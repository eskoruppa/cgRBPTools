from __future__ import annotations

import numpy as np
import scipy as sp
from numpy.linalg import slogdet, solve
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import pearsonr
from .se3 import poses2junctions, junctions2parameters, junctions2dynamics, parameters2junctions
from ..core.topology import CGRBPTopology


def dynamicparams2stiffness(
    dynamic_params: np.ndarray,
    subtract_mean: bool = False,
    chunk_size: int | None = None,
) -> np.ndarray:
    """
    Compute the stiffness matrix from dynamic parameters by calculating the
    covariance and inverting it.

    Parameters
    ----------
    dynamic_params : (NSteps, nbps, 6) or (NSteps, nbps*6) ndarray
        The dynamic parameters (e.g., junction parameters) for each sample.
        A 3-D array of shape ``(NSteps, nbps, 6)`` is automatically reshaped
        to ``(NSteps, nbps*6)`` before the covariance is computed.
    subtract_mean : bool, optional
        Whether to subtract the mean from the dynamic parameters before
        computing the covariance. Default is False.
    chunk_size : int or None, optional
        If given, the covariance accumulation is performed in chunks of this
        many frames.  Use this when ``NSteps`` is so large that holding the
        full ``(NSteps, nbps*6)`` array in memory is acceptable but the BLAS
        kernel's temporary workspace is not.  If ``None`` (default), the
        entire array is passed to a single BLAS ``dsyrk`` call via
        ``X.T @ X``, which is already memory-optimal for ordinary use.

    Returns
    -------
    stiffmat : (nbps*6, nbps*6) ndarray
        The stiffness (inverse covariance) matrix.
    """
    if dynamic_params.ndim == 3:
        nsteps = dynamic_params.shape[0]
        dynamic_params = dynamic_params.reshape(nsteps, -1)

    nsteps, n_features = dynamic_params.shape

    if subtract_mean:
        dynamic_params = dynamic_params - dynamic_params.mean(axis=0)

    if chunk_size is not None:
        cov = np.zeros((n_features, n_features), dtype=dynamic_params.dtype)
        for start in range(0, nsteps, chunk_size):
            chunk = dynamic_params[start : start + chunk_size]
            cov += chunk.T @ chunk
        cov /= nsteps
    else:
        cov = (dynamic_params.T @ dynamic_params) / nsteps

    stiffmat = np.linalg.inv(cov)
    return stiffmat

def align_euler_angles(
    params: np.ndarray,
    known_gs: np.ndarray | None = None,
    pi_threshold: float = np.pi * 0.5,
) -> np.ndarray:
    """Correct antipodal branch-cut ambiguities in the Euler-vector (rotation)
    components of SE3 parameters.

    When the rotation angle ‖Ω‖ is near π, the SO(3) logarithmic map has an
    antipodal ambiguity: Ω and −Ω represent the same rotation.  This causes
    individual frames to appear on the 'wrong' branch, shifting the empirical
    mean by ~2π relative to the true groundstate (most visible at ~5 bp/bead
    CG resolution where the helical twist per junction is ≈ π).

    The correction rule is: for each sample, if −Ω_t is closer (in Euclidean
    distance) to the reference than Ω_t is — equivalently, if Ω_t · Ω_ref < 0
    — replace Ω_t with −Ω_t.  Only vectors with ‖Ω_t‖ > ``pi_threshold`` are
    candidates, preventing spurious flips of near-identity rotations.

    Parameters
    ----------
    params : (NSteps, nbps, 6) ndarray
        SE3 parameter array.  First 3 entries per junction are the rotational
        Euler-vector components; last 3 are translational (not modified).
    known_gs : (nbps, 6) ndarray or None
        Known groundstate parameters.  If provided, ``known_gs[:, :3]`` is
        used as the per-junction reference for the flip decision.  If None,
        the per-junction componentwise median across the sample is used as a
        robust reference (reliable when fewer than half the samples are on the
        wrong branch).
    pi_threshold : float, optional
        Minimum Euler-vector magnitude (radians) below which a sample is never
        flipped.  Default is π/2.  Prevents accidental inversion of
        small-angle (near-identity) rotations for which the antipodal
        ambiguity does not exist.

    Returns
    -------
    corrected_params : ndarray
        Copy of ``params`` with corrected rotational components.
    """
    if params.ndim != 3 or params.shape[-1] != 6:
        raise ValueError(
            f"params must have shape (NSteps, nbps, 6), got {params.shape}"
        )

    params = params.copy()
    rot = params[:, :, :3]

    if known_gs is not None:
        ref = np.asarray(known_gs)[:, :3]
    else:
        # Componentwise median is robust when the majority is on the correct branch
        ref = np.median(rot, axis=0)

    # Dot product of each sample with its per-junction reference: (NSteps, nbps)
    dot = np.einsum('tji,ji->tj', rot, ref)

    # Magnitude of each sample's rotation vector: (NSteps, nbps)
    mag = np.linalg.norm(rot, axis=-1)

    # Identify flipped samples: dot < 0 AND magnitude above threshold
    flip_mask = (dot < 0) & (mag > pi_threshold)

    # Correct with a 2π shift, NOT a plain sign flip.
    #
    # For a true rotation angle θ_true > π the SO(3) log map returns the
    # antipodal representation Ω_log = -(2π - θ_true) * n̂_true, which has
    #   • opposite direction to the true axis n̂_true
    #   • magnitude ‖Ω_log‖ = 2π - θ_true  (< π)
    #
    # The correct representative on the "extended" branch is:
    #   Ω_corrected = (2π - ‖Ω_log‖) * (-Ω_log / ‖Ω_log‖)
    #              = -(2π / ‖Ω_log‖ - 1) * Ω_log
    #
    # At ‖Ω_log‖ = π this reduces to the plain sign flip.
    # For ‖Ω_log‖ < π the corrected magnitude 2π - ‖Ω_log‖ > π, which is
    # the intended "beyond π" extension that removes the arithmetic-mean bias.
    safe_mag = np.where(flip_mask, mag, 1.0)
    scale = np.where(flip_mask, -(2.0 * np.pi / safe_mag - 1.0), 1.0)
    params[:, :, :3] = scale[:, :, np.newaxis] * rot
    return params


def compute_groundstate_riemannian(
    junctions: np.ndarray,
    init_gs: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    """Compute the Fréchet mean groundstate on SE(3) via Riemannian gradient descent.

    For the Y-convention LAMMPS potential the energy is harmonic in
    ``Φ_dynamic = log(S^{-1} · G)``, so the optimal groundstate *S* satisfies
    ``⟨log(S^{-1} · G_i)⟩ = 0``, which is the Fréchet mean of {G_i} on SE(3).

    The standard intrinsic mean iteration (Riemannian gradient descent with
    unit step size) is::

        S ← S · exp( ⟨log(S^{-1} · G_i)⟩ )

    This converges to the Fréchet mean when the distribution is sufficiently
    concentrated (‖Φ_dynamic‖ < π almost surely), which holds for typical DNA
    MD trajectories.

    Parameters
    ----------
    junctions : (NSteps, nbps, 4, 4) ndarray
        SE(3) junction matrices for each snapshot and base-pair step.
    init_gs : (nbps, 6) ndarray or None
        Initial groundstate parameters (Euler vector + translation).  If None,
        the branch-corrected arithmetic mean of the junctions is used as a
        warm start.
    max_iter : int, optional
        Maximum number of iterations.  Default 50.
    tol : float, optional
        Convergence tolerance: iteration stops when
        ``max |μ| < tol``.  Default 1e-8.

    Returns
    -------
    gs_params : (nbps, 6) ndarray
        Groundstate SE(3) parameters, i.e. ``log(S)`` expressed as
        Euler vector + translation.
    """
    junctions = np.asarray(junctions, dtype=float)
    if junctions.ndim != 4 or junctions.shape[-2:] != (4, 4):
        raise ValueError(
            f"junctions must have shape (NSteps, nbps, 4, 4), got {junctions.shape}"
        )
    nbps = junctions.shape[1]

    if init_gs is not None:
        init_gs = np.asarray(init_gs, dtype=float)
        if init_gs.shape != (nbps, 6):
            raise ValueError(
                f"init_gs must have shape ({nbps}, 6), got {init_gs.shape}"
            )
        S = parameters2junctions(init_gs)
    else:
        # Branch-corrected arithmetic mean as warm start
        params0 = junctions2parameters(junctions)
        params0 = align_euler_angles(params0)
        mean0   = np.mean(params0, axis=0)
        S = parameters2junctions(mean0) 

    for i in range(max_iter):
        print(f"Riemannian mean iteration: computing gradient... (step {i+1}/{max_iter})")
        # D_i = S^{-1} · G_i  (tangent-space coordinates at S)
        dyn = junctions2dynamics(junctions, static_junctions=S)
        phi = junctions2parameters(dyn)
        phi = align_euler_angles(phi)

        # Riemannian gradient = tangent-space mean
        mu = np.mean(phi, axis=0)

        # Retract: S ← S · exp(μ)
        exp_mu = parameters2junctions(mu)
        S = np.matmul(S, exp_mu)

        print(f'  max |μ| = {np.max(np.abs(mu)):.2e}')
        if np.max(np.abs(mu)) < tol:
            break

    # Convert the converged S back to Lie-algebra coordinates
    gs_params = junctions2parameters(S)
    return gs_params


def eval_gs_and_stiffness(
    poses: np.ndarray, 
    topol: CGRBPTopology, 
    use_known_gs: bool = False,
    plot_twist_savefig: str | None = None,
    plot_twist_junction: int = 0,
) -> tuple[np.ndarray, np.ndarray]:

    junctions   = poses2junctions(poses)

    if topol.subtract_groundstate:
        params_raw  = junctions2parameters(junctions)
        params      = align_euler_angles(params_raw, known_gs=topol.get_groundstate() if use_known_gs else None)

        if plot_twist_savefig is not None:
            plot_twist_alignment(
                params_raw, params,
                junction_idx=plot_twist_junction,
                savefig=plot_twist_savefig,
            )
        mean_params = np.mean(params, axis=0)
        dynamic_params = params - topol.get_groundstate() if use_known_gs else params - mean_params

    else:
        print(f'Computing groundstate via Riemannian mean for stiffness calculation...')
        mean_params = compute_groundstate_riemannian(
            junctions,
            init_gs=topol.get_groundstate() if use_known_gs else None,
        )
        dynamic_junctions = junctions2dynamics(junctions, static_params=topol.get_groundstate() if use_known_gs else mean_params)
        dynamic_params = junctions2parameters(dynamic_junctions)

    stiffmat = dynamicparams2stiffness(dynamic_params)
    return mean_params, stiffmat


def eval_gs_and_diagonal_stiffness(
    poses: np.ndarray,
    topol: CGRBPTopology,
    use_known_gs: bool = False,
    plot_twist_savefig: str | None = None,
    plot_twist_junction: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    
    junctions   = poses2junctions(poses)

    if topol.subtract_groundstate:
        params_raw  = junctions2parameters(junctions)
        params      = align_euler_angles(params_raw, known_gs=topol.get_groundstate() if use_known_gs else None)

        if plot_twist_savefig is not None:
            plot_twist_alignment(
                params_raw, params,
                junction_idx=plot_twist_junction,
                savefig=plot_twist_savefig,
            )
        mean_params = np.mean(params, axis=0)
        dynamic_params = params - topol.get_groundstate() if use_known_gs else params - mean_params

    else:
        print(f'Computing groundstate via Riemannian mean for stiffness calculation...')
        mean_params = compute_groundstate_riemannian(
            junctions,
            init_gs=topol.get_groundstate() if use_known_gs else None,
        )
        dynamic_junctions = junctions2dynamics(junctions, static_params=topol.get_groundstate() if use_known_gs else mean_params)
        dynamic_params = junctions2parameters(dynamic_junctions)

    var = np.mean(dynamic_params**2, axis=0)
    stiff = 1.0 / var
    return mean_params, stiff

def diagonal_marginals(stiffmat: np.ndarray | sp.sparse.spmatrix, n_neighbors: int = 10) -> np.ndarray:
    """
    Computes the diagnomal marginals of a stiffness matrix, i.e., the effective 
    stiffness of each individual degree of freedom.
    
    This function processes the stiffness matrix by iterating through 6x6 blocks and
    computing the marginal stiffness for each block by considering n_neighbors
    neighboring blocks, inverting the combined matrix, and extracting the diagonal
    entries of the center block.
    
    Parameters:
    -----------
    stiffmat : np.ndarray or sp.sparse.spmatrix
        The stiffness matrix. Can be a dense numpy array or a sparse scipy matrix.
        Expected to have shape (N*6, N*6) where N is the number of 6x6 blocks.
    n_neighbors : int, optional
        Number of neighboring 6x6 blocks to include on each side. Default is 10.
    
    Returns:
    --------
    marginals : np.ndarray
        Array of shape (N, 6) containing the marginal effective stiffness values,
        where N is the number of 6x6 blocks. Each row contains the diagonal entries
        of the inverted covariance matrix for that block and its neighbors.
    
    Notes:
    ------
    - For boundary blocks, the number of neighbors will be reduced as appropriate
    - The function takes the reciprocal of the diagonal entries of the inverted matrix
    - The computation is performed block by block for numerical stability
    """
    # Convert to dense array if sparse
    if sp.sparse.issparse(stiffmat):
        stiffmat = stiffmat.toarray()
    
    # Determine the number of 6x6 blocks
    n_blocks = stiffmat.shape[0] // 6
    if stiffmat.shape[0] % 6 != 0 or stiffmat.shape[1] % 6 != 0:
        raise ValueError("Stiffness matrix dimensions must be divisible by 6")
    
    # Initialize output array
    marginals = np.zeros((n_blocks, 6))
    
    # Process each 6x6 block
    for block_idx in range(n_blocks):
        # Determine the range of neighboring blocks to include
        start_block = max(0, block_idx - n_neighbors)
        end_block = min(n_blocks, block_idx + n_neighbors + 1)
        
        # Extract the submatrix containing the block and its neighbors
        start_row = start_block * 6
        end_row = end_block * 6
        
        submatrix = stiffmat[start_row:end_row, start_row:end_row]
        
        # Invert the submatrix
        try:
            submatrix_inv = np.linalg.inv(submatrix)
        except np.linalg.LinAlgError:
            # If inversion fails, use pseudo-inverse
            submatrix_inv = np.linalg.pinv(submatrix)
        
        # Calculate the position of the current block within the submatrix
        center_block_offset = (block_idx - start_block) * 6
        center_block_end = center_block_offset + 6
        
        # Extract the diagonal entries corresponding to the center block
        diag_entries = np.diag(submatrix_inv[center_block_offset:center_block_end, 
                                             center_block_offset:center_block_end])
        
        # Take the reciprocal to get the marginal effective stiffness
        marginals[block_idx, :] = 1.0 / diag_entries
    
    return marginals


def kullbackleibler_divergence_2(K1, K2, normalized: bool = False, symmetrize: bool = True):
    """
    Compute Kullback–Leibler divergence between two stiffness matrices K1 and K2,
    interpreted as inverse covariance matrices of zero-mean Gaussians.

    D_KL( N(0, K1^{-1}) || N(0, K2^{-1}) )

    Parameters
    ----------
    K1, K2 : (n, n) ndarray
        Symmetric positive definite stiffness matrices.
    normalized : bool, optional
        If True, divide the result by n. Default is False.
    symmetrize : bool, optional
        If True, return the symmetrized KL divergence
        0.5 * (D_KL(p1||p2) + D_KL(p2||p1)). Default is True.

    Returns
    -------
    float
        KL divergence (>= 0).
    """

    K1 = np.asarray(K1, dtype=float)
    K2 = np.asarray(K2, dtype=float)

    if K1.shape != K2.shape:
        raise ValueError("K1 and K2 must have the same shape")

    n = K1.shape[0]

    # trace(K2^{-1} K1) using linear solve (more stable than explicit inverse)
    A = solve(K2, K1)
    trace_term = np.trace(A)

    # log(det(K2)/det(K1)) using slogdet for numerical stability
    sign1, logdet1 = slogdet(K1)
    sign2, logdet2 = slogdet(K2)

    if sign1 <= 0 or sign2 <= 0:
        raise ValueError("Matrices must be positive definite")

    logdet_ratio = logdet2 - logdet1

    kl = 0.5 * (trace_term - n + logdet_ratio)

    if symmetrize:
        A_rev = solve(K1, K2)
        trace_term_rev = np.trace(A_rev)
        kl_rev = 0.5 * (trace_term_rev - n - logdet_ratio)
        kl = 0.5 * (kl + kl_rev)

    if normalized:
        kl /= n
    return kl


def kullbackleibler_divergence(K1, K2, mu1=None, mu2=None, normalize: bool = False, symmetrize: bool = True):
    """Kullback-Leibler divergence D_KL(p1 || p2) for two multivariate
    Gaussians parameterised by their stiffness (precision) matrices.

    Parameters
    ----------
    K1, K2 : (d, d) array_like
        Symmetric positive-definite stiffness (precision) matrices of
        distributions p1 and p2.
    mu1, mu2 : (d,) array_like, optional
        Mean vectors.  If omitted, equal means are assumed and the
        Mahalanobis term vanishes.
    normalize : bool, optional
        If True, divide the result by d. Default is False.
    symmetrize : bool, optional
        If True, return the symmetrized KL divergence
        0.5 * (D_KL(p1||p2) + D_KL(p2||p1)). Default is True.

    Returns
    -------
    float
        D_KL(p1 || p2) or the symmetrized version (in nats). Returns nan
        if either matrix is not positive definite.

    Notes
    -----
    For zero-mean Gaussians with precision matrices K the density is

        p(x) ∝ exp(-½ xᵀ K x),   Σ = K⁻¹.

    The KL divergence is

        D_KL = ½ [ tr(K₂ K₁⁻¹) − d + ln det K₁ − ln det K₂
                   + (μ₂−μ₁)ᵀ K₂ (μ₂−μ₁) ].

    All inversions are replaced by Cholesky solves for stability.
    """
    
    K1 = np.asarray(K1, dtype=float)
    K2 = np.asarray(K2, dtype=float)

    if K1.shape != K2.shape:
        raise ValueError("K1 and K2 must have the same shape")

    d = K1.shape[0]

    try:
        # Cholesky factorisations  K = L Lᵀ
        L1, low1 = cho_factor(K1)
        L2, low2 = cho_factor(K2)
    except np.linalg.LinAlgError:
        return np.nan

    # log-determinants via Cholesky diagonal
    logdet_K1 = 2.0 * np.sum(np.log(np.diag(L1)))
    logdet_K2 = 2.0 * np.sum(np.log(np.diag(L2)))

    # tr(K₂ K₁⁻¹) = tr(K₁⁻¹ K₂):  solve K₁ X = K₂, then take trace
    K1inv_K2 = cho_solve((L1, low1), K2)
    trace_term = np.trace(K1inv_K2)

    # Mahalanobis term
    if mu1 is not None and mu2 is not None:
        dmu = np.asarray(mu2, dtype=float) - np.asarray(mu1, dtype=float)
        mahal = dmu @ K2 @ dmu
    else:
        mahal = 0.0

    kl = 0.5 * (trace_term - d + logdet_K1 - logdet_K2 + mahal)

    if symmetrize:
        # D_KL(p2||p1): swap roles of K1/K2 and mu1/mu2
        K2inv_K1 = cho_solve((L2, low2), K1)
        trace_term_rev = np.trace(K2inv_K1)
        mahal_rev = dmu @ K1 @ dmu if mu1 is not None and mu2 is not None else 0.0
        kl_rev = 0.5 * (trace_term_rev - d + logdet_K2 - logdet_K1 + mahal_rev)
        kl = 0.5 * (kl + kl_rev)

    if normalize:
        kl /= d
    return kl


def frobenius_difference(K_target, K_MD, normalize=False):
    """
    Compute the Frobenius norm of the difference between two matrices.

    Parameters
    ----------
    K_target : (n, n) ndarray
        Reference stiffness matrix.
    K_MD : (n, n) ndarray
        Stiffness matrix obtained from simulation.
    normalize : bool, default False
        If True, normalize the Frobenius norm by the norm of K_target.

    Returns
    -------
    float
        Frobenius norm (optionally normalized).
    """
    K_target = np.asarray(K_target)
    K_MD = np.asarray(K_MD)

    if K_target.shape != K_MD.shape:
        raise ValueError("K_target and K_MD must have the same shape")

    frob_diff = np.linalg.norm(K_MD - K_target, 'fro')

    if normalize:
        frob_norm = np.linalg.norm(K_target, 'fro')
        return frob_diff / frob_norm
    else:
        return frob_diff


def pearson_matrix_correlation(K_target, K_MD, normalize=False):
    """
    Compute the Pearson correlation coefficient between two matrices
    by flattening them into vectors.

    Parameters
    ----------
    K_target : (n, n) ndarray
        Reference stiffness matrix.
    K_MD : (n, n) ndarray
        Stiffness matrix obtained from simulation.
    normalize : bool, default False
        Included for API consistency, not used internally.

    Returns
    -------
    float
        Pearson correlation coefficient (-1 to 1).
    """
    K_target = np.asarray(K_target)
    K_MD = np.asarray(K_MD)

    if K_target.shape != K_MD.shape:
        raise ValueError("K_target and K_MD must have the same shape")

    vec_target = K_target.flatten()
    vec_MD = K_MD.flatten()

    r, _ = pearsonr(vec_target, vec_MD)
    return r


def plot_twist_alignment(
    params_raw: np.ndarray,
    params_aligned: np.ndarray,
    junction_idx: int = 0,
    savefig: str | None = None,
) -> None:
    """Plot the twist (rotational Euler-vector component 2) for a single
    junction over all snapshots, comparing uncorrected and corrected values.

    Four-panel layout:
      Left column  – time series (snapshot index vs Ω₂)
      Right column – histogram of Ω₂ values
    Upper row is uncorrected; lower row is corrected.

    Parameters
    ----------
    params_raw : (NSteps, nbps, 6) ndarray
        Raw SE3 parameters before antipodal alignment.
    params_aligned : (NSteps, nbps, 6) ndarray
        SE3 parameters after antipodal alignment.
    junction_idx : int, optional
        Which junction to inspect.  Default is 0.
    savefig : str or None, optional
        Base file path (without extension) to save the figure.  PNG and SVG
        are written.  If None, the figure is shown interactively.
    """
    import matplotlib.pyplot as plt

    raw_twist     = params_raw[:, junction_idx, 2]
    aligned_twist = params_aligned[:, junction_idx, 2]
    frames        = np.arange(raw_twist.shape[0])

    mean_raw = np.mean(raw_twist)
    mean_aln = np.mean(aligned_twist)

    fig, axes = plt.subplots(2, 2, figsize=(14, 5),
                             gridspec_kw={'width_ratios': [3, 1]})

    # --- top-left: raw time series ---
    axes[0, 0].plot(frames, raw_twist, lw=0.3, color='steelblue', alpha=0.5)
    axes[0, 0].axhline(mean_raw, color='red', lw=1.2, ls='--',
                       label=f'mean = {mean_raw:.4f} rad')
    axes[0, 0].set_ylabel('Twist $\\Omega_3$ (rad)')
    axes[0, 0].set_title(f'Uncorrected — junction {junction_idx}')
    axes[0, 0].legend(fontsize=8)

    # --- top-right: raw histogram ---
    bin_edges = np.linspace(raw_twist.min(), raw_twist.max(),
                            min(200, int(raw_twist.shape[0] ** 0.5) + 20))
    axes[0, 1].hist(raw_twist, bins=bin_edges, color='steelblue', alpha=0.7,
                    orientation='vertical')
    axes[0, 1].axvline(mean_raw, color='red', lw=1.2, ls='--')
    axes[0, 1].set_xlabel('Twist $\\Omega_3$ (rad)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution (raw)')

    # --- bottom-left: aligned time series ---
    axes[1, 0].plot(frames, aligned_twist, lw=0.3, color='darkorange', alpha=0.5)
    axes[1, 0].axhline(mean_aln, color='red', lw=1.2, ls='--',
                       label=f'mean = {mean_aln:.4f} rad')
    axes[1, 0].set_ylabel('Twist $\\Omega_3$ (rad)')
    axes[1, 0].set_xlabel('Snapshot index')
    axes[1, 0].set_title(f'Corrected — junction {junction_idx}')
    axes[1, 0].legend(fontsize=8)

    # --- bottom-right: aligned histogram ---
    bin_edges_aln = np.linspace(aligned_twist.min(), aligned_twist.max(),
                                min(200, int(aligned_twist.shape[0] ** 0.5) + 20))
    axes[1, 1].hist(aligned_twist, bins=bin_edges_aln, color='darkorange',
                    alpha=0.7, orientation='vertical')
    axes[1, 1].axvline(mean_aln, color='red', lw=1.2, ls='--')
    axes[1, 1].set_xlabel('Twist $\\Omega_3$ (rad)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Distribution (corrected)')

    plt.tight_layout()

    if savefig is not None:
        plt.savefig(f'{savefig}.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{savefig}.svg', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()