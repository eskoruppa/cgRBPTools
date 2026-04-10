from __future__ import annotations

import numpy as np
import scipy as sp
from numpy.linalg import slogdet, solve
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import pearsonr
from .se3 import poses2junctions, junctions2parameters, junctions2dynamics 
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

    # Accumulate X^T X in chunks to bound peak extra memory to
    # O(chunk_size * n_features) rather than the full array size.
    if chunk_size is not None:
        cov = np.zeros((n_features, n_features), dtype=dynamic_params.dtype)
        for start in range(0, nsteps, chunk_size):
            chunk = dynamic_params[start : start + chunk_size]
            cov += chunk.T @ chunk
        cov /= nsteps
    else:
        # Single BLAS dsyrk call — O(n_features^2) extra memory.
        cov = (dynamic_params.T @ dynamic_params) / nsteps

    stiffmat = np.linalg.inv(cov)
    return stiffmat

def eval_gs_and_stiffness(poses: np.ndarray, topol: CGRBPTopology, use_known_gs: bool = False) -> tuple[np.ndarray, np.ndarray]:
    junctions = poses2junctions(poses)
    params = junctions2parameters(junctions)
    mean_params = np.mean(params, axis=0)

    if use_known_gs:
        gs = topol.get_groundstate()
    else:
        gs = mean_params

    if topol.subtract_groundstate:
        dynamic_params = params - gs
    else:
        dynamic_junctions = junctions2dynamics(junctions, static_params=gs)
        dynamic_params = junctions2parameters(dynamic_junctions)

    stiffmat = dynamicparams2stiffness(dynamic_params)
    return mean_params, stiffmat

def eval_gs_and_diagonal_stiffness(poses: np.ndarray, topol: CGRBPTopology, use_known_gs: bool = False) -> tuple[np.ndarray, np.ndarray]:
    junctions = poses2junctions(poses)
    params = junctions2parameters(junctions)
    mean_params = np.mean(params, axis=0)

    if use_known_gs:
        gs = topol.get_groundstate()
    else:
        gs = mean_params

    if topol.subtract_groundstate:
        dynamic_params = params - gs
    else:
        dynamic_junctions = junctions2dynamics(junctions, static_params=gs)
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