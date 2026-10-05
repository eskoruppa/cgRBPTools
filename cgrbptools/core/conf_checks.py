"""
Consistency and plausibility checks for configurations, and adjustment of the linking number.

The checks evaluate the configuration the way the LAMMPS CG-RBP styles (bond styles rbp and
rbpfene) do, such that problems surface before a simulation is started.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import scipy as sp
from scipy.spatial import cKDTree
from scipy.stats import chi2

from .topology import (
    CGRBPTopology,
    LMP_TOPOL_GROUNDSTATE_MAX_FACTOR,
    LMP_TOPOL_GROUNDSTATE_MIN_FACTOR,
)
from .conf_import import CONF_MAX_LISTED_INDICES, ConfigurationValidationError, _listed
from ..evals.se3 import junctions2dynamics, junctions2parameters, poses2junctions, poses2parameters

# Bonds shorter than this (nm) mark coinciding consecutive beads
CONF_ZERO_BOND = 1e-8
# Median cosine between third triad axis and bond direction below which the triads are considered
# not to follow the chain (e.g. transposed triads or reversed order)
CONF_MIN_MEDIAN_ALIGNMENT = 0.5
# Non-neighbouring beads closer than this fraction of the median bond length are reported
CONF_OVERLAP_FRACTION = 0.5
# A closing bond longer than this multiple of the median bond length means the chain is not closed
CONF_MAX_CLOSURE_FACTOR = 2.0

# FENE regimes of bond style rbpfene (LAMMPS CG-RBP package, bond_rbp_fene.cpp), in terms of
# rlogarg = 1 - (r - Rc)^2 / (R0 - Rc)^2 for bonds with r >= Rc
FENE_RLOGARG_MIN = 0.1     # below: the potential is continued linearly and LAMMPS warns
FENE_RLOGARG_ABORT = -3.0  # at or below: LAMMPS aborts with "Bad RBP FENE bond"

# Significance level of the energy check (Bonferroni-corrected over the junctions)
CONF_ENERGY_ALPHA = 1e-3
# Up to this matrix dimension marginal covariances are obtained by dense inversion
CONF_DENSE_INVERSION_MAX_DIM = 3000
# Half width (in junctions) of the window used for the marginals of large closed chains
CONF_MARGINAL_WINDOW = 20

# Largest deviation of the linking number from an integer that is accepted silently
CONF_LK_INTEGER_TOL = 0.1
# Largest twist per step (rad) that may be added to adjust the linking number
CONF_MAX_ADDED_TWIST = 0.5 * np.pi


###################################################################################################
# Geometry

def bond_vectors(positions: np.ndarray, closed: bool = False) -> np.ndarray:
    """Vectors from each bead to the next one, including the closing bond for closed chains."""
    if closed:
        return np.roll(positions, -1, axis=0) - positions
    return positions[1:] - positions[:-1]


def check_geometry(poses: np.ndarray, groundstate: np.ndarray, closed: bool = False) -> list[str]:
    """
    Check the geometry of a configuration against the model.

    Raises for coinciding consecutive beads, bond lengths incompatible with the model (wrong
    length unit or resolution), a closing bond much longer than the other bonds (closed chains),
    and triads whose third axis does not follow the chain (e.g. transposed triads). Returns
    warnings for individual steps along which the third triad axis points against the chain and
    for non-neighbouring beads closer than half the median bond length.

    Parameters
    ----------
    poses : np.ndarray
        Poses with shape (N, 4, 4), positions in nm.
    groundstate : np.ndarray
        Groundstate of the model with shape (nbps, 6), translations in nm.
    closed : bool, optional
        Closed (circular) topology. Default is False.

    Returns
    -------
    list of str
        Warnings.

    Raises
    ------
    ConfigurationValidationError
    """
    warnings = []
    pos = poses[:, :3, 3]
    bonds = bond_vectors(pos, closed)
    lengths = np.linalg.norm(bonds, axis=1)

    zero = np.flatnonzero(lengths <= CONF_ZERO_BOND)
    if len(zero) > 0:
        raise ConfigurationValidationError(
            f'The configuration contains coinciding consecutive beads (zero bond length) at bonds '
            f'{_listed(zero)} (0-based; bond i connects beads i and i+1).'
        )

    chain_lengths = lengths[:-1] if closed else lengths
    median_bond = np.median(chain_lengths)
    model_bond = np.median(np.linalg.norm(groundstate[:, 3:], axis=1))
    ratio = median_bond / model_bond
    if not LMP_TOPOL_GROUNDSTATE_MIN_FACTOR <= ratio <= LMP_TOPOL_GROUNDSTATE_MAX_FACTOR:
        msg = (
            f'The bonds of the configuration have a median length of {median_bond:.3g} nm, while '
            f'the model expects about {model_bond:.3g} nm per bead step (ratio {ratio:.3g}).'
        )
        if 7.0 <= ratio <= 14.0:
            msg += ' The positions appear to be given in Angstrom (--conf_units angstrom).'
        msg += (
            ' Check the length unit of the positions (--conf_units) and the resolution of the '
            'configuration (--conf_resolution).'
        )
        raise ConfigurationValidationError(msg)

    if closed and lengths[-1] > CONF_MAX_CLOSURE_FACTOR * median_bond:
        raise ConfigurationValidationError(
            f'The closing bond (last to first bead) is {lengths[-1]:.3g} nm long, '
            f'{lengths[-1] / median_bond:.1f} times the median bond length. The configuration does '
            f'not appear to be closed.'
        )

    alignment = np.einsum('ni,ni->n', poses[: len(bonds), :3, 2], bonds / lengths[:, None])
    if np.median(alignment) < CONF_MIN_MEDIAN_ALIGNMENT:
        raise ConfigurationValidationError(
            f'The third triad axis does not follow the chain (median cosine between third triad '
            f'axis and bond direction: {np.median(alignment):.2f}). The triads may be stored as '
            f'rows instead of columns (transposed), or the order of the poses may be reversed.'
        )
    against = np.flatnonzero(alignment <= 0.0)
    if len(against) > 0:
        warnings.append(
            f'At {len(against)} step(s) the third triad axis points against the chain direction '
            f'(0-based bond indices: {_listed(against)}).'
        )

    num = len(pos)
    if num > 3:
        cutoff = CONF_OVERLAP_FRACTION * median_bond
        pairs = cKDTree(pos).query_pairs(cutoff, output_type='ndarray')
        if len(pairs) > 0:
            separation = np.abs(pairs[:, 0] - pairs[:, 1])
            if closed:
                separation = np.minimum(separation, num - separation)
            pairs = pairs[separation > 1]
        if len(pairs) > 0:
            pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
            listed = ', '.join(f'{i}-{j}' for i, j in pairs[:CONF_MAX_LISTED_INDICES])
            warnings.append(
                f'{len(pairs)} pair(s) of non-neighbouring beads are closer than {cutoff:.3g} nm '
                f'(half the median bond length): {listed}. The chain may intersect itself.'
            )
    return warnings


###################################################################################################
# FENE

def check_fene(
    poses: np.ndarray,
    Rc: float,
    R0: float,
    closed: bool = False,
    abort_is_error: bool = True,
) -> list[str]:
    """
    Check the bond lengths against the FENE term of bond style rbpfene.

    The FENE term acts on bonds of length r >= Rc and diverges at R0. LAMMPS continues it
    linearly once rlogarg = 1 - (r-Rc)^2/(R0-Rc)^2 drops below 0.1 (with warnings) and aborts
    for rlogarg <= -3.

    Parameters
    ----------
    poses : np.ndarray
        Poses with shape (N, 4, 4), positions in simulation units.
    Rc, R0 : float
        FENE onset and divergence distances in simulation units.
    closed : bool, optional
        Closed (circular) topology. Default is False.
    abort_is_error : bool, optional
        Raise if LAMMPS would abort (default). Otherwise this is reported as a warning.

    Returns
    -------
    list of str
        Warnings.

    Raises
    ------
    ConfigurationValidationError
        If abort_is_error and a bond is stretched into the regime in which LAMMPS aborts.
    """
    warnings = []
    lengths = np.linalg.norm(bond_vectors(poses[:, :3, 3], closed), axis=1)
    active = lengths >= Rc
    if not active.any():
        return warnings
    rlogarg = 1.0 - ((lengths - Rc) / (R0 - Rc)) ** 2
    abort = np.flatnonzero(active & (rlogarg <= FENE_RLOGARG_ABORT))
    linear = np.flatnonzero(active & (rlogarg > FENE_RLOGARG_ABORT) & (rlogarg < FENE_RLOGARG_MIN))
    regular = np.flatnonzero(active & (rlogarg >= FENE_RLOGARG_MIN))
    units = (' Rc, R0 and the bond lengths are in simulation units; the FENE coefficients are given in nm '
             '(and kT/nm^2) and rescaled with the unit length.')
    if len(abort) > 0:
        msg = (
            f'{len(abort)} bond(s) are stretched so far beyond the FENE limit (Rc = {Rc:g}, '
            f'R0 = {R0:g}, longest bond {lengths.max():.3g}) that LAMMPS aborts with '
            f'"Bad RBP FENE bond" (0-based bond indices: {_listed(abort)}).' + units
        )
        if abort_is_error:
            raise ConfigurationValidationError(msg)
        warnings.append(msg)
    if len(linear) > 0:
        warnings.append(
            f'{len(linear)} bond(s) exceed the FENE limit (Rc = {Rc:g}, R0 = {R0:g}). LAMMPS '
            f'continues the potential linearly and prints warnings (0-based bond indices: '
            f'{_listed(linear)}).' + units
        )
    if len(regular) > 0:
        warnings.append(
            f'The FENE term is active from the start for {len(regular)} bond(s) with length >= '
            f'Rc = {Rc:g} (0-based bond indices: {_listed(regular)}).'
        )
    return warnings


###################################################################################################
# Elastic energy

def deformations(
    poses: np.ndarray,
    groundstate: np.ndarray,
    closed: bool = False,
    subtract_groundstate: bool = False,
) -> np.ndarray:
    """
    Deformation of every junction relative to the groundstate, as evaluated by the rbp styles.

    By default the deformation of a junction g is the rotation vector and translation of
    g0^-1 g, with g0 the groundstate junction. With subtract_groundstate the groundstate
    parameters are subtracted from the parameters of g.

    Returns
    -------
    np.ndarray
        Deformations with shape (n_junctions, 6).
    """
    if subtract_groundstate:
        return poses2parameters(poses, closed=closed) - groundstate
    junctions = poses2junctions(poses, closed=closed)
    return junctions2parameters(junctions2dynamics(junctions, static_params=groundstate))


def marginal_stiffness_blocks(stiffmat, closed: bool = False) -> np.ndarray:
    """
    Marginal stiffness of every junction, i.e. the inverse of the 6x6 diagonal blocks of the
    covariance matrix stiffmat^-1.

    Small systems are inverted densely. Large open chains use an exact recursion for block
    tridiagonal matrices (linear in the number of junctions). Large closed chains use the
    inverse of a window of 2*CONF_MARGINAL_WINDOW+1 junctions around each junction, which
    slightly overestimates the marginal stiffness.

    Raises
    ------
    np.linalg.LinAlgError
        If stiffmat is not positive definite.
    """
    stiffmat = sp.sparse.csr_matrix(stiffmat)
    dim = stiffmat.shape[0]
    if dim <= CONF_DENSE_INVERSION_MAX_DIM:
        cov = _spd_inverse(stiffmat.toarray())
        n = dim // 6
        cov_blocks = np.array([cov[6 * i:6 * i + 6, 6 * i:6 * i + 6] for i in range(n)])
    else:
        # many small inversions: threads only add overhead
        with _single_threaded():
            if not closed:
                cov_blocks = _blocktridiag_cov_blocks(stiffmat)
            else:
                cov_blocks = _windowed_cov_blocks(stiffmat)
    return np.linalg.inv(cov_blocks)


def _single_threaded():
    """Context restricting the BLAS thread pools to one thread (if threadpoolctl is available)."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return nullcontext()
    return threadpool_limits(limits=1)


def _spd_inverse(mat: np.ndarray) -> np.ndarray:
    """Inverse of a symmetric positive definite matrix (raises LinAlgError otherwise)."""
    linv = np.linalg.inv(np.linalg.cholesky(mat))
    return linv.T @ linv


def _block_bandwidth(stiffmat, closed: bool = False) -> int:
    """Largest distance (in junctions) between coupled junctions."""
    coo = stiffmat.tocoo()
    if coo.nnz == 0:
        return 0
    n = stiffmat.shape[0] // 6
    dist = np.abs(coo.row // 6 - coo.col // 6)
    if closed:
        dist = np.minimum(dist, n - dist)
    return int(dist.max())


def _blocktridiag_cov_blocks(stiffmat) -> np.ndarray:
    """Exact diagonal covariance blocks for an open chain via the block tridiagonal recursion."""
    dim = stiffmat.shape[0]
    size = 6 * max(_block_bandwidth(stiffmat), 1)
    slices = [slice(s, min(s + size, dim)) for s in range(0, dim, size)]
    diag = [stiffmat[s, s].toarray() for s in slices]
    upper = [stiffmat[slices[k], slices[k + 1]].toarray() for k in range(len(slices) - 1)]

    # left-connected inverses
    g = [_spd_inverse(diag[0])]
    for k in range(1, len(slices)):
        g.append(_spd_inverse(diag[k] - upper[k - 1].T @ g[k - 1] @ upper[k - 1]))
    # diagonal blocks of the full inverse
    cov = [None] * len(slices)
    cov[-1] = g[-1]
    for k in range(len(slices) - 2, -1, -1):
        cov[k] = g[k] + g[k] @ upper[k] @ cov[k + 1] @ upper[k].T @ g[k]

    blocks = []
    for c in cov:
        for j in range(c.shape[0] // 6):
            blocks.append(c[6 * j:6 * j + 6, 6 * j:6 * j + 6])
    return np.array(blocks)


def _windowed_cov_blocks(stiffmat) -> np.ndarray:
    """Approximate diagonal covariance blocks for a closed chain from windows around each junction."""
    n = stiffmat.shape[0] // 6
    half = max(CONF_MARGINAL_WINDOW, 4 * _block_bandwidth(stiffmat, closed=True))
    blocks = np.empty((n, 6, 6))
    for i in range(n):
        idx = np.arange(i - half, i + half + 1) % n
        dof = (6 * idx[:, None] + np.arange(6)).ravel()
        cov = _spd_inverse(stiffmat[dof][:, dof].toarray())
        blocks[i] = cov[6 * half:6 * half + 6, 6 * half:6 * half + 6]
    return blocks


def energy_check(
    poses: np.ndarray,
    topology: CGRBPTopology,
    alpha: float = CONF_ENERGY_ALPHA,
) -> tuple[list[str], list[str]]:
    """
    Assess the elastic energy of a configuration within the Hamiltonian that is simulated.

    Deformations and stiffness are evaluated as by the rbp styles, using the coefficients as
    written to the database (limited to the coupling range and rounded). The total energy is
    compared with its equilibrium distribution (E = chi2(6 n_junctions)/2 in kT), and the
    deformation of every junction with its marginal equilibrium distribution (chi2 with 6
    degrees of freedom, Bonferroni-corrected over the junctions).

    Parameters
    ----------
    poses : np.ndarray
        Poses with shape (N, 4, 4), positions in simulation units.
    topology : CGRBPTopology
        Topology with couplings set.
    alpha : float, optional
        Significance level. Default is CONF_ENERGY_ALPHA.

    Returns
    -------
    tuple[list[str], list[str]]
        Information lines and warnings.
    """
    closed = topology.closed
    groundstate = topology._reconstruct_groundstate()
    stiffmat = topology._reconstruct_stiffness_matrix().tocsr()
    delta = deformations(poses, groundstate, closed, topology.subtract_groundstate)
    num = len(delta)
    flat = delta.reshape(-1)
    energy = 0.5 * flat @ (stiffmat @ flat) * topology.unit_energy
    mean = 3.0 * num
    info = [
        f'Elastic energy of the configuration: {energy:.1f} kT for {num} junctions '
        f'(equilibrium mean {mean:.0f} kT, ratio {energy / mean:.2f}).'
    ]
    warnings = []
    p_total = chi2.sf(2.0 * energy, 6 * num)
    if p_total < alpha:
        warnings.append(
            f'The elastic energy ({energy:.1f} kT) is far above the equilibrium expectation of '
            f'{mean:.0f} +- {np.sqrt(mean):.1f} kT (p = {p_total:.1e}). The configuration is strongly '
            f'deformed with respect to the model.'
        )

    try:
        marginals = marginal_stiffness_blocks(stiffmat, closed)
    except np.linalg.LinAlgError:
        try:
            marginals = marginal_stiffness_blocks(topology.get_stiffness_matrix(), closed)
        except np.linalg.LinAlgError:
            warnings.append(
                'The stiffness matrix is not positive definite; the per-junction energy test was skipped.'
            )
            return info, warnings
        warnings.append(
            f'The simulated stiffness matrix (coupling range {topology.coupling_range}) is not '
            f'positive definite; the per-junction test uses the full stiffness matrix instead.'
        )
    chi = np.einsum('ni,nij,nj->n', delta, marginals, delta) * topology.unit_energy
    flagged = np.flatnonzero(chi2.sf(chi, 6) < alpha / num)
    if len(flagged) > 0:
        order = flagged[np.argsort(chi[flagged])[::-1]]
        listed = ', '.join(f'{i} ({0.5 * chi[i]:.1f} kT)' for i in order[:CONF_MAX_LISTED_INDICES])
        if len(order) > CONF_MAX_LISTED_INDICES:
            listed += f' and {len(order) - CONF_MAX_LISTED_INDICES} more'
        warnings.append(
            f'{len(flagged)} of {num} junction(s) are deformed far beyond thermal fluctuations '
            f'(p < {alpha / num:.1e} per junction in equilibrium). 0-based junction index '
            f'(marginal energy): {listed}.'
        )
    return info, warnings


###################################################################################################
# Linking number

def relaxed_linking_number(topology: CGRBPTopology, bp_groundstate: np.ndarray | None = None) -> float:
    """
    Relaxed linking number of a closed topology: the total intrinsic twist in turns.

    The twist of a coarse-grained step is only defined up to full turns (about 343 degrees of
    ten base pairs are represented as -17 degrees). If the base-pair-level groundstate of the
    sequence the coarse-grained parameters were generated for is given, the full turns of every
    composite step are restored from it. Otherwise the result may be off by an integer, which
    does not affect excess linking numbers.

    Parameters
    ----------
    topology : CGRBPTopology
        Closed topology with groundstate.
    bp_groundstate : np.ndarray, optional
        Base-pair-level groundstate with shape (nbp*composite_size, 6), the composite step k
        comprising the base-pair steps k*composite_size ... (k+1)*composite_size - 1.
    """
    twists = topology.groundstate[:, 2]
    lk0 = float(np.sum(twists) / (2 * np.pi))
    if bp_groundstate is None:
        return lk0
    composite_size = len(bp_groundstate) // len(twists)
    if composite_size * len(twists) != len(bp_groundstate):
        raise ValueError(
            f'The base-pair groundstate ({len(bp_groundstate)} steps) does not match the '
            f'{len(twists)} composite steps of the topology.'
        )
    bp_twists = bp_groundstate[:, 2].reshape(len(twists), composite_size).sum(axis=1)
    return lk0 + float(np.sum(np.round((bp_twists - twists) / (2 * np.pi))))


def linking_number(poses: np.ndarray, topology: CGRBPTopology) -> tuple[float, float, float]:
    """
    Excess twist, writhe and excess linking number (in turns) of a closed configuration,
    relative to the groundstate of the topology. Requires the PyLk submodule.
    """
    try:
        from ..evals.link import poses2link
    except ImportError as e:
        raise ImportError(
            'Evaluating the linking number requires the PyLk submodule (cgrbptools/evals/PyLk).'
        ) from e
    twist, writhe, link = poses2link(poses, closed=True, groundstate=topology.groundstate)
    return float(twist), float(writhe), float(link)


def add_twist_turns(poses: np.ndarray, turns: int) -> np.ndarray:
    """
    Return poses with turns full turns of twist added uniformly: triad i is rotated about its
    own third axis by 2*pi*turns*i/N. Positions are unchanged.
    """
    num = len(poses)
    angles = 2 * np.pi * turns * np.arange(num) / num
    cos, sin = np.cos(angles), np.sin(angles)
    rz = np.zeros((num, 3, 3))
    rz[:, 0, 0] = cos
    rz[:, 0, 1] = -sin
    rz[:, 1, 0] = sin
    rz[:, 1, 1] = cos
    rz[:, 2, 2] = 1.0
    out = np.array(poses, dtype=np.float64, copy=True)
    out[:, :3, :3] = poses[:, :3, :3] @ rz
    return out


def adjust_excess_link(
    poses: np.ndarray,
    topology: CGRBPTopology,
    excess_link: float | None = None,
    bp_groundstate: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Report the linking number of a closed configuration and optionally adjust it.

    If excess_link is given, full turns of twist are added uniformly (see add_twist_turns) such
    that the linking number becomes round(Lk0 + excess_link), with Lk0 the relaxed linking
    number. This is the convention of ConfBuilder.circular.

    Parameters
    ----------
    poses : np.ndarray
        Poses with shape (N, 4, 4) of a closed configuration.
    topology : CGRBPTopology
        Closed topology with groundstate.
    excess_link : float, optional
        Requested excess linking number. If None, the configuration is not changed.
    bp_groundstate : np.ndarray, optional
        Base-pair-level groundstate used to restore the full turns of coarse-grained steps in
        the reported absolute linking numbers (see relaxed_linking_number).

    Returns
    -------
    tuple[np.ndarray, list[str], list[str]]
        Poses, information lines and warnings.

    Raises
    ------
    ConfigurationValidationError
        If the adjustment would add more than CONF_MAX_ADDED_TWIST of twist per step.
    """
    info, warnings = [], []
    lk0 = relaxed_linking_number(topology, bp_groundstate)
    twist, writhe, dlk = linking_number(poses, topology)
    lk = lk0 + dlk
    lk_int = int(round(lk))
    info.append(
        f'Linking number of the configuration: Lk = {lk:.3f} (relaxed Lk0 = {lk0:.3f}; excess '
        f'twist {twist:.3f}, writhe {writhe:.3f}, excess link {dlk:.3f}).'
    )
    if abs(lk - lk_int) > CONF_LK_INTEGER_TOL:
        warnings.append(
            f'The linking number of the configuration ({lk:.3f}) is not close to an integer. The '
            f'ring may not be closed properly or may be very strongly deformed.'
        )
    if excess_link is None:
        return poses, info, warnings

    target = int(round(lk0 + excess_link))
    turns = target - lk_int
    if turns == 0:
        info.append(
            f'The linking number matches the requested excess link (Lk = {target}, excess link '
            f'{target - lk0:.3f}).'
        )
        return poses, info, warnings
    per_step = 2 * np.pi * turns / len(poses)
    if abs(per_step) > CONF_MAX_ADDED_TWIST:
        raise ConfigurationValidationError(
            f'Adjusting the linking number by {turns} turns would add {np.degrees(per_step):.1f} '
            f'degrees of twist per step (limit {np.degrees(CONF_MAX_ADDED_TWIST):.0f} degrees).'
        )
    poses = add_twist_turns(poses, turns)
    _, _, dlk_new = linking_number(poses, topology)
    info.append(
        f'Added {turns:+d} turn(s) of twist ({np.degrees(per_step):+.2f} degrees per step): '
        f'Lk = {lk0 + dlk_new:.3f}, target {target} (excess link {target - lk0:.3f}).'
    )
    if abs(lk0 + dlk_new - target) > CONF_LK_INTEGER_TOL:
        warnings.append(
            f'After the adjustment the linking number ({lk0 + dlk_new:.3f}) deviates from the '
            f'target {target}.'
        )
    return poses, info, warnings
