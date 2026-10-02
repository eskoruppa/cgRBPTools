from __future__ import annotations

import math
import numpy as np
from ..SO3 import so3

# Direct imports needed for numba JIT-to-JIT calls in the _optimized variants
from ..SO3.so3.SE3 import _se3_inverse_sv as _se3_inverse
from ..SO3.so3.Euler import _se3_rotmat2euler_sv as _se3_rotmat2euler
from ..SO3.so3.pyConDec.pycondec import cond_jit

# Numerical constants (mirrored from SO3/so3/Euler.py)
_EULER_EPSILON    = 1e-12
_EULER_SERIES_SMALL = 1e-4


def poses2junctions(poses: np.ndarray, closed: bool = False, optimized: bool = True) -> np.ndarray:
    """Convert a sequence of SE3 poses to junctions.

    Parameters
    ----------
    poses : np.ndarray, shape (*batch, n_poses, 4, 4)
        SE3 pose matrices.  *batch may be empty (single snapshot) or contain
        any number of leading dimensions (e.g. multiple simulations).
    closed : bool, default False
        If True, treat the chain as topologically closed: an additional
        junction connecting the last pose back to the first is appended,
        yielding n_poses junctions instead of n_poses-1.
    optimized : bool, default True
        If True, delegates to poses2junctions_optimized (vectorised NumPy).

    Returns
    -------
    np.ndarray, shape (*batch, n_junc, 4, 4)
        n_junc == n_poses if closed else n_poses-1.
    """
    if optimized:
        return poses2junctions_optimized(poses, closed=closed)
    batch_shape = poses.shape[:-3]
    n_poses     = poses.shape[-3]
    flat        = poses.reshape(-1, n_poses, 4, 4)
    N           = flat.shape[0]
    n_junc      = n_poses if closed else n_poses - 1
    out         = np.zeros((N, n_junc, 4, 4))
    for i in range(N):
        for j in range(n_poses - 1):
            out[i, j] = so3.se3_inverse(flat[i, j]) @ flat[i, j + 1]
        if closed:
            out[i, n_poses - 1] = so3.se3_inverse(flat[i, n_poses - 1]) @ flat[i, 0]
    return out.reshape(batch_shape + (n_junc, 4, 4))


def junctions2parameters(junctions: np.ndarray, optimized: bool = True) -> np.ndarray:
    """Convert a sequence of junctions to SE3 parameters.

    Parameters
    ----------
    junctions : np.ndarray, shape (*batch, n_junc, 4, 4)
    optimized : bool, default True
        If True, delegates to junctions2parameters_optimized (numba JIT).

    Returns
    -------
    np.ndarray, shape (*batch, n_junc, 6)
    """
    if optimized:
        return junctions2parameters_optimized(junctions)
    batch_shape = junctions.shape[:-3]
    n_junc      = junctions.shape[-3]
    flat        = junctions.reshape(-1, n_junc, 4, 4)
    N           = flat.shape[0]
    out         = np.zeros((N, n_junc, 6))
    for i in range(N):
        for j in range(n_junc):
            out[i, j] = so3.se3_rotmat2euler(flat[i, j])
    return out.reshape(batch_shape + (n_junc, 6))


def poses2parameters(poses: np.ndarray, closed: bool = False, optimized: bool = True) -> np.ndarray:
    """Convert a sequence of SE3 poses to SE3 parameters.

    Parameters
    ----------
    poses : np.ndarray, shape (*batch, n_poses, 4, 4)
    closed : bool, default False
        If True, treat the chain as topologically closed: an additional
        parameter set for the junction connecting the last pose back to the
        first is appended, yielding n_poses parameters instead of n_poses-1.
    optimized : bool, default True
        If True, delegates to poses2parameters_optimized (numba JIT).

    Returns
    -------
    np.ndarray, shape (*batch, n_junc, 6)
        n_junc == n_poses if closed else n_poses-1.
    """
    if optimized:
        return poses2parameters_optimized(poses, closed=closed)
    batch_shape = poses.shape[:-3]
    n_poses     = poses.shape[-3]
    flat        = poses.reshape(-1, n_poses, 4, 4)
    N           = flat.shape[0]
    n_junc      = n_poses if closed else n_poses - 1
    out         = np.zeros((N, n_junc, 6))
    for i in range(N):
        for j in range(n_poses - 1):
            gij = so3.se3_inverse(flat[i, j]) @ flat[i, j + 1]
            out[i, j] = so3.se3_rotmat2euler(gij)
        if closed:
            gij = so3.se3_inverse(flat[i, n_poses - 1]) @ flat[i, 0]
            out[i, n_poses - 1] = so3.se3_rotmat2euler(gij)
    return out.reshape(batch_shape + (n_junc, 6))


def parameters2junctions(params: np.ndarray, optimized: bool = True) -> np.ndarray:
    """Convert a sequence of SE3 parameters to junctions.

    Parameters
    ----------
    params : np.ndarray, shape (*batch, n_junc, 6)
        *batch may be empty (single set of junctions).
    optimized : bool, default True
        If True, delegates to parameters2junctions_optimized (vectorised NumPy).

    Returns
    -------
    np.ndarray, shape (*batch, n_junc, 4, 4)
    """
    if optimized:
        return parameters2junctions_optimized(params)
    batch_shape = params.shape[:-2]
    n_junc      = params.shape[-2]
    flat        = params.reshape(-1, n_junc, 6)
    N           = flat.shape[0]
    out         = np.zeros((N, n_junc, 4, 4))
    for i in range(N):
        for j in range(n_junc):
            out[i, j] = so3.se3_euler2rotmat(flat[i, j])
    return out.reshape(batch_shape + (n_junc, 4, 4))


def junctions2dynamics(junctions: np.ndarray, static_junctions: np.ndarray | None = None, static_params: np.ndarray | None = None, optimized: bool = True) -> np.ndarray:
    """Compute the dynamic junctions relative to a static groundstate.

    Parameters
    ----------
    junctions        : np.ndarray, shape (*batch, n_junc, 4, 4)
    static_junctions : np.ndarray | None, shape (n_junc, 4, 4)
        If None, computed from static_params or as the mean over all batch
        dimensions of junctions.
    static_params    : np.ndarray | None, shape (n_junc, 6)
    optimized : bool, default True
        If True, delegates to junctions2dynamics_optimized (vectorised NumPy).

    Returns
    -------
    np.ndarray, shape (*batch, n_junc, 4, 4)
    """
    if optimized:
        return junctions2dynamics_optimized(junctions, static_junctions=static_junctions, static_params=static_params)
    if static_junctions is None:
        if static_params is not None:
            static_junctions = parameters2junctions(static_params, optimized=False)
        else:
            params = junctions2parameters(junctions, optimized=False)
            # Average over ALL batch dimensions → groundstate shape (n_junc, 6)
            static_params = np.mean(params.reshape(-1, params.shape[-2], params.shape[-1]), axis=0)
            static_junctions = parameters2junctions(static_params, optimized=False)
    # static_junctions is (n_junc, 4, 4) — broadcast over all batch dims
    batch_shape = junctions.shape[:-3]
    n_junc      = junctions.shape[-3]
    flat        = junctions.reshape(-1, n_junc, 4, 4)
    N           = flat.shape[0]
    s_flat      = static_junctions.reshape(n_junc, 4, 4)
    out         = np.zeros((N, n_junc, 4, 4))
    for i in range(N):
        for j in range(n_junc):
            out[i, j] = so3.se3_inverse(s_flat[j]) @ flat[i, j]
    return out.reshape(batch_shape + (n_junc, 4, 4))


def poses2junctions_optimized(poses: np.ndarray, closed: bool = False) -> np.ndarray:
    """Vectorised version of poses2junctions.  Accepts arbitrary leading batch dims.

    Parameters
    ----------
    poses : np.ndarray, shape (*batch, n_poses, 4, 4)
    closed : bool, default False
        If True, treat the chain as topologically closed (append the
        last-to-first junction), yielding n_poses junctions instead of
        n_poses-1.

    Returns
    -------
    np.ndarray, shape (*batch, n_junc, 4, 4)
        n_junc == n_poses if closed else n_poses-1.
    """
    batch_shape = poses.shape[:-3]
    n_poses     = poses.shape[-3]
    flat        = poses.reshape(-1, n_poses, 4, 4)   # (N, n_poses, 4, 4)

    if closed:
        cur = flat                       # (N, n_poses, 4, 4)
        nxt = np.roll(flat, -1, axis=1)  # next pose, wrapping last -> first
    else:
        cur = flat[:, :-1]               # (N, n_poses-1, 4, 4)
        nxt = flat[:, 1:]

    R      = cur[:, :, :3, :3]   # (N, n_junc, 3, 3)
    t      = cur[:, :, :3,  3]   # (N, n_junc, 3)
    R_next = nxt[:, :, :3, :3]   # (N, n_junc, 3, 3)
    t_next = nxt[:, :, :3,  3]   # (N, n_junc, 3)
    Rt     = R.swapaxes(-1, -2)  # batch R^T

    N      = flat.shape[0]
    n_junc = n_poses if closed else n_poses - 1
    junctions = np.zeros((N, n_junc, 4, 4))
    junctions[:, :, :3, :3] = np.matmul(Rt, R_next)
    junctions[:, :, :3,  3] = np.einsum('...ij,...j->...i', Rt, t_next - t)
    junctions[:, :,  3,  3] = 1.0
    return junctions.reshape(batch_shape + (n_junc, 4, 4))


@cond_jit(nopython=True, cache=True)
def _junctions2parameters_kernel(flat: np.ndarray) -> np.ndarray:
    """JIT kernel: flat input (N, n_junc, 4, 4) → (N, n_junc, 6)."""
    N      = flat.shape[0]
    n_junc = flat.shape[1]
    out    = np.zeros((N, n_junc, 6))
    for i in range(N):
        for j in range(n_junc):
            out[i, j] = _se3_rotmat2euler(flat[i, j])
    return out


def junctions2parameters_optimized(junctions: np.ndarray) -> np.ndarray:
    """Numba-JIT version of junctions2parameters.  Accepts arbitrary leading batch dims.

    rotmat2euler has complex branching (near-identity / near-π) that makes
    numpy vectorisation impractical; numba JIT-to-JIT calls are the fastest path.

    Parameters
    ----------
    junctions : np.ndarray, shape (*batch, n_junc, 4, 4)

    Returns
    -------
    np.ndarray, shape (*batch, n_junc, 6)
    """
    batch_shape = junctions.shape[:-3]
    n_junc      = junctions.shape[-3]
    out = _junctions2parameters_kernel(junctions.reshape(-1, n_junc, 4, 4))
    return out.reshape(batch_shape + (n_junc, 6))


@cond_jit(nopython=True, cache=True)
def _poses2parameters_kernel(flat: np.ndarray, closed: bool) -> np.ndarray:
    """JIT kernel: flat input (N, n_poses, 4, 4) → (N, n_junc, 6).

    n_junc == n_poses if closed else n_poses-1.  When closed, the wrap-around
    junction (last pose → first pose) is appended.
    """
    N       = flat.shape[0]
    n_poses = flat.shape[1]
    n_junc  = n_poses if closed else n_poses - 1
    out     = np.zeros((N, n_junc, 6))
    for i in range(N):
        for j in range(n_poses - 1):
            gij = _se3_inverse(flat[i, j]) @ flat[i, j + 1]
            out[i, j] = _se3_rotmat2euler(gij)
        if closed:
            gij = _se3_inverse(flat[i, n_poses - 1]) @ flat[i, 0]
            out[i, n_poses - 1] = _se3_rotmat2euler(gij)
    return out


def poses2parameters_optimized(poses: np.ndarray, closed: bool = False) -> np.ndarray:
    """Numba-JIT version of poses2parameters.  Accepts arbitrary leading batch dims.

    Same rationale as junctions2parameters_optimized: numba beats numpy
    because the SE3→parameters conversion (rotmat2euler) branches heavily.

    Parameters
    ----------
    poses : np.ndarray, shape (*batch, n_poses, 4, 4)
    closed : bool, default False
        If True, treat the chain as topologically closed (append the
        last-to-first junction), yielding n_poses parameters instead of
        n_poses-1.

    Returns
    -------
    np.ndarray, shape (*batch, n_junc, 6)
        n_junc == n_poses if closed else n_poses-1.
    """
    batch_shape = poses.shape[:-3]
    n_poses     = poses.shape[-3]
    n_junc      = n_poses if closed else n_poses - 1
    out = _poses2parameters_kernel(poses.reshape(-1, n_poses, 4, 4), closed)
    return out.reshape(batch_shape + (n_junc, 6))


def _batch_euler2rotmat(vrot: np.ndarray) -> np.ndarray:
    """Vectorised Rodrigues formula: (..., 3) → (..., 3, 3).

    Handles the two special cases (near-zero rotation, small-angle series)
    with np.where, keeping the computation fully in numpy/BLAS.
    """
    shape      = vrot.shape[:-1]
    flat       = vrot.reshape(-1, 3)     # (N, 3)
    N          = flat.shape[0]

    Om_sq = flat[:, 0]**2 + flat[:, 1]**2 + flat[:, 2]**2  # (N,)
    Om    = np.sqrt(Om_sq)                                   # (N,)

    # ---- coefficients A = sinc(Om) and B = (1-cos)/Om^2 ------------------
    use_series = (Om < _EULER_SERIES_SMALL) & (Om >= _EULER_EPSILON)
    use_full   = Om >= _EULER_SERIES_SMALL

    Om2 = Om_sq
    Om4 = Om2 * Om2

    A_ser  = 1.0 - Om2 / 6.0  + Om4 / 120.0
    B_ser  = 0.5 - Om2 / 24.0 + Om4 / 720.0
    A_full = np.where(use_full, np.sin(Om)         / np.where(use_full, Om,    1.0), 0.0)
    B_full = np.where(use_full, (1.0 - np.cos(Om)) / np.where(use_full, Om_sq, 1.0), 0.0)

    A = np.where(use_series, A_ser, A_full)
    B = np.where(use_series, B_ser, B_full)

    # ---- Rodrigues fill ----------------------------------------------------
    x, y, z   = flat[:, 0], flat[:, 1], flat[:, 2]
    xx, yy, zz = x*x,  y*y,  z*z
    xy, xz, yz = x*y,  x*z,  y*z

    R = np.empty((N, 3, 3))
    R[:, 0, 0] = 1.0 - B*(yy + zz)
    R[:, 1, 1] = 1.0 - B*(xx + zz)
    R[:, 2, 2] = 1.0 - B*(xx + yy)
    R[:, 0, 1] = B*xy - A*z
    R[:, 1, 0] = B*xy + A*z
    R[:, 0, 2] = B*xz + A*y
    R[:, 2, 0] = B*xz - A*y
    R[:, 1, 2] = B*yz - A*x
    R[:, 2, 1] = B*yz + A*x

    # near-zero rotations → identity (already correct because B=A=0 for Om=0)
    return R.reshape(shape + (3, 3))


def parameters2junctions_optimized(params: np.ndarray) -> np.ndarray:
    """Vectorised version of parameters2junctions.  Accepts arbitrary leading batch dims.

    The Rodrigues formula (euler2rotmat) is cleanly vectorisable with
    np.where-based masking of the two threshold cases, making numpy the
    superior choice over numba here.

    Parameters
    ----------
    params : np.ndarray, shape (*batch, n_junc, 6)
        *batch may be empty.

    Returns
    -------
    np.ndarray, shape (*batch, n_junc, 4, 4)
    """
    # _batch_euler2rotmat already handles arbitrary leading dims via reshape internally
    vrot   = params[..., :3]   # (*batch, n_junc, 3)
    vtrans = params[..., 3:]   # (*batch, n_junc, 3)

    R = _batch_euler2rotmat(vrot)   # (*batch, n_junc, 3, 3)

    shape = params.shape[:-1]
    junctions = np.zeros(shape + (4, 4))
    junctions[..., :3, :3] = R
    junctions[..., :3,  3] = vtrans
    junctions[...,  3,  3] = 1.0
    return junctions


def junctions2dynamics_optimized(
    junctions: np.ndarray,
    static_junctions: np.ndarray | None = None,
    static_params: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorised version of junctions2dynamics.  Accepts arbitrary leading batch dims.

    Parameters
    ----------
    junctions        : np.ndarray, shape (*batch, n_junc, 4, 4)
    static_junctions : np.ndarray | None, shape (n_junc, 4, 4)
        Groundstate junctions (not batched). If None, computed as the mean
        over all batch dimensions.
    static_params    : np.ndarray | None, shape (n_junc, 6)

    Returns
    -------
    np.ndarray, shape (*batch, n_junc, 4, 4)
    """
    if static_junctions is None:
        if static_params is not None:
            static_junctions = parameters2junctions_optimized(static_params)
        else:
            params = junctions2parameters_optimized(junctions)
            # Average over ALL batch dimensions → (n_junc, 6)
            static_params = np.mean(params.reshape(-1, params.shape[-2], params.shape[-1]), axis=0)
            static_junctions = parameters2junctions_optimized(static_params)

    # static_junctions: (n_junc, 4, 4) — flatten junctions over all batch dims
    batch_shape = junctions.shape[:-3]
    n_junc      = junctions.shape[-3]
    flat        = junctions.reshape(-1, n_junc, 4, 4)   # (N, J, 4, 4)
    s           = static_junctions.reshape(n_junc, 4, 4) # (J, 4, 4)

    R_s   = s[:, :3, :3]           # (J, 3, 3)
    t_s   = s[:, :3,  3]           # (J, 3)
    R_s_T = R_s.swapaxes(-1, -2)   # (J, 3, 3)  R_s^T

    # inv(g_s) @ g  =  [[R_s^T @ R,  R_s^T @ (t - t_s)], [0, 1]]
    R_dyn = np.matmul(R_s_T[np.newaxis], flat[:, :, :3, :3])          # (N, J, 3, 3)
    t_dyn = np.einsum('...ij,...j->...i', R_s_T[np.newaxis],
                      flat[:, :, :3, 3] - t_s[np.newaxis])             # (N, J, 3)

    dyn = np.zeros_like(flat)
    dyn[:, :, :3, :3] = R_dyn
    dyn[:, :, :3,  3] = t_dyn
    dyn[:, :,  3,  3] = 1.0
    return dyn.reshape(batch_shape + (n_junc, 4, 4))
