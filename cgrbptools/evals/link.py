from __future__ import annotations

import numpy as np

# from ..SO3.so3.pyConDec.pycondec import cond_jit
# from ..SO3 import so3
from .se3 import poses2junctions, junctions2dynamics, junctions2parameters
from .PyLk import pylk
from ..io.console_output import print_progress


def poses2twist(
    poses: np.ndarray,
    closed: bool = False,
    groundstate: np.ndarray | None = None,
) -> np.ndarray:
    """Total excess twist (in turns) accumulated along the chain.

    The twist is measured relative to a static reference via junctions2dynamics.

    Parameters
    ----------
    poses : np.ndarray, shape (*batch, n_poses, 4, 4)
    closed : bool
        Whether the chain is a closed loop (adds the wrap-around junction).
    groundstate : np.ndarray | None, shape (n_junc, 6)
        Intrinsic groundstate junction parameters used as the reference for the
        excess twist. If None, junctions2dynamics falls back to the trajectory
        mean -- which centres the excess twist at zero by construction and makes
        the result depend on the ensemble rather than on the intrinsic shape.
        Pass the known groundstate (e.g. topol.groundstate) for a reproducible,
        physically meaningful reference.
    """
    junc = poses2junctions(poses, closed=closed)
    dyn = junctions2dynamics(junc, static_params=groundstate)
    dparams = junctions2parameters(dyn)
    twist = np.sum(dparams[..., 2], axis=-1) / (2 * np.pi)
    return twist

def poses2writhe(
    poses: np.ndarray,
    closed: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    pos = poses[..., :3, 3]                 # (*batch, n_poses, 3)
    batch_shape = pos.shape[:-2]
    n_poses     = pos.shape[-2]
    flat        = pos.reshape(-1, n_poses, 3)   # (N, n_poses, 3)
    N           = flat.shape[0]

    if verbose:
        print("Computing writhe of trajectory...")
    writhe = np.zeros(N)
    for i in range(N):
        writhe[i] = pylk.writhe(flat[i], closed=closed)
        if verbose:
            print_progress(i + 1, N, prefix='Computing writhe')
    return writhe.reshape(batch_shape)

def poses2link(
    poses: np.ndarray,
    closed: bool = False,
    groundstate: np.ndarray | None = None,
    verbose: bool = False,
) -> np.ndarray:
    twist = poses2twist(poses, closed=closed, groundstate=groundstate)
    writhe = poses2writhe(poses, closed=closed, verbose=verbose)
    link = twist + writhe

    data = np.stack([twist, writhe, link], axis=-1)
    return data