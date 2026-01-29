from __future__ import annotations

import sys, os
import argparse
import numpy as np
from ..SO3 import so3

def extract_se3_parameters(poses: np.ndarray) -> np.ndarray:
    """Extract SE3 parameters from a sequence of poses.
    Parameters:
    -----------
    poses : np.ndarray
        Array of shape (n_snapshots, n_poses, 4, 4) containing the SE3 poses.
    Returns:
    np.ndarray
        Array of shape (n_snapshots, n_poses-1, 6) containing the extracted SE3 parameters.
    """
    n_snap  = poses.shape[0]
    n_poses = poses.shape[1]
    params = np.zeros((n_snap, n_poses-1, 6))
    for i in range(n_snap):
        for j in range(n_poses-1):
            gij = so3.se3_inverse(poses[i,j]) @ poses[i,j+1]
            params[i,j] = so3.se3_rotmat2euler(gij)
    return params

