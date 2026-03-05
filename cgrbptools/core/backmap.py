from __future__ import annotations

import sys
import numpy as np
from scipy.interpolate import splprep, splev

from ..SO3 import so3
from .topology import CGRBPTopology
# from .lmp_conf import CGRBPConf


CGRBP_BACKMAP_CLOSURE_MAX_DISTANCE_PER_BP = 0.68
CGRBP_BACKMAP_CLOSURE_OPEN_ADD_ROT = 0.6

def spline_interpolate_constant_arclength_segments(
    pts: np.ndarray,
    composite_size: int,
    k: int = 3,
    s_smooth: float = 0.0,
) -> np.ndarray:
    """
    Interpolate a 3D curve through pts using a spline and resample so that
    between each consecutive pair of original points there are (composite_size-1)
    additional points, equally spaced in (approximate) arc length along the spline.

    Output has shape ((N-1)*composite_size + 1, 3) and satisfies:
        curve[i*composite_size] == pts[i]  (up to numerical floating equality; endpoints are enforced)

    Parameters
    ----------
    pts : (N,3) array
    composite_size : int
        Number of samples per original segment (including the segment start).
        I.e., inserts composite_size-1 points between consecutive originals.
    k : int
        Spline degree (3=cubic). Requires N > k.
    s_smooth : float
        Smoothing factor for splprep. 0.0 interpolates exactly.

    Returns
    -------
    curve : ((N-1)*composite_size + 1, 3) array
    """
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("pts must be an (N,3) array")
    if composite_size < 1:
        raise ValueError("composite_size must be >= 1")

    N = pts.shape[0]
    if N < 2:
        return pts.copy()
    if N <= k:
        raise ValueError(f"Need N > k for splprep. Got N={N}, k={k}")

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Initial parameter guess: normalized cumulative polyline length
    d = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
    t = np.r_[0.0, np.cumsum(d)]
    t = t / t[-1] if t[-1] > 0 else t

    # Fit spline (interpolating if s_smooth=0)
    tck, u_pts = splprep([x, y, z], u=t, s=s_smooth, k=k)

    # Build dense map u -> s(u) (spline arc length approx)
    n_dense = max(5000, 100 * N)
    u_dense = np.linspace(0.0, 1.0, n_dense)
    xd, yd, zd = splev(u_dense, tck)
    ds = np.sqrt(np.diff(xd)**2 + np.diff(yd)**2 + np.diff(zd)**2)
    s_dense = np.r_[0.0, np.cumsum(ds)]

    # Helper: arc length at any u via dense interpolation
    def s_of_u(u):
        return np.interp(u, u_dense, s_dense)

    curve_parts = []
    eps = 1e-15

    for i in range(N - 1):
        u0, u1 = float(u_pts[i]), float(u_pts[i + 1])
        s0, s1 = float(s_of_u(u0)), float(s_of_u(u1))

        if abs(s1 - s0) < eps:
            # Degenerate segment (points coincide / zero length along spline)
            u_seg = np.linspace(u0, u1, composite_size + 1)
        else:
            # Targets equally spaced in arc length within this segment
            s_targets = np.linspace(s0, s1, composite_size + 1)
            u_seg = np.interp(s_targets, s_dense, u_dense)

            # Enforce exact endpoints so originals land exactly at i*composite_size
            u_seg[0] = u0
            u_seg[-1] = u1

        xs, ys, zs = splev(u_seg, tck)
        seg_curve = np.column_stack([xs, ys, zs])

        # Concatenate without duplicating the shared point at segment boundaries
        if i > 0:
            seg_curve = seg_curve[1:]

        curve_parts.append(seg_curve)

    curve = np.vstack(curve_parts)

    # Hard-enforce exact originals at the required indices (optional but matches your requirement)
    curve[::composite_size] = pts

    return curve

def linear_interpolate_constant_arclength_segments(
    pts: np.ndarray,
    composite_size: int,
) -> np.ndarray:
    """
    Piecewise-linear interpolation between 3D points with constant arc-length
    spacing per segment.

    Between pts[i] and pts[i+1], inserts composite_size-1 equally spaced points.
    Output shape: ((N-1)*composite_size + 1, 3)
    Ensures: curve[i*composite_size] == pts[i]

    Parameters
    ----------
    pts : (N,3) array
        Input tracepoints.
    composite_size : int
        Number of samples per original segment (including the start point).

    Returns
    -------
    curve : ((N-1)*composite_size + 1, 3) array
    """
    pts = np.asarray(pts, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("pts must be an (N,3) array")
    if composite_size < 1:
        raise ValueError("composite_size must be >= 1")

    N = pts.shape[0]
    if N < 2:
        return pts.copy()
    curve = np.empty(((N - 1) * composite_size + 1, 3), dtype=float)
    for i in range(N - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        alphas = np.linspace(0.0, 1.0, composite_size + 1)
        seg = (1.0 - alphas[:, None]) * p0 + alphas[:, None] * p1
        if i > 0:
            seg = seg[1:]
        start = i * composite_size
        curve[start:start + seg.shape[0]] = seg
    # Enforce exact originals
    curve[::composite_size] = pts
    return curve


def dna_backmap(
    conf: "CGRBPConf",
    # topology: CGRBPTopology | None = None,
    composite_size: int | None = None,
    spline_interpolation: bool = True,
    closed: bool | None = None,
    verbose: bool = False,
    ) -> np.ndarray:
    """Backmap coarse-grained rigid base pair configuration to base pair step configuration."""
    
    if not conf.topology_set():
        raise ValueError("CGRBPConf must have a topology set for backmapping.")
    topology = conf.topology
    
    if composite_size is None and topology is None:
        raise ValueError("Either composite_site or topology must be provided")
    if topology is not None:
        composite_size = topology.chars_per_atom    
    # check if topology and conf are compatible
    if topology is not None:
        if len(conf.poses) != topology.nbp:
            raise ValueError("Incompatible conf and topology: number of CGRBP poses does not match topology.")
    
    if verbose:
        print(f'Backmapping configuration with {len(conf.poses)} CGRBP poses at composite size {composite_size}.')
    
    conf_poses = conf.poses_in_nm()
    
    _closed = False
    if topology is not None:
        _closed = topology.closed
    if closed is not None:
        _closed = closed

    def _avg_discretization(poses: np.ndarray) -> float:
        total_dist = 0.0
        nsteps = len(poses) - 1
        for i in range(nsteps):
            total_dist += np.linalg.norm(poses[i+1,:3,3] - poses[i,:3,3])
        return total_dist / nsteps
    
    def _append_single_pose(poses: np.ndarray, composite_size: int) -> np.ndarray:
        avg_disc = _avg_discretization(poses)
        add_pose = poses[-1].copy()
        add_pose[:3,3] += add_pose[:3,2] * avg_disc
        R = so3.euler2rotmat((0.0, 0.0, CGRBP_BACKMAP_CLOSURE_OPEN_ADD_ROT*composite_size))
        add_pose[:3,:3] = add_pose[:3,:3] @ R
        return np.concatenate((poses, add_pose[None]), axis=0)

    if _closed:
        if verbose:
            print(" Backmapping closed configuration")
        # check if first and last poses are at right distance
        dist = np.linalg.norm(conf_poses[0,:3,3] - conf_poses[-1,:3,3])
        if dist <= CGRBP_BACKMAP_CLOSURE_MAX_DISTANCE_PER_BP*composite_size:
            if verbose:
                print(f' First and last pose are within {dist:.2f} nm, closing the loop.')
            cg_poses = np.concatenate((conf_poses, conf_poses[0][None]), axis=0)
        else:
            if verbose:
                print(" First and last pose are not within closure distance, extending along last tangent.")
            # avg_disc = _avg_discretization(conf_poses)
            # add_pose = conf_poses[-1].copy()
            # add_pose[:3,3] += add_pose[:3,2] * avg_disc
            # R = so3.euler2rotmat((0.0, 0.0, CGRBP_BACKMAP_CLOSURE_OPEN_ADD_ROT*composite_size))
            # add_pose[:3,:3] = add_pose[:3,:3] @ R
            # cg_poses = np.concatenate((conf_poses, add_pose[None]), axis=0)
            cg_poses = _append_single_pose(conf_poses, composite_size)
    else:  
        if verbose:
            print(" Backmapping open configuration")
        if (len(conf_poses)-1)*composite_size + 1 == len(topology.sequence):
            cg_poses = conf_poses.copy()
        else:
            if verbose: 
                print(" Extending last pose to match number of base pairs.")
            # avg_disc = _avg_discretization(conf_poses)
            # add_pose = conf_poses[-1].copy()
            # add_pose[:3,3] += add_pose[:3,2] * avg_disc
            # R = so3.euler2rotmat((0.0, 0.0, CGRBP_BACKMAP_CLOSURE_OPEN_ADD_ROT*composite_size))
            # add_pose[:3,:3] = add_pose[:3,:3] @ R
            # cg_poses = np.concatenate((conf_poses, add_pose[None]), axis=0)
            cg_poses = _append_single_pose(conf_poses, composite_size)
    pts = cg_poses[:, :3, 3].copy()
    
    if spline_interpolation:
        if verbose:
            print(" Using spline interpolation for backmapping.")
        curve = spline_interpolate_constant_arclength_segments(pts, composite_size)
    else:
        if verbose:
            print(" Using linear interpolation for backmapping.")    
        curve = linear_interpolate_constant_arclength_segments(pts, composite_size)
        
    poses = np.zeros((len(curve),4,4)) 
    if closed:
        poses[-1] = poses[0]
    if not _closed:
        poses[-1] = conf_poses[-1]
    nsteps = len(pts)-1

    def _project_to_plane(R: np.ndarray):
        v1, v2, tangent = R[:,0], R[:,1], R[:,2]
        utan = tangent / np.linalg.norm(tangent)
        dot1 = np.dot(v1, utan)
        dot2 = np.dot(v2, utan)
        if dot1**2 < dot2**2:
            p1 = v1 - dot1 * utan
            p1 = p1 / np.linalg.norm(p1)
            p2 = np.cross(utan, p1)
        else:
            p2 = v2 - dot2 * utan
            p2 = p2 / np.linalg.norm(p2)
            p1 = np.cross(p2, utan)
        return np.column_stack((p1, p2, utan))

    for i in range(len(cg_poses)):
        poses[i*composite_size] = cg_poses[i]

    gauge_twist = 2*np.pi / 10.5 * composite_size
    # print(f'Aimed twist per composite: {gauge_twist*180/np.pi:.2f} deg')    
    for i in range(nsteps):
        idx = i*composite_size        
        R1 = cg_poses[i,:3,:3]
        R2 = cg_poses[i+1,:3,:3]    
        mid = so3.midstep(R1,R2)
        accu_tw = 0.0
        for j in range(1,composite_size):
            id = idx + j
            tangent = curve[id+1] - curve[id]
            tangent /= np.linalg.norm(tangent)
            R = mid.copy()
            R[:,2] = tangent
            R_proj = _project_to_plane(R)
            poses[id,:3,3] = curve[id]
            poses[id,:3,:3] = R_proj
            poses[id,3,3] = 1.0
            accu_tw += so3.rotmat2euler(poses[id-1,:3,:3].T @ poses[id,:3,:3])[2]  

        accu_tw += so3.rotmat2euler(poses[idx+composite_size-1,:3,:3].T @ poses[idx+composite_size,:3,:3])[2] 
        aim_tw = accu_tw + np.round((gauge_twist - accu_tw) / (2*np.pi)) * 2*np.pi
        tw_per_step = aim_tw / composite_size
        for j in range(1,composite_size):
            id = idx + j
            T1 = poses[id-1,:3,:3]
            T2 = poses[id,:3,:3]
            Om = so3.rotmat2euler( T1.T @ T2 )
            poses[id,:3,:3] = T1 @ so3.euler2rotmat((Om[0],Om[1],tw_per_step))
        
    if _closed:
        poses = poses[:-1]
    else:
        poses = poses[:len(topology.sequence)]
    return poses