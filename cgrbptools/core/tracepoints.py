"""
Initial configurations from tracepoints.

A smooth curve is interpolated through the tracepoints and poses are placed along it, spaced in
proportion to the intrinsic rise of the steps. The triads follow the curve and carry the intrinsic
twist of the steps. All other ground-state coordinates (tilt, roll, shift and slide) are ignored.

Since only rise and twist enter, every step of the reference chain is a screw motion along the
third triad axis, and the construction does not depend on whether translations are expressed in
the midstep or in the triad frame. Twist is the third component of the rotation (Euler) vector of
a step, as everywhere in cgrbptools.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicHermiteSpline, splev, splprep
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from ..SO3 import so3
from .conf_checks import bond_vectors
from .conf_import import _listed, validate_poses

# Consecutive tracepoints closer than this fraction of the longest tracepoint spacing coincide
# (for closed curves a last tracepoint repeating the first one is removed)
TRACE_COINCIDE_FRACTION = 1e-8
# Tracepoints whose second singular value is below this fraction of the first lie on a line; a
# tangent with a component perpendicular to that line above this value leads off the line
TRACE_COLLINEAR_FRACTION = 1e-8
# Where the unconstrained spline is slower than this (in units of the chord-length parameter, in
# which a regular curve has speed close to 1), a prescribed tangent is scaled with speed 1 instead
TRACE_MIN_REFERENCE_SPEED = 0.1
# Largest angle (rad) between the third axis of first_triad and a tangent given at tracepoint 0
TRACE_FIRST_TRIAD_TANGENT_TOL = np.radians(0.1)
# Weight of the end points of open chains in the smoothing spline, in units of sqrt(N): pins them
# to within about a hundredth of the smoothing distance
TRACE_SMOOTHING_END_WEIGHT = 100.0
# Arc-length table: Gauss-Legendre sub-intervals per spline segment (at least) and per pose
TRACE_ARCLENGTH_MIN_SUBDIVISIONS = 16
TRACE_ARCLENGTH_SUBDIVISIONS_PER_POSE = 4
# Placement: relative spread of distance/rise over the steps at which the placement is accepted.
# The tolerance is raised to the precision attainable with the coordinates (PRECISION times the
# machine epsilon times the largest coordinate over the shortest distance). A placement that
# still has not converged after MAX_ITER iterations is accepted if its spread is below ACCEPT.
TRACE_PLACEMENT_TOL = 1e-10
TRACE_PLACEMENT_PRECISION = 1e3
TRACE_PLACEMENT_MAX_ITER = 100
TRACE_PLACEMENT_ACCEPT = 1e-6
# A step whose straight-line distance is below this fraction of the arc length it spans doubles
# back within the step (2/pi: a half circle)
TRACE_MIN_CHORD_ARC_RATIO = 2.0 / np.pi
# Twist correction: tolerance (rad) and iteration limit of the per-step inversion and closure
TRACE_TWIST_TOL = 1e-13
TRACE_TWIST_MAX_ITER = 50
# Largest deviation of the excess link of a closed chain from an integer
TRACE_INTEGER_TOL = 1e-9
# Stretch beyond max_fene (relative to the rise) attributed to rounding
TRACE_FENE_TOL = 1e-9
# Warnings: stretch |distance/rise - 1| without rescaling, bend (rad) between consecutive poses,
# non-neighbouring poses closer than this fraction of the median distance, and closure twist
# (fraction of pi) beyond which both closing directions are about equally close
TRACE_WARN_STRETCH = 0.1
TRACE_WARN_BEND = 0.5 * np.pi
TRACE_OVERLAP_FRACTION = 0.5
TRACE_AMBIGUOUS_CLOSURE_FRACTION = 0.9

# 3-point Gauss-Legendre rule on [-1, 1]
_GAUSS_NODES = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
_GAUSS_WEIGHTS = np.array([5.0, 8.0, 5.0]) / 9.0


@dataclass
class TracepointInfo:
    """
    Diagnostics of tracepoints_to_poses. Angles are in radians, lengths in the unit of the
    tracepoints.

    Attributes
    ----------
    stretch : float
        Distance between consecutive poses on the curve through the tracepoints divided by the
        intrinsic rise. It is the same for all steps.
    scale : float
        Factor by which the curve was scaled about the centroid of the tracepoints (1/stretch with
        rescale, otherwise 1).
    max_rise_deviation : float
        Largest absolute difference between the distance of consecutive poses and the rise.
    closure_twist : float
        Twist added in total to close the relaxed ring. The twist of every step is shifted by
        (closure_twist + 2 pi excess_link) / num_steps. 0 for open chains.
    excess_twist_per_step : float
        Twist added to every step for the excess link.
    max_bend : float
        Largest angle between the third axes of consecutive poses.
    max_twist_residual : float
        Largest difference between the twist of a step and its target (intrinsic twist plus the
        closure and excess twist per step).
    closest_pair : tuple[int, int] or None
        Closest pair of non-neighbouring poses (None if there is no such pair).
    closest_distance : float or None
        Distance of the closest pair.
    smoothing_rms : float
        Root-mean-square distance by which smoothing moved the tracepoints.
    warnings : list of str
        Warnings issued.
    """
    stretch: float
    scale: float
    max_rise_deviation: float
    closure_twist: float
    excess_twist_per_step: float
    max_bend: float
    max_twist_residual: float
    closest_pair: tuple[int, int] | None
    closest_distance: float | None
    smoothing_rms: float
    warnings: list[str] = field(default_factory=list)


###################################################################################################
# Input

def parse_groundstate(
    groundstate: np.ndarray | float,
    num_poses: int | None = None,
    closed: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Intrinsic twist and rise of every step from a ground state.

    The ground state is given per step with 1 (rise), 2 (twist, rise) or 6 (tilt, roll, twist,
    shift, slide, rise) coordinates. Coordinates other than twist and rise are ignored.

    - A scalar is a rise; a 1D array holds the coordinates of a single step. Both are applied
      to every step and require num_poses.
    - A 2D array (num_steps, d) holds one step per row. A single row together with num_poses is
      applied to every step. Otherwise the rows define the number of poses: num_steps + 1 for
      open and num_steps for closed chains; num_poses, if given, has to agree.

    Parameters
    ----------
    groundstate : float or np.ndarray
        Ground state, twist in radians.
    num_poses : int, optional
        Number of poses.
    closed : bool, optional
        Closed chain: the last step connects the last pose to the first one. Default is False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Twist and rise, each with shape (num_steps,).

    Raises
    ------
    ValueError
        If the ground state cannot be interpreted or does not match num_poses.
    """
    gs = np.asarray(groundstate, dtype=float)
    single = gs.ndim < 2
    if gs.ndim == 0:
        gs = gs.reshape(1, 1)
    elif gs.ndim == 1:
        if len(gs) not in (1, 2, 6):
            raise ValueError(
                f'A 1D ground state holds the coordinates of a single step and needs 1 (rise), 2 '
                f'(twist, rise) or 6 entries, got {len(gs)}. Pass per-step values as a 2D array of '
                f'shape (num_steps, d).'
            )
        gs = gs[None]
    elif gs.ndim != 2 or gs.shape[1] not in (1, 2, 6):
        raise ValueError(
            f'The ground state must have shape (num_steps, d) with d = 1 (rise), 2 (twist, rise) '
            f'or 6 (tilt, roll, twist, shift, slide, rise), got shape {gs.shape}.'
        )
    if len(gs) == 0:
        raise ValueError('The ground state is empty.')
    if not np.all(np.isfinite(gs)):
        raise ValueError('The ground state contains non-finite values.')

    chain = 'closed' if closed else 'open'
    if num_poses is not None:
        if isinstance(num_poses, bool) or not isinstance(num_poses, (int, np.integer)):
            raise TypeError(f'num_poses must be an integer, got {num_poses!r}.')
        num_steps = int(num_poses) if closed else int(num_poses) - 1
        if single or len(gs) == 1:
            gs = np.repeat(gs, max(num_steps, 0), axis=0)
        elif len(gs) != num_steps:
            raise ValueError(
                f'The ground state holds {len(gs)} steps, which make '
                f'{len(gs) if closed else len(gs) + 1} poses for an {chain} chain, but num_poses '
                f'is {num_poses}.'
            )
    elif single:
        raise ValueError(
            'A ground state given for a single step is applied to every step and requires '
            'num_poses.'
        )

    num_poses = len(gs) if closed else len(gs) + 1
    min_poses = 3 if closed else 2
    if num_poses < min_poses:
        raise ValueError(f'An {chain} chain needs at least {min_poses} poses, got {num_poses}.')

    dim = gs.shape[1]
    rise = gs[:, {1: 0, 2: 1, 6: 5}[dim]].copy()
    twist = np.zeros(len(gs)) if dim == 1 else gs[:, {2: 0, 6: 2}[dim]].copy()
    if np.any(rise <= 0):
        raise ValueError(
            f'The rise has to be positive (0-based steps with rise <= 0: '
            f'{_listed(np.flatnonzero(rise <= 0))}).'
        )
    return twist, rise


def _parse_tracepoints(
    tracepoints: np.ndarray,
    tangent_persistence: float | np.ndarray,
    closed: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions, tangents (NaN rows where none is given) and tangent persistence per tracepoint."""
    arr = np.asarray(tracepoints, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 3:
        points = arr.copy()
        tangents = np.full_like(points, np.nan)
    elif arr.ndim == 3 and arr.shape[1:] == (2, 3):
        points = arr[:, 0].copy()
        tangents = arr[:, 1].copy()
    else:
        raise ValueError(
            f'tracepoints must have shape (N, 3) (positions) or (N, 2, 3) (positions and '
            f'tangents), got shape {arr.shape}.'
        )
    if len(points) < 2:
        raise ValueError(f'At least 2 tracepoints are required, got {len(points)}.')
    if not np.all(np.isfinite(points)):
        raise ValueError('The positions of the tracepoints contain non-finite values.')

    nan = np.isnan(tangents)
    partial = np.flatnonzero(nan.any(axis=1) & ~nan.all(axis=1))
    if len(partial) > 0:
        raise ValueError(
            f'Tangents have to be given completely or marked as free with NaN in all three '
            f'components (0-based tracepoints with partly NaN tangents: {_listed(partial)}).'
        )
    given = ~nan.all(axis=1)
    if not np.all(np.isfinite(tangents[given])):
        raise ValueError('The tangents contain infinite values.')
    zero = np.flatnonzero(given & (np.linalg.norm(np.nan_to_num(tangents), axis=1) == 0.0))
    if len(zero) > 0:
        raise ValueError(
            f'Tangents must not have zero length; mark free tangents with NaN (0-based '
            f'tracepoints: {_listed(zero)}).'
        )

    persistence = np.asarray(tangent_persistence, dtype=float)
    if persistence.ndim == 0:
        persistence = np.full(len(points), float(persistence))
    elif persistence.shape != (len(points),):
        raise ValueError(
            f'tangent_persistence must be a scalar or have one entry per tracepoint '
            f'({len(points)}), got shape {persistence.shape}.'
        )
    else:
        persistence = persistence.copy()
    if not np.all(np.isfinite(persistence)) or np.any(persistence <= 0):
        raise ValueError('tangent_persistence has to be positive and finite.')

    # closed curves are often given with the first tracepoint repeated at the end
    if closed and len(points) > 2:
        spacing = np.linalg.norm(np.diff(points, axis=0), axis=1).max()
        if np.linalg.norm(points[-1] - points[0]) <= TRACE_COINCIDE_FRACTION * spacing:
            if not given[0] and given[-1]:
                tangents[0] = tangents[-1]
                persistence[0] = persistence[-1]
            points, tangents, persistence = points[:-1], tangents[:-1], persistence[:-1]
    return points, tangents, persistence


def _check_curve_geometry(points: np.ndarray, tangents: np.ndarray, closed: bool) -> None:
    """Reject coinciding consecutive tracepoints and closed curves that cannot form a loop."""
    chords = np.linalg.norm(bond_vectors(points, closed), axis=1)
    coincide = np.flatnonzero(chords <= TRACE_COINCIDE_FRACTION * chords.max())
    if len(coincide) > 0:
        raise ValueError(
            f'Consecutive tracepoints coincide (0-based index i stands for tracepoints i and '
            f'i+1{", the last one wrapping around to 0" if closed else ""}: {_listed(coincide)}).'
        )
    if not closed:
        return

    given = ~np.isnan(tangents[:, 0])
    if len(points) == 2 and not given.all():
        raise ValueError(
            'A closed curve through 2 tracepoints needs tangents at both of them; otherwise it '
            'runs back and forth along a line. Provide at least 3 tracepoints or both tangents.'
        )
    _, sv, vt = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    if len(sv) < 2 or sv[1] <= TRACE_COLLINEAR_FRACTION * sv[0]:
        unit = tangents[given] / np.linalg.norm(tangents[given], axis=1)[:, None]
        off_line = np.linalg.norm(unit - np.outer(unit @ vt[0], vt[0]), axis=1)
        if not np.any(off_line > TRACE_COLLINEAR_FRACTION):
            raise ValueError(
                'The tracepoints of the closed curve lie on a straight line, so the curve would '
                'run back and forth along it. Provide tracepoints off the line or tangents leading '
                'off it.'
            )


def _first_triad_tangent(first_triad: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Third axis of first_triad, checked against a tangent given at tracepoint 0."""
    axis = first_triad[:, 2].copy()
    if not np.isnan(tangent[0]):
        cos = np.dot(tangent, axis) / np.linalg.norm(tangent)
        angle = np.arccos(np.clip(cos, -1.0, 1.0))
        if angle > TRACE_FIRST_TRIAD_TANGENT_TOL:
            raise ValueError(
                f'The third axis of first_triad and the tangent at tracepoint 0 point in different '
                f'directions ({np.degrees(angle):.3g} degrees apart, tolerance '
                f'{np.degrees(TRACE_FIRST_TRIAD_TANGENT_TOL):g} degrees). Give only one of them or '
                f'make them agree.'
            )
    return axis


###################################################################################################
# Curve

def _smooth_tracepoints(points: np.ndarray, smoothing: float, closed: bool) -> np.ndarray:
    """
    Tracepoints moved onto a cubic smoothing spline that misses them by about smoothing (root mean
    square). The end points of open curves stay in place.
    """
    num = len(points)
    if num < 4:
        raise ValueError(
            f'Smoothing fits a cubic smoothing spline, which needs at least 4 tracepoints '
            f'({num} given). Pass smoothing=None for fewer tracepoints.'
        )
    if smoothing == 0:
        return points.copy()
    ext = np.vstack([points, points[:1]]) if closed else points
    u = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ext, axis=0), axis=1))]
    weights = np.ones(len(ext))
    if not closed:
        weights[[0, -1]] = TRACE_SMOOTHING_END_WEIGHT * np.sqrt(num)
    tck, _ = splprep(ext.T, u=u, w=weights, s=num * smoothing**2, k=3, per=int(closed), quiet=1)
    smoothed = np.column_stack(splev(u[:num], tck))
    if not closed:
        smoothed[[0, -1]] = points[[0, -1]]
    return smoothed


def _c2_slopes(u: np.ndarray, points: np.ndarray, closed: bool, fixed: np.ndarray) -> np.ndarray:
    """
    Slopes at the knots of the cubic Hermite spline through points with knots u.

    Rows of fixed that are not NaN prescribe the slope. At the other knots the second derivative
    is continuous, and free ends of open curves have zero second derivative (natural spline).
    For closed curves u has one entry more than points, the curve returning to points[0].
    """
    num = len(points)
    idx = np.arange(num)
    h = np.diff(u)
    secant = bond_vectors(points, closed) / h[:, None]
    is_fixed = ~np.isnan(fixed[:, 0])

    rows, cols, vals = [idx[is_fixed]], [idx[is_fixed]], [np.ones(is_fixed.sum())]
    rhs = np.zeros((num, 3))
    rhs[is_fixed] = fixed[is_fixed]

    # continuity of the second derivative at knot i (segments i-1 and i)
    inner = ~is_fixed if closed else ~is_fixed & (idx > 0) & (idx < num - 1)
    i = idx[inner]
    left, right = (i - 1) % num, i % len(h)
    hl, hr = h[left], h[right]
    rows += [i, i, i]
    cols += [left, i, (i + 1) % num]
    vals += [1.0 / hl, 2.0 * (1.0 / hl + 1.0 / hr), 1.0 / hr]
    rhs[i] = 3.0 * (secant[left] / hl[:, None] + secant[right] / hr[:, None])

    if not closed:
        if not is_fixed[0]:
            rows += [[0, 0]]
            cols += [[0, 1]]
            vals += [[2.0, 1.0]]
            rhs[0] = 3.0 * secant[0]
        if not is_fixed[-1]:
            rows += [[num - 1, num - 1]]
            cols += [[num - 2, num - 1]]
            vals += [[1.0, 2.0]]
            rhs[-1] = 3.0 * secant[-1]

    # duplicate entries (closed curves through 2 tracepoints) are summed
    mat = csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(num, num))
    return np.asarray(spsolve(mat, rhs)).reshape(num, 3)


def _build_curve(
    points: np.ndarray,
    tangents: np.ndarray,
    persistence: np.ndarray,
    closed: bool,
) -> CubicHermiteSpline:
    """
    Curve through the tracepoints, parametrized by the cumulative distance between them.

    Without tangents this is the natural (open) or periodic (closed) cubic spline. A given tangent
    prescribes the direction of the curve at its tracepoint; the speed there is that of the spline
    without tangents, multiplied by the tangent persistence. The curve is C2 at the other
    tracepoints and C1 at those with tangents.
    """
    ext = np.vstack([points, points[:1]]) if closed else points
    u = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ext, axis=0), axis=1))]
    slopes = _c2_slopes(u, points, closed, np.full_like(points, np.nan))
    given = ~np.isnan(tangents[:, 0])
    if given.any():
        speed = np.linalg.norm(slopes, axis=1)
        speed[speed < TRACE_MIN_REFERENCE_SPEED] = 1.0
        fixed = np.full_like(points, np.nan)
        unit = tangents[given] / np.linalg.norm(tangents[given], axis=1)[:, None]
        fixed[given] = (persistence[given] * speed[given])[:, None] * unit
        slopes = _c2_slopes(u, points, closed, fixed)
    if closed:
        slopes = np.vstack([slopes, slopes[:1]])
    return CubicHermiteSpline(u, ext, slopes, axis=0)


def _arclength_table(curve: CubicHermiteSpline, num_poses: int) -> tuple[np.ndarray, np.ndarray]:
    """Curve parameter and arc length at the edges of a fine subdivision of the curve."""
    knots = curve.x
    h = np.diff(knots)
    counts = np.maximum(
        TRACE_ARCLENGTH_MIN_SUBDIVISIONS,
        np.ceil(TRACE_ARCLENGTH_SUBDIVISIONS_PER_POSE * num_poses * h / (knots[-1] - knots[0])),
    ).astype(int)
    seg = np.repeat(np.arange(len(h)), counts)
    frac = (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)) / np.repeat(counts, counts)
    edges = np.r_[knots[seg] + frac * h[seg], knots[-1]]

    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * np.diff(edges)
    nodes = mid[:, None] + half[:, None] * _GAUSS_NODES
    speed = np.linalg.norm(curve(nodes.ravel(), 1), axis=1).reshape(nodes.shape)
    return edges, np.r_[0.0, np.cumsum(half * (speed @ _GAUSS_WEIGHTS))]


###################################################################################################
# Placement

def _place_poses(
    curve: CubicHermiteSpline,
    rise: np.ndarray,
    closed: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Curve parameters and positions of the poses, spaced such that the straight-line distances of
    consecutive poses are proportional to the rise. The first pose sits at the start of the curve;
    the last one at its end (open) or one step before returning to the start (closed).

    Returns the parameters, the positions and the common ratio distance/rise.
    """
    edges, arclength = _arclength_table(curve, len(rise) + 1)
    weights = rise.copy()
    for _ in range(TRACE_PLACEMENT_MAX_ITER):
        s = np.r_[0.0, np.cumsum(weights)] * (arclength[-1] / weights.sum())
        u = np.interp(s, arclength, edges)
        u[-1] = edges[-1]
        pos = curve(u)
        dist = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        ratio = dist / np.diff(s)
        tight = np.flatnonzero(ratio < TRACE_MIN_CHORD_ARC_RATIO)
        if len(tight) > 0:
            raise ValueError(
                f'The curve turns too tightly for the step length: between consecutive poses it '
                f'doubles back on itself (0-based steps: {_listed(tight)}). Use more poses, '
                f'tracepoints describing a smoother path, or smoothing.'
            )
        stretch = dist / rise
        spread = (stretch.max() - stretch.min()) / stretch.mean()
        precision = TRACE_PLACEMENT_PRECISION * np.finfo(float).eps * np.abs(pos).max() / dist.min()
        if spread <= max(TRACE_PLACEMENT_TOL, precision):
            break
        weights = rise / ratio
    else:
        if spread > TRACE_PLACEMENT_ACCEPT:
            raise ValueError(
                'The poses could not be placed with distances proportional to the rise. The curve '
                'may turn too tightly for the step length; use more poses, tracepoints describing a '
                'smoother path, or smoothing.'
            )
    if closed:
        u, pos = u[:-1], pos[:-1]
    return u, pos, float(stretch.mean())


###################################################################################################
# Frames and twist

def _wrap(angle):
    """Angle(s) wrapped to [-pi, pi)."""
    return np.mod(np.asarray(angle) + np.pi, 2 * np.pi) - np.pi


def _reference_axes(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per pose two unit vectors completing z to a right-handed frame."""
    e = np.zeros_like(z)
    e[np.arange(len(z)), np.argmin(np.abs(z), axis=1)] = 1.0
    x = e - np.einsum('ni,ni->n', e, z)[:, None] * z
    x /= np.linalg.norm(x, axis=1)[:, None]
    return x, np.cross(z, x)


def _parallel_transport(
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    closed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transport angles and bend angles of the steps.

    The minimal rotation turning z[j] into z[j+1] maps the reference frame (x[j], y[j], z[j]) to
    the reference frame of pose j+1 rotated by the transport angle about z[j+1]. The bend angle is
    the angle between z[j] and z[j+1].
    """
    num = len(z) if closed else len(z) - 1
    z0, x0 = z[:num], x[:num]
    z1, x1, y1 = (np.roll(a, -1, axis=0)[:num] for a in (z, x, y))
    k = np.cross(z0, z1)
    cos = np.einsum('ni,ni->n', z0, z1)
    reverse = np.flatnonzero(cos <= -1.0 + 1e-12)
    if len(reverse) > 0:
        raise ValueError(
            f'The curve reverses its direction between consecutive poses (0-based steps: '
            f'{_listed(reverse)}). Use more poses or tracepoints describing a smoother path.'
        )
    # Rodrigues' formula for the minimal rotation, applied to x0
    v = cos[:, None] * x0 + np.cross(k, x0) + k * (np.einsum('ni,ni->n', k, x0) / (1.0 + cos))[:, None]
    alpha = np.arctan2(np.einsum('ni,ni->n', v, y1), np.einsum('ni,ni->n', v, x1))
    beta = np.arctan2(np.linalg.norm(k, axis=1), cos)
    return alpha, beta


def _bent_step_twist(beta: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """
    Twist (third component of the rotation vector) of a step that bends by beta about an axis
    perpendicular to the third triad axis and turns by tau about it.
    """
    c = np.cos(beta / 2)
    w = np.clip(c * np.cos(tau / 2), -1.0, 1.0)
    sin_half = np.sqrt(1.0 - w**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        twist = 2.0 * np.arccos(w) * c * np.sin(tau / 2) / sin_half
    return np.where(sin_half > 1e-12, twist, tau)


def _bent_step_twist_slope(beta: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Derivative of _bent_step_twist with respect to tau."""
    c = np.cos(beta / 2)
    sx, cx = np.sin(tau / 2), np.cos(tau / 2)
    w = np.clip(c * cx, -1.0, 1.0)
    sin_half = np.sqrt(1.0 - w**2)
    half = np.arccos(w)
    with np.errstate(divide='ignore', invalid='ignore'):
        slope = c * (c * sx**2 / sin_half**2 + half * cx / sin_half - half * w * c * sx**2 / sin_half**3)
    return np.where(sin_half > 1e-12, slope, 1.0)


def _solve_bent_step_twist(
    beta: np.ndarray,
    target: np.ndarray,
    guess: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Turning angles tau in (-pi, pi) for which a step bending by beta has the twist target
    (inverse of _bent_step_twist, by safeguarded Newton iterations starting from guess, by
    default the target).

    The twist increases monotonically with tau, from -pi cos(beta/2) to pi cos(beta/2). Targets
    outside that range keep tau = target; they are flagged in the returned mask. Also returns the
    slope of the twist at the solution (1 for flagged steps).
    """
    unreachable = np.abs(target) >= np.pi * np.cos(beta / 2) * (1.0 - 1e-9)
    goal = np.where(unreachable, 0.0, target)
    lo, hi = np.full_like(goal, -np.pi), np.full_like(goal, np.pi)
    tau = goal.copy() if guess is None else np.where(unreachable, 0.0, np.clip(guess, -np.pi, np.pi))
    for _ in range(TRACE_TWIST_MAX_ITER):
        res = _bent_step_twist(beta, tau) - goal
        lo = np.where(res < 0, tau, lo)
        hi = np.where(res > 0, tau, hi)
        slope = _bent_step_twist_slope(beta, tau)
        step = tau - res / slope
        new = np.where(~np.isfinite(step) | (step <= lo) | (step >= hi), 0.5 * (lo + hi), step)
        done = np.max(np.abs(new - tau), initial=0.0) <= TRACE_TWIST_TOL
        tau = new
        if done:
            break
    return np.where(unreachable, target, tau), unreachable, np.where(unreachable, 1.0, slope)


def _step_turns(
    target: np.ndarray,
    beta: np.ndarray,
    correct: bool,
    guess: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Turning angles about the third axis that give the steps the target twist. Without correction
    they equal the target; the twist of a bent step then deviates at second order in the bend.
    guess is an estimate of the turning angles (e.g. from a nearby target). Also returns the
    steps whose target cannot be reached and the slope of the twist with respect to the turning
    angle.
    """
    if not correct:
        return target, np.zeros(len(target), dtype=bool), np.ones(len(target))
    wrapped = _wrap(target)
    tau, unreachable, slope = _solve_bent_step_twist(
        beta, wrapped, None if guess is None else guess - (target - wrapped)
    )
    return target + (tau - wrapped), unreachable, slope


def _closed_turns(
    twist: np.ndarray,
    beta: np.ndarray,
    holonomy: float,
    excess_link: float,
    correct: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Turning angles of a closed chain and the uniform shift of the twist per step.

    The frames close if the transport angle around the loop (holonomy) plus the total turning
    angle is a multiple of 2 pi. The relaxed ring is closed by the smallest uniform shift of the
    twist per step; the excess link adds whole turns on top. The number of turns follows from the
    turning angles of the unclosed relaxed chain; the shift is then found by Newton iterations.
    """
    tau, unreachable, slope = _step_turns(twist, beta, correct)
    total = tau.sum() + _wrap(-holonomy - tau.sum()) + 2 * np.pi * excess_link
    delta = 0.0
    for _ in range(TRACE_TWIST_MAX_ITER):
        res = tau.sum() - total
        # tolerance above the rounding error of the sum
        if abs(res) <= TRACE_TWIST_TOL * (1.0 + np.abs(tau).sum()):
            break
        # d tau_j / d delta = 1 / slope_j
        change = -res / np.sum(1.0 / slope)
        delta += change
        tau, unreachable, slope = _step_turns(twist + delta, beta, correct, tau + change / slope)
    return tau, unreachable, delta


###################################################################################################
# Configuration

def tracepoints_to_poses(
    tracepoints: np.ndarray,
    groundstate: np.ndarray | float,
    num_poses: int | None = None,
    closed: bool = False,
    rescale: bool = False,
    excess_link: float = 0.0,
    max_fene: float | None = None,
    tangent_persistence: float | np.ndarray = 1.0,
    smoothing: float | None = None,
    first_triad: np.ndarray | None = None,
    twist_correction: bool = True,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, TracepointInfo]:
    """
    Poses along a smooth curve through tracepoints, with the intrinsic rise and twist of a ground
    state.

    The curve is a cubic spline through the tracepoints, parametrized by the distance between
    them: natural (zero curvature at the ends) for open chains, periodic for closed ones. Tangents
    may be prescribed at all or some tracepoints.

    The first pose sits at the first tracepoint. The poses are spaced such that the straight-line
    distances between consecutive poses are proportional to the rise of the steps. For open
    chains the last pose sits at the last tracepoint; for closed chains the last step returns to
    the first pose. Without rescale all steps are stretched or compressed by the same factor
    (TracepointInfo.stretch); with rescale the curve is scaled about the centroid of the
    tracepoints such that the distances equal the rise.

    The third axis of every triad is the tangent of the curve. The first triad is first_triad, or
    else the triad of ConfBuilder.straight for that tangent. Along the chain the triads are
    parallel transported (rotated minimally from tangent to tangent) and turned about the tangent
    by the intrinsic twist:

    - open chains: the excess link adds 2 pi excess_link / num_steps of twist to every step,
      turning the last pose by 2 pi excess_link about its tangent. It need not be an integer.
    - closed chains: the frames close only if the twist plus the transport angle around the loop
      (2 pi times the writhe, modulo 2 pi) is a whole number of turns. The relaxed ring is closed
      by the smallest uniform change of the twist per step (TracepointInfo.closure_twist), and
      the excess link adds whole turns on top. It counts relative to the relaxed ring for this
      path, which differs from the convention of ConfBuilder.circular (relative to the total
      intrinsic twist) by about the writhe for non-planar paths.

    The first pose is never changed by the excess link.

    Parameters
    ----------
    tracepoints : np.ndarray
        Positions with shape (N, 3), or positions and tangents with shape (N, 2, 3). Tangents may
        have any length; tangents marked NaN are free. A closed curve that repeats its first
        tracepoint at the end has the repetition removed.
    groundstate : float or np.ndarray
        Ground state per step: rise, (twist, rise) or (tilt, roll, twist, shift, slide, rise),
        twist in radians and rise in the length unit of the tracepoints. See parse_groundstate.
    num_poses : int, optional
        Number of poses. Required if the ground state is given for a single step.
    closed : bool, optional
        Closed chain. Requires at least 3 tracepoints, or 2 with tangents at both. Default False.
    rescale : bool, optional
        Scale the curve such that the distances between poses equal the rise. Default False.
    excess_link : float, optional
        Turns of twist added on top of the relaxed state. Has to be an integer for closed chains.
        Default 0.
    max_fene : float, optional
        Largest allowed stretch of a step beyond its rise (distance - rise). Compression is not
        limited. Default None (no limit).
    tangent_persistence : float or np.ndarray, optional
        How far the curve follows a prescribed tangent before it turns, scalar or one value per
        tracepoint. 1 gives the speed of the spline without tangents; larger values follow the
        tangent further. Default 1.
    smoothing : float, optional
        For noisy tracepoints: root-mean-square distance by which the curve may miss the
        tracepoints. The ends of open chains stay in place. Requires at least 4 tracepoints.
        Default None (the curve passes through the tracepoints).
    first_triad : np.ndarray, optional
        Triad (3x3, columns are the axes) of the first pose. Its third axis prescribes the tangent
        at the first tracepoint.
    twist_correction : bool, optional
        Match the twist of every step exactly. Without correction, the twist of a bent step
        deviates from its target at second order in the bend angle. Default True.
    return_info : bool, optional
        Also return a TracepointInfo with diagnostics. Default False.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, TracepointInfo]
        Poses with shape (num_poses, 4, 4), positions in the unit of the tracepoints, and the
        diagnostics if return_info is set. Warnings are also issued as UserWarning.

    Raises
    ------
    ValueError
        If the input is invalid, the curve turns too tightly for the step length, or a step is
        stretched beyond max_fene. ConfigurationValidationError (a ValueError) if first_triad is
        not a rotation matrix.
    """
    twist, rise = parse_groundstate(groundstate, num_poses, closed)
    num_steps = len(rise)
    points, tangents, persistence = _parse_tracepoints(tracepoints, tangent_persistence, closed)

    excess_link = float(excess_link)
    if not np.isfinite(excess_link):
        raise ValueError('excess_link has to be finite.')
    if closed:
        if abs(excess_link - round(excess_link)) > TRACE_INTEGER_TOL:
            raise ValueError(
                f'For closed chains excess_link has to be an integer (got {excess_link:g}): the '
                f'first pose has to return onto itself.'
            )
        excess_link = float(round(excess_link))
    if max_fene is not None and not (np.isfinite(max_fene) and max_fene >= 0):
        raise ValueError(f'max_fene has to be non-negative, got {max_fene}.')
    if smoothing is not None and not (np.isfinite(smoothing) and smoothing >= 0):
        raise ValueError(f'smoothing has to be non-negative, got {smoothing}.')
    if first_triad is not None:
        triad = np.asarray(first_triad, dtype=float)
        if triad.shape != (3, 3):
            raise ValueError(f'first_triad must have shape (3, 3), got {triad.shape}.')
        pose = np.eye(4)
        pose[:3, :3] = triad
        first_triad = validate_poses(pose[None], name='first_triad')[0][0, :3, :3]
        tangents[0] = _first_triad_tangent(first_triad, tangents[0])

    _check_curve_geometry(points, tangents, closed)
    centroid = points.mean(axis=0)

    # warnings are issued right away, such that they also show if a later step fails
    notes = []

    def note(msg: str) -> None:
        notes.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)

    smoothing_rms = 0.0
    if smoothing is not None:
        smoothed = _smooth_tracepoints(points, smoothing, closed)
        smoothing_rms = float(np.sqrt(np.mean(np.sum((smoothed - points)**2, axis=1))))
        points = smoothed
        _check_curve_geometry(points, tangents, closed)

    given = np.flatnonzero(~np.isnan(tangents[:, 0]))
    if len(given) > 0:
        num = len(points)
        nxt = (given + 1) % num if closed else np.minimum(given + 1, num - 1)
        prv = (given - 1) % num if closed else np.maximum(given - 1, 0)
        chain_dir = points[nxt] - points[prv]
        dots = np.einsum('ni,ni->n', tangents[given], chain_dir)
        against = given[(dots <= 0) & (np.linalg.norm(chain_dir, axis=1) > 0)]
        if len(against) > 0:
            note(
                f'The tangents at tracepoint(s) {_listed(against)} (0-based) point against the '
                f'direction from the previous to the next tracepoint; the curve forms a loop there.'
            )

    # positions
    curve = _build_curve(points, tangents, persistence, closed)
    u, positions, stretch = _place_poses(curve, rise, closed)
    scale = 1.0
    if rescale:
        scale = 1.0 / stretch
        positions = centroid + (positions - centroid) * scale
    elif abs(stretch - 1.0) > TRACE_WARN_STRETCH:
        note(
            f'The poses are spaced {stretch:.4g} times the intrinsic rise: the curve through the '
            f'tracepoints is {"longer" if stretch > 1 else "shorter"} than the chain. Set '
            f'rescale=True to scale the curve to the intrinsic rise.'
        )
    deviation = np.linalg.norm(bond_vectors(positions, closed), axis=1) - rise
    if max_fene is not None and np.any(deviation - max_fene > TRACE_FENE_TOL * rise):
        worst = int(np.argmax(deviation))
        raise ValueError(
            f'The poses are spaced {stretch:.4g} times the intrinsic rise, which stretches step '
            f'{worst} (0-based) by {deviation[worst]:.4g} beyond its rise, more than max_fene = '
            f'{max_fene:g}. Relax max_fene, or set rescale=True to scale the curve to the '
            f'intrinsic rise.'
        )

    # triads: parallel transport along the tangents, turned about them by the twist
    z = curve(u, 1)
    z /= np.linalg.norm(z, axis=1)[:, None]
    if first_triad is None:
        first_triad = so3.rotmat_align_vector(np.array([0.0, 0.0, 1.0]), np.ascontiguousarray(z[0]))
    x_ref, y_ref = _reference_axes(z)
    alpha, beta = _parallel_transport(z, x_ref, y_ref, closed)
    excess_per_step = 2 * np.pi * excess_link / num_steps
    if closed:
        tau, unreachable, delta = _closed_turns(twist, beta, float(alpha.sum()), excess_link, twist_correction)
        closure_twist = num_steps * delta - 2 * np.pi * excess_link
    else:
        delta = excess_per_step
        tau, unreachable, _ = _step_turns(twist + delta, beta, twist_correction)
        closure_twist = 0.0

    theta0 = np.arctan2(first_triad[:, 0] @ y_ref[0], first_triad[:, 0] @ x_ref[0])
    psi = theta0 + np.r_[0.0, np.cumsum(alpha[: len(z) - 1] + tau[: len(z) - 1])]
    cos, sin = np.cos(psi)[:, None], np.sin(psi)[:, None]
    poses = np.zeros((len(z), 4, 4))
    poses[:, :3, 0] = cos * x_ref + sin * y_ref
    poses[:, :3, 1] = cos * y_ref - sin * x_ref
    poses[:, :3, 2] = z
    poses[:, :3, 3] = positions
    poses[:, 3, 3] = 1.0
    poses[0, :3, :3] = first_triad

    # diagnostics
    if closed and abs(closure_twist) > TRACE_AMBIGUOUS_CLOSURE_FRACTION * np.pi:
        note(
            f'Closing the relaxed ring needs {np.degrees(closure_twist):+.1f} degrees of twist in '
            f'total; closing it in the other direction would need almost as much. The two '
            f'topoisomers are about equally close to the relaxed state.'
        )
    if unreachable.any():
        note(
            f'The twist of {unreachable.sum()} step(s) cannot be matched exactly: a step bent by '
            f'beta carries at most pi*cos(beta/2) of twist in the rotation-vector parametrization. '
            f'These steps keep the uncorrected twist (0-based steps: '
            f'{_listed(np.flatnonzero(unreachable))}).'
        )
    measured = _bent_step_twist(beta, _wrap(tau))
    twist_residual = float(np.max(np.abs(_wrap(measured - (twist + delta)))))
    max_bend = float(beta.max())
    if max_bend > TRACE_WARN_BEND:
        note(
            f'Consecutive poses bend by up to {np.degrees(max_bend):.1f} degrees (step '
            f'{int(np.argmax(beta))}, 0-based). The curve turns sharply relative to the step '
            f'length.'
        )

    closest_pair, closest_distance = None, None
    if len(positions) > 3 or not closed:
        k = min(4, len(positions))
        dist, idx = cKDTree(positions).query(positions, k=k)
        sep = np.abs(idx - np.arange(len(positions))[:, None])
        if closed:
            sep = np.minimum(sep, len(positions) - sep)
        dist = np.where(sep > 1, dist, np.inf)
        if np.isfinite(dist).any():
            i, j = np.unravel_index(np.argmin(dist), dist.shape)
            closest_pair = tuple(sorted((int(i), int(idx[i, j]))))
            closest_distance = float(dist[i, j])
            if closest_distance < TRACE_OVERLAP_FRACTION * np.median(rise + deviation):
                note(
                    f'Non-neighbouring poses {closest_pair[0]} and {closest_pair[1]} are '
                    f'{closest_distance:.3g} apart, less than half the median distance between '
                    f'consecutive poses. The chain may intersect itself.'
                )

    if not return_info:
        return poses
    info = TracepointInfo(
        stretch=stretch,
        scale=scale,
        max_rise_deviation=float(np.max(np.abs(deviation))),
        closure_twist=float(closure_twist),
        excess_twist_per_step=float(excess_per_step),
        max_bend=max_bend,
        max_twist_residual=twist_residual,
        closest_pair=closest_pair,
        closest_distance=closest_distance,
        smoothing_rms=smoothing_rms,
        warnings=notes,
    )
    return poses, info
