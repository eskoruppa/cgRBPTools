"""Tests for cgrbptools.core.tracepoints: poses along a curve through tracepoints."""
import warnings

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from cgrbptools.SO3 import so3
from cgrbptools.core import tracepoints as tp
from cgrbptools.core.tracepoints import TracepointInfo, parse_groundstate, tracepoints_to_poses
from cgrbptools.evals.se3 import poses2parameters

TWIST, RISE = 0.6, 0.34


def ellipse(n, closed=True, a=12.0, b=8.0, c=3.0, end=1.5 * np.pi):
    s = np.linspace(0, 2 * np.pi, n, endpoint=False) if closed else np.linspace(0, end, n)
    return np.c_[a * np.cos(s), b * np.sin(s), c * np.sin(2 * s)]


def circle(n, radius=10.0):
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[radius * np.cos(s), radius * np.sin(s), np.zeros(n)]


def build(*args, **kwargs):
    """tracepoints_to_poses with warnings suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return tracepoints_to_poses(*args, **kwargs)


def distances(poses, closed=False):
    pos = poses[:, :3, 3]
    return np.linalg.norm(np.roll(pos, -1, axis=0) - pos if closed else np.diff(pos, axis=0), axis=1)


def twists(poses, closed=False):
    return poses2parameters(poses, closed=closed)[:, 2]


def wrapped(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def rotz(angle):
    return so3.euler2rotmat(np.array([0.0, 0.0, angle]))


def random_rotation(seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


###################################################################################################
# Ground state

def test_groundstate_formats():
    tw, rise = parse_groundstate(0.34, num_poses=5)
    np.testing.assert_array_equal(tw, np.zeros(4))
    np.testing.assert_array_equal(rise, np.full(4, 0.34))

    tw, rise = parse_groundstate([0.6, 0.34], num_poses=5, closed=True)
    np.testing.assert_array_equal(tw, np.full(5, 0.6))
    np.testing.assert_array_equal(rise, np.full(5, 0.34))

    gs6 = np.array([0.1, 0.2, 0.6, 0.3, 0.4, 0.34])
    tw, rise = parse_groundstate(gs6, num_poses=3)
    np.testing.assert_array_equal(tw, [0.6, 0.6])
    np.testing.assert_array_equal(rise, [0.34, 0.34])

    tw, rise = parse_groundstate(gs6[None], num_poses=4)
    assert len(tw) == 3

    per_step = np.c_[np.linspace(0.5, 0.7, 7), np.linspace(0.3, 0.4, 7)]
    tw, rise = parse_groundstate(per_step)
    np.testing.assert_array_equal(tw, per_step[:, 0])
    np.testing.assert_array_equal(rise, per_step[:, 1])
    assert len(parse_groundstate(per_step, num_poses=8)[0]) == 7
    assert len(parse_groundstate(per_step, num_poses=7, closed=True)[0]) == 7

    tw, rise = parse_groundstate(np.full((4, 1), 0.34))
    np.testing.assert_array_equal(tw, np.zeros(4))


@pytest.mark.parametrize('gs, kwargs', [
    ([0.6, 0.34], {}),                                          # single step without num_poses
    (np.ones((5, 2)), {'num_poses': 5}),                        # 5 steps make 6 open poses
    (np.ones(3), {'num_poses': 4}),                             # 1D with 3 entries
    (np.ones((4, 3)), {}),                                      # 3 columns
    (np.c_[np.zeros(3), [0.3, 0.0, 0.3]], {}),                  # zero rise
    (np.ones((2, 2)), {'closed': True}),                        # closed chain of 2 poses
    (np.ones((1, 2)), {'closed': True}),                        # one step, closed, no num_poses
])
def test_groundstate_errors(gs, kwargs):
    with pytest.raises(ValueError):
        parse_groundstate(gs, **kwargs)


def test_num_poses_has_to_be_an_integer():
    with pytest.raises(TypeError):
        parse_groundstate(0.34, num_poses=5.0)


###################################################################################################
# Placement

def test_open_spacing_proportional_to_rise():
    rise = 0.34 * (1 + 0.3 * np.sin(np.arange(199)))
    gs = np.c_[np.full(199, TWIST), rise]
    pts = ellipse(15, closed=False)
    poses, info = build(pts, gs, return_info=True)
    ratio = distances(poses) / rise
    assert np.ptp(ratio) < 1e-9 * ratio.mean()
    assert ratio.mean() == pytest.approx(info.stretch)
    np.testing.assert_allclose(poses[0, :3, 3], pts[0], atol=1e-12)
    np.testing.assert_allclose(poses[-1, :3, 3], pts[-1], atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(poses[:, :3, :3]), 1.0)


@pytest.mark.parametrize('closed', [False, True])
def test_rescale_matches_rise_and_scales_about_centroid(closed):
    pts = ellipse(15, closed=closed)
    gs = np.c_[np.full(150, TWIST), np.full(150, RISE)]
    free, info = build(pts, gs, closed=closed, return_info=True)
    scaled, info_scaled = build(pts, gs, closed=closed, rescale=True, return_info=True)
    np.testing.assert_allclose(distances(scaled, closed), RISE, rtol=1e-9)
    assert info_scaled.scale == pytest.approx(1 / info.stretch)
    assert info_scaled.max_rise_deviation < 1e-9
    centroid = pts.mean(axis=0)
    np.testing.assert_allclose(
        scaled[:, :3, 3], centroid + (free[:, :3, 3] - centroid) / info.stretch, atol=1e-9
    )
    np.testing.assert_allclose(scaled[:, :3, :3], free[:, :3, :3], atol=1e-12)


def test_two_tracepoints_give_a_straight_chain():
    direction = np.array([1.0, 2.0, 2.0]) / 3
    poses = build(np.array([[0, 0, 0], 3 * direction * 10]), [TWIST, RISE], num_poses=11)
    np.testing.assert_allclose(distances(poses), 3.0, rtol=1e-12)
    np.testing.assert_allclose(poses[:, :3, 2], np.tile(direction, (11, 1)), atol=1e-12)
    np.testing.assert_allclose(twists(poses), TWIST, atol=1e-12)
    # default first triad: the alignment of ConfBuilder.straight
    np.testing.assert_allclose(poses[0, :3, :3], so3.rotmat_align_vector(np.array([0.0, 0, 1]), direction), atol=1e-12)


def test_tight_turn_raises():
    hairpin = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0], [10.5, 0.5, 0], [10, 1, 0], [5, 1, 0], [0, 1, 0.0]])
    with pytest.raises(ValueError, match='too tightly'):
        tracepoints_to_poses(hairpin, 1.0, num_poses=4)


###################################################################################################
# Twist

def test_twist_correction_matches_twist_exactly():
    pts = ellipse(10, closed=False, a=3.0, b=2.0, c=1.0)
    gs = np.c_[TWIST + 0.1 * np.sin(np.arange(59)), np.full(59, RISE)]
    corrected = build(pts, gs, rescale=True)
    plain, info = build(pts, gs, rescale=True, twist_correction=False, return_info=True)
    np.testing.assert_allclose(twists(corrected), gs[:, 0], atol=1e-11)
    deviation = np.abs(twists(plain) - gs[:, 0]).max()
    assert 1e-7 < deviation < 1e-2
    assert info.max_twist_residual == pytest.approx(deviation, rel=1e-6)
    # the correction only turns the triads about their tangents
    np.testing.assert_allclose(corrected[:, :3, 2:], plain[:, :3, 2:], atol=1e-12)


@pytest.mark.parametrize('correct', [False, True])
def test_open_excess_link(correct):
    pts = ellipse(12, closed=False)
    gs = [TWIST, RISE]
    base = build(pts, gs, num_poses=120, twist_correction=correct)
    twisted, info = build(pts, gs, num_poses=120, excess_link=0.37, twist_correction=correct, return_info=True)
    per_step = 2 * np.pi * 0.37 / 119
    assert info.excess_twist_per_step == pytest.approx(per_step)
    np.testing.assert_allclose(twisted[:, :3, 3], base[:, :3, 3], atol=1e-12)
    np.testing.assert_allclose(twisted[0], base[0], atol=1e-12)
    if correct:
        np.testing.assert_allclose(twists(twisted) - twists(base), per_step, atol=1e-11)
    else:
        # the last pose is turned by exactly 2 pi excess_link about its tangent
        np.testing.assert_allclose(twisted[-1, :3, :3], base[-1, :3, :3] @ rotz(2 * np.pi * 0.37), atol=1e-11)


@pytest.mark.parametrize('closed', [False, True])
def test_wrapped_twist_gives_the_same_poses(closed):
    pts = ellipse(12, closed=closed)
    gs = np.c_[np.full(100, 4.2), np.full(100, 3.4)]
    unwrapped = build(pts, gs, closed=closed)
    gs[:, 0] -= 2 * np.pi
    np.testing.assert_allclose(build(pts, gs, closed=closed), unwrapped, atol=1e-9)


def test_unreachable_twist_keeps_uncorrected_twist():
    # steps bent by 30 degrees carry at most pi*cos(15 deg) = 174 degrees of twist
    pts = circle(12)[:7]
    with pytest.warns(UserWarning, match='cannot be matched exactly'):
        poses = tracepoints_to_poses(pts, [3.1, 1.0], num_poses=7, rescale=True)
    assert np.all(np.abs(twists(poses)) < 3.1)


###################################################################################################
# Closed chains

@pytest.mark.parametrize('correct', [False, True])
def test_planar_ring_closes_with_smallest_twist_change(correct):
    poses, info = build(circle(20), [TWIST, RISE], num_poses=150, closed=True, rescale=True,
                        twist_correction=correct, return_info=True)
    measured = twists(poses, closed=True)
    # without correction the twist of the bent steps deviates at second order in the bend
    np.testing.assert_allclose(measured, TWIST + info.closure_twist / 150, atol=1e-11 if correct else 1e-3)
    # a planar ring has no writhe: the closure only rounds the intrinsic twist to whole turns (up
    # to the second-order difference between turning angle and twist of the bent steps)
    expected = wrapped(-150 * TWIST)
    if correct:
        assert info.closure_twist == pytest.approx(expected, abs=0.05)
    else:
        assert info.closure_twist == pytest.approx(expected, abs=1e-12)
    assert abs(info.closure_twist) < np.pi


def test_closed_linking_number():
    pytest.importorskip('cgrbptools.evals.PyLk.pylk')
    from cgrbptools.evals.link import poses2writhe

    pts = ellipse(12)
    lk0 = 200 * TWIST / (2 * np.pi)
    links = []
    for excess in (0, 2, -1):
        poses, info = build(pts, [TWIST, RISE], num_poses=200, closed=True, rescale=True,
                            excess_link=excess, return_info=True)
        excess_twist = np.sum(twists(poses, closed=True) - TWIST) / (2 * np.pi)
        links.append(lk0 + excess_twist + poses2writhe(poses, closed=True))
        assert abs(info.closure_twist) < np.pi
    assert abs(links[0] - round(links[0])) < 0.01
    np.testing.assert_allclose(np.diff(links), [2, -3], atol=1e-3)


def test_closed_excess_link_has_to_be_an_integer():
    with pytest.raises(ValueError, match='integer'):
        tracepoints_to_poses(circle(10), [TWIST, RISE], num_poses=100, closed=True, excess_link=0.5)


def test_closed_curve_needs_three_tracepoints_or_two_tangents():
    with pytest.raises(ValueError, match='2 tracepoints'):
        tracepoints_to_poses(circle(4)[:2], [TWIST, RISE], num_poses=50, closed=True)
    two = np.array([[[0, 0, 0], [0, 1, 0]], [[4, 0, 0], [0, -1, 0]]], dtype=float)
    poses = build(two, [TWIST, RISE], num_poses=40, closed=True, rescale=True)
    np.testing.assert_allclose(distances(poses, closed=True), RISE, rtol=1e-9)


def test_collinear_closed_curve_raises():
    line = np.c_[np.arange(5.0), np.zeros(5), np.zeros(5)]
    with pytest.raises(ValueError, match='straight line'):
        tracepoints_to_poses(line, [TWIST, RISE], num_poses=50, closed=True)


def test_repeated_closing_tracepoint_is_removed():
    pts = ellipse(12)
    expected = build(pts, [TWIST, RISE], num_poses=100, closed=True)
    np.testing.assert_allclose(
        build(np.vstack([pts, pts[:1]]), [TWIST, RISE], num_poses=100, closed=True), expected, atol=1e-12
    )


###################################################################################################
# Curve and tangents

@pytest.mark.parametrize('closed', [False, True])
def test_spline_without_tangents_is_natural_or_periodic(closed):
    pts = ellipse(9, closed=closed) + np.random.default_rng(3).normal(size=(9, 3))
    ext = np.vstack([pts, pts[:1]]) if closed else pts
    u = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ext, axis=0), axis=1))]
    slopes = tp._c2_slopes(u, pts, closed, np.full_like(pts, np.nan))
    reference = CubicSpline(u, ext, axis=0, bc_type='periodic' if closed else 'natural')
    np.testing.assert_allclose(slopes, reference(u[:len(pts)], 1), atol=1e-10)


def test_tangents_and_persistence():
    pts = ellipse(8, closed=False)
    tangents = np.full_like(pts, np.nan)
    tangents[[0, 3, 7]] = [[0, 5, 0], [-1, 0, 0.2], [1, 1, 0]]
    persistence = np.full(8, 1.0)
    curve = tp._build_curve(pts, tangents, persistence, closed=False)
    knots = curve.x
    slopes = curve(knots, 1)
    for i in (0, 3, 7):
        np.testing.assert_allclose(slopes[i] / np.linalg.norm(slopes[i]), tangents[i] / np.linalg.norm(tangents[i]), atol=1e-12)
    # C2 at the free knots
    for i in (1, 2, 4, 5, 6):
        np.testing.assert_allclose(curve(knots[i] - 1e-9, 2), curve(knots[i] + 1e-9, 2), atol=1e-5)
    # natural end condition is replaced by the tangent; persistence scales the speed
    persistence[3] = 2.5
    stretched = tp._build_curve(pts, tangents, persistence, closed=False)
    assert np.linalg.norm(stretched(knots[3], 1)) == pytest.approx(2.5 * np.linalg.norm(slopes[3]))

    poses = build(np.stack([pts, tangents], axis=1), [TWIST, RISE], num_poses=200)
    np.testing.assert_allclose(poses[0, :3, 2], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(poses[-1, :3, 2], np.array([1, 1, 0]) / np.sqrt(2), atol=1e-9)


def test_tangent_input_errors_and_warnings():
    pts = ellipse(6, closed=False)
    tangents = np.full_like(pts, np.nan)
    tangents[2, 0] = 1.0
    with pytest.raises(ValueError, match='NaN in all three'):
        tracepoints_to_poses(np.stack([pts, tangents], axis=1), [TWIST, RISE], num_poses=50)
    tangents[2] = 0.0
    with pytest.raises(ValueError, match='zero length'):
        tracepoints_to_poses(np.stack([pts, tangents], axis=1), [TWIST, RISE], num_poses=50)
    tangents[2] = -(pts[3] - pts[1])
    with pytest.warns(UserWarning, match='point against'):
        tracepoints_to_poses(np.stack([pts, tangents], axis=1), [TWIST, 0.01], num_poses=3000)
    # the warning is issued before the loop turns out to be too tight for the step length
    with pytest.warns(UserWarning, match='point against'), pytest.raises(ValueError, match='too tightly'):
        tracepoints_to_poses(np.stack([pts, tangents], axis=1), [TWIST, RISE], num_poses=400, rescale=True)


###################################################################################################
# First triad

def test_first_triad():
    pts = ellipse(10, closed=False)
    triad = random_rotation()
    poses = build(pts, [TWIST, RISE], num_poses=100, first_triad=triad)
    np.testing.assert_array_equal(poses[0, :3, :3], triad)
    np.testing.assert_allclose(twists(poses), TWIST, atol=1e-11)

    tangents = np.full_like(pts, np.nan)
    tilted = triad @ so3.euler2rotmat(np.array([np.radians(0.01), 0.0, 0.0]))
    tangents[0] = 3 * tilted[:, 2]
    agreeing = build(np.stack([pts, tangents], axis=1), [TWIST, RISE], num_poses=100, first_triad=triad)
    np.testing.assert_allclose(agreeing, poses, atol=1e-12)

    tilted = triad @ so3.euler2rotmat(np.array([np.radians(1.0), 0.0, 0.0]))
    tangents[0] = tilted[:, 2]
    with pytest.raises(ValueError, match='first_triad'):
        tracepoints_to_poses(np.stack([pts, tangents], axis=1), [TWIST, RISE], num_poses=100, first_triad=triad)
    with pytest.raises(ValueError):
        tracepoints_to_poses(pts, [TWIST, RISE], num_poses=100, first_triad=-triad)


###################################################################################################
# FENE limit, smoothing, diagnostics

def test_max_fene_limits_stretching_only():
    pts = ellipse(12, closed=False)
    _, info = build(pts, [TWIST, RISE], num_poses=200, return_info=True)
    stretch = 1.2
    num = int(round(info.stretch * 199 / stretch)) + 1
    _, info = build(pts, [TWIST, RISE], num_poses=num, return_info=True)
    excess = (info.stretch - 1) * RISE
    with pytest.raises(ValueError, match='max_fene.*rescale=True'):
        build(pts, [TWIST, RISE], num_poses=num, max_fene=0.9 * excess)
    build(pts, [TWIST, RISE], num_poses=num, max_fene=1.1 * excess)
    build(pts, [TWIST, RISE], num_poses=num, max_fene=0.0, rescale=True)
    # compression is not limited
    build(pts, [TWIST, RISE], num_poses=2 * num, max_fene=0.0)


def test_smoothing():
    rng = np.random.default_rng(5)
    clean = ellipse(60, closed=False, end=1.8 * np.pi)
    noisy = clean + 0.3 * rng.normal(size=clean.shape)
    rough, info_rough = build(noisy, [TWIST, RISE], num_poses=300, return_info=True)
    smooth, info = build(noisy, [TWIST, RISE], num_poses=300, smoothing=0.3, return_info=True)
    assert 0.15 < info.smoothing_rms < 0.45
    assert info.max_bend < 0.5 * info_rough.max_bend
    # the ends of open chains stay in place
    np.testing.assert_allclose(smooth[0, :3, 3], noisy[0], atol=1e-12)
    np.testing.assert_allclose(smooth[-1, :3, 3], noisy[-1], atol=1e-9)
    # closed chains
    _, info = build(ellipse(40) + 0.3 * rng.normal(size=(40, 3)), [TWIST, RISE], num_poses=300,
                    closed=True, smoothing=0.3, return_info=True)
    assert info.smoothing_rms > 0.1
    with pytest.raises(ValueError, match='at least 4 tracepoints'):
        tracepoints_to_poses(noisy[:3], [TWIST, RISE], num_poses=30, smoothing=0.3)


def test_info_and_warnings():
    pts = ellipse(12, closed=False)
    with pytest.warns(UserWarning, match='rescale=True'):
        poses, info = tracepoints_to_poses(pts, [TWIST, RISE], num_poses=50, return_info=True)
    assert isinstance(info, TracepointInfo)
    assert info.stretch > 1.1 and info.scale == 1.0
    assert info.closure_twist == 0.0
    assert any('rescale=True' in w for w in info.warnings)
    assert info.closest_pair is not None

    # an open chain that returns to its start overlaps with itself
    loop = ellipse(13, closed=False, end=2 * np.pi)
    with pytest.warns(UserWarning, match='intersect'):
        _, info = tracepoints_to_poses(loop, [TWIST, RISE], num_poses=200, rescale=True, return_info=True)
    assert info.closest_pair == (0, 199)
    assert info.closest_distance == pytest.approx(0.0, abs=1e-9)


def test_regular_input_issues_no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        tracepoints_to_poses(ellipse(12), [TWIST, RISE], num_poses=500, closed=True, rescale=True)
