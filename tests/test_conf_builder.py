"""Tests for cgrbptools.core.conf_builder: configurations from tracepoints."""
import contextlib
import io
import sys
import warnings

import numpy as np
import pytest

from conftest import EXAMPLES, build_topology
from cgrbptools import CGRBPTopology, ConfBuilder
from cgrbptools.PolyCG.polycg import gen_params, load_sequence
from cgrbptools.core.conf_checks import FENE_RLOGARG_MIN
from cgrbptools.core.tracepoints import TracepointInfo, tracepoints_to_poses
from cgrbptools.evals.se3 import poses2parameters


def topology(seqfile='200bp', cg=10, closed=False, centered=False, unit_length=None, fene=None):
    """Topology as built by tests/conftest.build_topology, optionally rescaled and with FENE."""
    with contextlib.redirect_stdout(io.StringIO()):
        topol, _ = build_topology(seqfile, cg=cg, closed=closed, centered=centered)
        if fene is not None:
            topol.set_fene(*fene)
        if unit_length is not None:
            topol.set_unit_length(unit_length)
    return topol


def build(*args, **kwargs):
    """ConfBuilder.from_tracepoints with warnings suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ConfBuilder.from_tracepoints(*args, **kwargs)


def long_ring(cg, repeats):
    """Closed crystal topology for the 1kbp example sequence repeated several times."""
    seq = load_sequence(str(EXAMPLES / '1kbp')) * repeats
    with contextlib.redirect_stdout(io.StringIO()):
        params = gen_params('crystal', seq, composite_size=cg, closed=True, allow_crop=True)
        topol = CGRBPTopology(coupling_range=1, decimals=4, closed=True)
        topol.set_params(params.cg_shape_params, params.cg_stiffmat)
        topol.set_sequence(seq, chars_per_atom=cg)
    return topol


def chain_length(topol):
    """Contour length of the chain in nm."""
    return topol.groundstate[:, 5].sum() * topol.unit_length


def scaled(points, length, closed):
    """Points scaled such that the polygon through them has the given length."""
    ext = np.vstack([points, points[:1]]) if closed else points
    return points * length / np.linalg.norm(np.diff(ext, axis=0), axis=1).sum()


def arc(length):
    s = np.linspace(0, 1.5 * np.pi, 12)
    return scaled(np.c_[10 * np.cos(s), 7 * np.sin(s), 3 * np.sin(s)], length, closed=False)


def ring(length, height=0.0, n=24):
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return scaled(np.c_[1.3 * np.cos(s), 0.8 * np.sin(s), height * np.sin(2 * s)], length, closed=True)


def toroid(length, coils=8, n=160):
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 0.8 + 0.12 * np.cos(coils * s)
    return scaled(np.c_[r * np.cos(s), r * np.sin(s), 0.12 * np.sin(coils * s)], length, closed=True)


def twist_link(conf):
    """
    Approximate linking number of a gently bent planar ring: the twist of its steps in turns
    (misses up to about pi*bend^2/8 per strongly bent step).
    """
    return poses2parameters(conf.poses, closed=True)[:, 2].sum() / (2 * np.pi)


def ribbon_link(conf):
    """Linking number of the ribbon along the first triad axes (requires PyLk)."""
    from cgrbptools.evals.PyLk.pylk import triads2link
    pos = conf.positions
    width = 0.02 * np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1))
    return triads2link(pos, conf.triads, radius=width, closed=True)


###################################################################################################
# Construction and units

def test_poses_follow_tracepoints_to_poses():
    topol = topology()
    pts = arc(chain_length(topol))
    conf, info = build(topol, pts, return_info=True)
    expected = tracepoints_to_poses(pts, topol.groundstate, num_poses=topol.nbp)
    assert conf.topology is topol and conf.nbp == topol.nbp
    assert isinstance(info, TracepointInfo)
    np.testing.assert_allclose(conf.poses, expected, atol=1e-12)
    np.testing.assert_allclose(poses2parameters(conf.poses)[:, 2], topol.groundstate[:, 2], atol=1e-10)
    scaled_conf = build(topol, pts, rescale=True)
    np.testing.assert_allclose(np.linalg.norm(np.diff(scaled_conf.positions, axis=0), axis=1),
                               topol.groundstate[:, 5], rtol=1e-9)


def test_options_are_passed_on():
    topol = topology()
    pts = arc(chain_length(topol))
    tangents = np.full_like(pts, np.nan)
    tangents[5] = pts[6] - pts[4] + [0.0, 0.0, 1.0]
    # a first triad along the start of the path, turned about it
    z = (pts[1] - pts[0]) / np.linalg.norm(pts[1] - pts[0])
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    options = dict(first_triad=np.column_stack([x, np.cross(z, x), z]), tangent_persistence=2.0,
                   twist_correction=False)
    traced = np.stack([pts, tangents], axis=1)
    conf = build(topol, traced, mass=2.5, **options)
    expected = tracepoints_to_poses(traced, topol.groundstate, num_poses=topol.nbp, **options)
    np.testing.assert_allclose(conf.poses, expected, atol=1e-12)
    assert conf.mass == 2.5


@pytest.mark.parametrize('closed', [False, True])
def test_units(closed):
    reference = topology(closed=closed)
    topol = topology(closed=closed, unit_length=3.4)
    length = chain_length(topol)
    pts = ring(length, height=0.3) if closed else arc(length)
    expected = build(reference, pts, smoothing=0.2).poses_in_nm()
    conf, info = build(topol, pts, smoothing=0.2, return_info=True)
    np.testing.assert_allclose(conf.poses_in_nm(), expected, atol=1e-9)
    if not closed:
        # smoothing keeps the ends of open chains in place
        np.testing.assert_allclose(conf.positions[0], pts[0] / 3.4, atol=1e-12)
    # inputs and diagnostics are in conf_units
    for units, factor in (('angstrom', 10.0), ('sim', 1 / 3.4)):
        other, other_info = build(topol, pts * factor, conf_units=units, smoothing=0.2 * factor,
                                  return_info=True)
        np.testing.assert_allclose(other.poses_in_nm(), expected, atol=1e-9)
        assert other_info.stretch == pytest.approx(info.stretch, rel=1e-9)
        assert other_info.smoothing_rms == pytest.approx(info.smoothing_rms * factor, rel=1e-6)
        assert other_info.max_rise_deviation == pytest.approx(info.max_rise_deviation * factor, rel=1e-6)
    with pytest.raises(ValueError, match='Unknown length unit'):
        ConfBuilder.from_tracepoints(topol, pts, conf_units='um')


def test_length_unit_mistakes():
    topol = topology(unit_length=0.5)
    pts = arc(chain_length(topol))
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        ConfBuilder.from_tracepoints(topol, pts)
        ConfBuilder.from_tracepoints(topol, 10 * pts, conf_units='angstrom')
    # Angstrom read as nm: implausible spacing
    with pytest.raises(ValueError, match="conf_units='angstrom'"):
        build(topol, 10 * pts)
    build(topol, 10 * pts, rescale=True)
    # simulation units read as nm, and nm read as simulation units
    with pytest.warns(UserWarning, match="conf_units='sim'"):
        ConfBuilder.from_tracepoints(topol, pts / 0.5)
    with pytest.warns(UserWarning, match="conf_units='nm'"):
        ConfBuilder.from_tracepoints(topol, pts, conf_units='sim')


def test_topology_without_sequence():
    topol = topology()
    bare = CGRBPTopology(coupling_range=1, decimals=4)
    with contextlib.redirect_stdout(io.StringIO()):
        bare.set_params(topol.groundstate, topol.stiffness_matrix)
    pts = arc(chain_length(topol))
    np.testing.assert_allclose(build(bare, pts).poses, build(topol, pts).poses, atol=1e-12)


@pytest.mark.parametrize('seqfile, cg, closed, centered', [
    ('200bp', 1, False, False),
    ('1kbp', 5, True, True),
    ('200bp', 10, False, True),
    ('1kbp', 10, True, False),
])
def test_topologies(seqfile, cg, closed, centered):
    topol = topology(seqfile, cg=cg, closed=closed, centered=centered)
    length = chain_length(topol)
    conf = build(topol, ring(length, height=0.2) if closed else arc(length), rescale=True)
    assert conf.nbp == topol.nbp and conf.topology is topol
    np.testing.assert_allclose(np.linalg.det(conf.triads), 1.0)
    pos = conf.positions
    bonds = np.roll(pos, -1, axis=0) - pos if closed else np.diff(pos, axis=0)
    np.testing.assert_allclose(np.linalg.norm(bonds, axis=1), topol.groundstate[:, 5], rtol=1e-9)


###################################################################################################
# Excess link

def test_open_excess_link_as_straight():
    topol = topology()
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, chain_length(topol)]])
    np.testing.assert_allclose(build(topol, pts).poses, build(topol, pts, excess_link=0).poses, atol=1e-12)
    # the link reference does not matter for open chains
    for excess in (0, 1.5, -2):
        for reference in ('lk0', 'path'):
            conf = build(topol, pts, excess_link=excess, link_reference=reference)
            np.testing.assert_allclose(conf.poses, ConfBuilder.straight(topol, excess_link=excess).poses, atol=1e-10)


def test_planar_ring_as_circular():
    topol = topology('1kbp', closed=True)
    pts = ring(chain_length(topol))
    relaxed = build(topol, pts, rescale=True)
    assert round(twist_link(relaxed)) == round(topol.groundstate[:, 2].sum() / (2 * np.pi))
    for excess in (0, 2, -3, 0.4):
        conf = build(topol, pts, rescale=True, excess_link=excess)
        assert round(twist_link(conf)) == round(twist_link(ConfBuilder.circular(topol, excess_link=excess)))
        np.testing.assert_allclose(conf.positions, relaxed.positions, atol=1e-12)
        if float(excess).is_integer():
            path = build(topol, pts, rescale=True, excess_link=excess, link_reference='path')
            np.testing.assert_allclose(path.poses, conf.poses, atol=1e-12)


def test_link_references_on_a_writhed_ring():
    pytest.importorskip('cgrbptools.evals.PyLk.pylk')
    topol = topology('1kbp', closed=True)
    lk0 = topol.groundstate[:, 2].sum() / (2 * np.pi)
    pts = toroid(chain_length(topol))
    relaxed, info = build(topol, pts, rescale=True, return_info=True)
    relaxed_lk = ribbon_link(relaxed)
    # the relaxed ring takes up the writhe of the path (about -2.8 turns)
    assert relaxed_lk == pytest.approx(round(relaxed_lk), abs=1e-3)
    assert relaxed_lk <= round(lk0) - 2
    for excess in (0, 2, -1, 0.4):
        conf, info = build(topol, pts, rescale=True, excess_link=excess, return_info=True)
        assert ribbon_link(conf) == pytest.approx(round(lk0 + excess), abs=1e-3)
        np.testing.assert_allclose(conf.positions, relaxed.positions, atol=1e-12)
        turns = info.excess_twist_per_step * topol.nbp / (2 * np.pi)
        assert turns == pytest.approx(round(lk0 + excess) - round(relaxed_lk), abs=1e-9)
    path = build(topol, pts, rescale=True, excess_link=2, link_reference='path')
    assert ribbon_link(path) == pytest.approx(round(relaxed_lk) + 2, abs=1e-3)


def test_long_planar_ring_as_circular():
    # at 4 bp per bead the two linking-number counts differ by about three quarters of a turn for
    # this ring (3 kbp; see KNOWN_ISSUES.md); the linking number follows ConfBuilder.circular
    topol = long_ring(cg=4, repeats=3)
    pts = ring(chain_length(topol))
    for excess in (0, 2):
        conf = build(topol, pts, rescale=True, excess_link=excess)
        path = build(topol, pts, rescale=True, excess_link=excess, link_reference='path')
        np.testing.assert_allclose(conf.poses, path.poses, atol=1e-12)
        assert round(twist_link(conf)) == round(twist_link(ConfBuilder.circular(topol, excess_link=excess)))


def flower(length, petals=14, amplitude=0.1, n=400):
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1 + amplitude * np.cos(petals * s)
    return scaled(np.c_[r * np.cos(s), r * np.sin(s), np.zeros(n)], length, closed=True)


def test_wiggly_planar_ring():
    # the bent steps of this planar ring (up to 42 degrees) turn about the tangents by almost a
    # turn more in total than their twist; the sum of the twist missed the linking number silently
    pytest.importorskip('cgrbptools.evals.PyLk.pylk')
    topol = topology('1kbp', cg=4, closed=True)
    lk0 = topol.groundstate[:, 2].sum() / (2 * np.pi)
    pts = flower(chain_length(topol))
    for excess in (0, 1, -1):
        conf, info = build(topol, pts, rescale=True, excess_link=excess, return_info=True)
        assert ribbon_link(conf) == pytest.approx(round(lk0 + excess), abs=1e-3)
        assert not any('not close to an integer' in msg for msg in info.warnings)


def test_wiggly_planar_ring_without_pylk(monkeypatch):
    # planar rings do not need PyLk, however strongly they bend
    import cgrbptools.evals.PyLk as pylk_package
    topol = topology('1kbp', cg=4, closed=True)
    pts = flower(chain_length(topol))
    expected = build(topol, pts, rescale=True, excess_link=1).poses
    monkeypatch.setitem(sys.modules, 'cgrbptools.evals.PyLk.pylk', None)
    monkeypatch.delattr(pylk_package, 'pylk', raising=False)
    conf = build(topol, pts, rescale=True, excess_link=1)
    np.testing.assert_allclose(conf.poses, expected, atol=1e-12)


def test_lk0_warnings_describe_the_returned_ring():
    # at 5 bp per bead the twist of some steps of the twist-relaxed ring cannot be matched
    # exactly; with the turns added the returned ring has other such steps, or none
    topol = topology('1kbp', cg=5, closed=True)
    for petals, amplitude, excess in ((10, 0.15, -3), (14, 0.1, -1)):
        pts = flower(chain_length(topol), petals=petals, amplitude=amplitude)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            conf, info = ConfBuilder.from_tracepoints(topol, pts, rescale=True, excess_link=excess,
                                                      return_info=True)
        assert [str(w.message) for w in caught] == info.warnings
        turns = round(info.excess_twist_per_step * topol.nbp / (2 * np.pi))
        assert turns != 0
        direct, direct_info = build(topol, pts, rescale=True, excess_link=turns, link_reference='path',
                                    return_info=True)
        np.testing.assert_allclose(conf.poses, direct.poses, atol=1e-10)
        unmatched = [msg for msg in info.warnings if 'cannot be matched exactly' in msg]
        assert unmatched == [msg for msg in direct_info.warnings if 'cannot be matched exactly' in msg]


def test_path_reference_requires_whole_turns():
    topol = topology(closed=True)
    pts = ring(chain_length(topol))
    with pytest.raises(ValueError, match="link_reference='lk0'"):
        ConfBuilder.from_tracepoints(topol, pts, excess_link=0.5, link_reference='path')
    build(topol, pts, excess_link=0.5)


def test_warnings_are_issued_once():
    topol = topology('1kbp', closed=True)
    # stretched planar ring with whole turns added on top of the relaxed ring
    pts = ring(1.3 * chain_length(topol))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _, info = ConfBuilder.from_tracepoints(topol, pts, excess_link=3, return_info=True)
    messages = [str(w.message) for w in caught]
    assert info.excess_twist_per_step != 0
    assert any('rescale=True' in msg for msg in messages)
    assert len(messages) == len(set(messages))
    assert messages == info.warnings


def test_warnings_are_not_repeated_with_other_numbers():
    # the twist-relaxed ring closes ambiguously; the rebuild with the excess link would repeat
    # that warning with a slightly different closure twist
    topol = topology('1kbp', closed=True, centered=True)
    radius = chain_length(topol) / (2 * np.pi)
    s = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    pts = np.c_[1.1 * radius * np.cos(s), 0.9 * radius * np.sin(s), np.zeros(20)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _, info = ConfBuilder.from_tracepoints(topol, pts, excess_link=2, return_info=True)
    messages = [str(w.message) for w in caught]
    assert sum(msg.startswith('Closing the relaxed ring') for msg in messages) == 1
    assert messages == info.warnings
    assert info.excess_twist_per_step != 0.0


def test_twist_limits():
    topol = topology()
    pts = arc(chain_length(topol))
    with pytest.raises(ValueError, match='degrees of twist per step'):
        build(topol, pts, excess_link=topol.nbps / 3)
    ring_topol = topology('1kbp', closed=True)
    with pytest.raises(ValueError, match="degrees of twist per step.*link_reference='path'"):
        build(ring_topol, ring(chain_length(ring_topol)), rescale=True, excess_link=ring_topol.nbp / 3)
    # added twist pushes composite steps close to 180 degrees beyond 180 degrees, which is
    # reported as a warning
    closed = topology('1kbp', cg=5, closed=True)
    with pytest.warns(UserWarning, match='exceeds 180 degrees.*turn\\(s\\) less'):
        ConfBuilder.from_tracepoints(closed, ring(chain_length(closed)), rescale=True, excess_link=2)


def homogeneous(twist, nbps=100, closed=True):
    """Topology from CGRBPTopology.homogeneous_params with pure twist and a rise of 3.4 nm."""
    topol = CGRBPTopology(decimals=4, closed=closed)
    with contextlib.redirect_stdout(io.StringIO()):
        topol.homogeneous_params(groundstate=np.array([0.0, 0.0, twist, 0.0, 0.0, 3.4]),
                                 local_stiffness=np.array([12.0, 12.0, 29.0, 200.0, 200.0, 200.0]),
                                 nbps=nbps, closed=closed)
    return topol


@pytest.mark.parametrize('closed', [False, True])
def test_twist_given_with_full_turns(closed):
    # 10 bp per bead at 10.5 bp/turn: 342.9 degrees per step, the same rotation as -17.1 degrees
    if closed:
        pytest.importorskip('cgrbptools.evals.PyLk.pylk')
    turned = homogeneous(2 * np.pi * 10 / 10.5, closed=closed)
    reduced = homogeneous(2 * np.pi * 10 / 10.5 - 2 * np.pi, closed=closed)
    length = chain_length(turned)
    pts = toroid(length) if closed else arc(length)
    lk0 = reduced.groundstate[:, 2].sum() / (2 * np.pi)
    for kwargs in ({}, {'excess_link': 0}, {'excess_link': 2}):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            conf, info = ConfBuilder.from_tracepoints(turned, pts, rescale=True, return_info=True,
                                                      **kwargs)
        expected, expected_info = build(reduced, pts, rescale=True, return_info=True, **kwargs)
        np.testing.assert_allclose(conf.poses, expected.poses, atol=1e-10)
        assert info.warnings == expected_info.warnings
        assert not any('exceeds 180 degrees' in str(w.message) for w in caught)
        if closed and kwargs:
            assert ribbon_link(conf) == pytest.approx(round(lk0 + kwargs['excess_link']), abs=1e-3)


###################################################################################################
# Steps twisted by about 180 degrees

def homogeneous_ring(nbp, twist_deg, rise=1.7):
    """Closed topology with the same twist and rise at every step (hand-made groundstate)."""
    groundstate = np.zeros((nbp, 6))
    groundstate[:, 2], groundstate[:, 5] = np.radians(twist_deg), rise
    topol = CGRBPTopology(coupling_range=1, decimals=4, closed=True)
    with contextlib.redirect_stdout(io.StringIO()):
        topol.set_params(groundstate, np.eye(6 * nbp))
    return topol


def intended_link(conf, target, substeps=4):
    """
    Linking number of the ribbon along the first triad axes with every step counted with its
    intended twist (requires PyLk). Every step is subdivided: the frame bends along the minimal
    rotation between the tangents and turns about the tangent by the angle closest to target.
    """
    from scipy.spatial.transform import Rotation
    from cgrbptools.evals.PyLk.pylk import triads2link
    pos, triads = conf.positions, conf.triads
    nxt = np.roll(np.arange(len(pos)), -1)
    z0, z1 = triads[:, :, 2], triads[nxt, :, 2]
    axis = np.cross(z0, z1)
    sin = np.linalg.norm(axis, axis=1)
    bend = axis / sin[:, None] * np.arctan2(sin, np.einsum('ij,ij->i', z0, z1))[:, None]
    x0, x1 = Rotation.from_rotvec(bend).apply(triads[:, :, 0]), triads[nxt, :, 0]
    turn = np.arctan2(np.einsum('ij,ij->i', x1, np.cross(z1, x0)), np.einsum('ij,ij->i', x1, x0))
    turn += 2 * np.pi * np.round((target - turn) / (2 * np.pi))
    idx = np.repeat(np.arange(len(pos)), substeps)
    frac = np.tile(np.arange(substeps) / substeps, len(pos))
    frames = (Rotation.from_rotvec(frac[:, None] * bend[idx]) * Rotation.from_matrix(triads[idx])
              * Rotation.from_rotvec(np.outer(frac * turn[idx], [0.0, 0.0, 1.0]))).as_matrix()
    points = pos[idx] + frac[:, None] * (pos[nxt] - pos)[idx]
    width = 0.02 * np.median(np.linalg.norm(pos[nxt] - pos, axis=1))
    return triads2link(points, frames, radius=width, closed=True)


def final_target(topol, info):
    """Intended twist of every step of a closed configuration built by from_tracepoints."""
    added = (info.closure_twist + info.excess_twist_per_step * topol.nbp) / topol.nbp
    return topol.groundstate[:, 2] + added


def test_relaxed_ring_with_steps_beyond_180_degrees():
    pytest.importorskip('cgrbptools.evals.PyLk.pylk')
    topol = homogeneous_ring(200, 179.6)
    pts = toroid(chain_length(topol), coils=3)
    lk0 = topol.groundstate[:, 2].sum() / (2 * np.pi)
    # closing the relaxed ring adds about 0.9 degrees per step: every step twists beyond 180
    # degrees, which the ribbon between consecutive beads counts as turned the other way round
    _, relaxed = build(topol, pts, rescale=True, return_info=True)
    assert np.all(topol.groundstate[:, 2] + relaxed.closure_twist / topol.nbp > np.pi)
    # the same rotations expressed one turn lower per step
    shifted = homogeneous_ring(200, 179.6 - 360)
    for excess in (0, -1, 2):
        conf, info = build(topol, pts, rescale=True, excess_link=excess, return_info=True)
        assert intended_link(conf, final_target(topol, info)) == pytest.approx(round(lk0 + excess), abs=1e-3)
        np.testing.assert_allclose(build(shifted, pts, rescale=True, excess_link=excess).poses,
                                   conf.poses, atol=1e-9)


def test_strongly_bent_steps_near_180_degrees():
    pytest.importorskip('cgrbptools.evals.PyLk.pylk')
    # bends of up to 20 degrees per step: between consecutive beads the ribbon turns the other way
    # round also for steps twisted by slightly less than 180 degrees
    topol = homogeneous_ring(60, 179.5)
    pts = toroid(chain_length(topol), coils=5)
    lk0 = topol.groundstate[:, 2].sum() / (2 * np.pi)
    for excess in (0, -1):
        conf, info = build(topol, pts, rescale=True, excess_link=excess, return_info=True)
        assert intended_link(conf, final_target(topol, info)) == pytest.approx(round(lk0 + excess), abs=1e-3)


def test_planar_ring_with_steps_near_180_degrees():
    # the planar count takes every step on the branch of its intended twist, also near 180 degrees
    twist = np.full(100, 171.4)
    twist[::20] = 179.5
    topol = homogeneous_ring(100, twist)
    pts = ring(chain_length(topol))
    for excess in (0, 1):
        conf = build(topol, pts, rescale=True, excess_link=excess)
        path = build(topol, pts, rescale=True, excess_link=excess, link_reference='path')
        np.testing.assert_allclose(conf.poses, path.poses, atol=1e-12)


def test_steps_twisted_beyond_minus_180_degrees():
    # shortest-rotation counts are one turn higher for every step twisted beyond -180 degrees
    topol = homogeneous_ring(100, -179.8)
    with pytest.warns(UserWarning, match='exceeds 180 degrees.*turn\\(s\\) more'):
        ConfBuilder.from_tracepoints(topol, ring(chain_length(topol)), rescale=True, excess_link=-1)


###################################################################################################
# FENE

UNIT_LENGTH = 3.4
FENE_SIM = (200, 1.1, 1.35)   # in units of 3.4 nm (the bead spacing) and kT
FENE = (FENE_SIM[0] / UNIT_LENGTH**2, FENE_SIM[1] * UNIT_LENGTH, FENE_SIM[2] * UNIT_LENGTH)   # kT/nm^2, nm


def straight_path(topol, stretch):
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, stretch * chain_length(topol)]])


def test_fene_limit():
    topol = topology(unit_length=UNIT_LENGTH, fene=FENE)
    np.testing.assert_allclose(topol.get_fene(), FENE_SIM, rtol=1e-12)
    _, rc, r0 = topol.get_fene()
    bound = rc + (r0 - rc) * np.sqrt(1 - FENE_RLOGARG_MIN)
    stretch = bound / topol.groundstate[:, 5].max()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        conf, info = ConfBuilder.from_tracepoints(topol, straight_path(topol, stretch * (1 - 1e-6)),
                                                  return_info=True)
    assert np.linalg.norm(np.diff(conf.positions, axis=0), axis=1).max() < bound
    assert any('active from the start' in msg for msg in info.warnings)
    assert not any('linearly' in str(w.message) or 'aborts' in str(w.message) for w in caught)
    with pytest.raises(ValueError, match='regularly.*rescale=True'):
        build(topol, straight_path(topol, stretch * (1 + 1e-6)))
    build(topol, straight_path(topol, 1.5), rescale=True)
    build(topol, straight_path(topol, 1.5), max_fene=2.0)
    with pytest.warns(UserWarning, match='linearly'):
        ConfBuilder.from_tracepoints(topol, straight_path(topol, 1.5), max_fene=None)
    with pytest.raises(ValueError, match='max_fene = 0.5'):
        build(topol, straight_path(topol, 1.5), max_fene=0.5)


def test_fene_coefficients_in_wrong_units():
    # simulation-unit values passed to set_fene, which takes nm
    topol = topology(unit_length=UNIT_LENGTH, fene=FENE_SIM)
    with pytest.raises(ValueError, match='set_fene takes Rc and R0 in nm'):
        build(topol, straight_path(topol, 1.0))
    build(topol, straight_path(topol, 1.0), max_fene=None)


def test_fene_of_a_topology_read_from_a_database(tmp_path):
    topol = topology(unit_length=UNIT_LENGTH, fene=FENE)
    with contextlib.redirect_stdout(io.StringIO()):
        topol.write_database(tmp_path / 'topol', add_extension=False)
        loaded = CGRBPTopology.read_database(tmp_path / 'topol')
    # precondition, see KNOWN_ISSUES.md (FENE coefficients lost by read_database)
    assert loaded.has_fene and getattr(loaded, 'fene_Rc', None) is None
    with pytest.raises(ValueError, match='regularly'):
        build(loaded, straight_path(loaded, 1.4))
    build(loaded, straight_path(loaded, 1.4), rescale=True)
    # rebuilding the couplings of a loaded topology drops the FENE coefficients
    with contextlib.redirect_stdout(io.StringIO()):
        loaded.set_unit_length(2.0)
    with pytest.raises(ValueError, match='not available'):
        build(loaded, straight_path(loaded, 1.0))
    build(loaded, straight_path(loaded, 1.0), max_fene=None)


def test_fene_limit_of_a_closed_chain():
    topol = topology(closed=True, unit_length=UNIT_LENGTH, fene=FENE)
    pts = ring(1.5 * chain_length(topol))
    with pytest.raises(ValueError, match='regularly'):
        build(topol, pts)
    conf = build(topol, pts, rescale=True)
    np.testing.assert_allclose(np.linalg.norm(np.roll(conf.positions, -1, axis=0) - conf.positions, axis=1),
                               topol.groundstate[:, 5], rtol=1e-9)


def test_no_fene_no_limit():
    topol = topology()
    with pytest.warns(UserWarning, match='rescale=True'):
        ConfBuilder.from_tracepoints(topol, straight_path(topol, 1.5))


###################################################################################################
# Validation and optional dependencies

def test_invalid_arguments():
    topol = topology()
    pts = arc(chain_length(topol))
    with pytest.raises(TypeError):
        ConfBuilder.from_tracepoints(None, pts)
    with pytest.raises(TypeError):
        ConfBuilder.from_tracepoints(topol, pts, 1.0)
    with pytest.raises(ValueError, match='groundstate'):
        ConfBuilder.from_tracepoints(CGRBPTopology(), pts)
    for kwargs in ({'mass': 0}, {'link_reference': 'circle'}, {'max_fene': 'yes'},
                   {'excess_link': np.inf}, {'excess_link': '1'}, {'excess_link': True},
                   {'max_fene': -1.0}):
        with pytest.raises(ValueError):
            build(topol, pts, **kwargs)
    with pytest.raises(ValueError, match='shape'):
        build(topol, pts[:, :2])


def test_without_pylk(monkeypatch):
    import cgrbptools.evals.PyLk as pylk_package
    monkeypatch.setitem(sys.modules, 'cgrbptools.evals.PyLk.pylk', None)
    monkeypatch.delattr(pylk_package, 'pylk', raising=False)
    topol = topology('1kbp', closed=True)
    length = chain_length(topol)
    with pytest.raises(ImportError, match="link_reference='path'"):
        build(topol, toroid(length), rescale=True, excess_link=0)
    build(topol, toroid(length), rescale=True)
    build(topol, toroid(length), rescale=True, excess_link=1, link_reference='path')
    build(topol, ring(length), rescale=True, excess_link=1)
