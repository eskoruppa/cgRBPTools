"""Tests for cgrbptools.core.backmap: backmapping of closed chains."""
import numpy as np
import pytest

from cgrbptools import ConfBuilder
from cgrbptools.core.backmap import dna_backmap, spline_interpolate_constant_arclength_segments


def irregular_loop(poses, noise=0.5, seed=3):
    """Circular poses with randomly displaced bead positions (nm)."""
    poses = poses.copy()
    poses[:, :3, 3] += noise * np.random.default_rng(seed).normal(size=(len(poses), 3))
    return poses


@pytest.mark.parametrize('shift', [1, 7, 13])
def test_closed_backmap_does_not_depend_on_first_bead(closed_cg10, shift):
    # a non-periodic spline through the closed loop kinks the curve at bead 0, so relabeling the
    # beads moves the kink; the periodic spline gives the same base pairs for any first bead
    topol, _, poses = closed_cg10
    poses = irregular_loop(poses)
    cg = topol.chars_per_atom
    ref = dna_backmap(ConfBuilder.from_poses(poses, topol))
    rolled = dna_backmap(ConfBuilder.from_poses(np.roll(poses, -shift, axis=0), topol))
    np.testing.assert_allclose(rolled, np.roll(ref, -shift * cg, axis=0), atol=1e-4)


def test_closed_topology_with_open_ends_is_backmapped(closed_cg10):
    # ends beyond the closure distance are extended along the last tangent, not fitted periodically
    topol, _, poses = closed_cg10
    poses = poses.copy()
    poses[:, :3, 3] *= 2.5
    bp_poses = dna_backmap(ConfBuilder.from_poses(poses, topol))
    assert bp_poses.shape == (len(topol.sequence), 4, 4)
    assert np.isfinite(bp_poses).all()


def test_periodic_spline_requires_repeated_first_point():
    pts = np.c_[np.cos(np.arange(6)), np.sin(np.arange(6)), np.zeros(6)]
    with pytest.raises(ValueError, match='pts\\[-1\\] == pts\\[0\\]'):
        spline_interpolate_constant_arclength_segments(pts, 5, periodic=True)
