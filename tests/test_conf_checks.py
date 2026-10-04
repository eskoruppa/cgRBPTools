"""Tests for cgrbptools.core.conf_checks: geometry, FENE, energy and linking-number checks."""
import numpy as np
import pytest
import scipy as sp

from cgrbptools.core import conf_checks as cc
from cgrbptools.core.conf_import import ConfigurationValidationError


def banded_spd(n, bandwidth, closed, seed=1):
    rng = np.random.default_rng(seed)
    mat = np.zeros((6 * n, 6 * n))
    for i in range(n):
        for d in range(bandwidth + 1):
            j = i + d
            if j >= n and not closed:
                continue
            j %= n
            block = rng.normal(size=(6, 6)) * (0.3 if d else 1.0)
            if d == 0:
                block = block @ block.T + 30 * np.eye(6)
            mat[6 * i:6 * i + 6, 6 * j:6 * j + 6] += block
            if d:
                mat[6 * j:6 * j + 6, 6 * i:6 * i + 6] += block.T
    return sp.sparse.csr_matrix(mat)


def dense_cov_blocks(mat):
    cov = np.linalg.inv(mat.toarray())
    return np.array([cov[6 * i:6 * i + 6, 6 * i:6 * i + 6] for i in range(mat.shape[0] // 6)])


###################################################################################################
# Marginals

@pytest.mark.parametrize('bandwidth', [1, 2, 3])
def test_block_tridiagonal_recursion_is_exact(bandwidth):
    mat = banded_spd(203, bandwidth, closed=False)
    with cc._single_threaded():
        blocks = cc._blocktridiag_cov_blocks(mat)
    np.testing.assert_allclose(blocks, dense_cov_blocks(mat), rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize('bandwidth', [1, 2])
def test_windowed_marginals_for_closed_chains(bandwidth):
    mat = banded_spd(203, bandwidth, closed=True)
    with cc._single_threaded():
        blocks = cc._windowed_cov_blocks(mat)
    np.testing.assert_allclose(blocks, dense_cov_blocks(mat), rtol=1e-6, atol=1e-12)


def test_large_systems_use_the_sparse_paths(monkeypatch):
    monkeypatch.setattr(cc, 'CONF_DENSE_INVERSION_MAX_DIM', 100)
    for closed in (False, True):
        mat = banded_spd(120, 1, closed=closed)
        np.testing.assert_allclose(
            cc.marginal_stiffness_blocks(mat, closed=closed), np.linalg.inv(dense_cov_blocks(mat)), rtol=1e-6
        )


def test_not_positive_definite_is_detected():
    mat = banded_spd(50, 1, closed=False).toarray()
    mat[0, 0] = -5.0
    with pytest.raises(np.linalg.LinAlgError):
        cc._blocktridiag_cov_blocks(sp.sparse.csr_matrix(mat))
    with pytest.raises(np.linalg.LinAlgError):
        cc.marginal_stiffness_blocks(mat)


###################################################################################################
# Energy

def test_ground_state_has_zero_energy(open_cg10):
    topol, _, poses = open_cg10
    np.testing.assert_allclose(cc.deformations(poses, topol.groundstate), 0.0, atol=1e-12)
    info, warnings = cc.energy_check(poses, topol)
    assert 'Elastic energy of the configuration: 0.0 kT' in info[0]
    assert warnings == []


def test_strong_local_deformation_is_flagged(open_cg10):
    topol, _, poses = open_cg10
    bent = poses.copy()
    rot = sp.spatial.transform.Rotation.from_euler('x', 60, degrees=True).as_matrix()
    # bend the chain by 60 degrees at bead 10
    pivot = bent[10, :3, 3].copy()
    for i in range(10, len(bent)):
        bent[i, :3, :3] = rot @ bent[i, :3, :3]
        bent[i, :3, 3] = pivot + rot @ (bent[i, :3, 3] - pivot)
    _, warnings = cc.energy_check(bent, topol)
    assert any('junction(s) are deformed far beyond thermal fluctuations' in w and ' 9 (' in w for w in warnings)


def test_subtract_groundstate_convention(open_cg10):
    topol, _, poses = open_cg10
    np.testing.assert_allclose(cc.deformations(poses, topol.groundstate, subtract_groundstate=True), 0.0, atol=1e-12)


###################################################################################################
# Geometry

def test_geometry_of_ground_state_passes(open_cg10, closed_cg10):
    topol, _, poses = open_cg10
    assert cc.check_geometry(poses, topol.groundstate) == []
    topol, _, ring = closed_cg10
    assert cc.check_geometry(ring, topol.groundstate, closed=True) == []


def test_geometry_errors(open_cg10, closed_cg10):
    topol, _, poses = open_cg10
    scaled = poses.copy()
    scaled[:, :3, 3] *= 10
    with pytest.raises(ConfigurationValidationError, match='Angstrom'):
        cc.check_geometry(scaled, topol.groundstate)
    dup = poses.copy()
    dup[5] = dup[4]
    with pytest.raises(ConfigurationValidationError, match='zero bond length'):
        cc.check_geometry(dup, topol.groundstate)
    reversed_order = poses[::-1].copy()
    with pytest.raises(ConfigurationValidationError, match='does not follow the chain'):
        cc.check_geometry(reversed_order, topol.groundstate)
    topol_closed, _, ring = closed_cg10
    opened = ring.copy()
    opened[-1, :3, 3] += 20.0
    with pytest.raises(ConfigurationValidationError, match='not appear to be closed'):
        cc.check_geometry(opened, topol_closed.groundstate, closed=True)


def test_overlap_warning(open_cg10):
    topol, _, poses = open_cg10
    folded = poses.copy()
    folded[15, :3, 3] = folded[3, :3, 3] + np.array([0.5, 0.0, 0.0])
    warnings = cc.check_geometry(folded, topol.groundstate)
    assert any('non-neighbouring beads' in w and '3-15' in w for w in warnings)


###################################################################################################
# FENE

def test_fene_regimes():
    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[:, 2, 3] = [0.0, 1.0, 2.15, 4.0]   # bonds 1.0, 1.15, 1.85
    warnings = cc.check_fene(poses, Rc=1.1, R0=1.35, abort_is_error=False)
    joined = ' '.join(warnings)
    assert 'LAMMPS aborts' in joined and 'continues the potential linearly' not in joined
    with pytest.raises(ConfigurationValidationError, match='Bad RBP FENE bond'):
        cc.check_fene(poses, Rc=1.1, R0=1.35)
    poses[:, 2, 3] = [0.0, 1.0, 2.15, 3.55]   # bonds 1.0, 1.15 (active), 1.40 (linear regime)
    warnings = cc.check_fene(poses, Rc=1.1, R0=1.35)
    assert any('continues the potential linearly' in w for w in warnings)
    assert any('active from the start for 1 bond' in w for w in warnings)
    assert cc.check_fene(poses, Rc=2.0, R0=2.5) == []


###################################################################################################
# Linking number

def test_relaxed_linking_number_restores_full_turns(closed_cg10):
    topol, params, _ = closed_cg10
    lk_cg = cc.relaxed_linking_number(topol)
    lk = cc.relaxed_linking_number(topol, params.shape_params)
    assert lk - lk_cg == pytest.approx(round(lk - lk_cg))
    assert lk == pytest.approx(200 / 10.5, abs=0.5)


def test_twist_adjustment(closed_cg10):
    pytest.importorskip('cgrbptools.evals.PyLk.pylk')
    topol, params, ring = closed_cg10
    _, info, warnings = cc.adjust_excess_link(ring, topol, bp_groundstate=params.shape_params)
    assert warnings == []
    lk0 = cc.relaxed_linking_number(topol, params.shape_params)
    for dlk in (3, -2):
        adjusted, info, _ = cc.adjust_excess_link(ring, topol, excess_link=dlk, bp_groundstate=params.shape_params)
        np.testing.assert_array_equal(adjusted[:, :3, 3], ring[:, :3, 3])
        np.testing.assert_allclose(adjusted[:, :3, 2], ring[:, :3, 2], atol=1e-12)  # tangents unchanged
        _, _, excess = cc.linking_number(adjusted, topol)
        assert lk0 + excess == pytest.approx(round(lk0 + dlk), abs=cc.CONF_LK_INTEGER_TOL)
    with pytest.raises(ConfigurationValidationError, match='degrees of twist per step'):
        cc.adjust_excess_link(ring, topol, excess_link=10)
