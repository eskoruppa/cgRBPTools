"""End-to-end tests of the configuration import in lmp_input (run as a subprocess)."""
import numpy as np
import pytest
import scipy as sp

from conftest import EXAMPLES, build_topology, read_data, run_lmp_input
from cgrbptools import ConfBuilder

SEQ = EXAMPLES / '200bp'
BASE_ARGS = ['-seqfn', SEQ, '-m', 'crystal']


def lmp(tmp_path, *args):
    return run_lmp_input([*BASE_ARGS, *args], cwd=tmp_path)


def relative_poses(poses):
    """Poses relative to the first one (removes the global placement)."""
    inv = np.linalg.inv(poses[0])
    return np.einsum('ij,njk->nik', inv, poses)


@pytest.fixture(scope='module')
def gs_poses(tmp_path_factory):
    """Exact ground-state poses (nm) at bead (cg 10) and base-pair (cg 1) resolution, open and closed."""
    d = tmp_path_factory.mktemp('poses')
    files = {}
    for cg in (10, 1):
        topol, _ = build_topology(cg=cg)
        files[f'open_cg{cg}'] = d / f'open_cg{cg}.npy'
        np.save(files[f'open_cg{cg}'], ConfBuilder.build(topol, 'gs').poses_in_nm())
        topol, _ = build_topology(cg=cg, closed=True)
        files[f'closed_cg{cg}'] = d / f'closed_cg{cg}.npy'
        np.save(files[f'closed_cg{cg}'], ConfBuilder.circular(topol, excess_link=0).poses_in_nm())
    return files


def test_bead_resolution_round_trip_is_exact(tmp_path, gs_poses):
    ref = lmp(tmp_path, '-cg', 10, '-conf', 'gs', '-o', tmp_path / 'ref')
    imp = lmp(tmp_path, '-cg', 10, '-conffn', gs_poses['open_cg10'], '-o', tmp_path / 'imp')
    assert ref.returncode == 0 and imp.returncode == 0, imp.stderr
    assert (tmp_path / 'ref.data').read_bytes() == (tmp_path / 'imp.data').read_bytes()
    assert (tmp_path / 'ref.db').read_bytes() == (tmp_path / 'imp.db').read_bytes()
    assert 'Elastic energy of the configuration: 0.0 kT' in imp.stdout
    # re-importing the written configuration leaves it untouched and reproduces the run
    again = lmp(tmp_path, '-cg', 10, '-conffn', tmp_path / 'imp_conf.npy', '-o', tmp_path / 'imp')
    assert again.returncode == 0 and 'left unchanged' in again.stdout
    assert (tmp_path / 'ref.data').read_bytes() == (tmp_path / 'imp.data').read_bytes()


@pytest.mark.parametrize('centered', [False, True])
def test_base_pair_resolution_is_coarse_grained_like_the_model(tmp_path, gs_poses, centered):
    flag = ['-centered'] if centered else []
    ref = lmp(tmp_path, '-cg', 10, *flag, '-conf', 'gs', '-o', tmp_path / 'ref')
    imp = lmp(tmp_path, '-cg', 10, *flag, '-conffn', gs_poses['open_cg1'], '-vis', '-o', tmp_path / 'imp')
    assert ref.returncode == 0 and imp.returncode == 0, imp.stderr
    assert 'at base-pair resolution -> 20 beads' in imp.stdout
    np.testing.assert_allclose(
        relative_poses(read_data(tmp_path / 'imp.data')), relative_poses(read_data(tmp_path / 'ref.data')), atol=1e-10
    )
    assert np.load(tmp_path / 'imp_conf.npy').shape == (20, 4, 4)


def test_mismatch_and_truncation(tmp_path, gs_poses):
    short = tmp_path / 'short.npy'
    np.save(short, np.load(gs_poses['open_cg10'])[:15])
    res = lmp(tmp_path, '-cg', 10, '-conffn', short, '-o', tmp_path / 'x')
    assert res.returncode != 0
    assert 'contains 15 poses, but the sequence (200 bp, composite size 10, open) requires 20 poses' in res.stderr
    assert not (tmp_path / 'x.db').exists()
    res = lmp(tmp_path, '-cg', 10, '-conffn', short, '-trunc', '--conf_resolution', 'cg', '-o', tmp_path / 't')
    assert res.returncode == 0, res.stderr
    assert len((tmp_path / 't.seq').read_text().strip()) == 150
    assert len(read_data(tmp_path / 't.data')) == 15
    res = lmp(tmp_path, '-cg', 10, '-eid', 150, '-conffn', gs_poses['open_cg1'], '-o', tmp_path / 'e')
    assert res.returncode != 0 and 'before -sid/-eid were applied' in res.stderr


def test_closed_configuration_and_linking_number(tmp_path, gs_poses):
    pytest.importorskip('cgrbptools.evals.PyLk.pylk')
    ring = np.load(gs_poses['closed_cg10'])
    repeated = tmp_path / 'repeated.npy'
    np.save(repeated, np.concatenate([ring, ring[:1]]))
    res = lmp(tmp_path, '-cg', 10, '-closed', '-conffn', repeated, '-o', tmp_path / 'c0')
    assert res.returncode == 0, res.stderr
    assert 'repeated pose was removed' in res.stderr and 'Lk = 19.0' in res.stdout
    res = lmp(tmp_path, '-cg', 10, '-closed', '-conffn', gs_poses['closed_cg10'], '-dlk', 3, '-o', tmp_path / 'c3')
    assert res.returncode == 0, res.stderr
    assert 'Added +3 turn(s) of twist' in res.stdout and 'target 22' in res.stdout
    res = lmp(tmp_path, '-cg', 10, '-closed', '-conffn', gs_poses['closed_cg1'], '-o', tmp_path / 'c1')
    assert res.returncode == 0 and 'at base-pair resolution -> 20 beads' in res.stdout


def test_flag_errors(tmp_path, gs_poses):
    res = lmp(tmp_path, '-cg', 10, '--conf_frame', 0, '-o', tmp_path / 'x')
    assert res.returncode == 2 and '--conf_frame applies only to an imported configuration' in res.stderr
    res = lmp(tmp_path, '-cg', 10, '-closed', '-conffn', gs_poses['closed_cg10'], '-trunc', '-o', tmp_path / 'x')
    assert res.returncode == 2 and 'not available for closed topologies' in res.stderr
    res = lmp(tmp_path, '-cg', 10, '-conf', 'gs', '-conffn', gs_poses['open_cg10'], '-o', tmp_path / 'x')
    assert res.returncode == 2 and 'not allowed with argument' in res.stderr
    traj = tmp_path / 'traj.npy'
    np.save(traj, np.stack([np.load(gs_poses['open_cg10'])] * 2))
    res = lmp(tmp_path, '-cg', 10, '-conffn', traj, '-o', tmp_path / 'x')
    assert res.returncode != 0 and '-1 selects the last snapshot' in res.stderr
    res = lmp(tmp_path, '-cg', 10, '-conffn', traj, '--conf_frame', -1, '-o', tmp_path / 'x')
    assert res.returncode == 0, res.stderr


def test_units(tmp_path, gs_poses):
    poses = np.load(gs_poses['open_cg10'])
    poses[:, :3, 3] *= 10
    angstrom = tmp_path / 'angstrom.npy'
    np.save(angstrom, poses)
    res = lmp(tmp_path, '-cg', 10, '-conffn', angstrom, '-o', tmp_path / 'x')
    assert res.returncode != 0 and '--conf_units angstrom' in res.stderr
    res = lmp(tmp_path, '-cg', 10, '-conffn', angstrom, '--conf_units', 'angstrom', '-o', tmp_path / 'a')
    assert res.returncode == 0, res.stderr
    res = lmp(tmp_path, '-cg', 10, '-ul', 3.4, '-conffn', gs_poses['open_cg10'], '-o', tmp_path / 'u')
    assert res.returncode == 0, res.stderr
    ref = lmp(tmp_path, '-cg', 10, '-ul', 3.4, '-conf', 'gs', '-o', tmp_path / 'uref')
    np.testing.assert_allclose(read_data(tmp_path / 'u.data'), read_data(tmp_path / 'uref.data'), atol=1e-12)


def test_no_crop_requires_fitting_length(tmp_path):
    res = lmp(tmp_path, '-cg', 10, '-nc', '-conf', 'gs', '-o', tmp_path / 'x')
    assert res.returncode != 0
    assert '-nc/--no_crop is set, but the sequence would have to be cropped' in res.stderr
    res = run_lmp_input(['-seqfn', EXAMPLES / '201bp', '-m', 'crystal', '-cg', 10, '-nc', '-conf', 'gs',
                         '-o', tmp_path / 'y'], cwd=tmp_path)
    assert res.returncode == 0, res.stderr


def test_unit_energy(tmp_path):
    for ue in (1, 2):
        res = lmp(tmp_path, '-cg', 10, '-ue', ue, '-coeffs', '-o', tmp_path / f'ue{ue}')
        assert res.returncode == 0, res.stderr
    stiff1 = sp.sparse.load_npz(tmp_path / 'ue1_stiff.npz').toarray()
    stiff2 = sp.sparse.load_npz(tmp_path / 'ue2_stiff.npz').toarray()
    np.testing.assert_allclose(stiff2, stiff1 / 2)
    np.testing.assert_array_equal(np.load(tmp_path / 'ue1_gs.npy'), np.load(tmp_path / 'ue2_gs.npy'))


def test_fene_and_strict_checks(tmp_path, gs_poses):
    fene = ['-fene', 200, 1.1, 1.35]
    res = lmp(tmp_path, '-cg', 10, '-closed', '-conf', 'circ', *fene, '-o', tmp_path / 'b')
    assert res.returncode == 0 and 'LAMMPS aborts' in res.stderr
    res = lmp(tmp_path, '-cg', 10, '-closed', '-conffn', gs_poses['closed_cg10'], *fene, '-o', tmp_path / 'i')
    assert res.returncode != 0 and 'Bad RBP FENE bond' in res.stderr
    res = lmp(tmp_path, '-cg', 10, '-closed', '-conffn', gs_poses['closed_cg10'], *fene, '-ul', 3.4, '-o', tmp_path / 'u')
    assert res.returncode == 0, res.stderr
    pytest.importorskip('cgrbptools.evals.PyLk.pylk')
    res = lmp(tmp_path, '-cg', 10, '-closed', '-conffn', gs_poses['closed_cg10'], '-dlk', 3,
              '--strict_conf_check', '-o', tmp_path / 's')
    assert res.returncode != 0 and '--strict_conf_check' in res.stderr


def test_excess_link_warnings(tmp_path, gs_poses):
    res = lmp(tmp_path, '-cg', 10, '-conffn', gs_poses['open_cg10'], '-dlk', 2, '-o', tmp_path / 'o')
    assert res.returncode == 0 and 'no effect for an imported open configuration' in res.stderr
    res = lmp(tmp_path, '-cg', 10, '-conf', 'gs', '-dlk', 2, '-o', tmp_path / 'g')
    assert res.returncode == 0 and 'no effect for -conf ground_state' in res.stderr
