"""Tests for cgrbptools.core.conf_import: loading, validation and matching of external configurations."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import random_poses
from cgrbptools.core.conf_import import (
    ConfigurationLoadError,
    ConfigurationMismatchError,
    ConfigurationValidationError,
    expected_num_beads,
    load_poses,
    match_sequence_and_poses,
    num_cropped_bp,
    positions_to_nm,
    remove_repeated_pose,
    truncated_seq_len,
    validate_poses,
    write_poses,
)


def mapped_files():
    return Path('/proc/self/maps').read_text()


###################################################################################################
# Loading: independence from the source

def test_npy_returns_independent_float64_copy(tmp_path):
    f = tmp_path / 'conf.npy'
    ref = random_poses(12)
    np.save(f, ref)
    out = load_poses(f)
    assert type(out) is np.ndarray and out.base is None
    assert out.dtype == np.float64 and out.shape == (12, 4, 4)
    np.testing.assert_array_equal(out, ref)


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='uses /proc/self/maps')
def test_npy_file_is_unmapped_after_loading(tmp_path):
    f = tmp_path / 'conf_unmap.npy'
    np.save(f, random_poses(5))
    load_poses(f)
    assert str(f) not in mapped_files()
    traj = tmp_path / 'traj_unmap.npy'
    np.save(traj, np.stack([random_poses(5, s) for s in range(4)]))
    load_poses(traj, frame=2)
    assert str(traj) not in mapped_files()


def test_overwriting_the_input_file_after_loading(tmp_path):
    f = tmp_path / 'x_conf.npy'
    ref = random_poses(8)
    np.save(f, ref)
    out = load_poses(f)
    np.save(f, ref * 2.0)
    np.testing.assert_array_equal(out, ref)
    f.unlink()
    np.testing.assert_array_equal(out, ref)


def test_array_and_memmap_input_are_copied(tmp_path):
    src = random_poses(4)
    out = load_poses(src)
    out[0, 0, 3] = 99.0
    assert src[0, 0, 3] != 99.0
    f = tmp_path / 'mm.npy'
    np.save(f, src)
    out = load_poses(np.load(f, mmap_mode='r'))
    assert type(out) is np.ndarray and out.base is None


###################################################################################################
# Loading: invalid data

def test_object_and_pickled_data_are_rejected(tmp_path):
    f = tmp_path / 'obj.npy'
    np.save(f, np.array([{'a': 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ConfigurationLoadError, match='Pickled or object data'):
        load_poses(f)
    f = tmp_path / 'pickled.npy'
    with open(f, 'wb') as fh:
        pickle.dump(random_poses(3), fh)
    with pytest.raises(ConfigurationLoadError, match='Pickled or object data'):
        load_poses(f)
    f = tmp_path / 'obj.npz'
    np.savez(f, poses=np.array([None], dtype=object))
    with pytest.raises(ConfigurationLoadError, match='Pickled or object data'):
        load_poses(f)


@pytest.mark.parametrize('arr', [
    np.zeros((3, 4, 4), dtype=[('x', 'f8')]),
    np.zeros((3, 4, 4), dtype=complex),
    np.zeros((3, 4, 4), dtype=bool),
    np.full((3, 4, 4), 'a'),
])
def test_non_numeric_dtypes_are_rejected(tmp_path, arr):
    with pytest.raises(ConfigurationLoadError, match='unsupported data type'):
        load_poses(arr)
    f = tmp_path / 'bad.npy'
    np.save(f, arr)
    with pytest.raises(ConfigurationLoadError, match='unsupported data type'):
        load_poses(f)


def test_nan_and_inf_are_rejected_with_indices():
    p = random_poses(20)
    p[3, 0, 0] = np.nan
    p[7, 1, 3] = np.inf
    with pytest.raises(ConfigurationLoadError, match=r'2 of 20 poses \(0-based indices: 3, 7\)'):
        load_poses(p)
    p = random_poses(20)
    p[:13, 2, 3] = -np.inf
    with pytest.raises(ConfigurationLoadError, match='13 of 20 poses .* and 3 more'):
        load_poses(p)


def test_nan_in_unselected_frame_is_ignored():
    traj = np.stack([random_poses(5, s) for s in range(3)])
    traj[0, 2, 0, 0] = np.nan
    np.testing.assert_array_equal(load_poses(traj, frame=1), traj[1])
    with pytest.raises(ConfigurationLoadError, match='NaN'):
        load_poses(traj, frame=0)


def test_integer_and_float32_are_converted():
    p = random_poses(3)
    np.testing.assert_array_equal(load_poses(p.astype(np.float32)), p.astype(np.float32).astype(np.float64))
    assert load_poses(np.tile(np.eye(4, dtype=np.int64), (3, 1, 1))).dtype == np.float64


@pytest.mark.parametrize('shape', [(5, 3, 3), (5, 16), (5, 4, 4, 1), (4, 4), (2, 3, 5, 4, 4)])
def test_wrong_shapes_are_rejected(shape):
    with pytest.raises(ConfigurationLoadError, match='has shape'):
        load_poses(np.zeros(shape))


def test_frame_selection():
    traj = np.stack([random_poses(5, s) for s in range(4)])
    with pytest.raises(ConfigurationLoadError, match=r'trajectory of 4 snapshots.*--conf_frame.*-1 selects'):
        load_poses(traj)
    np.testing.assert_array_equal(load_poses(traj, frame=-1), traj[3])
    np.testing.assert_array_equal(load_poses(traj, frame=np.int64(0)), traj[0])
    for bad in (4, -5):
        with pytest.raises(ConfigurationLoadError, match='out of range'):
            load_poses(traj, frame=bad)
    for bad in (True, 1.0, '1'):
        with pytest.raises(TypeError):
            load_poses(traj, frame=bad)
    with pytest.raises(ConfigurationLoadError, match='single configuration'):
        load_poses(random_poses(3), frame=0)


def test_npz_variants(tmp_path):
    p = random_poses(6)
    np.savez(tmp_path / 'a.npz', poses=p)
    np.testing.assert_array_equal(load_poses(tmp_path / 'a.npz'), p)
    np.savez(tmp_path / 'b.npz', positions=p[:, :3, 3], triads=p[:, :3, :3], extra=np.arange(3))
    np.testing.assert_array_equal(load_poses(tmp_path / 'b.npz'), p)
    traj = np.stack([random_poses(6, s) for s in range(3)])
    np.savez(tmp_path / 'c.npz', positions=traj[:, :, :3, 3], triads=traj[:, :, :3, :3])
    np.testing.assert_array_equal(load_poses(tmp_path / 'c.npz', frame=-1), traj[-1])
    with pytest.raises(ConfigurationLoadError, match='trajectory of 3 snapshots'):
        load_poses(tmp_path / 'c.npz')


def test_npz_key_errors(tmp_path):
    p = random_poses(4)
    np.savez(tmp_path / 'both.npz', poses=p, positions=p[:, :3, 3], triads=p[:, :3, :3])
    with pytest.raises(ConfigurationLoadError, match='only one representation'):
        load_poses(tmp_path / 'both.npz')
    np.savez(tmp_path / 'half.npz', positions=p[:, :3, 3])
    with pytest.raises(ConfigurationLoadError, match="'positions' but not 'triads'"):
        load_poses(tmp_path / 'half.npz')
    np.savez(tmp_path / 'none.npz', coords=p)
    with pytest.raises(ConfigurationLoadError, match=r"Found: \['coords'\]"):
        load_poses(tmp_path / 'none.npz')
    np.savez(tmp_path / 'mismatch.npz', positions=p[:3, :3, 3], triads=p[:, :3, :3])
    with pytest.raises(ConfigurationLoadError, match=r'Expected \(N, 3\) and \(N, 3, 3\)'):
        load_poses(tmp_path / 'mismatch.npz')


def test_file_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_poses(tmp_path / 'missing.npy')
    f = tmp_path / 'conf.txt'
    f.write_text('1 2 3')
    with pytest.raises(ConfigurationLoadError, match="Unsupported configuration file type '.txt'"):
        load_poses(f)
    with pytest.raises(ConfigurationLoadError, match='contains no poses'):
        load_poses(np.zeros((0, 4, 4)))


###################################################################################################
# Units and validity

def test_positions_to_nm():
    p = random_poses(3)
    np.testing.assert_array_equal(positions_to_nm(p, 'nm'), p)
    np.testing.assert_allclose(positions_to_nm(p, 'angstrom')[:, :3, 3], p[:, :3, 3] / 10)
    np.testing.assert_allclose(positions_to_nm(p, 'sim', unit_length=3.4)[:, :3, 3], p[:, :3, 3] * 3.4)
    np.testing.assert_array_equal(positions_to_nm(p, 'angstrom')[:, :3, :3], p[:, :3, :3])
    with pytest.raises(ValueError):
        positions_to_nm(p, 'um')


def test_exact_rotations_are_left_untouched():
    p = random_poses(10)
    out, note = validate_poses(p)
    assert note is None
    np.testing.assert_array_equal(out, p)


def test_small_deviations_are_repaired():
    p = random_poses(10)
    noisy = p.copy()
    noisy[2, :3, :3] += 2e-5
    out, note = validate_poses(noisy)
    assert 're-orthonormalized 1 rotation block' in note
    r = out[2, :3, :3]
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(r, p[2, :3, :3], atol=1e-4)
    np.testing.assert_array_equal(out[[0, 1, 3]], p[[0, 1, 3]])


def test_large_deviations_require_force():
    p = random_poses(10)
    p[4, :3, :3] *= 1.01
    with pytest.raises(ConfigurationValidationError, match=r'--force_orthogonalize.*') as e:
        validate_poses(p)
    assert '0-based indices: 4' in str(e.value)
    out, note = validate_poses(p, force_orthogonalize=True)
    assert 'forced' in note
    np.testing.assert_allclose(np.linalg.det(out[:, :3, :3]), 1.0)


def test_improper_degenerate_and_transposed_poses_are_rejected():
    p = random_poses(5)
    p[1, :3, 0] *= -1
    with pytest.raises(ConfigurationValidationError, match='improper rotation'):
        validate_poses(p, force_orthogonalize=True)
    p = random_poses(5)
    p[3, :3, :3] = 0.0
    with pytest.raises(ConfigurationValidationError, match='degenerate'):
        validate_poses(p, force_orthogonalize=True)
    p = random_poses(5)
    with pytest.raises(ConfigurationValidationError, match='transposed'):
        validate_poses(np.transpose(p, (0, 2, 1)))


def test_repeated_pose(closed_cg10):
    _, _, ring = closed_cg10
    out, note = remove_repeated_pose(ring)
    assert note is None and len(out) == len(ring)
    out, note = remove_repeated_pose(np.concatenate([ring, ring[:1]]))
    assert 'repeated pose was removed' in note
    np.testing.assert_array_equal(out, ring)
    near = np.concatenate([ring, ring[:1]])
    near[-1, :3, 3] += 0.05
    with pytest.raises(ConfigurationValidationError, match='nearly coincides'):
        remove_repeated_pose(near)


###################################################################################################
# Matching configuration and sequence

@pytest.mark.parametrize('cg, center, closed, seq_len, expected', [
    (10, 0, False, 200, 20), (10, 0, False, 201, 21), (10, 0, False, 205, 21), (10, 5, False, 200, 20),
    (10, 5, False, 206, 21), (7, 3, False, 1000, 143), (1, 0, False, 200, 200), (10, 0, True, 200, 20),
    (10, 5, True, 200, 20),
])
def test_expected_num_beads(cg, center, closed, seq_len, expected):
    assert expected_num_beads(seq_len, cg, closed, center) == expected


def test_truncated_seq_len_yields_the_beads_without_cropping_issues():
    for cg in (2, 3, 7, 10):
        for center in (0, cg // 2):
            for beads in (2, 5, 13):
                crop_len = truncated_seq_len(beads, cg, center, allow_crop=True)
                assert expected_num_beads(crop_len, cg, False, center) == beads
                assert crop_len % cg == 0  # complete last block, valid for centered beads
                nocrop_len = truncated_seq_len(beads, cg, center, allow_crop=False)
                assert expected_num_beads(nocrop_len, cg, False, center) == beads
                assert num_cropped_bp(nocrop_len, cg, False, center) == 0


def test_exact_matches_at_both_resolutions():
    seq = 'A' * 200
    beads = random_poses(20)
    m = match_sequence_and_poses(seq, beads, 10)
    assert m.resolution == 'cg' and m.sequence == seq and m.bp_poses is None
    np.testing.assert_array_equal(m.bead_poses, beads)
    bp = random_poses(200)
    m = match_sequence_and_poses(seq, bp, 10)
    assert m.resolution == 'bp'
    np.testing.assert_array_equal(m.bead_poses, bp[::10])
    np.testing.assert_array_equal(m.bp_poses, bp)
    m = match_sequence_and_poses(seq, bp, 10, center_offset=5)
    np.testing.assert_array_equal(m.bead_poses, bp[5::10])
    m = match_sequence_and_poses(seq, bp, 10, closed=True, center_offset=5)
    np.testing.assert_array_equal(m.bead_poses, bp[5::10])


def test_mismatch_message():
    with pytest.raises(ConfigurationMismatchError) as e:
        match_sequence_and_poses('A' * 200, random_poses(15), 10, name='The test configuration')
    msg = str(e.value)
    assert 'contains 15 poses' in msg and 'requires 20 poses at bead resolution or 200 poses at base-pair' in msg
    assert '-trunc/--truncate_to_match' in msg
    with pytest.raises(ConfigurationMismatchError, match='truncation is not available for closed'):
        match_sequence_and_poses('A' * 200, random_poses(15), 10, closed=True)
    with pytest.raises(ConfigurationMismatchError, match='before -sid/-eid were applied'):
        match_sequence_and_poses('A' * 150, random_poses(200), 10, uncropped_seq_len=200)


def test_truncation():
    seq = 'ACGT' * 50
    beads = random_poses(20)
    m = match_sequence_and_poses(seq, beads[:15], 10, truncate=True, resolution='cg')
    assert m.sequence == seq[:150] and len(m.bead_poses) == 15
    m = match_sequence_and_poses(seq, beads[:15], 10, truncate=True, resolution='cg', allow_crop=False)
    assert len(m.sequence) == 141 and len(m.bead_poses) == 15
    m = match_sequence_and_poses(seq[:150], beads, 10, truncate=True, resolution='cg')
    assert m.sequence == seq[:150]
    np.testing.assert_array_equal(m.bead_poses, beads[:15])
    bp = random_poses(200)
    m = match_sequence_and_poses(seq[:150], bp, 10, truncate=True, resolution='bp')
    np.testing.assert_array_equal(m.bead_poses, bp[:150:10])
    np.testing.assert_array_equal(m.bp_poses, bp[:150])
    # configuration ending inside a block: the block is kept complete, its base pairs have no poses
    m = match_sequence_and_poses(seq, bp[:145], 10, truncate=True, resolution='bp')
    assert len(m.sequence) == 150 and len(m.bead_poses) == 15 and m.bp_poses is None
    m = match_sequence_and_poses(seq, bp[:145], 10, truncate=True, resolution='bp', center_offset=5)
    assert len(m.sequence) == 140
    np.testing.assert_array_equal(m.bead_poses, bp[5:140:10])
    with pytest.raises(ConfigurationMismatchError, match='--conf_resolution'):
        match_sequence_and_poses(seq, beads[:15], 10, truncate=True)
    with pytest.raises(ConfigurationMismatchError, match='not available for closed'):
        match_sequence_and_poses(seq, beads[:15], 10, truncate=True, closed=True)


###################################################################################################
# Output

def test_write_poses_protects_the_input(tmp_path):
    f = tmp_path / 'x_conf.npy'
    p = random_poses(5)
    np.save(f, p)
    assert 'left unchanged' in write_poses(f, p.copy(), input_file=f)
    np.testing.assert_array_equal(np.load(f), p)
    assert 'moved to' in write_poses(f, p[:3], input_file=f)
    np.testing.assert_array_equal(np.load(tmp_path / 'x_conf_#1.npy'), p)
    np.testing.assert_array_equal(np.load(f), p[:3])
    other = tmp_path / 'y_conf.npy'
    assert 'written' in write_poses(other, p, input_file=f)
    np.testing.assert_array_equal(np.load(other), p)
