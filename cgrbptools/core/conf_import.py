"""
Import of externally generated configurations.

An external configuration is an array of SE(3) poses (4x4 matrices with the triad in the
upper-left 3x3 block, its columns being the triad vectors, and the position in the last
column). It is either given at bead resolution (one pose per coarse-grained bead) or at
base-pair resolution (one pose per base pair of the sequence), in which case it is
coarse-grained by retaining the frames of the beads, exactly as the elastic model does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..io.backups import backup_filename

# Keys accepted in .npz configuration files: either the full poses or positions plus triads
CONF_NPZ_POSES_KEY = 'poses'
CONF_NPZ_POSITIONS_KEY = 'positions'
CONF_NPZ_TRIADS_KEY = 'triads'

# Maximum number of offending pose indices listed in error messages
CONF_MAX_LISTED_INDICES = 10

# Length units accepted for imported positions and their conversion factors to nm
# ('sim' denotes simulation units, i.e. multiples of the unit length)
CONF_UNITS = ('nm', 'angstrom', 'sim')
CONF_UNITS_TO_NM = {'nm': 1.0, 'angstrom': 0.1}

# Resolutions of imported configurations: deduced from the number of poses, base pairs or beads
CONF_RESOLUTIONS = ('auto', 'bp', 'cg')

# Rotation blocks deviating from orthonormality by up to this value (max. abs. entry of R^T R - I)
# are re-orthonormalized automatically; larger deviations require force_orthogonalize.
CONF_ORTHO_TOL = 1e-4
# Rotation blocks deviating by less than this are considered exact and left untouched
CONF_ORTHO_EXACT = 1e-12
# Rotation blocks with a singular value below this are degenerate and cannot be repaired
CONF_DEGENERATE_TOL = 1e-6

# A closed configuration whose last pose repeats the first one within this tolerance (nm for the
# position, absolute for the rotation entries) has the repeated pose removed
CONF_DUPLICATE_TOL = 1e-8
# A last pose closer to the first one than this fraction of the median bond length is rejected
CONF_NEAR_DUPLICATE_FRACTION = 0.1


class ConfigurationError(ValueError):
    """Base class for errors raised for an external configuration."""


class ConfigurationLoadError(ConfigurationError):
    """Raised when an external configuration cannot be loaded or has an invalid format."""


class ConfigurationValidationError(ConfigurationError):
    """Raised when an external configuration fails a consistency or plausibility check."""


class ConfigurationMismatchError(ConfigurationError):
    """Raised when the number of poses does not match the sequence and coarse-graining."""


def load_poses(
    source: str | Path | np.ndarray,
    frame: int | None = None,
) -> np.ndarray:
    """
    Load an external configuration as an array of SE(3) poses.

    Accepted sources are
    - a numpy array of shape (N, 4, 4) (single configuration) or (T, N, 4, 4) (trajectory),
    - a .npy file holding such an array,
    - a .npz file holding either the key 'poses' with such an array, or the keys 'positions'
      ((N, 3) or (T, N, 3)) and 'triads' ((N, 3, 3) or (T, N, 3, 3)).

    The selected configuration is always returned as an independent float64 array, so the
    source (array or file) may be modified or overwritten afterwards. .npy files are memory
    mapped, such that only the selected snapshot of a trajectory is read from disk; it is
    copied out of the map before the file is released. Members of .npz files cannot be
    memory mapped and are read completely.

    Parameters
    ----------
    source : str, Path or np.ndarray
        Configuration array or path to a .npy or .npz file.
    frame : int, optional
        Snapshot to select from a trajectory. Negative values count from the end, i.e. -1
        selects the last snapshot. Required for trajectories, must be None for a single
        configuration.

    Returns
    -------
    np.ndarray
        Poses with shape (N, 4, 4) and dtype float64.

    Raises
    ------
    TypeError
        If frame is not an integer.
    FileNotFoundError
        If the configuration file does not exist.
    ConfigurationLoadError
        If the file type is not supported, the data is not a plain integer or floating point
        array (pickled objects, structured, complex, boolean or string data), the shape is
        invalid, the frame selection is missing, superfluous or out of range, or the selected
        configuration is empty or contains NaN or infinite values.

    Notes
    -----
    Only the format is checked here. Whether the poses are valid SE(3) elements is checked
    separately.
    """
    if frame is not None and (isinstance(frame, bool) or not isinstance(frame, (int, np.integer))):
        raise TypeError(f'frame must be an integer or None, got {type(frame).__name__}.')

    if isinstance(source, np.ndarray):
        name = 'The configuration array'
        poses = _select_poses(source, frame, name)
    else:
        path = Path(source)
        name = f"Configuration file '{path}'"
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file '{path}' does not exist.")
        suffix = path.suffix.lower()
        if suffix == '.npy':
            try:
                data = np.load(path, mmap_mode='r', allow_pickle=False)
            except ValueError as e:
                raise ConfigurationLoadError(
                    f'{name} does not contain a plain numeric array ({e}). '
                    f'Pickled or object data is not accepted.'
                ) from e
            # _select_poses returns a copy, so dropping the map here releases the file
            try:
                poses = _select_poses(data, frame, name)
            finally:
                del data
        elif suffix == '.npz':
            with np.load(path, allow_pickle=False) as npz:
                poses = _poses_from_npz(npz, frame, name)
        else:
            raise ConfigurationLoadError(
                f"Unsupported configuration file type '{path.suffix}' ({path}). Expected .npy or .npz."
            )

    if len(poses) == 0:
        raise ConfigurationLoadError(f'{name} contains no poses.')

    finite = np.isfinite(poses).all(axis=(1, 2))
    if not finite.all():
        bad = np.flatnonzero(~finite)
        raise ConfigurationLoadError(
            f'{name} contains NaN or infinite values in {len(bad)} of {len(poses)} poses '
            f'(0-based indices: {_listed(bad)}).'
        )
    return poses


def _listed(indices) -> str:
    """Comma separated list of the first CONF_MAX_LISTED_INDICES indices."""
    listed = ', '.join(str(i) for i in indices[:CONF_MAX_LISTED_INDICES])
    if len(indices) > CONF_MAX_LISTED_INDICES:
        listed += f' and {len(indices) - CONF_MAX_LISTED_INDICES} more'
    return listed


def _lc(name: str) -> str:
    """Name with lowercase first letter, for use inside a sentence."""
    return name[0].lower() + name[1:]


def _check_numeric(arr: np.ndarray, name: str) -> None:
    """Reject anything but plain integer or floating point arrays."""
    if arr.dtype.kind not in 'iuf':
        raise ConfigurationLoadError(
            f'{name} has unsupported data type {arr.dtype}. Only integer or floating point '
            f'arrays are accepted.'
        )


def _select_frame(arr: np.ndarray, frame: int | None, single_ndim: int, name: str) -> np.ndarray:
    """Return arr itself for a single configuration or the selected snapshot of a trajectory.

    arr has to have single_ndim (single configuration) or single_ndim+1 (trajectory) dimensions.
    """
    if arr.ndim == single_ndim:
        if frame is not None:
            raise ConfigurationLoadError(
                f'Frame {frame} was selected, but {_lc(name)} holds a single '
                f'configuration, not a trajectory.'
            )
        return arr

    nframes = arr.shape[0]
    if frame is None:
        raise ConfigurationLoadError(
            f'{name} holds a trajectory of {nframes} snapshots. Select one with frame '
            f'(command line: --conf_frame); -1 selects the last snapshot.'
        )
    if not -nframes <= frame < nframes:
        raise ConfigurationLoadError(
            f'Frame {frame} is out of range for the trajectory of {nframes} snapshots in '
            f'{_lc(name)} (valid: {-nframes} to {nframes - 1}).'
        )
    return arr[frame]


def _select_poses(arr: np.ndarray, frame: int | None, name: str) -> np.ndarray:
    """Validate a pose array and return a float64 copy of the selected configuration."""
    _check_numeric(arr, name)
    if arr.ndim not in (3, 4) or arr.shape[-2:] != (4, 4):
        raise ConfigurationLoadError(
            f'{name} has shape {arr.shape}. Expected (N, 4, 4) for a single configuration or '
            f'(T, N, 4, 4) for a trajectory.'
        )
    return np.array(_select_frame(arr, frame, 3, name), dtype=np.float64, copy=True)


def _npz_member(npz, key: str, name: str) -> np.ndarray:
    try:
        return npz[key]
    except ValueError as e:
        raise ConfigurationLoadError(
            f"{name}: entry '{key}' is not a plain numeric array ({e}). "
            f"Pickled or object data is not accepted."
        ) from e


def _poses_from_npz(npz, frame: int | None, name: str) -> np.ndarray:
    """Read the poses from a .npz file, given either as 'poses' or as 'positions' and 'triads'."""
    keys = set(npz.files)
    has_poses = CONF_NPZ_POSES_KEY in keys
    has_positions = CONF_NPZ_POSITIONS_KEY in keys
    has_triads = CONF_NPZ_TRIADS_KEY in keys

    if has_poses and (has_positions or has_triads):
        raise ConfigurationLoadError(
            f"{name} contains both '{CONF_NPZ_POSES_KEY}' and "
            f"'{CONF_NPZ_POSITIONS_KEY}'/'{CONF_NPZ_TRIADS_KEY}'. Provide only one representation."
        )
    if has_poses:
        return _select_poses(_npz_member(npz, CONF_NPZ_POSES_KEY, name), frame, name)
    if not has_positions and not has_triads:
        raise ConfigurationLoadError(
            f"{name} contains none of the expected entries ('{CONF_NPZ_POSES_KEY}', or "
            f"'{CONF_NPZ_POSITIONS_KEY}' and '{CONF_NPZ_TRIADS_KEY}'). Found: {sorted(keys)}."
        )
    if has_positions != has_triads:
        present, missing = (
            (CONF_NPZ_POSITIONS_KEY, CONF_NPZ_TRIADS_KEY) if has_positions
            else (CONF_NPZ_TRIADS_KEY, CONF_NPZ_POSITIONS_KEY)
        )
        raise ConfigurationLoadError(
            f"{name} contains '{present}' but not '{missing}'. Both are required."
        )

    positions = _npz_member(npz, CONF_NPZ_POSITIONS_KEY, name)
    triads = _npz_member(npz, CONF_NPZ_TRIADS_KEY, name)
    _check_numeric(positions, f"{name} ('{CONF_NPZ_POSITIONS_KEY}')")
    _check_numeric(triads, f"{name} ('{CONF_NPZ_TRIADS_KEY}')")
    if (
        positions.ndim not in (2, 3)
        or positions.shape[-1] != 3
        or triads.shape[-2:] != (3, 3)
        or triads.shape[:-2] != positions.shape[:-1]
    ):
        raise ConfigurationLoadError(
            f"{name} has '{CONF_NPZ_POSITIONS_KEY}' of shape {positions.shape} and "
            f"'{CONF_NPZ_TRIADS_KEY}' of shape {triads.shape}. Expected (N, 3) and (N, 3, 3) for "
            f"a single configuration or (T, N, 3) and (T, N, 3, 3) for a trajectory."
        )

    sel_positions = _select_frame(positions, frame, 2, name)
    sel_triads = _select_frame(triads, frame, 3, name)
    poses = np.zeros((len(sel_positions), 4, 4), dtype=np.float64)
    poses[:, :3, :3] = sel_triads
    poses[:, :3, 3] = sel_positions
    poses[:, 3, 3] = 1.0
    return poses


###################################################################################################
# Units and validity

def positions_to_nm(poses: np.ndarray, conf_units: str = 'nm', unit_length: float = 1.0) -> np.ndarray:
    """
    Return a copy of the poses with the positions converted to nm.

    Parameters
    ----------
    poses : np.ndarray
        Poses with shape (N, 4, 4).
    conf_units : str, optional
        Length unit of the positions: 'nm' (default), 'angstrom', or 'sim' for simulation units,
        i.e. multiples of unit_length.
    unit_length : float, optional
        Unit length in nm. Only used for conf_units='sim'.

    Returns
    -------
    np.ndarray
        Poses with positions in nm.
    """
    if conf_units not in CONF_UNITS:
        raise ValueError(f"Unknown length unit '{conf_units}'. Expected one of: {', '.join(CONF_UNITS)}.")
    factor = unit_length if conf_units == 'sim' else CONF_UNITS_TO_NM[conf_units]
    out = np.array(poses, dtype=np.float64, copy=True)
    if factor != 1.0:
        out[:, :3, 3] *= factor
    return out


def validate_poses(
    poses: np.ndarray,
    ortho_tol: float = CONF_ORTHO_TOL,
    force_orthogonalize: bool = False,
    name: str = 'The configuration',
) -> tuple[np.ndarray, str | None]:
    """
    Check that the poses are SE(3) elements and repair small deviations of the rotations.

    Rotation blocks that deviate from orthonormality (largest absolute entry of R^T R - I) by up
    to ortho_tol are replaced by the closest rotation matrix. Larger deviations raise an error
    unless force_orthogonalize is set. Mirrored (determinant -1) and degenerate rotation blocks
    are always rejected, as is a last row other than [0, 0, 0, 1].

    Parameters
    ----------
    poses : np.ndarray
        Poses with shape (N, 4, 4).
    ortho_tol : float, optional
        Largest deviation from orthonormality that is repaired automatically.
    force_orthogonalize : bool, optional
        Repair rotation blocks deviating by more than ortho_tol as well. Default is False.
    name : str, optional
        Name of the configuration used in messages.

    Returns
    -------
    tuple[np.ndarray, str | None]
        Validated copy of the poses, and a note describing the repair (None if no rotation
        block was changed).

    Raises
    ------
    ConfigurationValidationError
        If a pose cannot be interpreted as an SE(3) element or the deviation from
        orthonormality exceeds ortho_tol without force_orthogonalize.
    """
    poses = np.array(poses, dtype=np.float64, copy=True)

    last_row_dev = np.max(np.abs(poses[:, 3, :] - np.array([0.0, 0.0, 0.0, 1.0])), axis=1)
    bad = np.flatnonzero(last_row_dev > ortho_tol)
    if len(bad) > 0:
        msg = (
            f'{name} contains poses whose last row differs from [0, 0, 0, 1] '
            f'(0-based indices: {_listed(bad)}).'
        )
        if np.allclose(poses[:, :3, 3], 0.0) and np.any(np.abs(poses[:, 3, :3]) > ortho_tol):
            msg += (
                ' The positions appear to be stored in the last row, i.e. the matrices are '
                'transposed. Positions have to be stored in the last column.'
            )
        raise ConfigurationValidationError(msg)
    poses[:, 3, :] = (0.0, 0.0, 0.0, 1.0)

    rot = poses[:, :3, :3]
    u, s, vt = np.linalg.svd(rot)
    degenerate = np.flatnonzero(s[:, -1] < CONF_DEGENERATE_TOL)
    if len(degenerate) > 0:
        raise ConfigurationValidationError(
            f'{name} contains degenerate rotation blocks that cannot be repaired '
            f'(0-based indices: {_listed(degenerate)}).'
        )
    improper = np.flatnonzero(np.linalg.det(rot) < 0)
    if len(improper) > 0:
        raise ConfigurationValidationError(
            f'{name} contains improper rotation blocks (determinant -1, i.e. mirrored triads) that '
            f'cannot be repaired (0-based indices: {_listed(improper)}). Check the handedness of '
            f'the triads.'
        )

    dev = np.max(np.abs(np.einsum('nji,njk->nik', rot, rot) - np.eye(3)), axis=(1, 2))
    to_fix = np.flatnonzero(dev > CONF_ORTHO_EXACT)
    if len(to_fix) == 0:
        return poses, None
    max_dev = dev.max()
    if max_dev > ortho_tol and not force_orthogonalize:
        above = np.flatnonzero(dev > ortho_tol)
        raise ConfigurationValidationError(
            f'{name} contains {len(above)} rotation block(s) deviating from orthonormality by more '
            f'than {ortho_tol:g} (largest deviation {max_dev:.2e}; 0-based indices: '
            f'{_listed(above)}). Use --force_orthogonalize (force_orthogonalize=True) to replace '
            f'them by the closest rotation matrices anyway.'
        )
    # det(rot) > 0, hence u @ vt is a proper rotation
    poses[to_fix, :3, :3] = u[to_fix] @ vt[to_fix]
    note = (
        f'{name}: re-orthonormalized {len(to_fix)} rotation block(s) '
        f'(largest deviation {max_dev:.2e}'
    )
    note += ', above the tolerance; forced).' if max_dev > ortho_tol else ').'
    return poses, note


def remove_repeated_pose(poses: np.ndarray, name: str = 'The configuration') -> tuple[np.ndarray, str | None]:
    """
    Remove a last pose that repeats the first one (closed configurations).

    A closed configuration lists every pose once, the bond closing the ring connects the last
    pose to the first one. A last pose identical to the first one (positions within
    CONF_DUPLICATE_TOL nm and rotation entries within CONF_DUPLICATE_TOL) is removed. A last
    pose that nearly coincides with the first one (closer than CONF_NEAR_DUPLICATE_FRACTION of
    the median bond length) is rejected.

    Parameters
    ----------
    poses : np.ndarray
        Poses with shape (N, 4, 4) and positions in nm.
    name : str, optional
        Name of the configuration used in messages.

    Returns
    -------
    tuple[np.ndarray, str | None]
        Poses without the repeated pose, and a note if a pose was removed (None otherwise).

    Raises
    ------
    ConfigurationValidationError
        If the last pose nearly coincides with the first one.
    """
    if len(poses) < 3:
        return poses, None
    pos = poses[:, :3, 3]
    dist = np.linalg.norm(pos[-1] - pos[0])
    rot_dev = np.max(np.abs(poses[-1, :3, :3] - poses[0, :3, :3]))
    if dist <= CONF_DUPLICATE_TOL and rot_dev <= CONF_DUPLICATE_TOL:
        return poses[:-1].copy(), (
            f'{name} repeats its first pose at the end. The repeated pose was removed, '
            f'{len(poses) - 1} poses remain.'
        )
    median_bond = np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1))
    if dist < CONF_NEAR_DUPLICATE_FRACTION * median_bond:
        raise ConfigurationValidationError(
            f'The last pose of {_lc(name)} nearly coincides with the first one (distance '
            f'{dist:.3g} nm, median bond length {median_bond:.3g} nm). A closed configuration has '
            f'to list every pose once: remove the repeated last pose.'
        )
    return poses, None


###################################################################################################
# Matching configuration and sequence

def expected_num_beads(
    seq_len: int,
    composite_size: int,
    closed: bool = False,
    center_offset: int = 0,
) -> int:
    """
    Number of beads (coarse-grained triads) generated for a sequence.

    Bead k is base pair k*composite_size + center_offset. Open chains are covered up to their
    last complete composite step, base pairs beyond are cropped. Closed chains require seq_len to
    be a multiple of composite_size.

    Parameters
    ----------
    seq_len : int
        Number of base pairs.
    composite_size : int
        Number of base pairs per bead.
    closed : bool, optional
        Closed (circular) topology. Default is False.
    center_offset : int, optional
        Base pair of the first bead (composite_size//2 for centered beads, 0 otherwise).
    """
    if closed:
        return seq_len // composite_size
    return (seq_len - 1 - center_offset) // composite_size + 1


def num_cropped_bp(
    seq_len: int,
    composite_size: int,
    closed: bool = False,
    center_offset: int = 0,
) -> int:
    """Number of base pairs at the end of an open sequence that do not complete a composite step."""
    if closed:
        return 0
    return (seq_len - 1 - center_offset) % composite_size


def truncated_seq_len(
    num_beads: int,
    composite_size: int,
    center_offset: int = 0,
    allow_crop: bool = True,
) -> int:
    """
    Length to which an open sequence is truncated such that it yields num_beads beads.

    With cropping allowed, the block of the last bead is kept complete (num_beads*composite_size
    base pairs), which keeps centered beads valid. Without cropping the sequence ends on the base
    pair of the last bead.
    """
    if allow_crop:
        return num_beads * composite_size
    return (num_beads - 1) * composite_size + center_offset + 1


@dataclass
class MatchResult:
    """
    Result of matching an external configuration to a sequence.

    Attributes
    ----------
    sequence : str
        The sequence, truncated at its end if required.
    bead_poses : np.ndarray
        Poses of the beads, shape (num_beads, 4, 4).
    resolution : str
        Resolution of the configuration: 'cg' (one pose per bead) or 'bp' (one pose per base pair).
    bp_poses : np.ndarray or None
        Base-pair poses covering the full (truncated) sequence. Only available for base-pair
        resolution, and only if the configuration covers every base pair.
    notes : list of str
        Descriptions of the performed truncations.
    """
    sequence: str
    bead_poses: np.ndarray
    resolution: str
    bp_poses: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)


def match_sequence_and_poses(
    seq: str,
    poses: np.ndarray,
    composite_size: int,
    closed: bool = False,
    center_offset: int = 0,
    allow_crop: bool = True,
    resolution: str = 'auto',
    truncate: bool = False,
    uncropped_seq_len: int | None = None,
    name: str = 'The configuration',
) -> MatchResult:
    """
    Match the poses of an external configuration to the beads generated for a sequence.

    The configuration is either at bead resolution (one pose per bead) or at base-pair
    resolution (one pose per base pair). The latter is coarse-grained by retaining the frames of
    the beads, i.e. base pairs k*composite_size + center_offset, exactly as the elastic
    parameters are coarse-grained. With resolution='auto' the resolution is deduced from the
    number of poses, which is unambiguous for exact matches.

    If the number of poses does not match, an error is raised, unless truncate is set (open
    chains only). Then the longer of the two is truncated at its end: either the configuration is
    cut to the poses required by the sequence, or the sequence is cut to the beads covered by the
    configuration (see truncated_seq_len).

    Parameters
    ----------
    seq : str
        Sequence (after any selection of a subsequence).
    poses : np.ndarray
        Poses with shape (N, 4, 4).
    composite_size : int
        Number of base pairs per bead.
    closed : bool, optional
        Closed (circular) topology. Default is False.
    center_offset : int, optional
        Base pair of the first bead (composite_size//2 for centered beads, 0 otherwise).
    allow_crop : bool, optional
        Whether base pairs beyond the last complete composite step may be cropped. Determines the
        length to which the sequence is truncated. Default is True.
    resolution : str, optional
        'auto' (default), 'bp' or 'cg'.
    truncate : bool, optional
        Truncate the longer of sequence and configuration to match the shorter one. Only for open
        chains. Default is False.
    uncropped_seq_len : int, optional
        Length of the sequence before a subsequence was selected. Only used for hints in error
        messages.
    name : str, optional
        Name of the configuration used in messages.

    Returns
    -------
    MatchResult

    Raises
    ------
    ConfigurationMismatchError
        If the number of poses does not match and truncate is not set, if truncate is set for a
        closed topology, or if the resolution cannot be deduced.
    """
    if resolution not in CONF_RESOLUTIONS:
        raise ValueError(f"Unknown resolution '{resolution}'. Expected one of: {', '.join(CONF_RESOLUTIONS)}.")
    cg = composite_size
    seq_len = len(seq)
    num_poses = len(poses)
    if closed and truncate:
        raise ConfigurationMismatchError(
            'Truncation (-trunc/--truncate_to_match) is not available for closed topologies. '
            'Closed configurations have to match the sequence exactly.'
        )
    if closed and seq_len % cg != 0:
        raise ValueError(
            f'For closed topology the length of the sequence ({seq_len}) must be a multiple of the '
            f'composite size ({cg}).'
        )
    num_beads = expected_num_beads(seq_len, cg, closed, center_offset)

    def mismatch(res: str) -> ConfigurationMismatchError:
        return _mismatch_error(
            name, num_poses, seq_len, num_beads, cg, closed, center_offset, res, uncropped_seq_len
        )

    if cg == 1:
        res = 'cg'
    elif resolution != 'auto':
        res = resolution
    elif num_poses == num_beads:
        res = 'cg'
    elif num_poses == seq_len:
        res = 'bp'
    elif truncate:
        raise ConfigurationMismatchError(
            f'{name} contains {num_poses} poses, which matches neither the {num_beads} beads (bead '
            f'resolution) nor the {seq_len} base pairs (base-pair resolution) of the sequence. With '
            f'-trunc/--truncate_to_match the resolution cannot be deduced from the number of poses. '
            f'Specify it with --conf_resolution bp or --conf_resolution cg.'
        )
    else:
        raise mismatch('auto')

    notes = []
    expected = num_beads if res == 'cg' else seq_len
    if num_poses != expected:
        if not truncate:
            raise mismatch(res)
        if num_poses > expected:
            poses = poses[:expected]
            notes.append(
                f'Truncated the configuration from {num_poses} to {expected} poses to match the sequence.'
            )
        else:
            if res == 'cg':
                beads = num_poses
            else:
                if num_poses <= center_offset:
                    raise ConfigurationMismatchError(
                        f'{name} contains {num_poses} base-pair poses, which do not reach the first '
                        f'bead at base pair {center_offset}.'
                    )
                beads = -(-(num_poses - center_offset) // cg)
            new_len = min(seq_len, truncated_seq_len(beads, cg, center_offset, allow_crop))
            if new_len < seq_len:
                seq = seq[:new_len]
                notes.append(
                    f'Truncated the sequence from {seq_len} to {new_len} bp to match the {beads} '
                    f'beads covered by the configuration.'
                )

    final_len = len(seq)
    final_beads = expected_num_beads(final_len, cg, closed, center_offset)
    bp_poses = None
    if res == 'cg':
        bead_poses = poses
    else:
        bead_poses = poses[center_offset::cg][:final_beads]
        if len(poses) >= final_len:
            bp_poses = poses[:final_len]
        else:
            notes.append(
                f'The configuration has no poses for the last {final_len - len(poses)} base pair(s) '
                f'of the sequence, which lie beyond the last bead.'
            )
    if len(bead_poses) != final_beads:
        raise RuntimeError(
            f'Internal error: {len(bead_poses)} bead poses for {final_beads} beads after matching.'
        )
    if final_beads < 2:
        raise ConfigurationMismatchError(
            f'{name} yields {final_beads} bead(s). At least two beads are required.'
        )
    return MatchResult(
        sequence=seq,
        bead_poses=np.array(bead_poses, copy=True),
        resolution=res,
        bp_poses=None if bp_poses is None else np.array(bp_poses, copy=True),
        notes=notes,
    )


def _mismatch_error(
    name: str,
    num_poses: int,
    seq_len: int,
    num_beads: int,
    composite_size: int,
    closed: bool,
    center_offset: int,
    resolution: str,
    uncropped_seq_len: int | None,
) -> ConfigurationMismatchError:
    """Error stating the expected and the provided number of poses."""
    if composite_size == 1:
        expected = f'{num_beads} poses (one per base pair)'
    elif resolution == 'cg':
        expected = f'{num_beads} poses at bead resolution'
    elif resolution == 'bp':
        expected = f'{seq_len} poses at base-pair resolution'
    else:
        expected = f'{num_beads} poses at bead resolution or {seq_len} poses at base-pair resolution'
    centered = ', centered' if center_offset else ''
    topology = 'closed' if closed else 'open'
    msg = (
        f'Configuration mismatch: {_lc(name)} contains {num_poses} poses, but the sequence '
        f'({seq_len} bp, composite size {composite_size}{centered}, {topology}) requires {expected}.'
    )
    if closed:
        msg += (
            ' Closed configurations have to match the sequence exactly; truncation is not '
            'available for closed topologies.'
        )
    else:
        msg += (
            ' The mismatch can be resolved with -trunc/--truncate_to_match, which truncates the '
            'longer of the two (sequence or configuration) at its end to match the shorter one.'
        )
    if uncropped_seq_len is not None and uncropped_seq_len != seq_len:
        uncropped_beads = expected_num_beads(uncropped_seq_len, composite_size, closed, center_offset)
        if num_poses in (uncropped_seq_len, uncropped_beads):
            msg += (
                f' Note: the configuration matches the full sequence of {uncropped_seq_len} bp '
                f'before -sid/-eid were applied. It has to correspond to the selected part of the '
                f'sequence.'
            )
    return ConfigurationMismatchError(msg)


###################################################################################################
# Output

def write_poses(filename: str | Path, poses: np.ndarray, input_file: str | Path | None = None) -> str:
    """
    Save poses to a .npy file without destroying the input configuration.

    If filename is the input configuration file, the file is left untouched if it already holds
    the given poses. Otherwise the original is moved aside (backup name with suffix _#N) before
    the poses are written.

    Parameters
    ----------
    filename : str or Path
        Output file (.npy).
    poses : np.ndarray
        Poses to save.
    input_file : str or Path, optional
        Path of the configuration file that was imported.

    Returns
    -------
    str
        Description of what was written.
    """
    path = Path(filename)
    is_input = (
        input_file is not None
        and path.exists()
        and Path(input_file).exists()
        and path.resolve() == Path(input_file).resolve()
    )
    if is_input:
        existing = np.load(path, allow_pickle=False)
        if existing.shape == poses.shape and np.array_equal(existing, poses):
            return f'{path} is the input configuration and already holds these poses; left unchanged.'
        backup = backup_filename(path)
        path.rename(backup)
        np.save(path, poses)
        return f'{path} was the input configuration. It was moved to {backup} and replaced.'
    np.save(path, poses)
    return f'Configuration written to {path}.'
