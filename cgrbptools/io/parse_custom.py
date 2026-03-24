from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence
import atexit
import os
import signal
import sys
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from ..SO3 import so3
from ..evals.se3 import poses2junctions, junctions2parameters, junctions2dynamics
from .console_output import print_progress


# ---------------------------------------------------------------------------
# Module-level temp-file registry
# ---------------------------------------------------------------------------
# Tracks paths of all active memmap backing files so they can be removed on
# interpreter exit or on OS signals, even if the memmap was not yet GC'd.

_active_tmp_files: set[str] = set()


def _cleanup_tmp_files() -> None:
    """Delete every registered temp file that still exists."""
    for p in list(_active_tmp_files):
        try:
            os.unlink(p)
        except OSError:
            pass
        _active_tmp_files.discard(p)


atexit.register(_cleanup_tmp_files)

# SIGTERM does not trigger atexit by default – bridge it.
_prev_sigterm = signal.getsignal(signal.SIGTERM)


def _sigterm_handler(signum: int, frame) -> None:  # type: ignore[type-arg]
    _cleanup_tmp_files()
    if callable(_prev_sigterm):
        _prev_sigterm(signum, frame)
    sys.exit(128 + signum)


try:
    signal.signal(signal.SIGTERM, _sigterm_handler)
except OSError:
    pass  # may fail in some restricted environments


class _FileCleanupGuard:
    """
    Deletes a memmap backing file when garbage-collected.

    Stored as ``_guard`` on every ``_ManagedMemmap`` instance (and its views)
    so its lifetime is automatically tied to the data.  The atexit / SIGTERM
    handlers serve as a backstop in case GC does not run before shutdown.
    """

    __slots__ = ('_path',)

    def __init__(self, path: str) -> None:
        self._path = path

    def __del__(self) -> None:
        # Guard against module-level globals already being None during
        # interpreter shutdown.
        path = self._path
        try:
            reg = _active_tmp_files
            if reg is not None:
                reg.discard(path)
        except Exception:
            pass
        try:
            _os = os
            if _os is not None:
                _os.unlink(path)
        except Exception:
            pass


class _ManagedMemmap(np.memmap):
    """
    A ``np.memmap`` subclass that owns its backing file via a
    ``_FileCleanupGuard`` stored as ``_guard``.

    The guard is propagated to every view / slice created from this array
    via ``__array_finalize__``, so the file stays alive as long as any
    derived array exists, and is deleted automatically when all references
    are gone.  Callers never need to handle the guard explicitly.
    """

    def __new__(
        cls,
        path: str,
        dtype: type = np.float64,
        mode: str = 'w+',
        shape: tuple | None = None,
        guard: _FileCleanupGuard | None = None,
    ) -> '_ManagedMemmap':
        obj = super().__new__(cls, path, dtype=dtype, mode=mode, shape=shape)
        obj._guard = guard
        return obj

    def __array_finalize__(self, obj: object) -> None:
        # Called whenever a new array is created from an existing one
        # (view, slice, ufunc output …).  Propagate the guard so the
        # file outlives all derived arrays.
        self._guard = getattr(obj, '_guard', None)


def _available_memory_bytes() -> int:
    """
    Return the number of bytes of available (free + reclaimable) RAM.

    Reads ``MemAvailable`` from ``/proc/meminfo`` (Linux).  Returns 0 if
    the file cannot be read, which will always trigger the memmap path
    (safe default).
    """
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def _read_box(box_type_str: str, f: Any) -> np.ndarray:
    """Read 3 box-bound lines from *f* and return a flat (9,) array."""
    rows = [list(map(float, f.readline().split())) for _ in range(3)]
    box = np.zeros(9, dtype=float)
    if "xy xz yz" in box_type_str:  # triclinic
        for i, row in enumerate(rows):
            box[i * 2], box[i * 2 + 1], box[6 + i] = row
    else:                            # orthogonal
        for i, row in enumerate(rows):
            box[i * 2], box[i * 2 + 1] = row
    return box


def _scan_custom(filename: str):
    """
    Fast first-pass scan of a LAMMPS custom dump file.

    Reads only the header lines of each frame (skipping atom data) to collect
    metadata without loading atom coordinates into memory.

    Returns
    -------
    n_frames : int
    n_atoms : int
    n_cols : int
    col_names : list[str]
    box_data : np.ndarray  shape (9,) – from the first timestep
    """
    with open(filename, 'r') as f:
        # ------------------------------------------------------------------ #
        # Read the first frame to capture metadata                           #
        # ------------------------------------------------------------------ #
        if not f.readline():          # ITEM: TIMESTEP
            return 0, 0, 0, [], None
        f.readline()                  # timestep value

        line = f.readline()           # ITEM: NUMBER OF ATOMS
        if not line.startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError(f"Expected 'ITEM: NUMBER OF ATOMS', but got: {line.strip()}")
        n_atoms = int(f.readline())

        line = f.readline()           # ITEM: BOX BOUNDS
        if not line.startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"Expected 'ITEM: BOX BOUNDS', but got: {line.strip()}")
        box_data = _read_box(line.strip().split(' ', 3)[3], f)

        line = f.readline()           # ITEM: ATOMS
        if not line.startswith("ITEM: ATOMS"):
            raise ValueError(f"Expected 'ITEM: ATOMS', but got: {line.strip()}")
        col_names = line.strip().split()[2:]
        n_cols = len(col_names)

        for _ in range(n_atoms):      # skip atom data
            f.readline()

        n_frames = 1

        # ------------------------------------------------------------------ #
        # Remaining frames – validate consistency, skip atom data            #
        # ------------------------------------------------------------------ #
        while True:
            line = f.readline()       # ITEM: TIMESTEP
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                raise ValueError(f"Expected 'ITEM: TIMESTEP', but got: {line.strip()}")
            f.readline()              # timestep value

            line = f.readline()       # ITEM: NUMBER OF ATOMS
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"Expected 'ITEM: NUMBER OF ATOMS', but got: {line.strip()}")
            cur_atoms = int(f.readline())
            if cur_atoms != n_atoms:
                raise ValueError(
                    f"Number of atoms changed from {n_atoms} to {cur_atoms} at frame {n_frames}."
                )

            line = f.readline()       # ITEM: BOX BOUNDS (3 data lines)
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Expected 'ITEM: BOX BOUNDS', but got: {line.strip()}")
            f.readline(); f.readline(); f.readline()

            line = f.readline()       # ITEM: ATOMS
            if not line.startswith("ITEM: ATOMS"):
                raise ValueError(f"Expected 'ITEM: ATOMS', but got: {line.strip()}")
            cur_cols = line.strip().split()[2:]
            if cur_cols != col_names:
                raise ValueError(f"Column layout changed at frame {n_frames}.")

            for _ in range(n_atoms):  # skip atom data
                f.readline()

            n_frames += 1

    return n_frames, n_atoms, n_cols, col_names, box_data


def parse_custom(
    filename: str,
    start: int = 0,
    last: int = -1,
    stride: int = 1,
    verbose: bool = False,
) -> Dict:
    """
    Parses a LAMMPS custom dump file and returns data in a dictionary format.

    This function expects the number of atoms and the number of dumped arguments
    to be constant across all timesteps in the dump file.

    Frame selection
    ---------------
    ``start``, ``last``, and ``stride`` work on the **0-based snapshot index**
    as it appears in the file (independent of the LAMMPS timestep value).

    The selected indices form the sequence::

        start, start + stride, start + 2*stride, ...

    up to and including ``last`` (or the last snapshot in the file when
    ``last == -1`` or ``last`` exceeds the total frame count).

    Example: ``start=3, stride=10`` reads snapshots 3, 13, 23, …

    Memory efficiency
    -----------------
    The function uses a two-pass strategy to avoid peak-memory doubling:

    1. A fast first pass reads *only* header lines to determine the total
       frame count, atom count, and column layout.
    2. A second pass fills a pre-allocated ``np.memmap`` array on disk,
       so atom data never accumulates in RAM as a Python list.  Skipped
       frames are advanced with bare ``readline()`` calls.

    Temp file
    ---------
    If the required array would exceed **1/4 of available RAM**, the data are
    stored in a disk-backed ``_ManagedMemmap``; otherwise a plain pre-allocated
    ``np.ndarray`` is used and no file is written.

    When a temp file is created it is placed next to the input file and named
    ``<stem>.custom_read_tmp``.  It is deleted automatically when:

    * the returned ``data['data']`` array is garbage-collected — a
        ``_FileCleanupGuard`` embedded in the ``_ManagedMemmap`` calls
        ``os.unlink`` from its ``__del__`` method when the last reference
        is dropped;
    * the Python interpreter exits normally (``atexit`` handler);
    * the process receives ``SIGTERM`` (server shutdown, ``kill``, etc.);
    * the user presses Ctrl-C (``SIGINT`` → ``KeyboardInterrupt`` → ``atexit``).

    Args:
        filename (str): The filename of the LAMMPS custom dump file.
        start (int): 0-based index of the first snapshot to read. Snapshots
            before this index are skipped entirely. Default: 0.
        last (int): 0-based index of the last snapshot to read (inclusive).
            If -1, or larger than the total number of snapshots, all snapshots
            from ``start`` onward are read (subject to ``stride``). Default: -1.
        stride (int): Read every ``stride``-th snapshot within the selected
            range. Default: 1 (read every snapshot).

    Returns:
        dict: A dictionary containing the parsed data:
              - 'box' (np.ndarray): A 2D NumPy array of shape (3, 3)
                representing the box dimensions from the **FIRST** timestep.
              - 'timesteps' (np.ndarray): A 1D NumPy array containing the
                timestep numbers of the *selected* snapshots (length N_sel).
              - 'args' (list): A list of strings with the names of the dumped
                attributes (e.g., ['id', 'type', 'mol', 'x', 'y', 'z',
                'ix', 'iy', 'iz']).
              - 'data' (np.ndarray | np.memmap): A 3D array of shape
                (N_sel, A, C), where N_sel = number of selected snapshots,
                A = number of atoms per snapshot, C = number of columns.
                A plain ``np.ndarray`` is used when the data fits in 1/4 of
                available RAM; a disk-backed ``_ManagedMemmap`` is used
                otherwise.
              - '_memmap_path' (str): Path of the backing ``.custom_read_tmp``
                file.  Present only when a memmap was used.

    Raises:
        FileNotFoundError: If the specified filename does not exist.
        ValueError: If the dump file format is unexpected, or if the number
                    of atoms or columns changes between timesteps, or if
                    ``stride`` < 1 or ``start`` < 0.
        RuntimeError: For other unexpected parsing errors.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}.")

    try:
        # ------------------------------------------------------------------ #
        # Pass 1 – fast metadata scan (no atom data held in RAM)             #
        # ------------------------------------------------------------------ #
        if verbose:
            print("Scanning file for metadata and frame count...")
        n_frames, n_atoms, n_cols, col_names, box_raw = _scan_custom(filename)

        if n_frames == 0:
            return {
                'box': np.array([]),
                'timesteps': np.array([]),
                'args': [],
                'data': np.array([]).reshape(0, 0, 0),
            }

        # ------------------------------------------------------------------ #
        # Resolve frame selection                                            #
        # ------------------------------------------------------------------ #
        effective_last = (n_frames - 1) if (last < 0 or last >= n_frames) else last
        # Selected 0-based frame indices (ordered)
        selected_range = range(start, effective_last + 1, stride)
        n_selected = len(selected_range)

        if n_selected == 0:
            return {
                'box': np.array([]),
                'timesteps': np.array([]),
                'args': col_names,
                'data': np.array([]).reshape(0, 0, 0),
            }

        # ------------------------------------------------------------------ #
        # Allocate output arrays                                             #
        # ------------------------------------------------------------------ #
        timesteps_array = np.empty(n_selected, dtype=np.int64)

        # Decide between in-RAM pre-allocation and a disk-backed memmap.
        # Use memmap when the required array would exceed 1/4 of the
        # currently available RAM; use np.empty otherwise.
        required_bytes = n_selected * n_atoms * n_cols * np.dtype(np.float64).itemsize
        use_memmap = required_bytes > _available_memory_bytes() // 4

        tmp_path: str | None = None
        if use_memmap:
            tmp_path = str(
                Path(filename).parent / (Path(filename).stem + '.custom_read_tmp')
            )
            _active_tmp_files.add(tmp_path)
            guard = _FileCleanupGuard(tmp_path)
            data_matrix: np.ndarray = _ManagedMemmap(
                tmp_path, dtype=np.float64, mode='w+',
                shape=(n_selected, n_atoms, n_cols),
                guard=guard,
            )
        else:
            data_matrix = np.empty((n_selected, n_atoms, n_cols), dtype=np.float64)

        # ------------------------------------------------------------------ #
        # Pass 2 – fill data_matrix, skipping unselected frames             #
        # Use an iterator over the selected range – O(1) auxiliary memory.  #
        # ------------------------------------------------------------------ #
        sel_iter = iter(selected_range)
        next_sel: int | None = next(sel_iter, None)
        out_idx = 0

        with open(filename, 'r') as f:
            for raw_idx in range(effective_last + 1):
                # ITEM: TIMESTEP
                line = f.readline()
                if not line:
                    break
                if not line.startswith("ITEM: TIMESTEP"):
                    raise ValueError(f"Expected 'ITEM: TIMESTEP', but got: {line.strip()}")
                ts_value = int(f.readline())

                # Skip: ITEM: NUMBER OF ATOMS, n_atoms,
                #        ITEM: BOX BOUNDS, 3 box lines,
                #        ITEM: ATOMS header  → 7 lines total
                for _ in range(7):
                    f.readline()

                if raw_idx == next_sel:
                    timesteps_array[out_idx] = ts_value
                    frame_view = data_matrix[out_idx]
                    for i in range(n_atoms):
                        atom_line_str = f.readline()
                        if not atom_line_str:
                            raise ValueError("Unexpected end of file while reading atom data.")
                        frame_view[i] = atom_line_str.split()
                    out_idx += 1
                    if verbose and (out_idx % max(1, n_selected // 500) == 0 or out_idx == n_selected):
                        print_progress(out_idx, n_selected, prefix='Reading frames')
                    next_sel = next(sel_iter, None)
                else:
                    for _ in range(n_atoms):
                        f.readline()

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {filename}") from None
    except Exception as e:
        raise RuntimeError(f"An error occurred while parsing the dump file: {e}") from e

    box = np.zeros((3, 3))
    box[:, 0] = box_raw[0:6:2]
    box[:, 1] = box_raw[1:6:2]
    box[:, 2] = box_raw[6:]

    result: Dict[str, Any] = {
        'box': box,
        'timesteps': timesteps_array,
        'args': col_names,
        'data': data_matrix,
    }
    if tmp_path is not None:
        result['_memmap_path'] = tmp_path
    return result


def parse_custom_old(filename: str) -> Dict:
    """
    Parses a LAMMPS custom dump file and returns data in a dictionary format.

    This function expects the number of atoms and the number of dumped arguments
    to be constant across all timesteps in the dump file.

    Args:
        filename (str): The filename of the LAMMPS custom dump file.

    Returns:
        dict: A dictionary containing the parsed data:
              - 'box' (np.ndarray): A 1D NumPy array of 9 floats
                [xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz]
                representing the box dimensions from the **FIRST** timestep.
              - 'timesteps' (np.ndarray): A 1D NumPy array containing the timestep numbers (length N).
              - 'args' (list): A list of strings with the names of the dumped attributes
                (e.g., ['id', 'type', 'mol', 'x', 'y', 'z', 'ix', 'iy', 'iz']).
              - 'data' (np.ndarray): A 3D NumPy array of shape (N, A, C), where:
                N = number of timesteps
                A = number of atoms per timestep
                C = number of arguments (columns) per atom

    Raises:
        FileNotFoundError: If the specified filename does not exist.
        ValueError: If the dump file format is unexpected, or if the number of atoms
                    or columns changes between timesteps.
        RuntimeError: For other unexpected parsing errors.
    """
    all_timesteps = []
    all_atom_data_2d_frames = []
    
    initial_num_atoms = -1
    initial_num_cols_per_atom = -1
    dumped_column_names = None
    initial_box_data = None 

    try:
        with open(filename, 'r') as f:
            while True:
                # ITEM: TIMESTEP
                line = f.readline()
                if not line: # End of file
                    break # No more frames to read
                if not line.startswith("ITEM: TIMESTEP"):
                    raise ValueError(f"Expected 'ITEM: TIMESTEP', but got: {line.strip()}")
                timestep = int(f.readline().strip())

                # ITEM: NUMBER OF ATOMS
                line = f.readline()
                if not line.startswith("ITEM: NUMBER OF ATOMS"):
                    raise ValueError(f"Expected 'ITEM: NUMBER OF ATOMS', but got: {line.strip()}")
                num_atoms = int(f.readline().strip())

                # ITEM: BOX BOUNDS
                line = f.readline()
                if not line.startswith("ITEM: BOX BOUNDS"):
                    raise ValueError(f"Expected 'ITEM: BOX BOUNDS', but got: {line.strip()}")
                box_type_str = line.strip().split(' ', 3)[3]

                x_line = list(map(float, f.readline().strip().split()))
                y_line = list(map(float, f.readline().strip().split()))
                z_line = list(map(float, f.readline().strip().split()))

                current_box_array = np.zeros(9, dtype=float)
                if "xy xz yz" in box_type_str: # Triclinic box
                    if not (len(x_line) == 3 and len(y_line) == 3 and len(z_line) == 3):
                        raise ValueError("Triclinic box bounds expected 3 values per line (lo, hi, tilt)")
                    current_box_array[0], current_box_array[1], current_box_array[6] = x_line
                    current_box_array[2], current_box_array[3], current_box_array[7] = y_line
                    current_box_array[4], current_box_array[5], current_box_array[8] = z_line
                else: 
                    if not (len(x_line) == 2 and len(y_line) == 2 and len(z_line) == 2):
                        raise ValueError("Orthogonal box bounds expected 2 values per line (lo, hi)")
                    current_box_array[0], current_box_array[1] = x_line
                    current_box_array[2], current_box_array[3] = y_line
                    current_box_array[4], current_box_array[5] = z_line

                line = f.readline()
                if not line.startswith("ITEM: ATOMS"):
                    raise ValueError(f"Expected 'ITEM: ATOMS', but got: {line.strip()}")
                column_names = line.strip().split(' ')[2:]

                if initial_num_atoms == -1:
                    initial_num_atoms = num_atoms
                    initial_num_cols_per_atom = len(column_names)
                    dumped_column_names = column_names
                    initial_box_data = current_box_array 
                else:
                    if num_atoms != initial_num_atoms:
                        raise ValueError(
                            f"Number of atoms changed from {initial_num_atoms} to {num_atoms} "
                            f"at timestep {timestep}. Cannot form a consistent (N, A, C) matrix."
                        )
                    if len(column_names) != initial_num_cols_per_atom:
                        raise ValueError(
                            f"Number of columns changed from {initial_num_cols_per_atom} to {len(column_names)} "
                            f"at timestep {timestep}. This is unexpected for 'dump custom'."
                        )
                    if column_names != dumped_column_names:
                        raise ValueError(
                            f"Column names changed at timestep {timestep}. This is unexpected for 'dump custom'."
                        )

                # Read atom data for the current frame
                atom_data_current_frame = np.empty((num_atoms, initial_num_cols_per_atom), dtype=float)
                for i in range(num_atoms):
                    atom_line_str = f.readline().strip()
                    if not atom_line_str:
                        raise ValueError("Unexpected end of file while reading atom data.")
                    atom_data_current_frame[i, :] = list(map(float, atom_line_str.split()))
                
                all_timesteps.append(timestep)
                all_atom_data_2d_frames.append(atom_data_current_frame)

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {filename}") from None
    except Exception as e:
        raise RuntimeError(f"An error occurred while parsing the dump file: {e}") from e
    
    # Handle empty files or parsing failures gracefully
    if not all_timesteps:
        return {
            'box': np.array([]), 
            'timesteps': np.array([]), 
            'args': [], 
            'data': np.array([]).reshape(0,0,0),
        }

    box = np.zeros((3,3))
    box[:,0] = initial_box_data[:6][0::2]
    box[:,1] = initial_box_data[:6][1::2]
    box[:,2] = initial_box_data[6:]

    data_matrix_NAC = np.array(all_atom_data_2d_frames, dtype=float)
    timesteps_array = np.array(all_timesteps, dtype=int)
    data = {
        'box': box,
        'timesteps': timesteps_array,
        'args': dumped_column_names,
        'data': data_matrix_NAC,
    }
    return data



def arg2id(data: Mapping[str, Any], arg: str) -> int | None:
    """Return the column index for `arg`, or None if not present."""
    try:
        args = data["args"]
    except KeyError:
        raise KeyError('data must contain key "args".') from None
    arg_map = data.get("_arg2id")
    if isinstance(arg_map, dict):
        return arg_map.get(arg)
    try:
        return args.index(arg)
    except ValueError:
        return None


def select_args(data: Mapping[str, Any], arglist: Sequence[str]) -> np.ndarray:
    """Select columns in data['data'] corresponding to arg names in `arglist`."""
    try:
        args = data["args"]
        arr = data["data"]
    except KeyError as e:
        raise KeyError(f"data must contain key {e!s}.") from None
    arg_map = {name: i for i, name in enumerate(args)}
    try:
        ids = [arg_map[name] for name in arglist]
    except KeyError as e:
        raise ValueError(f'arg "{e.args[0]}" not found.') from None
    return arr[:, :, ids]


def parse_args(
    fn: str,
    arglist: Sequence[str],
    start: int = 0,
    last: int = -1,
    stride: int = 1,
) -> dict[str, Any]:
    """Parse file, keep only selected args and snapshots in returned dict."""
    data = parse_custom(fn, start=start, last=last, stride=stride)
    data["data"] = select_args(data, arglist)
    data["_arg2id"] = {name: i for i, name in enumerate(data["args"])}
    return data



@dataclass(slots=True)
class LoadCustom:
    filename: str | Path
    start: int = 0
    last: int = -1
    stride: int = 1
    verbose: bool = False

    box: np.ndarray = field(init=False)
    timesteps: np.ndarray = field(init=False)
    args: list[str] = field(init=False)
    data: np.ndarray = field(init=False)
    arg2id: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        datadict = parse_custom(
            self.filename, start=self.start, last=self.last, stride=self.stride, verbose=self.verbose
        )
        self.box = datadict['box']
        self.timesteps = datadict['timesteps']
        self.args = datadict['args']
        self.data = datadict['data']
        self.arg2id = {name: i for i, name in enumerate(self.args)}

    def id(self, arg: str) -> int | None:
        """Return column index for `arg`, or None if not present."""
        return self.arg2id.get(arg)

    def select(self, arglist: Sequence[str]) -> np.ndarray:
        """Return selected columns as array (N, A, len(arglist))."""
        try:
            ids = [self.arg2id[name] for name in arglist]
        except KeyError as e:
            raise ValueError(f'arg "{e.args[0]}" not found.') from None
        return self.data[:, :, ids]

    @staticmethod
    def unwrap_pos(pos: np.ndarray, ipos: np.ndarray, box: np.ndarray) -> np.ndarray:
        L = (box[:, 1] - box[:, 0]).astype(pos.dtype, copy=False)
        return pos + ipos.astype(pos.dtype, copy=False) * L

    @property
    def positions(self) -> np.ndarray:
        return self.get_positions(unwrap=True)

    def get_positions(self, unwrap: bool = True) -> np.ndarray:
        """
        Return positions as (N, A, 3).

        unwrap=True:
        - prefer xu/yu/zu
        - else require x/y/z + ix/iy/iz and unwrap via image flags
        - else raise (no fallback to wrapped coords)

        unwrap=False:
        - prefer x/y/z
        - else if xu/yu/zu and box are present, wrap back via modulo
        - else raise
        """
        has_xyz = all(k in self.arg2id for k in ("x", "y", "z"))
        has_xu  = all(k in self.arg2id for k in ("xu", "yu", "zu"))
        has_ix  = all(k in self.arg2id for k in ("ix", "iy", "iz"))

        if unwrap:
            if has_xu:
                return self.select(("xu", "yu", "zu"))
            if has_xyz and has_ix:
                pos = self.select(("x", "y", "z"))
                ipos = self.select(("ix", "iy", "iz"))
                return self.unwrap_pos(pos, ipos, self.box)

            missing = []
            if not has_xu:
                missing.append("(xu,yu,zu)")
            if not (has_xyz and has_ix):
                missing.append("(x,y,z + ix,iy,iz)")
            raise ValueError(
                "unwrap=True requires unwrapped coordinates. Missing: "
                + " or ".join(missing)
                + "."
            )

        if has_xyz:
            return self.select(("x", "y", "z"))
        if has_xu and self.box is not None and self.box.size > 0:
            xu = self.select(("xu", "yu", "zu"))
            lo = self.box[:, 0]
            L = self.box[:, 1] - lo
            return (xu - lo) % L + lo
        raise ValueError(
            'unwrap=False requires wrapped coordinates (x,y,z). '
            'Neither (x,y,z) nor (xu,yu,zu) with box dimensions are present.'
        )

    def pos(self, unwrap: bool = True) -> np.ndarray:
        return self.get_positions(unwrap=unwrap)

    def triads(self) -> np.ndarray:
        """
        Build rotation matrices of shape (N, A, 3, 3) from quaternions.

        Requires quaternion components:
        (c_quat[1], c_quat[2], c_quat[3], c_quat[4])

        Uses external: so3.quats2mats(quats) mapping (N,A,4)->(N,A,3,3).
        """
        quat_keys = ("c_quat[1]", "c_quat[2]", "c_quat[3]", "c_quat[4]")
        has_quats = all(k in self.arg2id for k in quat_keys)

        if not has_quats:
            missing = [k for k in quat_keys if k not in self.arg2id]
            raise ValueError(
                "triads() requires quaternion coordinates. Missing: "
                + ", ".join(missing)
                + "."
            )
        quats = self.select(quat_keys)
        return so3.quats2mats(quats)


    def poses(self, unwrap: bool = True, reduced: bool = False) -> np.ndarray:
        """
        Return SE(3) blocks taus of shape (N, A, 3, 4) or (N, A, 4, 4).

        Parameters
        ----------
        unwrap : bool, optional
            Controls how translational coordinates are obtained.
            If True, unwrapped positions are required and are constructed from
            (xu, yu, zu) if present, or from (x, y, z) together with image flags
            (ix, iy, iz). If False, wrapped coordinates (x, y, z) are required.

        reduced : bool, optional
            If True, return reduced SE(3) blocks of shape (3, 4), corresponding to
            the matrix [R | t]. If False, return full homogeneous matrices of shape
            (4, 4), with the last row fixed to [0, 0, 0, 1].

        Returns
        -------
        taus : np.ndarray
            Array of SE(3) blocks with shape:
            - (N, A, 3, 4) if reduced is True
            - (N, A, 4, 4) if reduced is False
            where N is the number of timesteps and A the number of atoms.
        """
        pos = self.pos(unwrap=unwrap)
        R = self.triads()

        if R.shape[:2] != pos.shape[:2]:
            raise ValueError(
                f"Incompatible shapes: triads has {R.shape[:2]} frames/atoms, "
                f"but pos has {pos.shape[:2]}."
            )
        if R.shape[-2:] != (3, 3) or pos.shape[-1] != 3:
            raise ValueError(f"Unexpected shapes: triads={R.shape}, pos={pos.shape}.")

        if reduced:
            taus = np.empty((pos.shape[0], pos.shape[1], 3, 4), dtype=np.result_type(R, pos))
        else:
            taus = np.empty((pos.shape[0], pos.shape[1], 4, 4), dtype=np.result_type(R, pos))
            taus[:,:,3,3] = 1
            
        taus[..., :3, :3] = R
        taus[..., :3, 3] = pos
        return taus
    
    def get_parameters(self, dynamic: bool = False) -> np.ndarray:
        junctions = poses2junctions(self.poses(unwrap=True, reduced=False))
        if dynamic:
            return junctions2parameters(junctions2dynamics(junctions))
        else:             
            return junctions2parameters(junctions)

    def get_mean_params(self) -> np.ndarray:
        params = self.get_parameters(dynamic=False)
        return np.mean(params, axis=0)
    
    
    



if __name__ == "__main__":
    
    fn = sys.argv[1]
    
    data = parse_custom(fn)
    print('contained args:')    
    for arg in data['args']:
        print(arg)
    print(f'data shape: {data["data"].shape}')
    print(f'box: {data["box"]}')
    
    print(arg2id(data,'type'))
    print(arg2id(data,'iz'))
    print(arg2id(data,'it'))
    
    print(data['data'].shape)
    
    arglist = ['id','x','y','z','ix','iy','iz']
    data = parse_args(fn,arglist)
    
    print(data['data'].shape)
    print(data['data'][-1,-1])
    
    # arglist = ['id','x','y','z','ix','iy','iz','xxx']
    # data = parse_args(fn,arglist)
    
    lmp = LoadCustom(fn)
    taus = lmp.se3(unwrap=True)
    
    print(taus.shape)
    print(taus[-1,-5:])