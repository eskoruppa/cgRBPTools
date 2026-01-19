from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Callable,  Tuple
import sys
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from ..SO3 import so3


def parse_custom(filename: str) -> Dict:
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


def parse_args(fn: str, arglist: Sequence[str]) -> dict[str, Any]:
    """Parse file and keep only selected args in returned dict."""
    data = parse_custom(fn)
    data["data"] = select_args(data, arglist)
    data["_arg2id"] = {name: i for i, name in enumerate(data["args"])}
    return data



@dataclass(slots=True)
class LMPCustom:
    filename: str | Path

    box: np.ndarray = field(init=False)
    timesteps: np.ndarray = field(init=False)
    args: list[str] = field(init=False)
    data: np.ndarray = field(init=False)
    arg2id: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        datadict = parse_custom(self.filename)
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

    def pos(self, unwrap: bool = True) -> np.ndarray:
        """
        Return positions as (N, A, 3).

        unwrap=True:
        - prefer xu/yu/zu
        - else require x/y/z + ix/iy/iz and unwrap via image flags
        - else raise (no fallback to wrapped coords)

        unwrap=False:
        - require x/y/z
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
        raise ValueError('unwrap=False requires wrapped coordinates (x,y,z), but they are not present.')

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


    def se3(self, unwrap: bool = True, reduced: bool = False) -> np.ndarray:
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
    
    lmp = LMPCustom(fn)
    taus = lmp.se3(unwrap=True)
    
    print(taus.shape)
    print(taus[-1,-5:])