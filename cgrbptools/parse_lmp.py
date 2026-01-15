import numpy as np
import sys

def parse_out(filepath: str) -> dict:
    """
    Parses a LAMMPS custom dump file and returns data in a dictionary format.

    This function expects the number of atoms and the number of dumped arguments
    to be constant across all timesteps in the dump file.

    Args:
        filepath (str): The path to the LAMMPS custom dump file.

    Returns:
        dict: A dictionary containing the parsed data:
              - 'box' (np.ndarray): A 1D NumPy array of 9 floats
                [xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz]
                representing the box dimensions from the **FIRST** timestep.
              - 'timesteps' (np.ndarray): A 1D NumPy array containing the timestep numbers (length N).
              - 'args' (list): A list of strings with the names of the dumped attributes
                (e.g., ['id', 'x', 'y', 'z', 'q[0]', 'q[1]', 'q[2]', 'q[3]']).
              - 'data' (np.ndarray): A 3D NumPy array of shape (N, A, C), where:
                N = number of timesteps
                A = number of atoms per timestep
                C = number of arguments (columns) per atom

    Raises:
        FileNotFoundError: If the specified filepath does not exist.
        ValueError: If the dump file format is unexpected, or if the number of atoms
                    or columns changes between timesteps.
        RuntimeError: For other unexpected parsing errors.
    """
    all_timesteps = []
    all_atom_data_2d_frames = [] # Will store (A, C) arrays for each frame
    
    initial_num_atoms = -1
    initial_num_cols_per_atom = -1
    dumped_column_names = None
    initial_box_data = None # Store box from the first frame

    try:
        with open(filepath, 'r') as f:
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

                # Standardize box representation to always be 9 elements:
                # [xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz]
                current_box_array = np.zeros(9, dtype=float)
                if "xy xz yz" in box_type_str: # Triclinic box
                    if not (len(x_line) == 3 and len(y_line) == 3 and len(z_line) == 3):
                        raise ValueError("Triclinic box bounds expected 3 values per line (lo, hi, tilt)")
                    current_box_array[0], current_box_array[1], current_box_array[6] = x_line
                    current_box_array[2], current_box_array[3], current_box_array[7] = y_line
                    current_box_array[4], current_box_array[5], current_box_array[8] = z_line
                else: # Orthogonal box
                    if not (len(x_line) == 2 and len(y_line) == 2 and len(z_line) == 2):
                        raise ValueError("Orthogonal box bounds expected 2 values per line (lo, hi)")
                    current_box_array[0], current_box_array[1] = x_line
                    current_box_array[2], current_box_array[3] = y_line
                    current_box_array[4], current_box_array[5] = z_line
                    # xy, xz, yz remain 0.0

                # ITEM: ATOMS header (column names)
                line = f.readline()
                if not line.startswith("ITEM: ATOMS"):
                    raise ValueError(f"Expected 'ITEM: ATOMS', but got: {line.strip()}")
                column_names = line.strip().split(' ')[2:]

                # --- Consistency Checks for (N, A, C) Matrix ---
                if initial_num_atoms == -1: # First frame setup
                    initial_num_atoms = num_atoms
                    initial_num_cols_per_atom = len(column_names)
                    dumped_column_names = column_names
                    initial_box_data = current_box_array # Store the first box
                else: # Subsequent frames: validate consistency
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
        raise FileNotFoundError(f"Error: File not found at {filepath}") from None
    except Exception as e:
        raise RuntimeError(f"An error occurred while parsing the dump file: {e}") from e
    
    # Handle empty files or parsing failures gracefully
    if not all_timesteps:
        return {
            'box': np.array([]), # Empty box
            'timesteps': np.array([]), # Empty timesteps array
            'args': [], # Empty args list
            'data': np.array([]).reshape(0,0,0), # Empty 3D data array
        }

    # Stack all 2D atom data frames into a single 3D NumPy array
    data_matrix_NAC = np.array(all_atom_data_2d_frames, dtype=float)
    timesteps_array = np.array(all_timesteps, dtype=int)

    data = {
        'box': initial_box_data,
        'timesteps': timesteps_array,
        'args': dumped_column_names,
        'data': data_matrix_NAC,
    }
    return data



if __name__ == "__main__":
    
    fn = sys.argv[1]
    
    data = parse_out(fn)

    print(data['data'].shape)
    print(data['timesteps'])
    print(data['box'].shape)
    print(data['box'])

