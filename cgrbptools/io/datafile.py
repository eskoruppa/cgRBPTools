from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from .lmp_topol import CGRBPTopology
from .lmp_conf import CGRBPConf
from .backups import backup_filename

# Constants
CGRBP_DEFAULT_MARGIN_FRACTION = 0.05
CGRBP_DEFAULT_BOX_DECIMALS = 1
CGRBP_DATA_EXTENSION = '.data'


def cgrbp_datafile(
    filename: str | Path,
    topology: CGRBPTopology,
    config: CGRBPConf,
    box: np.ndarray = None,
    hybrid: bool = False,
    include_coeffs: bool = False,
    add_extension: bool = True,
    box_decimals: int = CGRBP_DEFAULT_BOX_DECIMALS,
    overwrite: bool = False
    ) -> Path:
    """Generate a LAMMPS data file for custom bond/angle/dihedral model.
    
    Args:
        filename: Output file path
        topology: CGRBP topology containing bonds, angles, dihedrals
        config: CGRBP configuration with positions and orientations
        box: Simulation box bounds [shape: (3,2)], auto-calculated if None
        hybrid: Use hybrid style formatting for coefficients
        include_coeffs: Include coefficient sections in output
        add_extension: Automatically add .data extension if missing
        box_decimals: Decimal precision for box dimensions
        overwrite: If False (default), backup existing file before writing; if True, overwrite
        
    Returns:
        Path object to the created data file
        
    Raises:
        TypeError: If topology or config have incorrect types
        ValueError: If topology and config are inconsistent
        OSError: If file cannot be written
    """
    # Input validation (Suggestion 1)
    if not isinstance(topology, CGRBPTopology):
        raise TypeError(f"topology must be CGRBPTopology, got {type(topology).__name__}")
    if not isinstance(config, CGRBPConf):
        raise TypeError(f"config must be CGRBPConf, got {type(config).__name__}")
    if topology is None or config is None:
        raise ValueError("topology and config cannot be None")
    
    # Validate consistency (Suggestion 2)
    if config.nbp != len(config.positions):
        raise ValueError(f"Inconsistent config: nbp={config.nbp} but positions has {len(config.positions)} entries")
    
    # Convert filename to Path and add extension
    filepath = Path(filename)
    if add_extension and filepath.suffix.lower() != CGRBP_DATA_EXTENSION:
        filepath = filepath.with_suffix(CGRBP_DATA_EXTENSION)
    
    # Calculate box if not provided
    if box is None:
        box = config.extended_bounds(margin_fraction=CGRBP_DEFAULT_MARGIN_FRACTION, square_box=True)
    
    # Validate box bounds (Suggestion 3)
    if box.shape != (3, 2):
        raise ValueError(f"box must have shape (3, 2), got {box.shape}")
    positions = config.positions
    if np.any(positions < box[:, 0]) or np.any(positions > box[:, 1]):
        raise ValueError("Some atom positions are outside the specified box bounds")
    
    rounded_box = np.round(box, decimals=box_decimals)
    
    # Build content using list accumulation for better performance (Suggestions 19, 20)
    lines = []
    
    # Header
    lines.append('\n')
    lines.append('\n')
    
    # Counts section
    lines.append(f'{len(config.positions)} atoms\n')
    lines.append(f'{len(topology.bonds)} bonds\n')
    lines.append(f'{len(topology.angles)} angles\n')
    lines.append(f'{len(topology.dihedrals)} dihedrals\n')
    lines.append(f'1 atom types\n')
    lines.append(f'{len(topology.bondtypes)} bond types\n')
    lines.append(f'{len(topology.angletypes)} angle types\n')
    lines.append(f'{len(topology.dihedraltypes)} dihedral types\n')
    lines.append(f'{config.nbp} ellipsoids\n')
    lines.append('\n')
    
    # Box dimensions
    lines.append(f'{rounded_box[0,0]} {rounded_box[0,1]} xlo xhi\n')
    lines.append(f'{rounded_box[1,0]} {rounded_box[1,1]} ylo yhi\n')
    lines.append(f'{rounded_box[2,0]} {rounded_box[2,1]} zlo zhi\n')
    
    # Masses section
    if len(topology.bondtypes) > 0:
        lines.append('\nMasses\n\n')
        for mass_string in config.mass_strings():
            lines.append(f'{mass_string}\n')
        lines.append('\n')
    
    # Atoms section
    if config.nbp > 0:
        lines.append('\nAtoms\n\n')
        atom_strs = config.atom_strings()
        for atom_str in atom_strs:
            lines.append(f'{atom_str}\n')
        lines.append('\n')  
    
    # Ellipsoids section
    if config.nbp > 0:
        lines.append('\nEllipsoids\n\n')
        ellipsoid_strs = config.ellipsoid_strings()
        for ellipsoid_str in ellipsoid_strs:
            lines.append(f'{ellipsoid_str}\n')
        lines.append('\n')  
    
    # Coefficients sections (if requested)
    if include_coeffs:
        if len(topology.bondtypes) > 0:
            lines.append('\nBond Coeffs\n\n')
            for bondtype in topology.bondtypes:
                lines.append(f'{bondtype.to_str(hybrid=hybrid)}\n')
            lines.append('\n')

        if len(topology.angletypes) > 0:
            lines.append('\nAngle Coeffs\n\n')
            for angletype in topology.angletypes:
                lines.append(f'{angletype.to_str(hybrid=hybrid)}\n')
            lines.append('\n')

        if len(topology.dihedraltypes) > 0:
            lines.append('\nDihedral Coeffs\n\n')
            for dihedraltype in topology.dihedraltypes:
                lines.append(f'{dihedraltype.to_str(hybrid=hybrid)}\n')
            lines.append('\n')

    # Bonds section
    if len(topology.bonds) > 0:
        lines.append('\nBonds\n\n')
        for bond in topology.bonds:
            lines.append(f'{bond.to_str()}\n')
        lines.append('\n')

    # Angles section
    if len(topology.angles) > 0:
        lines.append('\nAngles\n\n')
        for angle in topology.angles:
            lines.append(f'{angle.to_str()}\n')
        lines.append('\n')
    
    # Dihedrals section
    if len(topology.dihedrals) > 0:
        lines.append('\nDihedrals\n\n')
        for dihedral in topology.dihedrals:
            lines.append(f'{dihedral.to_str()}\n')
        lines.append('\n')
    
    # Backup existing file if needed (Suggestion 21)
    if filepath.exists() and not overwrite:
        backup_path = backup_filename(filepath)
        try:
            filepath.rename(backup_path)
        except OSError as e:
            raise OSError(f"Failed to backup existing file to {backup_path}: {e}") from e
    
    # Write all content at once (Suggestion 4: error handling)
    try:
        with open(filepath, 'w') as f:
            f.write(''.join(lines))
    
    except OSError as e:
        raise OSError(f"Failed to write data file to {filepath}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error while generating data file: {e}") from e
    
    return filepath