from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from .lmp_topol import CGRBPTopology
from .lmp_config import CGRBPConfig



def gen_datafile(
    filename: str,
    topology : CGRBPTopology,
    config: CGRBPConfig,
    box: np.ndarray = None,
    hybrid: bool = False,
    include_coeffs: bool = False,
    add_extension: bool = True
    ) -> None:

    ext = '.data'
    if add_extension and os.path.splitext(filename)[-1].lower() != ext:
        filename += ext
    
    with open(filename,'w') as f:
        f.write(f'\n\n')
        # atoms
        f.write(f'{len(config.positions)} atoms\n')
        f.write(f'{len(topology.bonds)} bonds\n')
        f.write(f'{len(topology.angles)} angles\n')
        f.write(f'{len(topology.dihedrals)} dihedrals\n')
        f.write(f'1 atom types\n')
        f.write(f'{len(topology.bondtypes)} bond types\n')
        f.write(f'{len(topology.angletypes)} angle types\n')
        f.write(f'{len(topology.dihedraltypes)} dihedral types\n')
        f.write(f'{config.nbp} ellipsoids\n')
        f.write('\n')
        
        if box is None:
            box = config.extended_bounds(margin_fraction=0.05, square_box=True)
        
        rbox = np.round(box,decimals=1)
        
        f.write(f'{rbox[0,0]} {rbox[0,1]} xlo xhi\n')
        f.write(f'{rbox[1,0]} {rbox[1,1]} ylo yhi\n')
        f.write(f'{rbox[2,0]} {rbox[2,1]} zlo zhi\n')
        

        if len(topology.bondtypes) > 0:
            f.write(f'\nMasses\n\n')
            for mass_sting in config.mass_strings():
                f.write(f'{mass_sting}\n')
            f.write('\n')
        
        if config.nbp > 0:
            f.write(f'\nAtoms\n\n')
            atomstrs = config.atom_strings()
            for atomstr in atomstrs:
                f.write(f'{atomstr}\n')
            f.write('\n')  
            
        if config.nbp > 0:
            f.write(f'\nEllipsoids\n\n')
            ellipsstrs = config.ellipsoid_strings()
            for ellipsstr in ellipsstrs:
                f.write(f'{ellipsstr}\n')
            f.write('\n')  
        
        if include_coeffs:
            if len(topology.bondtypes) > 0:
                f.write(f'\nBond Coeffs\n\n')
                for bondtype in topology.bondtypes:
                    f.write(f'{bondtype.to_str(hybrid=hybrid)}\n')
                f.write('\n')

            if len(topology.angletypes) > 0:
                f.write(f'\nAngle Coeffs\n\n')
                for angletype in topology.angletypes:
                    f.write(f'{angletype.to_str(hybrid=hybrid)}\n')
                f.write('\n')

            if len(topology.dihedraltypes) > 0:
                f.write(f'\nDihedral Coeffs\n\n')
                for dihedraltype in topology.dihedraltypes:
                    f.write(f'{dihedraltype.to_str(hybrid=hybrid)}\n')
                f.write('\n')

        if len(topology.bonds) > 0:
            f.write(f'\nBonds\n\n')
            for bond in topology.bonds:
                f.write(f'{bond.to_str()}\n')
            f.write('\n')

        if len(topology.angles) > 0:
            f.write(f'\nAngles\n\n')
            for angle in topology.angles:
                f.write(f'{angle.to_str()}\n')
            f.write('\n')
            
        if len(topology.dihedrals) > 0:
            f.write(f'\nDihedrals\n\n')
            for dihedral in topology.dihedrals:
                f.write(f'{dihedral.to_str()}\n')
            f.write('\n')