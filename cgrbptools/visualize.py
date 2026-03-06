from __future__ import annotations

import sys, os
import argparse
from pathlib import Path

from .core.topology import CGRBPTopology
from .core.conf_builder import ConfBuilder
from .io.parse_custom import LoadCustom

###################################################################################################
###################################################################################################
###################################################################################################

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Visualize snapshots from a LAMMPS custom output file using ChimeraX.")
    parser.add_argument(
        '-in',   
        '--custom_file',      
        type=str, default = None,
        required = True,
        help='Path to the LAMMPS custom output file.'
    )
    parser.add_argument(
        '-db',   
        '--database_file',      
        type=str, default = None,
        required = False,
        help='Path to the database file.'
    )
    parser.add_argument(
        '-s',   
        '--snapshot',      
        type=int, default = -1,
        required = False,
        help=''
    )
    parser.add_argument(
        '-all',       
        '--all',    
        action='store_true',
        help='Visualize all snapshots.') 
    parser.add_argument(
        '-stride',       
        '--stride',    
        type=int, default=1,
        required=False,
        help='Stride for visualizing snapshots when visualizing all.'
    ) 
    args = parser.parse_args()
    
    # load lmps custom output file
    custom_file = Path(args.custom_file)
    if not custom_file.exists():
        raise ValueError(f"Custom parameter file '{custom_file}' does not exist.")
    
    basefn = custom_file.with_suffix('')
    dbfn = Path(args.database_file) if args.database_file is not None else None
    if dbfn is None:
        dbfn = basefn.with_suffix('.db')
        # CHECK IF FILE FileExists:
        if not dbfn.exists():
            raise ValueError(f"Database file '{dbfn}' does not exist. Please provide a valid database file using the '-db' argument.")
     
    topol = CGRBPTopology.read_database(dbfn)
    custom = LoadCustom(custom_file)
    poses = custom.poses(unwrap=True, reduced=False)
    
    if not args.all:
        if args.snapshot >= len(poses) or args.snapshot < -len(poses):
            raise ValueError(f"Snapshot index '{args.snapshot}' is out of range. The number of available snapshots is {len(poses)}.")
        conf = ConfBuilder.from_poses(poses[args.snapshot],topol)
        conf.visualize_chimerax(basefn, include_bps_triads=True)
    else:
        for i in range(0,len(poses),args.stride):
            print(f'Visualizing snapshot {i} / {len(poses)}')
            conf = ConfBuilder.from_poses(poses[i],topol)
            snapfn = basefn.with_name(basefn.stem + f'_snapshots')
            conf.visualize_chimerax(snapfn / f'snapshot_{i:04d}', include_bps_triads=True)
