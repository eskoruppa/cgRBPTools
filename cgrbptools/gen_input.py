from __future__ import annotations

import sys, os
import argparse
import numpy as np
from pathlib import Path

# load sequence from sequence file
from .PolyCG.polycg import gen_params, load_sequence, visualize_chimerax, visualize_pdb, visualize_xyz, write_seqfile

from .io.lmp_topol import CGRBPTopology
from .io.conf_builder import ConfBuilder
from .io.unit_conversion import RescaleUnits
from .io.matrix_methods import rescale_stiff
from .io.backmap import dna_backmap

GEN_INPUT_CGNAP_SETNAME = 'curves_plus'

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="Generate cgRBP input files")
    parser.add_argument(
        '-seqfn',   
        '--sequence_file',      
        type=str, default = None,
        help='Path to DNA sequence file (.seq extension). Used to derive output filename if -o not specified.')
    parser.add_argument(
        '-seq',     
        '--sequence',           
        type=str, default = None,
        help='DNA sequence as string (alternative to -seqfn). Requires -o to specify output filename.')
    parser.add_argument(
        '-m',       
        '--model',              
        type=str, default = 'cgnaplus', choices=['cgnaplus','md','crystall'],
        help='DNA model for parameter generation (default: cgnaplus)')
    parser.add_argument(
        '-cg',      
        '--composite_size',     
        type=int, default = 1,
        help='Number of base pairs per coarse-grained bead (default: 1 for all-atom)')
    parser.add_argument(
        '-cr',      
        '--coupling_range',     
        type=int, default = 1,
        help='Range of beyond nearest-neighbor interactions. Only local couplings if set to 0. (default: 1)')
    parser.add_argument(
        '-dec',     
        '--decimals',           
        type=int, default = 2,
        help='Number of decimal places for output formatting (default: 2)')
    parser.add_argument(
        '-mass',    
        '--mass',           
        type=float, default = 1.0,
        help='Mass of atoms in output (default: 1.0)')
    parser.add_argument(
        '-closed',  
        '--closed',             
        action='store_true',
        help='Generate closed (circular) configuration with couplings across the periodic boundary') 
    parser.add_argument(
        '-center',  
        '--centered',             
        action='store_true',
        help='Place retained triads at the center of coarse-grained blocks.') 
        
    parser.add_argument(
        '-fene',
        '--bond_fene_coeffs',
        nargs=3,
        type=float,
        metavar=("K", "Rc", "R0"),
        default=None,
        help="Include native FENE bond in rbp bond couplings. Requires three arguments (K, Rc, R0).",
    )
    
    parser.add_argument(
        '-nc',      
        '--no_crop',            
        action='store_true',
        help='Disable automatic sequence cropping for boundary conditions') 
    parser.add_argument(
        '-np',      
        '--no_partial',         
        action='store_true',
        help='Disable partial block assembly for stiffness matrix') 
    parser.add_argument(
        '-sid',     
        '--start_id',           
        type=int, default=0,
        help='Starting base pair index for subsection generation (default: 0)') 
    parser.add_argument(
        '-eid',     
        '--end_id',             
        type=int, default=None,
        help='Ending base pair index for subsection generation (default: full sequence)') 
    parser.add_argument(
        '-o',       
        '--output_basename',    
        type=str, default = None, required=False,
        help='Base filename for output files (required if using -seq, otherwise derived from -seqfn)')
    parser.add_argument(
        '-stiff',       
        '--safe_stiffmat',    
        action='store_true',
        help='Save stiffness matrix to .npy file') 
    parser.add_argument(
        '-gs',       
        '--safe_groundstate',   
        action='store_true',
        help='Save groundstate configuration to .npy file')
    
    
    parser.add_argument(
        '-xyz',     
        '--gen_xyz',            
        action='store_true',
        help='Generate XYZ coordinate file')
    parser.add_argument(
        '-pdb',     
        '--gen_pdb',            
        action='store_true',
        help='Generate PDB structure file')
    parser.add_argument(
        '-vis',     
        '--visualize_cgrbp',    
        action='store_true',
        help='Generate ChimeraX visualization script (.cxc)')
    parser.add_argument(
        '-bpst',    
        '--include_bps_triads', 
        action='store_true',
        help='Include base pair step triads in visualization (requires -vis)') 
    parser.add_argument(
        '-nodup',    
        '--remove_duplicate',
        type=bool, default=True, 
        help='Remove duplicate coupling styles: bonds styles, angle styles, dihedral styles (default: True)') 
    

    #########################################
    ############# INCLUDE HELP
    
    parser.add_argument('-lr',      '--len_rescale',        type=float, default = 1.0)
    parser.add_argument('-ls',      '--length_scale',       type=float, default = 1.0)
    args = parser.parse_args()
    

    ##################################################################################################################
    # additional edits
    
    allow_crop      = not args.no_crop
    allow_partial   = not args.no_partial
    
    include_fene = False
    bond_fene_coeffs = None
    if args.bond_fene_coeffs is not None:
        bond_fene_coeffs = np.array(args.bond_fene_coeffs, dtype=float)
        include_fene = True
    
    if args.output_basename is not None:
        outname = args.output_basename
    else:
        if args.sequence_file is not None:
            outname = str(args.sequence_file).replace('.seq','')
        else:
            raise ValueError(f'No output name or sequence filename specified')
    
    len_rescale = args.len_rescale
    length_scale = args.length_scale
    
    if length_scale != 1:
        len_rescale = 1. / length_scale
    
    ##################################################################################################################
    # load sequence
    
    seq = args.sequence
    if seq is None:
        if args.sequence_file is None:
            raise ValueError(f'Requires either a sequence (-seq) or a sequence file (-seqfn)')
        seq = load_sequence(args.sequence_file)
       
    cgnap_setname = GEN_INPUT_CGNAP_SETNAME
    

    params = gen_params(
        args.model, 
        seq,
        composite_size=args.composite_size,
        closed=args.closed,
        start_id=args.start_id,
        end_id=args.end_id,
        allow_partial=allow_partial,
        allow_crop=allow_crop,
        cgnap_setname = cgnap_setname,
        verbose=True,
    )
    
    if args.output_basename is None:
        base_fn = Path(args.sequence_file)
    else:
        base_fn = Path(args.output_basename)

    if params.is_coarse_grained:
        stiff = params.cg_stiffmat
        shape = params.cg_shape_params
    else: 
        stiff = params.stiffmat
        shape = params.shape_params
        
        
    # X0 = np.array([[0,0,0.6,0,0,3.4]])
    # M0 = np.eye(6)
    # M0[0,0] =  40.0
    # M0[1,1] =  40.0
    # M0[2,2] =  100.0
    # M0[3,3] =  200.0
    # M0[4,4] =  200.0
    # M0[5,5] =  200.0
    
    # stiff = np.zeros(stiff.shape)
    # for i in range(stiff.shape[0] // 6):
    #     shape[i,:] = X0
    #     stiff[i*6:(i+1)*6,i*6:(i+1)*6] = M0
    # ##################################################
    
    # print(f'len_rescale = {len_rescale}')
    # print(f'rec_len_res = {1./len_rescale}')
    # sys.exit()
        
    rescale = RescaleUnits(length_factor=len_rescale)
    shape,stiff = rescale.rescale_model(shape,stiff)
    # mean_disc_len = np.mean(shape[:,5])
    # length_rescale_factor = 1./3.4
    ##################################################
    ########## RESCALING #############################
    rot_rescale = np.sqrt(0.667)
    rise_rescale  = np.sqrt(0.16)
    
    stiff = rescale_stiff(stiff,rot_rescale,entries=[0,1,2])
    stiff = rescale_stiff(stiff,rise_rescale,entries=[5])
    ########## RESCALING #############################
    ##################################################

    
    topol = CGRBPTopology(coupling_range=args.coupling_range,decimals=args.decimals,check_existing_types=args.remove_duplicate,closed=args.closed)
    if include_fene:
        topol.set_fene(*bond_fene_coeffs)
    topol.set_params(shape,stiff)
    
    # topol.set_fene(200,1.1,1.35)
    # topol.remove_fene()
    # topol.set_coupling_range(2)
    # topol.set_closed(closed)
    
    if args.end_id is not None:
        cropped_seq = seq[args.start_id:args.end_id]
    else:
        cropped_seq = seq[args.start_id:]
    topol.set_sequence(cropped_seq,chars_per_atom=args.composite_size,centered=args.centered)
    topol.write_database(base_fn,add_extension=True)
    
    # topol2 = CGRBPTopology.read_database(base_fn.with_suffix('.db'),check_existing_types=check_existing_types)
    # topol2.write_database(base_fn.with_name(base_fn.stem + '_copy'),add_extension=True)

    # print(topol.closed, topol2.closed)
    # print(topol.nbp, topol2.nbp)
    # print(topol.nbps, topol2.nbps)
    # print(topol.coupling_range, topol2.coupling_range)
    # print(topol.num_bonds, topol2.num_bonds)
    # print(topol.num_angles, topol2.num_angles)
    # print(topol.num_dihedrals, topol2.num_dihedrals)
    # print(topol.num_bond_types, topol2.num_bond_types)
    # print(topol.num_angle_types, topol2.num_angle_types)
    # print(topol.num_dihedral_types, topol2.num_dihedral_types)  
    # print(topol.seqs_set, topol2.seqs_set)
    # print(topol.seqs_centered, topol2.seqs_centered)
    # print(topol.chars_per_atom, topol2.chars_per_atom)
    

    conf = ConfBuilder.straight(topol,mass=args.mass)
    
    print(np.min(conf.poses[:,:3,3],axis=0))
    print(np.max(conf.poses[:,:3,3],axis=0))
    box = conf.extended_bounds(0.0,square_box=True)
    print(box)
    box = conf.extended_bounds(0.2,square_box=True)
    print(box)
    
    
    conf = ConfBuilder.circular(topol,mass=args.mass)
    
    # conf = ConfBuilder.ground_state(topol,mass=args.mass)
    
    
    # from .io.backmap import dna_backmap
    
    # dna_backmap(conf,topology=topol)
    
    
    # sys.exit()
    
    np.set_printoptions(precision=1,linewidth=300,suppress=True)
    
    # conf = ConfBuilder.ground_state(topol,mass=args.mass)
    
    
    
    
    vis_seq = params.sequence
    # if args.closed:
    #     vis_seq += vis_seq[0]
    
    # visualization cgrbp
    if args.visualize_cgrbp:
        if args.composite_size > 1:
            bead_radius = args.composite_size*0.34*0.5
            bp_poses = dna_backmap(conf,topology=topol,verbose=True)
        else:
            bead_radius = 0
            bp_poses = conf.poses
        
        
        visualize_chimerax(base_fn, vis_seq, args.composite_size, poses=bp_poses, start_id=args.start_id, bead_radius=bead_radius,include_bps_triads=args.include_bps_triads) 
        
    if args.gen_pdb and not args.visualize_cgrbp:
        visualize_pdb(base_fn, vis_seq, shape_params=params.shape_params)
        
    if args.gen_xyz:
        visualize_xyz(base_fn, args.composite_size, shape_params=params.shape_params, start_id=args.start_id)
    
    
    
    
    
    
    # conf = ConfBuilder.circular(topol,radius=None,mass=args.mass)

    
    # sys.exit()
    
    
    
    # gen_datafile(outname,topol,conf,hybrid=False,box=box)
    # gen_database_file(outname,topol,seq=seq,composite_size=composite_size,start_id=start_id,end_id=end_id)
        
    # if safe_stiffmat:
    #     stifffn = outname + '_stiff.npy'
    #     np.save(stifffn,stiff.toarray())
    
    # if safe_groundstate:
    #     shape_fn = outname + '_shape.npy'
    #     np.save(shape_fn,shape)
        
