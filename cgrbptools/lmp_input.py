from __future__ import annotations

import sys, os
import argparse
import numpy as np
from pathlib import Path

from .PolyCG.polycg import gen_params, load_sequence, write_seqfile
# from .PolyCG.polycg import visualize_chimerax, visualize_pdb, visualize_xyz

from .core.lmp_topol import CGRBPTopology
from .core.conf_builder import ConfBuilder
from .core.matrix_methods import rescale_stiff
# from .io.unit_conversion import RescaleUnits
# from .io.backmap import dna_backmap

GEN_INPUT_CGNAP_SETNAME = 'curves_plus'

def parse_rescaling(resc_args, ndim=6):
    scale = [1.0] * ndim
    for group in resc_args:
        if len(group) < 2:
            raise argparse.ArgumentTypeError(
                "-r/--resc requires: factor dim [dim ...]"
            )
        try:
            factor = float(group[0])
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid scale factor for rescaling'{group[0]}'")

        for d_str in group[1:]:
            try:
                d = int(d_str)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid rescaling dimension '{d_str}'. Only the first argument can be a float factor; the rest must be integer dimensions.")

            if not (0 <= d < ndim):
                raise argparse.ArgumentTypeError(f"Rescaling dimension {d} out of range (0..{ndim-1})")

            scale[d] = factor
    return scale

###################################################################################################
###################################################################################################
###################################################################################################

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
        type=int, default = 4,
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
        '-centered',  
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
        help="Include native FENE bond in rbp bond couplings. Requires three arguments (K, Rc, R0).",)
    
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
        '-coeffs',       
        '--safe_coeffs',    
        action='store_true',
        help='Save stiffness matrix and shape to file.') 
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
        
    parser.add_argument(
        '-ul', 
        '--unit_length', 
        type=float, 
        default = 1.0,
        help='Set length unit length in nm. Rescales parameters to chosen scale. If set to, for example, 3.4, all length are rescaled by 1./3.4 (default: 1.0 nm)') 
        
    parser.add_argument(
        '-ue', 
        '--unit_energy', 
        type=float, 
        default = 1.0,
        help='Set energy unit energy in kT. Rescales parameters to chosen scale. (default: 1.0 kT)') 
    
    parser.add_argument(
        "-sr", "--stiff_resc",
        action="append",
        nargs="+",
        default=[],
        metavar=("FACTOR", "DIM"),
        help="Rescale stiffness of invidual dimensions: -r factor dim [dim ...]. Can be repeated.")
    
    parser.add_argument(
        '-conf',
        "--configuration_method",
        type=lambda s: s.strip().lower(),
        choices=sorted(ConfBuilder.mapping_dict().keys()),  # accept aliases
        default=None,
        help=(
            "Mode selection. Allowed values: "
            + ", ".join(sorted(ConfBuilder.mapping_dict().keys()))
            + ". Defaults to None."
        ),)
    args = parser.parse_args()
    
    ##################################################################################################################
    # additional edits
    allow_crop      = not args.no_crop
    allow_partial   = not args.no_partial
    
    ###################################################
    ########## Process bond FENE ######################
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
    
    
    ###################################################
    ########## Load sequence ##########################
    seq = args.sequence
    if seq is None:
        if args.sequence_file is None:
            raise ValueError(f'Requires either a sequence (-seq) or a sequence file (-seqfn)')
        seq = load_sequence(args.sequence_file)
      
    ###################################################
    ########## Crop sequence ##########################
    
    if args.end_id is not None:
        seq = seq[args.start_id:args.end_id]
    else:
        seq = seq[args.start_id:]
    start_id = 0
    end_id = None
        
    ###################################################
    ########## Generate parameters ####################
    cgnap_setname = GEN_INPUT_CGNAP_SETNAME
    params = gen_params(
        args.model, 
        seq,
        composite_size=args.composite_size,
        closed=args.closed,
        start_id=start_id,
        end_id=end_id,
        allow_partial=allow_partial,
        allow_crop=allow_crop,
        cgnap_setname = cgnap_setname,
        verbose=True,
    )
    
    ###################################################
    ########## Determine output base filename #########
    if args.output_basename is None:
        base_fn = Path(args.sequence_file)
    else:
        base_fn = Path(args.output_basename)

    ###################################################
    ########## Select shape and stiffness matrices ####
    if params.is_coarse_grained:
        stiff = params.cg_stiffmat
        shape = params.cg_shape_params
    else: 
        stiff = params.stiffmat
        shape = params.shape_params
        
    # ##################################################
    # ########## Rescale length units ##################
    # if args.unit_length != 1.0:
    #     rescale = RescaleUnits(length_factor=1./args.unit_length)
    #     shape,stiff = rescale.rescale_model(shape,stiff)
    
    ##################################################
    ########## Rescale stiffnesses ###################
    scale = parse_rescaling(args.stiff_resc, ndim=6)    
    for i in range(len(scale)):
        if scale[i] != 1.0:
            stiff = rescale_stiff(stiff,scale[i],entries=[i])

    ##################################################
    ########## Generate topology #####################
    topol = CGRBPTopology(
        coupling_range=args.coupling_range,
        decimals=args.decimals,
        check_existing_types=args.remove_duplicate,
        closed=args.closed
    )
    if include_fene:
        topol.set_fene(*bond_fene_coeffs)
    topol.set_params(shape,stiff)
    topol.set_unit_energy(args.unit_energy)
    topol.set_unit_length(args.unit_length)
        
    # topol.set_fene(200,1.1,1.35)
    # topol.remove_fene()
    # topol.set_coupling_range(2)
    # topol.set_closed(closed)
    
    ##################################################
    ########## Write topology ########################
    topol.set_sequence(seq,chars_per_atom=args.composite_size,centered=args.centered)
    topol.write_database(base_fn,add_extension=True)

    ##################################################
    ########## Build configuration ###################
    conf = None
    gen_conf = args.configuration_method is not None
    if gen_conf:
        conf_method_key = args.configuration_method
        print(f"Generating configuration using method: {ConfBuilder.mapping_dict()[conf_method_key]} (input: {conf_method_key})")
        conf = ConfBuilder.build(topol, conf_method_key, mass=args.mass)
        box = conf.extended_bounds(0.5,square_box=True)
    
        conf.write_datafile(
            outname,
            topol,
            box = box,
            hybrid = False,
            include_coeffs = False,
            add_extension = True,
            box_decimals = args.decimals,
            overwrite = True,
        )  
    
    ##################################################
    ########## Visualizations ########################
    if conf is None:   
        conf = ConfBuilder.build(topol, 'ground_state', mass=args.mass)
    
    ##################################################
    ########## Generate ChimeraX script ##############    
    if args.visualize_cgrbp:
        conf.visualize_chimerax(base_fn, include_bps_triads=args.include_bps_triads)
    
    ##################################################
    ########## Generate PDB ##########################
    if args.gen_pdb and not args.visualize_cgrbp:
        conf.visualize_pdb(base_fn)
     
    ##################################################
    ########## Generate XYZ ##########################
    if args.gen_xyz:
        conf.visualize_xyz(base_fn)
    
    
        
    # ##################################################
    # ########## Generate ChimeraX script ##############
    # if args.visualize_cgrbp:
    #     if conf is None:
    #         conf = ConfBuilder.build(topol, 'ground_state', mass=args.mass)
    
    #     if args.composite_size > 1:
    #         bead_radius = args.composite_size*0.34*0.5
    #         bp_poses = dna_backmap(conf,verbose=True)
    #     else:
    #         bead_radius = 0
    #         bp_poses = conf.poses
    #     visualize_chimerax(base_fn, seq, args.composite_size, poses=bp_poses, start_id=topol.center_pos, bead_radius=bead_radius,include_bps_triads=args.include_bps_triads) 
        
    # ##################################################
    # ########## Generate PDB ##########################
    # if args.gen_pdb and not args.visualize_cgrbp:
    #     if conf is None:
    #         conf = ConfBuilder.build(topol, 'ground_state', mass=args.mass)
    #     visualize_pdb(base_fn, seq, poses=conf.poses_in_nm)
     
    # ##################################################
    # ########## Generate XYZ ##########################
    # if args.gen_xyz:
    #     visualize_xyz(base_fn, args.composite_size, poses=conf.poses, start_id=topol.center_pos)
    
    # ##################################################
    # ########## Save coefficients #####################
    # if args.safe_coeffs:
    #     if args.composite_size > 1:
    #         params.save_cg_coeffs(base_fn.with_name(base_fn.stem + '_closed') if args.closed else base_fn)
    #     else:
    #         params.save_coeffs(base_fn.with_name(base_fn.stem + '_closed') if args.closed else base_fn)
    
    ##################################################
    ########## Write sequence file ###################
    seqfn = base_fn.with_suffix('.seq')
    write_seqfile(seqfn,params.sequence,add_extension=True)
    