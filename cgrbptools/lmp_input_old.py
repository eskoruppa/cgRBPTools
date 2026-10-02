from __future__ import annotations

import sys, os
import argparse

import numpy as np
import scipy as sp
from pathlib import Path

from .PolyCG.polycg import gen_params, load_sequence, write_seqfile
# from .PolyCG.polycg import visualize_chimerax, visualize_pdb, visualize_xyz

from .core.topology import CGRBPTopology
from .core.conf_builder import ConfBuilder
from .core.matrix_methods import rescale_stiff
# from .io.backmap import dna_backmap

GEN_INPUT_CGNAP_SETNAME = 'curves_plus'

def parse_rescaling(resc_args, parser=None, ndim=6):
    def fail(msg):
        if parser is not None:
            parser.error(msg)
        raise argparse.ArgumentTypeError(msg)

    scale = [1.0] * ndim
    for group in resc_args:
        if len(group) < 2:
            fail("-sr/--stiff_resc requires: FACTOR DIM [DIM ...]")
        try:
            factor = float(group[0])
        except ValueError:
            fail(f"Invalid scale factor for rescaling '{group[0]}'")

        for d_str in group[1:]:
            try:
                d = int(d_str)
            except ValueError:
                fail(f"Invalid rescaling dimension '{d_str}'. Only the first argument can be a float factor; the rest must be integer dimensions.")

            if not (0 <= d < ndim):
                fail(f"Rescaling dimension {d} out of range (0..{ndim-1})")

            if scale[d] != 1.0 and scale[d] != factor:
                print(f"Warning: dimension {d} rescaled more than once; using last factor {factor} (overriding {scale[d]}).", file=sys.stderr)
            scale[d] = factor
    return scale


def format_rescaling(resc_args):
    """Reconstruct the -sr/--stiff_resc command line from the parsed argument groups."""
    return ' '.join('-sr ' + ' '.join(str(entry) for entry in group) for group in resc_args)


def write_rescaling_file(filename, resc_args, add_extension=True):
    """Write the rescaling command to a .rescale file. Returns the written filename."""
    filename = Path(filename)
    if add_extension and filename.suffix != '.rescale':
        filename = filename.with_suffix('.rescale')
    with open(filename, 'w') as f:
        f.write(format_rescaling(resc_args) + '\n')
    return filename


def limit_num_cores(num_cores):
    """Restrict the BLAS/OpenMP thread pools to num_cores threads.

    The *_NUM_THREADS environment variables are only read by these libraries when they are
    loaded, which has already happened by the time this module is imported (numpy comes in
    via the cgrbptools package). They are set here purely so that any subprocess inherits
    the same limit; threadpool_limits applies it to the already loaded pools.

    Returns the limiter, which the caller must keep alive: the previous limits are restored
    once it goes out of scope.

    threadpoolctl is imported lazily so that the module remains usable without it as long as
    no thread limit is requested.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        raise ImportError(
            "Limiting the number of threads requires the 'threadpoolctl' package "
            "(pip install threadpoolctl). Alternatively, set OMP_NUM_THREADS and "
            "OPENBLAS_NUM_THREADS in the environment before starting python."
        )

    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = f"{num_cores}"
    return threadpool_limits(limits=num_cores)

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
        type=str, default = 'cgnaplus', choices=['cgnaplus','md','crystal'],
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
        help='Number of decimal places for output formatting (default: 4)')
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
        '-nopart',
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
        '-sr', '--stiff_resc',
        action='append',
        nargs='+',
        default=[],
        metavar=('FACTOR', 'DIM'),
        help='Rescale stiffness of individual dimensions: -sr FACTOR DIM [DIM ...]. '
             'Repeat the flag for different factors, e.g. -sr 2.0 0 -sr 5.0 3 4 '
             'scales dim 0 by 2.0 and dims 3,4 by 5.0.')

    parser.add_argument(
        '-racg',
        '--rescale_after_cg',
        action='store_true',
        default=False,
        help="Apply the --stiff_resc rescaling to the coarse-grained stiffness matrix AFTER "
             "coarse-graining instead of to the base-pair-step stiffness beforehand "
             "(inside gen_params), which is the default. Because coarse-graining "
             "mixes degrees of freedom, the two orders generally give different results. "
             "The default (rescaling before coarse-graining) requires a PolyCG version that "
             "supports the 'dof_rescale' argument.")
    
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
        ),
        )
    parser.add_argument(
        '-dlk', 
        '--excess_link', 
        type=float, 
        default = 0.0,
        help='Set excess linking number for configuration generation. (default: 0.0)')

    parser.add_argument(
        '-be',
        '--bext',
        '--box_extend',
        '--box-extend',
        dest='box_extend',
        type=float,
        default=0.2,
        help='Relative extension of the simulation box beyond the bounds of the generated '
             'configuration. (default: 0.2)')

    parser.add_argument(
        '-np',
        '--num_procs',
        type=int,
        default=None,
        help='Limit the number of threads used by numpy/scipy (BLAS, OpenMP). '
             'If not set, the thread pools are left at their defaults.')
    args = parser.parse_args()

    ##################################################
    ########## Limit number of threads ###############
    # Only restrict the thread pools when the user explicitly asked for it. The reference is
    # kept for the lifetime of the run so the limit is not restored prematurely.
    if args.num_procs is not None:
        if args.num_procs < 1:
            parser.error(f'-np/--num_procs must be at least 1 (got {args.num_procs})')
        _threadpool_limiter = limit_num_cores(args.num_procs)

    ###################################################
    ########## Print passed setup #####################
    print('#' * 64)
    print('# Passed setup:')
    for key, value in sorted(vars(args).items()):
        print(f'#   {key:<24} = {value}')
    print('#' * 64)

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

    scale = parse_rescaling(args.stiff_resc, parser=parser, ndim=6)
    gen_params_kwargs = {} if args.rescale_after_cg else {'dof_rescale': scale}

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
        **gen_params_kwargs,
    )
    
    ###################################################
    ########## Determine output base filename #########
    if args.output_basename is None:
        if args.sequence_file is not None:
            base_fn = str(args.sequence_file).replace('.seq','')
            if args.composite_size > 0:
                base_fn += f'_cg{args.composite_size}'
        else:
            raise ValueError(f'No output name or sequence filename specified')
    else:
        base_fn = args.output_basename
    base_fn = Path(base_fn)

    ###################################################
    ########## Select shape and stiffness matrices ####
    if params.is_coarse_grained:
        stiff = params.cg_stiffmat
        shape = params.cg_shape_params
    else: 
        stiff = params.stiffmat
        shape = params.shape_params
            
    ##################################################
    ########## Rescale stiffnesses ###################
    # Only when -racg is set; by default the rescaling is already applied before
    # coarse-graining (see gen_params above).
    if args.rescale_after_cg:
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
    box_extension = args.box_extend

    conf = None
    gen_conf = args.configuration_method is not None
    if gen_conf:
        conf_method_key = args.configuration_method
        print(f"Generating configuration using method: {ConfBuilder.mapping_dict()[conf_method_key]} (input: {conf_method_key})")
        conf = ConfBuilder.build(topol, conf_method_key, mass=args.mass, excess_link=args.excess_link)
        box = conf.extended_bounds(box_extension,square_box=True)
    
        conf.write_datafile(
            base_fn,
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
        conf = ConfBuilder.build(topol, 'ground_state', mass=args.mass, excess_link=args.excess_link)
    
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
    #         conf = ConfBuilder.build(topol, 'ground_state', mass=args.mass, excess_link=args.excess_link)
    
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
    #         conf = ConfBuilder.build(topol, 'ground_state', mass=args.mass, excess_link=args.excess_link)
    #     visualize_pdb(base_fn, seq, poses=conf.poses_in_nm)
     
    # ##################################################
    # ########## Generate XYZ ##########################
    # if args.gen_xyz:
    #     visualize_xyz(base_fn, args.composite_size, poses=conf.poses, start_id=topol.center_pos)
    
    ##################################################
    ########## Save coefficients #####################
    if args.safe_coeffs:
        resc_stiffmat = topol.get_stiffness_matrix()
        resc_gs = topol.get_groundstate()

        if sp.sparse.issparse(resc_stiffmat):
            sp.sparse.save_npz(base_fn.with_name(base_fn.stem + '_stiff.npz'),resc_stiffmat)
        else:
            np.save(base_fn.with_name(base_fn.stem + '_stiff.npy'),resc_stiffmat)
        np.save(base_fn.with_name(base_fn.stem + '_gs.npy'),resc_gs)

    ##################################################
    ########## Write sequence file ###################
    seqfn = base_fn.with_suffix('.seq')
    write_seqfile(seqfn,params.sequence,add_extension=True)

    ##################################################
    ########## Write rescaling file ##################
    # Only written if rescaling was requested via -sr/--stiff_resc.
    if args.stiff_resc:
        write_rescaling_file(base_fn.with_suffix('.rescale'), args.stiff_resc)


    