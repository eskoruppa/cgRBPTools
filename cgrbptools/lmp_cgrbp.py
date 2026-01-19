from __future__ import annotations

import sys, os
import argparse
import numpy as np
import scipy as sp
from typing import Any, Callable, Dict, List, Tuple

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar
from functools import cached_property

from .cgrbp_methods.lmp_topol import CGRBPTopology
from .cgrbp_methods.lmp_config import CGRBPConfigBuilder, CGRBPConfig
from .cgrbp_methods.lmp_unit_conversion import RescaleUnits

# # load models
# from .PolyCG.polycg import cgnaplus_bps_params
# from .PolyCG.polycg import GenStiffness
# # load partial stiffness generation
# from .PolyCG.polycg import partial_stiff
# # load coarse graining methods
# from .cg import coarse_grain
# load sequence from sequence file
from .PolyCG.polycg import load_sequence
# # write sequence file
# from .utils.seq import write_seqfile
# load so3 
from .SO3 import so3
# load visualization methods
from .PolyCG.polycg import cgvisual

from .gen_params import gen_params


def gen_database_file(
    filename: str,
    topology: CGRBPTopology,
    add_extension: bool = True,
    seq: str | None = None,
    composite_size: int = 1,
    start_id: int | None = None,
    end_id: int | None = None 
    ) -> None:
    
    ext = '.db'
    if add_extension and os.path.splitext(filename)[-1].lower() != ext:
        filename += ext
    
    with open(filename,'w') as f:
        
        f.write(f'{topology.nbps + 1} triads\n\n')
        
        if seq is not None:
            if end_id is not None:
                seq = seq[:end_id]
            if start_id is not None:
                seq = seq[start_id:]
            
            pseqs = [seq[ii*composite_size:(ii+1)*composite_size] for ii in range(topology.nbps+1)]
            pseqs[-1] = pseqs[-1][:1]
            
            f.write(f'\nSeqs\n\n')
            for i,pseq in enumerate(pseqs):
                f.write(f'{i+1} {pseq.upper()}\n')
        
        if len(topology.bondtypes) > 0:
            f.write(f'\nBond Coeffs\n\n')
            for bondtype in topology.bondtypes:
                f.write(f'{bondtype.to_str(hybrid=False)}\n')
            f.write('\n')

        if len(topology.angletypes) > 0:
            f.write(f'\nAngle Coeffs\n\n')
            for angletype in topology.angletypes:
                f.write(f'{angletype.to_str(hybrid=False)}\n')
            f.write('\n')

        if len(topology.dihedraltypes) > 0:
            f.write(f'\nDihedral Coeffs\n\n')
            for dihedraltype in topology.dihedraltypes:
                f.write(f'{dihedraltype.to_str(hybrid=False)}\n')
            f.write('\n')



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
            atomstrs = conf.atom_strings()
            for atomstr in atomstrs:
                f.write(f'{atomstr}\n')
            f.write('\n')  
            
        if config.nbp > 0:
            f.write(f'\nEllipsoids\n\n')
            ellipsstrs = conf.ellipsoid_strings()
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
            
            

def matrix_copy(A):
    if sp.sparse.issparse(A):
        return A.copy()
    else:
        return np.array(A, copy=True)

def rescale_kth(stiff, k: int, factor: float):
    if not (0 <= k < 6):
        raise ValueError("k must be an integer in the range 0..5")
    n = stiff.shape[0]
    d = np.ones(n)
    d[k::6] = factor
    if sp.sparse.issparse(stiff):
        D = sp.sparse.diags(
            d,
            format=stiff.format if hasattr(stiff, "format") else "csr"
        )
        return D @ stiff @ D
    else:
        return (d[:, None] * stiff) * d[None, :]

def rescale_stiff(stiff,factor,entries=[]):
    if len(entries) == 0:
        rescaled_stiff = stiff * factor**2
        return rescaled_stiff
    else:
        rescaled_stiff = matrix_copy(stiff)
        for k in entries:
            rescaled_stiff = rescale_kth(rescaled_stiff,k,factor)
        return rescaled_stiff


def is_positive_definite(A, tol=0.0):
    A = np.asarray(A)

    # Must be square
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False

    # Must be (numerically) symmetric
    if not np.allclose(A, A.T, atol=tol):
        return False

    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False



if __name__ == "__main__":
    
    np.set_printoptions(precision=1,linewidth=300,suppress=True)
    
    parser = argparse.ArgumentParser(description="Generate PolyMC input files")
    parser.add_argument('-mdl',     '--model',              type=str, default = 'cgnaplus', choices=['cgnaplus','lankas','olson'])
    parser.add_argument('-cg',      '--composite_size',     type=int, default = 1)
    parser.add_argument('-seqfn',   '--sequence_file',      type=str, default = None)
    parser.add_argument('-seq',     '--sequence',           type=str, default = None)
    parser.add_argument('-cr',      '--coupling_range',     type=int, default = 4)
    parser.add_argument('-dec',     '--decimals',           type=int, default = 2)
    parser.add_argument('-m',       '--mass',               type=float, default = 1.0)
    parser.add_argument('-closed',  '--closed',             action='store_true') 
    parser.add_argument('-nc',      '--no_crop',            action='store_true') 
    parser.add_argument('-np',      '--no_partial',         action='store_true') 
    parser.add_argument('-stiff',   '--safe_stiffmat',      action='store_true') 
    parser.add_argument('-gs',      '--safe_groundstate',   action='store_true') 
    parser.add_argument('-sid',     '--start_id',           type=int, default=0) 
    parser.add_argument('-eid',     '--end_id',             type=int, default=None) 
    parser.add_argument('-o',       '--output_basename',    type=str, default = None, required=False)
    # parser.add_argument('-nv',      '--no_visualization',   action='store_true') 
    
    # parser.add_argument(
    #     '-seqfn', '--sequence_file',
    #     type=str,
    #     nargs='+',
    #     default=None,
    #     help="One or more sequence file paths"
    # )
    parser.add_argument(
        '-fene',
        '--bond_fene_coeffs',
        nargs=3,
        type=float,
        metavar=("K", "Rc", "R0"),
        default=None,
        help="Coefficients for bond style rbpfene",
    )
    # parser.add_argument('-xyz',     '--gen_xyz',            action='store_true') 
    # parser.add_argument('-pdb',     '--gen_pdb',            action='store_true') 
     
    args = parser.parse_args()
    
    model           = args.model
    composite_size  = args.composite_size
    coupling_range  = args.coupling_range
    decimals        = args.decimals
    mass            = args.mass
    seqfn           = args.sequence_file
    seq             = args.sequence
    closed          = args.closed
    allow_crop      = not args.no_crop
    safe_stiffmat   = args.safe_stiffmat
    safe_groundstate = args.safe_groundstate
        
    allow_partial   = not args.no_partial
    start_id        = args.start_id
    end_id          = args.end_id
    
    bond_fene_coeffs = None
    if args.bond_fene_coeffs is not None:
        bond_fene_coeffs = np.array(args.bond_fene_coeffs, dtype=float)
    
    if args.output_basename is not None:
        outname = args.output_basename
    else:
        if seqfn is not None:
            outname = seqfn.replace('.seq','')
        else:
            raise ValueError(f'No output name or sequence filename specified')
    
    ##################################################################################################################
    
    if seq is None:
        if seqfn is None:
            raise ValueError(f'Requires either a sequence (-seq) or a sequence file (-seqfn)')
        seq = load_sequence(seqfn)
       
    sparse = True
    cgnap_setname = 'curves_plus'
    
    params = gen_params(
        model , 
        seq,
        composite_size,
        closed=closed,
        sparse=sparse,
        start_id=start_id,
        end_id=end_id,
        allow_partial=allow_partial,
        allow_crop=allow_crop,
        cgnap_setname = cgnap_setname
    )
    
    for key in params:
        print(key)
    
    print(params['gs'].shape)
    print(params['stiff'].shape)
    print(params['seq'])
    
    if 'cg_stiff' in params:
        stiff   = params['cg_stiff']
        gs      = params['cg_gs']
    else: 
        stiff   = params['stiff']
        gs      = params['gs']
    
    # stiff = params['stiff'][60:660,60:660].toarray()
    # cgstiff = params['stiff'][6:66,6:66].toarray()
        
    # ##########################################
    # posdef = is_positive_definite(stiff)
    # print(f'M posdef   : {posdef}')
    # posdef = is_positive_definite(cgstiff)
    # print(f'Mxg posdef : {posdef}')
    
    # for k in range(50):
    #     nkpd = 0
    #     for i in range(params['stiff'].shape[0]//6 - k):
    #         pd = is_positive_definite(params['stiff'][i*6:(i+1+k)*6,i*6:(i+1+k)*6].toarray())
    #         if not pd:
    #             nkpd += 1
    #             # break
    #             # print(f'block {i} not positive definite')
    #     print(f'{k}-block, {nkpd} not positive definite')
    #     # if kpd:
    #     #     print(f'{k}-block are positive definite')
    #     # else:
    #     #     print(f'{k}-block are not positive definite')
    # sys.exit()  
    
    ##########################################
    
    mean_disc_len = np.mean(gs[:,5])
    length_rescale_factor = 1./3.4
        
    rescale = RescaleUnits(length_factor=length_rescale_factor)
    gs,stiff = rescale.rescale_model(gs,stiff)
    
    ##################################################
    ########## RESCALING #############################
    rot_rescale = np.sqrt(0.667)
    rise_rescale  = np.sqrt(0.16)
    
    stiff = rescale_stiff(stiff,rot_rescale,entries=[0,1,2])
    stiff = rescale_stiff(stiff,rise_rescale,entries=[5])
    ########## RESCALING #############################
    ##################################################
    
    # gs[:,0] = 0
    # gs[:,1] = 0
    # gs[:,3] = 0
    # gs[:,4] = 0
    
    check_existing_types = False
    
    # topol = CGRBPTopology(gs,stiff,coupling_range=coupling_range,decimals=decimals,extra_bond=bond_fene_coeffs)
    topol = CGRBPTopology(gs,stiff,coupling_range=coupling_range,decimals=decimals,extra_bond=bond_fene_coeffs,check_existing_types=check_existing_types)
    
    conf = CGRBPConfigBuilder.straight_with_twist(topol,mass=mass)
    box = conf.extended_bounds(0.2,square_box=True)
    
    gen_datafile(outname,topol,conf,hybrid=False,box=box)
    gen_database_file(outname,topol,seq=seq,composite_size=composite_size,start_id=start_id,end_id=end_id)
        
    if safe_stiffmat:
        stifffn = outname + '_stiff.npy'
        np.save(stifffn,stiff.toarray())
    
    if safe_groundstate:
        gsfn = outname + '_gs.npy'
        np.save(gsfn,gs)