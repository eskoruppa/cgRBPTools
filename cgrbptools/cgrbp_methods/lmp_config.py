from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Any
import numpy as np

# import numpy as np
# import scipy as sp
# from scipy.sparse import spmatrix
# import hashlib
# from typing import Any, Callable, Dict, List, Tuple

# from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
# from typing import ClassVar
# from functools import cached_property

from ..SO3 import so3
from .lmp_topol import CGRBPTopology



LMP_RBP_DIMS = 6

##################################################################################################################
##################################################################################################################
# Build Molecule Configuration

@dataclass
class CGRBPConfig:
    poses: np.ndarray
    mass: float
    
    def __post_init__(self):
        self.nbp = len(self.poses)
    
    @property
    def positions(self):
        return self.poses[:,:3,3]

    @property
    def triads(self):
        return self.poses[:,:3,:3]
    
    @property
    def quaternions(self):
        return so3.mats2quats(self.poses[:,:3,:3])
    
    @property
    def bounds(self):
        return np.array([np.min(self.positions,axis=0),np.max(self.positions,axis=0)]).T
    
    def dimsize(self, bounds:np.ndarray = None):
        if bounds is None:
            bounds = self.bounds
        return bounds[:,1] - bounds[:,0]
    
    def extended_bounds(self, margin_fraction: float, square_box: bool = False):
        bounds = self.bounds
        dimsize = self.dimsize(bounds=bounds)
        margin = np.max(margin_fraction * dimsize)
        bounds[:,0] -= margin
        bounds[:,1] += margin
        if square_box:
            dim = self.dimsize(bounds=bounds)
            mdim = np.max(dim)
            for i in range(len(bounds)):
                if dim[i] < mdim:
                    ext = (mdim-dim[i])*0.5
                    bounds[i,0] -= ext
                    bounds[i,1] += ext
        return bounds
    
    def atom_strings(self): 
        strs = []
        for i,pos in enumerate(self.positions):
            pstr = f'{i+1} 1 {pos[0]} {pos[1]} {pos[2]} 1 1 1'
            strs.append(pstr)
        return(strs)
    
    def ellipsoid_strings(self):
        strs = []
        for i,quat in enumerate(self.quaternions):
            pstr = f'{i+1} 1 1.0000001 0.9999999 {quat[0]} {quat[1]} {quat[2]} {quat[3]}'
            strs.append(pstr)
        return(strs)
    
    def mass_strings(self):
        return [f'{1} {self.mass}']


    
    
            


class CGRBPConfigBuilder:
    def __init__(self, topology: CGRBPTopology):
        self.topology = topology

    @classmethod
    def straight_with_twist(self, topology: CGRBPTopology, mass: float, dir: np.ndarray = np.array([0,0,1])) -> CGRBPConfig:
        nbp = topology.nbps + 1
        poses = np.zeros((nbp,4,4),dtype=float)
        poses[0] = np.eye(4)
        
        for i,X0 in enumerate(topology.groundstate):
            Xstr = np.zeros(X0.shape)
            Xstr[2] = X0[2]
            Xstr[5] = X0[5]
            
            g = so3.se3_euler2rotmat(Xstr)
            poses[i+1] = poses[i] @ g
        return CGRBPConfig(poses,mass)

        

    # def ground_state(self) -> RBPConfiguration:
    #     ...

    # def import_config(self, positions, orientations) -> RBPConfiguration:
    #     ...