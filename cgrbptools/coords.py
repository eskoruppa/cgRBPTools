from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt
from .SO3 import so3



def se3_deforms(se3: np.ndarray, gs: np.ndarray, subtract: bool = True) -> np.ndarray:
    
    if gs.shape[-1] != 6 and (gs.shape[-1] != 4 or gs.shape[-2] != 4):
        raise ValueError(f'Invalid dimension of gs. Elements of gs should be either coordinates (6 components) or elements of se3 (4x4 matrices).')
    
    if subtract:
        if gs.shape[-1] == 4:
            X0 = se3_mats2coords(gs)
        else:
            X0 = gs
        
        
    else:
        if gs.shape[-1] == 6:
            s = so3.se3_
            
            
        

    
    


if __name__ == "__main__":
    
    from .parse_custom import LMPCustom
    
    fn = sys.argv[1]
    lmp = LMPCustom(fn)
    taus = lmp.se3(unwrap=True)
    
    taus = taus[-1,-1]
    
    