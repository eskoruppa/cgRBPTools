from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt
from .SO3 import so3


def se3_coords_deforms(taus: np.ndarray, groundstate: np.ndarray, subtract: bool = True) -> np.ndarray:
    
    if taus.shape[-3]-1 != groundstate.shape[0]:
        raise ValueError(f'Incompatible dimensions of se3 and groundstate.')
    
    if groundstate.shape[-1] != 6 and (groundstate.shape[-1] != 4 or groundstate.shape[-2] != 4):
        raise ValueError(f'Invalid dimension of gs. Elements of gs should be either coordinates (6 components) or elements of se3 (4x4 matrices).')
        
    if subtract:
        if groundstate.shape[-1] == 4:
            X0 = so3.se3_rotmats2eulers(groundstate)
        else:
            X0 = groundstate
        Xd = np.zeros((taus.shape[0],taus.shape[1]-1,6),dtype=np.double)
        for s in range(len(taus)):
            for i in range(len(taus[0])-1):
                Xd[s,i] = so3.se3_rotmat2euler(so3.se3_inverse(taus[s,i]) @ taus[s,i+1]) - X0[i]
        return Xd

    else:
        if groundstate.shape[-1] == 6:
            s = so3.se3_eulers2rotmats(groundstate)
        else:
            s = groundstate
            
        Yd = np.zeros((taus.shape[0],taus.shape[1]-1,6),dtype=np.double)
        for s in range(len(taus)):
            for i in range(len(taus[0])-1):
                Yd[s,i] = so3.se3_rotmat2euler(so3.se3_inverse(s[i]) @ so3.se3_inverse(taus[s,i]) @ taus[s,i+1] )
        return Yd
    
    
def se3_coords_mean(taus: np.ndarray) -> np.ndarray:
    X = np.zeros((taus.shape[0],taus.shape[1]-1,6),dtype=np.double)
    for s in range(len(taus)):
        for i in range(len(taus[0])-1):
            X[s,i] = so3.se3_rotmat2euler(so3.se3_inverse(taus[s,i]) @ taus[s,i+1])
    Xs = np.mean(X,axis=0)
    return Xs


def eval_stiffmat(Xd: np.ndarray):
    if len(Xd.shape) == 3: 
        Xd = Xd.reshape(len(Xd),len(Xd[0])*len(Xd[0,0]))
    cov = np.cov(Xd.T)
    stiff = np.linalg.inv(cov)
    return stiff
    
            
            
        

    
    


if __name__ == "__main__":
    
    from .parse_custom import LMPCustom
    np.set_printoptions(precision=2, suppress=True,linewidth=200)
    
    
    
    fn = sys.argv[1]
    lmp = LMPCustom(fn)
    taus = lmp.se3(unwrap=True)
    
    Xs = se3_coords_mean(taus)
    
    # gsfn = fn.replace('.custom','_gs.npy')
    gsfn = fn.split('_ts')[0] + '_gs.npy'
    gs = np.load(gsfn)
    
    # stifffn = fn.replace('.custom','_stiff.npy')
    stifffn = fn.split('_ts')[0] + '_stiff.npy'
    stiff_control = np.load(stifffn)
    
    
    # print(Xs[-10:])
    # print(gs[-10:])
    # sys.exit()
    
    Xd = se3_coords_deforms(taus,gs,subtract=True)
    
    X = se3_coords_deforms(taus,np.zeros(gs.shape),subtract=True)
    print(X[-100,-10:])
    print(Xd[-100,-10:])

    # print(Xd.shape)
    
    
    # k = 1./(np.dot(Xd[:,50,-1],Xd[:,50,-1])/len(Xd))
    # print(k)
    # sys.exit()
    
    stiff = eval_stiffmat(Xd)    
    
    nblock = 18
    for i in range(len(stiff)//nblock):
        print('\n\n##########################################\n')
        print('reference')
        print(stiff_control[i*nblock:(i+1)*nblock,i*nblock:(i+1)*nblock])
        
        print('sampled')
        print(stiff[i*nblock:(i+1)*nblock,i*nblock:(i+1)*nblock])
        
        print('difference')
        print(stiff[i*nblock:(i+1)*nblock,i*nblock:(i+1)*nblock]-stiff_control[i*nblock:(i+1)*nblock,i*nblock:(i+1)*nblock])
    
    print(f'\n\n {len(taus)} snapshots found')