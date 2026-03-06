from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt
from .SO3 import so3

    
def se3_mats2coords(se3: np.ndarray) -> np.ndarray:
    """
    Convert SE(3) matrices to 6D coordinates via so3.se3_rotmat2euler.

    Accepts:
      - a single matrix: shape (M, M) (typically (4,4) or (3,4))
      - a batch: shape (..., M, M)

    Returns:
      - shape (6,) for a single matrix
      - shape (..., 6) for a batch
    """
    se3 = np.asarray(se3)

    if se3.ndim == 2:
        return so3.se3_rotmat2euler(se3)

    lead_shape = se3.shape[:-2]
    n = int(np.prod(lead_shape))
    se3_flat = se3.reshape((n,) + se3.shape[-2:])

    try:
        X_flat = so3.se3_rotmat2euler(se3_flat)
        X_flat = np.asarray(X_flat, dtype=np.float64)
        return X_flat.reshape(lead_shape + (6,))
    except Exception:
        X_flat = np.empty((n, 6), dtype=np.float64)
        for i in range(n):
            X_flat[i] = so3.se3_rotmat2euler(se3_flat[i])
        return X_flat.reshape(lead_shape + (6,))
    
    
def se3_coords2mats(X: np.ndarray) -> np.ndarray:
    """
    Convert 6D SE(3) coordinates to SE(3) matrices via so3.se3_euler2rotmat.

    Accepts:
      - a single coordinate vector: shape (6,)
      - a batch: shape (..., 6)

    Returns:
      - a single SE(3) matrix for (6,) input
      - a batch of SE(3) matrices for (..., 6) input, with shape (..., M, M)
        where M is whatever so3.se3_euler2rotmat returns (typically 4x4 or 3x4).
    """
    X = np.asarray(X)
    
    if X.ndim == 1:
        if X.shape[0] != 6:
            raise ValueError(f"Expected shape (6,), got {X.shape}.")
        return so3.se3_euler2rotmat(X)

    if X.shape[-1] != 6:
        raise ValueError(f"Expected last dimension to be 6, got {X.shape}.")

    lead_shape = X.shape[:-1]
    n = int(np.prod(lead_shape))
    X_flat = X.reshape((n, 6))

    try:
        se3_flat = so3.se3_euler2rotmat(X_flat)
        se3_flat = np.asarray(se3_flat)
        return se3_flat.reshape(lead_shape + se3_flat.shape[-2:])
    except Exception:
        first = np.asarray(so3.se3_euler2rotmat(X_flat[0]))
        out = np.empty((n,) + first.shape, dtype=first.dtype)
        out[0] = first
        for i in range(1, n):
            out[i] = so3.se3_euler2rotmat(X_flat[i])
        return out.reshape(lead_shape + first.shape)


def eval_deforms(triads,S,pos,vs):
    Phis = np.zeros((len(triads),len(triads[0])-1,3))
    for s in range(len(Phis)):
        for i in range(len(Phis[0])):
            Phis[s,i] = so3.rotmat2euler(S.T @ triads[s,i].T @ triads[s,i+1]) 
    
    d = np.zeros((len(pos),len(pos[0])-1,3))
    for s in range(len(pos)):
        for i in range(len(Phis[0])):
            d[s,i] = S.T @ (triads[s,i].T @ (pos[s,i+1]-pos[s,i]) - vs)
            
    sh = list(Phis.shape)
    sh[-1] = 6
    Y = np.zeros(sh)  
    Y[:,:,:3] = Phis
    Y[:,:,3:] = d
    return Y

def eval_deforms_add(triads,srot,pos,strans):
    Phis = np.zeros((len(triads),len(triads[0])-1,3))
    for s in range(len(Phis)):
        for i in range(len(Phis[0])):
            Phis[s,i] = so3.rotmat2euler(triads[s,i].T @ triads[s,i+1]) - srot 
    
    d = np.zeros((len(pos),len(pos[0])-1,3))
    for s in range(len(pos)):
        for i in range(len(Phis[0])):
            d[s,i] = (triads[s,i].T @ (pos[s,i+1]-pos[s,i]) - strans)
    
    # x = np.arange(len(d))
    # # for s in range(len(d[0])):
    # plt.plot(x,pos[:,0,0])
    # plt.plot(x,pos[:,1,0])
    # plt.show()
           
    sh = list(Phis.shape)
    sh[-1] = 6
    Y = np.zeros(sh)  
    Y[:,:,:3] = Phis
    Y[:,:,3:] = d
    return Y
    
def evalstiff(Y):
    cov = np.cov(Y.T)
    stiff = np.linalg.inv(cov)
    return stiff



def eval_energy(fn,savefn=None,dofs=1,ylabel=None):
    data = np.loadtxt(fn, skiprows=2)

    # print(data.shape)
    # data = data[200000:500000]
    
    
    time = data[:,0]
    epot = data[:,1]
    
    # time = time[epot/(N*6) < 3]
    # epot = epot[epot/(N*6) < 3]
    
    print(f'<E> = {np.mean(epot)/(dofs)}')
    
    # --- 2. Define Figure Size in Inches ---
    # Matplotlib uses inches for figsize, so convert cm to inches (1 inch = 2.54 cm)
    width_cm = 8.6
    height_cm = 5.0

    plt.figure(figsize=(width_cm / 2.54,  height_cm / 2.54))

    plt.plot(time, epot/(dofs), label='Potential Energy',color='black',lw=0.5)

    # --- 4. Add Labels, Title, and Grid ---
    plt.xlabel('Time')
    if ylabel is None:
        plt.ylabel('Potential Energy')
    else:
        plt.ylabel(ylabel)
        
    # plt.title('Potential Energy vs. Time')
    plt.grid(True, linestyle='--', alpha=0.7) # Add a grid for readability

    # Optional: Add a legend if you have multiple lines
    # plt.legend()

    plt.tight_layout() # Adjusts plot parameters for a tight layout

    if savefn is not None:
        plt.savefig(savefn+'.png', dpi=300) # dpi (dots per inch) for higher resolution PNG
        plt.savefig(savefn+'.pdf')
    else:
        plt.show()
    plt.close()


# =========================
# Numerical constants
# =========================

import math
from .SO3.so3.pyConDec.pycondec import cond_jit

DEF_EULER_EPSILON = 1e-12
DEF_EULER_CLOSE_TO_ONE = 1.0 - 1e-10
DEF_EULER_CLOSE_TO_MINUS_ONE = -1.0 + 1e-10

DEF_EULER_SERIES_SMALL = 1e-4
DEF_THETA_SCALE_SMALL = 1e-6
DEF_AXIS_NORM_EPS = 1e-15
DEF_AXIS_COMP_EPS = 1e-15


@cond_jit(nopython=True, cache=True)
def euler2rotmat(Omega: np.ndarray) -> np.ndarray:
    Om = math.sqrt(Omega[0]*Omega[0] + Omega[1]*Omega[1] + Omega[2]*Omega[2])
    if Om < DEF_EULER_EPSILON:
        return np.eye(3, dtype=np.double)

    if Om < DEF_EULER_SERIES_SMALL:
        Om2 = Om * Om
        Om4 = Om2 * Om2
        A = 1.0 - Om2 / 6.0 + Om4 / 120.0
        B = 0.5 - Om2 / 24.0 + Om4 / 720.0
    else:
        A = math.sin(Om) / Om
        B = (1.0 - math.cos(Om)) / (Om * Om)

    x, y, z = Omega[0], Omega[1], Omega[2]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z

    R = np.empty((3, 3), dtype=np.double)
    R[0, 0] = 1.0 - B * (yy + zz)
    R[1, 1] = 1.0 - B * (xx + zz)
    R[2, 2] = 1.0 - B * (xx + yy)

    R[0, 1] = B * xy - A * z
    R[1, 0] = B * xy + A * z
    R[0, 2] = B * xz + A * y
    R[2, 0] = B * xz - A * y
    R[1, 2] = B * yz - A * x
    R[2, 1] = B * yz + A * x
    return R


@cond_jit(nopython=True, cache=True)
def rotmat2euler(R: np.ndarray) -> np.ndarray:
    out = np.empty(3, dtype=np.double)

    val = 0.5 * ((R[0, 0] + R[1, 1] + R[2, 2]) - 1.0)
    if val > 1.0:
        val = 1.0
    elif val < -1.0:
        val = -1.0

    if val > DEF_EULER_CLOSE_TO_ONE:
        out[0] = 0.0; out[1] = 0.0; out[2] = 0.0
        return out

    if val < DEF_EULER_CLOSE_TO_MINUS_ONE:
        r00, r11, r22 = R[0, 0], R[1, 1], R[2, 2]

        # pick dominant diagonal; compute axis; deterministic fallback if denom tiny
        if (r00 >= r11) and (r00 >= r22):
            t = 0.5 * (r00 + 1.0);  t = 0.0 if t < 0.0 else t
            ax = math.sqrt(t)
            if ax < DEF_AXIS_COMP_EPS:
                out[0] = 0.0; out[1] = math.pi; out[2] = 0.0
                return out
            ay = R[0, 1] / (2.0 * ax)
            az = R[0, 2] / (2.0 * ax)
        elif r11 >= r22:
            t = 0.5 * (r11 + 1.0);  t = 0.0 if t < 0.0 else t
            ay = math.sqrt(t)
            if ay < DEF_AXIS_COMP_EPS:
                out[0] = math.pi; out[1] = 0.0; out[2] = 0.0
                return out
            ax = R[0, 1] / (2.0 * ay)
            az = R[1, 2] / (2.0 * ay)
        else:
            t = 0.5 * (r22 + 1.0);  t = 0.0 if t < 0.0 else t
            az = math.sqrt(t)
            if az < DEF_AXIS_COMP_EPS:
                out[0] = math.pi; out[1] = 0.0; out[2] = 0.0
                return out
            ax = R[0, 2] / (2.0 * az)
            ay = R[1, 2] / (2.0 * az)

        nrm = math.sqrt(ax*ax + ay*ay + az*az)
        if nrm < DEF_AXIS_NORM_EPS:
            out[0] = math.pi; out[1] = 0.0; out[2] = 0.0
            return out

        s = math.pi / nrm
        out[0] = ax * s; out[1] = ay * s; out[2] = az * s
        return out

    th = math.acos(val)
    vx = R[2, 1] - R[1, 2]
    vy = R[0, 2] - R[2, 0]
    vz = R[1, 0] - R[0, 1]

    if th < DEF_THETA_SCALE_SMALL:
        th2 = th * th
        scale = 0.5 + th2 / 12.0
    else:
        scale = 0.5 * th / math.sin(th)

    out[0] = scale * vx
    out[1] = scale * vy
    out[2] = scale * vz
    return out



if __name__ == "__main__":
    
    from .parse_custom import LoadCustom
    
    fn = sys.argv[1]
    lmp = LoadCustom(fn)
    taus = lmp.se3(unwrap=True)
    
    # X = se3_mats2coords(taus)
    
    # taus1 = se3_coords2mats(X)
    
    taus = taus[-1,-1]
    print((taus-se3_coords2mats(se3_mats2coords(taus))).sum())
    
    print((taus-so3.se3_euler2rotmat(so3.se3_rotmat2euler(taus))).sum())
    
    # print(taus)
    # print(so3.se3_euler2rotmat(so3.se3_rotmat2euler(taus)))
    
    
    taus = taus[:3,:3]
    print((taus-euler2rotmat(rotmat2euler(taus))).sum())
    
    
    tau = taus
    
    X1 = so3.rotmat2euler(tau)
    X2 = rotmat2euler(tau)
    
    
    tau1 = so3.euler2rotmat(X1)
    tau2 = euler2rotmat(X2)
    
    print((X1-X2).sum())
    print((tau1-tau2).sum())
    
    
    
    
    
    
    
    # fn = sys.argv[1]
    # data = parse_custom(fn)
    
    # # N=8
    
    # # dofs = (N-1)*6
    # # Efn = "../Runs/dump1.epot"
    # # savefn = 'Epot'
    # # eval_energy(Efn,savefn=savefn,dofs=dofs)
    
    # # dofs = N*3
    # # Efn = "../Runs/dump1.krot"
    # # savefn = 'Krot'
    # # ylabel = 'Rot Kinetic Energy'
    # # eval_energy(Efn,savefn=savefn,dofs=dofs,ylabel=ylabel)
    
    
    # print(data['args'])

    # print(data['data'].shape)

    # pos   = data['data'][:,:,1:4]
    # quats = data['data'][:,:,4:8]
    
    # print(pos.shape)
    
    # # excl = 3
    # # pos = pos[:,excl:-excl]
    # # print(pos.shape)
    
    # # print(pos.shape)
    # # quats = quats[:,excl:-excl]
    # # print(pos.shape)
    
    
    # triads = quats2rotmats(quats)
    # print(triads.shape)
    
    # Phi0 = np.array([0.0, 0.0, 0.0])
    # vs   = np.array([0.0, 0.0, 1.0])
    
    # S = so3.euler2rotmat(Phi0)
    
    # # Y = eval_deforms(triads,S,pos,vs)
    # Y = eval_deforms_add(triads,Phi0,pos,vs)
    # print(Y.shape)
    
    # # for i in range(len(Y)):
    # #     print(Y[i,0,3:])
    
    # tY = Y.reshape((len(Y)*len(Y[0]),len(Y[0,0])))
    # print(tY.shape)
    
    # print(np.mean(tY,axis=0))
    # print(np.mean(tY[:,-1]))
    # stiff = evalstiff(tY)
    
    # print(stiff)