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










# def se3_deforms(se3: np.ndarray, gs: np.ndarray, subtract: bool = True) -> np.ndarray:
    
#     if gs.shape[-1] != 6 and (gs.shape[-1] != 4 or gs.shape[-2] != 4):
#         raise ValueError(f'Invalid dimension of gs. Elements of gs should be either coordinates (6 components) or elements of se3 (4x4 matrices).')
    
#     if subtract:
#         if gs.shape[-1] == 4:
#             X0 = so3.se3_coords(gs)
#         else:
#             X0 = gs
        
        
#     else:
#         if gs.shape[-1] == 6:
            
        
        
    
#     if gs.shape[-1] == 6:
        
    
    
    


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

# True zero-rotation cutoff (do NOT use for numerical stabilization)
DEF_EULER_EPSILON = 1e-12

# Thresholds for detecting angles close to 0 and close to pi via val = 0.5*(tr(R)-1)
DEF_EULER_CLOSE_TO_ONE = 1.0 - 1e-10
DEF_EULER_CLOSE_TO_MINUS_ONE = -1.0 + 1e-10

# Series / stability thresholds (kept outside for clarity + reproducibility)
DEF_EULER_SERIES_SMALL = 1e-4      # for sin(x)/x and (1-cos(x))/x^2
DEF_THETA_SCALE_SMALL = 1e-6       # for theta/(2 sin theta) scaling in log map
DEF_AXIS_NORM_EPS = 1e-15          # avoid division by ~0 when normalizing axis
DEF_AXIS_COMP_EPS = 1e-15          # avoid division by ~0 in pi-axis extraction


# =========================
# so(3) → SO(3)
# =========================

# @cond_jit(nopython=True, cache=True)
def euler2rotmat(Omega: np.ndarray) -> np.ndarray:
    """
    Euler-Rodrigues / exponential map from so(3) to SO(3).

    Numerically stable for small ||Omega|| by using series expansions for
    sin(x)/x and (1-cos(x))/x^2.
    """
    Om = np.linalg.norm(Omega)

    # Identity for (near) zero rotation
    if Om < DEF_EULER_EPSILON:
        return np.eye(3, dtype=np.double)

    # Stable evaluation of:
    #   A = sin(Om)/Om
    #   B = (1-cos(Om))/Om^2
    if Om < DEF_EULER_SERIES_SMALL:
        Om2 = Om * Om
        Om4 = Om2 * Om2
        # sin(x)/x = 1 - x^2/6 + x^4/120
        A = 1.0 - Om2 / 6.0 + Om4 / 120.0
        # (1-cos(x))/x^2 = 1/2 - x^2/24 + x^4/720
        B = 0.5 - Om2 / 24.0 + Om4 / 720.0
    else:
        A = np.sin(Om) / Om
        B = (1.0 - np.cos(Om)) / (Om * Om)

    x = Omega[0]
    y = Omega[1]
    z = Omega[2]

    # Rodrigues formula: R = I + A*[w]_x + B*[w]_x^2 (expanded)
    R = np.empty((3, 3), dtype=np.double)

    xx = x * x
    yy = y * y
    zz = z * z

    xy = x * y
    xz = x * z
    yz = y * z

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


# =========================
# SO(3) → so(3)
# =========================

# @cond_jit(nopython=True, cache=True)
def rotmat2euler(R: np.ndarray) -> np.ndarray:
    """
    Inverse of Euler Rodriguez Formula (log map SO(3)->so(3)),
    returning the rotation vector Omega (3,).

    Robustness features (as in the earlier version):
      - clamp trace-derived val into [-1, 1] to avoid arccos NaNs
      - stable small-angle handling for th/(2 sin th)
      - robust angle≈pi handling using "largest diagonal" axis extraction
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    val = 0.5 * (tr - 1.0)

    # Clamp to valid arccos domain
    if val > 1.0:
        val = 1.0
    elif val < -1.0:
        val = -1.0

    # angle ~ 0
    if val > DEF_EULER_CLOSE_TO_ONE:
        return np.zeros(3, dtype=np.double)

    # angle ~ pi
    if val < DEF_EULER_CLOSE_TO_MINUS_ONE:
        r00 = R[0, 0]
        r11 = R[1, 1]
        r22 = R[2, 2]

        # Compute axis components with safeguards against tiny negative due to roundoff
        if (r00 >= r11) and (r00 >= r22):
            t = 0.5 * (r00 + 1.0)
            if t < 0.0:
                t = 0.0
            ax = np.sqrt(t)
            if ax < DEF_AXIS_COMP_EPS:
                # fallback (rare): choose another axis deterministically
                t1 = 0.5 * (r11 + 1.0)
                if t1 < 0.0:
                    t1 = 0.0
                ay = np.sqrt(t1)
                t2 = 0.5 * (r22 + 1.0)
                if t2 < 0.0:
                    t2 = 0.0
                az = np.sqrt(t2)
                if ay >= az:
                    return np.array([0.0, np.pi, 0.0], dtype=np.double)
                return np.array([0.0, 0.0, np.pi], dtype=np.double)
            ay = R[0, 1] / (2.0 * ax)
            az = R[0, 2] / (2.0 * ax)

        elif r11 >= r22:
            t = 0.5 * (r11 + 1.0)
            if t < 0.0:
                t = 0.0
            ay = np.sqrt(t)
            if ay < DEF_AXIS_COMP_EPS:
                t0 = 0.5 * (r00 + 1.0)
                if t0 < 0.0:
                    t0 = 0.0
                ax = np.sqrt(t0)
                t2 = 0.5 * (r22 + 1.0)
                if t2 < 0.0:
                    t2 = 0.0
                az = np.sqrt(t2)
                if ax >= az:
                    return np.array([np.pi, 0.0, 0.0], dtype=np.double)
                return np.array([0.0, 0.0, np.pi], dtype=np.double)
            ax = R[0, 1] / (2.0 * ay)
            az = R[1, 2] / (2.0 * ay)

        else:
            t = 0.5 * (r22 + 1.0)
            if t < 0.0:
                t = 0.0
            az = np.sqrt(t)
            if az < DEF_AXIS_COMP_EPS:
                t0 = 0.5 * (r00 + 1.0)
                if t0 < 0.0:
                    t0 = 0.0
                ax = np.sqrt(t0)
                t1 = 0.5 * (r11 + 1.0)
                if t1 < 0.0:
                    t1 = 0.0
                ay = np.sqrt(t1)
                if ax >= ay:
                    return np.array([np.pi, 0.0, 0.0], dtype=np.double)
                return np.array([0.0, np.pi, 0.0], dtype=np.double)
            ax = R[0, 2] / (2.0 * az)
            ay = R[1, 2] / (2.0 * az)

        # Normalize axis, then scale by pi
        nrm = np.sqrt(ax * ax + ay * ay + az * az)
        if nrm < DEF_AXIS_NORM_EPS:
            return np.array([np.pi, 0.0, 0.0], dtype=np.double)
        inv = np.pi / nrm
        return np.array([ax * inv, ay * inv, az * inv], dtype=np.double)

    # general case
    th = np.arccos(val)

    # vee(R - R^T)
    vx = R[2, 1] - R[1, 2]
    vy = R[0, 2] - R[2, 0]
    vz = R[1, 0] - R[0, 1]

    # scale = th / (2 sin(th)) with stable small-angle approximation
    # th/(2 sin th) ~ 0.5 + th^2/12 for th -> 0
    if th < DEF_THETA_SCALE_SMALL:
        th2 = th * th
        scale = 0.5 + th2 / 12.0
    else:
        scale = 0.5 * th / np.sin(th)

    return np.array([scale * vx, scale * vy, scale * vz], dtype=np.double)



if __name__ == "__main__":
    
    from .parse_custom import LMPCustom
    
    fn = sys.argv[1]
    lmp = LMPCustom(fn)
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