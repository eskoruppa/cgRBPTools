import numpy as np
import scipy as sp
from scipy.sparse import spmatrix
from dataclasses import dataclass, field

@dataclass(frozen=True)
class RescaleUnits:
    length_factor: float  
    rotation_factor: float = 1
    energy_factor: float = 1

    def rescale_groundstate(self, X0: np.ndarray):
        X0 = np.asarray(X0)

        if X0.ndim == 1:
            if X0.size % 6 != 0:
                raise ValueError("Groundstate must have 6N entries.")
            N = X0.size // 6
            X0_reshaped = X0.reshape((N, 6))
            X_rescaled = X0_reshaped.copy()

        elif X0.ndim == 2:
            if X0.shape[1] != 6:
                raise ValueError("Groundstate must be of shape (N, 6).")
            X_rescaled = X0.copy()

        else:
            raise ValueError("Groundstate must be 1D (6N,) or 2D (N,6).")

        X_rescaled[:, 0:3] *= self.rotation_factor
        X_rescaled[:, 3:6] *= self.length_factor

        if X0.ndim == 1:
            return X_rescaled.reshape(-1)
        else:
            return X_rescaled

    def rescale_stiffness(self, M: np.ndarray | spmatrix):
        
        if not isinstance(M, (np.ndarray, spmatrix)):
            raise TypeError("M must be a numpy array or a scipy sparse matrix.")
        if M.ndim != 2:
            raise ValueError("Stiffness matrix must be 2-dimensional.")
        nrow, ncol = M.shape
        if nrow != ncol:
            raise ValueError("Stiffness matrix must be square.")
        if nrow % 6 != 0:
            raise ValueError("Stiffness matrix dimension must be 6N x 6N.")

        N = nrow // 6
        s = np.empty(6 * N, dtype=float)
        for i in range(N):
            base = 6 * i
            s[base:base+3] = self.rotation_factor
            s[base+3:base+6] = self.length_factor

        if sp.sparse.isspmatrix(M):
            coo = M.tocoo(copy=True)
            i = coo.row
            j = coo.col
            data = coo.data
            data = self.energy_factor * data / (s[i] * s[j])
            M_rescaled = sp.sparse.coo_matrix((data, (i, j)), shape=coo.shape)
            return M_rescaled.asformat(M.getformat())
        else:
            M_rescaled = np.array(M, dtype=float, copy=True)
            scale_matrix = s[:, None] * s[None, :]
            M_rescaled *= self.energy_factor
            M_rescaled /= scale_matrix
            return M_rescaled
        
    def rescale_model(
        self,
        X0: np.ndarray,
        M: np.ndarray | spmatrix,
    ) -> tuple[np.ndarray, np.ndarray | spmatrix]:

        X_rescaled = self.rescale_groundstate(X0)
        M_rescaled = self.rescale_stiffness(M)

        X0_arr = np.asarray(X0)
        if X0_arr.ndim == 1:
            N_gs = X0_arr.size // 6
        elif X0_arr.ndim == 2:
            N_gs = X0_arr.shape[0]
        else:
            raise RuntimeError("Unexpected groundstate dimensionality.")

        nrow = M.shape[0]
        N_M = nrow // 6

        if N_gs != N_M:
            raise ValueError(
                f"Inconsistent sizes: groundstate corresponds to N={N_gs}, "
                f"stiffness matrix corresponds to N={N_M}."
            )

        return X_rescaled, M_rescaled

        
if __name__ == "__main__":
    
    np.set_printoptions(linewidth=400, precision=2)
    M = np.ones((6,6))
    X = np.ones((6))
    rescale = RescaleUnits(length_factor=1./10,rotation_factor=3.1415,energy_factor=10)
    Xr = rescale.rescale_groundstate(X)
    Mr = rescale.rescale_stiffness(M)
    print(Xr)
    print(Mr)