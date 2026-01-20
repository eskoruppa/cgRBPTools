from __future__ import annotations

import numpy as np
import scipy as sp
from scipy.sparse import spmatrix
from collections.abc import Sequence
from numpy.typing import ArrayLike
        

def matrix_copy(A: np.ndarray | spmatrix) -> np.ndarray | spmatrix:
    """
    Return a deep copy of a dense or sparse matrix.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        Input matrix.

    Returns
    -------
    numpy.ndarray or scipy.sparse.spmatrix
        A deep copy of the input matrix.
    """

    if A is None:
        raise TypeError("A must not be None.")

    if not hasattr(A, "shape"):
        raise TypeError("A must be a numpy array or a scipy sparse matrix.")

    if len(A.shape) != 2:
        raise ValueError(f"A must be 2-dimensional, got shape {A.shape}.")

    if sp.sparse.issparse(A):
        return A.copy()
    else:
        return np.array(A, copy=True)
    

def rescale_kth(stiff: np.ndarray | spmatrix, k: int, factor: float) -> np.ndarray | spmatrix:
    """
    Rescale the k-th degree of freedom by scaling the corresponding rows and columns.

    This method rescales the stiffness matrix by multiplying all rows and columns
    associated with the k-th degree of freedom (k in 0..5) by ``sqrt(factor)``.
    Equivalently, it applies a similarity transformation

        K -> D K D

    where ``D`` is diagonal with entries equal to ``sqrt(factor)`` for the k-th
    degree of freedom and unity elsewhere.

    As a consequence:
    - diagonal stiffness entries associated with the k-th degree of freedom are
      multiplied by ``factor``,
    - off-diagonal entries coupling the k-th degree of freedom to others are
      multiplied by ``sqrt(factor)``,
    - all other entries are unchanged.

    Parameters
    ----------
    stiff : numpy.ndarray or scipy.sparse.spmatrix
        Square stiffness matrix with dimension a multiple of 6.
    k : int
        Degree-of-freedom index in the range 0..5.
    factor : float
        Positive rescaling factor for the diagonal stiffness entries of the
        selected degree of freedom.

    Returns
    -------
    numpy.ndarray or scipy.sparse.spmatrix
        Rescaled stiffness matrix (same type as input).
    """

    if not isinstance(k, int):
        raise TypeError("k must be an int.")
    if not (0 <= k < 6):
        raise ValueError("k must be an integer in the range 0..5.")

    if not isinstance(factor, (int, float, np.number)):
        raise TypeError("factor must be a real number.")
    
    factor = float(factor)
    if not np.isfinite(factor):
        raise ValueError("factor must be finite (not inf/NaN).")
    if factor <= 0:
        raise ValueError(f'factor must be a positive number, encountered factor = {factor}')

    if not hasattr(stiff, "shape") or len(stiff.shape) != 2:
        raise TypeError("stiff must be a 2D numpy array or a scipy sparse matrix.")

    n, m = stiff.shape
    if n != m:
        raise ValueError(f"stiff must be square, got shape {stiff.shape}.")
    if n == 0:
        raise ValueError("stiff must be non-empty.")
    if n % 6 != 0:
        raise ValueError(f"stiff dimension must be a multiple of 6, got {n}.")

    sqfac = np.sqrt(factor)

    d = np.ones(n, dtype=float)
    d[k::6] = sqfac

    if sp.sparse.issparse(stiff):
        fmt = getattr(stiff, "format", "csr") or "csr"
        D = sp.sparse.diags(d, format=fmt)
        return D @ stiff @ D
    else:
        stiff = np.asarray(stiff)
        return (d[:, None] * stiff) * d[None, :]
    

def rescale_stiff(
    stiff: np.ndarray | spmatrix,
    factor: float,
    entries: Sequence[int] | None = None,
) -> np.ndarray | spmatrix:
    """
    Rescale selected degrees of freedom by scaling corresponding rows and columns.

    This method rescales the stiffness matrix by multiplying rows and columns
    associated with selected degrees of freedom by ``sqrt(factor)``. Internally,
    it applies a similarity transformation

        K -> D K D

    where ``D`` is diagonal with entries equal to ``sqrt(factor)`` for the selected
    degrees of freedom and unity elsewhere.

    As a result:
    - diagonal stiffness entries associated with selected degrees of freedom are
      multiplied by ``factor``,
    - off-diagonal entries coupling selected and unselected degrees of freedom
      are multiplied by ``sqrt(factor)``,
    - couplings between two selected degrees of freedom are multiplied by
      ``factor``,
    - all other entries are unchanged.

    If ``entries`` is None or empty, all rows and columns are rescaled uniformly,
    and the stiffness matrix is multiplied by ``factor``.

    Parameters
    ----------
    stiff : numpy.ndarray or scipy.sparse.spmatrix
        Square stiffness matrix with size compatible with 6 degrees of freedom
        per site.
    factor : float
        Positive rescaling factor applied to diagonal stiffness entries of the
        selected degrees of freedom.
    entries : sequence of int or None, optional
        Degree-of-freedom indices to rescale (each in the range 0..5). If None or
        empty, all degrees of freedom are rescaled uniformly.

    Returns
    -------
    numpy.ndarray or scipy.sparse.spmatrix
        Rescaled stiffness matrix. The return type matches the input type.
    """ 

    if not isinstance(factor, (int, float, np.number)):
        raise TypeError("factor must be a real number.")
    
    factor = float(factor)
    if not np.isfinite(factor):
        raise ValueError("factor must be finite (not inf/NaN).")
    if factor <= 0:
        raise ValueError(f'factor must be a positive number, encountered factor = {factor}')

    if not hasattr(stiff, "shape") or len(stiff.shape) != 2:
        raise TypeError("stiff must be a 2D numpy array or a scipy sparse matrix.")
    
    n, m = stiff.shape
    if n != m:
        raise ValueError(f"stiff must be square, got shape {stiff.shape}.")
    if n == 0:
        raise ValueError("stiff must be non-empty.")
    if n % 6 != 0:
        raise ValueError(f"stiff dimension must be a multiple of 6, got {n}.")

    if entries is None or len(entries) == 0:
        return stiff * factor 

    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise TypeError("entries must be a sequence of ints (e.g., [0, 3, 5]) or None.")

    cleaned: list[int] = []
    for i, k in enumerate(entries):
        if not isinstance(k, int):
            raise TypeError(f"entries[{i}] must be an int, got {type(k)}.")
        if not (0 <= k < 6):
            raise ValueError(f"entries[{i}] must be in range 0..5, got {k}.")
        cleaned.append(k)

    cleaned = sorted(set(cleaned))
    rescaled = stiff.copy() if sp.sparse.issparse(stiff) else np.array(stiff, copy=True)
    for k in cleaned:
        rescaled = rescale_kth(rescaled, k, factor)

    return rescaled



def is_positive_definite(A: ArrayLike, tol: float = 0.0) -> bool:
    """
    Check whether a matrix is (numerically) symmetric positive definite.

    Parameters
    ----------
    A : array-like
        Input matrix.
    tol : float, optional
        Absolute tolerance used for the symmetry check. Default is 0.0.

    Returns
    -------
    bool
        True if A is symmetric positive definite, False otherwise.
    """
    
    # --- tolerance check ---
    if not isinstance(tol, (int, float, np.number)):
        raise TypeError("tol must be a real number.")
    tol = float(tol)
    if tol < 0:
        raise ValueError("tol must be >= 0.")
    if not np.isfinite(tol):
        raise ValueError("tol must be finite.")

    # --- array conversion ---
    A = np.asarray(A)

    # --- shape check ---
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False

    # --- finite entries check ---
    if not np.all(np.isfinite(A)):
        return False

    # --- symmetry check ---
    if not np.allclose(A, A.T, atol=tol, rtol=0.0):
        return False

    # --- positive definiteness via Cholesky ---
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
