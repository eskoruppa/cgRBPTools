from __future__ import annotations

import sys, os
import argparse
import numpy as np
import scipy as sp
from ..SO3 import so3


def marginalize_stiffness_matrix(stiffmat: np.ndarray | sp.sparse.spmatrix, n_neighbors: int = 10) -> np.ndarray:
    """
    Compute the marginal effective stiffness of each 6x6 block in the stiffness matrix.
    
    This function processes the stiffness matrix by iterating through 6x6 blocks and
    computing the marginal stiffness for each block by considering n_neighbors
    neighboring blocks, inverting the combined matrix, and extracting the diagonal
    entries of the center block.
    
    Parameters:
    -----------
    stiffmat : np.ndarray or sp.sparse.spmatrix
        The stiffness matrix. Can be a dense numpy array or a sparse scipy matrix.
        Expected to have shape (N*6, N*6) where N is the number of 6x6 blocks.
    n_neighbors : int, optional
        Number of neighboring 6x6 blocks to include on each side. Default is 10.
    
    Returns:
    --------
    marginals : np.ndarray
        Array of shape (N, 6) containing the marginal effective stiffness values,
        where N is the number of 6x6 blocks. Each row contains the diagonal entries
        of the inverted covariance matrix for that block and its neighbors.
    
    Notes:
    ------
    - For boundary blocks, the number of neighbors will be reduced as appropriate
    - The function takes the reciprocal of the diagonal entries of the inverted matrix
    - The computation is performed block by block for numerical stability
    """
    # Convert to dense array if sparse
    if sp.sparse.issparse(stiffmat):
        stiffmat = stiffmat.toarray()
    
    # Determine the number of 6x6 blocks
    n_blocks = stiffmat.shape[0] // 6
    if stiffmat.shape[0] % 6 != 0 or stiffmat.shape[1] % 6 != 0:
        raise ValueError("Stiffness matrix dimensions must be divisible by 6")
    
    # Initialize output array
    marginals = np.zeros((n_blocks, 6))
    
    # Process each 6x6 block
    for block_idx in range(n_blocks):
        # Determine the range of neighboring blocks to include
        start_block = max(0, block_idx - n_neighbors)
        end_block = min(n_blocks, block_idx + n_neighbors + 1)
        
        # Extract the submatrix containing the block and its neighbors
        start_row = start_block * 6
        end_row = end_block * 6
        
        submatrix = stiffmat[start_row:end_row, start_row:end_row]
        
        # Invert the submatrix
        try:
            submatrix_inv = np.linalg.inv(submatrix)
        except np.linalg.LinAlgError:
            # If inversion fails, use pseudo-inverse
            submatrix_inv = np.linalg.pinv(submatrix)
        
        # Calculate the position of the current block within the submatrix
        center_block_offset = (block_idx - start_block) * 6
        center_block_end = center_block_offset + 6
        
        # Extract the diagonal entries corresponding to the center block
        diag_entries = np.diag(submatrix_inv[center_block_offset:center_block_end, 
                                             center_block_offset:center_block_end])
        
        # Take the reciprocal to get the marginal effective stiffness
        marginals[block_idx, :] = 1.0 / diag_entries
    
    return marginals
