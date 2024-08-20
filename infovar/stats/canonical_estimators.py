from typing import Tuple, Union

import numpy as np
from scipy.linalg import sqrtm

__all__ = [
    "contraction_matrix",
    "canonical_corr",
    "cca"
]

def contraction_matrix(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the contraction matrix as well as the matricial square-root of the covariance matrices.
    """
    
    # Number of pixels
    L = X.shape[0]
    
    # Centering X and Y
    Xc = X - np.mean(X, 0)
    Yc = Y - np.mean(Y, 0)

    # Unbiased covariance and crosses covariance matrices of X and Y
    R_XX = Xc.T @ Xc / (L-1)
    R_YY = Yc.T @ Yc / (L-1)
    R_XY = Xc.T @ Yc / (L-1)

    # Matricial square-root of covariance matrices
    R_XX_Pud = sqrtm(R_XX)
    R_YY_Pud = sqrtm(R_YY)

    # Contraction matrix (which singular values are the canonical correlations)   
    M = np.linalg.solve(R_XX_Pud, R_XY) @ np.linalg.inv(R_YY_Pud)
    
    return M, R_XX_Pud, R_YY_Pud

def canonical_corr(X: np.ndarray, Y: np.ndarray, max: bool=True) -> Union[float, np.ndarray]:
    """
    Returns the canonical correlation coefficient of data X and Y.
    If `max` is False, returns all the singular values in decreasing order.
    """

    # Contraction matrix (which singular values are the canonical correlations)   
    M, _, __ = contraction_matrix(X, Y)

    # Singular values decomposition
    _, S, __ = np.linalg.svd(M)
    
    if max:
        return S[0] # S is already sorted in decreasing order
    
    return S

def cca(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO
    """

    # Contraction matrix (which singular values are the canonical correlations)   
    M, R_XX_Pud, R_YY_Pud = contraction_matrix(X, Y)

    # Décomposition en valeurs singulières pour avoir les correlations canoniques
    UX, S, UY = np.linalg.svd(M) # Vérifier l'ordre des outputs
    
    # Projection coefficients
    JX = np.linalg.solve(R_XX_Pud, UX[:, 0])
    JY = np.linalg.solve(R_YY_Pud, UY[:, 0])

    # Constraint of sum to 1 over the coefficient
    JX /= abs(JX).sum()
    JY /= abs(JY).sum()

    return JX, JY, S[0]