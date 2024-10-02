from typing import List, Union

import numpy as np

__all__ = [
    "condh_to_mse_gaussian",
    "condh_to_rmse_gaussian",
    "corr_to_info_gaussian_1d",
    "corr_to_info_gaussian_nd",
    "info_to_corr_gaussian"
]

# Lower bound of MSE
def condh_to_mse_gaussian(
    condh: Union[float, np.ndarray],
    dim: int=1,
    base: float=2
) -> Union[float, np.ndarray]:
    """
    Converts conditional differential entropy into estimation mean squared error (MSE) under multivariate Gaussian assumption.

    Parameters
    ----------
    condh : Union[float, np.ndarray]
        Conditional differential entropy.
    dim : int, optional
        Dimension of multivariate Gaussian variable, by default 1 (univariate case).
    base : float, optional
        Base of differential entropy, by default 2 (bits).

    Returns
    -------
    Union[float, np.ndarray]
        Estimation mean squared error.
    """
    return dim / (2*np.pi*np.e) * np.exp(2/dim*condh*np.log(base))

# Lower bound of RMSE
def condh_to_rmse_gaussian(
    condh: Union[float, np.ndarray],
    dim: int=1,
    base: float=2
) -> Union[float, np.ndarray]:
    """
    Converts conditional differential entropy into estimation root mean squared error (RMSE) under multivariate Gaussian assumption.

    Parameters
    ----------
    condh : Union[float, np.ndarray]
        Conditional differential entropy.
    dim : int, optional
        Dimension of multivariate Gaussian variable, by default 1 (univariate case).
    base : float, optional
        Base of differential entropy, by default 2 (bits).

    Returns
    -------
    Union[float, np.ndarray]
        Estimation root mean squared error.
    """
    return np.sqrt(condh_to_mse_gaussian(condh, dim=dim, base=base))

# Correlation to amount of information
def corr_to_info_gaussian_1d(
    rho: Union[float, np.ndarray],
    base: float=2
) -> float:
    """
    Converts Pearson correlation coefficient into mutual information under univariate Gaussian asumption.

    Parameters
    ----------
    rho : Union[float, np.ndarray]
        Pearson correlation coefficient or array of correlation coefficients.
    base : float, optional
        Base of mutual information, by default 2 (bits).

    Returns
    -------
    float
        Mutual information between the two subsets of variables.
    """
    assert isinstance(rho, (float, np.ndarray))
    return - 0.5 * np.log(1 - rho**2) / np.log(base)

def corr_to_info_gaussian_nd(
    C: np.ndarray,
    I1: List[int],
    I2: List[int],
    base: float=2
) -> float:
    """
    Converts covariance matrix into mutual information under multivariate Gaussian asumption.

    Parameters
    ----------
    C : np.ndarray
        Full covariance matrix of multivariate normal variable.
    I1 : List[int]
        Indices of first subset of variables.
    I2 : List[int]
        Indices of second subset of variables.
    base : float, optional
        Base of mutual information, by default 2 (bits).

    Returns
    -------
    float
        Mutual information between the two subsets of variables.
    """
    C1, C2 = C[I1, I1], C[I2, I2]
    return 0.5 * np.log(np.det(C1) * np.det(C2) / np.det(C)) / np.log(base)

# Information to correlation
def info_to_corr_gaussian(
    mi: float,
    base: float=2
) -> float:
    """
    Converts mutual information into a Pearson correlation coefficient under multivariate Gaussian asumption.

    Parameters
    ----------
    mi : float
        Mutual information.
    base : float, optional
        Base of mutual information, by default 2 (bits).

    Returns
    -------
    float
        Correlation coefficient.
    """
    return np.sqrt(1 - base**(-2*mi))
