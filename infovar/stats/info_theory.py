from typing import Union

import numpy as np

__all__ = [
    "mse_lower_bound",
    "rmse_lower_bound",
    "corr_to_info",
    "info_to_corr"
]


# Lower bound of MSE
def mse_lower_bound(cond_h: Union[float, np.ndarray], base: float=2) :
    return 1/(2*np.pi*np.e) * np.exp(2*cond_h*np.log(base))

# Lower bound of RMSE
def rmse_lower_bound(cond_h: Union[float, np.ndarray], base: float=2) :
    return np.sqrt(mse_lower_bound(cond_h, base = base))

# Correlation to amount of information
def corr_to_info(rho, base: float=2):
    return - 0.5 * np.log(1 - rho**2) / np.log(base)

# Information to correlation
def info_to_corr(mi, base: float=2):
    return np.sqrt(1 - base**(-2*mi))
