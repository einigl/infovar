from typing import Union

import numpy as np

# Lower bound of MSE
def mse_lower_bound(cond_h: Union[float, np.ndarray], base: float=2) :
    return 1/(2*np.pi*np.e) * np.exp(2*cond_h*np.log(base))

# Lower bound of RMSE
def rmse_lower_bound(cond_h: Union[float, np.ndarray], base: float=2) :
    return np.sqrt(mse_lower_bound(cond_h, base = base))

# Correlation to bits
def corr_to_bits(rho, base: float=2):
    return - 0.5 * np.log(1 - rho**2) / np.log(base)

# Bits to correlation
def bits_to_corr(mi, base: float=2):
    return np.sqrt(1 - base**(-2*mi))