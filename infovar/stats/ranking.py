from typing import Optional, Union

import numpy as np
from tqdm import tqdm

from scipy import stats, integrate

def prob_higher(
    mus: np.ndarray,
    sigmas: np.ndarray,
    idx: Optional[int]=None,
    approx: bool=True,
    pbar: bool=False,
) -> Union[np.ndarray, float]:
    """
    Returns the probability
    Ref: https://stats.stackexchange.com/questions/44139/what-is-px-1x-2-x-1x-3-x-1x-n
    """
    if not isinstance(mus, np.ndarray):
        mus = np.array(mus)
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array(sigmas)

    assert mus.shape == sigmas.shape
    n_vars = mus.size

    def integrand(t: np.ndarray, i: int) -> np.ndarray:
        res = stats.norm.logpdf(t, loc=mus[i], scale=sigmas[i])
        for j in range(n_vars):
            if j != i:
                res += stats.norm.logcdf(t, loc=mus[j], scale=sigmas[j])
        return np.exp(res)
    # We restrict to a bounded interval to prevent large integration errors due to the small domain where the integrand is not zero
    n_sigma = 5

    # In approx mode, only the probabilities of line combinations whose +/- n_sigma_approx intervals overlap are calculated
    if approx:
        i_higher = np.argmax(mus)
        n_sigma_approx = 1.5
        significant = (mus + n_sigma_approx*sigmas) > (mus[i_higher] - n_sigma_approx*sigmas[i_higher])
    else:
        significant = np.ones_like(mus, dtype=bool)

    # If the user wants to compute all probabilities
    if idx is None:
        probs = np.zeros(n_vars)# * np.nan
        _iterable = tqdm(range(n_vars)) if pbar else range(n_vars)
        for i in _iterable:
            if  significant[i]:
                bounds = (mus[i]-n_sigma*sigmas[i], mus[i]+n_sigma*sigmas[i])
                fun = lambda t: integrand(t, i)
                p = integrate.quad(fun, *bounds)[0]
                probs[i] = p
        return probs
    
    # If the user has asked for a particular probability
    bounds = (mus[idx]-n_sigma*sigmas[idx], mus[idx]+n_sigma*sigmas[idx])
    fun = lambda t: integrand(t, idx)
    return integrate.quad(fun, *bounds)[0]
