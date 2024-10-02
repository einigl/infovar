from abc import ABC, abstractmethod
from math import floor
from random import shuffle

import numpy as np

from .statistics import Statistic

__all__ = [
    "Resampling",
    "Bootstrapping",
    "Subsampling"
]


class Resampling(ABC):

    @abstractmethod
    def compute_sigma(
        self,
        variables: np.ndarray,
        targets: np.ndarray,
        stat: Statistic,
        **kwargs
    ) -> float:
        """
        Estimates the standard deviation of the estimator `stat`.

        Parameters
        ----------
        variables : np.ndarray
            Variable data. Must be a 2D array.
        targets : np.ndarray
            Target data. Must be a 2D array with the same number of rows than `variables`.
        stat : Statistic
            Estimator whose variance is to be estimated.

        Returns
        -------
        float
            Estimate of estimator standard deviation.
        """
        pass

class Bootstrapping(Resampling):

    def compute_sigma(
        self,
        variables: np.ndarray,
        targets: np.ndarray,
        stat: Statistic,
        n: int=10,
    ) -> float:
        """
        Estimates the standard deviation of the estimator `stat` using by bootstrap. This method permits to estimate the variance of an estimator for a given data distribution. It consists in creating new datasets from the same distribution by drawing with replacement samples from existing data.

        Parameters
        ----------
        variables : np.ndarray
            Variable data. Must be a 2D array.
        targets : np.ndarray
            Target data. Must be a 2D array with the same number of rows than `variables`.
        stat : Statistic
            Estimator whose variance is to be estimated.
        n : int, optional
            Number of bootstrap samples, by default 10

        Returns
        -------
        float
            Estimate of estimator standard deviation.
        """
        assert isinstance(variables, np.ndarray) and variables.ndim == 2
        assert isinstance(targets, np.ndarray) and targets.ndim == 2
        assert variables.shape[0] == targets.shape[0]
        assert isinstance(stat, Statistic)

        if n < 2:
            raise ValueError(f"Number of different subset size `n` must be greater than or equal to 2 (here: {n})")

        values = np.zeros(n)
        for i in range(n):
            idx = np.random.choice(variables.shape[0], variables.shape[0], replace=True)
            values[i] = stat(variables[idx], targets[idx])
        return np.std(values)

class Subsampling(Resampling):

    def compute_sigma(
        self,
        variables: np.ndarray,
        targets: np.ndarray,
        stat: Statistic,
        n: int=5,
        min_samples: int=20,
        min_subsets: int=5,
        decades: float=2
    ) -> float:
        """
        Estimates the standard deviation of the estimator `stat` using the approach proposed in Holmes, C. M., & Nemenman, I. (2019). It assumes that the variance of the estimator depends on the number of samples N as Var[stat](N) = B/N, with B being a parameter to be estimated that depends on the data distribution.
        This function assumes that the previous relation is true for the given estimator and compute its variance for several number of samples N by subsampling the dataset. This permit to estimate the value of B.

        Parameters
        ----------
        variables : np.ndarray
            Variable data. Must be a 2D array.
        targets : np.ndarray
            Target data. Must be a 2D array with the same number of rows than `variables`.
        stat : Statistic
            Estimator whose variance is to be estimated.
        n : int, optional
            Number of different subset sizes, by default 5.
        min_samples : int, optional
            Minimum number of samples required for a subset, by default 20.
        min_subsets : int, optional
            Minimum number of subsets for a given subset size, by default 5.
        decades : float, optional
            Maximum orders of magnitude between the largest and smallest subset sizes, by default 2.

        Returns
        -------
        float
            Estimate of estimator standard deviation.
        """
        assert isinstance(variables, np.ndarray) and variables.ndim == 2
        assert isinstance(targets, np.ndarray) and targets.ndim == 2
        assert variables.shape[0] == targets.shape[0]
        assert isinstance(stat, Statistic)
        if n < 2:
            raise ValueError(f"Number of different subset size `n` must be greater than or equal to 2 (here: {n})")
        if min_samples > variables.shape[0]:
            raise ValueError(f"Number of samples {variables.shape[0]} must be greater than or equal to `min_samples` ({min_samples})")

        N = variables.shape[0]

        # Max subset size
        max_Ni = floor(N / min_subsets)
        # Minimum subset size
        min_Ni = max(min_samples, 10**(-decades) * max_Ni)
        # Defines subset sizes
        Nis = np.logspace(np.log10(min_Ni), np.log10(max_Ni), n).astype(int)
        # Keep only distincts sizes (in case where the number of sample is not high enough to get `n` different sizes).
        Nis = np.unique(Nis)
        # Check that there is enough subset sizes to estimate B
        if Nis.size < 2:
            return np.nan

        # Random subsampling
        ni_list = [] # List of number of subset for the different subset sizes
        sigma2_list = [] # List of variances for the different subset sizes
        idx = list(range(N))
        for Ni in Nis:
            ni = floor(N / Ni) # Number of subset
            ni_list.append(ni)
            shuffle(idx)
            mis = []
            for i in range(ni):
                sidx = idx[i*Ni:(i+1)*Ni]
                mis.append(stat(
                    variables[sidx],
                    targets[sidx]
                ))
            sigma2_list.append(np.var(mis))

        nis = np.array(ni_list)
        sigma2s = np.array(sigma2_list)

        # Eq. (8) in Holmes, C. M., & Nemenman, I. (2019)
        return np.sqrt( np.sum((nis-1)/nis * sigma2s) / np.sum(nis - 1) )
