from abc import ABC, abstractmethod
from math import floor
from random import shuffle

import numpy as np

from .statistics import Statistic

__all__ = [
    "Resampling",
    "Boostrapping",
    "Subsampling"
]


class Resampling(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def compute_sigma(
        self,
        variables: np.ndarray,
        targets: np.ndarray,
        stat: Statistic
    ) -> float:
        pass

class Bootstrapping(Resampling):

    def compute_sigma(
        self,
        variables: np.ndarray,
        targets: np.ndarray,
        stat: Statistic,
        n: int=10,
    ) -> float:
        if n < 2:
            return np.nan
        assert variables.shape[0] == targets.shape[0]
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
        decades: float=2,

    ) -> float:
        if n < 2:
            return np.nan

        assert variables.shape[0] == targets.shape[0]
        N = variables.shape[0]

        max_Ni = floor(N / min_subsets)
        min_Ni = max(min_samples, 10**(-decades) * max_Ni)
        Nis = np.logspace(np.log10(min_Ni), np.log10(max_Ni), n).astype(int)
        Nis = np.unique(Nis)
        if Nis.size < 2:
            return np.nan

        ni_list = []
        sigma2_list = []
        idx = list(range(N))
        for Ni in Nis:
            ni = floor(N / Ni)
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

        return np.sqrt( np.sum((nis-1)/nis * sigma2s) / np.sum(nis - 1) )
