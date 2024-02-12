from abc import ABC, abstractmethod

import numpy as np

from .entropy_estimators import mi, centropy
from .canonical_estimators import canonical_corr
from .info_theory import corr_to_info

__all__ = [
    "Statistic",
    "MI",
    "Condh",
    "Corr",
]

class Statistic(ABC):
    """
    Abstract class for all statistics used in procedure
    Can be used to defined your own.
    """

    def __init__(self):
        pass

    @abstractmethod
    def __call__(
        self,
        variables: np.ndarray,
        targets: np.ndarray
    ):
        pass

class MI(Statistic):
    """
    TODO
    """

    def __call__(
        self,
        variables: np.ndarray,
        targets: np.ndarray
    ):
        _variables = variables + np.random.normal(0, 1e-3, variables.shape)
        _targets = targets + np.random.normal(0, 1e-3, targets.shape)
        return mi(_variables, _targets)

class Condh(Statistic):
    """
    TODO
    """

    def __call__(
        self,
        variables: np.ndarray,
        targets: np.ndarray
    ):
        _variables = variables + np.random.normal(0, 1e-3, variables.shape)
        _targets = targets + np.random.normal(0, 1e-3, targets.shape)
        return centropy(_variables, _targets)

class Corr(Statistic):
    """
    TODO
    """

    def __call__(
        self,
        variables: np.ndarray,
        targets: np.ndarray
    ):
        return canonical_corr(variables, targets, max=True).item()

class LinearInfo(Statistic):
    """
    TODO
    """

    def __call__(
        self,
        variables: np.ndarray,
        targets: np.ndarray
    ):
        rhos = canonical_corr(variables, targets, max=False)
        return np.sum(corr_to_info(rhos))
