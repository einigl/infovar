from abc import ABC, abstractmethod

import numpy as np

from .entropy_estimators import mi, centropy
from .canonical_estimators import canonical_corr

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
        # TODO: gérer ici la dégénérescence et l'ajout de bruit
        # _Yn, _Xn = degeneracy_handling(
        #     _Y, _X, duplicates_y=True, duplicates_x=False
        # )
        return mi(variables, targets)

class Condh(Statistic):
    """
    TODO
    """

    def __call__(
        self,
        variables: np.ndarray,
        targets: np.ndarray
    ):
        return centropy(targets, variables)

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
