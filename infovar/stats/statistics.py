from abc import ABC, abstractmethod

import numpy as np
from scipy.special import erfinv

from .entropy_estimators import mi, centropy
from .canonical_estimators import canonical_corr
from .info_theory import corr_to_info

__all__ = [
    "Statistic",
    "MI",
    "Condh",
    "Corr",
    "GaussInfo",
    "GaussInfoReparam"
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
    Mutual information estimator.
    """

    def __call__(
        self,
        variables: np.ndarray,
        targets: np.ndarray
    ) -> np.ndarray:
        """
        Estimate.

        Parameters
        ----------
        variables : np.ndarray
            Variable data.
        targets : np.ndarray
            Target data.

        Returns
        -------
        np.ndarray
            Estimated mutual information.
        """
        _variables = self.marginaly_gaussianize(variables)
        _targets = self.marginaly_gaussianize(targets)

        return mi(_variables, _targets)
    
    @staticmethod
    def gaussianize(x: np.ndarray) -> np.ndarray:
        """
        Reparameterization of of univariate distribution into Gaussian.
        Raise an error if `x` does not have one dimension.

        Parameters
        ----------
        x : np.ndarray
            Univariate data to reparameterize.

        Returns
        -------
        np.ndarray
            Reparameterized data.
        """
        assert x.ndim == 1
        n = x.size
        order = np.argsort(np.argsort(x))
        r = np.arange(n) + 1
        return np.sqrt(2) * erfinv((2*r[order] - (n+1))/n)

    @staticmethod
    def marginaly_gaussianize(x: np.ndarray) -> np.ndarray:
        """
        Reparameterization of marginal distributions of multivariate data into Gaussian ones.

        Parameters
        ----------
        x : np.ndarray
            Multidimensional data to reparameterize.

        Returns
        -------
        np.ndarray
            Reparameterized data.
        """
        if x.ndim == 1:
            return MI.gaussianize(x)
        return np.column_stack([
            MI.gaussianize(x[:, i]) for i in range(x.shape[1])
        ])

class Condh(Statistic):
    """
    Conditional differential entropy.
    """

    def __call__(
        self,
        variables: np.ndarray,
        targets: np.ndarray
    ) -> np.ndarray:
        """
        Estimate.

        Parameters
        ----------
        variables : np.ndarray
            Variable data.
        targets : np.ndarray
            Target data.

        Returns
        -------
        np.ndarray
            Estimated conditional differential entropy.
        """
        _variables = variables + np.random.normal(0, 1e-3, variables.shape)
        _targets = targets + np.random.normal(0, 1e-3, targets.shape)
        return centropy(_targets, _variables)

class Corr(Statistic):
    """
    Canonical correlation.
    """

    def __call__(
        self,
        variables: np.ndarray,
        targets: np.ndarray
    ) -> np.ndarray:
        """
        Estimate.

        Parameters
        ----------
        variables : np.ndarray
            Variable data.
        targets : np.ndarray
            Target data.

        Returns
        -------
        np.ndarray
            Estimated canonical correlation coefficient.
        """
        return canonical_corr(variables, targets, max=True).item()

class GaussInfo(Statistic):
    """
    Mutual information under multivariate Gaussian assumption.
    """

    def __call__(
        self,
        variables: np.ndarray,
        targets: np.ndarray
    ) -> np.ndarray:
        """
        Estimate.

        Parameters
        ----------
        variables : np.ndarray
            Variable data.
        targets : np.ndarray
            Target data.

        Returns
        -------
        np.ndarray
            Estimated mutual information.
        """
        rhos = canonical_corr(variables, targets, max=False)
        return np.sum(corr_to_info(rhos))

class GaussInfoReparam(Statistic):
    """
    Mutual information under multivariate Gaussian assumption after Gaussian reparameterization of marginals.
    """

    def __call__(
        self,
        variables: np.ndarray,
        targets: np.ndarray
    ) -> np.ndarray:
        """
        Estimate.

        Parameters
        ----------
        variables : np.ndarray
            Variable data.
        targets : np.ndarray
            Target data.

        Returns
        -------
        np.ndarray
            Estimated mutual information.
        """
        _variables = MI.marginaly_gaussianize(variables)
        _targets = MI.marginaly_gaussianize(targets)

        rhos = canonical_corr(_variables, _targets, max=False)
        return np.sum(corr_to_info(rhos))
