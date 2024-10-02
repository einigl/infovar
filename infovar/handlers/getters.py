from random import shuffle
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["StandardGetter"]


class StandardGetter:
    """
    Class implementing a `get` function for handlers.
    """

    def __init__(
        self,
        x_names: List[str],
        y_names: List[str],
        x: np.ndarray,
        y: np.ndarray
    ):
        """
        Initializer.

        Parameters
        ----------
        x_names : List[str]
            Variable names.
        y_names : List[str]
            Target names.
        x : np.ndarray
            Variable data. Must have x.shape[0] == len(x_names).
        y : np.ndarray
            Target data. Must have y.shape[0] == len(y_names).
        """
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == len(x_names)
        assert y.shape[1] == len(y_names)

        self.x_names = x_names #: Variable names.
        self.y_names = y_names #: Target names.
        self.x = x #: Variable data
        self.y = y #: Target data

    def get(
        self,
        x_features: List[str],
        y_features: List[str],
        restrictions: Dict[str, Tuple[float]],
        max_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns variable and target data that verifies the restrictions provided by `restriction` dictionnary. If `max_samples` is not None, it precises the maximum number of random samples to draw.

        Parameters
        ----------
        x_features : List[str]
            Names of features to return.
        y_features : List[str]
            Names of targets to return.
        restrictions : Dict[str, Tuple[float]]
            Dictionnary of restrictions on variable or target values.
        max_samples : Optional[int], optional
            If not None, maximum number of random samples to draw, by default None.

        Returns
        -------
        np.ndarray
            Selected variable data.
        np.ndarray
            Selected target data.
        """
        assert set(x_features) <= set(self.x_names)
        assert set(y_features) <= set(self.y_names)

        x_idx = [self.x_names.index(v) for v in x_features]
        _x = np.column_stack([self.x[:, i] for i in x_idx])

        y_idx = [self.y_names.index(v) for v in y_features]
        _y = np.column_stack([self.y[:, i] for i in y_idx])

        filt = np.ones(_y.shape[0], dtype="bool")

        # Remove non-finite pixels in variables
        filt &= np.isfinite(_x).all(axis=1)

        # Remove pixels out of the targets ranges (including NaNs)
        if restrictions is None:
            restrictions = {}
        for key, (low, upp) in restrictions.items():
            if key in self.x_names:
                i = self.x_names.index(key)
                filt &= np.logical_and(
                    self.x[:, i] >= (low or -float("inf")),
                    self.x[:, i] <= (upp or float("inf")),
                )
            elif key in self.y_names:
                i = self.y_names.index(key)
                filt &= np.logical_and(
                    self.y[:, i] >= (low or -float("inf")),
                    self.y[:, i] <= (upp or float("inf")),
                )

        _x = _x[filt]
        _y = _y[filt]

        if max_samples is None or _x.shape[0] <= max_samples:
            return _x, _y

        idx = list(range(_x.shape[0]))
        shuffle(idx)
        idx = idx[:max_samples]

        return _x[idx, :], _y[idx, :]
