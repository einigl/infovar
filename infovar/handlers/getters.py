from random import shuffle
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["StandardGetter"]


class StandardGetter:
    def __init__(
        self, x_names: List[str], y_names: List[str], x: np.ndarray, y: np.ndarray
    ):
        self.x_names = x_names
        self.y_names = y_names
        self.x = x
        self.y = y

    def get(
        self,
        x_features: List[str],
        y_features: List[str],
        restrictions: Dict[str, Tuple[float]],
        max_samples: Optional[int] = None,
    ):
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
