from typing import List

import numpy as np

__all__ = [
    "ContinuousHelper"
]

class ContinuousHelper:

    def __init__(self):
        pass

    def prob(
        self,
        values: List[np.ndarray],
        sigmas: List[np.ndarray]
    ) -> List[np.ndarray]:
        assert isinstance(values, List)
        assert isinstance(sigmas, List)
        assert len(values) == len(sigmas)
        raise NotImplementedError("TODO")

    def argmax(
        self,
        data: List[np.ndarray]
    ) -> np.ndarray:
        assert isinstance(data, List)
        raise NotImplementedError("TODO")

    def smooth_label_map(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        assert data.ndim == 2
        raise NotImplementedError("TODO")
