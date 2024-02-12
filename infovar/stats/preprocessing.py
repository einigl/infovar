import numpy as np

__all__ = [
    "break_degeneracy"
]


def break_degeneracy(
    data: np.ndarray
) -> np.ndarray:
    """
    Measures the sample step and add an adequate noise to break degeneracy.
    data: np.ndarray of shape (N, dim)
    """
    diffs = np.diff(np.sort(data, axis=0), axis=0)
    diffs = np.where(diffs==0, np.nan, diffs)
    
    steps = np.nanmin(diffs, axis=0)
    
    datac = (np.random.random(size=data.shape)-0.5) * steps
    return datac
