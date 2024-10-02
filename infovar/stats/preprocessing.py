import numpy as np

__all__ = ["break_degeneracy"]


def break_degeneracy(data: np.ndarray) -> np.ndarray:
    """
    Measures the sample step and add an adequate noise to break degeneracy (i.e., eliminate duplicates). Allows k-nearest neighbor estimators (e.g., entropy) to be used with data that, without processing, would cause the algorithms to fail.
    Note: this function does not work in all situations (for instance when applying a logarithm).

    Parameters
    ----------
    data : np.ndarray
        Data with potential duplicates.

    Returns
    -------
    np.ndarray
        Data without duplicates. If no duplicates are found, no changes are made.
    """
    diffs = np.diff(np.sort(data, axis=0), axis=0)
    diffs = np.where(diffs == 0, np.nan, diffs)

    steps = np.nanmin(diffs, axis=0)

    datac = 0.5 * (np.random.random(size=data.shape) - 0.5) * steps
    return data + datac
