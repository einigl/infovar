#!/usr/bin/env python
# Based on the package by Greg Ver Steeg
# See readme.pdf for documentation
# Or go to http://www.isi.edu/~gregv/npeet.html

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import BallTree, KDTree

__all__ = [
    "entropy",
    "centropy",
    "mi"
]


# Nearest neighbors-based estimators

def entropy(
    x: np.ndarray,
    k: int=3,
    base: float=2
) -> float:
    """
    The classic K-L k-nearest neighbor continuous entropy estimator
        x should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert x.ndim in (1, 2)
    N = x.shape[0]

    assert k <= N - 1

    if x.ndim == 1:
        x = np.expand_dims(x, 1)
    
    n_elements, n_features = x.shape
    x = _add_noise(x)
    tree = _build_tree(x)
    nn = _query_neighbors(tree, x, k)
    const = digamma(n_elements) - digamma(k) + n_features * np.log(2)

    return (const + n_features * np.log(nn).mean()) / np.log(base)

def centropy(
    x: np.ndarray,
    y: np.ndarray,
    k: int=3, base:
    float=2
) -> float:
    """
    The classic K-L k-nearest neighbor continuous entropy estimator for the
        entropy of X conditioned on Y.
    """
    assert x.ndim in (1, 2)
    assert y.ndim in (1, 2)
    assert x.shape[0] == y.shape[0]
    N = x.shape[0]

    assert k <= N - 1

    if x.ndim == 1:
        x = np.expand_dims(x, 1)
    if y.ndim == 1:
        y = np.expand_dims(y, 1)

    xy = np.column_stack((x, y))
    entropy_union_xy = entropy(xy, k=k, base=base)
    entropy_y = entropy(y, k=k, base=base)

    return entropy_union_xy - entropy_y

def mi(
    x: np.ndarray,
    y: np.ndarray,
    k: int=3,
    base: float=2
) -> float:
    """
    Mutual information of x and y (conditioned on z if z is not None)
    x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
    if x is a one-dimensional scalar and we have four samples
    """
    assert x.ndim in (1, 2)
    assert y.ndim in (1, 2)
    assert x.shape[0] == y.shape[0]
    N = x.shape[0]

    assert k <= N - 1

    if x.ndim == 1:
        x = np.expand_dims(x, 1)
    if y.ndim == 1:
        y = np.expand_dims(y, 1)

    # Small additive noise
    x = _add_noise(x)
    y = _add_noise(y)

    # Find nearest neighbors in joint space, p=inf means max-norm
    points = np.column_stack((x, y))
    tree = _build_tree(points)
    dvec = _query_neighbors(tree, points, k)

    # Kraskov formula
    res = digamma(k) - (_avgdigamma(x, dvec) + _avgdigamma(y, dvec)) + digamma(N)
    return res / np.log(base)


## Helpers

def _add_noise(x, intens=1e-10):
    """
    Small noise to break degeneracy.
    """
    return x + intens * np.random.random_sample(x.shape)

def _query_neighbors(tree, x, k):
    return tree.query(x, k=k+1)[0][:, k]

def _count_neighbors(tree, x, r) -> int:
    return tree.query_radius(x, r, count_only=True)

def _avgdigamma(points, dvec) -> float:
    """
    Finds number of neighbors in some radius in the marginal space.
    Returns expectation value of <psi(nx)>.
    """
    tree = _build_tree(points)
    dvec = dvec - 1e-15
    num_points = _count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))

def _build_tree(points):
    if points.shape[1] >= 20:
        return BallTree(points, metric='chebyshev')
    return KDTree(points, metric='chebyshev')
