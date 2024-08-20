import numpy as np
import pytest

from infovar.stats import Statistic, MI, Bootstrapping, Subsampling

@pytest.fixture(scope="module")
def x() -> np.ndarray:
    return np.random.normal(0, 1, size=(1000, 1))

@pytest.fixture(scope="module")
def y() -> np.ndarray:
    return np.random.normal(0, 1, size=(1000, 1))

@pytest.fixture(scope="module")
def stat() -> Statistic:
    return MI()

def test_bootstrapping(x: np.ndarray, y: np.ndarray, stat: Statistic):
    boot = Bootstrapping()
    boot.compute_sigma(x, y, stat)

def test_subsampling(x: np.ndarray, y: np.ndarray, stat: Statistic):
    subs = Subsampling()
    subs.compute_sigma(x, y, stat)
