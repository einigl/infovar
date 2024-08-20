import numpy as np
import pytest

from infovar.stats import MI, Condh, Corr, LinearInfo, LinearInfoReparam

@pytest.fixture(scope="module")
def x() -> np.ndarray:
    return np.random.normal(0, 1, size=(1000, 1))

@pytest.fixture(scope="module")
def y() -> np.ndarray:
    return np.random.normal(0, 1, size=(1000, 1))

def test_mi(x: np.ndarray, y: np.ndarray):
    mi = MI()
    mi(x, y)

def test_condh(x: np.ndarray, y: np.ndarray):
    condh = Condh()
    condh(x, y)

def test_corr(x: np.ndarray, y: np.ndarray):
    corr = Corr()
    corr(x, y)

def test_linearinfo(x: np.ndarray, y: np.ndarray):
    lininfo = LinearInfo()
    lininfo(x, y)

def test_linearinfo_reparam(x: np.ndarray, y: np.ndarray):
    lininfo = LinearInfoReparam()
    lininfo(x, y)
