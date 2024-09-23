import numpy as np

from infovar.stats.canonical_estimators import cca

def test_cca_1d():
    X = np.random.normal(0, 1, size=100)
    Y = 2*X
    JX, JY, rho = cca(X, Y)
    assert np.isclose(rho, 1.)
    assert np.isclose(np.abs(JX), 1.).all()
    assert np.isclose(np.abs(JY), 1.).all()

def test_cca_2d():
    a = np.random.normal(0, 1, size=100)
    b = np.random.normal(0, 1, size=100)
    c = np.random.normal(0, 1, size=100)

    X = np.column_stack([a+b, a-b])
    Y = np.column_stack([a+c, c])
    _, __, rho = cca(X, Y)

    assert np.isclose(rho, 1.)
