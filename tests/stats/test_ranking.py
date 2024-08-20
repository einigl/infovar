import numpy as np

from infovar.stats import prob_higher

def test_prob_higher_equal():
    values = np.array([0., 0.])
    stds = np.array([1., 1.])
    p = prob_higher(values, stds)
    assert len(p) == 2
    assert np.isclose(p[0], 0.5)
    assert np.isclose(p[1], 0.5)

def test_prob_higher_approx():
    values = np.array([0., 3.1])
    stds = np.array([1., 1.])
    p = prob_higher(values, stds, approx=True)
    print(p)
    assert np.isclose(p[0], 0) and np.isclose(p[1], 1)
    p = prob_higher(values, stds, approx=False)
    assert p[0] > 0 and p[1] < 1

def test_prob_higher_index():
    values = np.array([0., 0.])
    stds = np.array([1., 1.])
    p = prob_higher(values, stds, 1)
    assert isinstance(p, float)
    assert np.isclose(p, 0.5)
