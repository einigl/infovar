import numpy as np

from infovar.stats import break_degeneracy

def test_break_degeneracy():
    a = np.array([1., 2., 3., 2.])
    b = break_degeneracy(a)
    assert np.unique(b).size == a.size
