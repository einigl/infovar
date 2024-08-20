import numpy as np

from infovar.stats import break_degeneracy

def test_break_degeneracy():
    a = np.array([1.0, 2.0, 3.0, 2.0])
    b = break_degeneracy(a)
    assert np.unique(b).size == 3
    