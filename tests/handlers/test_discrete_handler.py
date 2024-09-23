import os
from typing import Dict, List, Any

import numpy as np
import pytest

from infovar import DiscreteHandler, StandardGetter

@pytest.fixture(scope="module")
def dhandler() -> DiscreteHandler:
    x1 = np.random.normal(0, 1, size=(1000, 1))
    x2 = np.random.normal(0, 1, size=(1000, 1))
    n = np.random.normal(0, 1, size=(1000, 1))
    y = (x1 + x2)**2 + 0.1 * n
    getter = StandardGetter(
        ["x1", "x2"], ["y"],
        np.column_stack([x1, x2]), y
    )

    dhandler = DiscreteHandler()
    dhandler.set_getter(getter.get)
    dhandler.set_restrictions({
        "all": {},
        "neg": {"y": [None, 0]},
        "pos": {"y": [0, None]}
    })
    dhandler.set_path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    )
    if os.path.isdir(dhandler.save_path):
        dhandler.remove(None) # Just in case cleanup failed during last pytest run

    assert isinstance(str(dhandler), str)
    assert dhandler.overview() is None

    return dhandler

@pytest.fixture(scope="module")
def settings() -> Dict[str, Any]:
    return {
        "restrictions": ["all", "neg", "pos"],
        "min_samples": 200,
        "statistics": ["mi", "corr"],
        "uncertainty": {
            "mi": {
                "name": "subsampling",
                "args": {}
            },
            "corr": {
                "name": "bootstrapping",
                "args": {}
            }
        }
    }

def test_update(dhandler: DiscreteHandler, settings: Dict[str, Any]):
    filename = dhandler.get_filename("y")

    assert not os.path.isfile(filename)

    dhandler.update(
        ["x1", "x2"],
        "y",
        settings
    )

    assert os.path.isfile(filename)
    time1 = os.path.getmtime(filename)

    dhandler.update(
        ["x1", "x2"],
        "y",
        settings,
        iterable_x=True
    )

    time2 = os.path.getmtime(filename)
    assert time2 != time1

    # The following should have no effect
    dhandler.update(
        ["x1", "x2"],
        "y",
        settings
    )

    time3 = os.path.getmtime(filename)
    assert time3 == time2

@pytest.mark.run(after='test_update')
def test_overwrite(dhandler: DiscreteHandler, settings: Dict[str, Any]):
    filename = dhandler.get_filename("y")

    time0 = os.path.getmtime(filename)
    
    dhandler.overwrite(
        ["x1", "x2"],
        "y",
        settings
    )

    time1 = os.path.getmtime(filename)
    assert time1 != time0

    dhandler.overwrite(
        ["x1", "x2"],
        "y",
        settings,
        iterable_x=True
    )
    
    time2 = os.path.getmtime(filename)
    assert time2 != time0

@pytest.mark.run(after="test_overwrite")
def test_store_error(dhandler: DiscreteHandler, settings: Dict[str, Any]):
    dhandler.restrictions = None
    print("STORE ERROR")
    with pytest.raises(Exception):
        dhandler.store(
            "x1",
            "y",
            settings
        )
    # dhandler.

@pytest.mark.run(after='test_store_error')
def test_read(dhandler: DiscreteHandler):
    entry = dhandler.read(
        ["x1", "x2"], "y", "all"
    )
    assert isinstance(entry, Dict)

    entries = dhandler.read(
        ["x1", "x2"], "y", "neg",
        iterable_x=True
    )
    assert isinstance(entries, List)
    assert len(entries) == 2

@pytest.mark.run(after='test_read')
def test_read_default(dhandler: DiscreteHandler):
    with pytest.raises(Exception):
        entry = dhandler.read(
            "z", "y", "all"
        )

    entry = dhandler.read(
        "z", "y", "all",
        default=None
    )
    assert entry is None

    with pytest.raises(Exception):
        entry = dhandler.read(
            "x1", "y", "whatever"
        )

    entry = dhandler.read(
        "x1", "y", "whatever",
        default=None
    )
    assert entry is None

@pytest.mark.run(after='test_read_default')
def test_get_available(dhandler: DiscreteHandler):
    res = dhandler.get_available_targets()
    assert len(res) == 1
    res = dhandler.get_available_variables("y")
    assert len(res) == 3
    res = dhandler.get_available_restrictions("y", "x1")
    assert len(res) == 3
    res = dhandler.get_available_restrictions("y", ["x1", "x2"])
    assert len(res) == 3
    res = dhandler.get_available_stats("y", "x1", "all")
    assert len(res) == 2
    res = dhandler.get_available_stats("y", ["x1", "x2"], "neg")
    assert len(res) == 2

@pytest.mark.run(after='test_get_available')
def test_cleanup(dhandler: DiscreteHandler):
    dhandler.delete_stats("y", "mi")
    stats = dhandler.get_available_stats("y", ["x1", "x2"], "all")
    assert "mi" not in stats

    dhandler.remove("y")
    assert "y" not in dhandler.get_available_targets()

    dhandler.remove(None)
    assert len(dhandler.get_existing_saves()) == 0
