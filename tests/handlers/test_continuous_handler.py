import os
from typing import Dict, Any

import numpy as np
import pytest

from infovar import ContinuousHandler, StandardGetter

@pytest.fixture(scope="module")
def chandler() -> ContinuousHandler:
    x1 = np.random.normal(0, 1, size=(2000, 1))
    x2 = np.random.normal(0, 1, size=(2000, 1))
    n = np.random.normal(0, 1, size=(2000, 1))
    y1 = (x1 + x2)**2 + 0.1 * n
    y2 = (x1 - x2)**2 + 0.1 * n
    getter = StandardGetter(
        ["x1", "x2"], ["y1", "y2"],
        np.column_stack([x1, x2]), np.column_stack([y1, y2])
    )

    chandler = ContinuousHandler()
    chandler.set_getter(getter.get)
    chandler.set_path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data-continuous")
    )
    if os.path.isdir(chandler.save_path):
        chandler.remove(None, None) # Just in case cleanup failed during last pytest run

    assert isinstance(str(chandler), str)
    assert chandler.overview() is None

    return chandler

@pytest.fixture(scope="module")
def settings_profile() -> Dict[str, Any]:
    return {
        "windows": {
            "features": "y1",
            "bounds": [-3, 3],
            "bounds_include_windows": False,
            "scale": "linear",
            "length": 1,
            "points": 5,
        },
        "statistics": ["mi"],
        "uncertainty": {
            "mi": {
                "name": "subsampling",
                "args": {}
            }
        }
    }

@pytest.fixture(scope="module")
def settings_map() -> Dict[str, Any]:
    return {
        "windows": {
            "features": ["y1", "y2"],
            "bounds": [[1, 5], [1, 5]],
            "bounds_include_windows": True,
            "scale": ["log", "log"],
            "num_windows": 3,
            "overlap": "50%",
        },
        "min_samples": 200,
        "max_samples": 500,
        "statistics": ["mi"]
    }

def test_update_profile(chandler: ContinuousHandler, settings_profile: Dict[str, Any]):
    filename = chandler.get_filename("x1", "y1")
    assert not os.path.isfile(filename)

    chandler.update(
        "x1",
        "y1",
        settings_profile
    )

    assert os.path.isfile(filename)

    filename = chandler.get_filename(["x1", "x2"], "y1")
    assert not os.path.isfile(filename)

    chandler.update(
        ["x1", "x2"],
        "y1",
        settings_profile
    )

    assert os.path.isfile(filename)

    # The following should have no effect
    filename = chandler.get_filename("x1", "y1")
    time1 = os.path.getmtime(filename)

    chandler.update(
        "x1",
        "y1",
        settings_profile
    )

    time2 = os.path.getmtime(filename)
    assert time2 == time1

@pytest.mark.run(after='test_update_profile')
def test_update_map(chandler: ContinuousHandler, settings_map: Dict[str, Any]):
    filename = chandler.get_filename("x1", "y2")
    assert not os.path.isfile(filename)
    print(settings_map)
    print()

    chandler.update(
        "x1",
        "y2",
        settings_map
    )

    assert os.path.isfile(filename)

    filename = chandler.get_filename(["x1", "x2"], "y2")
    assert not os.path.isfile(filename)

    chandler.update(
        ["x1", "x2"],
        "y2",
        settings_map
    )

    assert os.path.isfile(filename)

    # The following should have no effect
    filename = chandler.get_filename("x1", "y2")
    time1 = os.path.getmtime(filename)

    chandler.update(
        "x1",
        "y2",
        settings_map
    )

    time2 = os.path.getmtime(filename)
    assert time2 == time1

@pytest.mark.run(after='test_update_map')
def test_overwrite_profile(chandler: ContinuousHandler, settings_profile: Dict[str, Any]):
    filename = chandler.get_filename("x1", "y1")
    time1 = os.path.getmtime(filename)

    chandler.overwrite(
        "x1",
        "y1",
        settings_profile
    )

    time2 = os.path.getmtime(filename)
    assert time2 != time1

@pytest.mark.run(after='test_overwrite_profile')
def test_overwrite_map(chandler: ContinuousHandler, settings_map: Dict[str, Any]):
    filename = chandler.get_filename("x1", "y2")
    time1 = os.path.getmtime(filename)
    
    chandler.overwrite(
        "x1",
        "y2",
        settings_map
    )

    time2 = os.path.getmtime(filename)
    assert time2 != time1

@pytest.mark.run(after='test_overwrite_map')
def test_read(chandler: ContinuousHandler):
    entry = chandler.read(
        "x1",
        "y1",
        "y1"
    )
    assert isinstance(entry, Dict)
    entry = chandler.read(
        "x1",
        "y2",
        ["y1", "y2"],
    )
    assert isinstance(entry, Dict)

@pytest.mark.run(after='test_read')
def test_get_available(chandler: ContinuousHandler):
    res = chandler.get_available_targets()
    assert len(res) == 2
    res = chandler.get_available_variables("y1")
    assert len(res) == 2
    res = chandler.get_available_window_features("x1", "y2")
    assert len(res) == 1
    res = chandler.get_available_stats("x1", "y2", ["y1", "y2"])
    assert len(res) == 1

@pytest.mark.run(after='test_get_available')
def test_cleanup(chandler: ContinuousHandler):
    chandler.delete_stats("y1", "mi", "x1")
    stats = chandler.get_available_stats("x1", "y1", ["y1"])
    assert "mi" not in stats

    assert len(chandler.get_existing_saves()) == 4
    chandler.remove("x1", "y1")
    assert len(chandler.get_existing_saves()) == 3
    chandler.remove("x1", None)
    assert len(chandler.get_existing_saves()) == 2
    chandler.remove(None, "y1")
    assert len(chandler.get_existing_saves()) == 1
    chandler.remove(None, None)
    assert len(chandler.get_existing_saves()) == 0
