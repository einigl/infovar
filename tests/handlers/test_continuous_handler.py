import os
from typing import Dict, List, Any

import numpy as np
import pytest

from infovar import ContinuousHandler, StandardGetter

@pytest.fixture
def handler() -> ContinuousHandler:
    x1 = np.random.normal(0, 1, size=(2000, 1))
    x2 = np.random.normal(0, 1, size=(2000, 1))
    n = np.random.normal(0, 1, size=(2000, 1))
    y1 = (x1 + x2)**2 + 0.1 * n
    y2 = (x1 - x2)**2 + 0.1 * n
    getter = StandardGetter(
        ["x1", "x2"], ["y1", "y2"],
        np.column_stack([x1, x2]), np.column_stack([y1, y2])
    )

    handler = ContinuousHandler()
    handler.set_getter(getter.get)
    handler.set_path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data-continuous")
    )

    print(str(handler))
    print(handler.overview())

    return handler

@pytest.fixture
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

@pytest.fixture
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

@pytest.mark.order(1)
def test_update_profile(handler: ContinuousHandler, settings_profile: Dict[str, Any]):
    handler.update(
        "x1",
        "y1",
        settings_profile
    )

    handler.update(
        ["x1", "x2"],
        "y1",
        settings_profile
    )

@pytest.mark.order(2)
def test_update_map(handler: ContinuousHandler, settings_map: Dict[str, Any]):
    handler.update(
        "x1",
        "y1",
        settings_map
    )

    handler.update(
        ["x1", "x2"],
        "y1",
        settings_map
    )

@pytest.mark.order(3)
def test_overwrite_profile(handler: ContinuousHandler, settings_profile: Dict[str, Any]):
    handler.overwrite(
        "x1",
        "y1",
        settings_profile
    )

    handler.overwrite(
        ["x1", "x2"],
        "y1",
        settings_profile
    )

@pytest.mark.order(4)
def test_overwrite_map(handler: ContinuousHandler, settings_map: Dict[str, Any]):
    handler.overwrite(
        "x1",
        "y1",
        settings_map
    )

    handler.overwrite(
        ["x1", "x2"],
        "y1",
        settings_map
    )

@pytest.mark.order(5)
def test_read(handler: ContinuousHandler):
    entry = handler.read(
        "x1",
        "y1",
        "y1"
    )
    assert isinstance(entry, Dict)
    entry = handler.read(
        "x1",
        "y1",
        ["y1", "y2"],
    )
    assert isinstance(entry, Dict)

@pytest.mark.order(6)
def test_get_available(handler: ContinuousHandler):
    # handler.get_available_targets
    # handler.get_available_variables
    # handler.get_available_window_features
    # handler.get_available_stats
    pass

@pytest.mark.order(7)
def test_cleanup(handler: ContinuousHandler):
    handler.delete_stats("y1", "mi", "x1")
    handler.remove(None, None)
