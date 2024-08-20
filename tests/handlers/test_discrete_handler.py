import os
from typing import Dict, List, Any

import numpy as np
import pytest

from infovar import DiscreteHandler, StandardGetter

@pytest.fixture
def handler() -> DiscreteHandler:
    x1 = np.random.normal(0, 1, size=(1000, 1))
    x2 = np.random.normal(0, 1, size=(1000, 1))
    n = np.random.normal(0, 1, size=(1000, 1))
    y = (x1 + x2)**2 + 0.1 * n
    getter = StandardGetter(
        ["x1", "x2"], ["y"],
        np.column_stack([x1, x2]), y
    )

    handler = DiscreteHandler()
    handler.set_getter(getter.get)
    handler.set_restrictions({
        "all": {},
        "neg": {"y": [None, 0]},
        "pos": {"y": [0, None]}
    })
    handler.set_path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    )

    print(str(handler))
    print(handler.overview())

    return handler

@pytest.fixture
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

@pytest.mark.order(1)
def test_update(handler: DiscreteHandler, settings: Dict[str, Any]):
    handler.update(
        ["x1", "x2"],
        "y",
        settings
    )

    handler.update(
        ["x1", "x2"],
        "y",
        settings,
        iterable_x=True
    )

@pytest.mark.order(2)
def test_overwrite(handler: DiscreteHandler, settings: Dict[str, Any]):
    handler.overwrite(
        ["x1", "x2"],
        "y",
        settings
    )

    handler.overwrite(
        ["x1", "x2"],
        "y",
        settings,
        iterable_x=True
    )

@pytest.mark.order(3)
def test_read(handler: DiscreteHandler):
    entry = handler.read(
        ["x1", "x2"],
        "y",
        "all"
    )
    assert isinstance(entry, Dict)
    entries = handler.read(
        ["x1", "x2"],
        "y",
        "neg",
        iterable_x=True
    )
    assert isinstance(entries, List)

@pytest.mark.order(4)
def test_get_available(handler: DiscreteHandler):
    handler.get_available_targets()
    handler.get_available_variables("y")
    handler.get_available_restrictions("y", "x1")
    handler.get_available_restrictions("y", ["x1", "x2"])
    handler.get_available_stats("y", "x1", "all")
    handler.get_available_stats("y", ["x1", "x2"], "neg")

@pytest.mark.order(5)
def test_cleanup(handler: DiscreteHandler):
    handler.delete_stats("y", "mi")
    handler.remove("y")
    handler.remove(None)
    pass