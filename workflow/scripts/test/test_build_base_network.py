import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from build_base_network import (
    add_dclines,
    assign_link_length_and_efficiency,
    impute_cross_substation_line_ratings,
)


class FakeNetwork:
    def __init__(self):
        self.buses = pd.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
            },
            index=["a", "b"],
        )
        self.links = pd.DataFrame()

    def madd(self, component, names, suffix="", **kwargs):
        index = [f"{name}{suffix}" for name in names]
        data = {}
        for key, value in kwargs.items():
            if isinstance(value, str) or not hasattr(value, "__iter__"):
                data[key] = [value] * len(index)
            else:
                data[key] = list(value)
        self.links = pd.concat([self.links, pd.DataFrame(data, index=index)], axis=0)


def _make_dclines():
    return pd.DataFrame(
        {
            "from_bus_id": ["a"],
            "to_bus_id": ["b"],
            "Pt": [1000.0],
        },
        index=["dc1"],
    )


def test_add_dclines_with_losses_creates_directional_links():
    n = add_dclines(FakeNetwork(), _make_dclines())

    assert list(n.links.index) == ["dc1_fwd", "dc1_rev"]
    assert n.links.at["dc1_fwd", "bus0"] == "a"
    assert n.links.at["dc1_fwd", "bus1"] == "b"
    assert n.links.at["dc1_rev", "bus0"] == "b"
    assert n.links.at["dc1_rev", "bus1"] == "a"


def test_impute_cross_substation_line_ratings_uses_local_voltage_median():
    buses = pd.DataFrame(
        {
            "sub_id": [1, 2, 3, 4, 5, 6],
            "v_nom": [345.0] * 6,
        },
        index=["a", "b", "c", "d", "e", "f"],
    )
    branches = pd.DataFrame(
        {
            "from_bus_id": ["a", "c", "e"],
            "to_bus_id": ["b", "d", "f"],
            "rateA": [0.0, 1000.0, 1400.0],
            "branch_device_type": ["Line"] * 3,
            "interconnect": ["Eastern"] * 3,
        },
        index=["missing", "rated_low", "rated_high"],
    )

    result = impute_cross_substation_line_ratings(branches, buses)

    assert result.at["missing", "rateA"] == 1200.0
    assert branches.at["missing", "rateA"] == 0.0


def test_impute_cross_substation_line_ratings_preserves_same_substation_zero():
    buses = pd.DataFrame(
        {
            "sub_id": [1, 1, 2, 3],
            "v_nom": [230.0] * 4,
        },
        index=["a", "b", "c", "d"],
    )
    branches = pd.DataFrame(
        {
            "from_bus_id": ["a", "c"],
            "to_bus_id": ["b", "d"],
            "rateA": [0.0, 500.0],
            "branch_device_type": ["Line", "Line"],
            "interconnect": ["Eastern", "Eastern"],
        },
        index=["internal", "rated"],
    )

    result = impute_cross_substation_line_ratings(branches, buses)

    assert result.at["internal", "rateA"] == 0.0
