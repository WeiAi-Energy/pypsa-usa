import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from add_demand import apply_ac_demand_losses


def test_apply_ac_demand_losses_scales_ac_carrier():
    demand = pd.DataFrame({"bus_1": [100.0], "bus_2": [200.0]})

    scaled = apply_ac_demand_losses(demand, "AC")

    pd.testing.assert_frame_equal(scaled, demand * 1.05)


def test_apply_ac_demand_losses_only_affects_ac_carrier():
    demand = pd.DataFrame({"bus_1": [100.0], "bus_2": [200.0]})

    scaled = apply_ac_demand_losses(demand, "residential-electricity")

    pd.testing.assert_frame_equal(scaled, demand)
