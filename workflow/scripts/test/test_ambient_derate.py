"""Tests for the temperature-dependent forced-outage derate of thermal units (ambient_derate.py)."""

import os

import numpy as np
import pandas as pd
import pytest
from add_electricity import broadcast_investment_horizons_index
from ambient_derate import (
    DEFAULT_OUTAGE_CURVES,
    DEFAULT_PRIME_MOVERS,
    aggregate_bus_temperature,
    ambient_derate_profiles,
    ambient_derate_settings,
    apply_ambient_derate,
    forced_outage_rate,
    load_outage_curves,
)

WORKFLOW = os.path.join(os.path.dirname(__file__), "..", "..")


@pytest.fixture
def curves():
    return load_outage_curves(os.path.join(WORKFLOW, DEFAULT_OUTAGE_CURVES))


def test_curves_cover_murphy_table(curves):
    assert set(curves.columns) == {
        "combined_cycle",
        "combustion_turbine",
        "diesel",
        "hydro_and_psh",
        "nuclear",
        "steam",
    }
    assert list(curves.index) == list(range(-15, 40, 5))
    assert curves.loc[35, "nuclear"] == pytest.approx(0.124)
    assert set(DEFAULT_PRIME_MOVERS.values()) <= set(curves.columns)


def test_forced_outage_rate_interpolates_inside_the_table(curves):
    cc = curves["combined_cycle"]
    # published points are reproduced
    assert forced_outage_rate(np.array([-15.0, 10.0, 35.0]), cc) == pytest.approx([0.149, 0.025, 0.072])
    # linear between them
    assert forced_outage_rate(np.array([32.5]), cc)[0] == pytest.approx((0.041 + 0.072) / 2)


def test_forced_outage_rate_extrapolates_through_the_outer_points(curves):
    cc = curves["combined_cycle"]
    slope_high = (0.072 - 0.041) / 5
    assert forced_outage_rate(np.array([40.0]), cc)[0] == pytest.approx(0.072 + 5 * slope_high)
    slope_low = (0.081 - 0.149) / 5
    assert forced_outage_rate(np.array([-20.0]), cc)[0] == pytest.approx(0.149 - 5 * slope_low)


def test_forced_outage_rate_is_capped_and_non_negative(curves):
    # extreme cold blows the CT curve past the cap
    assert forced_outage_rate(np.array([-40.0]), curves["combustion_turbine"])[0] == pytest.approx(0.4)
    assert forced_outage_rate(np.array([-40.0]), curves["combustion_turbine"], max_outage=0.3)[0] == pytest.approx(0.3)
    falling = pd.Series([0.05, 0.02], index=[0, 10])
    assert forced_outage_rate(np.array([50.0]), falling)[0] == 0.0


def test_forced_outage_rate_keeps_frame_labels(curves):
    t = pd.DataFrame({"a": [0.0, 30.0], "b": [35.0, -15.0]}, index=pd.date_range("2030", periods=2, freq="h"))
    out = forced_outage_rate(t, curves["nuclear"])
    assert out.index.equals(t.index) and list(out.columns) == ["a", "b"]
    assert out.loc[t.index[0], "b"] == pytest.approx(0.124)


def test_aggregate_bus_temperature_is_area_weighted():
    t = pd.DataFrame({"s1": [10.0, 20.0], "s2": [30.0, 40.0], "s3": [5.0, 5.0]})
    area = pd.Series({"s1": 1.0, "s2": 3.0, "s3": 2.0})
    busmap = pd.Series({"s1": "c1", "s2": "c1", "s3": "c2"})
    out = aggregate_bus_temperature(t, area, busmap)
    assert out["c1"].tolist() == pytest.approx([25.0, 35.0])
    assert out["c2"].tolist() == pytest.approx([5.0, 5.0])


def test_profiles_follow_the_bus_temperature_and_skip_unmapped(curves):
    t = pd.DataFrame({"z1": [35.0, 10.0], "z2": [-15.0, 10.0]})
    gens = pd.DataFrame(
        {"bus": ["z1", "z2", "z1", "z9"], "carrier": ["nuclear", "CCGT", "geothermal", "CCGT"]},
        index=["nuc", "cc", "geo", "far"],
    )
    out = ambient_derate_profiles(t, gens, DEFAULT_PRIME_MOVERS, curves)
    # geothermal has no curve, z9 has no temperature
    assert sorted(out.columns) == ["cc", "nuc"]
    assert out["nuc"].tolist() == pytest.approx([1 - 0.124, 1 - 0.019])
    assert out["cc"].tolist() == pytest.approx([1 - 0.149, 1 - 0.025])


def test_unknown_prime_mover_is_rejected(curves):
    t = pd.DataFrame({"z1": [10.0]})
    gens = pd.DataFrame({"bus": ["z1"], "carrier": ["CCGT"]}, index=["g"])
    with pytest.raises(ValueError, match="gas_turbine"):
        ambient_derate_profiles(t, gens, {"CCGT": "gas_turbine"}, curves)


def test_settings_fall_back_to_defaults():
    s = ambient_derate_settings({})
    assert s["prime_movers"] == DEFAULT_PRIME_MOVERS
    assert s["max_outage"] == 0.4
    s = ambient_derate_settings({"ambient_derate": {"prime_movers": {"CCGT": "steam"}, "max_outage": 0.3}})
    assert s == {"prime_movers": {"CCGT": "steam"}, "max_outage": 0.3}


def test_apply_sets_hourly_p_max_pu_and_leaves_existing_profiles(base_network, curves):
    n = base_network
    n.generators.loc[n.generators.carrier == "gas", "carrier"] = "CCGT"
    ccgt = n.generators.index[n.generators.carrier == "CCGT"]
    # one CCGT already carries a profile (a copy of an existing fleet): untouched
    inherited = ccgt[0]
    n.generators_t.p_max_pu[inherited] = 0.5

    hours = n.snapshots.get_level_values(1)
    temperature = pd.DataFrame(
        {bus: np.linspace(-20, 40, len(hours)) for bus in n.buses.index},
        index=pd.DatetimeIndex(hours),
    )
    n.generators.loc[ccgt, "p_min_pu"] = 0.9

    derated = apply_ambient_derate(
        n,
        temperature,
        ambient_derate_settings({}),
        curves,
        broadcast_investment_horizons_index,
    )

    assert inherited not in derated
    assert (n.generators_t.p_max_pu[inherited] == 0.5).all()
    for g in derated:
        assert n.generators.at[g, "carrier"] == "CCGT"
        expected = 1 - forced_outage_rate(temperature[n.generators.at[g, "bus"]].to_numpy(), curves["combined_cycle"])
        assert n.generators_t.p_max_pu[g].to_numpy() == pytest.approx(expected, abs=1e-3)
        assert n.generators.at[g, "p_max_pu"] == 1.0
        # p_min_pu pulled under the lowest available fraction
        assert n.generators.at[g, "p_min_pu"] <= 0.95 * n.generators_t.p_max_pu[g].min() + 1e-9
    # wind/solar profiles are not touched
    assert not set(derated) & set(n.generators.index[n.generators.carrier.isin(["onwind", "solar"])])
