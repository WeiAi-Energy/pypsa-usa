import os
import sys

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from capacity_derates import (
    DYNAMIC_PRIME_MOVER_BY_CARRIER,
    MAX_EXTRAPOLATED_OUTAGE_FORCED,
    TEMP_MAX,
    TEMP_MIN,
    apply_dynamic_component_derates,
    apply_temperature_derates,
    load_forced_outage_temperature_curves,
    outage_fraction,
    read_region_temperatures,
)

CURVE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "repo_data",
    "forced_outage",
    "outage_forced_temperature_murphy2019.csv",
)


def _snapshots(periods=3):
    timesteps = pd.date_range("2050-01-01 00:00", periods=periods, freq="h")
    return pd.MultiIndex.from_arrays(
        [np.repeat(2050, periods), timesteps],
        names=["period", "timestep"],
    )


def _network(snapshots):
    n = pypsa.Network()
    n.add("Bus", "bus_a", carrier="AC")
    n.add("Bus", "bus_b", carrier="AC")
    n.snapshots = snapshots
    n.set_investment_periods(periods=[2050])
    n.add("Generator", "ccgt", bus="bus_a", carrier="CCGT", p_nom=100.0)
    n.add("Generator", "nuke", bus="bus_b", carrier="nuclear", p_nom=200.0)
    n.add("Generator", "coal", bus="bus_a", carrier="coal", p_nom=300.0)
    return n


def _temperature_table(snapshots, bus_a, bus_b):
    return pd.DataFrame({"bus_a": bus_a, "bus_b": bus_b}, index=snapshots)


def test_forced_outage_curves_span_full_integer_temperature_range():
    curves = load_forced_outage_temperature_curves(CURVE_PATH)

    assert curves.index.min() == TEMP_MIN
    assert curves.index.max() == TEMP_MAX
    assert curves.index.is_monotonic_increasing
    assert not curves.isna().any().any()
    assert curves.to_numpy().min() >= 0.0
    assert curves.to_numpy().max() <= MAX_EXTRAPOLATED_OUTAGE_FORCED
    assert {"combined_cycle", "combustion_turbine", "steam", "nuclear", "hydro_and_psh"}.issubset(curves.columns)


def test_forced_outage_curves_reproduce_tabulated_values():
    """Integer temperatures present in the source table must survive interpolation."""
    curves = load_forced_outage_temperature_curves(CURVE_PATH)

    assert curves.at[10, "combined_cycle"] == pytest.approx(0.025)
    assert curves.at[-15, "combined_cycle"] == pytest.approx(0.149)
    assert curves.at[35, "combined_cycle"] == pytest.approx(0.072)


def test_outage_fraction_interpolates_between_curve_points():
    curves = load_forced_outage_temperature_curves(CURVE_PATH)
    temperatures = pd.DataFrame({"g": [10.0, 12.5, 15.0]})

    outage = outage_fraction(curves, "combined_cycle", temperatures)

    midpoint = (curves.at[12, "combined_cycle"] + curves.at[13, "combined_cycle"]) / 2
    assert outage.at[0, "g"] == pytest.approx(curves.at[10, "combined_cycle"])
    assert outage.at[2, "g"] == pytest.approx(curves.at[15, "combined_cycle"])
    assert outage.at[1, "g"] == pytest.approx(midpoint, abs=1e-3)


def test_outage_fraction_rejects_unknown_prime_mover():
    curves = load_forced_outage_temperature_curves(CURVE_PATH)

    with pytest.raises(KeyError, match="prime mover"):
        outage_fraction(curves, "fusion", pd.DataFrame({"g": [10.0]}))


def test_derates_apply_per_bus_temperature_and_skip_unmapped_carriers():
    snapshots = _snapshots()
    n = _network(snapshots)
    curves = load_forced_outage_temperature_curves(CURVE_PATH)
    # bus_a runs cold-to-hot, bus_b stays at the curve minimum region.
    temperatures = _temperature_table(snapshots, [-15.0, 10.0, 35.0], [10.0, 10.0, 10.0])

    apply_dynamic_component_derates(n, "Generator", DYNAMIC_PRIME_MOVER_BY_CARRIER, temperatures, curves)

    ccgt = n.generators_t.p_max_pu["ccgt"]
    assert ccgt.tolist() == pytest.approx(
        [
            1.0 - curves.at[-15, "combined_cycle"],
            1.0 - curves.at[10, "combined_cycle"],
            1.0 - curves.at[35, "combined_cycle"],
        ],
    )

    nuke = n.generators_t.p_max_pu["nuke"]
    assert nuke.tolist() == pytest.approx([1.0 - curves.at[10, "nuclear"]] * 3)

    # `coal` is intentionally absent from the mapping, so it must stay untouched.
    assert "coal" not in n.generators_t.p_max_pu.columns
    assert n.generators.at["coal", "p_max_pu"] == 1.0


def test_derates_compose_with_an_existing_p_max_pu_timeseries():
    snapshots = _snapshots()
    n = _network(snapshots)
    curves = load_forced_outage_temperature_curves(CURVE_PATH)
    n.generators_t.p_max_pu = pd.DataFrame({"ccgt": [0.5, 0.5, 0.5]}, index=snapshots)
    temperatures = _temperature_table(snapshots, [10.0] * 3, [10.0] * 3)

    apply_dynamic_component_derates(n, "Generator", {"CCGT": "combined_cycle"}, temperatures, curves)

    expected = 0.5 * (1.0 - curves.at[10, "combined_cycle"])
    assert n.generators_t.p_max_pu["ccgt"].tolist() == pytest.approx([expected] * 3)


def test_only_the_tes_discharger_half_is_derated(tmp_path):
    snapshots = _snapshots()
    n = _network(snapshots)
    n.add("Bus", "bus_a tes", carrier="tes")
    n.add("Link", "bus_a tes charger", bus0="bus_a", bus1="bus_a tes", carrier="tes", p_nom=10.0)
    n.add("Link", "bus_a tes discharger", bus0="bus_a tes", bus1="bus_a", carrier="tes", p_nom=10.0)

    temperatures = _temperature_table(snapshots, [10.0] * 3, [10.0] * 3)
    table = temperatures.reset_index()
    path = tmp_path / "region_temperature.csv"
    table.to_csv(path, index=False)

    apply_temperature_derates(n, str(path), CURVE_PATH)

    curves = load_forced_outage_temperature_curves(CURVE_PATH)
    expected = 1.0 - curves.at[10, "combined_cycle"]
    assert n.links_t.p_max_pu["bus_a tes discharger"].tolist() == pytest.approx([expected] * 3)
    assert "bus_a tes charger" not in n.links_t.p_max_pu.columns


def test_read_region_temperatures_rejects_snapshot_mismatch(tmp_path):
    snapshots = _snapshots()
    temperatures = _temperature_table(snapshots, [10.0] * 3, [10.0] * 3)
    path = tmp_path / "region_temperature.csv"
    temperatures.reset_index().to_csv(path, index=False)

    assert read_region_temperatures(str(path), snapshots).shape == (3, 2)

    with pytest.raises(ValueError, match="not aligned"):
        read_region_temperatures(str(path), _snapshots(periods=4))


def test_derates_fall_back_to_national_mean_for_unknown_buses():
    snapshots = _snapshots()
    n = _network(snapshots)
    curves = load_forced_outage_temperature_curves(CURVE_PATH)
    # bus_b has no temperature column, so it must use the mean of what is available.
    temperatures = pd.DataFrame({"bus_a": [-15.0, 10.0, 35.0]}, index=snapshots)

    apply_dynamic_component_derates(n, "Generator", DYNAMIC_PRIME_MOVER_BY_CARRIER, temperatures, curves)

    nuke = n.generators_t.p_max_pu["nuke"]
    assert nuke.tolist() == pytest.approx(
        [
            1.0 - curves.at[-15, "nuclear"],
            1.0 - curves.at[10, "nuclear"],
            1.0 - curves.at[35, "nuclear"],
        ],
    )
