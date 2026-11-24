"""Representative periods: selection in prepare_network and the solve-side constraints."""

import os
import sys

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from _helpers import get_multiindex_snapshots
from opts.representative_periods import add_representative_period_constraints, block_ids
from opts.reserves import add_ERM_constraints
from select_representative_periods import META_KEY, apply_representative_periods, national_features, select_periods

DAYS = 10
PERIODS = [2030, 2040]
CONFIG = {"enable": True, "period_length": 24, "number": 3}


def _daily(values, hours=24):
    return np.repeat(np.asarray(values, dtype=float), hours)


def _features(days=DAYS):
    rng = np.random.default_rng(0)
    hour = np.tile(np.arange(24), days)
    solar = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * _daily(rng.uniform(0.6, 1.0, days))
    wind = _daily(rng.uniform(0.3, 0.6, days)) + 0.05 * np.cos(hour / 24 * 2 * np.pi)
    load = _daily(rng.uniform(900, 1100, days)) + 100 * np.sin(hour / 24 * 2 * np.pi)
    return pd.DataFrame({"wind": wind, "solar": solar, "load": load})


def test_select_periods_extremes_and_weights():
    features = _features()
    features.loc[3 * 24 : 4 * 24 - 1, "load"] += 500  # day 3: peak demand ...
    features.loc[3 * 24 : 4 * 24 - 1, "wind"] = 0.01  # ... and the deepest wind lull
    features.loc[5 * 24 : 6 * 24 - 1, "wind"] = 0.05  # day 5: second-deepest wind lull
    features.loc[7 * 24 : 8 * 24 - 1, "solar"] *= 0.1  # day 7: solar lull

    entries = select_periods(features, pd.Series(1.0, index=features.index), steps=24, number=3)

    extremes = {e["label"]: e["block"] for e in entries if e["kind"] == "extreme"}
    assert extremes == {"demand_max": 3, "wind_min": 5, "solar_min": 7}
    assert [e["block"] for e in entries] == sorted(e["block"] for e in entries)
    typical = [e for e in entries if e["kind"] == "typical"]
    assert len(typical) == 3
    assert sum(e["members"] for e in typical) == DAYS - 3
    # whole days and no remainder: a typical period's weight is its cluster size
    assert all(e["weight"] == pytest.approx(e["members"]) for e in typical)
    assert sum(e["weight"] for e in entries) * 24 == pytest.approx(DAYS * 24)


def test_select_periods_keeps_total_with_remainder():
    features = _features(DAYS + 1).iloc[: DAYS * 24 + 5].reset_index(drop=True)
    hours = pd.Series(1.0, index=features.index)

    entries = select_periods(features, hours, steps=24, number=2)

    assert all(e["block"] < DAYS for e in entries)
    assert sum(e["weight"] for e in entries) * 24 == pytest.approx(hours.sum())


def test_select_periods_rejects_too_many_typical_periods():
    with pytest.raises(ValueError, match="exceeds"):
        select_periods(_features(), pd.Series(1.0, index=range(DAYS * 24)), steps=24, number=DAYS - 2)


def _network(days=DAYS):
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        {"start": "2030-01-01 00:00", "end": f"2030-01-{days:02d} 23:00", "inclusive": "both"},
        PERIODS,
    )
    n.set_investment_periods(PERIODS)
    n.add(
        "Bus",
        "z1",
        carrier="AC",
        country="US",
        reeds_zone="p1",
        reeds_state="CA",
        interconnect="western",
        nerc_reg="NERC1",
    )
    n.add("Carrier", ["AC", "onwind", "solar", "gas", "battery"])

    features = _features(days)
    features.loc[2 * 24 : 3 * 24 - 1, "load"] += 400  # demand peak on day 2
    tile = len(PERIODS)
    n.add("Load", "load", bus="z1", p_set=pd.Series(np.tile(features.load, tile), index=n.snapshots))

    # two wind candidates: the one with the larger p_nom_max sets the national lull
    big = features.wind.copy()
    big[6 * 24 : 7 * 24] = 0.02
    small = features.wind.copy()
    small[4 * 24 : 5 * 24] = 0.0
    for name, profile, p_nom_max in (("wind_big", big, 1000), ("wind_small", small, 10)):
        n.add(
            "Generator",
            name,
            bus="z1",
            carrier="onwind",
            p_nom_extendable=True,
            p_nom_max=p_nom_max,
            capital_cost=500,
            p_max_pu=pd.Series(np.tile(profile, tile), index=n.snapshots),
            build_year=2030,
            lifetime=40,
        )
    n.add(
        "Generator",
        "solar",
        bus="z1",
        carrier="solar",
        p_nom_extendable=True,
        p_nom_max=2000,
        capital_cost=300,
        p_max_pu=pd.Series(np.tile(features.solar, tile), index=n.snapshots),
        build_year=2030,
        lifetime=40,
    )
    n.add(
        "Generator",
        "gas",
        bus="z1",
        carrier="gas",
        p_nom_extendable=True,
        capital_cost=800,
        marginal_cost=60,
        ramp_limit_up=0.2,
        ramp_limit_down=0.2,
        build_year=2030,
        lifetime=40,
    )
    n.add(
        "StorageUnit",
        "battery",
        bus="z1",
        carrier="battery",
        p_nom_extendable=True,
        capital_cost=100,
        max_hours=4,
        efficiency_store=0.9,
        efficiency_dispatch=0.9,
        cyclic_state_of_charge=True,
        cyclic_state_of_charge_per_period=True,
        build_year=2030,
        lifetime=40,
    )
    return n


def test_national_features_weight_capacity_factors_by_p_nom_max():
    n = _network()
    features = national_features(n, 2030)
    sns = n.snapshots[n.snapshots.get_level_values("period") == 2030]
    cf = n.generators_t.p_max_pu.loc[sns]
    expected = (cf.wind_big * 1000 + cf.wind_small * 10) / 1010
    np.testing.assert_allclose(features.wind.to_numpy(), expected.to_numpy())
    np.testing.assert_allclose(features.load.to_numpy(), n.loads_t.p_set.loc[sns, "load"].to_numpy())


def test_apply_representative_periods_reduces_and_reweights():
    n = _network()
    before = n.snapshot_weightings.groupby(level="period").sum()

    metadata = apply_representative_periods(n, CONFIG)

    assert metadata["period_hours"] == 24
    for period in PERIODS:
        sns = n.snapshots[n.snapshots.get_level_values("period") == period]
        assert len(sns) == (3 + CONFIG["number"]) * 24
        w = n.snapshot_weightings.loc[period]
        assert w.objective.sum() == pytest.approx(before.loc[period, "objective"])
        assert w.generators.sum() == pytest.approx(before.loc[period, "generators"])
        assert (w.stores == 1).all()

        entries = metadata["periods"][str(period)]
        starts = [pd.Timestamp(e["start"]) for e in entries]
        assert starts == sorted(starts)
        assert all(pd.Timestamp(e["end"]) - pd.Timestamp(e["start"]) == pd.Timedelta(hours=23) for e in entries)
        extremes = {e["label"]: pd.Timestamp(e["start"]).day - 1 for e in entries if e["kind"] == "extreme"}
        assert extremes["demand_max"] == 2
        assert extremes["wind_min"] == 6  # the big candidate's lull, not the small one's
    assert n.generators_t.p_max_pu.index.equals(n.snapshots)


def test_apply_representative_periods_after_averaging():
    n = _network()
    m = n.copy(snapshots=[])
    sns = n.snapshots[n.snapshots.get_level_values("timestep").hour % 6 == 0]
    m.set_snapshots(sns)
    m.investment_periods = n.investment_periods
    m.snapshot_weightings = n.snapshot_weightings.loc[sns] * 6
    for attr in ("p_max_pu",):
        m.generators_t[attr] = n.generators_t[attr].loc[sns]
    m.loads_t.p_set = n.loads_t.p_set.loc[sns]

    apply_representative_periods(m, CONFIG)

    assert len(m.snapshots) == len(PERIODS) * (3 + CONFIG["number"]) * 4
    assert (m.snapshot_weightings.stores == 6).all()
    assert m.snapshot_weightings.loc[2030].objective.sum() == pytest.approx(DAYS * 24)


def test_period_length_must_be_a_whole_number_of_snapshots():
    n = _network()
    with pytest.raises(ValueError, match="whole number"):
        apply_representative_periods(n, {**CONFIG, "period_length": 2.5})


def test_metadata_survives_netcdf(tmp_path):
    n = _network()
    metadata = apply_representative_periods(n, CONFIG)
    n.meta = {META_KEY: metadata}
    path = tmp_path / "n.nc"
    n.export_to_netcdf(path)
    assert pypsa.Network(path).meta[META_KEY] == metadata


@pytest.fixture
def representative_network():
    n = _network()
    n.meta = {META_KEY: apply_representative_periods(n, CONFIG)}
    return n


def _blocks(n):
    ids = block_ids(n, n.model.variables["Generator-p"].indexes["snapshot"])
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.append(starts[1:], len(ids)) - 1
    return ids, starts, ends


def test_block_ids_none_without_representative_periods():
    n = _network()
    n.optimize.create_model(multi_investment_periods=True)
    assert block_ids(n, n.model.variables["Generator-p"].indexes["snapshot"]) is None


def test_storage_cycles_within_each_period(representative_network):
    n = representative_network
    status, _ = n.optimize(
        solver_name="highs",
        multi_investment_periods=True,
        extra_functionality=add_representative_period_constraints,
    )
    assert status == "ok"

    ids, starts, ends = _blocks(n)
    assert len(starts) == len(PERIODS) * (3 + CONFIG["number"])
    soc = n.storage_units_t.state_of_charge["battery"].to_numpy()
    store = n.storage_units_t.p_store["battery"].to_numpy()
    dispatch = n.storage_units_t.p_dispatch["battery"].to_numpy()
    # first snapshot of each period follows that period's last
    np.testing.assert_allclose(
        soc[starts],
        soc[ends] + 0.9 * store[starts] - dispatch[starts] / 0.9,
        atol=1e-5,
    )
    # so no energy moves between periods
    net = pd.Series(0.9 * store - dispatch / 0.9).groupby(ids).sum()
    np.testing.assert_allclose(net, 0, atol=1e-5)


def test_ramp_limits_dropped_between_periods(representative_network):
    n = representative_network
    n.optimize.create_model(multi_investment_periods=True)
    sns = n.model.variables["Generator-p"].indexes["snapshot"]
    before = {name: int((n.model.constraints[name].labels != -1).sum()) for name in n.model.constraints}

    add_representative_period_constraints(n, sns)

    _, starts, _ = _blocks(n)
    for name in ("Generator-p-ramp_limit_up", "Generator-p-ramp_limit_down"):
        labels = n.model.constraints[name].labels.sel(name="gas").to_numpy()
        assert (labels[starts[1:]] == -1).all()
        within = np.setdiff1d(np.arange(1, len(labels)), starts)
        assert (labels[within] != -1).all()
        # PyPSA already skips the first snapshot of the second investment period
        assert before[name] - int((n.model.constraints[name].labels != -1).sum()) == len(starts) - len(PERIODS)


def test_erm_shadow_storage_cycles_within_each_period(representative_network):
    n = representative_network

    def extra_functionality(n, sns):
        add_representative_period_constraints(n, sns)
        add_ERM_constraints(n, sns, regional_erm_data={"all": 0.15})

    status, _ = n.optimize(solver_name="highs", multi_investment_periods=True, extra_functionality=extra_functionality)
    assert status == "ok"

    _, starts, ends = _blocks(n)
    sol = n.model.solution
    soc = sol["StorageUnit-state_of_charge_RESERVES"].sel(name="battery").to_numpy()
    store = sol["StorageUnit-p_store_RESERVES"].sel(name="battery").to_numpy()
    dispatch = sol["StorageUnit-p_dispatch_RESERVES"].sel(name="battery").to_numpy()
    np.testing.assert_allclose(
        soc[starts],
        soc[ends] + 0.9 * store[starts] - dispatch[starts] / 0.9,
        atol=1e-5,
    )
