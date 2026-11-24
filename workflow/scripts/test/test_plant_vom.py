"""Unit tests for plant VOM and new-build parameters.

- ``add_sectors.convert_generators_2_links`` replaces gas/coal/oil generators
  with links that draw from the state fuel bus, which prices the fuel. The link
  must still carry the plant's VOM.
- ``add_extra_components`` keeps the existing fleet only as retiring capacity;
  every new build takes the cost table of its investment period.
- ``add_electricity.apply_dynamic_pricing`` adds each plant's own VOM.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import add_extra_components as aec
from _helpers import get_multiindex_snapshots
from add_electricity import apply_dynamic_pricing, broadcast_investment_horizons_index
from add_sectors import convert_generators_2_links
from ambient_derate import DEFAULT_OUTAGE_CURVES, apply_ambient_derate, forced_outage_rate, load_outage_curves
from build_powerplants import filter_outliers_zscore
from solve_network import freeze_prior_periods

pytestmark = pytest.mark.fast

COSTS = pd.DataFrame(
    {
        "opex_variable_per_mwh": {"gas": 2.0, "CCGT-95CCS": 6.0, "solar": 0.0, "onwind": 1.5},
        "opex_fixed_per_kw": {"gas": 10.0, "CCGT-95CCS": 30.0, "solar": 20.0, "onwind": 25.0},
        "annualized_capex_fom": {"gas": 1e5, "CCGT-95CCS": 2e5, "solar": 5e4, "onwind": 7e4},
        "cost_recovery_period_years": {"gas": 30, "CCGT-95CCS": 30, "solar": 30, "onwind": 30},
        "lifetime": {"gas": 30, "CCGT-95CCS": 30, "solar": 30, "onwind": 25},
        "marginal_cost": {"gas": 30.0, "CCGT-95CCS": 40.0, "solar": 0.0, "onwind": 1.0},
        "efficiency": {"gas": 0.5, "CCGT-95CCS": 0.45, "solar": 0.0, "onwind": 0.0},
        "heat_rate_mmbtu_per_mwh": {"gas": 6.824, "CCGT-95CCS": 7.58, "solar": 0.0, "onwind": 0.0},
        "co2_emissions": {"gas": 0.2, "CCGT-95CCS": 0.01, "CCGT": 0.2, "solar": 0.0, "onwind": 0.0},
    },
)


# ---------------------------------------------------------------------------
# convert_generators_2_links
# ---------------------------------------------------------------------------


@pytest.fixture
def sector_network(base_network):
    """base_network with its gas plants as CCGTs, plus state fuel and co2 buses."""
    n = base_network
    n.buses["STATE"] = n.buses.reeds_state
    for state in ("CA", "TX"):
        n.add("Bus", f"{state} gas", carrier="gas")
        n.add("Bus", f"{state} pwr-co2", carrier="co2")
    n.generators.loc[["gas1", "gas2"], "carrier"] = "CCGT"
    n.generators.loc[["gas1", "gas2"], "efficiency"] = 0.5
    n.generators.loc[["gas1", "gas2"], "vom_cost"] = [3.0, 4.0]
    return n


def test_link_carries_vom_per_unit_of_fuel(sector_network):
    n = sector_network
    convert_generators_2_links(n, "CCGT", " gas", co2_intensity=0.2)

    links = n.links.loc[["gas1", "gas2"]]
    assert links.bus0.tolist() == ["CA gas", "TX gas"]
    # 3 and 4 $/MWh_el at 50 % efficiency is 1.5 and 2 $ per MWh of gas burnt
    assert links.marginal_cost.tolist() == pytest.approx([1.5, 2.0])


def test_time_varying_fuel_cost_is_not_copied(sector_network):
    n = sector_network
    # fuel + VOM per MWh_el, as dynamic fuel pricing writes it
    n.generators_t.marginal_cost["gas1"] = 25.0

    convert_generators_2_links(n, "CCGT", " gas")

    assert "gas1" not in n.links_t.marginal_cost.columns
    assert n.links.at["gas1", "marginal_cost"] == pytest.approx(1.5)


def test_plant_without_vom_cost_gets_zero_and_a_warning(sector_network, caplog):
    n = sector_network
    n.generators.loc["gas2", "vom_cost"] = np.nan

    convert_generators_2_links(n, "CCGT", " gas")

    assert n.links.at["gas2", "marginal_cost"] == 0
    assert "No VOM for 1 CCGT generators" in caplog.text


# ---------------------------------------------------------------------------
# new builds in add_extra_components
# ---------------------------------------------------------------------------


@pytest.fixture
def fleet_network(base_network):
    """base_network with plant efficiency and VOM on its existing gas fleet."""
    n = base_network
    n.generators["land_region"] = ""
    n.generators.loc[["gas1", "gas2"], "efficiency"] = 0.4
    n.generators.loc[["gas1", "gas2"], "vom_cost"] = [3.0, 4.0]
    n.add("Carrier", "gas")
    return n


def test_split_keeps_the_fleet_and_adds_no_thermal_unit(fleet_network):
    n = fleet_network
    n.generators_t.marginal_cost["gas2"] = 20.0  # dynamic fuel price

    aec.split_retirement_gens(n, COSTS, ["gas"], economic=True)

    existing = n.generators.loc["gas2 existing"]
    assert (existing.efficiency, existing.vom_cost, existing.p_nom, existing.p_nom_max) == (0.4, 4.0, 200, 200)
    assert existing.p_nom_extendable
    assert existing.capital_cost == pytest.approx(10.0e3)  # fixed O&M only
    assert (n.generators_t.marginal_cost["gas2 existing"] == 20.0).all()
    # nothing is priced from the fleet: no new unit, no stale time series
    assert "gas2" not in n.generators.index
    assert all("gas2" not in df.columns for df in n.generators_t.values())


def test_technical_retirement_fixes_the_existing_part(fleet_network):
    n = fleet_network

    aec.split_retirement_gens(n, COSTS, ["gas"], economic=False)

    existing = n.generators.loc["gas2 existing"]
    assert (existing.p_nom, existing.p_nom_extendable, existing.efficiency) == (200, False, 0.4)


@pytest.mark.parametrize(("retirement", "retirable"), [("economic", True), ("technical", False)])
def test_myopic_freeze_retires_existing_only_with_economic_retirement(fleet_network, retirement, retirable):
    n = fleet_network
    aec.split_retirement_gens(n, COSTS, ["gas"], economic=retirement == "economic")
    aec.attach_new_thermal(n, COSTS, ["gas"], 2030)
    n.config = {"electricity": {"renewable_carriers": ["onwind", "solar"], "retirement": retirement}}
    n.generators["p_nom_opt"] = n.generators.p_nom

    freeze_prior_periods(n, 2030)

    existing = n.generators.loc["gas2 existing"]
    assert existing.p_nom_extendable == retirable
    assert existing.p_nom == 200
    assert not n.generators.at["z3 gas_2030", "p_nom_extendable"]  # prior new build is frozen either way


def test_new_thermal_takes_the_cost_table_not_the_fleet(fleet_network, monkeypatch):
    n = fleet_network
    aec.split_retirement_gens(n, COSTS, ["gas"], economic=True)

    added = aec.attach_new_thermal(n, COSTS, ["gas"], 2030)

    assert added.tolist() == ["z1 gas_2030", "z2 gas_2030", "z3 gas_2030"]
    new = n.generators.loc[added]
    assert (new.efficiency == 0.5).all()  # ATB, not the fleet's 0.4
    assert (new.vom_cost == 2.0).all()
    assert (new.marginal_cost == 30.0).all()
    assert (new.lifetime == 30).all()
    assert (new.capital_cost == 1e5).all()
    assert (new.p_nom == 0).all() and new.p_nom_extendable.all()
    # no fleet profile: the ambient derate sets p_max_pu from the bus temperature
    assert not n.generators_t.p_max_pu.columns.intersection(added).any()
    assert not n.generators_t.marginal_cost.columns.intersection(added).any()


def test_new_thermal_respects_min_year(fleet_network):
    n = fleet_network
    n.add("Carrier", "CCGT")

    added = aec.attach_new_thermal(n, COSTS, ["gas", "CCGT-95CCS"], 2030, min_years={"CCGT-95CCS": 2035})

    assert set(n.generators.loc[added, "carrier"]) == {"gas"}


def test_new_carrier_generators_get_vom(fleet_network):
    n = fleet_network
    n.add("Carrier", "CCGT")  # CO2 intensity is looked up on the base technology

    aec.attach_new_thermal(n, COSTS, ["CCGT-95CCS"], 2030)

    new = n.generators[n.generators.carrier == "CCGT-95CCS"]
    assert len(new) == len(n.buses)
    assert (new.vom_cost == 6.0).all()


def _two_period(n):
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 23:00", "inclusive": "both"},
        invest_periods=[2030, 2040],
    )
    n.set_investment_periods(periods=[2030, 2040])
    return n


ELEC_CONFIG = {
    "extendable_carriers": {"Generator": ["gas", "nuclear", "onwind", "solar"]},
    "conventional_carriers": ["gas", "nuclear"],
    "renewable_carriers": ["onwind", "solar"],
}
COSTS_NUCLEAR = pd.concat(
    [
        COSTS,
        pd.DataFrame(
            {
                "opex_variable_per_mwh": {"nuclear": 3.0},
                "opex_fixed_per_kw": {"nuclear": 150.0},
                "annualized_capex_fom": {"nuclear": 6e5},
                "cost_recovery_period_years": {"nuclear": 30},
                "lifetime": {"nuclear": 60},
                "marginal_cost": {"nuclear": 10.0},
                "efficiency": {"nuclear": 0.33},
                "heat_rate_mmbtu_per_mwh": {"nuclear": 10.4},
                "co2_emissions": {"nuclear": 0.0},
            },
        ),
    ],
)


@pytest.mark.parametrize("periods", [[2030], [2030, 2040]])
def test_one_new_build_per_bus_carrier_and_period(fleet_network, periods):
    """An existing extendable plant does not add a second, fleet priced, new-build option."""
    n = _two_period(fleet_network) if len(periods) == 2 else fleet_network
    n.add("Carrier", ["nuclear", "onwind", "solar"])
    n.add("Generator", "nuc1", bus="z2", carrier="nuclear", p_nom=1000, p_nom_extendable=True, efficiency=0.32)
    n.add("Generator", "nuc_planned", bus="z1", carrier="nuclear", p_nom=0, p_nom_extendable=True)
    n.generators.loc["wind1", "p_nom"] = 100.0

    thermal, sites = aec.split_existing_fleet(n, ELEC_CONFIG, COSTS_NUCLEAR, economic=True)
    for year in periods:
        aec.attach_renewable_vintages(n, COSTS_NUCLEAR, sites, year)
        aec.attach_new_thermal(n, COSTS_NUCLEAR, thermal, year)
    n.remove("Generator", sites.index)

    assert thermal == ["gas", "nuclear"]
    assert "nuc_planned" not in n.generators.index
    new = n.generators[n.generators.p_nom_extendable & ~n.generators.index.str.contains("existing")]
    per_option = new.groupby(["bus", "carrier", "build_year"]).size()
    assert (per_option == 1).all()
    nuclear = new[new.carrier == "nuclear"]
    assert len(nuclear) == len(n.buses) * len(periods)
    assert (nuclear.efficiency == 0.33).all() and (nuclear.lifetime == 60).all()
    assert n.generators.at["nuc1 existing", "p_nom"] == 1000


def test_renewable_vintages_keep_the_resource_and_take_the_cost_table(fleet_network):
    n = _two_period(fleet_network)
    n.generators_t.p_max_pu["wind1"] = 0.8
    n.generators.loc["wind1", ["p_nom", "land_region"]] = [100.0, "region_a"]
    costs = COSTS

    split = aec.split_retirement_gens(n, costs, ["onwind"], economic=False, keep_sites=True)
    assert split.tolist() == ["wind1"]
    existing = n.generators.loc["wind1 existing"]
    assert (existing.p_nom, existing.p_nom_max, existing.p_nom_extendable) == (100, 100, False)
    assert (n.generators_t.p_max_pu["wind1 existing"] == 0.8).all()

    sites = n.generators.loc[["wind1"]]
    for year in (2030, 2040):
        aec.attach_renewable_vintages(n, costs, sites, year)

    for year in (2030, 2040):
        new = n.generators.loc[f"wind1 {year}"]
        assert (new.bus, new.land_region, new.p_nom, new.p_nom_max) == ("z1", "region_a", 0, 500)
        assert (new.marginal_cost, new.vom_cost, new.lifetime, new.efficiency) == (1.0, 1.5, 25, 1)
        assert new.capital_cost == pytest.approx(7e4)
        assert (n.generators_t.p_max_pu[f"wind1 {year}"] == 0.8).all()


@pytest.mark.parametrize("listed", [True, False])
def test_renewable_carrier_without_sites_is_not_built(fleet_network, caplog, listed):
    """No offshore sites: no constant-output "offwind" at every bus, whether or not it is in renewable_carriers."""
    n = fleet_network
    config = {
        "extendable_carriers": {"Generator": ["gas", "offwind"]},
        "conventional_carriers": ["gas"],
        "renewable_carriers": ["onwind", "solar", *(["offwind"] if listed else [])],
    }

    thermal, sites = aec.split_existing_fleet(
        n,
        config,
        COSTS,
        economic=True,
        profile_carriers=["dataset", "onwind", "offwind", "solar"],
    )

    assert thermal == ["gas"]
    assert sites.empty
    assert "offwind" in caplog.text


def test_existing_and_new_thermal_both_get_the_ambient_derate(fleet_network):
    n = fleet_network
    settings = {"prime_movers": {"gas": "combined_cycle", "nuclear": "nuclear"}, "max_outage": 0.4}
    curves = load_outage_curves(os.path.join(os.path.dirname(__file__), "..", "..", DEFAULT_OUTAGE_CURVES))
    hours = n.snapshots.get_level_values(1)
    temperature = pd.DataFrame(
        {"z1": np.linspace(-20, 40, len(hours)), "z2": 35.0, "z3": np.linspace(40, -20, len(hours))},
        index=pd.DatetimeIndex(hours),
    )
    # the fleet as add_electricity derates it
    n.generators_t.p_max_pu = n.generators_t.p_max_pu.drop(columns=["gas1", "gas2"])
    apply_ambient_derate(n, temperature, settings, curves, broadcast_investment_horizons_index)
    fleet_profile = n.generators_t.p_max_pu["gas2"].copy()
    wind_profile = n.generators_t.p_max_pu["wind1"].copy()

    aec.split_retirement_gens(n, COSTS_NUCLEAR, ["gas"], economic=True)
    added = aec.attach_new_thermal(n, COSTS_NUCLEAR, ["gas", "nuclear"], 2030)
    # add_extra_components derates what has no profile yet
    apply_ambient_derate(n, temperature, settings, curves, broadcast_investment_horizons_index)

    pd.testing.assert_series_equal(n.generators_t.p_max_pu["gas2 existing"], fleet_profile, check_names=False)
    assert len(added) == 6
    for g in added:
        bus, carrier = n.generators.loc[g, ["bus", "carrier"]]
        curve = curves[settings["prime_movers"][carrier]]
        expected = 1 - forced_outage_rate(temperature[bus].to_numpy(), curve, max_outage=0.4)
        assert n.generators_t.p_max_pu[g].to_numpy() == pytest.approx(expected, abs=1e-3)
    pd.testing.assert_series_equal(n.generators_t.p_max_pu["wind1"], wind_profile)


# ---------------------------------------------------------------------------
# dynamic fuel prices in add_electricity
# ---------------------------------------------------------------------------


def test_dynamic_pricing_adds_each_plants_own_vom(fleet_network):
    n = fleet_network
    n.generators.loc["gas2", "vom_cost"] = np.nan  # falls back to the passed VOM
    hours = n.snapshots.get_level_values(1)
    fuel = pd.DataFrame({"CA": 8.0, "TX": 10.0}, index=hours)  # $/MWh_th

    apply_dynamic_pricing(n, "gas", "reeds_state", fuel, vom=2.5)

    mc = n.generators_t.marginal_cost
    assert mc["gas1"].to_numpy() == pytest.approx(8.0 / 0.4 + 3.0)
    assert mc["gas2"].to_numpy() == pytest.approx(10.0 / 0.4 + 2.5)


def test_dynamic_pricing_matches_state_names_and_eia_balancing_areas(fleet_network):
    """The wholesale price tables use state names and EIA balancing authorities."""
    n = fleet_network
    n.generators.loc[["gas1", "gas2"], "ba_eia"] = ["CISO", np.nan]
    hours = n.snapshots.get_level_values(1)

    # coarse to fine, as add_electricity applies them
    apply_dynamic_pricing(n, "gas", "state", pd.DataFrame({"California": 8.0, "Texas": 10.0}, index=hours))
    apply_dynamic_pricing(n, "gas", "balancing_area", pd.DataFrame({"CISO": 6.0}, index=hours))

    mc = n.generators_t.marginal_cost
    assert mc["gas1"].to_numpy() == pytest.approx(6.0 / 0.4 + 3.0)  # balancing area overrides state
    assert mc["gas2"].to_numpy() == pytest.approx(10.0 / 0.4 + 4.0)  # no balancing area, state price


# ---------------------------------------------------------------------------
# PUDL fuel cost / heat rate outlier filter
# ---------------------------------------------------------------------------


def test_zscore_filter_keeps_generators_without_spread():
    """One month, or a constant series, cannot be scored; a real outlier is still dropped."""
    data = pd.DataFrame(
        {
            "generator_name": ["single"] + ["flat"] * 3 + ["spread"] * 12,
            "fuel_cost_per_mwh": [20.0] + [15.0] * 3 + [10.0] * 11 + [100.0],
        },
    )

    kept = filter_outliers_zscore(data, "fuel_cost_per_mwh")

    assert kept.groupby("generator_name").size().to_dict() == {"flat": 3, "single": 1, "spread": 11}
    assert kept.fuel_cost_per_mwh.max() == 20.0
