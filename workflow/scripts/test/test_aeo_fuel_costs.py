"""
Test the AEO fuel costs of the cost data.

``build_cost_data`` reads the AEO2026 electric power prices of natural gas,
steam coal and distillate fuel oil (Table 3) for the case of ``costs: aeo:
scenario`` from the EIA API, in $/MMBtu of the dollar year named in their
unit, and writes them in $/MWh_th of the model's dollar year (2023 here).
The EIA layer is patched throughout.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import eia
from build_cost_data import build_aeo_fuel_costs
from money import deflator

# 2025$/MMBtu of each fuel in 2025, 2030 and 2050
AEO_PRICES = {"gas": [4.0, 5.0, 6.0], "coal": [2.0, 2.5, 3.0], "oil": [20.0, 21.0, 22.0]}
DOLLAR_YEAR = 2023
TO_MWH = 3.412 * deflator(2025, DOLLAR_YEAR)  # 2025$/MMBtu -> 2023$/MWh_th


@pytest.fixture
def patch_prices(monkeypatch):
    def _patch(units: str = "2025 $/MMBtu") -> None:
        def get_data(self, pivot=False):
            return pd.DataFrame(
                {"series-description": self.fuel, "value": AEO_PRICES[self.fuel], "units": units, "state": "USA"},
                index=pd.Index([2025, 2030, 2050], name="period"),
            )

        monkeypatch.setattr(eia.ProjectedFuelCost, "get_data", get_data)

    return _patch


def fuel_costs(aeo: pd.DataFrame) -> pd.Series:
    return aeo[aeo.parameter == "fuel_cost_real_per_mwhth"].set_index("pypsa-name").value


def test_url_targets_aeo2026_electric_power_prices():
    url = eia.ProjectedFuelCost("coal", "power", "KEY").get_api_call()

    assert url.startswith(f"{eia.API_BASE}aeo/2026/data/?api_key=KEY&")
    # AEO2026 has no Reference case: reference is the Counterfactual Baseline
    assert "facets[scenario][]=cb2026" in url
    assert "facets[seriesId][]=prce_real_elep_NA_stc_NA_NA_y13dlrpmmbtu" in url


def test_other_aeo2026_cases():
    url = eia.ProjectedFuelCost("gas", "power", "KEY", "high_oil_and_gas_supply").get_api_call()

    assert "facets[scenario][]=highogs" in url
    assert "facets[seriesId][]=prce_real_elep_NA_ng_NA_NA_y13dlrpmmbtu" in url


def test_unknown_case_is_rejected():
    with pytest.raises(eia.InputPropertyError):
        eia.ProjectedFuelCost("gas", "power", "KEY", "high").get_api_call()


def test_fuel_costs_are_2023_usd_per_mwh_th(patch_prices):
    patch_prices()

    costs = fuel_costs(build_aeo_fuel_costs("KEY", 2030, "reference", DOLLAR_YEAR))

    assert costs["natural_gas"] == pytest.approx(5.0 * TO_MWH)
    assert costs["coal"] == pytest.approx(2.5 * TO_MWH)
    assert costs["distillate_fuel_oil"] == pytest.approx(21.0 * TO_MWH)
    assert costs[["CCGT", "OCGT", "CCGT-95CCS", "CCGT-97CCS"]].tolist() == pytest.approx([5.0 * TO_MWH] * 4)
    assert costs[["coal-95CCS", "coal-99CCS"]].tolist() == pytest.approx([2.5 * TO_MWH] * 2)


def test_years_outside_the_aeo_take_the_nearest_year(patch_prices):
    patch_prices()

    assert fuel_costs(build_aeo_fuel_costs("KEY", 2024, "reference", DOLLAR_YEAR))["natural_gas"] == pytest.approx(
        4.0 * TO_MWH
    )


def test_other_dollar_year_is_converted(patch_prices):
    patch_prices(units="2024 $/MMBtu")

    costs = fuel_costs(build_aeo_fuel_costs("KEY", 2030, "reference", DOLLAR_YEAR))

    assert costs["natural_gas"] == pytest.approx(5.0 * 3.412 * deflator(2024, DOLLAR_YEAR))


def test_unit_without_dollar_year_is_rejected(patch_prices):
    patch_prices(units="$/MMBtu")

    with pytest.raises(ValueError, match="No dollar year"):
        build_aeo_fuel_costs("KEY", 2030, "reference", DOLLAR_YEAR)


def test_other_unit_is_rejected(patch_prices):
    patch_prices(units="2025 $/MWh")

    with pytest.raises(ValueError, match="\\$/MMBtu"):
        build_aeo_fuel_costs("KEY", 2030, "reference", DOLLAR_YEAR)


# every gas, coal and oil plant pays the AEO fuel costs


def test_oil_plants_take_the_distillate_cost(patch_prices):
    patch_prices()

    costs = fuel_costs(build_aeo_fuel_costs("KEY", 2030, "reference", DOLLAR_YEAR))

    assert costs["oil"] == pytest.approx(21.0 * TO_MWH)


def test_plants_pay_the_aeo_fuel_cost_of_each_period():
    import pypsa
    from _helpers import get_multiindex_snapshots
    from add_extra_components import apply_power_fuel_prices

    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 01:00", "inclusive": "both"},
        invest_periods=[2030, 2050],
    )
    n.set_investment_periods(periods=[2030, 2050])
    n.add("Bus", "p1")
    n.add(
        "Generator",
        ["p1 CCGT existing", "p1 coal-95CCS_2050", "p1 oil", "p1 nuclear existing", "p1 biomass existing", "p1 solar"],
        bus="p1",
        carrier=["CCGT", "coal-95CCS", "oil", "nuclear", "biomass", "solar"],
        efficiency=[0.5, 0.4, 0.25, 0.32, 0.25, 1],
        marginal_cost=[30, 40, 100, 10, 30, 0],
    )
    # an existing plant's own fuel cost is replaced
    n.generators["fuel_cost"] = [3.0, float("nan"), 15.0, 0.71, 2.0, 0.0]
    n.generators["vom_cost"] = [2.0, float("nan"), 3.0, 2.5, 5.0, 0.0]
    costs_dict = {
        period: pd.DataFrame(
            {"fuel_cost_real_per_mwhth": fuel, "opex_variable_per_mwh": [0.0, 4.0, 0.0, 0.0, 0.0]},
            index=["CCGT", "coal-95CCS", "oil", "nuclear", "biomass"],
        )
        for period, fuel in {2030: [10.0, 5.0, 50.0, 3.3, 19.9], 2050: [20.0, 6.0, 60.0, 3.3, 19.9]}.items()
    }

    apply_power_fuel_prices(n, costs_dict)

    mc = period_marginal_cost(n)
    assert mc["p1 CCGT existing"].tolist() == pytest.approx([2 + 10 / 0.5, 2 + 20 / 0.5])
    # without a VOM of its own, a plant takes the VOM of its carrier's cost data
    assert mc["p1 coal-95CCS_2050"].tolist() == pytest.approx([4 + 5 / 0.4, 4 + 6 / 0.4])
    assert mc["p1 oil"].tolist() == pytest.approx([3 + 50 / 0.25, 3 + 60 / 0.25])
    # nuclear and biomass plants pay the (ATB) fuel cost of the cost data too
    assert mc["p1 nuclear existing"].tolist() == pytest.approx([2.5 + 3.3 / 0.32] * 2)
    assert mc["p1 biomass existing"].tolist() == pytest.approx([5 + 19.9 / 0.25] * 2)
    assert mc["p1 solar"].tolist() == [0, 0]  # not a fuel plant: untouched
    assert n.generators.at["p1 CCGT existing", "marginal_cost"] == pytest.approx(2 + 10 / 0.5)
    # only plants whose cost changes between periods get a time series
    assert set(n.generators_t.marginal_cost.columns) == {"p1 CCGT existing", "p1 coal-95CCS_2050", "p1 oil"}


def period_marginal_cost(n):
    """Marginal cost of every generator in each investment period: its series, else its static value."""
    return n.get_switchable_as_dense("Generator", "marginal_cost").groupby(level="period").first()


def fuel_network(periods, plants):
    """A one-bus network over two hours of each period with fuel plants (name -> build_year, lifetime)."""
    import pypsa
    from _helpers import get_multiindex_snapshots

    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 01:00", "inclusive": "both"},
        invest_periods=periods,
    )
    n.set_investment_periods(periods=periods)
    n.add("Bus", "p1")
    for name, (carrier, build_year, lifetime) in plants.items():
        n.add("Generator", name, bus="p1", carrier=carrier, efficiency=0.5, build_year=build_year, lifetime=lifetime)
    n.generators["vom_cost"] = 1.0
    return n


def fuel_costs_by_period(prices):
    """Cost data of each period from {period: {carrier: fuel cost}}."""
    return {
        period: pd.DataFrame({"fuel_cost_real_per_mwhth": fuel, "opex_variable_per_mwh": 0.0})
        for period, fuel in ((p, pd.Series(f)) for p, f in prices.items())
    }


def test_single_period_needs_no_fuel_cost_series():
    from add_extra_components import apply_power_fuel_prices

    n = fuel_network([2030], {"CCGT existing": ("CCGT", 1990, 50), "OCGT 2030": ("OCGT", 2030, 30)})
    apply_power_fuel_prices(n, fuel_costs_by_period({2030: {"CCGT": 10.0, "OCGT": 12.0}}))

    assert n.generators_t.marginal_cost.empty
    assert n.generators.marginal_cost.tolist() == pytest.approx([1 + 10 / 0.5, 1 + 12 / 0.5])


def test_only_plants_with_changing_active_costs_get_a_series():
    from add_extra_components import apply_power_fuel_prices

    n = fuel_network(
        [2030, 2040, 2050],
        {
            "CCGT existing": ("CCGT", 1990, 100),  # active 2030-2050, price changes: series
            "CCGT 2030": ("CCGT", 2030, 15),  # active 2030 and 2040: series
            "CCGT 2050": ("CCGT", 2050, 30),  # active in 2050 only: static 2050 cost
            "coal retired": ("coal", 1980, 40),  # never active: static first-period cost
            "nuclear existing": ("nuclear", 1990, 100),  # active throughout, constant price: static
            "OCGT 2040": ("OCGT", 2040, 30),  # active 2040 and 2050 at the same price: static
        },
    )
    prices = {
        2030: {"CCGT": 10.0, "coal": 4.0, "nuclear": 3.0, "OCGT": 20.0},
        2040: {"CCGT": 14.0, "coal": 5.0, "nuclear": 3.0, "OCGT": 22.0},
        2050: {"CCGT": 18.0, "coal": 6.0, "nuclear": 3.0, "OCGT": 22.0},
    }
    apply_power_fuel_prices(n, fuel_costs_by_period(prices))

    assert list(n.generators_t.marginal_cost.columns) == ["CCGT existing", "CCGT 2030"]
    mc = period_marginal_cost(n)
    assert mc["CCGT existing"].tolist() == pytest.approx([21, 29, 37])
    assert mc["CCGT 2030"].tolist()[:2] == pytest.approx([21, 29])
    static = n.generators.marginal_cost
    assert static["CCGT 2050"] == pytest.approx(1 + 18.0 / 0.5)
    assert static["coal retired"] == pytest.approx(1 + 4.0 / 0.5)
    assert static["nuclear existing"] == pytest.approx(1 + 3.0 / 0.5)
    assert static["OCGT 2040"] == pytest.approx(1 + 22.0 / 0.5)


def test_active_period_costs_match_a_full_series():
    """In every period a plant is active in, its cost equals the full snapshot x plant table's."""
    import numpy as np
    from add_extra_components import apply_power_fuel_prices

    rng = np.random.default_rng(0)
    periods = [2030, 2035, 2040, 2050]
    carriers = ["CCGT", "OCGT", "coal", "oil", "nuclear", "biomass"]
    plants = {
        f"g{i}": (
            rng.choice(carriers),
            int(rng.choice([1970, 1995, 2030, 2035, 2040, 2050])),
            float(rng.choice([10, 20, 40, np.inf])),
        )
        for i in range(60)
    }
    n = fuel_network(periods, plants)
    n.generators["efficiency"] = rng.uniform(0.25, 0.6, len(n.generators))
    prices = {p: {c: float(rng.choice([3.0, 7.5, rng.uniform(1, 30)])) for c in carriers} for p in periods}
    costs_dict = fuel_costs_by_period(prices)

    apply_power_fuel_prices(n, costs_dict)

    # the full table: VOM + period fuel cost / efficiency for every plant and snapshot
    fuel = pd.DataFrame(prices).T
    full = fuel.loc[periods, n.generators.carrier].to_numpy() / n.generators.efficiency.to_numpy() + 1.0
    mc = period_marginal_cost(n).to_numpy()
    active = np.vstack([n.get_active_assets("Generator", p).to_numpy() for p in periods])
    assert active.any() and not active.all()
    np.testing.assert_array_equal(mc[active], full[active])


def test_ptc_after_period_fuel_costs():
    """The PTC comes off both the static cost and the series of a vintage."""
    from add_extra_components import apply_power_fuel_prices, apply_ptc

    n = fuel_network([2030, 2040], {"CCGT 2030": ("CCGT", 2030, 30), "CCGT 2040": ("CCGT", 2040, 30)})
    n.generators["p_nom_extendable"] = True
    costs_dict = fuel_costs_by_period({2030: {"CCGT": 10.0}, 2040: {"CCGT": 14.0}})
    for period, costs in costs_dict.items():
        costs["ptc_per_mwh"] = {2030: 5.0, 2040: 0.0}[period]

    apply_power_fuel_prices(n, costs_dict)
    apply_ptc(n, costs_dict)

    mc = period_marginal_cost(n)
    assert mc["CCGT 2030"].tolist() == pytest.approx([21 - 5, 29 - 5])  # series
    assert n.generators.at["CCGT 2040", "marginal_cost"] == pytest.approx(29)  # static, no credit
    assert "CCGT 2040" not in n.generators_t.marginal_cost


# nuclear, SMR and biomass pay the ATB fuel cost


def test_nuclear_and_biomass_pay_the_atb_fuel_cost():
    import numpy as np
    from build_cost_data import ATB_VALUE_COLS, process_atb_scenario

    atb = pd.DataFrame(
        {
            "pypsa-name": ["nuclear", "SMR", "biomass", "solar", "OCGT"],
            "fuel_cost_per_mwh": [10.0, 11.0, 78.0, np.nan, np.nan],  # USD/MWh_el
            "heat_rate_mmbtu_per_mwh": [10.5, 9.2, 13.3, np.nan, 9.7],
            "opex_variable_per_mwh": [2.0, 3.0, 5.0, 0.0, 5.0],
            "capex_per_kw": [7000.0, 9000.0, 5000.0, 1000.0, 1000.0],
            "cost_recovery_period_years": [30.0] * 5,
            "wacc_real": [0.05] * 5,
        },
    )
    for col in ATB_VALUE_COLS:
        if col not in atb:
            atb[col] = np.nan
    aeo = pd.DataFrame({"pypsa-name": ["natural_gas"], "parameter": ["fuel_cost_real_per_mwhth"], "value": [15.0]})

    costs = process_atb_scenario(atb, aeo).pivot(index="pypsa-name", columns="parameter", values="value")

    expected = [10.0 * 3.412 / 10.5, 11.0 * 3.412 / 9.2, 78.0 * 3.412 / 13.3]
    # the cost data is rounded to cents
    assert costs.loc[["nuclear", "SMR", "biomass"], "fuel_cost_real_per_mwhth"].tolist() == pytest.approx(
        expected, abs=0.01
    )
    # per MWh of electricity, the plants pay the ATB fuel cost
    assert costs.loc[["nuclear", "SMR", "biomass"], "fuel_cost"].tolist() == pytest.approx([10.0, 11.0, 78.0], abs=0.01)
