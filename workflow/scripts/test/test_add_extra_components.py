from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

from _helpers import calculate_annuity
from add_extra_components import (
    FLEXIBLE_ELECTROLYSIS_BUS_SUFFIX,
    attach_flexible_electrolysis,
    attach_tes_storageunits,
    carrier_new_build_buses,
)
from regional_cost import (
    SectorCosts,
    assign_bus_counties,
    bus_multiplier_table,
    carrier_multiplier,
    county_multiplier_table,
    overnight_delta_capital_cost,
    renewable_interconnection_capital_cost,
)


COSTS = Path(__file__).parents[2] / "repo_data" / "costs" / "simple_sector_costs.csv"


def network_with_ac_buses():
    n = pypsa.Network()
    n.set_investment_periods([2050])
    n.add("Bus", "ac_1", carrier="AC", x=1.0, y=2.0, country="US")
    n.add("Bus", "ac_2", carrier="AC", x=3.0, y=4.0, country="US")
    n.add("Bus", "dc_1", carrier="DC")
    # h2ptcreg is a custom bus attribute set on the DataFrame in build_base_network;
    # n.add() drops attributes that are not part of the Bus component schema.
    n.buses["h2ptcreg"] = pd.Series({"ac_1": "Texas", "ac_2": "California"})
    return n


def test_attach_tes_uses_reference_bus_store_link_topology():
    n = network_with_ac_buses()
    attach_tes_storageunits(n, str(COSTS))

    assert {"ac_1 tes", "ac_2 tes"}.issubset(n.buses.index)
    assert set(n.stores.index) == {"ac_1 tes", "ac_2 tes"}
    assert set(n.links.index) == {
        "ac_1 tes charger",
        "ac_2 tes charger",
        "ac_1 tes discharger",
        "ac_2 tes discharger",
    }
    assert n.stores.e_nom_extendable.all()
    assert n.stores.e_cyclic.all()
    assert np.allclose(n.stores.standing_loss, 0.0002)
    chargers = n.links[n.links.index.str.endswith(" tes charger")]
    dischargers = n.links[n.links.index.str.endswith(" tes discharger")]
    assert np.allclose(chargers.efficiency, 0.98)
    assert np.allclose(dischargers.efficiency, 0.5)
    assert np.allclose(dischargers.marginal_cost, 1.0)
    assert (n.links.carrier == "tes").all()


def test_attach_flexible_electrolysis_uses_reference_costs_and_zero_efficiency_links():
    n = network_with_ac_buses()
    n.add("Generator", "existing generator", bus="ac_1", carrier="gas", p_nom=100.0)
    n.add(
        "Generator",
        "new generator",
        bus="ac_2",
        carrier="solar",
        p_nom=0.0,
        p_nom_extendable=True,
        build_year=2050,
    )
    attach_flexible_electrolysis(
        n,
        {
            "enable": True,
            "annual_hydrogen_twh": 1512,
        },
        str(COSTS),
    )

    # One accounting H2 bus per h2ptcreg region, each fed by the AC buses in it.
    h2_buses = ["California" + FLEXIBLE_ELECTROLYSIS_BUS_SUFFIX, "Texas" + FLEXIBLE_ELECTROLYSIS_BUS_SUFFIX]
    assert set(n.buses.index[n.buses.carrier == "H2"]) == set(h2_buses)
    links = n.links[n.links.carrier == "electrolysis"]
    assert set(links.bus0) == {"ac_1", "ac_2"}
    assert links.loc["ac_1 flexible electrolysis", "bus1"] == "Texas" + FLEXIBLE_ELECTROLYSIS_BUS_SUFFIX
    assert links.loc["ac_2 flexible electrolysis", "bus1"] == "California" + FLEXIBLE_ELECTROLYSIS_BUS_SUFFIX
    assert links.p_nom_extendable.all()
    # Zero efficiency keeps the H2 accounting buses balanced without any sink component.
    assert np.allclose(links.efficiency, 0.0)
    assert n.generators[n.generators.bus.isin(h2_buses)].empty


def test_attach_flexible_electrolysis_skips_buses_without_an_h2ptcreg_region():
    n = network_with_ac_buses()
    n.buses.loc["ac_2", "h2ptcreg"] = np.nan

    attach_flexible_electrolysis(n, {"enable": True}, str(COSTS))

    links = n.links[n.links.carrier == "electrolysis"]
    assert set(links.bus0) == {"ac_1"}
    assert set(n.buses.index[n.buses.carrier == "H2"]) == {"Texas" + FLEXIBLE_ELECTROLYSIS_BUS_SUFFIX}


def test_attach_flexible_electrolysis_nation_uses_a_single_accounting_bus():
    n = network_with_ac_buses()

    attach_flexible_electrolysis(
        n,
        {"enable": True, "accounting_region": "nation"},
        str(COSTS),
    )

    national_bus = "nation" + FLEXIBLE_ELECTROLYSIS_BUS_SUFFIX
    assert set(n.buses.index[n.buses.carrier == "H2"]) == {national_bus}
    links = n.links[n.links.carrier == "electrolysis"]
    assert set(links.bus0) == {"ac_1", "ac_2"}
    assert set(links.bus1) == {national_bus}


def test_attach_flexible_electrolysis_nation_still_skips_buses_without_an_h2ptcreg_region():
    n = network_with_ac_buses()
    n.buses.loc["ac_2", "h2ptcreg"] = np.nan

    attach_flexible_electrolysis(
        n,
        {"enable": True, "accounting_region": "nation"},
        str(COSTS),
    )

    links = n.links[n.links.carrier == "electrolysis"]
    assert set(links.bus0) == {"ac_1"}


def test_attach_flexible_electrolysis_rejects_unknown_accounting_region():
    n = network_with_ac_buses()

    with pytest.raises(ValueError, match="accounting_region"):
        attach_flexible_electrolysis(
            n,
            {"enable": True, "accounting_region": "state"},
            str(COSTS),
        )


def test_nuclear_new_build_buses_are_restricted_to_existing_nuclear_sites():
    n = network_with_ac_buses()
    n.add("Generator", "existing nuclear", bus="ac_2", carrier="nuclear", p_nom=100.0)

    buses = carrier_new_build_buses(
        n,
        "nuclear",
        n.buses.index[n.buses.carrier == "AC"],
        pd.Index([]),
    )

    assert buses.tolist() == ["ac_2"]


def test_nuclear_new_build_buses_are_empty_without_existing_nuclear_sites():
    n = network_with_ac_buses()

    buses = carrier_new_build_buses(
        n,
        "nuclear",
        n.buses.index[n.buses.carrier == "AC"],
        pd.Index([]),
    )

    assert buses.empty


def test_flexible_electrolysis_rejects_nonpositive_electricity_input(tmp_path):
    n = network_with_ac_buses()
    costs = pd.read_csv(COSTS)
    mask = (costs["pypsa-name"] == "h2 electrolysis") & (costs["parameter"] == "electricity-input")
    costs.loc[mask, "value"] = 0.0
    broken = tmp_path / "simple_sector_costs.csv"
    costs.to_csv(broken, index=False)

    with pytest.raises(ValueError, match="must be positive"):
        attach_flexible_electrolysis(n, {"enable": True}, str(broken))


###
# Regional overnight-capex multipliers (regional_cost.py)
###


def _county_grid(path):
    """Write a 2x2 grid of square test counties spanning lon -100..-98, lat 40..42."""
    import geopandas as gpd
    from shapely.geometry import box

    counties = gpd.GeoDataFrame(
        {
            "GEOID": ["11111", "22222", "33333", "44444"],
            "geometry": [
                box(-100, 40, -99, 41),  # SW
                box(-99, 40, -98, 41),  # SE
                box(-100, 41, -99, 42),  # NW
                box(-99, 41, -98, 42),  # NE
            ],
        },
        crs="EPSG:4326",
    )
    counties.to_file(path, driver="GeoJSON")
    return str(path)


def _buses_for_grid():
    return pd.DataFrame(
        {"x": [-99.5, -98.5, -97.0], "y": [40.5, 41.5, 40.5]},
        index=["inside_sw", "inside_ne", "outside_east"],
    )


def test_renewable_interconnection_capital_cost_excludes_atb_grid_connection():
    annuity = 0.08
    actual = renewable_interconnection_capital_cost(
        capex_overnight_per_kw=1000.0,
        capex_construction_finance_factor=100.0,
        opex_fixed_per_kw=20.0,
        cost_trans_usd_per_mw=50_000.0,
        annuity_factor=annuity,
    )
    # Only overnight + construction finance are annuitized alongside the ReEDS
    # interconnection cost; ATB's own grid connection adder never appears.
    expected = annuity * ((1000.0 + 100.0) * 1e3 + 50_000.0) + 20.0 * 1e3
    assert actual == pytest.approx(expected)


def test_overnight_delta_capital_cost_is_identity_at_unit_multiplier():
    assert overnight_delta_capital_cost(123_456.0, 1000.0, 0.08, 1.0) == pytest.approx(123_456.0)
    # A 20% overnight premium adds only the annuitized overnight delta.
    assert overnight_delta_capital_cost(123_456.0, 1000.0, 0.08, 1.2) == pytest.approx(
        123_456.0 + 0.08 * 0.2 * 1000.0 * 1e3,
    )


def test_county_multiplier_table_expands_pipe_delimited_groups():
    reg_cap = pd.DataFrame(
        {"A|B": [0.10, -0.05], "NUCLEAR": [0.20, 0.0]},
        index=pd.Index(["p01001", "p01003"], name="r"),
    )
    table = county_multiplier_table(reg_cap, {"A|B": ["CCGT", "OCGT"], "NUCLEAR": ["nuclear"]})

    assert set(table.columns) == {"CCGT", "OCGT", "nuclear"}
    assert table.at["p01001", "CCGT"] == pytest.approx(1.10)
    assert table.at["p01001", "OCGT"] == pytest.approx(1.10)
    assert table.at["p01003", "CCGT"] == pytest.approx(0.95)
    assert table.at["p01001", "nuclear"] == pytest.approx(1.20)


def test_county_multiplier_table_skips_unknown_columns_and_rejects_empty_mapping():
    reg_cap = pd.DataFrame({"NUCLEAR": [0.2]}, index=pd.Index(["p01001"], name="r"))

    table = county_multiplier_table(reg_cap, {"NOT_A_COLUMN": ["CCGT"], "NUCLEAR": ["nuclear"]})
    assert list(table.columns) == ["nuclear"]

    with pytest.raises(ValueError, match="column_to_tech is required"):
        county_multiplier_table(reg_cap, {})


def test_assign_bus_counties_uses_containing_county_then_nearest(tmp_path):
    shapes = _county_grid(tmp_path / "counties.geojson")

    assigned = assign_bus_counties(_buses_for_grid(), shapes)

    # A point inside a polygon takes that polygon; the point east of the grid falls
    # back to the nearest one.
    assert assigned["inside_sw"] == "p11111"
    assert assigned["inside_ne"] == "p44444"
    assert assigned["outside_east"] == "p22222"


def test_assign_bus_counties_restricts_candidates_to_valid_counties(tmp_path):
    shapes = _county_grid(tmp_path / "counties.geojson")

    assigned = assign_bus_counties(_buses_for_grid(), shapes, valid_counties=["p11111", "p33333"])

    # Every bus must land on a county the cost table actually covers, including the
    # one whose own containing county was excluded.
    assert set(assigned) <= {"p11111", "p33333"}
    assert assigned["inside_sw"] == "p11111"
    assert assigned["inside_ne"] == "p33333"
    assert assigned["outside_east"] == "p11111"


def test_bus_multiplier_table_defaults_unmapped_carriers_and_buses(tmp_path):
    shapes = _county_grid(tmp_path / "counties.geojson")
    county_table = pd.DataFrame(
        {"CCGT": [1.10, 0.95]},
        index=pd.Index(["p11111", "p44444"], name="r"),
    )

    table = bus_multiplier_table(_buses_for_grid(), county_table, shapes)

    assert table.at["inside_sw", "CCGT"] == pytest.approx(1.10)
    assert table.at["inside_ne", "CCGT"] == pytest.approx(0.95)
    # p22222 contains outside_east's nearest polygon edge but carries no cost row,
    # so the bus snaps to the nearest *covered* county -- p44444, whose corner is
    # ~100 km away against ~170 km for p11111.
    assert table.at["outside_east", "CCGT"] == pytest.approx(0.95)

    # An unmapped carrier, and a disabled feature, both degrade to the scalar 1.0.
    assert carrier_multiplier(table, "nuclear", table.index) == 1.0
    assert carrier_multiplier(None, "CCGT", table.index) == 1.0


def test_sector_costs_overnight_multiplier_scales_annuity_only():
    costs = SectorCosts(str(COSTS))
    base = costs.annualized("h2 electrolysis")

    assert costs.annualized("h2 electrolysis", overnight_multiplier=1.0) == pytest.approx(base)

    annuity = calculate_annuity(costs.value("h2 electrolysis", "crp"), costs.wacc_real("h2 electrolysis"))
    fom_rate = costs.value("h2 electrolysis", "FOM") / 100
    scaled = costs.annualized("h2 electrolysis", overnight_multiplier=1.25)
    # base = (a + fom) * I  =>  I = base / (a + fom); the delta touches only `a`.
    assert scaled - base == pytest.approx(0.25 * annuity * base / (annuity + fom_rate))


def test_flexible_electrolysis_applies_per_bus_overnight_multiplier():
    n = network_with_ac_buses()
    bus_multipliers = pd.DataFrame({"electrolysis": [1.0, 1.5]}, index=["ac_1", "ac_2"])

    attach_flexible_electrolysis(n, {"enable": True}, str(COSTS), bus_multipliers=bus_multipliers)

    costs = SectorCosts(str(COSTS))
    base = costs.annualized("h2 electrolysis")
    annuity = calculate_annuity(costs.value("h2 electrolysis", "crp"), costs.wacc_real("h2 electrolysis"))
    fom_rate = costs.value("h2 electrolysis", "FOM") / 100

    links = n.links[n.links.carrier == "electrolysis"]
    assert len(links) == 2
    unscaled = links.loc[links.bus0 == "ac_1", "capital_cost"].iloc[0]
    scaled = links.loc[links.bus0 == "ac_2", "capital_cost"].iloc[0]
    assert unscaled == pytest.approx(base)
    assert scaled == pytest.approx(base + 0.5 * annuity * base / (annuity + fom_rate))


def test_tes_applies_per_bus_overnight_multiplier():
    n = network_with_ac_buses()
    bus_multipliers = pd.DataFrame({"tes": [1.0, 1.5]}, index=["ac_1", "ac_2"])

    attach_tes_storageunits(n, str(COSTS), bus_multipliers=bus_multipliers)

    costs = SectorCosts(str(COSTS))
    annuity = calculate_annuity(costs.value("TES", "crp"), costs.wacc_real("TES"))
    fom_rate = costs.value("TES", "FOM") / 100

    for component, name_suffix, param in (
        ("stores", " tes", "energy_cost"),
        ("links", " tes charger", "charge_cost"),
        ("links", " tes discharger", "discharge_cost"),
    ):
        df = getattr(n, component)
        unit = costs.value("TES", param)
        assert df.at["ac_1" + name_suffix, "capital_cost"] == pytest.approx((annuity + fom_rate) * unit)
        assert df.at["ac_2" + name_suffix, "capital_cost"] == pytest.approx((annuity * 1.5 + fom_rate) * unit)
