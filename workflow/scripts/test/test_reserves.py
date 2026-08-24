"""
Test the reserves constraints functionality.

This module contains tests for the reserve margin constraints in PyPSA-USA.
"""

import os
import sys

import numpy as np
import pandas as pd
import pypsa
import pytest
from pypsa.descriptors import (
    get_activity_mask,
)
from pypsa.descriptors import (
    get_switchable_as_dense as get_as_dense,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from opts.reserves import (
    ERM_REQUIREMENT,
    add_ERM_constraints,
    erm_requirement_name,
    store_ERM_duals,
)


def _erm_solution(n, name):
    """Reserve-state solution of a variable, as a snapshot-by-asset frame."""
    return n.model.solution[name].to_pandas()


@pytest.fixture
def reserve_margin_network(base_network):
    """
    Adapt base network for ERM constraint testing.

    Extends the base network with parameters relevant to reserve margin testing.
    """
    n = base_network.copy()

    # Create a higher peak in load profile for reserve margin testing
    load_profile_region1 = pd.Series(
        np.concatenate([np.linspace(800, 1200, 12), np.linspace(1200, 800, 12)]),
        index=n.snapshots,
    )

    load_profile_region2 = pd.Series(
        np.concatenate([np.linspace(500, 700, 18), np.linspace(700, 500, 6)]),
        index=n.snapshots,
    )

    # Update load profiles
    n.loads_t.p_set.loc[:, "load1"] = load_profile_region1
    n.loads_t.p_set.loc[:, "load2"] = load_profile_region1 * 0.75
    n.loads_t.p_set.loc[:, "load3"] = load_profile_region2

    return n


@pytest.fixture
def meshed_reserve_network(reserve_margin_network):
    """
    Close the z1-z2-z3 path into a loop, and give NERC1 a second boundary branch.

    The base network is radial, so z1 has only one line leaving it. A third line makes
    z1 the endpoint of two crossing branches, which is what the boundary-flow tests
    need to distinguish a signed sum from a single term.
    """
    n = reserve_margin_network.copy()
    n.add(
        "Line",
        "line3",
        bus0="z1",
        bus1="z3",
        carrier="AC",
        x=0.15,
        r=0.015,
        s_nom=400,
        s_nom_min=400,
        capital_cost=300,
        s_nom_extendable=True,
    )
    return n


@pytest.fixture
def multi_period_reserve_network(multi_period_base_network):
    """
    Adapt multi-period base network for ERM constraint testing.

    Extends the multi-period base network with peaky load profiles.
    """
    n = multi_period_base_network.copy()

    load_profile_region1 = pd.Series(
        np.tile(
            np.concatenate([np.linspace(800, 1200, 12), np.linspace(1200, 800, 12)]),
            2,  # two periods
        ),
        index=n.snapshots,
    )

    load_profile_region2 = pd.Series(
        np.tile(
            np.concatenate([np.linspace(500, 700, 18), np.linspace(700, 500, 6)]),
            2,
        ),
        index=n.snapshots,
    )

    n.loads_t.p_set.loc[:, "load1"] = load_profile_region1
    n.loads_t.p_set.loc[:, "load2"] = load_profile_region1 * 0.75
    n.loads_t.p_set.loc[:, "load3"] = load_profile_region2

    return n


def _bus_demand(n, buses):
    return n.loads_t.p_set.T.groupby(n.loads.bus).sum().T.reindex(columns=buses, fill_value=0.0)


def test_erm_increases_capacity(reserve_margin_network):
    """Test that ERM constraint of 0.14 results in more capacity built than no ERM."""
    # First run without ERM constraints
    n_no_erm = reserve_margin_network.copy()
    n_no_erm.optimize(solver_name="highs", multi_investment_periods=True)

    total_gen_capacity_no_erm = n_no_erm.generators.p_nom_opt.sum()
    total_storage_capacity_no_erm = n_no_erm.storage_units.p_nom_opt.sum()
    total_capacity_no_erm = total_gen_capacity_no_erm + total_storage_capacity_no_erm

    # Now run with ERM constraint of 0.14
    n_with_erm = reserve_margin_network.copy()

    def extra_functionality(n, snapshots):
        add_ERM_constraints(n, snapshots, regional_erm_data={"all": 0.14})

    n_with_erm.optimize(
        solver_name="highs",
        multi_investment_periods=True,
        extra_functionality=extra_functionality,
    )

    total_gen_capacity_with_erm = n_with_erm.generators.p_nom_opt.sum()
    total_storage_capacity_with_erm = n_with_erm.storage_units.p_nom_opt.sum()
    total_capacity_with_erm = total_gen_capacity_with_erm + total_storage_capacity_with_erm

    assert total_capacity_with_erm > total_capacity_no_erm, (
        f"ERM constraint (erm=0.14) should result in more capacity built. "
        f"Without ERM: {total_capacity_no_erm:.2f} MW, With ERM: {total_capacity_with_erm:.2f} MW"
    )

    assert n_with_erm.objective > n_no_erm.objective, (
        f"ERM constraint should increase system cost. "
        f"Without ERM: {n_no_erm.objective:.2f}, With ERM: {n_with_erm.objective:.2f}"
    )


def test_erm_increases_capacity_no_expandable_transmission(reserve_margin_network):
    """Test that ERM constraint of 0.14 results in more capacity built than no ERM, with no expandable lines or links."""

    def disable_transmission_expansion(n):
        """Disable expansion of lines and links, and add firm generation at z2 for feasibility."""
        n.lines["s_nom_extendable"] = False
        n.links["p_nom_extendable"] = False
        n.lines["s_nom"] = 2000
        n.links["p_nom"] = 2000
        n.add(
            "Generator",
            "gas_z2",
            bus="z2",
            p_nom=0,
            p_nom_extendable=True,
            carrier="gas",
            capital_cost=500,
            marginal_cost=20,
            p_max_pu=1.0,
            p_nom_max=5000,
            build_year=2030,
            lifetime=20,
        )
        return n

    # First run without ERM constraints
    n_no_erm = reserve_margin_network.copy()
    n_no_erm = disable_transmission_expansion(n_no_erm)
    n_no_erm.optimize(solver_name="highs", multi_investment_periods=True)

    total_gen_capacity_no_erm = n_no_erm.generators.p_nom_opt.sum()
    total_storage_capacity_no_erm = n_no_erm.storage_units.p_nom_opt.sum()
    total_capacity_no_erm = total_gen_capacity_no_erm + total_storage_capacity_no_erm

    # Now run with ERM constraint of 0.14
    n_with_erm = reserve_margin_network.copy()
    n_with_erm = disable_transmission_expansion(n_with_erm)

    def extra_functionality(n, snapshots):
        add_ERM_constraints(n, snapshots, regional_erm_data={"all": 0.14})

    n_with_erm.optimize(
        solver_name="highs",
        multi_investment_periods=True,
        extra_functionality=extra_functionality,
    )

    total_gen_capacity_with_erm = n_with_erm.generators.p_nom_opt.sum()
    total_storage_capacity_with_erm = n_with_erm.storage_units.p_nom_opt.sum()
    total_capacity_with_erm = total_gen_capacity_with_erm + total_storage_capacity_with_erm

    assert total_capacity_with_erm > total_capacity_no_erm, (
        f"ERM constraint (erm=0.14) should result in more capacity built (no expandable transmission). "
        f"Without ERM: {total_capacity_no_erm:.2f} MW, With ERM: {total_capacity_with_erm:.2f} MW"
    )

    assert n_with_erm.objective > n_no_erm.objective, (
        f"ERM constraint should increase system cost (no expandable transmission). "
        f"Without ERM: {n_no_erm.objective:.2f}, With ERM: {n_with_erm.objective:.2f}"
    )


def test_multi_period_erm_increases_capacity(multi_period_reserve_network):
    """Test that ERM increases capacity across multiple investment periods."""
    # Without ERM
    n_no_erm = multi_period_reserve_network.copy()
    n_no_erm.optimize(solver_name="highs", multi_investment_periods=True)

    total_capacity_no_erm = n_no_erm.generators.p_nom_opt.sum() + n_no_erm.storage_units.p_nom_opt.sum()

    # With ERM
    n_with_erm = multi_period_reserve_network.copy()

    def extra_functionality(n, snapshots):
        add_ERM_constraints(n, snapshots, regional_erm_data={"all": 0.15})

    n_with_erm.optimize(
        solver_name="highs",
        multi_investment_periods=True,
        extra_functionality=extra_functionality,
    )

    total_capacity_with_erm = n_with_erm.generators.p_nom_opt.sum() + n_with_erm.storage_units.p_nom_opt.sum()

    assert total_capacity_with_erm > total_capacity_no_erm, (
        f"ERM should increase capacity in multi-period. "
        f"Without: {total_capacity_no_erm:.2f}, With: {total_capacity_with_erm:.2f}"
    )


def _solution_lookup(n):
    """Map every variable label of a solved model to its value."""
    values = {}
    for name in n.model.variables:
        labels = np.asarray(n.model.variables[name].labels).ravel()
        solution = np.asarray(n.model.solution[name]).ravel()
        for label, value in zip(labels, solution):
            if label != -1:
                values[int(label)] = float(value)
    return values


def _requirement_row(n, region, snapshot):
    """Coefficients of one ERM requirement row, keyed by variable label."""
    constraint = n.model.constraints[erm_requirement_name(region)]
    selection = {"snapshot": snapshot}
    labels = constraint.vars.sel(selection).values.flat
    coeffs = constraint.coeffs.sel(selection).values.flat
    return {int(label): float(coeff) for label, coeff in zip(labels, coeffs) if label != -1}


def _label(n, name, **selection):
    """Label of a single variable of the model."""
    return int(n.model.variables[name].labels.sel(selection).item())


def _firm_capacity(n, bus, snapshots):
    """Available capacity of the non-extendable generators at ``bus``, per snapshot."""
    fixed_i = n.generators.index[(n.generators.bus == bus) & ~n.generators.p_nom_extendable]
    if fixed_i.empty:
        return pd.Series(0.0, index=snapshots)
    available = get_as_dense(n, "Generator", "p_max_pu", snapshots)[fixed_i].where(
        get_activity_mask(n, "Generator", snapshots, fixed_i),
        0.0,
    )
    return available.mul(n.generators.p_nom[fixed_i]).sum(axis=1)


def test_erm_requirement_is_enforced_per_region(reserve_margin_network):
    """One row per region and snapshot, and every row holds at the optimum."""
    n = reserve_margin_network.copy()

    erm_value = 0.30

    def extra_functionality(n, snapshots):
        add_ERM_constraints(n, snapshots, regional_erm_data={"NERC1": erm_value, "NERC2": erm_value})

    status, condition = n.optimize(
        solver_name="highs",
        multi_investment_periods=True,
        extra_functionality=extra_functionality,
    )
    assert status == "ok" and condition == "optimal", f"Optimization failed: {status}/{condition}"

    constraints = {region: n.model.constraints[erm_requirement_name(region)] for region in ("NERC1", "NERC2")}
    for region, constraint in constraints.items():
        assert set(constraint.coords["snapshot"].values) == set(n.snapshots)
        # A regional requirement, not a per-bus one: one row per snapshot.
        assert constraint.labels.size == len(n.snapshots)

    buses = n.buses.index[n.buses.carrier == "AC"]
    demand = _bus_demand(n, buses)
    values = _solution_lookup(n)

    for region, constraint in constraints.items():
        members = buses[n.buses.nerc_reg[buses] == region]

        # Fixed capacity is a constant and rides on the right-hand side.
        firm = sum((_firm_capacity(n, bus, n.snapshots) for bus in members), pd.Series(0.0, index=n.snapshots))
        expected_rhs = demand[members].sum(axis=1) * (1 + erm_value) - firm
        assert np.allclose(constraint.rhs.to_pandas(), expected_rhs, atol=1e-6), (
            f"{region} right-hand side should be (1 + erm) * demand less its firm capacity"
        )

        for snapshot in n.snapshots:
            row = _requirement_row(n, region, snapshot)
            lhs = sum(coeff * values[label] for label, coeff in row.items())
            rhs = float(constraint.rhs.sel(snapshot=snapshot).item())
            assert lhs >= rhs - 1e-4, f"{region} requirement violated at {snapshot}"


def test_erm_generator_coefficient_is_availability(reserve_margin_network):
    """An extendable generator enters its region's row at p_max_pu of that snapshot."""
    n = reserve_margin_network.copy()
    n.optimize.create_model(multi_investment_periods=True)
    add_ERM_constraints(n, n.snapshots, regional_erm_data={"NERC1": 0.15})

    p_max_pu = get_as_dense(n, "Generator", "p_max_pu", n.snapshots)
    extendable_at_z1 = n.generators.index[(n.generators.bus == "z1") & n.generators.p_nom_extendable]
    assert not extendable_at_z1.empty, "The fixture should have extendable capacity at z1"

    varying = [g for g in extendable_at_z1 if p_max_pu[g].nunique() > 1]
    assert varying, "At least one generator should have a time-varying availability"

    for snapshot in (n.snapshots[0], n.snapshots[9], n.snapshots[-1]):
        row = _requirement_row(n, "NERC1", snapshot)
        for gen in extendable_at_z1:
            label = _label(n, "Generator-p_nom", **{"Generator-ext": gen})
            assert row[label] == pytest.approx(p_max_pu.at[snapshot, gen]), (
                f"{gen} should enter the row at its availability factor"
            )

    # Generators outside the region stay out of the row.
    row = _requirement_row(n, "NERC1", n.snapshots[0])
    for gen in n.generators.index[(n.generators.bus != "z1") & n.generators.p_nom_extendable]:
        label = _label(n, "Generator-p_nom", **{"Generator-ext": gen})
        assert label not in row, f"{gen} is outside NERC1 and should not appear in its row"


def test_erm_load_shedding_reduces_served_demand_without_counting_as_capacity(
    reserve_margin_network,
):
    """Shedding dispatch relaxes ERM, while its 10 GW rating earns no firm credit."""
    erm_value = 0.15

    baseline = reserve_margin_network.copy()
    baseline.optimize.create_model(multi_investment_periods=True)
    add_ERM_constraints(
        baseline,
        baseline.snapshots,
        regional_erm_data={"NERC1": erm_value},
    )
    baseline_rhs = baseline.model.constraints[
        erm_requirement_name("NERC1")
    ].rhs.to_pandas()

    n = reserve_margin_network.copy()
    if "load" not in n.carriers.index:
        n.add("Carrier", "load")
    n.add(
        "Generator",
        "z1 load",
        bus="z1",
        carrier="load",
        p_nom=1e4,
        p_nom_extendable=False,
        marginal_cost=1e5,
    )
    n.optimize.create_model(multi_investment_periods=True)
    add_ERM_constraints(
        n,
        n.snapshots,
        regional_erm_data={"NERC1": erm_value},
    )

    # Excluding the 10 GW shedding rating leaves the constant RHS unchanged.
    constraint = n.model.constraints[erm_requirement_name("NERC1")]
    pd.testing.assert_series_equal(
        constraint.rhs.to_pandas(),
        baseline_rhs,
    )

    # Actual shedding subtracts from demand before the margin is applied:
    # physical capacity + (1 + erm) * shedding >= (1 + erm) * gross demand.
    for snapshot in (n.snapshots[0], n.snapshots[-1]):
        row = _requirement_row(n, "NERC1", snapshot)
        shedding = _label(
            n,
            "Generator-p",
            snapshot=snapshot,
            Generator="z1 load",
        )
        assert row[shedding] == pytest.approx(1.0 + erm_value)


def test_erm_boundary_flow_is_signed_by_direction(meshed_reserve_network):
    """A branch crossing the boundary enters with the sign of the in-region end."""
    n = meshed_reserve_network.copy()
    n.optimize.create_model(multi_investment_periods=True)
    add_ERM_constraints(n, n.snapshots, regional_erm_data={"NERC1": 0.15, "NERC2": 0.15})

    snapshot = n.snapshots[0]
    row_nerc1 = _requirement_row(n, "NERC1", snapshot)
    row_nerc2 = _requirement_row(n, "NERC2", snapshot)
    flow = {line: _label(n, "Line-s", snapshot=snapshot, Line=line) for line in n.lines.index}

    # NERC1 is z1 alone. line1 (z1->z2) and line3 (z1->z3) leave it from bus0.
    assert row_nerc1[flow["line1"]] == pytest.approx(-1.0)
    assert row_nerc1[flow["line3"]] == pytest.approx(-1.0)
    assert flow["line2"] not in row_nerc1, "line2 has neither end in NERC1"

    # NERC1 and NERC2 partition the network, so the same branches arrive at bus1.
    assert row_nerc2[flow["line1"]] == pytest.approx(1.0)
    assert row_nerc2[flow["line3"]] == pytest.approx(1.0)
    assert flow["line2"] not in row_nerc2, "line2 is internal to NERC2 and nets out"

    # link1 runs z1 -> z3, so it leaves NERC1 undiminished and arrives in NERC2
    # scaled by the link efficiency.
    link_flow = _label(n, "Link-p", snapshot=snapshot, Link="link1")
    assert row_nerc1[link_flow] == pytest.approx(-1.0)
    assert row_nerc2[link_flow] == pytest.approx(n.links.at["link1", "efficiency"])


def test_erm_boundary_flow_charges_half_the_branch_loss(meshed_reserve_network):
    """Where the base state models losses, each end of a branch pays half of it."""
    n = meshed_reserve_network.copy()
    n.optimize.create_model(multi_investment_periods=True, transmission_losses=1)
    add_ERM_constraints(n, n.snapshots, regional_erm_data={"NERC1": 0.15, "NERC2": 0.15})

    snapshot = n.snapshots[0]
    for region in ("NERC1", "NERC2"):
        row = _requirement_row(n, region, snapshot)
        for line in ("line1", "line3"):
            label = _label(n, "Line-loss", snapshot=snapshot, Line=line)
            assert row[label] == pytest.approx(-0.5), (
                f"{line} should charge half its loss to {region}, as the nodal balance does"
            )


def test_erm_all_expands_to_every_nerc_region(reserve_margin_network, caplog):
    """``all`` is shorthand for every NERC region, not one nationwide row."""
    n = reserve_margin_network.copy()
    n.optimize.create_model(multi_investment_periods=True)

    with caplog.at_level("INFO"):
        add_ERM_constraints(n, n.snapshots, regional_erm_data={"all": 0.15})

    assert {erm_requirement_name("NERC1"), erm_requirement_name("NERC2")} <= set(n.model.constraints)
    assert any("expanded to 2 NERC regions" in record.message for record in caplog.records)

    # A nationwide row would have no boundary at all, so the branches would drop out.
    row = _requirement_row(n, "NERC1", n.snapshots[0])
    assert _label(n, "Line-s", snapshot=n.snapshots[0], Line="line1") in row


def test_erm_explicit_region_overrides_all(reserve_margin_network):
    """An explicit region key wins over the value inherited from ``all``."""
    n = reserve_margin_network.copy()
    n.optimize.create_model(multi_investment_periods=True)
    add_ERM_constraints(n, n.snapshots, regional_erm_data={"all": 0.15, "NERC1": 0.40})

    buses = n.buses.index[n.buses.carrier == "AC"]
    demand = _bus_demand(n, buses)

    for region, erm_value in [("NERC1", 0.40), ("NERC2", 0.15)]:
        constraint = n.model.constraints[erm_requirement_name(region)]
        members = buses[n.buses.nerc_reg[buses] == region]
        firm = sum((_firm_capacity(n, bus, n.snapshots) for bus in members), pd.Series(0.0, index=n.snapshots))
        expected = demand[members].sum(axis=1) * (1 + erm_value) - firm
        assert np.allclose(constraint.rhs.to_pandas(), expected, atol=1e-6)


def test_erm_discharge_is_capped_by_rating_and_state_of_charge(reserve_margin_network):
    """The only reserve state left is the dischargers', bounded by rating and energy."""
    n = reserve_margin_network.copy()

    def extra_functionality(n, snapshots):
        add_ERM_constraints(n, snapshots, regional_erm_data={"all": 0.20})

    status, condition = n.optimize(
        solver_name="highs",
        multi_investment_periods=True,
        extra_functionality=extra_functionality,
    )
    assert status == "ok" and condition == "optimal", f"Optimization failed: {status}/{condition}"

    discharge = _erm_solution(n, "StorageUnit-p_dispatch_ERM")
    assert (discharge >= -1e-6).all().all(), "Reserve discharge should be non-negative"

    rating = n.storage_units.p_nom_opt * get_as_dense(n, "StorageUnit", "p_max_pu", n.snapshots)
    assert (discharge <= rating[discharge.columns] + 1e-4).all().all(), (
        "Reserve discharge should respect the rating of the installed capacity"
    )

    energy_limit = n.storage_units_t.state_of_charge * n.storage_units.efficiency_dispatch
    assert (discharge <= energy_limit[discharge.columns] + 1e-4).all().all(), (
        "Reserve discharge should be backed by the base-state state of charge"
    )

    # Nothing else is duplicated any more.
    for gone in ("Generator-p_ERM", "Line-s_ERM", "Line-loss_ERM", "ERM_bus_reserve"):
        assert gone not in n.model.variables, f"{gone} should no longer exist"
    for gone in ("Kirchhoff-Voltage-Law_ERM", "ERM_nodal_balance"):
        assert gone not in n.model.constraints, f"{gone} should no longer exist"


def test_multiple_non_overlapping_erms(reserve_margin_network):
    """Different regions should impose different reserve requirements."""
    n = reserve_margin_network.copy()

    erm_dict = {"NERC1": 0.15, "NERC2": 0.30}

    def extra_functionality(n, snapshots):
        add_ERM_constraints(n, snapshots, regional_erm_data=erm_dict)

    status, condition = n.optimize(
        solver_name="highs",
        multi_investment_periods=True,
        extra_functionality=extra_functionality,
    )
    assert status == "ok" and condition == "optimal", f"Optimization failed: {status}/{condition}"

    store_ERM_duals(n)

    assert {erm_requirement_name("NERC1"), erm_requirement_name("NERC2")} <= set(n.model.constraints), (
        "A per-region reserve requirement should exist"
    )

    buses = n.buses.index[n.buses.carrier == "AC"]
    demand = _bus_demand(n, buses)

    # z1 is in NERC1 (0.15), z2/z3 are in NERC2 (0.30)
    for region, erm_value in erm_dict.items():
        constraint = n.model.constraints[erm_requirement_name(region)]
        members = buses[n.buses.nerc_reg[buses] == region]
        firm = sum((_firm_capacity(n, bus, n.snapshots) for bus in members), pd.Series(0.0, index=n.snapshots))
        expected = demand[members].sum(axis=1) * (1 + erm_value) - firm
        assert np.allclose(constraint.rhs.to_pandas(), expected, atol=1e-6), (
            f"{region} should carry a {erm_value:.0%} margin on its own demand"
        )

    assert hasattr(n, "erm_region_price"), "Regional reserve prices should be stored in n.erm_region_price"
    assert sorted(n.erm_region_price.columns) == ["NERC1", "NERC2"]
    assert not n.erm_region_price.isnull().values.any()

    # The removed reserve state leaves no price behind either.
    assert "erm_price" not in n.buses_t
    assert "erm_reserve" not in n.buses_t


def test_tes_discharge_reserve_is_limited_by_connected_store_energy():
    """TES discharge reserve should be capped by the energy held in the connected store."""
    hours = pd.date_range("2030-01-01 00:00", periods=3, freq="h")
    snapshots = pd.MultiIndex.from_tuples(
        [(2030, ts) for ts in hours],
        names=["period", "timestep"],
    )

    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.set_investment_periods(periods=[2030])

    for carrier in ["AC", "tes", "gen", "load"]:
        n.add("Carrier", carrier, co2_emissions=0.0)

    n.add("Bus", "z1", carrier="AC")
    n.add("Bus", "z1 tes_2030", carrier="tes")
    n.buses.loc[:, "country"] = "US"
    n.buses.loc[:, "interconnect"] = "west"
    n.buses.loc[:, "region"] = "west"
    n.buses.loc[:, "nerc_reg"] = "NERC1"
    n.buses.loc[:, "reeds_state"] = "CA"
    n.buses.loc[:, "reeds_zone"] = "CA_Z1"

    n.add("Load", "l", bus="z1", carrier="load", p_set=pd.Series(0.0, index=snapshots))
    n.add("Generator", "g", bus="z1", carrier="gen", p_nom=100, marginal_cost=1.0, p_max_pu=1.0)
    n.add(
        "Store",
        "z1 tes store_2030",
        bus="z1 tes_2030",
        carrier="tes",
        e_nom=200,
        e_initial=100,
        e_cyclic=False,
        e_cyclic_per_period=False,
        standing_loss=0.0,
        build_year=2030,
        lifetime=30,
    )
    n.add(
        "Link",
        "z1 tes discharge_2030",
        bus0="z1 tes_2030",
        bus1="z1",
        carrier="tes",
        p_nom=100,
        efficiency=0.524,
        build_year=2030,
        lifetime=30,
    )

    n.optimize.create_model(multi_investment_periods=True)
    add_ERM_constraints(n, snapshots, regional_erm_data={"all": 0.15})

    constraint = n.model.constraints["Link-p-store-upper_ERM"]
    reserve_var = n.model.variables["Link-p_ERM"].labels.sel(
        snapshot=snapshots[0],
        Link="z1 tes discharge_2030",
    ).item()
    store_var = n.model.variables["Store-e"].labels.sel(
        snapshot=snapshots[0],
        Store="z1 tes store_2030",
    ).item()

    vars_at_snapshot = constraint.vars.sel(snapshot=snapshots[0], Link="z1 tes discharge_2030").values.flat
    coeffs_at_snapshot = constraint.coeffs.sel(snapshot=snapshots[0], Link="z1 tes discharge_2030").values.flat
    coeff_by_var = dict(zip(vars_at_snapshot, coeffs_at_snapshot))

    # eh * p_ERM - Store-e <= 0, with an elapsed hour of 1
    assert coeff_by_var[reserve_var] == pytest.approx(1.0)
    assert coeff_by_var[store_var] == pytest.approx(-1.0)
    assert constraint.rhs.sel(snapshot=snapshots[0], Link="z1 tes discharge_2030").item() == 0.0

    # The discharger reaches the electricity bus through its efficiency, and the
    # charge side of a store link is not a boundary crossing: the base-state Link-p
    # of a link with one end on a tes bus never enters the requirement.
    row = _requirement_row(n, "NERC1", snapshots[0])
    assert row[reserve_var] == pytest.approx(0.524)
    base_link_var = _label(n, "Link-p", snapshot=snapshots[0], Link="z1 tes discharge_2030")
    assert base_link_var not in row, "A store link is not an electric boundary branch"


def test_multi_period_erm_optimization(multi_period_reserve_network):
    """Test that multi-period network with ERM solves and creates one constraint (not per period)."""
    n = multi_period_reserve_network.copy()

    erm_dict = {"all": 0.15}

    def extra_functionality(n, snapshots):
        add_ERM_constraints(n, snapshots, regional_erm_data=erm_dict)

    status, condition = n.optimize(
        solver_name="highs",
        multi_investment_periods=True,
        extra_functionality=extra_functionality,
    )
    assert status == "ok" and condition == "optimal", f"Optimization failed: {status}/{condition}"

    # One requirement per region spanning both periods, not one per region per period
    erm_constraints = sorted(c for c in n.model.constraints if c.startswith(f"{ERM_REQUIREMENT}_"))
    assert erm_constraints == sorted(erm_requirement_name(r) for r in ("NERC1", "NERC2")), (
        f"Should have exactly one ERM requirement per region, found: {erm_constraints}"
    )
    for region in ("NERC1", "NERC2"):
        requirement = n.model.constraints[erm_requirement_name(region)]
        assert set(requirement.coords["snapshot"].values) == set(n.snapshots), (
            "The requirement should span all investment periods"
        )


def test_multi_period_erm_activity_masking(multi_period_reserve_network):
    """Verify that retiring generators don't contribute to reserves in periods after retirement."""
    n = multi_period_reserve_network.copy()

    # Confirm gas_retiring is in the network and has the expected lifetime
    assert "gas_retiring" in n.generators.index
    assert n.generators.loc["gas_retiring", "build_year"] == 2025
    assert n.generators.loc["gas_retiring", "lifetime"] == 10
    assert not n.generators.loc["gas_retiring", "p_nom_extendable"]
    assert n.generators.loc["gas_retiring", "bus"] == "z1"

    # Check activity mask: gas_retiring should be active in 2030 but not in 2040
    n._multi_invest = True
    activity = get_activity_mask(n, "Generator", n.snapshots)
    period_2030_mask = n.snapshots.get_level_values(0) == 2030
    period_2040_mask = n.snapshots.get_level_values(0) == 2040

    assert activity.loc[period_2030_mask, "gas_retiring"].all(), "gas_retiring should be active in all 2030 snapshots"
    assert not activity.loc[period_2040_mask, "gas_retiring"].any(), (
        "gas_retiring should be inactive in all 2040 snapshots"
    )

    # Run optimization with ERM
    def extra_functionality(n, snapshots):
        add_ERM_constraints(n, snapshots, regional_erm_data={"all": 0.15})

    status, condition = n.optimize(
        solver_name="highs",
        multi_investment_periods=True,
        extra_functionality=extra_functionality,
    )
    assert status == "ok" and condition == "optimal", f"Optimization failed: {status}/{condition}"
    assert erm_requirement_name("NERC1") in n.model.constraints

    # gas_retiring is fixed capacity at z1, so it rides on the NERC1 right-hand side.
    # After retirement its 150 MW must drop out of that constant.
    firm = _firm_capacity(n, "z1", n.snapshots)
    unmasked = (
        get_as_dense(n, "Generator", "p_max_pu", n.snapshots)[
            n.generators.index[(n.generators.bus == "z1") & ~n.generators.p_nom_extendable]
        ]
        .mul(n.generators.p_nom)
        .sum(axis=1)
    )
    assert np.allclose(firm[period_2030_mask], unmasked[period_2030_mask]), (
        "Nothing at z1 is retired in 2030, so masking should change nothing"
    )
    assert np.allclose(unmasked[period_2040_mask] - firm[period_2040_mask], 150.0), (
        "The retired 150 MW should be masked out of the 2040 firm capacity"
    )

    buses = n.buses.index[n.buses.carrier == "AC"]
    demand = _bus_demand(n, buses)
    expected_rhs = demand[["z1"]].sum(axis=1) * 1.15 - firm
    actual_rhs = n.model.constraints[erm_requirement_name("NERC1")].rhs.to_pandas()
    assert np.allclose(actual_rhs, expected_rhs, atol=1e-6), (
        "The NERC1 requirement should be built on the masked firm capacity"
    )
