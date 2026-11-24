"""
Test the reserves constraints functionality.

This module contains tests for the reserve margin constraints in PyPSA-USA.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from opts.reserves import add_ERM_constraints, store_ERM_duals


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


def _regional_reserve_slack(n, buses, erm_value):
    """Regional reserve capacity (generators, shadow storage, boundary flows) minus requirement."""
    buses = pd.Index(buses)
    snapshots = n.snapshots

    demand = n.loads_t.p.loc[:, n.loads.bus.isin(buses)].sum(axis=1)
    requirement = demand * (1.0 + erm_value)

    gens = n.generators.index[n.generators.bus.isin(buses)]
    p_max_pu = n.get_switchable_as_dense("Generator", "p_max_pu", snapshots)[gens]
    capacity = (p_max_pu * n.generators.p_nom_opt[gens]).sum(axis=1)

    # storage enters with its shadow (RESERVES) dispatch
    su = n.storage_units.index[n.storage_units.bus.isin(buses)]
    if not su.empty:
        shadow = (
            n.model.solution["StorageUnit-p_dispatch_RESERVES"].to_pandas()
            - n.model.solution["StorageUnit-p_store_RESERVES"].to_pandas()
        )
        capacity = capacity + shadow[su].sum(axis=1).values

    # lines and links enter with their actual flows across the region boundary
    def net_import(c, flow, efficiency=None):
        df = n.static(c)
        inflow = df.index[~df.bus0.isin(buses) & df.bus1.isin(buses)]
        outflow = df.index[df.bus0.isin(buses) & ~df.bus1.isin(buses)]
        into = flow[inflow] * (efficiency[inflow] if efficiency is not None else 1)
        return into.sum(axis=1) - flow[outflow].sum(axis=1)

    link_eff = n.get_switchable_as_dense("Link", "efficiency", snapshots)
    imports = net_import("Line", n.lines_t.p0) + net_import("Link", n.links_t.p0, link_eff)

    return capacity + imports - requirement, imports


def test_erm_constraint_binding(reserve_margin_network):
    """Test that the regional ERM constraint holds in every snapshot."""
    n = reserve_margin_network.copy()

    erm_value = 0.90

    def extra_functionality(n, sns):
        add_ERM_constraints(n, sns, regional_erm_data={"all": erm_value})

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)
    store_ERM_duals(n)

    # one row per snapshot, not per bus
    assert n.model.constraints["GlobalConstraint-all_ERM"].shape == (len(n.snapshots),)

    slack, imports = _regional_reserve_slack(n, n.buses.index, erm_value)
    assert (imports.abs() < 1e-6).all(), "No line or link crosses the boundary of region 'all'"
    assert (slack >= -0.1).all(), "Regional reserve capacity should cover the regional reserve requirement"
    assert (slack.abs() < 0.1).any(), "ERM of 0.90 should bind in at least one snapshot"


def test_erm_counts_boundary_flows(reserve_margin_network):
    """Only actual flows across a region's boundary enter its ERM constraint."""
    n = reserve_margin_network.copy()

    erm_dict = {"NERC1": 0.15, "NERC2": 0.15}

    def extra_functionality(n, sns):
        add_ERM_constraints(n, sns, regional_erm_data=erm_dict)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)
    store_ERM_duals(n)

    m = n.model

    def flow_labels(var, name):
        return set(m.variables[var].sel(name=name).labels.values.ravel())

    for region, buses in {"NERC1": ["z1"], "NERC2": ["z2", "z3"]}.items():
        con = m.constraints[f"GlobalConstraint-{region}_ERM"]
        assert con.shape == (len(n.snapshots),)

        # line1 (z1-z2) and link1 (z1-z3) cross the boundary; line2 (z2-z3) is inside NERC2
        labels = set(con.vars.values.ravel()) - {-1}
        assert labels & flow_labels("Line-s", "line1")
        assert labels & flow_labels("Link-p", "link1")
        assert not labels & flow_labels("Line-s", "line2")

        slack, _ = _regional_reserve_slack(n, buses, erm_dict[region])
        assert (slack >= -0.1).all(), f"ERM of {region} should hold in every snapshot"

    # each bus carries the dual of its own region
    prices = n.buses_t.erm_price
    assert (prices["z2"] == prices["z3"]).all()
    dual_nerc1 = m.dual["GlobalConstraint-NERC1_ERM"].to_pandas().reindex(n.snapshots)
    assert np.allclose(prices["z1"], dual_nerc1)


def test_multiple_non_overlapping_erms(reserve_margin_network):
    """Test that multiple ERM constraints work correctly for non-overlapping regions."""
    n = reserve_margin_network.copy()

    # Add large firm generators in each region so standalone regional ERM is feasible.
    # NERC1 (z1) and NERC2 (z3) have insufficient capacity without cross-region transmission.
    n.add(
        "Generator",
        "firm_nerc1",
        bus="z1",
        p_nom=2000,
        p_nom_extendable=False,
        carrier="gas",
        marginal_cost=50,
        p_max_pu=1.0,
        build_year=2030,
        lifetime=20,
    )
    n.add(
        "Generator",
        "firm_nerc2",
        bus="z3",
        p_nom=2000,
        p_nom_extendable=False,
        carrier="gas",
        marginal_cost=50,
        p_max_pu=1.0,
        build_year=2030,
        lifetime=20,
    )

    erm_dict = {"NERC1": 0.15, "NERC2": 0.30}

    def extra_functionality(n, sns):
        add_ERM_constraints(n, sns, regional_erm_data=erm_dict)

    try:
        n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)
    except Exception:
        assert False, "Optimization failed"

    store_ERM_duals(n)

    # Verify that ERM constraints were actually added
    erm_constraints = [c for c in n.model.constraints if "ERM" in c]
    assert len(erm_constraints) >= 2, f"Should have at least 2 ERM constraints, found {len(erm_constraints)}"

    # Verify each region has its own ERM constraint
    for region_name in erm_dict:
        constraint_name = f"GlobalConstraint-{region_name}_ERM"
        assert constraint_name in n.model.constraints, f"ERM constraint {constraint_name} should exist"

    # Verify that ERM duals are stored correctly
    assert hasattr(n.buses_t, "erm_price"), "ERM dual prices should be stored in n.buses_t.erm_price"

    # Check that we have ERM prices for both regions
    erm_price_data = n.buses_t.erm_price
    assert not erm_price_data.isnull().values.any(), "ERM price data should not contain any NaN values"


def test_erm_increases_capacity(reserve_margin_network):
    """Test that ERM constraint of 0.14 results in more capacity built than no ERM."""
    # First run without ERM constraints
    n_no_erm = reserve_margin_network.copy()
    n_no_erm.optimize(solver_name="glpk", multi_investment_periods=True)

    total_gen_capacity_no_erm = n_no_erm.generators.p_nom_opt.sum()
    total_storage_capacity_no_erm = n_no_erm.storage_units.p_nom_opt.sum()
    total_capacity_no_erm = total_gen_capacity_no_erm + total_storage_capacity_no_erm

    # Now run with ERM constraint of 0.14
    n_with_erm = reserve_margin_network.copy()

    def extra_functionality(n, sns):
        add_ERM_constraints(n, sns, regional_erm_data={"all": 0.14})

    n_with_erm.optimize(
        solver_name="glpk",
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
    n_no_erm.optimize(solver_name="glpk", multi_investment_periods=True)

    total_gen_capacity_no_erm = n_no_erm.generators.p_nom_opt.sum()
    total_storage_capacity_no_erm = n_no_erm.storage_units.p_nom_opt.sum()
    total_capacity_no_erm = total_gen_capacity_no_erm + total_storage_capacity_no_erm

    # Now run with ERM constraint of 0.14
    n_with_erm = reserve_margin_network.copy()
    n_with_erm = disable_transmission_expansion(n_with_erm)

    def extra_functionality(n, sns):
        add_ERM_constraints(n, sns, regional_erm_data={"all": 0.14})

    n_with_erm.optimize(
        solver_name="glpk",
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


def test_multi_period_erm_optimization(multi_period_reserve_network):
    """Test that multi-period network with ERM solves and creates one constraint (not per period)."""
    n = multi_period_reserve_network.copy()

    erm_dict = {"all": 0.15}

    def extra_functionality(n, sns):
        add_ERM_constraints(n, sns, regional_erm_data=erm_dict)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    # Verify one ERM constraint is created (not two per period)
    erm_constraints = [c for c in n.model.constraints if "ERM" in c]
    assert len(erm_constraints) == 1, (
        f"Should have exactly 1 ERM constraint, found {len(erm_constraints)}: {erm_constraints}"
    )
    assert "GlobalConstraint-all_ERM" in n.model.constraints


def test_multi_period_erm_increases_capacity(multi_period_reserve_network):
    """Test that ERM increases capacity across multiple investment periods."""
    # Without ERM
    n_no_erm = multi_period_reserve_network.copy()
    n_no_erm.optimize(solver_name="glpk", multi_investment_periods=True)

    total_capacity_no_erm = n_no_erm.generators.p_nom_opt.sum() + n_no_erm.storage_units.p_nom_opt.sum()

    # With ERM
    n_with_erm = multi_period_reserve_network.copy()

    def extra_functionality(n, sns):
        add_ERM_constraints(n, sns, regional_erm_data={"all": 0.15})

    n_with_erm.optimize(
        solver_name="glpk",
        multi_investment_periods=True,
        extra_functionality=extra_functionality,
    )

    total_capacity_with_erm = n_with_erm.generators.p_nom_opt.sum() + n_with_erm.storage_units.p_nom_opt.sum()

    assert total_capacity_with_erm > total_capacity_no_erm, (
        f"ERM should increase capacity in multi-period. "
        f"Without: {total_capacity_no_erm:.2f}, With: {total_capacity_with_erm:.2f}"
    )


def test_multi_period_erm_activity_masking(multi_period_reserve_network):
    """Verify that retiring generators don't contribute to reserves in periods after retirement."""
    n = multi_period_reserve_network.copy()

    # Confirm gas_retiring is in the network and has the expected lifetime
    assert "gas_retiring" in n.generators.index
    assert n.generators.loc["gas_retiring", "build_year"] == 2025
    assert n.generators.loc["gas_retiring", "lifetime"] == 10

    # Check activity mask: gas_retiring should be active in 2030 but not in 2040.
    # n._multi_invest must be True for get_activity_mask to use build_year/lifetime logic;
    # it is normally set by n.optimize(multi_investment_periods=True).
    n._multi_invest = True
    activity = n.components["Generator"].get_activity_mask(n.snapshots)
    period_2030_mask = n.snapshots.get_level_values(0) == 2030
    period_2040_mask = n.snapshots.get_level_values(0) == 2040

    assert activity.loc[period_2030_mask, "gas_retiring"].all(), "gas_retiring should be active in all 2030 snapshots"
    assert not activity.loc[period_2040_mask, "gas_retiring"].any(), (
        "gas_retiring should be inactive in all 2040 snapshots"
    )

    # Run optimization with ERM
    def extra_functionality(n, sns):
        add_ERM_constraints(n, sns, regional_erm_data={"all": 0.15})

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    # Verify the constraint was added successfully
    assert "GlobalConstraint-all_ERM" in n.model.constraints


def test_erm_zero_emission_period_excludes_fossil(multi_period_reserve_network, tmp_path):
    """CO2 limit=0 in 2040 forces more clean (nuclear) capacity when gas is excluded from ERM credit.

    When a planning horizon has a national CO2 limit of zero, emitting generators
    (gas, coal) receive no ERM capacity credit in that period. The optimizer must
    then satisfy the reserve margin requirement with clean firm capacity instead.
    """

    def _add_nuclear(n):
        """Add extendable nuclear at each bus so the zero-emission ERM is feasible."""
        for bus in n.buses.index:
            n.add(
                "Generator",
                f"nuclear_{bus}",
                bus=bus,
                carrier="nuclear",
                p_nom=0,
                p_nom_extendable=True,
                p_nom_max=5000,
                capital_cost=6000,
                marginal_cost=8,
                p_max_pu=0.92,
                build_year=2030,
                lifetime=60,
            )

    # Baseline: ERM with no CO2 restriction — gas can count toward ERM in all periods.
    # Nuclear (very expensive) should not be built when gas is available.
    n_base = multi_period_reserve_network.copy()
    _add_nuclear(n_base)

    def extra_base(n, sns):
        add_ERM_constraints(n, sns, regional_erm_data={"all": 0.15})

    n_base.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_base)
    nuclear_base = n_base.generators.filter(like="nuclear", axis=0).p_nom_opt.sum()

    # Zero-emission run: CO2 limit=0 in 2040 → gas excluded from ERM credit in 2040.
    # Nuclear must now be built to satisfy the 2040 reserve requirement.
    co2_csv = tmp_path / "co2.csv"
    pd.DataFrame(
        {
            "regions": ["all", "all"],
            "planning_horizon": [2030, 2040],
            "limit": [1e9, 0],
        },
    ).to_csv(co2_csv, index=False)

    n_zero = multi_period_reserve_network.copy()
    _add_nuclear(n_zero)
    cfg = {"electricity": {"regional_Co2_limits": str(co2_csv)}}

    def extra_zero(n, sns):
        add_ERM_constraints(n, sns, config=cfg, regional_erm_data={"all": 0.15})

    n_zero.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_zero)
    nuclear_zero = n_zero.generators.filter(like="nuclear", axis=0).p_nom_opt.sum()

    assert nuclear_zero >= nuclear_base, (
        f"Zero-emission ERM should force more nuclear capacity. "
        f"Baseline: {nuclear_base:.0f} MW, zero-emission: {nuclear_zero:.0f} MW"
    )
