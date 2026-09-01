"""
Test the content.py policy constraints functionality.

This module contains tests for the policy constraints in PyPSA-USA,
including Technology Capacity Targets (TCT), Renewable Portfolio Standards (RPS),
and Regional CO2 Limits.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from opts._helpers import get_region_buses
from prepare_network import average_every_nhours
from summary import get_node_emissions_timeseries

logger = logging.getLogger(__name__)

# Fixtures


@pytest.fixture
def policy_network(base_network):
    """
    Adapt base network for policy constraint testing (RPS, TCT, CO2 limits).

    Extends the base network with parameters needed for policy constraint testing.
    """
    n = base_network.copy()

    # For policy constraints, we want clearer isolation of regions
    # Add a nuclear generator (non-renewable but clean)
    n.add(
        "Generator",
        "nuclear1",
        bus="z3",
        p_nom=300,
        p_nom_extendable=True,
        carrier="nuclear",
        capital_cost=2500,
        marginal_cost=5,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
        p_nom_max=1500,
    )

    # Add a coal generator (high CO2 emissions)
    n.add(
        "Generator",
        "coal1",
        bus="z1",
        p_nom=500,
        p_nom_extendable=True,
        carrier="coal",
        capital_cost=2000,
        marginal_cost=30,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
        p_nom_max=1000,
    )

    # Add CO2 emissions to carriers
    n.carriers.loc["coal", "co2_emissions"] = 0.8  # tonnes CO2 per MWh
    n.carriers.loc["nuclear", "co2_emissions"] = 0.0
    n.carriers.loc["gas", "co2_emissions"] = 0.4  # tonnes CO2 per MWh

    # Add nice_name to carriers for emissions calculation
    n.carriers["nice_name"] = n.carriers.index

    return n


@pytest.fixture
def clustered_policy_network(policy_network):
    """Create a time-clustered version of the policy network."""
    return average_every_nhours(policy_network, "3h")


@pytest.fixture
def co2_config():
    """Create a config dictionary for regional CO2 limit constraints."""
    return {
        "electricity": {
            "regional_Co2_limits": os.path.join(os.path.dirname(__file__), "fixtures/regional_co2_limits.csv"),
        },
        "scenario": {
            "planning_horizons": ["2030"],
        },
    }


@pytest.fixture
def rps_config():
    """Create a config dictionary for RPS constraints."""
    # Create config dictionary
    config = {
        "electricity": {
            "portfolio_standards": os.path.join(os.path.dirname(__file__), "fixtures/portfolio_standards.csv"),
        },
    }

    # Create a mock snakemake object
    class MockSnakemake:
        def __init__(self):
            self.input = type(
                "obj",
                (object,),
                {
                    "rps_reeds": os.path.join(os.path.dirname(__file__), "fixtures/rps_reeds.csv"),
                    "ces_reeds": os.path.join(os.path.dirname(__file__), "fixtures/ces_reeds.csv"),
                },
            )
            self.params = type(
                "obj",
                (object,),
                {
                    "planning_horizons": [2030],
                },
            )

    snakemake = MockSnakemake()

    return config, snakemake


@pytest.fixture
def tct_config():
    """Create a config dictionary for TCT constraints."""
    return {
        "electricity": {
            "technology_capacity_targets": os.path.join(
                os.path.dirname(__file__),
                "fixtures/technology_capacity_targets.csv",
            ),
        },
    }


def test_add_regional_co2limit(policy_network, co2_config):
    """Test that regional CO2 limits are correctly added to the network."""
    from opts.policy import add_regional_co2limit

    n = policy_network
    config = co2_config

    # Add regional CO2 limits
    def extra_functionality(n, _):
        add_regional_co2limit(n, config)

    n.optimize(solver_name="highs", multi_investment_periods=True, extra_functionality=extra_functionality)

    # Check that constraints were added
    assert any("co2_limit" in c for c in n.model.constraints), "No CO2 limit constraints were added"

    # Get emissions data
    emissions = get_node_emissions_timeseries(n)
    # Check that emissions are within limits for each region
    # Get the regional CO2 limits from config file
    co2_limits = pd.read_csv(config["electricity"]["regional_Co2_limits"])
    epsilon = 1e-2  # Small numerical tolerance
    for _, row in co2_limits.iterrows():
        limit = row["limit"]
        region_list = [region.strip() for region in row.regions.split(",")]
        region_buses = get_region_buses(n, region_list)
        constraint_emissions = emissions.loc[:, region_buses.index].sum().sum()
        assert constraint_emissions <= limit + epsilon, f"Emissions in region {row.name} exceed limit of {limit}"


def test_add_regional_co2limit_clustered(clustered_policy_network, co2_config):
    """Test that regional CO2 limits are correctly added to a time-clustered network."""
    from opts.policy import add_regional_co2limit

    n = clustered_policy_network
    config = co2_config

    # Add regional CO2 limits
    def extra_functionality(n, _):
        add_regional_co2limit(n, config)

    n.optimize(solver_name="highs", multi_investment_periods=True, extra_functionality=extra_functionality)

    # Check that constraints were added
    assert any("co2_limit" in c for c in n.model.constraints), "No CO2 limit constraints were added"

    # Get emissions data
    emissions = get_node_emissions_timeseries(n)

    # Get the regional CO2 limits from config file
    co2_limits = pd.read_csv(config["electricity"]["regional_Co2_limits"])
    epsilon = 1e-2  # Small numerical tolerance
    for _, row in co2_limits.iterrows():
        limit = row["limit"]
        region_list = [region.strip() for region in row.regions.split(",")]
        region_buses = get_region_buses(n, region_list)
        constraint_emissions = emissions.loc[:, region_buses.index].sum().sum()
        assert constraint_emissions <= limit + epsilon, f"Emissions in region {row.name} exceed limit of {limit}"


def test_add_rps_constraints(policy_network, rps_config):
    """Test that RPS constraints are correctly added to the network."""
    from opts.policy import add_RPS_constraints

    n = policy_network
    config, snakemake = rps_config
    if "load" not in n.carriers.index:
        n.add("Carrier", "load")
    n.add(
        "Generator",
        "z1 load",
        bus="z1",
        carrier="load",
        p_nom=1e4,
        marginal_cost=1e5,
    )

    # Add RPS constraints
    def extra_functionality(n, _):
        add_RPS_constraints(n, config, snakemake=snakemake)

    n.optimize(solver_name="highs", multi_investment_periods=True, extra_functionality=extra_functionality)

    # Check that constraints were added
    assert any("rps_limit" in c for c in n.model.constraints), "No RPS limit constraints were added"
    assert any("ces_limit" in c for c in n.model.constraints), "No CES limit constraints were added"

    shedding_labels = set(
        n.model["Generator-p"]
        .labels.sel(Generator="z1 load")
        .to_numpy()
        .reshape(-1),
    )
    for name, constraint in n.model.constraints.items():
        if name.startswith("PortfolioStandard-"):
            constraint_labels = set(constraint.vars.to_numpy().reshape(-1))
            assert shedding_labels.isdisjoint(constraint_labels)


def test_add_rps_constraints_clustered(clustered_policy_network, rps_config):
    """Test that RPS constraints are correctly added to a time-clustered network."""
    from opts.policy import add_RPS_constraints

    n = clustered_policy_network
    config, snakemake = rps_config

    # Add RPS constraints
    def extra_functionality(n, _):
        add_RPS_constraints(n, config, snakemake=snakemake)

    n.optimize(solver_name="highs", multi_investment_periods=True, extra_functionality=extra_functionality)

    # Check that constraints were added
    assert any("rps_limit" in c for c in n.model.constraints), "No RPS limit constraints were added"
    assert any("ces_limit" in c for c in n.model.constraints), "No CES limit constraints were added"


def _portfolio_snakemake(tmp_path, rps_rows, ces_rows, planning_horizons):
    """Mock snakemake serving RPS/CES fractions written to ``tmp_path``."""
    paths = {}
    for name, rows in (("rps", rps_rows), ("ces", ces_rows)):
        path = tmp_path / f"{name}_fraction.csv"
        pd.DataFrame(rows, columns=["st", "t", "rps_all"]).to_csv(path, index=False)
        paths[name] = str(path)

    return type(
        "MockSnakemake",
        (object,),
        {
            "input": type("obj", (object,), {"rps_reeds": paths["rps"], "ces_reeds": paths["ces"]}),
            "params": type("obj", (object,), {"planning_horizons": planning_horizons}),
        },
    )()


def _full_standard_model(policy_network, tmp_path, ces_pct, rps_pct=0.3, horizons=(2030,)):
    """Build the model and apply RPS/CES with a CA CES at ``ces_pct``."""
    from opts.policy import add_RPS_constraints

    n = policy_network
    snakemake = _portfolio_snakemake(
        tmp_path,
        rps_rows=[["CA", h, rps_pct] for h in horizons] + [["TX", h, 0.4] for h in horizons],
        ces_rows=[["CA", h, ces_pct] for h in horizons] + [["TX", h, 0.5] for h in horizons],
        planning_horizons=list(horizons),
    )
    n.optimize.create_model(multi_investment_periods=True)
    add_RPS_constraints(n, {}, snakemake=snakemake)
    return n


def test_full_portfolio_standard_fixes_non_eligible_dispatch(policy_network, tmp_path):
    """A 100% CES fixes the state's non-eligible dispatch at zero instead of a share row."""
    n = _full_standard_model(policy_network, tmp_path, ces_pct=1.0)

    names = set(n.model.constraints)
    assert "PortfolioStandard-CA_2030_ces_limit" not in names
    constraint = n.model.constraints["PortfolioStandard-CA_2030_ces_zero_dispatch"]

    # z1 is the only CA bus; its gas and coal units are not CES eligible.
    blocked = ["coal1", "gas1"]
    expected = (
        n.model["Generator-p"].labels.sel(period=2030, Generator=blocked).values.reshape(-1)
    )
    assert sorted(constraint.vars.to_numpy().reshape(-1).tolist()) == sorted(expected.tolist())
    assert set(constraint.coeffs.to_numpy().reshape(-1).tolist()) == {1.0}
    assert set(np.asarray(constraint.rhs).ravel().tolist()) == {0.0}
    assert set(np.asarray(constraint.sign).ravel().tolist()) == {"="}

    # Partial standards elsewhere are untouched.
    assert "PortfolioStandard-CA_2030_rps_limit" in names
    assert "PortfolioStandard-TX_2030_ces_limit" in names


def test_full_portfolio_standard_keeps_capacity_for_adequacy(policy_network, tmp_path):
    """Only dispatch is fixed: the units and their p_nom stay, so ERM still sees them."""
    n = _full_standard_model(policy_network, tmp_path, ces_pct=1.0)

    # Both units survive with their rating intact, which is what the ERM requirement
    # reads (it is written on p_max_pu * p_nom, not on p).
    for name in ("gas1", "coal1"):
        assert name in n.generators.index
        assert n.generators.at[name, "p_nom"] > 0
        assert n.generators_t.p_max_pu.get(name, pd.Series([1.0])).max() > 0
    # coal1 is extendable, so it also keeps its capacity variable.
    assert "coal1" in n.model["Generator-p_nom"].indexes["Generator-ext"]


def test_full_portfolio_standard_spares_load_shedding(policy_network, tmp_path):
    """Load shedding is unserved energy, not generation, so it is not fixed to zero."""
    n = policy_network
    n.add("Generator", "z1 load", bus="z1", carrier="load", p_nom=1e4, marginal_cost=1e5)
    n = _full_standard_model(n, tmp_path, ces_pct=1.0)

    shedding = n.model["Generator-p"].labels.sel(period=2030, Generator=["z1 load"]).values.reshape(-1)
    constraint = n.model.constraints["PortfolioStandard-CA_2030_ces_zero_dispatch"]
    assert set(constraint.vars.to_numpy().reshape(-1)).isdisjoint(shedding.tolist())


def test_near_full_portfolio_standard_is_rounded_up(policy_network, tmp_path, caplog):
    """ReEDS' 0.999144 crosses the threshold, is enforced as full, and says so."""
    with caplog.at_level(logging.WARNING):
        n = _full_standard_model(policy_network, tmp_path, ces_pct=0.999144)

    assert "PortfolioStandard-CA_2030_ces_zero_dispatch" in set(n.model.constraints)
    assert any("rounded up to a full standard" in record.getMessage() for record in caplog.records)


def test_portfolio_standard_below_threshold_stays_a_share_row(policy_network, tmp_path):
    """Just below the threshold the standard remains an ordinary share constraint."""
    n = _full_standard_model(policy_network, tmp_path, ces_pct=0.9989)

    names = set(n.model.constraints)
    assert "PortfolioStandard-CA_2030_ces_limit" in names
    assert "PortfolioStandard-CA_2030_ces_zero_dispatch" not in names


def test_full_portfolio_standard_is_scoped_to_its_own_period(policy_network, tmp_path):
    """A standard that only reaches 100% later must not zero dispatch in earlier periods."""
    from opts.policy import add_RPS_constraints

    n = policy_network
    snakemake = _portfolio_snakemake(
        tmp_path,
        rps_rows=[["CA", 2030, 0.3]],
        ces_rows=[["CA", 2030, 0.5]],
        planning_horizons=[2030],
    )
    n.optimize.create_model(multi_investment_periods=True)
    add_RPS_constraints(n, {}, snakemake=snakemake)

    names = set(n.model.constraints)
    assert "PortfolioStandard-CA_2030_ces_zero_dispatch" not in names
    assert "PortfolioStandard-CA_2030_ces_limit" in names


def test_add_technology_capacity_target_constraints(policy_network, tct_config):
    """TCT creates minimum and maximum constraints without requiring a solver."""
    from opts.policy import add_technology_capacity_target_constraints

    n = policy_network
    n.optimize.create_model(multi_investment_periods=True)
    add_technology_capacity_target_constraints(n, tct_config)

    names = list(n.model.constraints)
    assert any(name.startswith("TCT-") and name.endswith("_min") for name in names)
    assert any(name.startswith("TCT-") and name.endswith("_max") for name in names)


def test_remove_tct_blocked_components_max_zero_forces_retirement(policy_network):
    """A ``max=0`` TCT row (e.g. reeds_GT_forced_retirement) removes both existing and candidate generators of that carrier/region."""
    from opts.policy import remove_tct_blocked_components

    n = policy_network

    # gas1 (bus z1 / reeds_state CA) is an existing, non-extendable generator.
    # Add a not-yet-built candidate in the same region/carrier to confirm both are purged.
    n.add(
        "Generator",
        "gas_candidate_ca",
        bus="z1",
        p_nom=0,
        p_nom_extendable=True,
        carrier="gas",
        capital_cost=450,
        marginal_cost=18,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
        p_nom_max=1000,
        build_year=2035,
        lifetime=20,
    )

    config = {
        "electricity": {
            "technology_capacity_targets": os.path.join(
                os.path.dirname(__file__),
                "fixtures/technology_capacity_targets_forced_retirement.csv",
            ),
        },
    }

    removed = remove_tct_blocked_components(n, config, components=("Generator", "StorageUnit", "Link"))

    assert set(removed["Generator"]) == {"gas1", "gas_candidate_ca"}
    assert "gas1" not in n.generators.index
    assert "gas_candidate_ca" not in n.generators.index
    # gas2 sits on bus z3 (reeds_state TX), outside the target's region, so it survives.
    assert "gas2" in n.generators.index
