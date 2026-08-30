import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import solve_network as solve_network_module
from solve_network import (
    add_electrolysis_hydrogen_target_constraint,
    h2ptcreg_hydrogen_shares,
)

SECTOR_COSTS = Path(__file__).parents[2] / "repo_data" / "costs" / "simple_sector_costs.csv"
HYDROGEN_DEMAND_SHARE = (
    Path(__file__).parents[2] / "repo_data" / "ReEDS_Constraints" / "hydrogen_demand_share.csv"
)


def test_solve_network_with_line_x_config(monkeypatch):
    captured = {}

    def fake_run_optimize(n, rolling_horizon, skip_iterations, cf_solving, **kwargs):
        captured["rolling_horizon"] = rolling_horizon
        captured["skip_iterations"] = skip_iterations
        captured["cf_solving"] = cf_solving
        captured["kwargs"] = kwargs

    monkeypatch.setattr(solve_network_module, "run_optimize", fake_run_optimize)
    monkeypatch.setattr(
        solve_network_module,
        "snakemake",
        SimpleNamespace(params=SimpleNamespace(foresight="perfect")),
        raising=False,
    )

    n = SimpleNamespace(lines=pd.DataFrame({"s_nom_extendable": [True]}))
    config = {
        "foresight": "perfect",
        "lines": {
            "convert_lines_to_line_x": {
                "enable": True,
            },
        },
    }
    solving = {
        "solver": {"options": "", "name": "gurobi"},
        "solver_options": {},
        "options": {},
    }

    solve_network_module.solve_network(n, config, solving)

    # Test passes if solve_network completes without error
    assert captured["kwargs"] is not None


def test_run_optimize_passes_extra_functionality_into_iterative_solver():
    captured = {}

    class FakeOptimizeAccessor:
        def optimize_transmission_expansion_iteratively(self, **kwargs):
            captured.update(kwargs)
            return "ok", "optimal"

    n = SimpleNamespace(optimize=FakeOptimizeAccessor())

    def fake_extra_functionality(network, snapshots):
        return None

    solve_network_module.run_optimize(
        n,
        rolling_horizon=False,
        skip_iterations=False,
        cf_solving={
            "track_iterations": True,
            "min_iterations": 2,
            "max_iterations": 3,
            "scheme": "slp",
            "trust_region": True,
        },
        extra_functionality=fake_extra_functionality,
    )

    assert captured["extra_functionality"] is fake_extra_functionality
    assert captured["track_iterations"] is True
    assert captured["min_iterations"] == 2
    assert captured["max_iterations"] == 3
    assert captured["scheme"] == "slp"
    assert captured["trust_region"] is True


def test_add_electrolysis_constraint_splits_hydrogen_target_across_h2ptcreg_regions():
    hours_2030 = pd.date_range("2030-01-01 00:00", "2030-01-01 02:00", freq="h")
    hours_2040 = pd.date_range("2040-01-01 00:00", "2040-01-01 02:00", freq="h")
    snapshots = pd.MultiIndex.from_tuples(
        [(2030, ts) for ts in hours_2030] + [(2040, ts) for ts in hours_2040],
        names=["period", "timestep"],
    )

    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.set_investment_periods(periods=[2030, 2040])

    n.add("Carrier", "AC")
    n.add("Bus", "b", carrier="AC")
    n.add("Bus", "b2", carrier="AC")
    n.add("Carrier", "gen")
    n.add("Carrier", "load")
    n.add("Carrier", "H2")
    n.add("Generator", "g", bus="b", carrier="gen", p_nom=1e6, marginal_cost=1.0, p_max_pu=1.0)
    n.add("Load", "l", bus="b", carrier="load", p_set=pd.Series(0.0, index=snapshots))
    # One accounting H2 bus per h2ptcreg region.
    n.add("Bus", "Texas flexible electrolysis H2", carrier="H2")
    n.add("Bus", "California flexible electrolysis H2", carrier="H2")
    n.add("Carrier", "electrolysis")
    efficiency = 1.0 / 1.351
    # Network links carry zero efficiency; the constraint must use the configured factor.
    n.add(
        "Link",
        "b flexible electrolysis",
        bus0="b",
        bus1="Texas flexible electrolysis H2",
        carrier="electrolysis",
        p_nom=0.0,
        p_nom_extendable=True,
        efficiency=0.0,
    )
    n.add(
        "Link",
        "b2 flexible electrolysis",
        bus0="b2",
        bus1="California flexible electrolysis H2",
        carrier="electrolysis",
        p_nom=0.0,
        p_nom_extendable=True,
        efficiency=0.0,
    )
    n.snapshot_weightings.loc[:, "generators"] = 2920.0

    n.optimize.create_model(multi_investment_periods=True)
    add_electrolysis_hydrogen_target_constraint(
        n,
        snapshots,
        {
            "flexible_electrolysis": {
                "enable": True,
                "annual_hydrogen_twh": 1512,
            },
        },
        str(SECTOR_COSTS),
        str(HYDROGEN_DEMAND_SHARE),
    )

    # Shares are renormalised over the regions present in the network, so the two
    # regional targets still add up to the configured national total.
    shares = h2ptcreg_hydrogen_shares(str(HYDROGEN_DEMAND_SHARE))
    modelled = shares[["California", "Texas"]]
    expected_rhs = (modelled / modelled.sum() * 1512.0).to_dict()

    link_p = n.model.variables["Link-p"].labels
    link_p_nom = n.model.variables["Link-p_nom"].labels
    expected_coeff = 2920.0 * efficiency / 1e6
    expected_capacity_coeff = 3 * expected_coeff
    region_links = {
        "Texas": "b flexible electrolysis",
        "California": "b2 flexible electrolysis",
    }

    for period, hours in ((2030, hours_2030), (2040, hours_2040)):
        for region, link in region_links.items():
            constraint = n.model.constraints[f"FlexibleElectrolysis-annual_hydrogen-{region}-{period}"]
            expected_vars = link_p.sel(
                snapshot=[(period, ts) for ts in hours],
                Link=[link],
            ).values.reshape(-1)

            assert constraint.rhs.item() == pytest.approx(expected_rhs[region])
            assert constraint.sign.item() == "="
            assert constraint.coeffs.to_numpy().tolist() == pytest.approx([expected_coeff] * 3)
            assert constraint.vars.to_numpy().tolist() == expected_vars.tolist()

            capacity_constraint = n.model.constraints[
                f"FlexibleElectrolysis-annual_capacity_energy-{region}-{period}"
            ]
            expected_capacity_var = link_p_nom.sel({"Link-ext": link}).item()
            assert capacity_constraint.rhs.item() == pytest.approx(expected_rhs[region])
            assert capacity_constraint.sign.item() == ">="
            assert capacity_constraint.coeffs.to_numpy().tolist() == pytest.approx(
                [expected_capacity_coeff],
            )
            assert capacity_constraint.vars.to_numpy().tolist() == [expected_capacity_var]

    assert sum(expected_rhs.values()) == pytest.approx(1512.0)


def _national_electrolysis_network(accounting_bus):
    """Single-period network whose electrolysis links share one accounting bus."""
    hours = pd.date_range("2030-01-01 00:00", "2030-01-01 02:00", freq="h")
    snapshots = pd.MultiIndex.from_tuples(
        [(2030, ts) for ts in hours],
        names=["period", "timestep"],
    )

    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.set_investment_periods(periods=[2030])

    n.add("Carrier", "AC")
    n.add("Carrier", "gen")
    n.add("Carrier", "load")
    n.add("Carrier", "H2")
    n.add("Carrier", "electrolysis")
    n.add("Bus", "b", carrier="AC")
    n.add("Bus", "b2", carrier="AC")
    n.add("Bus", accounting_bus, carrier="H2")
    n.add("Generator", "g", bus="b", carrier="gen", p_nom=1e6, marginal_cost=1.0)
    n.add("Load", "l", bus="b", carrier="load", p_set=pd.Series(0.0, index=snapshots))
    for ac_bus in ("b", "b2"):
        n.add(
            "Link",
            f"{ac_bus} flexible electrolysis",
            bus0=ac_bus,
            bus1=accounting_bus,
            carrier="electrolysis",
            p_nom=0.0,
            p_nom_extendable=True,
            efficiency=0.0,
        )
    n.snapshot_weightings.loc[:, "generators"] = 2920.0
    n.optimize.create_model(multi_investment_periods=True)
    return n, snapshots, hours


def test_add_electrolysis_constraint_pools_hydrogen_target_nationally():
    n, snapshots, hours = _national_electrolysis_network(
        "nation flexible electrolysis H2",
    )

    add_electrolysis_hydrogen_target_constraint(
        n,
        snapshots,
        {
            "flexible_electrolysis": {
                "enable": True,
                "annual_hydrogen_twh": 1512,
                "accounting_region": "nation",
            },
        },
        str(SECTOR_COSTS),
        str(HYDROGEN_DEMAND_SHARE),
    )

    # A single constraint over every electrolysis link, for the full national total.
    assert not any(
        name.startswith("FlexibleElectrolysis-annual_hydrogen-")
        and not name.startswith("FlexibleElectrolysis-annual_hydrogen-nation")
        for name in n.model.constraints
    )
    constraint = n.model.constraints["FlexibleElectrolysis-annual_hydrogen-nation-2030"]
    assert constraint.rhs.item() == pytest.approx(1512.0)
    assert constraint.sign.item() == "="

    efficiency = 1.0 / 1.351
    expected_coeff = 2920.0 * efficiency / 1e6
    assert constraint.coeffs.to_numpy().tolist() == pytest.approx([expected_coeff] * 6)
    expected_vars = (
        n.model.variables["Link-p"]
        .labels.sel(
            snapshot=[(2030, ts) for ts in hours],
            Link=["b flexible electrolysis", "b2 flexible electrolysis"],
        )
        .values.reshape(-1)
    )
    assert sorted(constraint.vars.to_numpy().tolist()) == sorted(expected_vars.tolist())

    # Both links contribute to the one capacity-adequacy constraint.
    capacity = n.model.constraints["FlexibleElectrolysis-annual_capacity_energy-nation-2030"]
    assert capacity.rhs.item() == pytest.approx(1512.0)
    assert len(capacity.vars.to_numpy().reshape(-1)) == 2


def test_add_electrolysis_constraint_rejects_accounting_region_network_mismatch():
    n, snapshots, _ = _national_electrolysis_network(
        "Texas flexible electrolysis H2",
    )

    with pytest.raises(ValueError, match="accounting_region"):
        add_electrolysis_hydrogen_target_constraint(
            n,
            snapshots,
            {
                "flexible_electrolysis": {
                    "enable": True,
                    "annual_hydrogen_twh": 1512,
                    "accounting_region": "nation",
                },
            },
            str(SECTOR_COSTS),
            str(HYDROGEN_DEMAND_SHARE),
        )


def test_electrolysis_representative_blocks_use_bounded_master_budgets():
    hours = pd.date_range("2030-01-01 00:00", periods=4, freq="h")
    snapshots = pd.MultiIndex.from_product(
        [[2030], hours],
        names=["period", "timestep"],
    )
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.set_investment_periods([2030])
    n.add("Carrier", "AC")
    n.add("Carrier", "H2")
    n.add("Carrier", "electrolysis")
    n.add("Bus", "b", carrier="AC")
    n.add("Bus", "Texas flexible electrolysis H2", carrier="H2")
    n.add("Generator", "g", bus="b", p_nom=100.0, marginal_cost=1.0)
    n.add("Load", "l", bus="b", p_set=0.0)
    n.add(
        "Link",
        "b flexible electrolysis",
        bus0="b",
        bus1="Texas flexible electrolysis H2",
        carrier="electrolysis",
        p_nom_extendable=True,
        efficiency=0.0,
    )
    n.snapshot_weightings.loc[:, "generators"] = 2190.0
    n.optimize.create_model(multi_investment_periods=True)

    target_twh = 0.01
    add_electrolysis_hydrogen_target_constraint(
        n,
        snapshots,
        {
            "flexible_electrolysis": {
                "enable": True,
                "annual_hydrogen_twh": target_twh,
            },
            "clustering": {
                "temporal": {
                    "representative_periods": {
                        "enable": True,
                        "period_length": 2.0 / 24.0,
                    },
                },
            },
        },
        str(SECTOR_COSTS),
        str(HYDROGEN_DEMAND_SHARE),
    )

    efficiency = 1.0 / 1.351
    p_nom_label = n.model["Link-p_nom"].labels.sel(
        {"Link-ext": "b flexible electrolysis"},
    ).item()
    budget_labels = []
    for block in range(2):
        budget_name = (
            f"FlexibleElectrolysis-hydrogen_budget-Texas-2030-{block}"
        )
        budget_label = n.model[budget_name].labels.item()
        budget_labels.append(budget_label)

        local = n.model.constraints[
            f"FlexibleElectrolysis-block_hydrogen-Texas-2030-{block}"
        ]
        local_terms = dict(
            zip(
                local.vars.to_numpy().reshape(-1),
                local.coeffs.to_numpy().reshape(-1),
            ),
        )
        assert local_terms[budget_label] == pytest.approx(-1.0)
        assert sum(value > 0.0 for value in local_terms.values()) == 2

        capacity = n.model.constraints[
            f"FlexibleElectrolysis-block_capacity-Texas-2030-{block}"
        ]
        capacity_terms = dict(
            zip(
                capacity.vars.to_numpy().reshape(-1),
                capacity.coeffs.to_numpy().reshape(-1),
            ),
        )
        assert capacity_terms[budget_label] == pytest.approx(1.0)
        assert capacity_terms[p_nom_label] == pytest.approx(
            -(2 * 2190.0 * efficiency / 1e6),
        )
        assert capacity.rhs.item() == pytest.approx(0.0)

    annual = n.model.constraints[
        "FlexibleElectrolysis-annual_hydrogen-Texas-2030"
    ]
    assert set(annual.vars.to_numpy().reshape(-1)) == set(budget_labels)
    assert annual.coeffs.to_numpy().reshape(-1).tolist() == pytest.approx(
        [1.0, 1.0],
    )
    assert annual.rhs.item() == pytest.approx(target_twh)


def _line_x_network():
    """Three-bus AC triangle whose LineX branches differ in what is extendable."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    for bus in ["a", "b", "c"]:
        n.add("Bus", bus, carrier="AC", v_nom=345)

    # extendable line and extendable SSSC
    n.add(
        "LineX", "ab", bus0="a", bus1="b", carrier="AC", x=0.1, r=0.01, s_nom=100,
        s_nom_extendable=True, s_nom_min=100, s_nom_max=400,
        sssc_nom_extendable=True, sssc_nom_max=1e6, capital_cost=1.0, capital_cost_sssc=-50.0,
    )
    # fixed line, extendable SSSC
    n.add(
        "LineX", "bc", bus0="b", bus1="c", carrier="AC", x=0.2, r=0.02, s_nom=80,
        s_nom_extendable=False,
        sssc_nom_extendable=True, sssc_nom_max=1e6, capital_cost_sssc=-50.0,
    )
    # nothing extendable, SSSC preset
    n.add(
        "LineX", "ac", bus0="a", bus1="c", carrier="AC", x=0.3, r=0.03, s_nom=60,
        s_nom_extendable=False, sssc_nom_extendable=False, sssc_nom=5.0,
    )
    n.add("Generator", "g", bus="a", carrier="AC", p_nom=500, marginal_cost=10.0)
    n.add("Load", "l", bus="c", carrier="AC", p_set=90.0)
    return n


def _line_x_config(ratio=1.0):
    return {"lines": {"convert_lines_to_line_x": {"sssc_nom_max_pu": ratio}}}


def test_line_x_sssc_line_capacity_constraint_rows():
    n = _line_x_network()
    n.optimize.create_model()

    solve_network_module.add_line_x_sssc_line_capacity_constraint(n, n.snapshots, _line_x_config())

    # the extendable line is capped against its own capacity variable
    ext = n.model.constraints["LineX-sssc_nom-line_capacity"]
    assert list(ext.indexes["LineX"]) == ["ab"]
    assert float(ext.rhs.item()) == pytest.approx(0.0)

    # the fixed line is capped against its constant rating
    fix = n.model.constraints["LineX-fix-sssc_nom-line_capacity"]
    assert list(fix.indexes["LineX"]) == ["bc"]
    assert float(fix.rhs.item()) == pytest.approx(80.0)


def test_line_x_sssc_line_capacity_constraint_binds_in_solution():
    n = _line_x_network()

    # capital_cost_sssc is negative in the fixture, so the optimum sits on the cap
    n.optimize(
        solver_name="highs",
        extra_functionality=lambda network, sns: solve_network_module.add_line_x_sssc_line_capacity_constraint(
            network, sns, _line_x_config()
        ),
    )

    assert n.line_xs.at["ab", "sssc_nom_opt"] == pytest.approx(n.line_xs.at["ab", "s_nom_opt"])
    assert n.line_xs.at["bc", "sssc_nom_opt"] == pytest.approx(n.line_xs.at["bc", "s_nom"])
    # a branch without an extendable SSSC keeps its preset rating
    assert n.line_xs.at["ac", "sssc_nom_opt"] == pytest.approx(5.0)


def test_line_x_sssc_line_capacity_constraint_respects_ratio_and_is_optional():
    n = _line_x_network()
    n.optimize.create_model()
    solve_network_module.add_line_x_sssc_line_capacity_constraint(n, n.snapshots, _line_x_config(ratio=0.5))
    assert float(n.model.constraints["LineX-fix-sssc_nom-line_capacity"].rhs.item()) == pytest.approx(40.0)

    disabled = _line_x_network()
    disabled.optimize.create_model()
    solve_network_module.add_line_x_sssc_line_capacity_constraint(disabled, disabled.snapshots, _line_x_config(ratio=None))
    assert not [name for name in disabled.model.constraints if "line_capacity" in name]
