import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
    region_links = {
        "Texas": "b flexible electrolysis",
        "California": "b2 flexible electrolysis",
    }

    for period, hours in ((2030, hours_2030), (2040, hours_2040)):
        for region, link in region_links.items():
            rate_labels = (
                n.model.variables[f"FlexibleElectrolysis-h2_rate-{region}-{period}"]
                .labels.to_numpy()
                .reshape(-1)
            )
            expected_vars = link_p.sel(
                snapshot=[(period, ts) for ts in hours],
                Link=[link],
            ).values.reshape(-1)

            # The fleet is aggregated per snapshot, one row per snapshot, so the
            # annual row never carries a term per link and snapshot.
            definition = n.model.constraints[
                f"FlexibleElectrolysis-h2_rate-{region}-{period}-definition"
            ]
            assert set(np.asarray(definition.sign).ravel().tolist()) == {"="}
            assert np.asarray(definition.rhs).ravel().tolist() == pytest.approx([0.0] * 3)
            assert np.asarray(definition.coeffs).ravel().tolist() == pytest.approx(
                [1.0, -efficiency] * 3,
            )
            assert np.asarray(definition.vars).ravel().tolist() == [
                label for pair in zip(rate_labels, expected_vars) for label in pair
            ]

            constraint = n.model.constraints[f"FlexibleElectrolysis-annual_hydrogen-{region}-{period}"]
            assert constraint.rhs.item() == pytest.approx(expected_rhs[region] * 1e3)
            assert constraint.sign.item() == "="
            assert constraint.coeffs.to_numpy().tolist() == pytest.approx([2920.0 / 1e3] * 3)
            assert constraint.vars.to_numpy().tolist() == rate_labels.tolist()

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
    assert constraint.rhs.item() == pytest.approx(1512.0 * 1e3)
    assert constraint.sign.item() == "="

    efficiency = 1.0 / 1.351
    rate_labels = (
        n.model.variables["FlexibleElectrolysis-h2_rate-nation-2030"]
        .labels.to_numpy()
        .reshape(-1)
    )

    # The annual row sums the per-snapshot aggregate, one term per snapshot.
    assert constraint.coeffs.to_numpy().tolist() == pytest.approx([2920.0 / 1e3] * 3)
    assert constraint.vars.to_numpy().tolist() == rate_labels.tolist()

    # The whole fleet enters through the aggregation rows instead.
    definition = n.model.constraints["FlexibleElectrolysis-h2_rate-nation-2030-definition"]
    expected_vars = (
        n.model.variables["Link-p"]
        .labels.sel(
            snapshot=[(2030, ts) for ts in hours],
            Link=["b flexible electrolysis", "b2 flexible electrolysis"],
        )
        .values.reshape(-1)
    )
    assert np.asarray(definition.rhs).ravel().tolist() == pytest.approx([0.0] * 3)
    assert np.asarray(definition.coeffs).ravel().tolist() == pytest.approx(
        [1.0, -efficiency, -efficiency] * 3,
    )
    assert sorted(np.asarray(definition.vars).ravel().tolist()) == sorted(
        rate_labels.tolist() + expected_vars.tolist(),
    )

    # Capacity adequacy follows from Link-p <= p_nom, so no separate
    # capacity-energy constraint is built.
    assert not [name for name in n.model.constraints if "capacity_energy" in name]


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


def test_electrolysis_representative_periods_use_single_annual_equality():
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

    # Representative periods must not introduce any per-block auxiliary
    # variables or constraints; the target stays a single annual equality.
    assert not [name for name in n.model.variables if "hydrogen_budget" in name]
    assert not [name for name in n.model.constraints if "-block_" in name]

    efficiency = 1.0 / 1.351
    annual = n.model.constraints[
        "FlexibleElectrolysis-annual_hydrogen-Texas-2030"
    ]
    rate_labels = (
        n.model.variables["FlexibleElectrolysis-h2_rate-Texas-2030"]
        .labels.to_numpy()
        .reshape(-1)
    )
    assert annual.vars.to_numpy().reshape(-1).tolist() == rate_labels.tolist()
    assert annual.coeffs.to_numpy().reshape(-1).tolist() == pytest.approx([2190.0 / 1e3] * 4)
    assert annual.rhs.item() == pytest.approx(target_twh * 1e3)

    # The links reach the target only through the per-snapshot aggregation rows.
    definition = n.model.constraints["FlexibleElectrolysis-h2_rate-Texas-2030-definition"]
    link_p_labels = n.model["Link-p"].labels.to_numpy().reshape(-1)
    assert np.asarray(definition.coeffs).ravel().tolist() == pytest.approx(
        [1.0, -efficiency] * 4,
    )
    assert np.asarray(definition.vars).ravel().tolist() == [
        label for pair in zip(rate_labels, link_p_labels) for label in pair
    ]


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


def _sssc_upper(n):
    variable = n.model.variables["LineX-sssc_nom"]
    return variable.upper.to_series()


def test_line_x_sssc_bound_follows_the_implied_per_branch_cap():
    n = _line_x_network()
    n.optimize.create_model()
    assert _sssc_upper(n).tolist() == [1e6, 1e6]

    solve_network_module.tighten_line_x_sssc_bound(n, n.snapshots, _line_x_config(ratio=0.5))

    # the extendable branch is bounded by its own s_nom_max, the fixed one by its
    # rating, both times through the configured share
    assert _sssc_upper(n)["ab"] == pytest.approx(200.0)
    assert _sssc_upper(n)["bc"] == pytest.approx(40.0)


def test_line_x_sssc_bound_also_respects_the_system_total():
    n = _line_x_network()
    n.optimize.create_model()

    config = _line_x_config(ratio=0.5)
    config["lines"]["convert_lines_to_line_x"]["sssc_tot_max"] = 30.0
    solve_network_module.tighten_line_x_sssc_bound(n, n.snapshots, config)

    # no single branch can exceed the budget shared by all of them
    assert _sssc_upper(n).tolist() == pytest.approx([30.0, 30.0])


def test_line_x_sssc_bound_survives_a_disabled_per_branch_share():
    """The two caps are independent: dropping the share keeps the system total."""
    n = _line_x_network()
    n.optimize.create_model()

    config = _line_x_config(ratio=None)
    config["lines"]["convert_lines_to_line_x"]["sssc_tot_max"] = 30.0
    solve_network_module.add_line_x_sssc_line_capacity_constraint(n, n.snapshots, config)
    # the per-branch rows are gone with the share
    assert not [name for name in n.model.constraints if "line_capacity" in name]

    solve_network_module.tighten_line_x_sssc_bound(n, n.snapshots, config)
    assert _sssc_upper(n).tolist() == pytest.approx([30.0, 30.0])


def test_line_x_sssc_bound_is_a_no_op_without_either_cap():
    n = _line_x_network()
    n.optimize.create_model()

    solve_network_module.tighten_line_x_sssc_bound(n, n.snapshots, _line_x_config(ratio=None))

    assert _sssc_upper(n).tolist() == [1e6, 1e6]


def test_line_x_sssc_bound_never_loosens_an_existing_one():
    n = _line_x_network()
    n.line_xs.loc["ab", "sssc_nom_max"] = 5.0
    n.optimize.create_model()

    solve_network_module.tighten_line_x_sssc_bound(n, n.snapshots, _line_x_config(ratio=1.0))

    assert _sssc_upper(n)["ab"] == pytest.approx(5.0)
    assert _sssc_upper(n)["bc"] == pytest.approx(80.0)


def test_line_x_sssc_bound_leaves_the_optimum_untouched(monkeypatch):
    """The tightened bound is implied by the rows, so it may not move the solution."""

    def solve(tighten):
        n = _line_x_network()
        if not tighten:
            monkeypatch.setattr(
                solve_network_module, "tighten_line_x_sssc_bound", lambda *a, **k: None
            )

        def extra(network, sns):
            solve_network_module.add_line_x_sssc_line_capacity_constraint(
                network, sns, _line_x_config(ratio=0.5)
            )
            solve_network_module.tighten_line_x_sssc_bound(
                network, sns, _line_x_config(ratio=0.5)
            )

        n.optimize(solver_name="highs", extra_functionality=extra)
        return n.line_xs.sssc_nom_opt.tolist(), float(n.objective)

    tightened = solve(True)
    # the patch is undone by the fixture at the end of the test
    loose = solve(False)

    assert tightened[0] == pytest.approx(loose[0])
    assert tightened[1] == pytest.approx(loose[1])


def test_iterative_optimize_kwargs_forwards_only_what_is_configured():
    """An unset key has to keep PyPSA's own default rather than become None."""
    assert solve_network_module._iterative_optimize_kwargs({}) == {}
    assert solve_network_module._iterative_optimize_kwargs({"proximal": None}) == {}

    # the remaining step control stays at PyPSA's defaults and is not forwarded
    forwarded = solve_network_module._iterative_optimize_kwargs(
        {"proximal": "off", "cost_threshold": 1.0e-4, "max_iterations": 20},
    )
    assert forwarded == {"proximal": "off"}


def test_run_standard_optimize_passes_the_proximal_norm_through(monkeypatch):
    captured = {}

    class FakeOptimize:
        def optimize_transmission_expansion_iteratively(self, **kwargs):
            captured.update(kwargs)
            return "ok", "optimal"

    network = SimpleNamespace(optimize=FakeOptimize())
    cf_solving = {"scheme": "slp", "trust_region": True, "proximal": "l2"}

    status, condition = solve_network_module._run_standard_optimize(
        network, rolling_horizon=False, skip_iterations=False, cf_solving=cf_solving
    )

    assert (status, condition) == ("ok", "optimal")
    assert captured["trust_region"] is True
    assert captured["proximal"] == "l2"
    # the convergence criterion is PyPSA's, not something this repo sets
    assert "cost_threshold" not in captured
    assert "cost_window" not in captured
