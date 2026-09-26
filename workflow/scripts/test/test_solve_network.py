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
    add_electrolysis_electricity_target_constraint,
    h2ptcreg_hydrogen_shares,
    store_electrolysis_duals,
)
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
            "proximal": True,
        },
        extra_functionality=fake_extra_functionality,
    )

    assert captured["extra_functionality"] is fake_extra_functionality
    assert captured["track_iterations"] is True
    assert captured["min_iterations"] == 2
    assert captured["max_iterations"] == 3
    assert captured["scheme"] == "slp"
    assert captured["proximal"] is True


def test_add_electrolysis_constraint_splits_electricity_target_across_h2ptcreg_regions():
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
    add_electrolysis_electricity_target_constraint(
        n,
        snapshots,
        {
            "flexible_electrolysis": {
                "enable": True,
                "annual_electricity_twh": 1512,
            },
        },
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

    # No auxiliary rate variable: the target is a single row per region and period.
    assert not [name for name in n.model.variables if "power_rate" in name]
    assert not [name for name in n.model.constraints if name.endswith("-definition")]

    for period, hours in ((2030, hours_2030), (2040, hours_2040)):
        for region, link in region_links.items():
            expected_vars = link_p.sel(
                snapshot=[(period, ts) for ts in hours],
                Link=[link],
            ).values.reshape(-1)

            # One term per link and snapshot, on the weighting in GWh_e.
            constraint = n.model.constraints[f"FlexibleElectrolysis-annual_electricity-{region}-{period}"]
            assert constraint.rhs.item() == pytest.approx(expected_rhs[region] * 1e3)
            assert constraint.sign.item() == "="
            assert constraint.coeffs.to_numpy().reshape(-1).tolist() == pytest.approx(
                [2920.0 / 1e3] * 3,
            )
            assert constraint.vars.to_numpy().reshape(-1).tolist() == expected_vars.tolist()

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


def test_add_electrolysis_constraint_pools_electricity_target_nationally():
    n, snapshots, hours = _national_electrolysis_network(
        "nation flexible electrolysis H2",
    )

    add_electrolysis_electricity_target_constraint(
        n,
        snapshots,
        {
            "flexible_electrolysis": {
                "enable": True,
                "annual_electricity_twh": 1512,
                "accounting_region": "nation",
            },
        },
        str(HYDROGEN_DEMAND_SHARE),
    )

    # A single constraint over every electrolysis link, for the full national total.
    assert not any(
        name.startswith("FlexibleElectrolysis-annual_electricity-")
        and not name.startswith("FlexibleElectrolysis-annual_electricity-nation")
        for name in n.model.constraints
    )
    constraint = n.model.constraints["FlexibleElectrolysis-annual_electricity-nation-2030"]
    assert constraint.rhs.item() == pytest.approx(1512.0 * 1e3)
    assert constraint.sign.item() == "="

    # No auxiliary rate variable: the whole fleet sits on the one annual row.
    assert not [name for name in n.model.variables if "power_rate" in name]
    assert not [name for name in n.model.constraints if name.endswith("-definition")]

    expected_vars = (
        n.model.variables["Link-p"]
        .labels.sel(
            snapshot=[(2030, ts) for ts in hours],
            Link=["b flexible electrolysis", "b2 flexible electrolysis"],
        )
        .values.reshape(-1)
    )
    assert constraint.coeffs.to_numpy().reshape(-1).tolist() == pytest.approx(
        [2920.0 / 1e3] * 6,
    )
    assert sorted(constraint.vars.to_numpy().reshape(-1).tolist()) == sorted(
        expected_vars.tolist(),
    )

    # Capacity adequacy follows from Link-p <= p_nom, so no separate
    # capacity-energy constraint is built.
    assert not [name for name in n.model.constraints if "capacity_energy" in name]


def test_add_electrolysis_constraint_target_is_electricity_not_hydrogen():
    """The annual row fixes the fleet's grid withdrawal, with no conversion factor."""
    n, snapshots, hours = _national_electrolysis_network(
        "nation flexible electrolysis H2",
    )

    add_electrolysis_electricity_target_constraint(
        n,
        snapshots,
        {
            "flexible_electrolysis": {
                "enable": True,
                "annual_electricity_twh": 1000.0,
                "accounting_region": "nation",
            },
        },
        str(HYDROGEN_DEMAND_SHARE),
    )

    # 1000 TWh_e over 3 snapshots weighted 2920 h each: the rate is the fleet's
    # electricity draw, so p sums to the target without the 1.351 electricity
    # input per unit of hydrogen.
    constraint = n.model.constraints["FlexibleElectrolysis-annual_electricity-nation-2030"]
    assert constraint.rhs.item() == pytest.approx(1000.0 * 1e3)
    # p enters on the snapshot weighting alone -- no efficiency, no hydrogen factor.
    assert constraint.coeffs.to_numpy().reshape(-1).tolist() == pytest.approx([2920.0 / 1e3] * 6)


def test_add_electrolysis_constraint_rejects_accounting_region_network_mismatch():
    n, snapshots, _ = _national_electrolysis_network(
        "Texas flexible electrolysis H2",
    )

    with pytest.raises(ValueError, match="accounting_region"):
        add_electrolysis_electricity_target_constraint(
            n,
            snapshots,
            {
                "flexible_electrolysis": {
                    "enable": True,
                    "annual_electricity_twh": 1512,
                    "accounting_region": "nation",
                },
            },
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
    add_electrolysis_electricity_target_constraint(
        n,
        snapshots,
        {
            "flexible_electrolysis": {
                "enable": True,
                "annual_electricity_twh": target_twh,
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
        str(HYDROGEN_DEMAND_SHARE),
    )

    # Representative periods must not introduce any per-block auxiliary
    # variables or constraints; the target stays a single annual equality.
    assert not [name for name in n.model.variables if "hydrogen_budget" in name]
    assert not [name for name in n.model.constraints if "-block_" in name]

    # No auxiliary rate variable either: one row carrying every link and snapshot.
    assert not [name for name in n.model.variables if "power_rate" in name]
    assert not [name for name in n.model.constraints if name.endswith("-definition")]

    annual = n.model.constraints[
        "FlexibleElectrolysis-annual_electricity-Texas-2030"
    ]
    link_p_labels = n.model["Link-p"].labels.to_numpy().reshape(-1)
    assert annual.vars.to_numpy().reshape(-1).tolist() == link_p_labels.tolist()
    assert annual.coeffs.to_numpy().reshape(-1).tolist() == pytest.approx([2190.0 / 1e3] * 4)
    assert annual.rhs.item() == pytest.approx(target_twh * 1e3)


def test_store_electrolysis_duals_recovers_the_marginal_cost(monkeypatch):
    """The stored price is per MWh_e and does not depend on the row scale."""
    hours = pd.date_range("2030-01-01 00:00", "2030-01-01 02:00", freq="h")
    snapshots = pd.MultiIndex.from_tuples(
        [(2030, ts) for ts in hours],
        names=["period", "timestep"],
    )
    config = {
        "flexible_electrolysis": {
            "enable": True,
            "annual_electricity_twh": 1.0,
            "accounting_region": "nation",
        },
    }

    def price_at(scale):
        n = pypsa.Network()
        n.set_snapshots(snapshots)
        n.set_investment_periods(periods=[2030])
        for carrier in ("AC", "gen", "load", "H2", "electrolysis"):
            n.add("Carrier", carrier)
        n.add("Bus", "b", carrier="AC")
        n.add("Bus", "nation flexible electrolysis H2", carrier="H2")
        # The only way to serve the target, at a known marginal cost.
        n.add("Generator", "g", bus="b", carrier="gen", p_nom=1e6, marginal_cost=7.0)
        n.add("Load", "l", bus="b", carrier="load", p_set=pd.Series(0.0, index=snapshots))
        n.add(
            "Link",
            "b flexible electrolysis",
            bus0="b",
            bus1="nation flexible electrolysis H2",
            carrier="electrolysis",
            p_nom=0.0,
            p_nom_extendable=True,
            efficiency=0.0,
        )
        # Both columns: the row uses ``generators``, the objective ``objective``.
        n.snapshot_weightings.loc[:, :] = 2920.0

        monkeypatch.setattr(solve_network_module, "ELECTROLYSIS_ROW_SCALE", scale)
        n.optimize(
            solver_name="highs",
            multi_investment_periods=True,
            extra_functionality=lambda n, sns: add_electrolysis_electricity_target_constraint(
                n, sns, config, str(HYDROGEN_DEMAND_SHARE),
            ),
        )
        store_electrolysis_duals(n)
        assert hasattr(n, "electrolysis_electricity_price"), "No electrolysis price was stored"
        assert list(n.electrolysis_electricity_price.index) == [
            "FlexibleElectrolysis-annual_electricity-nation-2030",
        ]
        return n.electrolysis_electricity_price.iloc[0]

    # Every extra MWh_e is served by the 7/MWh generator, so that is the price.
    unscaled = price_at(1.0)
    assert abs(unscaled) == pytest.approx(7.0, rel=1e-6)
    # Multiplying instead of dividing would separate these by 1e6.
    assert price_at(1e3) == pytest.approx(unscaled, rel=1e-6)


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

    # The damping strength is this repo's choice and config.default.yaml exposes
    # it, so a configured proximal_* value is forwarded. The convergence test is
    # calibrated inside PyPSA and is not, and ``max_iterations`` is not in this
    # set either - ``_run_standard_optimize`` passes it separately.
    forwarded = solve_network_module._iterative_optimize_kwargs(
        {
            "proximal": True,
            "proximal_weight": 0.1,
            "proximal_ceiling": 8,
            "progress_target": 0.5,
            "cost_threshold": 1.0e-4,
            "max_iterations": 20,
        },
    )
    assert forwarded == {
        "proximal": True,
        "proximal_weight": 0.1,
        "proximal_ceiling": 8,
    }
    # An unconfigured weight stays out, so PyPSA keeps its own default.
    assert solve_network_module._iterative_optimize_kwargs(
        {"proximal": True, "proximal_weight": None},
    ) == {"proximal": True}
    assert solve_network_module._iterative_optimize_kwargs({"proximal": False}) == {
        "proximal": False
    }


def test_run_standard_optimize_passes_the_proximal_switch_through(monkeypatch):
    captured = {}

    class FakeOptimize:
        def optimize_transmission_expansion_iteratively(self, **kwargs):
            captured.update(kwargs)
            return "ok", "optimal"

    network = SimpleNamespace(optimize=FakeOptimize())
    cf_solving = {"scheme": "slp", "proximal": True, "proximal_weight": 0.1}

    status, condition = solve_network_module._run_standard_optimize(
        network, rolling_horizon=False, skip_iterations=False, cf_solving=cf_solving
    )

    assert (status, condition) == ("ok", "optimal")
    assert captured["proximal"] is True
    # The damping strength is configurable here; the convergence criterion is
    # PyPSA's and is never set by this repo.
    assert captured["proximal_weight"] == 0.1
    assert "cost_threshold" not in captured
    assert "cost_window" not in captured

    captured.clear()
    solve_network_module._run_standard_optimize(
        SimpleNamespace(optimize=FakeOptimize()),
        rolling_horizon=False,
        skip_iterations=False,
        cf_solving={"scheme": "slp", "proximal": True},
    )
    assert "proximal_weight" not in captured


@pytest.mark.parametrize(
    ("options", "expected"),
    [({}, True), ({"split_capacity_by_representative_period": True}, True),
     ({"split_capacity_by_representative_period": False}, False)],
)
def test_extra_functionality_splits_capacity_last_unless_switched_off(monkeypatch, options, expected):
    calls = []
    for name in (
        "add_bidirectional_link_constraints",
        "add_representative_period_storage_constraints",
        "add_electrolysis_electricity_target_constraint",
        "add_line_x_sssc_total_max_constraint",
        "add_line_x_sssc_line_capacity_constraint",
        "tighten_line_x_sssc_bound",
        "split_capacity_by_representative_period",
    ):
        monkeypatch.setattr(solve_network_module, name, lambda *a, name=name, **k: calls.append(name))

    n = SimpleNamespace(opts=[], config={"solving": {"options": options}})
    solve_network_module.extra_functionality(n, pd.Index([]))

    assert ("split_capacity_by_representative_period" in calls) is expected
    if expected:
        assert calls[-1] == "split_capacity_by_representative_period"


@pytest.mark.parametrize(
    ("solver_name", "solver_options", "options", "warns"),
    [
        ("gurobi", {"Aggregate": 0}, {}, False),
        ("gurobi", {"aggregate": 1}, {}, True),
        ("gurobi", {}, {}, True),
        ("gurobi", {}, {"split_capacity_by_representative_period": False}, False),
        ("highs", {}, {}, False),
    ],
)
def test_warns_when_gurobi_aggregator_would_undo_the_capacity_split(caplog, solver_name, solver_options, options, warns):
    with caplog.at_level(logging.WARNING, logger=solve_network_module.logger.name):
        solve_network_module._warn_if_aggregator_undoes_capacity_split(options, solver_name, solver_options)

    assert ("Aggregate" in caplog.text) is warns
