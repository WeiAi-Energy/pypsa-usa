import os
import sys
from types import SimpleNamespace

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import solve_network as solve_network_module


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
            "diagnose_large_rhs_bounds": False,
            "track_iterations": True,
            "min_iterations": 2,
            "max_iterations": 3,
            "relaxation_factor": 1.25,
        },
        extra_functionality=fake_extra_functionality,
    )

    assert captured["extra_functionality"] is fake_extra_functionality
    assert captured["track_iterations"] is True
    assert captured["min_iterations"] == 2
    assert captured["max_iterations"] == 3
    assert captured["relaxation_factor"] == 1.25


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
