"""Convex piecewise expansion costs for lines, with several investment periods and ll limits."""

import pandas as pd
import pytest
from opts.piecewise_transmission import (
    BASE_COLUMN,
    COST_COLUMN,
    add_piecewise_line_costs,
    attach_piecewise_line_costs,
    correct_iterated_objective,
    line_cost_segments,
    piecewise_line_cost,
    read_piecewise_line_costs,
    store_piecewise_line_costs,
)
from opts.transmission import (
    add_period_transmission_limits,
    mark_staged_transmission,
    store_period_transmission_capacity,
)
from prepare_network import apply_transmission_limit, parse_ll_by_period, set_transmission_limit_by_period
from test_transmission_limits import PERIODS, two_bus_network

# Moving 1 MW from a to b saves 99/MWh in each of a period's two hours: 198 per MW and period.
CURVE = pd.DataFrame(
    {"line": "ab", "breakpoint": [50, 150, float("inf")], "capital_cost": [10.0, 100.0, 1000.0]},
)


def extra_functionality(n, snapshots):
    add_period_transmission_limits(n, snapshots)
    add_piecewise_line_costs(n, snapshots)


def solve(n, **kwargs):
    mark_staged_transmission(n)
    status, _ = n.optimize(
        multi_investment_periods=True,
        solver_name="highs",
        extra_functionality=extra_functionality,
        **kwargs,
    )
    assert status == "ok"
    store_period_transmission_capacity(n, n.investment_periods)
    store_piecewise_line_costs(n)
    return n


def expandable(n):
    """Let the line expand from its 100 MW without any ll limit."""
    n.lines.loc["ab", ["s_nom_min", "s_nom_extendable"]] = [100.0, True]
    return n


def opex(n):
    w = n.investment_period_weightings.objective
    dispatch = (n.generators_t.p * n.generators.marginal_cost).sum(axis=1)
    return (dispatch.groupby(level="period").sum() * w).sum()


def single_period(n):
    n.set_snapshots(n.snapshots[n.snapshots.get_level_values("period") == 2030])
    n.set_investment_periods([2030])
    return n


def test_segments():
    segments = line_cost_segments(CURVE.iloc[::-1], pd.Index(["ab"]))
    assert segments["lower"].tolist() == [0, 50, 150]
    assert segments["width"].tolist() == [50, 100, float("inf")]
    for added, cost in ((0, 0), (30, 300), (120, 7500), (200, 60500), (-5, 0)):
        assert piecewise_line_cost(segments, pd.Series({"ab": float(added)})).at["ab"] == cost


def test_read_csv(tmp_path):
    path = tmp_path / "curves.csv"
    path.write_text("line,breakpoint,capital_cost\n7,100,1.0\n7,,2.0\n")
    segments = line_cost_segments(read_piecewise_line_costs(path), pd.Index(["7"]))
    assert segments["width"].tolist() == [100, float("inf")]


@pytest.mark.parametrize(
    ("curve", "match"),
    [
        ({"line": "cd", "breakpoint": [50], "capital_cost": [1.0]}, "not in the network"),
        ({"line": "ab", "breakpoint": [50, 100], "capital_cost": [2.0, 1.0]}, "must not decrease"),
        ({"line": "ab", "breakpoint": [50, 50], "capital_cost": [1.0, 2.0]}, "repeated breakpoint"),
        ({"line": "ab", "breakpoint": [0, 50], "capital_cost": [1.0, 2.0]}, "must be positive"),
    ],
)
def test_invalid_curves(curve, match):
    with pytest.raises(ValueError, match=match):
        line_cost_segments(pd.DataFrame(curve), pd.Index(["ab"]))


def test_single_period():
    """100 MW are wanted: the 10/MW segment is built, the 100/MW one too, as both cost less than 198."""
    n = expandable(single_period(two_bus_network()))
    attach_piecewise_line_costs(n, CURVE)
    solve(n)
    assert n.lines.at["ab", "s_nom_opt"] == pytest.approx(200)
    assert n.lines.at["ab", COST_COLUMN] == pytest.approx(10 * 50 + 100 * 50)

    w = n.investment_period_weightings.objective[2030]
    capex = w * (0.01 * 100 + 10 * 50 + 100 * 50)
    assert n.objective + n.objective_constant == pytest.approx(opex(n) + capex)


def test_last_segment_caps_expansion():
    n = expandable(single_period(two_bus_network()))
    attach_piecewise_line_costs(n, pd.DataFrame({"line": "ab", "breakpoint": [40], "capital_cost": [1.0]}))
    solve(n)
    assert n.lines.at["ab", "s_nom_opt"] == pytest.approx(140)


def test_each_period_pays_the_curve_up_to_its_capacity():
    """
    2030 wants 100 MW more, 2040 200 MW: 2030 builds the first two segments, 2040 stops
    before the 1000/MW one. Each period pays the curve up to what it has.
    """
    n = two_bus_network()
    n.loads_t.p_set.loc[2040, "load"] = 300.0
    set_transmission_limit_by_period(n, parse_ll_by_period("2030vopt"))
    attach_piecewise_line_costs(n, CURVE)
    solve(n)
    assert n.lines.at["ab", "s_nom_2030_opt"] == pytest.approx(200)
    assert n.lines.at["ab", "s_nom_2040_opt"] == pytest.approx(250)
    assert n.lines.at["ab", f"{COST_COLUMN}_2030"] == pytest.approx(10 * 50 + 100 * 50)
    assert n.lines.at["ab", f"{COST_COLUMN}_2040"] == pytest.approx(10 * 50 + 100 * 100)

    w = n.investment_period_weightings.objective
    capex = sum(w[p] * (0.01 * 100 + n.lines.at["ab", f"{COST_COLUMN}_{p}"]) for p in PERIODS)
    assert n.objective + n.objective_constant == pytest.approx(opex(n) + capex)


def test_earlier_period_build_waits_for_the_cheaper_need():
    """A segment worth building only for the 2040 load is not paid in 2030."""
    n = two_bus_network()
    n.loads_t.p_set.loc[2030, "load"] = 100.0
    set_transmission_limit_by_period(n, parse_ll_by_period("2030vopt"))
    attach_piecewise_line_costs(n, CURVE)
    solve(n)
    assert n.lines.at["ab", "s_nom_2030_opt"] == pytest.approx(100)
    assert n.lines.at["ab", "s_nom_2040_opt"] == pytest.approx(250)


def test_myopic_horizons_measure_from_the_original_capacity():
    """The second horizon pays the curve from the original 100 MW, so what 2030 built keeps the cheap segments."""
    n = two_bus_network()
    n.loads_t.p_set.loc[2040, "load"] = 300.0
    set_transmission_limit_by_period(n, parse_ll_by_period("2030vopt"))
    attach_piecewise_line_costs(n, CURVE)
    for period in PERIODS:
        sns = n.snapshots[n.snapshots.get_level_values(0) == period]
        status, _ = n.optimize(
            snapshots=sns,
            multi_investment_periods=True,
            solver_name="highs",
            extra_functionality=extra_functionality,
        )
        assert status == "ok"
        store_period_transmission_capacity(n, [period])
        n.lines.loc["ab", "s_nom_min"] = max(n.lines.at["ab", "s_nom_min"], n.lines.at["ab", "s_nom_opt"])
    store_piecewise_line_costs(n)
    assert n.lines.at["ab", BASE_COLUMN] == 100
    assert n.lines.at["ab", "s_nom_2030_opt"] == pytest.approx(200)
    assert n.lines.at["ab", "s_nom_2040_opt"] == pytest.approx(250)


def test_volume_limit_per_period():
    """The curve does not change what an lv limit counts."""
    n = two_bus_network()
    set_transmission_limit_by_period(n, parse_ll_by_period("2030v1.2+2040v1.5"))
    attach_piecewise_line_costs(n, CURVE)
    solve(n)
    assert n.lines.at["ab", "s_nom_2030_opt"] == pytest.approx(120)
    assert n.lines.at["ab", "s_nom_2040_opt"] == pytest.approx(150)


def cost_limited_network():
    """capital_cost 10/MW gives an lc reference of 10 * 100 = 1000; added capacity costs 10, then 30/MW."""
    n = two_bus_network()
    n.loads_t.p_set["load"] = 300.0
    n.lines.at["ab", "capital_cost"] = 10.0
    curve = pd.DataFrame({"line": "ab", "breakpoint": [20, float("inf")], "capital_cost": [10.0, 30.0]})
    return n, curve


def test_cost_limit_per_period():
    """
    c1.5 leaves 500 for expansion: 20 MW at 10 and 10 MW at 30. c2.0 leaves 1000:
    20 MW at 10 and 26.67 MW at 30. At a linear 10/MW the line would add 50 and 100 MW.
    """
    n, curve = cost_limited_network()
    set_transmission_limit_by_period(n, parse_ll_by_period("2030c1.5+2040c2.0"))
    attach_piecewise_line_costs(n, curve)
    solve(n)
    assert n.lines.at["ab", "s_nom_2030_opt"] == pytest.approx(130)
    assert n.lines.at["ab", "s_nom_2040_opt"] == pytest.approx(100 + 20 + 800 / 30)


def test_pypsa_cost_limit():
    """A single period keeps PyPSA's own lc constraint, which counts the curve too."""
    n, curve = cost_limited_network()
    single_period(n)
    apply_transmission_limit(n, "c1.5")
    assert n.global_constraints.type.tolist() == ["transmission_expansion_cost_limit"]
    attach_piecewise_line_costs(n, curve)
    solve(n)
    assert n.lines.at["ab", "s_nom_opt"] == pytest.approx(130)


def test_iterative_expansion_with_fixed_lines_in_cost_limit():
    """
    PyPSA's last iteration fixes the line. The AC link next to it stays extendable and
    must still see the budget the line takes along its curve.

    Reference 10 * 100 + 15 * 100 = 2500, c1.2 leaves 500: the line's first 20 MW at 12
    (240), then the link at 15 (17.33 MW); the line's next segment at 40 is dearer.
    Counting the line at capital_cost 10 in the last iteration would give the link 2.67 MW more.
    """
    n = two_bus_network()
    n.loads_t.p_set["load"] = 400.0
    n.lines.at["ab", "capital_cost"] = 10.0
    n.add(
        "Link",
        "ab_link",
        bus0="a",
        bus1="b",
        carrier="AC",
        p_nom=100,
        p_min_pu=-1,
        length=100,
        capital_cost=15.0,
        build_year=2030,
        lifetime=100,
    )
    set_transmission_limit_by_period(n, parse_ll_by_period("2030c1.2"))
    curve = pd.DataFrame({"line": "ab", "breakpoint": [20, float("inf")], "capital_cost": [12.0, 40.0]})
    attach_piecewise_line_costs(n, curve)
    mark_staged_transmission(n)
    n.optimize.optimize_transmission_expansion_iteratively(
        multi_investment_periods=True,
        solver_name="highs",
        extra_functionality=extra_functionality,
        max_iterations=3,
    )
    store_period_transmission_capacity(n, n.investment_periods)
    for period in PERIODS:
        assert n.lines.at["ab", f"s_nom_{period}_opt"] == pytest.approx(120)
    assert n.links.at["ab_link", "p_nom_opt"] == pytest.approx(100 + 260 / 15)


def discounted(n, rate=0.05):
    """Ten-year periods discounted at ``rate``, weighted as prepare_network does."""
    years = pd.Series(10.0, index=PERIODS)
    start = years.cumsum().shift(fill_value=0.0)
    n.investment_period_weightings["years"] = years
    n.investment_period_weightings["objective"] = [
        sum((1 + rate) ** -t for t in range(int(s), int(s + y))) for s, y in zip(start, years)
    ]
    return n


def growing_network():
    """2030 wants 100 MW more (200 MW built), 2040 200 MW more (250 MW built), discounted at 5 %."""
    n = discounted(two_bus_network())
    n.loads_t.p_set.loc[2040, "load"] = 300.0
    set_transmission_limit_by_period(n, parse_ll_by_period("2030vopt"))
    attach_piecewise_line_costs(n, CURVE)
    return n


def iterate(n, **kwargs):
    mark_staged_transmission(n)
    status, _ = n.optimize.optimize_transmission_expansion_iteratively(
        multi_investment_periods=True,
        solver_name="highs",
        extra_functionality=extra_functionality,
        max_iterations=3,
        **kwargs,
    )
    assert status == "ok"
    return correct_iterated_objective(n, kwargs.get("snapshots"), multi_investment_periods=True)


def test_discounted_periods_pay_the_curve():
    """Each period pays the annual curve cost of its capacity, times its discounted years."""
    n = solve(growing_network())
    w = n.investment_period_weightings.objective
    assert w[2030] > w[2040] > 0
    assert n.lines.at["ab", "s_nom_2030_opt"] == pytest.approx(200)
    assert n.lines.at["ab", "s_nom_2040_opt"] == pytest.approx(250)
    capex = sum(w[p] * (0.01 * 100 + n.lines.at["ab", f"{COST_COLUMN}_{p}"]) for p in PERIODS)
    assert n.objective + n.objective_constant == pytest.approx(opex(n) + capex)


def test_iterative_objective_charges_the_curve():
    """
    The final solve of the iterative expansion fixes the line and pypsa-bill charges what it
    added at capital_cost (0.01); after the correction the objective charges the curve, as
    the single solve does, less the cost of the existing 100 MW, which the iterative
    expansion leaves out of its objective.
    """
    once = solve(growing_network())
    n = growing_network()
    correction = iterate(n)
    store_period_transmission_capacity(n, n.investment_periods)
    for period in PERIODS:
        assert n.lines.at["ab", f"s_nom_{period}_opt"] == pytest.approx(once.lines.at["ab", f"s_nom_{period}_opt"])

    w = n.investment_period_weightings.objective
    curve = {2030: 10 * 50 + 100 * 50, 2040: 10 * 50 + 100 * 100}
    added = {2030: 100, 2040: 150}
    assert correction == pytest.approx(sum(w[p] * (curve[p] - 0.01 * added[p]) for p in PERIODS))
    existing = sum(w[p] * 0.01 * 100 for p in PERIODS)
    assert n.objective == pytest.approx(once.objective + once.objective_constant - existing)


def test_iterative_myopic_horizon_charges_the_curve_above_the_last_build():
    """In the 2040 horizon s_nom_min holds the 200 MW built for 2030; the objective charges the curve above it."""
    sns = growing_network().snapshots
    sns = sns[sns.get_level_values(0) == 2040]

    def horizon():
        n = growing_network()
        n.lines.at["ab", "s_nom_min"] = 200.0
        return n

    once = horizon()
    status, _ = once.optimize(
        snapshots=sns,
        multi_investment_periods=True,
        solver_name="highs",
        extra_functionality=extra_functionality,
    )
    assert status == "ok"
    n = horizon()
    correction = iterate(n, snapshots=sns)
    assert n.lines.at["ab", "s_nom_opt"] == pytest.approx(250)

    w = n.investment_period_weightings.objective[2040]
    # the 50 MW above 200 cost 100/MW along the curve, not 0.01
    assert correction == pytest.approx(w * (100 - 0.01) * 50)
    below = w * (0.01 * 100 + 10 * 50 + 100 * 50)  # the 200 MW at s_nom_min, along the curve
    assert n.objective == pytest.approx(once.objective + once.objective_constant - below)
