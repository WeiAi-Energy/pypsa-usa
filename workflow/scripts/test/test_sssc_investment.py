"""SSSC investment on AC lines (lines: sssc), with per-period capacities under perfect foresight."""

import pandas as pd
import pypsa
import pytest
from _helpers import calculate_annuity
from opts.bidirectional_link import add_bidirectional_link_constraints
from opts.sssc import add_period_sssc_constraints, store_period_sssc_capacity
from opts.transmission import (
    add_period_transmission_limits,
    mark_staged_transmission,
    store_period_transmission_capacity,
)
from prepare_network import attach_sssc, parse_ll_by_period, set_transmission_limit_by_period

pytestmark = pytest.mark.skipif(
    "sssc_nom_extendable" not in pypsa.Network().lines.columns,
    reason="needs pypsa-bill (SSSC line attributes)",
)

PERIODS = [2030, 2040]
SSSC = {"enable": True, "capex": 80, "lifetime": 20, "wacc_real": 0.036, "fom_fraction": 0.015, "max_pu": 0.3}
X = 0.01  # an SSSC angle shift of d rad moves d / (3 X) MW from line ac onto the path a-b-c


def triangle(load_2030, load_2040, periods=PERIODS):
    """
    Cheap generation at ``a``, dear generation and the load at ``c``, lines ac, ab, bc.

    Two thirds of the load flows over ac (100 MW), one third over a-b-c (200 MW each),
    so ac congests above 150 MW of load unless SSSCs push flow onto a-b-c.
    """
    n = pypsa.Network()
    snapshots = pd.MultiIndex.from_product(
        [periods, pd.date_range("2030-01-01", periods=2, freq="h")],
        names=["period", "timestep"],
    )
    n.set_snapshots(snapshots)
    n.set_investment_periods(periods)
    n.add("Bus", ["a", "b", "c"], carrier="AC")
    n.add("Carrier", ["AC", "gas"])
    n.add("Generator", "cheap", bus="a", carrier="gas", p_nom=1000, marginal_cost=1)
    n.add("Generator", "dear", bus="c", carrier="gas", p_nom=1000, marginal_cost=100)
    loads = [load_2030, load_2040][: len(periods)]
    n.add("Load", "load", bus="c", p_set=pd.Series([v for v in loads for _ in range(2)], index=snapshots))
    for name, bus0, bus1, s_nom in [("ac", "a", "c", 100), ("ab", "a", "b", 200), ("bc", "b", "c", 200)]:
        n.add(
            "Line",
            name,
            bus0=bus0,
            bus1=bus1,
            carrier="AC",
            x=X,
            r=X / 10,
            s_nom=s_nom,
            length=100,
            capital_cost=0.01,
            build_year=periods[0],
            lifetime=100,
        )
    return n


def solve(n):
    def extra_functionality(n, snapshots):
        add_bidirectional_link_constraints(n)
        add_period_transmission_limits(n, snapshots)
        add_period_sssc_constraints(n, snapshots)

    multi = len(n.investment_periods) > 1
    mark_staged_transmission(n)
    status, _ = n.optimize(
        multi_investment_periods=multi,
        solver_name="highs",
        extra_functionality=extra_functionality,
    )
    assert status == "ok"
    store_period_transmission_capacity(n, n.investment_periods)
    store_period_sssc_capacity(n, n.investment_periods)
    return n


def test_attach_sssc():
    n = triangle(120, 120)
    n.add("Line", "new", bus0="a", bus1="c", carrier="AC", x=X, s_nom=0)
    attach_sssc(n, SSSC)

    cost = 80e3 * (calculate_annuity(20, 0.036) + 0.015)
    assert cost == pytest.approx(6880, rel=1e-3)
    with_sssc = ["ac", "ab", "bc"]
    assert n.lines.loc[with_sssc, "sssc_nom_extendable"].all()
    assert n.lines.loc[with_sssc, "capital_cost_sssc"].to_numpy() == pytest.approx(cost)
    assert n.lines.loc[with_sssc, "sssc_nom_max_pu"].to_numpy() == pytest.approx(0.3)
    # an SSSC enters the voltage law at the line's s_nom, which must be positive
    assert not n.lines.at["new", "sssc_nom_extendable"]


@pytest.mark.parametrize("sssc", [None, {**SSSC, "enable": False}])
def test_attach_sssc_disabled(sssc):
    n = triangle(120, 120)
    attach_sssc(n, sssc)
    assert not n.lines.sssc_nom_extendable.any()


def test_transport_model_skips_sssc():
    n = triangle(120, 120)
    attach_sssc(n, SSSC, transport_model=True)
    assert not n.lines.sssc_nom_extendable.any()


def test_single_period_relieves_congestion():
    n = triangle(180, 180, periods=[2030])
    attach_sssc(n, SSSC)
    n.lines["capital_cost_sssc"] = 1e-3
    solve(n)
    assert n.lines.sssc_nom_opt.sum() > 0
    assert not n.lines.columns.str.fullmatch(r"sssc_nom_\d{4}_opt").any()
    # the SSSCs move the 20 MW above ac's rating onto a-b-c: no dear generation
    assert n.generators_t.p["dear"].max() == pytest.approx(0, abs=1e-6)
    assert n.lines_t.p0["ac"].max() == pytest.approx(100)


def test_sssc_paid_and_used_only_from_build_period():
    """An SSSC needed only for the 2040 load is neither built, used nor paid in 2030."""
    n = triangle(120, 180)
    attach_sssc(n, SSSC)
    cost = 1.0
    n.lines["capital_cost_sssc"] = cost
    solve(n)

    assert n.lines.sssc_nom_2030_opt.to_numpy() == pytest.approx(0, abs=1e-6)
    assert n.lines.sssc_nom_2040_opt.sum() > 0
    assert n.lines.sssc_nom_2040_opt.to_numpy() == pytest.approx(n.lines.sssc_nom_opt.to_numpy())
    assert n.lines_t.q_sssc.loc[2030].abs().to_numpy().max() == pytest.approx(0, abs=1e-6)
    assert n.generators_t.p["dear"].max() == pytest.approx(0, abs=1e-6)

    w = n.investment_period_weightings.objective
    dispatch = (n.generators_t.p * n.generators.marginal_cost).sum(axis=1)
    opex = (dispatch.groupby(level="period").sum() * w).sum()
    capex = cost * n.lines.sssc_nom_2040_opt.sum() * w[2040]
    assert n.objective + n.objective_constant == pytest.approx(opex + capex)


def test_sssc_limited_by_line_capacity_of_each_period():
    """In 2030 the SSSCs are capped by the 2030 line capacity, not by the larger 2040 one."""
    n = triangle(200, 400)
    set_transmission_limit_by_period(n, parse_ll_by_period("2040vopt"))
    attach_sssc(n, SSSC)
    n.lines["capital_cost_sssc"] = 1e-3
    solve(n)

    s_2030, s_2040 = n.lines.s_nom_2030_opt, n.lines.s_nom_2040_opt
    assert s_2030.to_numpy() == pytest.approx(n.lines.s_nom.to_numpy())
    assert s_2040["ac"] > s_2030["ac"] + 1
    # 2030 needs a 33 MW shift but its SSSCs can move only 0.3 * 3 lines / (3 X) = 30 MW
    assert n.lines.sssc_nom_2030_opt.to_numpy() == pytest.approx(0.3 * s_2030.to_numpy(), rel=1e-6)
    assert (n.lines.sssc_nom_2040_opt <= 0.3 * s_2040 + 1e-6).all()
    assert n.lines.sssc_nom_2040_opt["ac"] > 0.3 * s_2030["ac"] + 1
    assert (n.lines_t.q_sssc.loc[2030].abs().max() <= n.lines.sssc_nom_2030_opt + 1e-6).all()
