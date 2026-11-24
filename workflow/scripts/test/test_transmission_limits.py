"""Per-period transmission limits from a per-period ll wildcard (e.g. 2030v1.2+2040v1.5)."""

import pandas as pd
import pypsa
import pytest
from opts.bidirectional_link import add_bidirectional_link_constraints
from opts.transmission import (
    add_period_transmission_limits,
    mark_staged_transmission,
    store_period_transmission_capacity,
    vintaged_branches,
)
from prepare_network import (
    apply_transmission_limit,
    halve_paired_link_length,
    parse_ll_by_period,
    set_transmission_limit_by_period,
)

PERIODS = [2030, 2040]


def two_bus_network():
    """Cheap generation at ``a``, dear generation and a growing load at ``b``, one 100 MW line."""
    n = pypsa.Network()
    snapshots = pd.MultiIndex.from_product(
        [PERIODS, pd.date_range("2030-01-01", periods=2, freq="h")],
        names=["period", "timestep"],
    )
    n.set_snapshots(snapshots)
    n.set_investment_periods(PERIODS)
    n.add("Bus", ["a", "b"], carrier="AC")
    n.add("Carrier", ["AC", "gas"])
    n.add("Generator", "cheap", bus="a", carrier="gas", p_nom=1000, marginal_cost=1)
    n.add("Generator", "dear", bus="b", carrier="gas", p_nom=1000, marginal_cost=100)
    load = pd.Series([200.0] * 2 + [300.0] * 2, index=snapshots)
    n.add("Load", "load", bus="b", p_set=load)
    n.add(
        "Line",
        "ab",
        bus0="a",
        bus1="b",
        carrier="AC",
        x=0.1,
        r=0.01,
        s_nom=100,
        length=100,
        capital_cost=0.01,
        build_year=2030,
        lifetime=100,
    )
    return n


def solve(n):
    def extra_functionality(n, snapshots):
        add_bidirectional_link_constraints(n)
        add_period_transmission_limits(n, snapshots)

    mark_staged_transmission(n)
    status, _ = n.optimize(
        multi_investment_periods=True,
        solver_name="highs",
        extra_functionality=extra_functionality,
    )
    assert status == "ok"
    store_period_transmission_capacity(n, n.investment_periods)
    return n


def test_parse_ll_by_period():
    assert parse_ll_by_period("v1.1") is None
    assert parse_ll_by_period("copt") is None
    assert parse_ll_by_period("2040v1.1+2050copt") == {2040: ("v", "1.1"), 2050: ("c", "opt")}
    with pytest.raises(ValueError):
        parse_ll_by_period("2040v1.1+v1.2")


def test_unknown_period_raises():
    with pytest.raises(ValueError, match="not planning horizons"):
        set_transmission_limit_by_period(two_bus_network(), {2050: ("v", "1.2")})


def test_each_period_capped():
    n = two_bus_network()
    set_transmission_limit_by_period(n, parse_ll_by_period("2030v1.2+2040v1.5"))
    assert n.lines.at["ab", "s_nom_extendable"]
    assert n.global_constraints.constant.to_dict() == {"lv_limit_2030": 12000.0, "lv_limit_2040": 15000.0}
    assert set(n.global_constraints.investment_period) == set(PERIODS)

    solve(n)
    assert n.lines.at["ab", "s_nom_2030_opt"] == pytest.approx(120)
    assert n.lines.at["ab", "s_nom_2040_opt"] == pytest.approx(150)
    flow = n.lines_t.p0["ab"]
    assert flow.loc[2030].max() == pytest.approx(120)
    assert flow.loc[2040].max() == pytest.approx(150)


def test_periods_before_first_entry_stay_fixed():
    n = two_bus_network()
    set_transmission_limit_by_period(n, parse_ll_by_period("2040v1.5"))
    solve(n)
    assert n.lines.at["ab", "s_nom_2030_opt"] == pytest.approx(100)
    assert n.lines_t.p0["ab"].loc[2030].max() == pytest.approx(100)
    assert n.lines.at["ab", "s_nom_2040_opt"] == pytest.approx(150)


def test_line_paid_only_from_build_period():
    """A line worth building only for the 2040 load is not charged in 2030."""
    n = two_bus_network()
    n.loads_t.p_set.loc[2030, "load"] = 100.0
    n.lines.at["ab", "capital_cost"] = 50.0
    set_transmission_limit_by_period(n, parse_ll_by_period("2030vopt"))
    solve(n)
    assert n.lines.at["ab", "s_nom_2030_opt"] == pytest.approx(100)
    assert n.lines.at["ab", "s_nom_2040_opt"] == pytest.approx(300)

    w = n.investment_period_weightings.objective
    dispatch = (n.generators_t.p * n.generators.marginal_cost).sum(axis=1)
    opex = (dispatch.groupby(level="period").sum() * w).sum()
    capex = 50.0 * (100 * w[2030] + 300 * w[2040])
    assert n.objective + n.objective_constant == pytest.approx(opex + capex)


def test_iterative_expansion_keeps_period_caps():
    """PyPSA's last iteration fixes lines at s_nom_opt; earlier periods keep their own capacity."""
    n = two_bus_network()
    set_transmission_limit_by_period(n, parse_ll_by_period("2030v1.2+2040v1.5"))
    n.optimize.optimize_transmission_expansion_iteratively(
        multi_investment_periods=True,
        solver_name="highs",
        extra_functionality=lambda n, snapshots: add_period_transmission_limits(n, snapshots),
        max_iterations=3,
    )
    store_period_transmission_capacity(n, n.investment_periods)
    assert n.lines.at["ab", "s_nom_2030_opt"] == pytest.approx(120)
    assert n.lines.at["ab", "s_nom_2040_opt"] == pytest.approx(150)
    assert n.lines_t.p0["ab"].loc[2030].max() == pytest.approx(120)


def test_transport_links():
    n = two_bus_network()
    n.remove("Line", "ab")
    n.add(
        "Link",
        "ab",
        bus0="a",
        bus1="b",
        carrier="AC",
        p_nom=100,
        p_min_pu=-1,
        length=100,
        capital_cost=0.01,
        build_year=2030,
        lifetime=100,
    )
    set_transmission_limit_by_period(n, parse_ll_by_period("2030v1.2+2040v1.5"))
    solve(n)
    assert n.links.at["ab", "p_nom_2030_opt"] == pytest.approx(120)
    assert n.links.at["ab", "p_nom_2040_opt"] == pytest.approx(150)
    assert n.links_t.p0["ab"].loc[2030].max() == pytest.approx(120)


def test_single_ll_applies_to_every_period():
    """With several periods a single ll caps each period and lines still expand gradually."""
    n = two_bus_network()
    n.loads_t.p_set.loc[2030, "load"] = 100.0
    n.lines.at["ab", "capital_cost"] = 50.0
    apply_transmission_limit(n, "v2.0")
    assert n.global_constraints.type.unique().tolist() == ["transmission_volume_expansion_limit_per_period"]
    assert n.global_constraints.constant.to_dict() == {"lv_limit_2030": 20000.0, "lv_limit_2040": 20000.0}

    solve(n)
    assert n.lines.at["ab", "s_nom_2030_opt"] == pytest.approx(100)
    assert n.lines.at["ab", "s_nom_2040_opt"] == pytest.approx(200)


def test_single_period_keeps_pypsa_limit():
    n = two_bus_network()
    n.set_snapshots(n.snapshots[n.snapshots.get_level_values("period") == 2030])
    n.set_investment_periods([2030])
    apply_transmission_limit(n, "v1.5")
    assert n.global_constraints.type.tolist() == ["transmission_volume_expansion_limit"]


def test_bidirectional_pair_expands_together():
    """A '_fwd'/'_rev' interface pair builds the same capacity in every period, as in solve_network."""
    n = two_bus_network()
    n.remove("Line", "ab")
    for name, bus0, bus1 in (("ab_fwd", "a", "b"), ("ab_rev", "b", "a")):
        n.add(
            "Link",
            name,
            bus0=bus0,
            bus1=bus1,
            carrier="AC",
            p_nom=100,
            length=100,
            capital_cost=0.01,
            build_year=2030,
            lifetime=100,
        )
    set_transmission_limit_by_period(n, parse_ll_by_period("2030v1.2+2040v1.5"))
    solve(n)
    for period, built in ((2030, 120), (2040, 150)):
        assert n.links.loc[["ab_fwd", "ab_rev"], f"p_nom_{period}_opt"].tolist() == pytest.approx([built, built])


def test_paired_links_count_once_in_volume_limit():
    """An HVDC line built as a '_fwd'/'_rev' pair counts its MW*km once, like a line."""
    n = two_bus_network()
    for name, bus0, bus1 in (("dc_fwd", "a", "b"), ("dc_rev", "b", "a")):
        n.add(
            "Link",
            name,
            bus0=bus0,
            bus1=bus1,
            carrier="DC",
            p_nom=50,
            length=200,
            capital_cost=0.01,
            build_year=2030,
            lifetime=100,
        )
    n.add("Link", ["h2_fwd", "h2_rev"], bus0=["a", "b"], bus1=["b", "a"], carrier="H2 pipeline", length=300)
    halve_paired_link_length(n)
    assert n.links.length.to_dict() == {"dc_fwd": 100, "dc_rev": 100, "h2_fwd": 300, "h2_rev": 300}

    # 100 MW x 100 km of line plus 50 MW x 200 km of HVDC corridor
    apply_transmission_limit(n, "v1.25")
    assert n.global_constraints.constant.unique().tolist() == [25000.0]
    solve(n)
    # per MW*km the line adds the most transfer: 100 * s_nom + 200 * 50 <= 25000
    assert n.lines.at["ab", "s_nom_opt"] == pytest.approx(150)
    assert n.links.loc[["dc_fwd", "dc_rev"], "p_nom_opt"].tolist() == pytest.approx([50, 50])


def test_vintaged_links_keep_their_own_capacity():
    """Transport-model links with a copy per later period get no per-period capacity."""
    n = two_bus_network()
    n.remove("Line", "ab")
    for name, p_nom, build_year in (("ab", 100, 2030), ("ab_2040", 0, 2040)):
        n.add(
            "Link",
            name,
            bus0="a",
            bus1="b",
            carrier="AC",
            p_nom=p_nom,
            p_min_pu=-1,
            length=100,
            capital_cost=0.01,
            build_year=build_year,
            lifetime=100,
        )
    assert vintaged_branches(n.links.index, PERIODS).tolist() == ["ab_2040", "ab"]
    set_transmission_limit_by_period(n, parse_ll_by_period("2030v1.2+2040v1.5"))
    solve(n)
    assert "Link-p_nom_2030" not in n.model.variables
    assert n.links.at["ab", "p_nom_opt"] == pytest.approx(120)
    assert n.links.at["ab_2040", "p_nom_opt"] == pytest.approx(30)
    assert n.links["p_nom_2030_opt"].to_dict() == pytest.approx({"ab": 120, "ab_2040": 0})
    assert n.links_t.p0.loc[2040].sum(axis=1).max() == pytest.approx(150)
