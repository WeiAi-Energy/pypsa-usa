import os
import sys

import numpy as np
import pandas as pd
import pypsa

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from simplify_network import (
    busmap_by_electrical_distance,
    clustering_from_busmap,
    merge_parallel_lines,
    reduce_low_degree_buses_and_merge_parallel_lines,
)


def _network_with_degree_two_bus():
    """A degree-two bus linked to two non-reducible K4-core buses."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    for bus in ("m", "a", "b", "c", "d"):
        n.add("Bus", bus)

    edges = [
        ("m", "a", 1.0),
        ("m", "b", 3.0),
        ("a", "b", 2.0),
        ("a", "c", 2.0),
        ("a", "d", 2.0),
        ("b", "c", 2.0),
        ("b", "d", 2.0),
        ("c", "d", 2.0),
    ]
    n.lines = pd.DataFrame(
        {
            "bus0": [edge[0] for edge in edges],
            "bus1": [edge[1] for edge in edges],
            "x": [edge[2] for edge in edges],
            "r": 0.1,
            "s_nom": 100.0,
            "s_nom_min": 0.0,
            "s_nom_max": 200.0,
            "length": 1.0,
            "type": "",
            "carrier": "AC",
        },
        index=[f"line_{i}" for i in range(len(edges))],
    )
    n.buses["Pd"] = 0.0
    n.buses.loc["m", "Pd"] = 80.0
    return n


def test_low_degree_bus_splits_capacity_and_averages_generator_attributes():
    n = _network_with_degree_two_bus()
    n.add(
        "Generator",
        "middle",
        bus="m",
        carrier="solar",
        p_nom=100.0,
        p_nom_max=200.0,
        efficiency=0.8,
        capital_cost=4.0,
        marginal_cost=2.0,
    )
    n.add(
        "Generator",
        "at_a",
        bus="a",
        carrier="solar",
        p_nom=25.0,
        p_nom_max=25.0,
        efficiency=1.0,
        capital_cost=8.0,
        marginal_cost=6.0,
    )
    n.generators_t.p_max_pu = pd.DataFrame(
        {"middle": [0.4, 0.8], "at_a": [0.8, 0.4]},
        index=n.snapshots,
    )

    reduced, _ = reduce_low_degree_buses_and_merge_parallel_lines(n)

    # x(m-a)=1 and x(m-b)=3, so the Kron split is 3/4 to a and 1/4 to b.
    assert reduced.buses.at["a", "Pd"] == 60.0
    assert reduced.buses.at["b", "Pd"] == 20.0

    at_a = reduced.generators.query("bus == 'a'").iloc[0]
    at_b = reduced.generators.query("bus == 'b'").iloc[0]
    assert at_a.p_nom == 100.0
    assert at_a.p_nom_max == 175.0
    assert at_b.p_nom == 25.0
    assert at_b.p_nom_max == 50.0
    np.testing.assert_allclose(at_a.efficiency, 0.85)
    np.testing.assert_allclose(at_a.capital_cost, 5.0)
    np.testing.assert_allclose(at_a.marginal_cost, 3.0)
    np.testing.assert_allclose(reduced.generators_t.p_max_pu[at_a.name], [0.5, 0.7])
    np.testing.assert_allclose(reduced.generators_t.p_max_pu[at_b.name], [0.4, 0.8])


def test_series_merge_sums_length_and_capital_cost():
    """A merged corridor costs what both of its segments cost, and a fresh
    parallel pair created by that merge (`m`'s series merge lands on `a`-`b`,
    which already has its own direct Line) is itself merged before the
    reduction settles -- proving the loop iterates to a joint fixed point
    rather than stopping after the first kind of reduction it finds."""
    n = _network_with_degree_two_bus()
    # $/MW proportional to length, as `update_transmission_costs` builds it.
    n.lines["capital_cost"] = n.lines.length * 7.0
    n.lines.loc["line_1", "length"] = 3.0
    n.lines.loc["line_1", "capital_cost"] = 21.0
    n.lines.loc["line_1", "s_nom"] = 60.0

    reduced, _ = reduce_low_degree_buses_and_merge_parallel_lines(n)

    # m-a and m-b combine in series into a new a-b Line (x=4, s_nom=60, the
    # narrower of the two), which then merges as a parallel pair with the
    # pre-existing a-b Line (line_2: x=2, s_nom=100) -- there is only one
    # Line left between a and b, and no bus in the K4 core dropped below
    # degree 3.
    assert (reduced.lines.bus0.isin(["a", "b"]) & reduced.lines.bus1.isin(["a", "b"])).sum() == 1
    merged = reduced.lines.loc["line_0"]
    assert {merged.bus0, merged.bus1} == {"a", "b"}
    # x_parallel = 1 / (1/4 + 1/2) = 4/3; r combines the same way (0.2 and 0.1).
    np.testing.assert_allclose(merged.x, 4 / 3)
    np.testing.assert_allclose(merged.r, 1 / 15)
    # The series segment saturates at 60*4/(4/3)=180; the direct Line at
    # 100*2/(4/3)=150 -- the direct Line binds.
    np.testing.assert_allclose(merged.s_nom, 150.0)
    # Parallel circuits run the same route, so length averages rather than sums.
    np.testing.assert_allclose(merged.length, 2.5)
    # Capacity-weighted average of the two segments' cost: (28*60 + 7*100) / 160.
    np.testing.assert_allclose(merged.capital_cost, 14.875)


def test_merge_parallel_lines_sets_capacity_from_whichever_branch_saturates_first():
    """Direct unit check of the parallel-merge formula, independent of the
    low-degree-bus machinery above."""
    lines = pd.DataFrame(
        {
            "bus0": ["p", "p"],
            "bus1": ["q", "q"],
            "x": [1.0, 2.0],
            "r": [1.0, 2.0],
            "s_nom": [100.0, 100.0],
            "length": [10.0, 12.0],
            "capital_cost": [50.0, 40.0],
        },
        index=["line_p", "line_q"],
    )

    merged, n_groups = merge_parallel_lines(lines, ["s_nom"])

    assert n_groups == 1
    assert len(merged) == 1
    row = merged.iloc[0]
    # x_parallel = 1 / (1/1 + 1/2) = 2/3; r combines the same way.
    np.testing.assert_allclose(row.x, 2 / 3)
    np.testing.assert_allclose(row.r, 2 / 3)
    # line_p saturates at 100*1/(2/3)=150 before line_q does at 100*2/(2/3)=300.
    np.testing.assert_allclose(row.s_nom, 150.0)
    np.testing.assert_allclose(row.length, 11.0)
    # Capacity-weighted: both segments carry the same s_nom, so it is a plain mean.
    np.testing.assert_allclose(row.capital_cost, 45.0)


def test_merge_parallel_lines_identical_lines_double_the_capacity():
    """Sanity check against the wrong formula: min(s_nom_i / x_i) / x_parallel
    would *halve* two identical lines' combined rating instead of doubling it."""
    lines = pd.DataFrame(
        {
            "bus0": ["p", "p"],
            "bus1": ["q", "q"],
            "x": [1.0, 1.0],
            "r": [0.1, 0.1],
            "s_nom": [100.0, 100.0],
        },
        index=["line_p", "line_q"],
    )

    merged, _ = merge_parallel_lines(lines, ["s_nom"])

    row = merged.iloc[0]
    np.testing.assert_allclose(row.x, 0.5)
    np.testing.assert_allclose(row.s_nom, 200.0)


def test_bus_with_a_dc_link_relocates_wholesale_to_the_majority_neighbour():
    """A degree-two bus carrying a Link is reduced like any other; the Link
    moves entirely to the neighbour with the larger Kron share rather than
    splitting or holding the bus out of the reduction."""
    n = _network_with_degree_two_bus()
    n.add("Link", "hvdc", bus0="m", bus1="c", p_nom=40.0)

    reduced, busmap = reduce_low_degree_buses_and_merge_parallel_lines(n)

    # x(m-a)=1 and x(m-b)=3, so `a` gets the majority (3/4) Kron share.
    assert "m" not in reduced.buses.index
    assert busmap.at["m"] == "a"
    # Its two Lines combine in series, same as the no-Link case.
    merged = reduced.lines.loc["line_0"]
    assert {merged.bus0, merged.bus1} == {"a", "b"}
    hvdc = reduced.links.loc["hvdc"]
    assert (hvdc.bus0, hvdc.bus1) == ("a", "c")
    assert hvdc.p_nom == 40.0
    # Demand weight still splits 3/4 to `a`, 1/4 to `b`, same as without a Link.
    assert reduced.buses.at["a", "Pd"] == 60.0
    assert reduced.buses.at["b", "Pd"] == 20.0


def test_link_stranded_as_a_self_loop_is_dropped():
    """If both of a Link's endpoints relocate onto the same surviving bus,
    the Link is a self-loop and gets dropped rather than left invalid."""
    n = _network_with_degree_two_bus()
    # `c` is degree-2 within the K4 core's Line set once `m`'s pair is added;
    # instead, force the self-loop directly by landing both Link ends on `m`
    # and `m`'s sole reduction target.
    n.add("Link", "hvdc", bus0="m", bus1="a", p_nom=40.0)

    reduced, busmap = reduce_low_degree_buses_and_merge_parallel_lines(n)

    assert "m" not in reduced.buses.index
    assert busmap.at["m"] == "a"
    assert "hvdc" not in reduced.links.index


def test_electrical_distance_busmap_respects_nerc_and_diameter_limits():
    n = pypsa.Network()
    for name, latitude, country in (
        ("a", 40.000, "p1"),
        ("b", 40.010, "p1"),
        ("c", 40.020, "p2"),
        ("d", 40.200, "p3"),
        ("e", 40.015, "p4"),
    ):
        n.add("Bus", name, x=-100.0, y=latitude)
        n.buses.loc[name, "country"] = country
    n.buses["reeds_zone"] = ["p1", "p1", "p2", "p3", "p4"]
    n.add("Line", "ab", bus0="a", bus1="b", x=1.0, r=0.0, s_nom=100.0)
    n.add("Line", "bc", bus0="b", bus1="c", x=1.0, r=0.0, s_nom=100.0)
    n.add("Line", "cd", bus0="c", bus1="d", x=1.0, r=0.0, s_nom=100.0)
    n.add("Line", "be", bus0="b", bus1="e", x=1.0, r=0.0, s_nom=100.0)

    busmap = busmap_by_electrical_distance(
        n,
        max_geographic_distance_km=10.0,
        max_electrical_distance_ohm=10.0,
        n_probes=128,
        seed=7,
        topological_boundary="reeds_zone",
    )

    assert busmap["a"] == busmap["b"]
    assert busmap["a"] != busmap["c"]  # ReEDS-zone boundary
    assert busmap["a"] != busmap["d"]  # geographic diameter
    assert busmap["a"] != busmap["e"]  # ReEDS-zone boundary
    clustered = clustering_from_busmap(n, busmap, line_length_factor=1.0)
    assert clustered.network.buses.at[busmap["a"], "country"] == "p1"
