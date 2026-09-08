import os
import sys
from functools import reduce

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from simplify_network import (
    EARTH_RADIUS_KM,
    SHORT_BRANCH_LENGTH_KM,
    _earth_centered_km,
    _effective_reactance_embedding,
    busmap_by_target_bus_count,
    clustering_from_busmap,
    contract_short_branches,
    identity_busmap,
    reduce_low_degree_buses,
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

    reduced, _ = reduce_low_degree_buses(n)

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
    """A merged corridor costs what both of its segments cost and is rated at the
    narrower of the two."""
    n = _network_with_degree_two_bus()
    # $/MW proportional to length, as `update_transmission_costs` builds it.
    n.lines["capital_cost"] = n.lines.length * 7.0
    n.lines.loc["line_1", "length"] = 3.0
    n.lines.loc["line_1", "capital_cost"] = 21.0
    n.lines.loc["line_1", "s_nom"] = 60.0

    reduced, _ = reduce_low_degree_buses(n)

    merged = reduced.lines.loc["line_0"]
    assert {merged.bus0, merged.bus1} == {"a", "b"}
    np.testing.assert_allclose(merged.x, 4.0)  # 1 + 3
    np.testing.assert_allclose(merged.r, 0.2)  # 0.1 + 0.1
    np.testing.assert_allclose(merged.s_nom, 60.0)  # the narrower section
    np.testing.assert_allclose(merged.length, 4.0)  # 1 + 3
    np.testing.assert_allclose(merged.capital_cost, 28.0)  # 7 + 21


def test_parallel_lines_left_by_a_series_merge_are_not_merged():
    """`m`'s series merge lands on `a`-`b`, which already has its own direct Line.

    Both survive with their own attributes: an SSSC can then be sited on either,
    which is what lets it rebalance the split. Merging them would rate the pair at
    ``min(x_i s_nom_i) / x_parallel`` and write that headroom off.
    """
    n = _network_with_degree_two_bus()

    reduced, _ = reduce_low_degree_buses(n)

    a_b = reduced.lines[reduced.lines.bus0.isin(["a", "b"]) & reduced.lines.bus1.isin(["a", "b"])]
    assert len(a_b) == 2
    assert set(a_b.index) == {"line_0", "line_2"}
    # the series merge of m-a (x=1) and m-b (x=3), and the untouched direct Line
    np.testing.assert_allclose(sorted(a_b.x), [2.0, 4.0])
    # no bus in the K4 core dropped below degree 3, so nothing else moved
    assert set(reduced.buses.index) == {"a", "b", "c", "d"}


def test_degree_two_bus_with_both_lines_to_one_neighbour_folds_in_as_a_stub():
    """A double stub has no series merge to make -- it folds in whole.

    Reaching the degree-2 branch with both Lines on one neighbour, a series merge
    would produce a self-loop and the Kron split would clone every one-port asset
    onto its single target.
    """
    n = _network_with_degree_two_bus()
    n.lines.loc["line_1", "bus1"] = "a"  # both of m's Lines now lead to a
    n.add("Generator", "gen_m", bus="m", carrier="onwind", p_nom=90.0, p_nom_max=90.0)
    n.add("Load", "load_m", bus="m", p_set=25.0)

    reduced, busmap = reduce_low_degree_buses(n)

    assert "m" not in reduced.buses.index
    assert busmap.at["m"] == "a"
    assert reduced.buses.at["a", "Pd"] == 80.0  # the whole weight, not a Kron share
    assert {"line_0", "line_1"}.isdisjoint(reduced.lines.index)
    assert len(reduced.lines) == 6
    # moved wholesale rather than split into clones
    assert list(reduced.generators.index) == ["gen_m"]
    assert reduced.generators.at["gen_m", "bus"] == "a"
    np.testing.assert_allclose(reduced.generators.at["gen_m", "p_nom"], 90.0)
    assert list(reduced.loads.index) == ["load_m"]
    assert reduced.loads.at["load_m", "bus"] == "a"
    np.testing.assert_allclose(reduced.loads.at["load_m", "p_set"], 25.0)


def _grid_network(rows=3, columns=6, split_column=3):
    """A rectangular mesh split into two ReEDS zones down the middle."""
    n = pypsa.Network()
    for column in range(columns):
        for row in range(rows):
            name = f"b{column}_{row}"
            n.add("Bus", name, x=-100.0 + 0.1 * column, y=40.0 + 0.1 * row)
            n.buses.loc[name, "reeds_zone"] = (
                "west" if column < split_column else "east"
            )
    index = 0
    for column in range(columns):
        for row in range(rows):
            for step_column, step_row in ((1, 0), (0, 1)):
                other = f"b{column + step_column}_{row + step_row}"
                if other in n.buses.index:
                    n.add(
                        "Line",
                        f"l{index}",
                        bus0=f"b{column}_{row}",
                        bus1=other,
                        x=1.0,
                        r=0.0,
                        s_nom=100.0,
                    )
                    index += 1
    return n


def _clusters_are_connected(n, busmap):
    """True when every cluster induces a connected subgraph of the line graph."""
    for cluster, members in busmap.groupby(busmap).groups.items():
        members = set(members)
        edges = n.lines[n.lines.bus0.isin(members) & n.lines.bus1.isin(members)]
        reached, frontier = set(), [next(iter(members))]
        while frontier:
            bus = frontier.pop()
            if bus in reached:
                continue
            reached.add(bus)
            frontier += list(edges.loc[edges.bus0 == bus, "bus1"])
            frontier += list(edges.loc[edges.bus1 == bus, "bus0"])
        if reached != members:
            return False
    return True


def test_effective_reactance_embedding_recovers_series_reactance_distance():
    """Rademacher Laplacian probes estimate the exact L+ distance on a path."""
    n = pypsa.Network()
    for bus in ("a", "b", "c", "d"):
        n.add("Bus", bus)
    for name, bus0, bus1, x in (
        ("ab", "a", "b", 1.0),
        ("bc", "b", "c", 2.0),
        ("cd", "c", "d", 3.0),
    ):
        n.add("Line", name, bus0=bus0, bus1=bus1, x=x, r=0.0, s_nom=100.0)

    embedding, _, _, _ = _effective_reactance_embedding(n, n_probes=4096, seed=123)
    estimated = ((embedding[:, None, :] - embedding[None, :, :]) ** 2).mean(axis=-1)
    # In a series network, effective reactance is the sum of line reactances
    # along the unique path between buses.
    exact = np.array(
        [[0.0, 1.0, 3.0, 6.0], [1.0, 0.0, 2.0, 5.0],
         [3.0, 2.0, 0.0, 3.0], [6.0, 5.0, 3.0, 0.0]],
    )
    upper = np.triu_indices_from(exact, k=1)
    np.testing.assert_allclose(estimated[upper], exact[upper], rtol=0.03)


def test_effective_reactance_embedding_matches_spielman_srivastava_form():
    """The implementation is Q W**0.5 B L+ up to transpose and scaling."""
    n = pypsa.Network()
    for bus in ("a", "b", "c"):
        n.add("Bus", bus)
    n.add("Line", "ab", bus0="a", bus1="b", x=2.0, r=0.0, s_nom=100.0)
    n.add("Line", "bc", bus0="b", bus1="c", x=3.0, r=0.0, s_nom=100.0)
    n_probes, seed = 17, 11

    embedding, _, _, _ = _effective_reactance_embedding(n, n_probes, seed)
    incidence = np.array([[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0]])
    conductance = np.array([1 / 2.0, 1 / 3.0])
    laplacian = (incidence * conductance) @ incidence.T
    signs = 2.0 * np.random.default_rng(seed).integers(
        0,
        2,
        size=(2, n_probes),
        dtype=np.int8,
    ) - 1.0
    # Paper notation uses B = incidence.T and Q = signs.T / sqrt(n_probes).
    paper_embedding = (
        (signs.T / np.sqrt(n_probes) * np.sqrt(conductance))
        @ incidence.T
        @ np.linalg.pinv(laplacian)
    )
    expected = ((paper_embedding[:, :, None] - paper_embedding[:, None, :]) ** 2).sum(axis=0)
    actual = ((embedding[:, None, :] - embedding[None, :, :]) ** 2).mean(axis=-1)
    np.testing.assert_allclose(actual, expected)


def test_earth_centered_coordinates_reproduce_spherical_chord_lengths():
    buses = pd.DataFrame({"x": [0.0, 90.0, 180.0], "y": [0.0, 0.0, 0.0]})
    coordinates = _earth_centered_km(buses)

    np.testing.assert_allclose(np.linalg.norm(coordinates, axis=1), EARTH_RADIUS_KM)
    np.testing.assert_allclose(
        np.linalg.norm(coordinates[0] - coordinates[1]),
        np.sqrt(2) * EARTH_RADIUS_KM,
    )
    np.testing.assert_allclose(
        np.linalg.norm(coordinates[0] - coordinates[2]),
        2 * EARTH_RADIUS_KM,
    )


def test_target_bus_count_hits_the_requested_cluster_count_exactly():
    n = _grid_network()
    for target in (2, 3, 5, 9, 17):
        busmap = busmap_by_target_bus_count(
            n, target, topological_boundary="reeds_zone",
        )
        assert busmap.nunique() == target


def test_target_bus_count_never_merges_across_a_region_boundary():
    n = _grid_network()
    busmap = busmap_by_target_bus_count(n, 4, topological_boundary="reeds_zone")
    zones = pd.DataFrame({"cluster": busmap, "zone": n.buses.reeds_zone})
    assert zones.groupby("cluster").zone.nunique().max() == 1


def test_target_bus_count_never_merges_across_an_island():
    """Two disconnected meshes and no region column: islands must stay apart."""
    n = _grid_network(rows=2, columns=4)
    severed = n.lines[
        n.lines.bus0.str.startswith("b1_") & n.lines.bus1.str.startswith("b2_")
    ].index
    n.lines = n.lines.drop(index=severed)
    # Really no region column: otherwise the unconditional ReEDS-zone guard cuts
    # the zone boundary too, and this fixture's zone split lands mid-island.
    n.buses = n.buses.drop(columns=["reeds_zone"])
    busmap = busmap_by_target_bus_count(n, 2)
    left = {bus for bus in n.buses.index if int(bus[1]) < 2}
    right = set(n.buses.index) - left
    assert len({busmap[bus] for bus in left}) == 1
    assert not {busmap[bus] for bus in left} & {busmap[bus] for bus in right}


def test_target_bus_count_clusters_are_connected_subgraphs():
    n = _grid_network()
    for target in (3, 6, 11):
        busmap = busmap_by_target_bus_count(
            n, target, topological_boundary="reeds_zone",
        )
        assert _clusters_are_connected(n, busmap)


def test_target_bus_count_rejects_a_target_below_the_component_count():
    n = _grid_network()
    with pytest.raises(ValueError, match="connected subgraphs"):
        busmap_by_target_bus_count(n, 1, topological_boundary="reeds_zone")


def test_target_bus_count_cuts_are_nested_across_targets():
    """A coarser cut must be a coarsening of a finer one: they share a tree."""
    n = _grid_network()
    fine = busmap_by_target_bus_count(n, 10, topological_boundary="reeds_zone")
    coarse = busmap_by_target_bus_count(n, 5, topological_boundary="reeds_zone")
    assert pd.DataFrame({"fine": fine, "coarse": coarse}).groupby(
        "fine",
    ).coarse.nunique().max() == 1


def test_target_bus_count_is_deterministic():
    n = _grid_network()
    first = busmap_by_target_bus_count(n, 7, seed=5, topological_boundary="reeds_zone")
    second = busmap_by_target_bus_count(n, 7, seed=5, topological_boundary="reeds_zone")
    pd.testing.assert_series_equal(first, second)


def test_lambda_electrical_shifts_the_merge_order_towards_electrical_proximity():
    """A geographically short but electrically weak tie loses ground as lambda grows.

    ``a-b`` is a long line with tiny reactance; ``b-c`` is short but highly
    reactive. At low lambda geography decides and ``b`` merges with ``c``; at
    high lambda the effective reactance decides and ``b`` merges with ``a``.
    """
    n = pypsa.Network()
    for name, longitude in (("a", -100.20), ("b", -100.00), ("c", -99.99), ("d", -99.00)):
        n.add("Bus", name, x=longitude, y=40.0)
    n.add("Line", "ab", bus0="a", bus1="b", x=0.01, r=0.0, s_nom=100.0)
    n.add("Line", "bc", bus0="b", bus1="c", x=50.0, r=0.0, s_nom=100.0)
    n.add("Line", "cd", bus0="c", bus1="d", x=50.0, r=0.0, s_nom=100.0)

    geographic = busmap_by_target_bus_count(n, 3, lambda_electrical=1e-3)
    electrical = busmap_by_target_bus_count(n, 3, lambda_electrical=1e3)
    assert geographic["b"] == geographic["c"]
    assert electrical["a"] == electrical["b"]


def test_target_bus_count_busmap_feeds_the_standard_clustering_wrapper():
    n = _grid_network()
    busmap = busmap_by_target_bus_count(n, 6, topological_boundary="reeds_zone")
    clustered = clustering_from_busmap(n, busmap, line_length_factor=1.0).network
    assert len(clustered.buses) == 6
    # Region membership has to survive onto the clustered buses: cluster_regions
    # and the downstream policy constraints key off it.
    assert set(clustered.buses.reeds_zone) == {"west", "east"}


def test_identity_busmap_leaves_the_network_untouched():
    """The neutral element a disabled stage contributes to the busmap chain."""
    n = _grid_network(rows=2, columns=3)
    busmap = identity_busmap(n)
    assert (busmap == n.buses.index).all()
    clustered = clustering_from_busmap(n, busmap, line_length_factor=1.0).network
    assert list(clustered.buses.index) == list(n.buses.index)
    assert len(clustered.lines) == len(n.lines)


def test_identity_busmap_composes_as_a_neutral_element():
    """A disabled later stage must not move any bus in the composed chain.

    Mirrors how ``main`` composes the three stage busmaps: each one is indexed
    by the buses that existed when that stage ran, so a disabled stage takes
    its identity map from the network as it stands at that point.
    """
    n = _grid_network(rows=2, columns=3)
    stage = busmap_by_target_bus_count(n, 3, topological_boundary="reeds_zone")
    clustered = clustering_from_busmap(n, stage, line_length_factor=1.0).network
    neutral = identity_busmap(clustered)
    composed = reduce(lambda left, right: left.map(right), [neutral], stage)
    pd.testing.assert_series_equal(composed, stage, check_names=False)


def _network_with_lines(edges, buses, zones=None):
    """Build a Line-only network from ``(bus0, bus1, length_km, x)`` tuples."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    for bus in buses:
        n.add("Bus", bus)
    n.lines = pd.DataFrame(
        {
            "bus0": [edge[0] for edge in edges],
            "bus1": [edge[1] for edge in edges],
            "length": [float(edge[2]) for edge in edges],
            "x": [float(edge[3]) for edge in edges],
            "r": 0.1,
            "s_nom": 100.0,
            "s_nom_min": 0.0,
            "s_nom_max": 200.0,
            "capital_cost": [10.0 * edge[2] for edge in edges],
            "type": "",
            "carrier": "AC",
        },
        index=[f"line_{i}" for i in range(len(edges))],
    )
    n.buses["Pd"] = 0.0
    if zones is not None:
        n.buses["reeds_zone"] = pd.Series(zones)
    return n


def _short_tie_in_a_k4(zones=None):
    """A metre-scale tie inside a K4 core, so both endpoints have degree three."""
    edges = [
        ("a", "b", 0.05, 0.001),
        ("a", "c", 40.0, 0.4),
        ("a", "d", 30.0, 0.3),
        ("b", "c", 35.0, 0.35),
        ("b", "d", 45.0, 0.45),
        ("c", "d", 50.0, 0.5),
    ]
    return _network_with_lines(edges, ("a", "b", "c", "d"), zones)


def test_short_branch_contraction_merges_the_two_substations():
    """The tie disappears, its buses become one, and demand weight adds."""
    n = _network_with_lines(
        [("a", "b", 0.05, 0.001), ("a", "c", 40.0, 0.4), ("b", "d", 30.0, 0.3), ("c", "d", 50.0, 0.5)],
        ("a", "b", "c", "d"),
    )
    n.buses.loc["a", "Pd"] = 30.0
    n.buses.loc["b", "Pd"] = 70.0

    n, busmap = contract_short_branches(n)

    # The larger demand weight keeps its own site, so "b" survives and "a" folds in.
    assert set(n.buses.index) == {"b", "c", "d"}
    assert busmap["a"] == "b"
    assert busmap["c"] == "c"
    assert n.buses.at["b", "Pd"] == pytest.approx(100.0)
    # The contracted branch is gone and the two it used to separate now meet at "b".
    assert len(n.lines) == 3
    assert 0.05 not in set(n.lines.length)
    assert set(zip(n.lines.bus0, n.lines.bus1)) == {("b", "c"), ("b", "d"), ("c", "d")}


def test_short_branch_contraction_reaches_endpoints_the_low_degree_pass_cannot():
    """A short tie between two degree-three buses: a no-op for the degree-1/2
    rule at any length, which is why this is a separate pass."""
    untouched, low_degree_busmap = reduce_low_degree_buses(_short_tie_in_a_k4())
    assert len(untouched.buses) == 4
    assert untouched.lines.length.min() == pytest.approx(0.05)
    assert (low_degree_busmap == low_degree_busmap.index).all()

    contracted, busmap = contract_short_branches(_short_tie_in_a_k4())
    assert len(contracted.buses) == 3
    assert contracted.lines.length.min() > SHORT_BRANCH_LENGTH_KM
    assert busmap["b"] == "a"


def test_short_branch_contraction_leaves_the_parallel_lines_it_creates():
    """Collapsing a-b in a K4 doubles up a-c/b-c and a-d/b-d, and both survive.

    Pooling them would rate each pair at ``min(x_i s_nom_i) / x_parallel`` and
    write off the headroom an SSSC could unlock by rebalancing the split.
    """
    n, _ = contract_short_branches(_short_tie_in_a_k4())

    pairs = [tuple(sorted(pair)) for pair in zip(n.lines.bus0, n.lines.bus1)]
    assert len(pairs) == 5
    assert sorted(pairs) == [("a", "c"), ("a", "c"), ("a", "d"), ("a", "d"), ("c", "d")]
    assert (n.lines.bus0 != n.lines.bus1).all()
    # Each keeps its own reactance and rating, so an SSSC can be sited on either.
    np.testing.assert_allclose(sorted(n.lines.loc[[i for i, p in zip(n.lines.index, pairs)
                                                   if p == ("a", "c")], "x"]), [0.35, 0.4])
    assert (n.lines.s_nom == 100.0).all()


def test_short_branch_contraction_collapses_a_chain_of_short_lines_at_once():
    """Short Lines sharing endpoints form one group, not a chain of pair merges."""
    n = _network_with_lines(
        [
            ("a", "b", 0.02, 0.001),
            ("b", "c", 0.03, 0.001),
            ("a", "d", 40.0, 0.4),
            ("c", "e", 30.0, 0.3),
            ("d", "e", 50.0, 0.5),
        ],
        ("a", "b", "c", "d", "e"),
    )
    n.buses.loc["b", "Pd"] = 5.0

    n, busmap = contract_short_branches(n)

    assert len(n.buses) == 3
    assert busmap["a"] == busmap["b"] == busmap["c"] == "b"
    assert n.buses.at["b", "Pd"] == pytest.approx(5.0)
    assert len(n.lines) == 3


def test_short_branch_contraction_refuses_to_cross_the_protected_zone():
    """A tie between two zones stays, so no zone loses load or generation."""
    n = _short_tie_in_a_k4(zones={"a": "west", "b": "east", "c": "east", "d": "east"})

    n, busmap = contract_short_branches(n)

    assert len(n.buses) == 4
    assert len(n.lines) == 6
    assert n.lines.length.min() == pytest.approx(0.05)
    assert (busmap == busmap.index).all()


def test_short_branch_contraction_is_a_no_op_without_short_lines():
    """Every real corridor is far above the threshold; the pass must not fire."""
    n = _network_with_lines(
        [("a", "b", 12.0, 0.1), ("b", "c", 40.0, 0.4), ("a", "c", 50.0, 0.5)],
        ("a", "b", "c"),
    )

    reduced, busmap = contract_short_branches(n)

    assert len(reduced.buses) == 3
    assert len(reduced.lines) == 3
    assert (busmap == busmap.index).all()


def test_short_branch_contraction_moves_one_port_assets_and_links_wholesale():
    """Merging two sites moves their assets across intact, not Kron-split."""
    n = _network_with_lines(
        [("a", "b", 0.05, 0.001), ("a", "c", 40.0, 0.4), ("b", "d", 30.0, 0.3), ("c", "d", 50.0, 0.5)],
        ("a", "b", "c", "d"),
    )
    n.buses.loc["b", "Pd"] = 70.0
    n.add("Generator", "wind_a", bus="a", carrier="onwind", p_nom=250.0, p_nom_max=250.0)
    n.add("Load", "load_a", bus="a", p_set=40.0)
    n.add("Bus", "dc")
    n.add("Link", "dc_a", bus0="a", bus1="dc", p_nom=500.0)

    n, _ = contract_short_branches(n)

    assert n.generators.at["wind_a", "bus"] == "b"
    assert n.generators.at["wind_a", "p_nom"] == pytest.approx(250.0)
    assert n.loads.at["load_a", "bus"] == "b"
    assert n.loads.at["load_a", "p_set"] == pytest.approx(40.0)
    assert n.links.at["dc_a", "bus0"] == "b"


def test_short_branch_contraction_busmap_feeds_the_standard_clustering_wrapper():
    """The busmap has to be a valid stage map over the pre-contraction index."""
    original = _short_tie_in_a_k4()
    _, busmap = contract_short_branches(_short_tie_in_a_k4())

    assert list(busmap.index) == list(original.buses.index)
    assert set(busmap) <= set(original.buses.index)
    clustered = clustering_from_busmap(original, busmap, line_length_factor=1.0).network
    assert len(clustered.buses) == 3
