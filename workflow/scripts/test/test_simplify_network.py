import logging
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
    aggregate_to_substations,
    busmap_by_target_bus_count,
    clustering_from_busmap,
    contract_short_branches,
    planning_horizon,
    retired_by,
    identity_busmap,
    parse_low_degree_reduction,
    reduce_low_degree_buses,
    blend_pooled_line_ratings,
    merged_series_rating,
    parse_line_rating,
    target_count_aggregation_strategies,
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


def _split_plant_network():
    """One plant on the degree-two bus, and one of the same carrier at a target."""
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
    return n


def test_low_degree_bus_splits_capacity_and_averages_generator_attributes():
    """Each Kron half pools into the same-carrier fleet at its target."""
    n = _split_plant_network()

    reduced, _ = reduce_low_degree_buses(n)

    # x(m-a)=1 and x(m-b)=3, so the Kron split is 3/4 to a and 1/4 to b.
    assert reduced.buses.at["a", "Pd"] == 60.0
    assert reduced.buses.at["b", "Pd"] == 20.0

    at_a = reduced.generators.query("bus == 'a'")
    at_b = reduced.generators.query("bus == 'b'")
    assert len(at_a) == len(at_b) == 1
    at_a, at_b = at_a.iloc[0], at_b.iloc[0]
    assert at_a.p_nom == 100.0
    assert at_a.p_nom_max == 175.0
    assert at_b.p_nom == 25.0
    assert at_b.p_nom_max == 50.0
    np.testing.assert_allclose(at_a.efficiency, 0.85)
    np.testing.assert_allclose(at_a.capital_cost, 5.0)
    np.testing.assert_allclose(at_a.marginal_cost, 3.0)
    np.testing.assert_allclose(reduced.generators_t.p_max_pu[at_a.name], [0.5, 0.7])
    np.testing.assert_allclose(reduced.generators_t.p_max_pu[at_b.name], [0.4, 0.8])


def test_stub_relocation_still_pools_the_generators_it_lands_on_one_bus():
    """A stub moves its plant intact, and it pools at the one bus, capacity-weighted."""
    n = _split_plant_network()
    n.lines.loc["line_1", "bus1"] = "a"  # both of m's Lines lead to a: a stub
    n.generators.loc["middle", "p_nom"] = 75.0
    n.generators.loc["middle", "p_nom_max"] = 150.0

    reduced, _ = reduce_low_degree_buses(n)

    at_a = reduced.generators.query("bus == 'a'")
    assert len(at_a) == 1
    merged = at_a.iloc[0]
    assert merged.p_nom == 100.0
    assert merged.p_nom_max == 175.0
    np.testing.assert_allclose(merged.efficiency, 0.85)
    np.testing.assert_allclose(merged.capital_cost, 5.0)
    np.testing.assert_allclose(merged.marginal_cost, 3.0)
    np.testing.assert_allclose(reduced.generators_t.p_max_pu[merged.name], [0.5, 0.7])


def test_low_degree_bus_splits_storage_units_by_the_kron_factor():
    """A battery is split exactly as a plant is, energy and stored charge included."""
    n = _network_with_degree_two_bus()
    n.add(
        "StorageUnit",
        "batt",
        bus="m",
        carrier="battery",
        p_nom=100.0,
        max_hours=4.0,
        state_of_charge_initial=200.0,
    )

    reduced, _ = reduce_low_degree_buses(n)

    units = reduced.storage_units
    assert set(units.index) == {"batt", "batt__split_b"}
    np.testing.assert_allclose(units.at["batt", "p_nom"], 75.0)
    np.testing.assert_allclose(units.at["batt__split_b", "p_nom"], 25.0)
    # `max_hours` is intensive, so the energy capacity follows the power split,
    # and the charge already in the reservoir is split with it.
    np.testing.assert_allclose(units.max_hours, 4.0)
    np.testing.assert_allclose(units.at["batt", "state_of_charge_initial"], 150.0)
    np.testing.assert_allclose(units.at["batt__split_b", "state_of_charge_initial"], 50.0)


def test_cascading_elimination_compounds_the_kron_split():
    """A chain of degree-two buses splits the same plant once per pass.

    The factors compound, so the plant's capacity still lands in the exact Kron
    ratio however deep the chain -- and pieces that the chain brings back onto one
    bus pool again. Both elimination orders give the same answer, as Kron requires.
    """
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    for bus in ("m1", "m2", "a", "b", "c", "d"):
        n.add("Bus", bus)
    edges = [
        ("m1", "a", 1.0),
        ("m1", "m2", 1.0),
        ("m2", "b", 2.0),
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
    n.add("Generator", "chained", bus="m1", carrier="solar", p_nom=100.0)

    reduced, _ = reduce_low_degree_buses(n)

    assert {"m1", "m2"}.isdisjoint(reduced.buses.index)
    clones = reduced.generators
    assert len(clones) == 2
    by_bus = clones.set_index("bus")
    np.testing.assert_allclose(by_bus.at["a", "p_nom"], 75.0)
    np.testing.assert_allclose(by_bus.at["b", "p_nom"], 25.0)


@pytest.mark.parametrize(
    "kwargs, s_nom",
    [
        ({}, (7.0 * 100.0 + 21.0 * 60.0) / 28.0),  # default: cost-weighted mean
        ({"series_rating": "min"}, 60.0),  # the narrower section
    ],
)
def test_series_merge_sums_length_and_capital_cost(kwargs, s_nom):
    """A merged corridor costs what both of its segments cost; its rating follows
    ``series_rating``."""
    n = _network_with_degree_two_bus()
    # $/MW proportional to length, as `update_transmission_costs` builds it.
    n.lines["capital_cost"] = n.lines.length * 7.0
    n.lines.loc["line_1", "length"] = 3.0
    n.lines.loc["line_1", "capital_cost"] = 21.0
    n.lines.loc["line_1", "s_nom"] = 60.0

    reduced, _ = reduce_low_degree_buses(n, **kwargs)

    merged = reduced.lines.loc["line_0"]
    assert {merged.bus0, merged.bus1} == {"a", "b"}
    np.testing.assert_allclose(merged.x, 4.0)  # 1 + 3
    np.testing.assert_allclose(merged.r, 0.2)  # 0.1 + 0.1
    np.testing.assert_allclose(merged.s_nom, s_nom)
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


def _bundle_between_two_clusters():
    """Two circuits A0-B0 and A1-B1 with unequal x*s, plus internal Lines."""
    n = _network_from_edges(
        [("a0", "b0", 1.0), ("a1", "b1", 4.0), ("a0", "a1", 1.0), ("b0", "b1", 1.0)],
        ["a0", "a1", "b0", "b1"],
    )
    n.lines.loc["line_1", ["s_nom", "s_nom_max"]] = [300.0, 400.0]
    busmap = pd.Series({"a0": "A", "a1": "A", "b0": "B", "b1": "B"})
    return n, busmap


@pytest.mark.parametrize(
    "line_rating, s_nom, s_nom_max",
    [
        # b = (1, 1/4); min(s/b) = min(100, 1200) -> 1.25 * 100
        ("parallel_bottleneck", 125.0, 250.0),
        ("sum", 400.0, 600.0),
    ],
)
def test_target_count_line_rating_sets_the_pooled_bundle_rating(line_rating, s_nom, s_nom_max):
    n, busmap = _bundle_between_two_clusters()
    strategies = target_count_aggregation_strategies({}, line_rating)
    lines = clustering_from_busmap(
        n, busmap, line_length_factor=1.0, aggregation_strategies=strategies,
    ).network.lines
    assert len(lines) == 1
    np.testing.assert_allclose(lines.s_nom.iloc[0], s_nom)
    np.testing.assert_allclose(lines.s_nom_max.iloc[0], s_nom_max)
    # The reactance is the parallel combination either way: 1 / (1 + 1/4).
    np.testing.assert_allclose(lines.x.iloc[0], 0.8)


def _unequal_series_chain():
    """a - m - b with segments rated 100 and 300 and costed 1 and 3."""
    n = _network_with_degree_two_bus()
    n.lines["capital_cost"] = 1.0
    n.lines.loc["line_0", ["s_nom", "capital_cost"]] = [100.0, 1.0]  # m-a
    n.lines.loc["line_1", ["s_nom", "capital_cost"]] = [300.0, 3.0]  # m-b
    n.lines["s_nom_max"] = np.inf
    return n


@pytest.mark.parametrize(
    "rule, s_nom",
    [("min", 100.0), ("cost_weighted", (1 * 100 + 3 * 300) / 4)],
)
def test_series_rating_sets_the_merged_corridor_rating(rule, s_nom):
    reduced, _ = reduce_low_degree_buses(_unequal_series_chain(), series_rating=rule)
    # The merged corridor inherits the first name in sort order; the K4 core's
    # own a-b Line (line_2) stands beside it untouched.
    merged = reduced.lines.loc["line_0"]
    assert {merged.bus0, merged.bus1} == {"a", "b"}
    np.testing.assert_allclose(merged.s_nom, s_nom)
    # Summed either way: widening the corridor past both segments costs both.
    np.testing.assert_allclose(merged.capital_cost, 4.0)
    assert np.isinf(merged.s_nom_max)
    np.testing.assert_allclose(reduced.lines.at["line_2", "s_nom"], 100.0)


def test_cost_weighted_series_rating_composes_across_passes():
    """Merging a merged segment again gives the cost-weighted mean of all three."""
    seg = lambda s, c: pd.Series({"s_nom": s, "capital_cost": c, "length": 1.0})
    first = merged_series_rating(seg(100.0, 1.0), seg(300.0, 3.0), "cost_weighted")
    second = merged_series_rating(seg(first, 4.0), seg(50.0, 2.0), "cost_weighted")
    np.testing.assert_allclose(second, (100 * 1 + 300 * 3 + 50 * 2) / 6)
    with pytest.raises(ValueError):
        merged_series_rating(seg(1.0, 1.0), seg(1.0, 1.0), "mean")


def test_unpooled_line_rating_keeps_every_inter_cluster_circuit():
    n, busmap = _bundle_between_two_clusters()
    clustered = clustering_from_busmap(
        n, busmap, line_length_factor=1.0,
        aggregation_strategies=target_count_aggregation_strategies({}, "unpooled"),
        pool_parallel_lines=False,
    ).network
    lines = clustered.lines.sort_values("x")
    # Both A-B circuits survive as they were; the two internal Lines are gone.
    assert len(lines) == 2
    assert set(map(frozenset, zip(lines.bus0, lines.bus1))) == {frozenset({"A", "B"})}
    np.testing.assert_allclose(lines.x, [1.0, 4.0])
    np.testing.assert_allclose(lines.s_nom, [100.0, 300.0])
    assert "_unpooled_line" not in clustered.lines.columns
    assert "_unpooled_line" not in n.lines.columns


def test_fractional_line_rating_blends_bottleneck_toward_sum():
    n, busmap = _bundle_between_two_clusters()
    n.lines.loc["line_2", "s_nom_max"] = np.inf  # an unbounded internal Line
    clustering = clustering_from_busmap(
        n, busmap, line_length_factor=1.0,
        aggregation_strategies=target_count_aggregation_strategies({}, 0.5),
    )
    blend_pooled_line_ratings(clustering.network, n.lines, clustering.linemap, 0.5)
    lines = clustering.network.lines
    assert len(lines) == 1
    # Halfway between the bottleneck (125, 250) and the sum (400, 600).
    np.testing.assert_allclose(lines.s_nom.iloc[0], 262.5)
    np.testing.assert_allclose(lines.s_nom_max.iloc[0], 425.0)


@pytest.mark.parametrize(
    "value, parsed",
    [(0, "parallel_bottleneck"), (1.0, "sum"), ("0.5", 0.5), (0.25, 0.25), ("sum", "sum"),
     ("unpooled", "unpooled")],
)
def test_parse_line_rating(value, parsed):
    assert parse_line_rating(value) == parsed


@pytest.mark.parametrize("value", [1.5, -0.1, "max", None])
def test_parse_line_rating_rejects_anything_else(value):
    with pytest.raises(ValueError):
        parse_line_rating(value)


def test_target_count_line_rating_leaves_the_callers_strategies_alone():
    base = {"lines": {"s_nom": "parallel_bottleneck"}, "generators": {"lifetime": "mean"}}
    strategies = target_count_aggregation_strategies(base, "sum")
    assert strategies["lines"]["s_nom"] == "sum"
    assert base["lines"]["s_nom"] == "parallel_bottleneck"
    assert strategies["generators"] == {"lifetime": "mean"}
    with pytest.raises(ValueError):
        target_count_aggregation_strategies(base, "max")


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


def _three_bus_line_with_investment_periods(periods=(2030, 2040, 2050)):
    """Three buses in a row, over the given planning horizons."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.investment_periods = list(periods)
    for bus in ("b0", "b1", "b2"):
        n.add("Bus", bus, v_nom=230.0)
    n.add("Line", "l0", bus0="b0", bus1="b1", x=0.1, r=0.01, s_nom=100.0, length=1.0)
    n.add("Line", "l1", bus0="b1", bus1="b2", x=0.1, r=0.01, s_nom=100.0, length=1.0)
    n.add("Carrier", "coal")
    return n


def _active_capacity(n):
    """Installed capacity pypsa treats as online, per investment period."""
    return {
        int(period): float(n.generators.p_nom[n.get_active_assets("Generator", period)].sum())
        for period in n.investment_periods
    }


def test_retirement_test_follows_pypsas_strict_bound():
    """pypsa's rule is `build_year <= period < build_year + lifetime`.

    A unit whose lifetime ends exactly in 2050 is already gone in 2050, so it
    must not share an aggregation group with one that survives the year.
    """
    n = _three_bus_line_with_investment_periods(periods=(2050,))
    n.add("Generator", "retires_in_2050", bus="b0", carrier="coal", build_year=1980, lifetime=70)
    n.add("Generator", "survives_2050", bus="b1", carrier="coal", build_year=1981, lifetime=70)

    retired = retired_by(n, planning_horizon(n))

    assert planning_horizon(n) == 2050
    assert retired["retires_in_2050"]
    assert not retired["survives_2050"]
    assert not n.get_active_assets("Generator", 2050)["retires_in_2050"]
    assert n.get_active_assets("Generator", 2050)["survives_2050"]


def test_clustering_keeps_retiring_and_surviving_plant_in_separate_generators():
    """Pooling the two would keep retired plant online; splitting them does not."""
    n = _three_bus_line_with_investment_periods()
    n.add("Generator", "g_old", bus="b0", carrier="coal", p_nom=100.0, build_year=1975, lifetime=70)
    n.add("Generator", "g_mid", bus="b1", carrier="coal", p_nom=100.0, build_year=1995, lifetime=70)
    n.add("Generator", "g_new", bus="b2", carrier="coal", p_nom=300.0, build_year=2005, lifetime=70)
    before = _active_capacity(n)

    clustered = clustering_from_busmap(
        n,
        pd.Series({"b0": "B", "b1": "B", "b2": "B"}),
        line_length_factor=1.0,
        aggregate_carriers=set(n.generators.carrier),
    ).network

    assert before == {2030: 500.0, 2040: 500.0, 2050: 400.0}
    assert _active_capacity(clustered) == before
    assert clustered.generators.p_nom.sum() == pytest.approx(n.generators.p_nom.sum())
    # The tag is stripped from `carrier`, which every cost and policy lookup
    # keys on, and kept in the name, which is all that tells the groups apart.
    assert set(clustered.generators.carrier) == {"coal"}
    assert len(clustered.generators) == 2
    assert clustered.generators.index.is_unique


def test_substation_aggregation_splits_retired_plant_too():
    n = _three_bus_line_with_investment_periods()
    n.buses["sub_id"] = [0, 0, 0]
    for column in ("interconnect", "state", "country", "county", "balancing_area", "reeds_zone", "reeds_ba", "reeds_state"):
        n.buses[column] = "x"
    n.add("Generator", "g_old", bus="b0", carrier="coal", p_nom=100.0, build_year=1975, lifetime=70)
    n.add("Generator", "g_new", bus="b1", carrier="coal", p_nom=100.0, build_year=1995, lifetime=70)

    clustered, _ = aggregate_to_substations(
        n,
        n.buses.sub_id.astype(int).astype(str),
        "reeds_zone",
        1.0,
        {},
    )

    assert _active_capacity(clustered) == {2030: 200.0, 2040: 200.0, 2050: 100.0}
    assert set(clustered.generators.carrier) == {"coal"}


def test_colocated_merge_keeps_retired_plant_apart_and_averages_the_dates():
    """A relocation that lands both kinds on one bus must not pool them."""
    n = _network_with_degree_two_bus()
    n.lines.loc["line_1", "bus1"] = "a"  # a stub, so m's plant moves intact
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.investment_periods = [2030, 2040, 2050]
    n.add("Generator", "old_m", bus="m", carrier="coal", p_nom=100.0, build_year=1975, lifetime=70)
    n.add("Generator", "mid_m", bus="m", carrier="coal", p_nom=100.0, build_year=1995, lifetime=70)
    n.add("Generator", "new_a", bus="a", carrier="coal", p_nom=300.0, build_year=2005, lifetime=70)

    reduced, _ = reduce_low_degree_buses(n)

    # The 1975 block retires in 2045, so it stays out of the group the 1995 and
    # 2005 units merge into, even though all three now sit on one bus.
    at_a = reduced.generators.query("bus == 'a'")
    assert len(at_a) == 2
    retiring = at_a[at_a.build_year < 1990]
    merged = at_a[at_a.build_year > 1990]
    assert float(retiring.p_nom.iloc[0]) == pytest.approx(100.0)
    assert float(merged.p_nom.iloc[0]) == pytest.approx(400.0)
    # 100 MW of 1995 plant and 300 MW of 2005 plant.
    assert int(merged.build_year.iloc[0]) == 2002
    assert float(merged.lifetime.iloc[0]) == pytest.approx(70.0)

    assert _active_capacity(reduced) == {2030: 500.0, 2040: 500.0, 2050: 400.0}


def test_fractional_lifetimes_survive_the_integer_build_year_column():
    """`build_year` is an integer, so a mean retirement date can round down.

    Both units here outlive 2050 by a few months, and their mean build year
    falls mid-year; truncating it would retire the whole aggregate in 2050.
    """
    n = _three_bus_line_with_investment_periods()
    n.add("Generator", "g_a", bus="b0", carrier="coal", p_nom=100.0, build_year=1980, lifetime=70.1)
    n.add("Generator", "g_b", bus="b1", carrier="coal", p_nom=100.0, build_year=1981, lifetime=69.5)
    assert not retired_by(n, planning_horizon(n)).any()

    clustered = clustering_from_busmap(
        n,
        pd.Series({"b0": "B", "b1": "B", "b2": "B"}),
        line_length_factor=1.0,
        aggregate_carriers=set(n.generators.carrier),
    ).network

    merged = clustered.generators.iloc[0]
    assert merged.build_year + merged.lifetime > 2050
    assert _active_capacity(clustered) == {2030: 200.0, 2040: 200.0, 2050: 200.0}


def test_single_planning_year_yields_at_most_two_existing_generators_per_carrier():
    """The repository default is one planning year, so one boundary decides all.

    Plant whose lifetime has run out by then is active in no period at all and
    belongs in its own block; everything else survives the whole horizon and
    keeps the plain `<bus> <carrier>` name.
    """
    n = _three_bus_line_with_investment_periods(periods=(2050,))
    n.add("Generator", "spent", bus="b0", carrier="coal", p_nom=100.0, build_year=1975, lifetime=70)
    n.add("Generator", "just_spent", bus="b1", carrier="coal", p_nom=200.0, build_year=1980, lifetime=70)
    n.add("Generator", "alive", bus="b2", carrier="coal", p_nom=300.0, build_year=1981, lifetime=70)

    clustered = clustering_from_busmap(
        n,
        pd.Series({"b0": "B", "b1": "B", "b2": "B"}),
        line_length_factor=1.0,
        aggregate_carriers=set(n.generators.carrier),
    ).network

    # 1980 + 70 == 2050 is retirement, not survival: pypsa's bound is strict.
    assert set(clustered.generators.index) == {"B coal", "B coal_retired"}
    assert clustered.generators.at["B coal", "p_nom"] == pytest.approx(300.0)
    assert clustered.generators.at["B coal_retired", "p_nom"] == pytest.approx(300.0)
    assert set(clustered.generators.carrier) == {"coal"}
    assert _active_capacity(clustered) == {2050: 300.0}
    assert clustered.generators.p_nom.sum() == pytest.approx(n.generators.p_nom.sum())


# ---------------------------------------------------------------------------
# lossless vs degree1and2
#
# The two levels share every line of `reduce_low_degree_buses` but the test for
# which degree-2 buses may go, so what follows pins that test: which buses each
# level is allowed to take, that `lossless` reaches a fixed point on its own, and
# that neither of the two approximations `lossless` exists to avoid -- a series
# merge that writes off rating, and a Kron split of a dispatchable asset -- can
# happen under it.
# ---------------------------------------------------------------------------


def _k4_edges():
    """Six Lines over a, b, c, d: every bus at degree 3, so none is reducible."""
    return [
        ("a", "b", 2.0),
        ("a", "c", 2.0),
        ("a", "d", 2.0),
        ("b", "c", 2.0),
        ("b", "d", 2.0),
        ("c", "d", 2.0),
    ]


def _network_from_edges(edges, buses):
    n = pypsa.Network()
    for bus in buses:
        n.add("Bus", bus)
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
    return n


def _stub_chain_on_a_k4(depth=1):
    """A chain of ``depth`` buses hanging off `a` of a K4 core, each carrying 10 MW.

    Only the tip of the chain starts at degree 1; every bus behind it starts at
    degree 2, equally rated and empty, so under ``lossless`` it may go either as a
    series merge or as a stub once the bus in front of it is gone. Either way the
    whole chain ends up at `a`, which takes a job for the fixed-point loop.
    """
    chain = ["a"] + [f"s{i}" for i in range(1, depth + 1)]
    edges = _k4_edges() + [(chain[i], chain[i + 1], 1.0) for i in range(depth)]
    n = _network_from_edges(edges, ("a", "b", "c", "d", *chain[1:]))
    n.buses.loc[chain[1:], "Pd"] = 10.0
    return n


def _double_stub_network():
    """`m` reaches only `a`, but by two Lines rather than one."""
    n = _network_with_degree_two_bus()
    n.lines.loc["line_1", "bus1"] = "a"
    return n


def _mixed_low_degree_network():
    """One degree-1 stub and one degree-2 bus on the same core.

    `s` hangs off `c`, which has three Lines of its own, so removing `s` leaves the
    core untouched and the two levels differ by exactly the degree-2 bus `m`,
    whose two Lines are rated 100 and 150 MW -- a merge would write off 50.
    """
    n = _network_with_degree_two_bus()
    n.lines.loc["line_1", "s_nom"] = 150.0
    n.add("Bus", "s")
    n.buses.loc["s", "Pd"] = 40.0
    n.lines.loc["line_stub"] = n.lines.loc["line_0"].copy()
    n.lines.loc["line_stub", ["bus0", "bus1"]] = ["c", "s"]
    return n


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("degree1and2", "degree1and2"),
        ("lossless", "lossless"),
        ("none", None),
        # The booleans the key took before the split. `true` has to keep meaning
        # degree1and2 -- the only behaviour it ever selected -- or a config written
        # against it would silently export a different network.
        (True, "degree1and2"),
        (False, None),
        (None, None),
        ("true", "degree1and2"),
        ("false", None),
        # YAML reads an unquoted no/off as False; the quoted forms land alongside.
        ("no", None),
        ("off", None),
        ("yes", "degree1and2"),
        # Case and surrounding space are not part of the mode name.
        ("LOSSLESS", "lossless"),
        ("  degree1and2\t", "degree1and2"),
    ],
)
def test_parse_low_degree_reduction_maps_every_accepted_spelling(value, expected):
    assert parse_low_degree_reduction(value) == expected


@pytest.mark.parametrize(
    "value",
    ["degree1", "degree2", "degree_1", "degree 1", "1", 2, "", "maybe"],
)
def test_parse_low_degree_reduction_rejects_anything_else(value):
    """A typo must not fall back to a default -- nor may the retired ``degree1``,
    whose stubs-only behaviour ``lossless`` replaced.

    Every mode runs without error and produces a network, so a silent fallback
    would swap the exported topology with nothing in the log to say so.
    """
    with pytest.raises(ValueError, match="low_degree_reduction"):
        parse_low_degree_reduction(value)


@pytest.mark.parametrize("level", [1, 2, None, "none", "degree1", "LOSSLESS", " lossless"])
def test_reduce_low_degree_buses_rejects_an_unsupported_level(level):
    """Only the two exact level names; the lenient spellings are the config
    parser's job, and switching the pass off is the caller's."""
    with pytest.raises(ValueError, match="level"):
        reduce_low_degree_buses(_network_with_degree_two_bus(), level=level)


def test_lossless_clears_a_stub_chain_and_the_weight_all_reaches_its_root():
    """Two buses deep, so the fixed point has to run twice to reach the far one.

    Whether `s1` goes as a series merge (its Lines are equally rated and it holds
    nothing dispatchable) or as a stub once `s2` is gone, 10 + 10 ends up at `a`.
    """
    n = _stub_chain_on_a_k4(depth=2)

    reduced, busmap = reduce_low_degree_buses(n, level="lossless")

    assert set(reduced.buses.index) == {"a", "b", "c", "d"}
    assert busmap.at["s1"] == "a"
    assert busmap.at["s2"] == "a"
    assert reduced.buses.at["a", "Pd"] == pytest.approx(20.0)
    # The core is untouched: only the two chain Lines went.
    assert set(reduced.lines.index) == {f"line_{i}" for i in range(6)}


def test_lossless_splices_out_an_equally_rated_empty_degree_two_bus():
    """Both of `m`'s Lines are rated 100 MW and it holds only demand weight.

    The corridor keeps 100 MW, so nothing is written off; `m`'s 80 MW is a fixed
    injection whose Kron split -- 3/4 to `a` behind x=1, 1/4 to `b` behind x=3 --
    is exact. This is precisely what degree1and2 would do to the same bus.
    """
    lossless, lossless_busmap = reduce_low_degree_buses(_network_with_degree_two_bus(), level="lossless")
    both, both_busmap = reduce_low_degree_buses(_network_with_degree_two_bus(), level="degree1and2")

    assert "m" not in lossless.buses.index
    assert lossless_busmap.at["m"] == "a"
    assert lossless.buses.at["a", "Pd"] == pytest.approx(60.0)
    assert lossless.buses.at["b", "Pd"] == pytest.approx(20.0)
    merged = lossless.lines.loc["line_0"]
    assert {merged.bus0, merged.bus1} == {"a", "b"}
    np.testing.assert_allclose(merged.x, 4.0)
    np.testing.assert_allclose(merged.s_nom, 100.0)
    assert "line_1" not in lossless.lines.index

    pd.testing.assert_frame_equal(lossless.lines.sort_index(), both.lines.sort_index())
    pd.testing.assert_series_equal(lossless_busmap, both_busmap)


def test_lossless_leaves_an_unequally_rated_degree_two_bus_alone():
    """Merging 100 MW with 150 MW would rate the corridor at 100 and write off 50,
    so under lossless `m` and both its segments stay as built."""
    n = _network_with_degree_two_bus()
    n.lines.loc["line_1", "s_nom"] = 150.0
    lines_before = set(n.lines.index)

    reduced, busmap = reduce_low_degree_buses(n, level="lossless")

    assert "m" in reduced.buses.index
    assert (busmap == busmap.index).all()
    assert reduced.buses.at["m", "Pd"] == pytest.approx(80.0)
    assert reduced.buses.at["a", "Pd"] == pytest.approx(0.0)
    assert set(reduced.lines.index) == lines_before
    np.testing.assert_allclose(reduced.lines.at["line_0", "x"], 1.0)
    np.testing.assert_allclose(reduced.lines.at["line_1", "x"], 3.0)


@pytest.mark.parametrize(
    ("column", "value", "default"),
    [("s_nom_max", 300.0, 200.0), ("s_max_pu", 0.5, 1.0), ("s_nom_extendable", True, False)],
)
def test_lossless_requires_every_rating_attribute_to_match(column, value, default):
    """Equal `s_nom` alone is not enough: the merge takes the minimum of the
    expansion bound too, and a mismatched `s_max_pu` or extendability would change
    what the corridor may carry or build."""
    n = _network_with_degree_two_bus()
    if column not in n.lines.columns:
        n.lines[column] = default
    n.lines.loc["line_1", column] = value

    reduced, _ = reduce_low_degree_buses(n, level="lossless")

    assert "m" in reduced.buses.index


def test_lossless_tolerates_float_noise_in_the_rating():
    n = _network_with_degree_two_bus()
    n.lines.loc["line_1", "s_nom"] = 100.0 * (1 + 1e-9)

    reduced, _ = reduce_low_degree_buses(n, level="lossless")

    assert "m" not in reduced.buses.index


def test_lossless_never_splits_a_plant_across_two_buses():
    """The other approximation lossless exists to avoid.

    Under degree1and2 this plant is cloned onto `a` and `b` in the Kron ratio,
    and the two halves then dispatch independently. Under lossless it is one
    plant on its own bus, as it was built -- even though the two Lines at `m` are
    equally rated.
    """
    n = _split_plant_network()

    reduced, _ = reduce_low_degree_buses(n, level="lossless")

    assert "m" in reduced.buses.index
    assert set(reduced.generators.index) == {"middle", "at_a"}
    assert reduced.generators.at["middle", "bus"] == "m"
    np.testing.assert_allclose(reduced.generators.at["middle", "p_nom"], 100.0)
    np.testing.assert_allclose(reduced.generators.at["middle", "p_nom_max"], 200.0)


def test_lossless_never_splits_a_storage_unit():
    n = _network_with_degree_two_bus()
    n.add("StorageUnit", "batt", bus="m", carrier="battery", p_nom=100.0, max_hours=4.0)

    reduced, _ = reduce_low_degree_buses(n, level="lossless")

    assert "m" in reduced.buses.index
    assert list(reduced.storage_units.index) == ["batt"]
    assert reduced.storage_units.at["batt", "bus"] == "m"


def test_lossless_never_relocates_a_link():
    """A Link is one converter pair with one site; moving it wholesale to the
    nearer neighbour is not lossless, so a bus holding either end stays."""
    n = _network_with_degree_two_bus()
    n.add("Bus", "far")
    n.add("Link", "dc", bus0="m", bus1="far", p_nom=50.0, carrier="DC")

    reduced, _ = reduce_low_degree_buses(n, level="lossless")

    assert "m" in reduced.buses.index
    assert reduced.links.at["dc", "bus0"] == "m"


def test_lossless_splits_a_load_by_the_kron_factor():
    """A Load is a fixed injection, so it does not block the merge and its Kron
    split is exact: 3/4 of it lands at `a` behind x=1, 1/4 at `b` behind x=3."""
    n = _network_with_degree_two_bus()
    n.add("Load", "demand", bus="m", p_set=40.0)

    reduced, _ = reduce_low_degree_buses(n, level="lossless")

    assert "m" not in reduced.buses.index
    by_bus = reduced.loads.groupby("bus").p_set.sum()
    assert by_bus.at["a"] == pytest.approx(30.0)
    assert by_bus.at["b"] == pytest.approx(10.0)


def test_lossless_collapses_an_equally_rated_chain_to_a_fixed_point():
    """`a - m1 - m2 - m3 - b`, every segment 100 MW and every middle bus empty:
    the chain is spliced a pair at a time over several passes and ends as one
    corridor whose reactance is the sum of its four segments."""
    edges = _k4_edges() + [
        ("a", "m1", 1.0),
        ("m1", "m2", 2.0),
        ("m2", "m3", 3.0),
        ("m3", "b", 4.0),
    ]
    n = _network_from_edges(edges, ("a", "b", "c", "d", "m1", "m2", "m3"))
    n.buses.loc[["m1", "m2", "m3"], "Pd"] = 10.0

    reduced, busmap = reduce_low_degree_buses(n, level="lossless")

    assert set(reduced.buses.index) == {"a", "b", "c", "d"}
    assert set(busmap.loc[["m1", "m2", "m3"]]) <= {"a", "b"}
    corridor = reduced.lines.loc[~reduced.lines.index.isin([f"line_{i}" for i in range(6)])]
    assert len(corridor) == 1
    np.testing.assert_allclose(corridor.x.iloc[0], 10.0)
    np.testing.assert_allclose(corridor.s_nom.iloc[0], 100.0)
    assert reduced.buses.Pd.sum() == pytest.approx(30.0)


def test_lossless_leaves_a_double_stub_standing():
    """`m` is a radial dead end reached only through `a`. degree1and2 folds it in
    as a stub; lossless leaves it, which keeps what lossless removes a subset of
    what degree1and2 removes and restricted to the two cases its name covers.
    """
    kept, busmap = reduce_low_degree_buses(_double_stub_network(), level="lossless")
    assert "m" in kept.buses.index
    assert (busmap == busmap.index).all()
    assert kept.buses.at["m", "Pd"] == pytest.approx(80.0)
    assert {"line_0", "line_1"} <= set(kept.lines.index)

    folded, folded_busmap = reduce_low_degree_buses(_double_stub_network(), level="degree1and2")
    assert "m" not in folded.buses.index
    assert folded_busmap.at["m"] == "a"
    assert folded.buses.at["a", "Pd"] == pytest.approx(80.0)


def test_lossless_removes_a_strict_subset_of_what_degree1and2_removes():
    """Why this is one three-valued key and not two independent switches: the
    levels are nested, so no combination of them could mean anything new."""
    ones, _ = reduce_low_degree_buses(_mixed_low_degree_network(), level="lossless")
    both, _ = reduce_low_degree_buses(_mixed_low_degree_network(), level="degree1and2")

    assert set(both.buses.index) < set(ones.buses.index)
    # `s` is the stub both take; `m`, unequally rated, only degree1and2 reaches.
    assert set(ones.buses.index) == {"a", "b", "c", "d", "m"}
    assert set(both.buses.index) == {"a", "b", "c", "d"}


def test_default_level_is_still_degree1and2():
    """Callers that predate the split have to keep the network they had."""
    default, default_busmap = reduce_low_degree_buses(_mixed_low_degree_network())
    explicit, explicit_busmap = reduce_low_degree_buses(
        _mixed_low_degree_network(),
        level="degree1and2",
    )

    assert set(default.buses.index) == set(explicit.buses.index)
    pd.testing.assert_series_equal(default_busmap, explicit_busmap)


def test_lossless_still_refuses_to_fold_a_stub_across_a_zone():
    """The zone guard sits on the shared path and needs its own check under this
    level.

    Folding `s1` into `a` would move its 10 MW out of `east`, which every
    ReEDS-facing constraint downstream is written against.
    """
    n = _stub_chain_on_a_k4(depth=1)
    n.buses["reeds_zone"] = "west"
    n.buses.loc["s1", "reeds_zone"] = "east"

    reduced, busmap = reduce_low_degree_buses(n, level="lossless")

    assert "s1" in reduced.buses.index
    assert busmap.at["s1"] == "s1"
    assert reduced.buses.at["s1", "Pd"] == pytest.approx(10.0)
    assert reduced.buses.at["a", "Pd"] == pytest.approx(0.0)


def test_lossless_skips_the_heterogeneity_line_when_it_merged_nothing(caplog):
    """With no series merge the metric has no groups and comes out NaN, which in a
    log reads as a broken measurement rather than an absent one. The level is
    named in the line that does go out, so the log says which one ran."""
    with caplog.at_level(logging.INFO, logger="simplify_network"):
        reduce_low_degree_buses(_mixed_low_degree_network(), level="lossless")

    assert "Series s_nom heterogeneity" not in caplog.text
    assert "Reduced low-degree buses (lossless)" in caplog.text
    assert "1 with unequally rated Lines" in caplog.text


def test_lossless_reports_zero_heterogeneity_for_its_merges(caplog):
    """Every string lossless collapses pools equal ratings, so the metric it logs
    is a check that has to read exactly 0."""
    with caplog.at_level(logging.INFO, logger="simplify_network"):
        reduce_low_degree_buses(_network_with_degree_two_bus(), level="lossless")

    assert "Series s_nom heterogeneity = 0.0000" in caplog.text


def test_degree1and2_mode_still_reports_the_series_heterogeneity(caplog):
    """The counterpart of the above: gating that log must not switch it off for the
    mode that does have something to report."""
    with caplog.at_level(logging.INFO, logger="simplify_network"):
        reduce_low_degree_buses(_mixed_low_degree_network(), level="degree1and2")

    assert "Series s_nom heterogeneity" in caplog.text
    assert "Reduced low-degree buses (degree1and2)" in caplog.text


@pytest.mark.parametrize(
    ("config_value", "surviving"),
    [
        ("degree1and2", {"a", "b", "c", "d"}),
        ("lossless", {"a", "b", "c", "d", "m"}),
        ("none", {"a", "b", "c", "d", "m", "s"}),
        # the spellings a config written before the split would carry
        (True, {"a", "b", "c", "d"}),
        (False, {"a", "b", "c", "d", "m", "s"}),
    ],
)
def test_the_config_value_reaches_the_network_it_names(config_value, surviving):
    """The join `simplify_network.__main__` makes, over one network that holds a
    case for each mode to disagree about.

    The busmap is checked alongside, for all three modes and not just the two that
    reduce: it has to stay total over the original buses and land inside the
    survivors, or the region dissolve and the exported ``busmap.csv`` -- which
    compose it with the contraction and target-count maps -- would lose the
    substations it is keyed by. A disabled pass hands back the identity for
    exactly that reason.
    """
    n = _mixed_low_degree_network()
    original = set(n.buses.index)

    level = parse_low_degree_reduction(config_value)
    if level is None:
        reduced, busmap = n, identity_busmap(n)
    else:
        reduced, busmap = reduce_low_degree_buses(n, level=level)

    assert set(reduced.buses.index) == surviving
    assert set(busmap.index) == original
    assert set(busmap) <= surviving
    # Every bus that went is accounted for, and every survivor maps to itself.
    for bus in surviving:
        assert busmap.at[bus] == bus
