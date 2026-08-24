# BY PyPSA-USA Authors
"""Aggregates network to substations and simplifies to a single voltage level.

This module also owns the aggregation helpers this stage drives: the
electrical-distance busmap, the pypsa clustering wrapper carrying the
repository-standard component strategies, the region dissolve, and the topology
plots. They used to sit in a separate ``cluster_network`` module that nothing but
this one ever imported, and there is no ``cluster_network`` rule -- every
aggregation pass runs here.
"""

import logging
import warnings
from functools import reduce

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pypsa
from _helpers import (
    LINK_FIXED_COST_COL,
    LINK_UNIT_COST_COL,
    configure_logging,
    read_network,
    recompute_link_transmission_costs,
    register_topology_carriers,
    update_p_nom_max,
)
from pypsa.clustering.spatial import get_clustering_from_busmap
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import splu
from sklearn.neighbors import BallTree

matplotlib.use("Agg")  # rendered to file inside a Snakemake job, never shown

import matplotlib.pyplot as plt  # noqa: E402
from cartopy import crs as ccrs  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

warnings.filterwarnings(action="ignore", category=UserWarning)

logger = logging.getLogger(__name__)

#: Deepest transformer chain to resolve when collapsing transformers away.
#: The TAMU network needs 4; the cap only guards against pathological data.
MAX_TRAFO_CHAIN_DEPTH = 16


def convert_to_per_unit(df):
    # Calculating base values per component
    df["base_impedance"] = df["v_nom"] ** 2 / df["s_nom"]
    df["base_susceptance"] = 1 / df["base_impedance"]

    # Converting to per-unit values
    df["resistance_pu"] = df["r"] / df["base_impedance"]
    df["reactance_pu"] = df["x"] / df["base_impedance"]
    df["susceptance_pu"] = df["b"] / df["base_susceptance"]

    # Dropping intermediate columns (optional)
    df = df.drop(["base_impedance", "base_susceptance"], axis=1)

    return df


def convert_to_voltage_level(n, new_voltage):
    """
    Converts network.lines parameters to a given voltage.

    Parameters
    ----------
    n (pypsa.Network): Network
    new_voltage (float): New voltage level
    """
    df = convert_to_per_unit(n.lines.copy())

    df["new_base_impedance"] = new_voltage**2 / df["s_nom"]

    # Convert per-unit values back to actual values using the new base impedance
    df["r"] = df["resistance_pu"] * df["new_base_impedance"]
    df["x"] = df["reactance_pu"] * df["new_base_impedance"]
    df["b"] = df["susceptance_pu"] / df["new_base_impedance"]

    df.v_nom = new_voltage

    # Dropping intermediate column
    df = df.drop(
        ["new_base_impedance", "resistance_pu", "reactance_pu", "susceptance_pu"],
        axis=1,
    )

    # Update network lines. No `type` is set: r/x/b were just recomputed by the
    # per-unit conversion above and are authoritative. Assigning a standard type here
    # used to leave every line typed until it was wiped after substation aggregation,
    # so any calculate_dependent_values() in that window silently replaced them all
    # with `per_length * length`.
    n.buses["v_nom"] = new_voltage
    n.lines = df
    return n


#: Bus columns holding demand weight. These are *extensive* quantities -- they
#: must follow a bus when it is collapsed into another, not be discarded with it.
EXTENSIVE_BUS_ATTRS = ("Pd", "LAF_state")


def remove_transformers(n):
    trafo_map = pd.Series(n.transformers.bus1.values, index=n.transformers.bus0.values)
    trafo_map = trafo_map[~trafo_map.index.duplicated(keep="first")]

    # Transformer chains run up to 4 deep in the TAMU data, so resolve them to a
    # fixed point rather than re-mapping once: a single pass leaves entries still
    # pointing at a bus that is itself dropped below, which would strand both the
    # component references and the bus attributes re-assigned afterwards.
    for _ in range(MAX_TRAFO_CHAIN_DEPTH):
        several_trafo_b = trafo_map.isin(trafo_map.index)
        if not several_trafo_b.any():
            break
        remapped = trafo_map.loc[several_trafo_b].map(trafo_map)
        if remapped.equals(trafo_map.loc[several_trafo_b]):
            break  # self-referential chain; further passes cannot resolve it
        trafo_map.loc[several_trafo_b] = remapped
    else:
        logger.warning(
            "Transformer chains not fully resolved after %s passes; %s buses still "
            "map onto a bus that is itself remapped.",
            MAX_TRAFO_CHAIN_DEPTH,
            int(trafo_map.isin(trafo_map.index).sum()),
        )

    missing_buses_i = n.buses.index.difference(trafo_map.index)
    missing = pd.Series(missing_buses_i, missing_buses_i)
    trafo_map = pd.concat([trafo_map, missing])

    # Carry demand weight across the transformer before the source-side buses are
    # dropped. Every transformer's bus0 and bus1 belong to the same substation and
    # share coordinates, so this is a within-substation move, but skipping it loses
    # the Pd sitting on transformer low sides outright -- 5.1% of national peak
    # demand across ~1,170 substations, ~1,045 of which are left at Pd == 0 despite
    # serving load. Downstream that weight drives both the EER demand allocation and
    # the low-degree reduction in `reduce_low_degree_buses_and_merge_parallel_lines`.
    for col in EXTENSIVE_BUS_ATTRS:
        if col in n.buses.columns:
            values = pd.to_numeric(n.buses[col], errors="coerce").fillna(0.0)
            n.buses[col] = values.groupby(trafo_map).sum().reindex(n.buses.index).fillna(0.0)

    for c in n.one_port_components | n.branch_components:
        df = n.df(c)
        for col in df.columns:
            if col.startswith("bus"):
                df[col] = df[col].map(trafo_map)

    n.mremove("Transformer", n.transformers.index)
    n.mremove("Bus", n.buses.index.difference(trafo_map))
    return n, trafo_map


def aggregate_to_substations(
    network: pypsa.Network,
    busmap,
    topological_boundaries: str,
    line_length_factor: float,
    aggregation_strategies=dict(),
):
    """Aggregate buses to substation level.

    ``line_length_factor`` is the configured routing factor, not a local constant:
    pypsa rescales ``x``/``r``/``capital_cost`` by ``new_length / old_length`` on every
    aggregation, so the factor here must be the same one ``assign_line_length`` used
    for it to cancel instead of biasing impedance and cost.
    """
    logger.info("Aggregating buses to substation level...")

    generator_strategies = apply_wind_solar_cf_aggregation_weights(
        network,
        aggregation_strategies.get("generators", dict()),
    )
    one_port_strategies = aggregation_strategies.get("one_ports", dict())

    clustering = get_clustering_from_busmap(
        network,
        busmap,
        aggregate_generators_weighted=True,
        aggregate_one_ports=["Load"],
        line_length_factor=line_length_factor,
        bus_strategies={
            "type": "max",
            "Pd": "sum",
            "LAF_state": "sum",
        },
        generator_strategies=generator_strategies,
        one_port_strategies=one_port_strategies,
    )

    substations = network.buses[
        [
            "sub_id",
            "interconnect",
            "state",
            "country",
            "county",
            "balancing_area",
            "reeds_zone",
            "reeds_ba",
            "reeds_state",
            "x",
            "y",
        ]
    ]
    substations = substations.drop_duplicates(subset=["sub_id"])
    substations.sub_id = substations.sub_id.astype(int).astype(str)
    substations.index = substations.sub_id

    match topological_boundaries:
        case "county":
            zone = substations.county
        case "reeds_zone":
            zone = substations.reeds_zone
        case "state":
            zone = substations.reeds_state
        case _:
            raise ValueError(
                "zonal_aggregation must be either balancing_area, country, or state",
            )

    network_s = clustering.network

    network_s.buses["interconnect"] = substations.interconnect
    network_s.buses["x"] = substations.x
    network_s.buses["y"] = substations.y
    network_s.buses["substation_lv"] = True
    network_s.buses["country"] = zone  # country field used bc pypsa algo aggregates based on country field

    if topological_boundaries == "reeds_zone" or topological_boundaries == "county":
        cols2drop = [
            "balancing_area",
            "substation_off",
            "sub_id",
            "state",
        ]
    elif topological_boundaries == "state":
        cols2drop = [
            "balancing_area",
            "substation_off",
            "sub_id",
            "county",
            "reeds_zone",
            "reeds_ba",
            "trans_reg",
            "trans_grp",
            "state",
        ]
    else:
        cols2drop = [
            "balancing_area",
            "state",
            "substation_off",
            "sub_id",
            "reeds_zone",
            "reeds_ba",
            "trans_reg",
            "trans_grp",
            "reeds_state",
        ]

    # Only drop columns that exist in the DataFrame
    cols2drop = [col for col in cols2drop if col in network_s.buses.columns]
    network_s.buses = network_s.buses.drop(columns=cols2drop)
    return network_s, clustering.busmap


def merge_colocated_generators(n: pypsa.Network, moved: set) -> int:
    """
    Combine same-carrier generators that a relocation left sharing one bus.

    Only groups containing a relocated generator are touched -- everything else
    was already aggregated per (bus, carrier) upstream and is left alone.

    Nominal capacities add.  Efficiency, capital/marginal costs and per-unit
    availability are averaged with the same effective capacity weights used by
    the clustering stages: ``p_nom`` for existing plant and finite
    ``p_nom_max`` for pure candidates.
    """
    generators = n.generators
    if generators.empty or not moved:
        return 0

    group_key = generators.bus.astype(str) + "|" + generators.carrier.astype(str)
    counts = group_key.value_counts()
    touched = {key for key in group_key.reindex(list(moved)).dropna() if counts.get(key, 0) > 1}
    if not touched:
        return 0

    p_nom = pd.to_numeric(generators.p_nom, errors="coerce").fillna(0.0).clip(lower=0.0)
    p_nom_max = pd.to_numeric(generators.p_nom_max, errors="coerce").replace(
        [np.inf, -np.inf], np.nan,
    )
    weights = p_nom.where(p_nom.gt(0.0), p_nom_max).where(lambda x: x.gt(0.0), 1.0)
    static_p_max_pu = pd.to_numeric(generators.p_max_pu, errors="coerce").fillna(1.0)
    profiles = n.generators_t.p_max_pu

    drop: list = []
    for key in touched:
        members = group_key.index[group_key == key]
        keep = min(members)

        weight = weights.reindex(members).fillna(0.0).clip(lower=0.0)
        if weight.sum() <= 0:
            weight = pd.Series(1.0, index=members)
        weight = weight / weight.sum()

        if any(member in profiles.columns for member in members):
            blended = sum(
                (profiles[member] if member in profiles.columns else static_p_max_pu[member])
                * weight[member]
                for member in members
            )
            n.generators_t.p_max_pu[keep] = blended
        else:
            generators.at[keep, "p_max_pu"] = float((static_p_max_pu.reindex(members) * weight).sum())

        for col in ("efficiency", "capital_cost", "marginal_cost", "p_min_pu"):
            if col not in generators.columns:
                continue
            values = pd.to_numeric(generators[col].reindex(members), errors="coerce")
            if values.notna().any():
                generators.at[keep, col] = float((values.fillna(0.0) * weight).sum())

        for col in ("p_nom", "p_nom_min", "p_nom_max"):
            if col in generators.columns:
                generators.at[keep, col] = pd.to_numeric(
                    generators[col].reindex(members),
                    errors="coerce",
                ).fillna(0.0).sum()

        drop.extend(member for member in members if member != keep)

    n.mremove("Generator", drop)
    logger.info(
        "Merged %s co-located generators into %s (p_nom_max-weighted p_max_pu).",
        len(drop) + len(touched),
        len(touched),
    )
    return len(touched)


NOMINAL_POWER_COLUMNS = ("p_nom", "p_nom_min", "p_nom_max")
EXTENSIVE_TIME_SERIES = {"p", "q", "p_set", "q_set", "inflow"}


def split_one_port_components(
    n: pypsa.Network,
    assignments: dict,
) -> set:
    """Split one-port assets between the surviving buses of an eliminated bus.

    A degree-two Kron elimination distributes an injection using the opposite
    branch reactance.  We apply that factor to power capacities (and extensive
    dispatch time series); per-unit quantities such as ``p_max_pu`` remain
    unchanged.  Clones keep the original carrier and intensive attributes so
    the capacity-weighted aggregation in this module
    computes their correct combined value.
    """
    moved_generators: set = set()
    for component in n.one_port_components:
        df = n.df(component)
        if df.empty or "bus" not in df.columns:
            continue
        panels = n.pnl(component)
        panel_additions = {attr: {} for attr in panels}
        moving = list(df.index[df.bus.isin(assignments)])
        for name in moving:
            targets = assignments[df.at[name, "bus"]]
            original = df.loc[name].copy()
            for position, (target, share) in enumerate(targets):
                asset_name = name
                if position:
                    asset_name = f"{name}__split_{target}"
                    suffix = 2
                    while asset_name in df.index:
                        asset_name = f"{name}__split_{target}_{suffix}"
                        suffix += 1
                    df.loc[asset_name] = original
                df.at[asset_name, "bus"] = target
                for column in NOMINAL_POWER_COLUMNS:
                    if column in df.columns:
                        value = pd.to_numeric(pd.Series([original[column]]), errors="coerce").iloc[0]
                        if pd.notna(value):
                            df.at[asset_name, column] = value * share

                for attr, panel in panels.items():
                    if name not in panel.columns:
                        continue
                    values = panel[name] * share if attr in EXTENSIVE_TIME_SERIES else panel[name]
                    if position:
                        # Appending one column at a time creates thousands of
                        # pandas blocks for large renewable fleets. Collect all
                        # clone columns and concatenate them once below.
                        panel_additions[attr][asset_name] = values
                    elif attr in EXTENSIVE_TIME_SERIES:
                        panel.loc[:, asset_name] = values
                if component == "Generator":
                    moved_generators.add(asset_name)
        for attr, additions in panel_additions.items():
            if additions:
                panels[attr] = pd.concat(
                    [panels[attr], pd.DataFrame(additions, index=panels[attr].index)],
                    axis=1,
                )
    return moved_generators


def combine_parallel_impedance(values: np.ndarray) -> float:
    """Standard parallel combination ``1 / sum(1/value)`` for ``r`` or ``x``.

    A non-positive value shorts every other branch in a real circuit, so it
    is treated the same way here rather than dividing by zero: the combined
    value is 0. The data should not contain zero-impedance lines, so this is
    a safety net, not a modelled case.
    """
    values = np.asarray(values, dtype=float)
    if np.any(values <= 0):
        return 0.0
    return float(1.0 / np.sum(1.0 / values))


def merge_parallel_lines(lines: pd.DataFrame, capacity_cols: list[str]) -> tuple[pd.DataFrame, int]:
    """Merge every group of two or more Lines sharing the same unordered bus pair.

    ``r`` and ``x`` combine with the standard parallel-impedance formula
    (:func:`combine_parallel_impedance`), applied independently -- the same
    simplification the series merge above makes by summing ``r`` and ``x``
    separately rather than doing complex-impedance arithmetic.

    Each capacity column (``s_nom`` and, where present, ``s_nom_min``/``max``)
    combines as ``min(capacity_i * x_i) / x_parallel``. Under the linearised
    (DC) power-flow assumption used throughout this module, a branch carries a
    fixed share ``x_parallel / x_i`` of the total transfer, so it saturates
    once that transfer reaches ``capacity_i * x_i / x_parallel``; the smallest
    such threshold across the group sets the merged corridor's rating. (Naively
    combining ``min(capacity_i / x_i)`` instead gets both the direction and the
    magnitude wrong -- for two identical lines it halves the combined rating
    instead of doubling it.) A degenerate zero-reactance group falls back to
    the smallest capacity present, which the data should not contain.

    ``length`` averages rather than sums: parallel circuits run the same
    physical route, so adding their lengths would double-count it.
    ``capital_cost`` is a capacity-weighted average, ``sum(cost_i * s_nom_i) /
    sum(s_nom_i)`` -- it is a per-MW rate, and this keeps the total investment
    (``cost * s_nom``, summed over the group) unchanged by the merge, matching
    how pypsa's own clustering treats parallel lines. Other static columns are
    inherited from the first name in sort order, as in the series-merge case.

    Returns the Lines frame with every such group replaced by one merged row,
    and the number of groups merged.
    """
    if lines.empty:
        return lines, 0

    # Vectorised min/max rather than a row-wise apply: this module runs on
    # networks with tens of thousands of Lines, and one pass here happens
    # every iteration of the caller's fixed-point loop.
    bus0_le = lines["bus0"] <= lines["bus1"]
    lo = lines["bus0"].where(bus0_le, lines["bus1"])
    hi = lines["bus1"].where(bus0_le, lines["bus0"])
    pair = pd.Series(list(zip(lo, hi)), index=lines.index)
    merge_groups = [sorted(names) for names in pair.groupby(pair).groups.values() if len(names) > 1]
    if not merge_groups:
        return lines, 0

    drop_names: list = []
    merged_rows: dict = {}
    for names in merge_groups:
        rows = lines.loc[names]
        keep = names[0]
        merged = rows.loc[keep].copy()

        x = rows["x"].to_numpy(dtype=float)
        merged["r"] = combine_parallel_impedance(rows["r"].to_numpy(dtype=float))
        x_p = combine_parallel_impedance(x)
        merged["x"] = x_p

        for col in capacity_cols:
            caps = rows[col].to_numpy(dtype=float)
            merged[col] = float(np.min(caps * x)) / x_p if x_p > 0 else float(np.min(caps))

        if "length" in rows.columns:
            merged["length"] = float(rows["length"].mean())
        if "capital_cost" in rows.columns and "s_nom" in rows.columns:
            caps = rows["s_nom"].to_numpy(dtype=float)
            total_cap = caps.sum()
            merged["capital_cost"] = (
                float((rows["capital_cost"].to_numpy(dtype=float) * caps).sum() / total_cap)
                if total_cap > 0
                else float(rows["capital_cost"].mean())
            )

        merged_rows[keep] = merged
        drop_names.extend(names)

    remaining = lines.drop(index=drop_names)
    add_df = pd.DataFrame(
        [row.to_dict() for row in merged_rows.values()],
        index=list(merged_rows.keys()),
    ).reindex(columns=lines.columns)
    return pd.concat([remaining, add_df]), len(merge_groups)


def reduce_low_degree_buses_and_merge_parallel_lines(n: pypsa.Network) -> tuple[pypsa.Network, pd.Series]:
    """
    Iteratively eliminate every bus of degree 1 or 2 and merge parallel Lines.

    Runs to a fixed point, so the result holds no bus with fewer than three
    incident Lines and no two buses joined by more than one Line: eliminating
    a bus can drop a neighbour to degree 2 or 1, and merging a parallel group
    can do the same, so each pass re-examines the whole graph from scratch.
    Every pass first merges parallel Lines (:func:`merge_parallel_lines`),
    then reduces low-degree buses; either step can create fresh opportunities
    for the other; the loop stops only once a full pass does neither.

    Three cases arise for a low-degree bus:

    * **degree 2** -- the two segments combine in series. Because parallel
      merging already ran this pass, its two neighbours are always distinct
      (two Lines to the same neighbour would already be one Line).
      ``r``/``x``/``length``/``capital_cost`` add; ``s_nom`` (and
      ``s_nom_min``/``max``) take the *minimum*, since a corridor's transfer
      capability is set by its narrowest section, not the sum of its parts.
      ``capital_cost`` is a per-MW figure proportional to length, so raising the
      merged corridor's rating by 1 MW means widening *both* segments and costs
      the sum -- unlike the parallel case above, which averages instead. Leaving
      it at one segment's value would survive the final clustering too, which
      rescales by ``capital_cost / length``: an n-segment chain would be priced
      at roughly 1/n of its true cost.
    * **degree 1** -- a stub. It folds into its single neighbour and the Line
      is dropped; nothing behind a radial tip constrains the rest of the grid.
      (A bus with two parallel Lines to the same neighbour ends up here too,
      one pass later, once those Lines have merged into one.)
    * **degree 0** -- left in place. Deleting it would discard its demand
      weight with nowhere to send it, so it is logged instead.

    Bus attributes and components move as follows:

    * ``Pd``/``LAF_state`` split between the two neighbours in the ratio that
      Kron elimination of the DC power-flow Laplacian prescribes -- the
      neighbour reached through reactance ``x1`` receives ``x2 / (x1 + x2)``.
      This is *exact* for every flow in the rest of the network, and since
      ``build_eer_demand`` allocates demand off these same weights, the load
      split is exact too. Handing the whole weight to one neighbour instead
      would misplace a fifth of national peak injection, a quarter of these
      buses sitting on splits worse than 70/30. A degree-1 bus has a single
      neighbour, which therefore takes everything.
    * One-port components split between the two neighbours by the same Kron
      factor.  Their nominal capacity is split, while efficiency, costs and
      per-unit availability remain intensive values and are capacity-weighted
      when same-carrier generators arrive at one bus.
    * Links relocate wholesale to whichever neighbour received the larger Kron
      share (the single neighbour, for a degree-1 bus) -- unlike one-port
      assets, a Link is one converter pair with one rating and is never split
      across two buses. A Link stranded as a self-loop by this move is dropped.

    Self-loops are dropped as they appear, so a bus's degree never counts one.
    A merged Line takes the first of its two names in sort order, and its
    remaining static attributes from that same segment; by this point every Line
    shares a nominal voltage courtesy of ``convert_to_voltage_level``.

    Returns the reduced network and a busmap over the *original* bus index
    giving, for each bus, the surviving bus that absorbed it. Anything keyed
    to the pre-reduction buses -- the Voronoi regions, and the substation-keyed
    renewable profiles that ``add_electricity`` joins through ``bus2sub`` --
    has to be pushed through this map, so it is composed across passes rather
    than discarded.
    """
    busmap = pd.Series(n.buses.index, index=n.buses.index, name="busmap")
    if n.lines.empty:
        return n, busmap

    split_cols = [c for c in EXTENSIVE_BUS_ATTRS if c in n.buses.columns]
    min_cols = [c for c in ("s_nom", "s_nom_min", "s_nom_max") if c in n.lines.columns]
    sum_cols = [c for c in ("length", "capital_cost") if c in n.lines.columns]

    parallel_merged = series_merged = stubs_removed = loops_removed = 0
    link_endpoints_relocated = 0
    relocated_generators: set = set()

    while True:
        lines = n.lines
        self_loops = lines.index[lines.bus0 == lines.bus1]
        if len(self_loops):
            n.lines = lines.drop(index=self_loops)
            loops_removed += len(self_loops)
            continue

        lines, n_parallel_groups = merge_parallel_lines(lines, min_cols)
        if n_parallel_groups:
            n.lines = lines
            parallel_merged += n_parallel_groups
            continue

        adj: dict = {}
        for name, bus0, bus1 in lines[["bus0", "bus1"]].itertuples(name=None):
            adj.setdefault(bus0, []).append((name, bus1))
            adj.setdefault(bus1, []).append((name, bus0))

        consumed: set = set()
        drop_lines: list = []
        new_lines: list = []
        assignments: dict = {}
        reassigned = {col: {} for col in split_cols}

        def hand_over(bus, targets):
            """Give ``bus``'s extensive attributes to ``targets``: (bus, share) pairs."""
            for col in split_cols:
                value = pd.to_numeric(n.buses.at[bus, col], errors="coerce")
                if pd.isna(value) or value == 0:
                    continue
                for target, share in targets:
                    reassigned[col][target] = reassigned[col].get(target, 0.0) + value * share

        for bus, edges in adj.items():
            if bus in consumed or len(edges) > 2:
                continue
            neighbours = {b for _, b in edges}
            if consumed & neighbours:
                continue  # a neighbour is going away this pass; retry next pass

            if len(edges) == 1:
                l1, b1 = edges[0]
                hand_over(bus, [(b1, 1.0)])
                assignments[bus] = [(b1, 1.0)]
                drop_lines.append(l1)
                consumed.add(bus)
                stubs_removed += 1
                continue

            # Parallel merging already ran this pass, so a degree-2 bus's two
            # Lines can never lead to the same neighbour here.
            (l1, b1), (l2, b2) = edges
            row1, row2 = lines.loc[l1], lines.loc[l2]
            merged = row1.copy()
            merged["bus0"], merged["bus1"] = b1, b2
            merged["r"] = row1.r + row2.r
            merged["x"] = row1.x + row2.x
            for col in sum_cols:
                merged[col] = row1[col] + row2[col]
            for col in min_cols:
                merged[col] = min(row1[col], row2[col])

            # Kron split: the neighbour behind x1 takes x2 / (x1 + x2). Equal
            # halves are the only sane fallback for a degenerate zero-reactance
            # pair, which the data should not contain.
            x_total = float(row1.x) + float(row2.x)
            share1 = float(row2.x) / x_total if x_total > 0 else 0.5
            hand_over(bus, [(b1, share1), (b2, 1.0 - share1)])
            assignments[bus] = [(b1, share1), (b2, 1.0 - share1)]

            # Inherit the first of the two names in sort order rather than
            # concatenating them: a chain seventeen segments deep would otherwise
            # build an unreadable identifier, and since the other name is retired
            # in the same step the index stays unique either way.
            new_lines.append((min(l1, l2), merged))
            drop_lines.extend([l1, l2])
            consumed.add(bus)
            series_merged += 1

        if not consumed:
            break

        # Bulk-replace n.lines (as convert_to_voltage_level does above) rather than
        # inserting rows one at a time via n.add()/`.loc[new_label]`: both of those
        # go through pypsa's per-row attribute validation, which silently drops
        # columns that aren't part of the official Line schema (e.g. v_nom,
        # interconnect, underwater_fraction are custom columns added in bulk
        # earlier in this pipeline) and leaves them NaN on the new rows.
        remaining = n.lines.drop(index=drop_lines)
        if new_lines:
            add_df = pd.DataFrame(
                [row.to_dict() for _, row in new_lines],
                index=[name for name, _ in new_lines],
            ).reindex(columns=n.lines.columns)
            remaining = pd.concat([remaining, add_df])
        n.lines = remaining

        # Targets are never themselves consumed in the same pass.  A degree-two
        # bus has two targets, so its one-port capacities must be split rather
        # than moved wholesale to the electrically nearer neighbour.
        relocated_generators.update(split_one_port_components(n, assignments))
        for col in split_cols:
            if reassigned[col]:
                addition = pd.Series(reassigned[col])
                n.buses[col] = (
                    pd.to_numeric(n.buses[col], errors="coerce").fillna(0.0)
                    + addition.reindex(n.buses.index).fillna(0.0)
                )

        # A Link is one converter pair with one rating, sited at a specific
        # substation -- unlike one-port assets it is never split between the
        # two Kron targets. It moves wholesale to whichever neighbour received
        # the larger share (the only neighbour, for a degree-1 bus).
        relocation = {bus: max(targets, key=lambda pair: pair[1])[0] for bus, targets in assignments.items()}
        if not n.links.empty:
            moved_links = (set(n.links.bus0) | set(n.links.bus1)) & set(relocation)
            if moved_links:
                n.links["bus0"] = n.links["bus0"].map(lambda b: relocation.get(b, b))
                n.links["bus1"] = n.links["bus1"].map(lambda b: relocation.get(b, b))
                link_endpoints_relocated += len(moved_links)

                self_loop_links = n.links.index[n.links.bus0 == n.links.bus1]
                if len(self_loop_links):
                    logger.warning(
                        "%s Links collapsed onto a single bus by low-degree reduction; dropping.",
                        len(self_loop_links),
                    )
                    n.mremove("Link", self_loop_links)

        n.mremove("Bus", list(consumed))
        busmap = busmap.map(lambda bus: relocation.get(bus, bus))

    merge_colocated_generators(n, relocated_generators)

    isolated = n.buses.index.difference(pd.Index(n.lines.bus0).union(n.lines.bus1))
    if len(isolated):
        logger.warning("%s buses left with no incident Line after reduction.", len(isolated))
    logger.info(
        "Reduced low-degree buses and parallel lines: %s parallel-line groups merged, "
        "%s series merges, %s degree-1 stubs, %s self-loops dropped, %s Link endpoints relocated.",
        parallel_merged,
        series_merged,
        stubs_removed,
        loops_removed,
        link_endpoints_relocated,
    )
    return n, busmap


# ---------------------------------------------------------------------------
# Aggregation helpers, migrated from the former cluster_network module
# ---------------------------------------------------------------------------


def apply_wind_solar_cf_aggregation_weights(
    n: pypsa.Network,
    generator_strategies: dict | None = None,
) -> dict:
    """
    Prepare generator aggregation with a capacity weight for every generator.

    ``p_nom_max`` is the weight that conserves a group's energy potential.
    The aggregate carries the summed ``p_nom_max``, so its profile has to be
    ``sum_i(p_nom_max_i * p_max_pu_i) / sum_i(p_nom_max_i)`` for
    ``p_nom_max_agg * p_max_pu_agg(t)`` to reproduce
    ``sum_i p_nom_max_i * p_max_pu_i(t)`` exactly.

    Weighting by energy potential (``p_nom_max * mean_CF``) instead yields a
    CF-weighted mean of CF, which is biased upward: it presumes the better
    sites inside a group get built first, while still advertising the group's
    entire potential as buildable at that inflated capacity factor.

    Existing capacity uses ``p_nom``.  A pure candidate has zero existing
    capacity, so it instead uses its finite positive ``p_nom_max``.  The same
    weight is applied to all intensive component attributes, including
    efficiency and costs, rather than relying on PyPSA's default ``p_nom``
    weight (which is zero for renewable supply-curve candidates).
    """
    generator_strategies = dict(generator_strategies or {})

    if n.generators.empty:
        return generator_strategies

    p_nom = (
        pd.to_numeric(
            n.generators.get("p_nom", pd.Series(0.0, index=n.generators.index)),
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )
    p_nom_max = pd.to_numeric(
        n.generators.get("p_nom_max", pd.Series(np.nan, index=n.generators.index)),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    weights = p_nom.where(p_nom.gt(0.0), p_nom_max).where(lambda x: x.gt(0.0), 1.0)
    n.generators["weight"] = weights

    # `weighted_average` uses the just-populated `weight` column.  This makes
    # capacity and all intensive economics/operating parameters consistent
    # during both substation simplification and final clustering.
    for attribute in (
        "efficiency",
        "capital_cost",
        "marginal_cost",
        "p_max_pu",
        "p_min_pu",
    ):
        generator_strategies[attribute] = "weighted_average"
    return generator_strategies




def clustering_from_busmap(
    n,
    busmap,
    line_length_factor: float,
    aggregate_carriers=None,
    aggregation_strategies=dict(),
):
    """Aggregate a network with the repository-standard component strategies.

    ``line_length_factor`` is required rather than defaulted: it sets the aggregated
    ``length``, and therefore the rescaling pypsa applies to ``x``/``r``/
    ``capital_cost``. A default would let a caller silently pick a routing factor
    different from the one the lengths were built with, which breaks that cancellation.
    """
    line_strategies = aggregation_strategies.get("lines", dict())
    generator_strategies = apply_wind_solar_cf_aggregation_weights(
        n,
        aggregation_strategies.get("generators", dict()),
    )
    one_port_strategies = aggregation_strategies.get("one_ports", dict())
    bus_strategies = {"Pd": "sum", "LAF_state": "sum"}
    return get_clustering_from_busmap(
        n,
        busmap,
        aggregate_generators_weighted=True,
        aggregate_generators_carriers=aggregate_carriers,
        aggregate_one_ports=["Load", "StorageUnit"],
        line_length_factor=line_length_factor,
        line_strategies=line_strategies,
        generator_strategies=generator_strategies,
        bus_strategies=bus_strategies,
        one_port_strategies=one_port_strategies,
        scale_link_capital_costs=False,
    )


def _effective_reactance_embedding(
    n: pypsa.Network,
    n_probes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate effective-reactance distances using Laplacian projections.

    The squared embedding distance estimates the effective reactance
    ``(e_i-e_j)\' L^-1 (e_i-e_j)`` in ohm, where
    ``L = B diag(1/x) B\'``. It uses only AC-line topology and supplied series
    reactances, never a solved dispatch or base power flow.
    """
    buses = n.buses.index
    bus_position = pd.Series(np.arange(len(buses)), index=buses)
    lines = n.lines.copy()
    x = pd.to_numeric(lines.x, errors="coerce")
    valid = (
        x.gt(0)
        & lines.bus0.isin(buses)
        & lines.bus1.isin(buses)
    )
    lines = lines.loc[valid]
    x = x.loc[valid].to_numpy(float)
    n_buses, n_lines = len(buses), len(lines)
    if n_lines == 0:
        return np.zeros((n_buses, n_probes)), np.arange(n_buses)

    bus0 = bus_position.loc[lines.bus0].to_numpy()
    bus1 = bus_position.loc[lines.bus1].to_numpy()
    incidence = sparse.coo_matrix(
        (
            np.r_[np.ones(n_lines), -np.ones(n_lines)],
            (np.r_[bus0, bus1], np.r_[np.arange(n_lines), np.arange(n_lines)]),
        ),
        shape=(n_buses, n_lines),
    ).tocsc()
    laplacian = incidence @ sparse.diags(1 / x) @ incidence.T
    _, islands = csgraph.connected_components(laplacian, directed=False)
    rng = np.random.default_rng(seed)
    rhs = incidence @ (
        np.sqrt(1 / x)[:, None] * rng.standard_normal((n_lines, n_probes))
    )
    embedding = np.zeros((n_buses, n_probes))
    for island in np.unique(islands):
        members = np.flatnonzero(islands == island)
        if len(members) > 1:
            reduced = laplacian[members][:, members][1:, 1:].tocsc()
            embedding[members[1:]] = splu(reduced).solve(rhs[members[1:]])
    return embedding, islands


def busmap_by_electrical_distance(
    n: pypsa.Network,
    max_geographic_distance_km: float = 10.0,
    max_electrical_distance_ohm: float = 10.0,
    n_probes: int = 128,
    seed: int = 123,
    topological_boundary: str | None = None,
) -> pd.Series:
    """Return a region-respecting, pairwise diameter-constrained busmap.

    Every group is a clique in the compatibility graph: any two buses in a
    group are in the same supplied topological region, at most
    ``max_geographic_distance_km`` apart, and have effective-reactance distance
    no larger than
    ``max_electrical_distance_ohm``. A deterministic dense-first clique cover
    gives a compact feasible aggregation without introducing a bus-count cap.
    """
    required = {"x", "y"}
    if topological_boundary is not None:
        required.add(topological_boundary)
    missing = required.difference(n.buses.columns)
    if missing:
        raise ValueError(
            "Electrical-distance clustering requires bus columns "
            f"{sorted(required)}; missing {sorted(missing)}.",
        )
    if max_geographic_distance_km <= 0 or max_electrical_distance_ohm <= 0:
        raise ValueError("Electrical-distance thresholds must be strictly positive.")
    if n_probes < 2:
        raise ValueError("Electrical-distance clustering requires at least two probes.")

    buses = n.buses
    if buses[["x", "y"]].isna().any().any():
        raise ValueError("Electrical-distance clustering requires coordinates for every bus.")

    embedding, islands = _effective_reactance_embedding(n, n_probes, seed)
    n_buses = len(buses)
    if topological_boundary is None:
        boundary = pd.Series("", index=buses.index, dtype="string")
    else:
        boundary = buses[topological_boundary].astype("string")
        boundary = boundary.where(
            boundary.notna(),
            pd.Series(buses.index.astype(str), index=buses.index).radd("__missing__"),
        )
    coordinates = np.radians(np.c_[buses.y.to_numpy(), buses.x.to_numpy()])
    neighbours = BallTree(coordinates, metric="haversine").query_radius(
        coordinates,
        r=max_geographic_distance_km / 6371.0088,
        return_distance=False,
    )
    adjacency = [set() for _ in range(n_buses)]
    compatible_edges = 0
    boundary_values = boundary.to_numpy()
    for bus, nearby in enumerate(neighbours):
        nearby = nearby[nearby > bus]
        if not len(nearby):
            continue
        electrical_distance = ((embedding[nearby] - embedding[bus]) ** 2).mean(axis=1)
        compatible = (
            (islands[nearby] == islands[bus])
            & (boundary_values[nearby] == boundary_values[bus])
            & (electrical_distance <= max_electrical_distance_ohm)
        )
        for neighbour in nearby[compatible]:
            adjacency[bus].add(int(neighbour))
            adjacency[neighbour].add(bus)
            compatible_edges += 1

    remaining = set(range(n_buses))
    busmap = pd.Series(index=buses.index, dtype=object, name="busmap")
    group = 0
    while remaining:
        pivot = max(remaining, key=lambda bus: (len(adjacency[bus] & remaining), -bus))
        members = [pivot]
        candidates = (adjacency[pivot] & remaining).copy()
        while candidates:
            member = max(candidates, key=lambda bus: (len(adjacency[bus] & candidates), -bus))
            members.append(member)
            candidates.intersection_update(adjacency[member])
        busmap.iloc[members] = f"electrical_{group}"
        remaining.difference_update(members)
        group += 1

    logger.info(
        "Electrical-distance clustering: %s buses -> %s clusters; %s compatible "
        "pairs; limits %.3f km, %.3f ohm, %s probes; topological boundary=%s.",
        n_buses,
        group,
        compatible_edges,
        max_geographic_distance_km,
        max_electrical_distance_ohm,
        n_probes,
        topological_boundary or "none",
    )
    return busmap


#: Map styling aligns with ``plot_network_maps``; state boundaries are the
#: only geographic outline intentionally shown by the topology diagnostics.
TOPOLOGY_FIGSIZE = (12, 12)
TOPOLOGY_DPI = 600


def plot_network_topology(
    n: pypsa.Network,
    path: str,
    wildcards=None,
    stage: str | None = None,
    state_boundaries: str | None = None,
    central_longitude: float | None = None,
    line_width_reference: float | None = None,
) -> None:
    """
    Draw every bus and Line of the clustered network onto a map.

    This is a check on the reduction, not a presentation figure: after
    ``reduce_low_degree_buses_and_merge_parallel_lines`` collapses the topology, the fastest way to see
    whether the surviving graph still looks like the grid it came from -- rather
    than a tangle of long-range shortcuts stitched between distant substations --
    is to look at it. Line width tracks ``s_nom`` so the transfer backbone stands
    out from the infill, and DC links are drawn separately because they are the
    only branches that cross an interconnect boundary.

    ``central_longitude`` and ``line_width_reference`` should be computed once
    from the raw, pre-reduction network and passed to every call for a given
    run: that keeps the map projection and the MW-to-linewidth scale identical
    across the after-transformers / after-substations / after-low-degree /
    final stages, so the four images are directly comparable rather than each
    silently rescaling itself to whatever survived that stage's reduction.
    """
    if n.buses.empty:
        logger.warning("No buses to plot; skipping topology map.")
        return

    ac_buses = n.buses[n.buses.carrier == "AC"] if "carrier" in n.buses.columns else n.buses
    if ac_buses.empty:
        ac_buses = n.buses

    if central_longitude is None:
        central_longitude = ac_buses.x.mean()

    fig, ax = plt.subplots(
        figsize=TOPOLOGY_FIGSIZE,
        subplot_kw={"projection": ccrs.EqualEarth(central_longitude)},
    )
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    boundaries = None
    if state_boundaries:
        states = gpd.read_file(state_boundaries)
        if not states.empty:
            bounds = states.total_bounds.astype(float)
            # This follows ``plot_network_maps.get_map_boundaries`` exactly.
            bounds[0] = min(bounds[2], bounds[0] + 2.0)
            bounds[2] = max(bounds[0], bounds[2] - 1.0)
            boundaries = tuple(bounds[[0, 2, 1, 3]])
            states.plot(
                ax=ax,
                facecolor="#EEEEEE",
                edgecolor="#ffffff",
                transform=ccrs.PlateCarree(),
                linewidth=1.0,
                zorder=0,
            )

    # Reference capacity for the MW-to-linewidth scale. Pass one in (computed
    # once from the raw network) so a given s_nom draws at the same width in
    # every stage's image; falling back to this stage's own 95th percentile
    # only applies when the caller wants a single, self-contained plot.
    s_nom = pd.to_numeric(n.lines.s_nom, errors="coerce").fillna(0.0)
    if line_width_reference is None or line_width_reference <= 0:
        reference = s_nom.quantile(0.95) if not s_nom.empty and s_nom.quantile(0.95) > 0 else 1.0
    else:
        reference = line_width_reference
    line_widths = (s_nom / reference).clip(upper=3.0) * 0.6 + 0.08
    bus_size = 2.2e-3 * (2000.0 / max(len(ac_buses), 1)) ** 0.5

    branch_components = ["Line"]
    link_widths = 0.0
    if not n.links.empty:
        branch_components.append("Link")
        link_widths = pd.Series(1.2, index=n.links.index)

    with plt.rc_context({"patch.linewidth": 0.0, "font.family": "Times New Roman"}):
        n.plot(
            ax=ax,
            bus_sizes=bus_size,
            bus_colors="#1f4e79",
            bus_alpha=0.7,
            line_widths=line_widths,
            line_colors="#4a7ba7",
            link_widths=link_widths,
            link_colors="#c0392b",
            boundaries=boundaries,
            color_geomap=False,
            branch_components=branch_components,
        )

    handles = [
        Line2D([], [], color="#4a7ba7", lw=1.6, label=f"AC lines ({len(n.lines):,})"),
        Line2D([], [], color="#1f4e79", marker="o", ls="", ms=5, label=f"buses ({len(n.buses):,})"),
    ]
    if not n.links.empty:
        handles.append(Line2D([], [], color="#c0392b", lw=1.6, label=f"DC links ({len(n.links):,})"))
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=11)

    subtitle = f"{len(n.buses):,} buses | {len(n.lines):,} AC lines"
    if stage:
        subtitle = f"{stage}\n{subtitle}"
    if wildcards is not None:
        parts = [f"{key}={value}" for key, value in dict(wildcards).items() if value != ""]
        if parts:
            subtitle += "\n" + "  |  ".join(parts)
    ax.set_title(
        "Network topology\n" + subtitle,
        fontsize=14,
        pad=16,
    )

    fig.savefig(path, dpi=TOPOLOGY_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "Wrote topology map with %s buses and %s lines to %s.",
        len(n.buses),
        len(n.lines),
        path,
    )


def cluster_regions(busmaps, input=None, output=None):
    """Create new geojson files for the clustered regions."""
    busmap = reduce(lambda x, y: x.map(y), busmaps[1:], busmaps[0])

    for which in ("regions_onshore", "regions_offshore"):
        regions = gpd.read_file(getattr(input, which))

        # Check if name column contains float values before indexing
        try:
            # Try to convert to float to see if values are numeric
            pd.to_numeric(regions["name"], errors="raise")
            is_float = True
        except:  # noqa: E722
            is_float = False

        # Reindex to set name as index
        regions = regions.reindex(columns=["name", "geometry"]).set_index("name")

        # Convert float indices to string representation of integers if needed
        if is_float:
            regions.index = regions.index.astype(float).astype(int).astype(str)

        # Dissolve regions according to busmap
        regions_c = regions.dissolve(busmap)
        regions_c.index.name = "name"
        regions_c = regions_c.reset_index()
        regions_c.to_file(getattr(output, which))


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("simplify_network", demand_level="High")
    configure_logging(snakemake)
    params = snakemake.params
    topological_boundaries = snakemake.params.topological_boundaries

    # Components are attached before topology reduction.  `read_network` keeps
    # the pickle hand-off's custom columns and generator time series intact.
    n = read_network(snakemake.input.network)
    register_topology_carriers(n)

    # Fix the map projection and the MW-to-linewidth scale once, from the raw
    # pre-reduction network, and reuse them for every topology plot below.
    # Otherwise each stage recomputes both from whatever survived that
    # reduction, so the four images silently use different map centers and
    # different capacity scales and end up impossible to compare by eye.
    raw_ac_buses = n.buses[n.buses.carrier == "AC"] if "carrier" in n.buses.columns else n.buses
    if raw_ac_buses.empty:
        raw_ac_buses = n.buses
    topology_central_longitude = raw_ac_buses.x.mean()
    raw_s_nom = pd.to_numeric(n.lines.s_nom, errors="coerce").fillna(0.0)
    topology_line_width_reference = (
        raw_s_nom.quantile(0.95) if not raw_s_nom.empty and raw_s_nom.quantile(0.95) > 0 else 1.0
    )

    n = convert_to_voltage_level(n, 230)
    n, _ = remove_transformers(n)
    plot_network_topology(
        n,
        snakemake.output.network_map_after_transformers,
        snakemake.wildcards,
        "After transformer removal",
        snakemake.input.state_boundaries,
        topology_central_longitude,
        topology_line_width_reference,
    )

    # new busmap definition
    busmap_to_sub = n.buses.sub_id.astype(int).astype(str).to_frame()

    n.links["underwater_fraction"] = 0

    n.buses.drop(columns=["substation_off"], inplace=True, errors="ignore")

    n, _ = aggregate_to_substations(
        n,
        busmap_to_sub.sub_id,
        topological_boundaries,
        params.length_factor,
        params.aggregation_strategies,
    )
    plot_network_topology(
        n,
        snakemake.output.network_map_after_substations,
        snakemake.wildcards,
        "After substation aggregation",
        snakemake.input.state_boundaries,
        topology_central_longitude,
        topology_line_width_reference,
    )

    n, reduction_busmap = reduce_low_degree_buses_and_merge_parallel_lines(n)
    plot_network_topology(
        n,
        snakemake.output.network_map_after_low_degree,
        snakemake.wildcards,
        "After low-degree reduction",
        snakemake.input.state_boundaries,
        topology_central_longitude,
        topology_line_width_reference,
    )

    if topological_boundaries in ["reeds_zone", "state"] and "county" in n.buses.columns:
        n.buses = n.buses.drop(columns=["county"])

    distance = params.electrical_distance
    busmap = busmap_by_electrical_distance(
        n,
        max_geographic_distance_km=distance["max_geographic_distance_km"],
        max_electrical_distance_ohm=distance["max_electrical_distance_ohm"],
        n_probes=distance.get("n_probes", 128),
        seed=distance.get("seed", 123),
        topological_boundary=topological_boundaries,
    )
    all_carriers = set(n.generators.carrier).union(set(n.storage_units.carrier))
    clustering = clustering_from_busmap(
        n,
        busmap,
        aggregate_carriers=all_carriers,
        line_length_factor=params.length_factor,
        aggregation_strategies=params.aggregation_strategies,
    )
    n = clustering.network
    update_p_nom_max(n)
    if "land_region" in n.generators.columns:
        n.generators["land_region"] = n.generators.land_region.fillna(n.generators.bus)
    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    register_topology_carriers(n)
    plot_network_topology(
        n,
        snakemake.output.network_map_after_electrical_distance,
        snakemake.wildcards,
        "After constrained electrical-distance aggregation",
        snakemake.input.state_boundaries,
        topology_central_longitude,
        topology_line_width_reference,
    )

    # The electrical-distance clustering above groups buses by geographic and
    # electrical distance and region membership -- it has no notion of the
    # resulting Line-graph degree, so it routinely leaves behind fresh
    # degree-1/2 buses that the first reduction pass never saw. Run the same
    # reduction again so the network handed to the model is free of them too.
    n, reduction_busmap_2 = reduce_low_degree_buses_and_merge_parallel_lines(n)
    register_topology_carriers(n)
    plot_network_topology(
        n,
        snakemake.output.network_map,
        snakemake.wildcards,
        "After second low-degree reduction",
        snakemake.input.state_boundaries,
        topology_central_longitude,
        topology_line_width_reference,
    )

    # The Voronoi regions are keyed by substations.  Compose all three maps --
    # first low-degree pass, electrical-distance clustering, second low-degree
    # pass -- before dissolving them into final regions and bus labels.
    busmaps = (reduction_busmap, clustering.busmap, reduction_busmap_2)
    cluster_regions(busmaps, snakemake.input, snakemake.output)

    # Anything still keyed by the raw substation ids -- the regions above, and
    # supply curves that `add_electricity` joins on `sub_id` -- needs to know
    # which surviving bus now stands for each one. The bus index at this point
    # *is* the substation id, so composing the maps gives exactly that.
    composed = reduce(lambda left, right: left.map(right), busmaps[1:], busmaps[0])
    composed.rename_axis("sub_id").rename("bus_id").to_csv(snakemake.output.busmap)

    # No Line-level counterpart is published. `clustering.linemap` covers only the
    # electrical-distance pass, so its index is intermediate Line names that match
    # nothing a caller holds, while merged Lines inherit the first of their names --
    # which makes a final name coincide with an unrelated original name often enough
    # (17k of 20k on the USA network) that joining the two would silently succeed and
    # be wrong. Tracing an original Line to its corridor needs name bookkeeping
    # through both low-degree passes as well; nothing consumes it today.

    # Substation aggregation, electrical-distance clustering and both low-degree
    # passes all move Link endpoints, so the distance their cost was priced on is
    # stale by now. Lines need no equivalent step: their `capital_cost` is strictly
    # proportional to `length`, which every aggregation rule above preserves.
    recompute_link_transmission_costs(n)
    # These are internal bookkeeping for recompute_link_transmission_costs --
    # nothing downstream of this rule needs them, so they don't belong in the
    # exported network.
    n.links.drop(columns=[LINK_UNIT_COST_COL, LINK_FIXED_COST_COL], errors="ignore", inplace=True)

    n.consistency_check()
    n.export_to_netcdf(snakemake.output.network)
