# BY PyPSA-USA Authors
"""Aggregates network to substations and simplifies to a single voltage level.

This module also owns the aggregation helpers this stage drives: the
target-bus-count busmap, the pypsa clustering wrapper carrying the
repository-standard component strategies, the region dissolve, and the
topology plots. They used to sit in a separate ``cluster_network`` module that nothing but
this one ever imported, and there is no ``cluster_network`` rule -- every
aggregation pass runs here.
"""

import logging
import warnings
from functools import reduce
from heapq import heappop, heappush
from typing import NamedTuple

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
from sklearn.cluster import ward_tree

matplotlib.use("Agg")  # rendered to file inside a Snakemake job, never shown

import matplotlib.pyplot as plt  # noqa: E402
from cartopy import crs as ccrs  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

warnings.filterwarnings(action="ignore", category=UserWarning)

logger = logging.getLogger(__name__)

#: Mean earth radius (WGS84 authalic), shared by every geographic distance here.
EARTH_RADIUS_KM = 6371.0088

#: Probe columns are i.i.d. Rademacher projections, following Spielman and
#: Srivastava's effective-resistance embedding. Any subsample is still a
#: Johnson-Lindenstrauss embedding whose relative error is uniform across
#: scales. PCA is the wrong knife here: it spends its dimensions on the largest
#: separations and blurs precisely the short ones that decide the early merges.
#:
#: The estimator's relative standard deviation is sqrt(2 / dims), so 128 columns
#: hold it near 13%. The width is close to free: on a 17k-bus mesh, widening
#: 32 -> 256 cost 29% more wall time, because the tree build is dominated by its
#: per-merge Python loop rather than by the feature width.
TARGET_COUNT_FEATURE_DIMS = 128

#: Cluster spread is reported for diagnostics, never enforced, so an exact
#: all-pairs diameter is not worth its quadratic cost on the biggest clusters.
SPREAD_SAMPLE_CAP = 64

#: Deepest transformer chain to resolve when collapsing transformers away.
#: The TAMU network needs 4; the cap only guards against pathological data.
MAX_TRAFO_CHAIN_DEPTH = 16

#: Bus column naming the zone no topology reduction may merge across. Every
#: downstream ReEDS-facing constraint (RPS, interzonal transfer limits, capacity
#: credit) is written per zone, so a bus that migrates into a neighbouring zone
#: -- or a cluster straddling two of them -- silently moves load and generation
#: between the accounting buckets those constraints police. Both the low-degree
#: reduction and the target-count clustering therefore refuse to cross it, which
#: also guarantees every zone keeps at least one bus. Networks built on a
#: boundary that drops the column (``state``) simply lose the guard.
PROTECTED_ZONE_COLUMN = "reeds_zone"

#: Label standing in for a bus whose :data:`PROTECTED_ZONE_COLUMN` is missing.
#: Such buses group together rather than each becoming its own zone, so absent
#: zone data relaxes the guard instead of freezing the reduction.
MISSING_ZONE_LABEL = "__missing_zone__"


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

    ``line_length_factor`` is the configured routing factor, not a local constant: it
    sets the aggregated *Link* lengths, which pypsa rebuilds as the crow-fly distance
    between the clustered buses times this factor, so it must be the same one
    ``assign_line_length`` used for Line and Link lengths to stay on one convention.
    Lines no longer consume it -- they take an ``s_nom``-weighted mean of the lengths
    of the circuits they replace, and their ``x``/``r``/``capital_cost`` are combined
    without any length rescaling.
    """
    logger.info("Aggregating buses to substation level...")

    generator_strategies = apply_wind_solar_cf_aggregation_weights(
        network,
        aggregation_strategies.get("generators", dict()),
    )
    one_port_strategies = aggregation_strategies.get("one_ports", dict())
    line_strategies = aggregation_strategies.get("lines", dict())

    clustering = get_clustering_from_busmap(
        network,
        busmap,
        aggregate_generators_weighted=True,
        aggregate_one_ports=["Load"],
        line_length_factor=line_length_factor,
        line_strategies=line_strategies,
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


def bus_zone_labels(n: pypsa.Network) -> pd.Series | None:
    """Zone label per bus for the no-crossing guard, or ``None`` if unavailable.

    Buses with no :data:`PROTECTED_ZONE_COLUMN` value share
    :data:`MISSING_ZONE_LABEL` so they can still merge with one another.
    """
    if PROTECTED_ZONE_COLUMN not in n.buses.columns:
        return None
    labels = n.buses[PROTECTED_ZONE_COLUMN].astype("string")
    return labels.where(labels.notna() & (labels != ""), MISSING_ZONE_LABEL).astype(str)


class MergeHeterogeneity(NamedTuple):
    """How unlike the members of each merged group were, as one scalar.

    ``ratio`` is ``sum_g(w_g * cv_g) / sum_g(w_g)``, where ``cv_g`` is the
    group's own ``std_g * n_g / sum_i value_i`` -- the dispersion inside the
    group, scaled by its size, over what it pooled. It is therefore a mean
    coefficient of variation over the merged groups: dimensionless and
    scale-free (doubling every rating leaves it unchanged), 0 when every group
    pooled identical members, and growing as merges pool increasingly unlike
    things.

    ``w_g`` is the MW at stake in that merge -- the summed ``s_nom`` of its
    members -- which is both the right emphasis and the currency the loss is
    denominated in: what a merge discards is transfer capability, and a wildly
    mismatched merge of two tiny stubs matters less than a mismatched merge on
    the transfer backbone. Where the pooled quantity already *is* ``s_nom`` (the
    series metric) this is just each group's share of the total, and ``ratio``
    collapses to ``sum_g(std_g * n_g) / sum_g(sum_i value_i)``. The parallel
    metric measures dispersion in ``x * s_nom`` but takes its weights
    separately, so both ratios weight a group by its capacity rather than by
    whatever each happens to measure dispersion in, and the two stay comparable.

    ``std`` is the population standard deviation (``ddof=0``). The sample
    variant is undefined for a group of one and inflates the very common
    two-member group by ``sqrt(2)`` for no reason; these groups are complete
    populations, not samples drawn from anything.
    """

    ratio: float
    groups: int
    members: int
    coefficients_of_variation: list[float]


def merge_heterogeneity(groups: list, weights: list | None = None) -> MergeHeterogeneity:
    """Fold per-group member values into a :class:`MergeHeterogeneity`.

    Each entry of ``groups`` holds the values one merge pooled -- the ``s_nom``
    of every segment in a collapsed series string, or the ``x * s_nom`` of every
    branch in a merged parallel bundle. Groups of fewer than two finite members
    merged nothing and are skipped, as are groups whose values sum to zero and
    so have no coefficient of variation.

    ``weights``, when given, holds one member-aligned array per group carrying
    the MW behind each pooled value; a group weighs the ``s_nom`` of exactly the
    members that survived the finite-value mask. Omitted, each group weighs its
    own summed values, which is the same thing whenever the pooled quantity is
    already ``s_nom``.
    """
    numerator = denominator = 0.0
    group_count = member_count = 0
    coefficients: list[float] = []
    for index, values in enumerate(groups):
        array = np.asarray(values, dtype=float)
        finite = np.isfinite(array)
        array = array[finite]
        if array.size < 2:
            continue
        total = float(array.sum())
        if total <= 0:
            continue
        deviation = float(array.std())
        coefficient = deviation * array.size / total
        weight = (
            total if weights is None else float(np.asarray(weights[index], dtype=float)[finite].sum())
        )
        numerator += weight * coefficient
        denominator += weight
        group_count += 1
        member_count += array.size
        coefficients.append(coefficient)
    ratio = numerator / denominator if denominator > 0 else float("nan")
    return MergeHeterogeneity(ratio, group_count, member_count, coefficients)


def summarize_distribution(values: list[float], fmt: str = "%.4g") -> str:
    """One-line ``n / mean / median / p90 / max`` summary for a log record.

    The per-event records behind these summaries run to five figures on the
    full network, so they go out at DEBUG and only this digest is logged at
    INFO.
    """
    if not values:
        return "n=0"
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return "n=0 (no finite samples)"
    parts = [
        ("mean", float(array.mean())),
        ("median", float(np.median(array))),
        ("p90", float(np.quantile(array, 0.9))),
        ("max", float(array.max())),
    ]
    return f"n={array.size}, " + ", ".join(f"{name}=" + fmt % value for name, value in parts)


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


def merge_parallel_lines(
    lines: pd.DataFrame,
    capacity_cols: list[str],
    group_stats: list | None = None,
) -> tuple[pd.DataFrame, int]:
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

    ``group_stats``, when given, collects one record per merged group for the
    caller's diagnostics: the branch names, each branch's ``s_nom`` (the weight
    the caller's ratio gives it), and for each branch its saturation angle
    ``x * s_nom`` -- the angle difference a branch carrying its own rating
    stands under, since DC flow puts ``x_i * s_nom_i`` across branch ``i`` at
    ``s_nom_i``. That is the quantity the merge is really pooling, because the
    ``min(capacity_i * x_i)`` above is exactly the smallest of these: a group
    whose branches share one saturation angle ``c`` saturates as one body and
    merges losslessly -- the formula returns ``c / x_parallel = sum(s_nom_i)``,
    the full sum of the ratings -- while a wide spread means the narrowest-angle
    branch caps the corridor and the headroom on the rest is written off. (The
    conductance-like ``s_nom / x`` is *not* this quantity and says nothing about
    the loss: two branches with equal ``s_nom / x`` but unequal ``x`` merge to
    half their summed rating.)

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

        if group_stats is not None and "s_nom" in capacity_cols:
            caps = rows["s_nom"].to_numpy(dtype=float)
            group_stats.append(
                {
                    "names": names,
                    "s_nom": caps,
                    "x_s_nom": np.where(x > 0, caps * x, np.nan),
                },
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

    A bus is only eliminated when every neighbour absorbing it sits in the same
    :data:`PROTECTED_ZONE_COLUMN` as the bus itself. Elimination hands the bus's
    demand weight, one-port capacity and Links to its neighbours, so allowing it
    across a zone boundary would move load and generation between the accounting
    buckets every ReEDS-facing constraint downstream is written against. The
    guard keeps a bus wherever the reduction would otherwise cross, which also
    means no zone can be emptied. Merged Lines still span zones -- it is the bus
    contents, not the corridor, that must stay put. If the column is absent the
    guard is skipped and logged.

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

    Two scalars are logged alongside the counts, one per kind of merge, because
    both discard information no later stage can recover. Each is a
    :func:`merge_heterogeneity` ratio -- a dimensionless mean coefficient of
    variation over the merged groups, each group weighted by the ``s_nom`` at
    stake in it, that is 0 when every merge pooled identical members:

    * **series** -- over the ``s_nom`` of every segment in each collapsed
      *string*, not each collapsed pair. A chain is collapsed two segments at a
      time over as many passes as it is long, so provenance is carried on the
      surviving Line and the string is banked only once it can grow no further
      (it is drained into a parallel bundle, dropped as a stub, or the fixed
      point is reached). The merged corridor takes the string's *minimum*
      rating, so this ratio tracks the headroom written off on every wider
      segment. A segment that came out of a parallel merge counts once, at the
      bundle's pooled rating, which is what it now is.
    * **parallel** -- over the ``x * s_nom`` of every branch in each merged
      bundle (see :func:`merge_parallel_lines`), the saturation angle the merge
      is really pooling. 0 means every bundle's branches saturate together and
      the merged corridor is rated at exactly their summed ``s_nom``; the ratio
      grows as the narrowest-angle branch caps more of the bundle. Dispersion is
      measured in saturation angle, but a bundle still weighs its summed
      ``s_nom``, so this ratio and the series one weight groups alike.

    Only the two ratios are logged, with the distribution of their per-group
    contributions for context; the per-merge records behind them run to tens of
    thousands on the full network and are not emitted.

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

    # Diagnostics: what each merge pooled. See the docstring.
    #
    # A series string is collapsed two segments at a time, over as many passes as
    # it is long, so the pairs seen inside the loop are not the strings the metric
    # is about. `segment_ratings` carries each surviving Line's provenance -- the
    # rating of every segment folded into it so far -- and a string is banked in
    # `collapsed_strings` once it can grow no further.
    segment_ratings: dict[str, list[float]] = {}
    collapsed_strings: list[list[float]] = []
    parallel_bundles: list[np.ndarray] = []
    parallel_weights: list[np.ndarray] = []
    track_series = "s_nom" in min_cols

    def close_string(name: str) -> None:
        """Bank a Line's series string; it is about to stop being extendable."""
        ratings = segment_ratings.pop(name, None)
        if ratings is None or len(ratings) < 2:
            return
        collapsed_strings.append(ratings)

    zones = bus_zone_labels(n)
    if zones is None:
        logger.warning(
            "Buses carry no '%s' column; low-degree reduction runs without the "
            "zone-crossing guard and may move demand between zones.",
            PROTECTED_ZONE_COLUMN,
        )
        bus_zone: dict = {}
    else:
        bus_zone = zones.to_dict()
    blocked_by_zone = 0

    while True:
        lines = n.lines
        self_loops = lines.index[lines.bus0 == lines.bus1]
        if len(self_loops):
            n.lines = lines.drop(index=self_loops)
            loops_removed += len(self_loops)
            for name in self_loops:
                close_string(name)
            continue

        parallel_groups: list = []
        lines, n_parallel_groups = merge_parallel_lines(lines, min_cols, parallel_groups)
        if n_parallel_groups:
            n.lines = lines
            parallel_merged += n_parallel_groups
            for group in parallel_groups:
                # Every branch in the bundle stops being extendable as a series
                # string here; the bundle becomes one fresh segment rated by the
                # parallel merge, which is what a later series merge should see.
                for name in group["names"]:
                    close_string(name)

                finite = np.isfinite(group["x_s_nom"])
                if finite.sum() > 1:
                    parallel_bundles.append(group["x_s_nom"][finite])
                    parallel_weights.append(group["s_nom"][finite])
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

            # The bus's contents are about to be handed to every neighbour, so
            # every neighbour has to sit in the bus's own zone. See the docstring.
            if bus in bus_zone and any(bus_zone.get(b) != bus_zone[bus] for b in neighbours):
                blocked_by_zone += 1
                continue

            if len(edges) == 1:
                l1, b1 = edges[0]
                hand_over(bus, [(b1, 1.0)])
                assignments[bus] = [(b1, 1.0)]
                drop_lines.append(l1)
                consumed.add(bus)
                stubs_removed += 1
                close_string(l1)
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

            if track_series:
                # Concatenate provenance rather than the two current ratings: a
                # segment that is itself a collapsed string contributes all of
                # its own segments, so the banked string spans the whole chain.
                # A segment that came out of a parallel merge has no entry and
                # counts once, at the bundle's pooled rating -- which is what it
                # now is, one segment of this string.
                segment_ratings[min(l1, l2)] = segment_ratings.pop(
                    l1, [float(row1["s_nom"])],
                ) + segment_ratings.pop(l2, [float(row2["s_nom"])])

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
    # Any string still standing at the fixed point is as long as it will get.
    for name in list(segment_ratings):
        close_string(name)

    series = merge_heterogeneity(collapsed_strings)
    parallel = merge_heterogeneity(parallel_bundles, parallel_weights)
    logger.info(
        "Series s_nom heterogeneity = %.4f -- sum(std(s_nom) * segments) / sum(s_nom) "
        "over %s collapsed strings spanning %s segments. 0 means every string pooled "
        "equally rated segments and lost no headroom to its narrowest one. "
        "Per-string contribution: %s.",
        series.ratio,
        series.groups,
        series.members,
        summarize_distribution(series.coefficients_of_variation, "%.4f"),
    )
    logger.info(
        "Parallel x*s_nom heterogeneity = %.4f -- s_nom-weighted mean of "
        "std(x*s_nom) * branches / sum(x*s_nom) over %s merged bundles spanning "
        "%s branches. 0 means every bundle pooled branches of equal saturation "
        "angle, which merge losslessly into their summed s_nom. Per-bundle "
        "contribution: %s.",
        parallel.ratio,
        parallel.groups,
        parallel.members,
        summarize_distribution(parallel.coefficients_of_variation, "%.4f"),
    )
    if zones is not None:
        surviving_zones = bus_zone_labels(n)
        logger.info(
            "Zone-crossing guard on '%s': %s reductions blocked; %s of %s zones still "
            "hold at least one bus.",
            PROTECTED_ZONE_COLUMN,
            blocked_by_zone,
            surviving_zones.nunique() if surviving_zones is not None else "?",
            zones.nunique(),
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
    *Link* lengths, which pypsa rebuilds as the crow-fly distance between the clustered
    buses times this factor. A default would let a caller silently pick a routing factor
    different from the one the lengths were built with. Lines ignore it: their length is
    an ``s_nom``-weighted mean over the circuits they replace, and their ``x``/``r``/
    ``capital_cost`` are combined without any length rescaling.
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


class ReactanceEmbedding(NamedTuple):
    """Laplacian embedding plus the exact edge set it was built from.

    ``bus0``/``bus1`` are positional indices into ``n.buses``, not names, and
    cover only the lines that actually entered the Laplacian. Callers needing a
    connectivity graph consistent with ``islands`` must build it from these two
    arrays -- deriving it from ``n.lines`` instead silently readmits the zero-
    and NaN-reactance lines dropped below, which would bridge buses that the
    island labels place in different connected components.
    """

    embedding: np.ndarray
    islands: np.ndarray
    bus0: np.ndarray
    bus1: np.ndarray


def _effective_reactance_embedding(
    n: pypsa.Network,
    n_probes: int,
    seed: int,
) -> ReactanceEmbedding:
    """Approximate effective-reactance distances using the S&S JL embedding.

    The squared embedding distance estimates the effective reactance
    ``(e_i-e_j)\' L^-1 (e_i-e_j)`` in ohm, where
    ``L = B diag(1/x) B\'``. It uses only AC-line topology and supplied series
    reactances, never a solved dispatch or base power flow. In the notation of
    Spielman and Srivastava (2011), it constructs the transpose of
    ``Q W**0.5 B L+`` with Rademacher ``Q``. The explicit ``1/sqrt(k)`` in
    ``Q`` is instead applied when callers average squared probe differences.
    The paper uses a near-linear approximate Laplacian solver; this routine
    uses direct sparse solves, so it introduces no additional solver error.
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
        empty = np.zeros(0, dtype=int)
        return ReactanceEmbedding(
            np.zeros((n_buses, n_probes)),
            np.arange(n_buses),
            empty,
            empty,
        )

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
    # Let ``S`` be this matrix of independent +/-1 signs. The paper's random
    # projection is ``Q = S.T / sqrt(n_probes)``. Solving the unnormalised
    # transpose here and averaging squared coordinate differences downstream
    # is exactly equivalent, while also making a selected subset of columns a
    # correctly normalised Rademacher projection in its own right.
    signs = 2.0 * rng.integers(
        0,
        2,
        size=(n_lines, n_probes),
        dtype=np.int8,
    ) - 1.0
    rhs = incidence @ (np.sqrt(1 / x)[:, None] * signs)
    embedding = np.zeros((n_buses, n_probes))
    for island in np.unique(islands):
        members = np.flatnonzero(islands == island)
        if len(members) > 1:
            reduced = laplacian[members][:, members][1:, 1:].tocsc()
            embedding[members[1:]] = splu(reduced).solve(rhs[members[1:]])
    return ReactanceEmbedding(embedding, islands, bus0, bus1)


def _earth_centered_km(buses: pd.DataFrame) -> np.ndarray:
    """Return Earth-centred Cartesian bus coordinates in kilometres.

    Euclidean distances in this three-dimensional feature space are spherical
    chord lengths, ``2 R sin(d_gc / (2 R))``, where ``d_gc`` is the great-circle
    distance. They are a globally well-behaved, monotonic approximation to arc
    length and avoid the latitude- and extent-dependent distortion of a local
    two-dimensional projection. A standard Euclidean Ward objective cannot
    represent great-circle arc lengths exactly, whereas these coordinates retain
    the correct Earth geometry with only three features.
    """
    latitude = np.radians(buses.y.to_numpy(float))
    longitude = np.radians(buses.x.to_numpy(float))
    return EARTH_RADIUS_KM * np.c_[
        np.cos(latitude) * np.cos(longitude),
        np.cos(latitude) * np.sin(longitude),
        np.sin(latitude),
    ]


def _calibrate_feature_scales(
    embedding: np.ndarray,
    bus0: np.ndarray,
    bus1: np.ndarray,
    geographic_km: np.ndarray,
) -> tuple[float, float]:
    """Return ``(km_ref, ohm_ref)`` read off the network's own typical line.

    Merges only ever happen between adjacent buses, so the scale that matters
    is the one spanned by a single line, not the diameter of the network. Both
    references are medians over the lines that entered the Laplacian, which
    makes them robust to a handful of very long or very reactive corridors and
    removes the need to hand-match two thresholds carrying different units.

    Calibration is deliberately global rather than per region: the tree cut
    downstream compares merge costs across regions, and per-region references
    would make those costs incommensurable.
    """
    if len(bus0) == 0:
        logger.warning(
            "Target-count clustering found no usable lines; falling back to "
            "unit feature scales.",
        )
        return 1.0, 1.0

    ohm_ref = float(np.median(((embedding[bus0] - embedding[bus1]) ** 2).mean(axis=1)))
    km_ref = float(
        np.median(np.linalg.norm(geographic_km[bus0] - geographic_km[bus1], axis=1)),
    )
    if not np.isfinite(ohm_ref) or ohm_ref <= 0:
        logger.warning("Median line effective reactance is %s; using 1 ohm.", ohm_ref)
        ohm_ref = 1.0
    if not np.isfinite(km_ref) or km_ref <= 0:
        logger.warning("Median line length is %s km; using 1 km.", km_ref)
        km_ref = 1.0
    return km_ref, ohm_ref


def _intra_boundary_components(
    n_buses: int,
    bus0: np.ndarray,
    bus1: np.ndarray,
    boundary: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Label the connected components of the line graph with region edges cut.

    Each label is therefore a connected subgraph lying wholly inside one
    topological region, which is exactly the unit that may be clustered. No
    later step can produce a group straddling a region or a synchronous
    island, and every cluster is guaranteed to be a connected subgraph -- a
    property the threshold-based clique cover never offered.
    """
    keep = boundary[bus0] == boundary[bus1]
    graph = sparse.coo_matrix(
        (np.ones(int(keep.sum())), (bus0[keep], bus1[keep])),
        shape=(n_buses, n_buses),
    )
    return csgraph.connected_components(graph, directed=False)


def _dsu_find(leaders: np.ndarray, node: int) -> int:
    """Union-find lookup with path compression, iterative to stay stack-safe."""
    root = node
    while leaders[root] != root:
        root = leaders[root]
    while leaders[node] != root:
        leaders[node], node = root, leaders[node]
    return root


def _cluster_spread(
    members_by_cluster: list[np.ndarray],
    geographic_km: np.ndarray,
    embedding: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster geographic (km) and effective-reactance (ohm) diameters."""
    geographic, electrical = [], []
    for members in members_by_cluster:
        if len(members) < 2:
            continue
        if len(members) > SPREAD_SAMPLE_CAP:
            members = rng.choice(members, SPREAD_SAMPLE_CAP, replace=False)
        points = geographic_km[members]
        geographic.append(
            np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1).max(),
        )
        probes = embedding[members]
        electrical.append(
            ((probes[:, None, :] - probes[None, :, :]) ** 2).mean(axis=-1).max(),
        )
    return np.array(geographic), np.array(electrical)


def busmap_by_target_bus_count(
    n: pypsa.Network,
    n_clusters: int,
    lambda_electrical: float = 1.0,
    n_probes: int = 256,
    n_feature_dims: int = TARGET_COUNT_FEATURE_DIMS,
    seed: int = 123,
    topological_boundary: str | None = None,
) -> pd.Series:
    """Return a busmap with exactly ``n_clusters`` groups, ranked by merge cost.

    The caller names the cluster count and the data supplies the distance
    thresholds. Buses are embedded in one feature space whose squared distance
    is

        ``lambda_electrical * X_eff / ohm_ref  +  d_geo^2 / km_ref^2``

    with both references calibrated off the network's own typical line, so the
    single remaining knob is the dimensionless ``lambda_electrical`` (1.0
    weights the two equally at that scale). Ward linkage under a connectivity
    constraint then builds one merge tree per connected intra-region subgraph,
    and a single cut taken globally across all trees -- always accepting the
    cheapest merge available next -- spends the cluster budget where merging
    costs least. Dense regions get compressed hard, sparse ones keep their
    detail, and no one has to apportion clusters between regions by hand.

    No cluster spans two :data:`PROTECTED_ZONE_COLUMN` zones: the zone boundary
    is cut alongside ``topological_boundary``, so every zone keeps at least one
    bus and no load or generation moves between zones. With a ``reeds_zone``
    topological boundary the two cuts are the same one.

    ``n_clusters`` is the count this pass produces, and the last reduction the
    pipeline applies -- it is the bus count of the exported network.
    """
    required = {"x", "y"}
    if topological_boundary is not None:
        required.add(topological_boundary)
    missing = required.difference(n.buses.columns)
    if missing:
        raise ValueError(
            "Target-count clustering requires bus columns "
            f"{sorted(required)}; missing {sorted(missing)}.",
        )
    if n_clusters < 1:
        raise ValueError("Target-count clustering requires at least one cluster.")
    if lambda_electrical <= 0:
        raise ValueError("lambda_electrical must be strictly positive.")
    if n_probes < 2:
        raise ValueError("Target-count clustering requires at least two probes.")
    if not 1 <= n_feature_dims <= n_probes:
        raise ValueError(
            f"n_feature_dims must lie in [1, n_probes={n_probes}]; got {n_feature_dims}.",
        )

    buses = n.buses
    if buses[["x", "y"]].isna().any().any():
        raise ValueError("Target-count clustering requires coordinates for every bus.")
    n_buses = len(buses)
    if n_clusters >= n_buses:
        logger.info(
            "Target-count clustering asked for %s clusters on %s buses; nothing to do.",
            n_clusters,
            n_buses,
        )
        return pd.Series(buses.index.astype(str), index=buses.index, name="busmap")

    embedding, _, bus0, bus1 = _effective_reactance_embedding(n, n_probes, seed)

    if topological_boundary is None:
        boundary = pd.Series("", index=buses.index, dtype="string")
    else:
        boundary = buses[topological_boundary].astype("string")
        boundary = boundary.where(
            boundary.notna(),
            pd.Series(buses.index.astype(str), index=buses.index).radd("__missing__"),
        )

    # Cut the reeds_zone boundary as well as the configured one, so no cluster
    # spans two zones and every zone keeps at least one bus. This is free when
    # the two coincide -- when `reeds_zone` is itself the configured topological
    # boundary -- and it is the only thing holding the line for coarser configured
    # boundaries such as `trans_reg`, `county`, or `state`. A cluster straddling
    # zones would pool load and generation across the buckets the downstream
    # ReEDS constraints are written against.
    zones = bus_zone_labels(n)
    if zones is None:
        logger.warning(
            "Buses carry no '%s' column; target-count clustering runs without the "
            "zone-crossing guard and may produce clusters spanning zones.",
            PROTECTED_ZONE_COLUMN,
        )
    elif topological_boundary != PROTECTED_ZONE_COLUMN:
        # Factorised pair rather than a concatenated string: no separator can be
        # mistaken for one that occurs inside a region or zone name.
        pair = pd.MultiIndex.from_arrays([boundary.astype(str), zones.astype(str)])
        boundary = pd.Series(
            pd.factorize(pair)[0].astype(str),
            index=buses.index,
            dtype="string",
        )

    geographic_km = _earth_centered_km(buses)
    km_ref, ohm_ref = _calibrate_feature_scales(embedding, bus0, bus1, geographic_km)

    # A subsample of the probes, not a projection of them -- see the comment on
    # TARGET_COUNT_FEATURE_DIMS. Sorted so the feature layout is reproducible.
    rng = np.random.default_rng(seed)
    probes = np.sort(rng.choice(n_probes, n_feature_dims, replace=False))
    feature = np.ascontiguousarray(
        np.c_[
            embedding[:, probes] * np.sqrt(lambda_electrical / (n_feature_dims * ohm_ref)),
            geographic_km / km_ref,
        ],
        dtype=np.float64,
    )

    n_components, components = _intra_boundary_components(
        n_buses,
        bus0,
        bus1,
        boundary.to_numpy(),
    )
    if n_clusters < n_components:
        raise ValueError(
            f"Target-count clustering cannot produce {n_clusters} clusters: the "
            f"network splits into {n_components} connected subgraphs once "
            f"{topological_boundary or 'no'} region boundaries are cut, and each "
            "must yield at least one cluster. Raise the target or coarsen the "
            "topological boundary.",
        )

    labels = np.arange(n_components)
    order = np.argsort(components, kind="stable")
    starts = np.searchsorted(components[order], labels)
    ends = np.searchsorted(components[order], labels, side="right")
    position_in_component = np.empty(n_buses, dtype=int)
    position_in_component[order] = np.arange(n_buses) - starts[components[order]]

    # Bucket the edges by component once. Rescanning the full edge list inside
    # the per-component loop below would be quadratic in the component count,
    # and cutting a fine topological boundary leaves thousands of them.
    edge_component = np.where(
        components[bus0] == components[bus1], components[bus0], -1,
    )
    edge_order = np.argsort(edge_component, kind="stable")
    edge_starts = np.searchsorted(edge_component[edge_order], labels)
    edge_ends = np.searchsorted(edge_component[edge_order], labels, side="right")

    # One Ward tree per component. ``representative`` carries, for every tree
    # node, one leaf standing for it, so accepted merges replay into a flat
    # union-find over buses without materialising cluster membership lists.
    trees: list[dict] = []
    frontier: list[tuple[float, int, int]] = []
    leaders = np.arange(n_buses)
    for component in range(n_components):
        positions = order[starts[component] : ends[component]]
        if len(positions) < 2:
            continue
        inside = edge_order[edge_starts[component] : edge_ends[component]]
        rows = position_in_component[bus0[inside]]
        cols = position_in_component[bus1[inside]]
        connectivity = sparse.coo_matrix(
            (np.ones(2 * len(rows)), (np.r_[rows, cols], np.r_[cols, rows])),
            shape=(len(positions), len(positions)),
        ).tocsr()
        children, connected, n_leaves, _, distances = ward_tree(
            feature[positions],
            connectivity=connectivity,
            return_distance=True,
        )
        if connected != 1:
            raise RuntimeError(
                f"Component {component} was built as a connected subgraph but "
                f"sklearn reports {connected} components; sklearn would bridge "
                "them and emit clusters spanning disconnected buses.",
            )
        trees.append(
            {
                "positions": positions,
                "children": children,
                # A connectivity constraint can make a later merge cheaper than
                # an earlier one, so raw Ward distances are not always monotone
                # here. The running maximum restores monotonicity without
                # reordering a tree's own merges, which must stay in sequence:
                # a merge's operands exist only once its predecessors are in.
                "keys": np.maximum.accumulate(distances),
                "representative": np.arange(2 * n_leaves - 1),
                "n_leaves": n_leaves,
            },
        )
        heappush(frontier, (float(trees[-1]["keys"][0]), len(trees) - 1, 0))

    accepted = 0
    while frontier and n_buses - accepted > n_clusters:
        _, tree_index, merge_index = heappop(frontier)
        tree = trees[tree_index]
        left, right = tree["children"][merge_index]
        representative, positions = tree["representative"], tree["positions"]
        root = _dsu_find(leaders, positions[representative[left]])
        leaders[_dsu_find(leaders, positions[representative[right]])] = root
        representative[tree["n_leaves"] + merge_index] = representative[left]
        accepted += 1
        following = merge_index + 1
        if following < len(tree["keys"]):
            heappush(frontier, (float(tree["keys"][following]), tree_index, following))

    roots = np.array([_dsu_find(leaders, bus) for bus in range(n_buses)])
    codes = pd.factorize(roots, sort=True)[0]
    busmap = pd.Series(
        [f"electrical_{code}" for code in codes],
        index=buses.index,
        name="busmap",
    )

    if zones is not None:
        zones_per_cluster = zones.groupby(busmap).nunique()
        if (zones_per_cluster > 1).any():
            raise RuntimeError(
                f"{int((zones_per_cluster > 1).sum())} target-count clusters span more "
                f"than one {PROTECTED_ZONE_COLUMN}; the zone boundary was supposed to "
                "be cut.",
            )
        logger.info(
            "Zone-crossing guard on '%s': %s zones in, %s zones still represented by "
            "the %s clusters out.",
            PROTECTED_ZONE_COLUMN,
            zones.nunique(),
            zones.groupby(busmap).first().nunique(),
            busmap.nunique(),
        )

    logger.info(
        "Target-count clustering: %s buses -> %s clusters over %s connected "
        "intra-region subgraphs; calibrated km_ref=%.3f km, ohm_ref=%.4f ohm, "
        "lambda=%.3f, %s of %s probes as features, seed=%s, boundary=%s.",
        n_buses,
        busmap.nunique(),
        n_components,
        km_ref,
        ohm_ref,
        lambda_electrical,
        n_feature_dims,
        n_probes,
        seed,
        topological_boundary or "none",
    )
    geographic_spread, electrical_spread = _cluster_spread(
        [np.flatnonzero(codes == code) for code in range(codes.max() + 1)],
        geographic_km,
        embedding,
        rng,
    )
    for name, spread, unit in (
        ("geographic", geographic_spread, "km"),
        ("effective-reactance", electrical_spread, "ohm"),
    ):
        if not len(spread):
            continue
        logger.info(
            "Cluster %s diameter over %s multi-bus clusters (%s): "
            "P50=%.3f P90=%.3f P99=%.3f max=%.3f.",
            name,
            len(spread),
            unit,
            *np.percentile(spread, [50, 90, 99]),
            spread.max(),
        )
    return busmap




def identity_busmap(n: pypsa.Network) -> pd.Series:
    """The busmap of a disabled stage: every bus maps to itself.

    Each reduction stage contributes its busmap to the region dissolve and to
    the exported ``busmap.csv``, so a stage that is switched off has to hand
    back the neutral element rather than drop out of the composition.
    """
    return pd.Series(n.buses.index, index=n.buses.index, name="busmap")



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

        snakemake = mock_snakemake("simplify_network", case="test")
    configure_logging(snakemake)
    params = snakemake.params
    topological_boundaries = snakemake.params.topological_boundaries
    low_degree_reduction = bool(getattr(params, "low_degree_reduction", True))

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

    if low_degree_reduction:
        n, reduction_busmap = reduce_low_degree_buses_and_merge_parallel_lines(n)
    else:
        logger.info("Low-degree reduction disabled; leaving degree-1/2 buses in place.")
        reduction_busmap = identity_busmap(n)
    plot_network_topology(
        n,
        snakemake.output.network_map_after_low_degree,
        snakemake.wildcards,
        "After low-degree reduction" if low_degree_reduction else "Low-degree reduction disabled",
        snakemake.input.state_boundaries,
        topology_central_longitude,
        topology_line_width_reference,
    )

    if topological_boundaries in ["reeds_zone", "state"] and "county" in n.buses.columns:
        n.buses = n.buses.drop(columns=["county"])

    target = params.target_count
    if target.get("enable", True):
        busmap = busmap_by_target_bus_count(
            n,
            n_clusters=int(target["n_clusters"]),
            lambda_electrical=float(target.get("lambda_electrical", 1.0)),
            n_probes=int(target.get("n_probes", 256)),
            n_feature_dims=int(target.get("n_feature_dims", TARGET_COUNT_FEATURE_DIMS)),
            seed=int(target.get("seed", 123)),
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
        target_busmap = clustering.busmap
    else:
        logger.info("Target-count clustering disabled; bus count left untouched.")
        target_busmap = identity_busmap(n)
    update_p_nom_max(n)
    if "land_region" in n.generators.columns:
        n.generators["land_region"] = n.generators.land_region.fillna(n.generators.bus)
    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    register_topology_carriers(n)
    # Target-count clustering is the last reduction the pipeline applies, so this
    # is the exported network. A second low-degree pass used to run here to clear
    # the degree-1/2 buses that clustering leaves behind; it is gone deliberately.
    # It reduced across cluster boundaries the target-count pass had just drawn,
    # undershooting the configured `n_clusters` by an amount nobody could predict
    # from the config, and its series merges kept collapsing exactly the corridors
    # clustering had chosen to keep distinct.
    plot_network_topology(
        n,
        snakemake.output.network_map,
        snakemake.wildcards,
        "After target-bus-count aggregation"
        if target.get("enable", True)
        else "Target-bus-count aggregation disabled",
        snakemake.input.state_boundaries,
        topology_central_longitude,
        topology_line_width_reference,
    )

    # The Voronoi regions are keyed by substations.  Compose both maps -- the
    # low-degree pass and the target-count clustering -- before dissolving them
    # into final regions and bus labels.
    busmaps = (reduction_busmap, target_busmap)
    cluster_regions(busmaps, snakemake.input, snakemake.output)

    # Anything still keyed by the raw substation ids -- the regions above, and
    # supply curves that `add_electricity` joins on `sub_id` -- needs to know
    # which surviving bus now stands for each one. The bus index at this point
    # *is* the substation id, so composing the maps gives exactly that.
    composed = reduce(lambda left, right: left.map(right), busmaps[1:], busmaps[0])
    composed.rename_axis("sub_id").rename("bus_id").to_csv(snakemake.output.busmap)

    # No Line-level counterpart is published. `clustering.linemap` covers only the
    # target-count pass, so its index is intermediate Line names that match
    # nothing a caller holds, while merged Lines inherit the first of their names --
    # which makes a final name coincide with an unrelated original name often enough
    # (17k of 20k on the USA network) that joining the two would silently succeed and
    # be wrong. Tracing an original Line to its corridor needs name bookkeeping
    # through the low-degree pass as well; nothing consumes it today.

    # Substation aggregation, the low-degree reduction and target-count clustering
    # all move Link endpoints, so the distance their cost was priced on is
    # stale by now. Lines need no equivalent step: their `capital_cost` is strictly
    # proportional to `length`, which every aggregation rule above preserves.
    recompute_link_transmission_costs(n)
    # These are internal bookkeeping for recompute_link_transmission_costs --
    # nothing downstream of this rule needs them, so they don't belong in the
    # exported network.
    n.links.drop(columns=[LINK_UNIT_COST_COL, LINK_FIXED_COST_COL], errors="ignore", inplace=True)

    n.consistency_check()
    n.export_to_netcdf(snakemake.output.network)
