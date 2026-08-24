"""
Representative-period selection: everything in one place.

This module owns the whole representative-period story:

* the tsam hierarchical-clustering selection itself, run on a small national-aggregate feature
  frame (capacity-weighted mean wind CF, capacity-weighted mean solar CF, total
  AC demand) computed straight from the raw ReEDS supply curves / CF tables and
  the EER demand h5;
* the snapshot definition it writes out (``snapshots.csv``, ``metadata.json``,
  ``profiles.png``) and the readers downstream rules use to consume it;
* the diagnostic profile plots.

It runs at the very top of the electricity workflow -- *before*
``build_renewable_profiles`` -- so the selected hours are known before any
per-bus profile or network exists. ``build_renewable_profiles`` then builds only
those hours, and every downstream rule attaches time series for those hours only,
so the full 15-weather-year x 8760 h timeline is never materialised anywhere.

The wind/solar features are aggregated over **every** site in the ReEDS supply
curve with no spatial aggregation and no region filter, so this step depends on
nothing but the raw data files.

Two pieces of the older in-network implementation are intentionally gone: the
"force one spring + one fall representative period" seasonal constraint, and the
post-hoc rescaling of non-extreme snapshots to match true annual energy totals.
Representative periods are used exactly as tsam selects them.

Extreme periods and weighting
-----------------------------
Extreme periods are tsam's job: they are requested through
``extremePeriodMethod="new_cluster_center"`` with ``addMeanMin`` / ``addMeanMax``
pointed straight at the clustering features. tsam then picks the period with the
lowest/highest period mean per selector, adds it as an *additional cluster
center*, and re-checks every other period against it -- a period closer to the
extreme than to its own medoid joins the extreme's cluster. Weights therefore
fall out of the cluster membership counts for representative and extreme periods
alike: an extreme week represents all the weeks that look like it, not only its
own hours. Snapshot weightings are finally rescaled so ``objective`` sums to
8760 h per planning horizon.

Because both kinds of period come out of one tsam run over one period partition,
extreme and representative periods necessarily share a length: the single
``period_length`` config key (see ``get_period_hours``). Note also that tsam
*drops* a requested extreme period that is already a cluster center rather than
falling back to the next-most-extreme candidate, so a run can legitimately return
fewer periods than ``number + len(include_extreme)``; ``validate_period_counts``
warns about it.

Timezone
--------
ReEDS capacity factors are UTC; the EER demand h5 is US Central Standard
Time (UTC-06:00). ``build_eer_demand.ReadEer`` performs the CST -> UTC roll
inside each 8760-hour block, which is why the demand series is read through that
class rather than from the h5 directly. Feeding a CST series into the feature
frame would shift the load feature six hours out of phase with the renewable
features and silently corrupt both the clustering and the ``demand_max`` extreme
window; ``build_national_feature_frame`` asserts index equality as a backstop.

Snapshot labels
---------------
Representative snapshots carry *synthetic* contiguous timestamps, so a block
sourced from December can be labelled with January dates. Never derive a season
or calendar date from a representative snapshot label -- use the
``source_timestep`` recorded alongside it.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from _helpers import configure_logging, get_weather_year_snapshots

logger = logging.getLogger(__name__)

# Renewable carriers aggregated into each national clustering feature. Carriers in
# the same group are combined by capacity into one mean-capacity-factor series.
WIND_CARRIERS = ("onwind", "offwind", "offwind_floating")
SOLAR_CARRIERS = ("solar",)
REEDS_FEATURE_GROUPS = {"wind": WIND_CARRIERS, "solar": SOLAR_CARRIERS}

_EXTREME_PERIOD_DEFAULTS = {
    "solar": "solar_min",
    "wind": "wind_min",
    "demand": "demand_max",
}
# Each extreme selector names the clustering feature tsam ranks periods on, and
# whether the lowest (addMeanMin) or highest (addMeanMax) period mean is wanted.
WIND_FEATURE = ("Generator", "p_max_pu", "wind")
SOLAR_FEATURE = ("Generator", "p_max_pu", "solar")
LOAD_FEATURE = ("Load", "p_set", "ac_load")
_EXTREME_PERIOD_SELECTORS = {
    "solar_min": {"feature": SOLAR_FEATURE, "mode": "min"},
    "solar_max": {"feature": SOLAR_FEATURE, "mode": "max"},
    "wind_min": {"feature": WIND_FEATURE, "mode": "min"},
    "wind_max": {"feature": WIND_FEATURE, "mode": "max"},
    "demand_min": {"feature": LOAD_FEATURE, "mode": "min"},
    "demand_max": {"feature": LOAD_FEATURE, "mode": "max"},
}


# Superseded by the single ``period_length``; a config still carrying one of these
# is rejected rather than silently falling back to a default.
_RETIRED_PERIOD_LENGTH_KEYS = ("representative_period_length", "extreme_period_length")


def get_period_hours(representative_periods):
    """
    Return the period length in hours from ``representative_periods.period_length``.

    Representative and extreme periods come out of a single tsam run over a single
    period partition, so tsam's one ``hoursPerPeriod`` governs both and there is
    exactly one length to configure.
    """
    retired = [key for key in _RETIRED_PERIOD_LENGTH_KEYS if representative_periods.get(key) is not None]
    if retired:
        raise ValueError(
            "Retired config key(s) "
            + ", ".join(f"representative_periods.{key}" for key in retired)
            + ": representative and extreme periods are selected by the same tsam run and share "
            "one length. Configure representative_periods.period_length instead.",
        )

    period_length = representative_periods.get("period_length")
    if period_length is None:
        raise ValueError("representative_periods.period_length must be configured.")
    days = float(period_length)
    if days <= 0:
        raise ValueError("representative_periods.period_length must be positive.")
    return days * 24.0


def get_extreme_period_selectors(representative_periods):
    """Return normalized extreme-period selectors from config."""
    include_extreme = representative_periods.get("include_extreme", [])
    allowed_entries = list(_EXTREME_PERIOD_DEFAULTS) + list(_EXTREME_PERIOD_SELECTORS)

    if isinstance(include_extreme, bool):
        return list(_EXTREME_PERIOD_DEFAULTS.values()) if include_extreme else []
    if include_extreme is None:
        return []

    if isinstance(include_extreme, str):
        categories = [include_extreme]
    elif isinstance(include_extreme, (list, tuple, set)):
        categories = list(include_extreme)
    else:
        raise ValueError(
            "representative_periods.include_extreme must be a bool, string, or list drawn from "
            f"{allowed_entries}.",
        )

    normalized = []
    invalid = []
    for category in categories:
        category_str = str(category).strip().lower()
        if category_str in _EXTREME_PERIOD_DEFAULTS:
            category_str = _EXTREME_PERIOD_DEFAULTS[category_str]
        if category_str in _EXTREME_PERIOD_SELECTORS:
            if category_str not in normalized:
                normalized.append(category_str)
        else:
            invalid.append(category)

    if invalid:
        raise ValueError(
            f"Unsupported representative_periods.include_extreme entries {invalid}. "
            f"Choose from {allowed_entries}.",
        )
    return normalized


def build_extreme_period_selector_columns(feature_t, extreme_selectors):
    """
    Translate configured extreme selectors into tsam's ``addMeanMin``/``addMeanMax``.

    The selectors point at the clustering features themselves rather than at
    duplicated helper columns: tsam requires every ``addMean*`` entry to be a
    column of the time series it clusters, and a duplicate would double the weight
    of that feature in the clustering distance. One feature can appear in both
    lists (``demand_min`` plus ``demand_max``); tsam handles the two independently.

    A selector whose feature is missing -- no wind carrier configured, say -- is
    skipped with a warning instead of failing the run.
    """
    add_mean_min = []
    add_mean_max = []
    for selector in extreme_selectors:
        column = _EXTREME_PERIOD_SELECTORS[selector]["feature"]
        if column not in feature_t.columns:
            logger.warning(
                "Skipping extreme period '%s': clustering feature %s is not available.",
                selector,
                column,
            )
            continue
        target = add_mean_min if _EXTREME_PERIOD_SELECTORS[selector]["mode"] == "min" else add_mean_max
        if column not in target:
            target.append(column)
    return add_mean_min, add_mean_max


def _build_contiguous_timestep_index(start, periods, timestep_hours):
    """Build a regular datetime index with the requested timestep spacing."""
    return pd.date_range(
        start=pd.Timestamp(start),
        periods=int(periods),
        freq=pd.to_timedelta(timestep_hours, unit="h"),
    )


def _get_period_steps(period_hours, timestep_hours, period_name, label):
    """Convert a period length in hours to an integral snapshot count."""
    steps = int(round(period_hours / timestep_hours))
    if not np.isclose(steps * timestep_hours, period_hours):
        raise ValueError(
            f"{period_name} ({period_hours}h) is incompatible with current timestep "
            f"resolution ({timestep_hours:g}h) for {label}.",
        )
    if steps <= 0:
        raise ValueError(f"Calculated {period_name} steps are non-positive for {label}.")
    return steps


def _is_expected_leap_day_gap(previous: pd.Timestamp, current: pd.Timestamp) -> bool:
    """Return true for the 25-hour gap created by removing 29 February."""
    return (
        previous.month == 2
        and previous.day == 28
        and previous.hour == 23
        and current.month == 3
        and current.day == 1
        and current.hour == 0
        and previous.is_leap_year
    )


def _build_contiguous_source_period_rows(source_index, steps_per_period, timestep_hours):
    """Build complete non-overlapping periods without crossing weather-year gaps."""
    source_index = pd.DatetimeIndex(source_index)
    steps_per_period = int(steps_per_period)
    if source_index.empty or steps_per_period <= 0:
        raise ValueError("Source snapshots and steps_per_period must be non-empty/positive.")

    expected_delta = pd.Timedelta(hours=float(timestep_hours))
    boundaries = [0]
    for position in range(1, len(source_index)):
        previous = source_index[position - 1]
        current = source_index[position]
        if current - previous != expected_delta and not _is_expected_leap_day_gap(previous, current):
            boundaries.append(position)
    boundaries.append(len(source_index))

    rows = []
    for block_start, block_end in zip(boundaries[:-1], boundaries[1:]):
        for start in range(block_start, block_end, steps_per_period):
            stop = start + steps_per_period
            if stop <= block_end:
                rows.append(np.arange(start, stop))
    if not rows:
        raise ValueError("No complete representative-period candidates are available.")
    return rows


def _build_representative_period_mapping(agg, period_ids):
    """
    Map each period label tsam returned to the source period it was taken from.

    ``clusterCenterIndices`` holds the medoid of cluster ``i`` at position ``i``,
    and ``extremePeriods`` records the label (``newClusterNo``) and source period
    (``stepNo``) of every extreme period tsam appended as a new cluster center.
    """
    source_periods = {
        int(label): int(center_idx)
        for label, center_idx in enumerate(getattr(agg, "clusterCenterIndices", None) or [])
    }
    for info in getattr(agg, "extremePeriods", {}).values():
        new_cluster = info.get("newClusterNo")
        step_no = info.get("stepNo")
        if new_cluster is not None and step_no is not None:
            source_periods[int(new_cluster)] = int(step_no)

    missing = [int(label) for label in period_ids if int(label) not in source_periods]
    if missing:
        raise ValueError(f"Unable to map representative periods to source periods for labels {missing}.")
    return source_periods


def _build_representative_snapshot_weightings(sw, matching_idx, typical_idx):
    """Build representative-period snapshot weightings by summing source weights per cluster."""
    sw_rep = pd.DataFrame(index=typical_idx, columns=sw.columns, dtype="float64")
    for col in sw.columns:
        sw_rep[col] = sw[col].groupby(matching_idx).sum().reindex(typical_idx).to_numpy()
    return sw_rep


def _scale_snapshot_weightings_to_total_hours(snapshot_weightings, target_hours=8760.0, reference_column="objective"):
    """Scale snapshot weightings so the reference column sums to ``target_hours``."""
    if reference_column not in snapshot_weightings.columns:
        raise KeyError(f"snapshot_weightings missing reference column '{reference_column}'.")
    total = float(snapshot_weightings[reference_column].sum())
    if total <= 0:
        raise ValueError(f"Cannot scale snapshot weightings with non-positive {reference_column} sum ({total}).")
    return snapshot_weightings * (float(target_hours) / total)


def _get_extreme_period_ids(agg, period_ids):
    """Return the period labels tsam created for extreme periods."""
    period_id_set = {int(period_id) for period_id in period_ids}
    extreme_period_ids = set()
    for info in getattr(agg, "extremePeriods", {}).values():
        new_cluster = info.get("newClusterNo")
        if new_cluster is not None and int(new_cluster) in period_id_set:
            extreme_period_ids.add(int(new_cluster))
    return sorted(extreme_period_ids)


def _build_period_entry(period_id, kind, steps, source_snapshots, weightings):
    """Build internal period metadata for one representative or extreme block."""
    source_snapshots = pd.DatetimeIndex(source_snapshots)
    return {
        "period_id": int(period_id),
        "kind": str(kind),
        "steps": int(steps),
        "source_snapshots": source_snapshots,
        "source_start": source_snapshots[0] if len(source_snapshots) else None,
        "source_end": source_snapshots[-1] if len(source_snapshots) else None,
        "weightings": weightings.copy(),
    }


def _assign_period_entry_snapshots(period_label, base_time, timestep_hours, period_entries):
    """Assign contiguous synthetic snapshots (labeled with ``period_label``) to each entry."""
    total_steps = sum(int(entry["steps"]) for entry in period_entries)
    timesteps = _build_contiguous_timestep_index(base_time, total_steps, timestep_hours)

    start = 0
    entries = []
    snapshot_parts = []
    for entry in period_entries:
        steps = int(entry["steps"])
        period_timesteps = pd.DatetimeIndex(timesteps[start : start + steps])
        period_snapshots = pd.MultiIndex.from_arrays(
            [np.repeat(period_label, len(period_timesteps)), period_timesteps],
            names=["period", "timestep"],
        )
        updated = dict(entry)
        updated["snapshots"] = period_snapshots
        entries.append(updated)
        snapshot_parts.append(period_snapshots)
        start += steps

    if not snapshot_parts:
        return entries, pd.MultiIndex.from_arrays([[], []], names=["period", "timestep"])
    return entries, snapshot_parts[0].append(snapshot_parts[1:])


def _build_period_snapshot_weightings(period_entries, scale_to_hours=True):
    """Concatenate per-period snapshot weightings and optionally rescale annually."""
    parts = []
    for entry in period_entries:
        weights = entry["weightings"].copy()
        weights.index = entry["snapshots"]
        parts.append(weights)
    snapshot_weightings = pd.concat(parts) if parts else pd.DataFrame()
    if scale_to_hours and not snapshot_weightings.empty:
        snapshot_weightings = _scale_snapshot_weightings_to_total_hours(
            snapshot_weightings, target_hours=8760.0, reference_column="objective",
        )
    return snapshot_weightings


def _get_source_bounds(entry):
    """Return the (start, end) weather hours one period entry was drawn from."""
    source_snapshots = entry.get("source_snapshots")
    if source_snapshots is not None and len(source_snapshots):
        source_snapshots = pd.DatetimeIndex(source_snapshots)
        return source_snapshots[0], source_snapshots[-1]
    return entry.get("source_start"), entry.get("source_end")


def _get_period_weather_years(entry):
    """Return the weather year(s) one period entry is sourced from."""
    source_snapshots = entry.get("source_snapshots")
    if source_snapshots is not None and len(source_snapshots):
        return sorted({int(timestamp.year) for timestamp in pd.DatetimeIndex(source_snapshots)})
    return sorted(
        {
            int(pd.Timestamp(timestamp).year)
            for timestamp in _get_source_bounds(entry)
            if timestamp is not None and not pd.isna(timestamp)
        },
    )


def serialize_representative_period_metadata(period_entries_by_label):
    """
    Convert representative-period source ranges to JSON-safe network metadata.

    ``start`` / ``end`` / ``weather_years`` describe the weather hours the block
    was drawn from, so downstream plots can label a period with its actual
    weather year. ``snapshot_start`` / ``snapshot_end`` are the synthetic
    contiguous labels the network carries (see the module docstring).
    """
    metadata = {}
    for label, entries in period_entries_by_label.items():
        periods = []
        for entry in entries:
            snapshots = entry.get("snapshots")
            snapshot_timesteps = (
                pd.DatetimeIndex(snapshots.get_level_values("timestep"))
                if isinstance(snapshots, pd.MultiIndex)
                else pd.DatetimeIndex([])
            )
            source_start, source_end = _get_source_bounds(entry)
            periods.append(
                {
                    "period_id": int(entry["period_id"]),
                    "kind": str(entry.get("kind", "representative")),
                    "steps": int(entry.get("steps", 0)),
                    "start": (
                        None
                        if source_start is None or pd.isna(source_start)
                        else pd.Timestamp(source_start).isoformat()
                    ),
                    "end": (
                        None if source_end is None or pd.isna(source_end) else pd.Timestamp(source_end).isoformat()
                    ),
                    "weather_years": _get_period_weather_years(entry),
                    "snapshot_start": (
                        None if snapshot_timesteps.empty else pd.Timestamp(snapshot_timesteps[0]).isoformat()
                    ),
                    "snapshot_end": (
                        None if snapshot_timesteps.empty else pd.Timestamp(snapshot_timesteps[-1]).isoformat()
                    ),
                },
            )
        metadata[str(label)] = {"periods": periods}
    return metadata


def select_period_entries(feature_t_full, source_index_full, representative_periods_cfg):
    """
    Select representative + extreme period entries for one source weather-year timeline.

    One tsam run does everything: hierarchical clustering picks ``number``
    representative periods (represented by the real historical period closest to
    each cluster centroid, tsam's ``medoidRepresentation``), and -- when
    ``include_extreme`` asks for them -- tsam's ``new_cluster_center`` method adds
    the configured extreme periods as further cluster centers and reassigns
    whatever is closer to them. Period weights are the resulting cluster
    membership counts (see the module docstring).

    Parameters
    ----------
    feature_t_full : pandas.DataFrame
        Aggregate clustering features (weighted wind/solar CF, total AC load),
        indexed by ``source_index_full``.
    source_index_full : pandas.DatetimeIndex
        Full source timeline (e.g. concatenated 15 weather years).
    representative_periods_cfg : dict
        The ``clustering.temporal.representative_periods`` config block.

    Returns
    -------
    list[dict]
        Period entries (without a "snapshots" label yet -- see
        ``_assign_period_entry_snapshots``), each carrying a ``source_snapshots``
        DatetimeIndex that indexes directly into the raw source data and a
        ``weightings`` frame holding that period's cluster weight.
    """
    import tsam.timeseriesaggregation as tsam

    number = int(representative_periods_cfg.get("number", 4))
    if number <= 0:
        raise ValueError("representative_periods.number must be a positive integer.")

    extreme_selectors = get_extreme_period_selectors(representative_periods_cfg)
    period_hours = get_period_hours(representative_periods_cfg)

    if feature_t_full.empty:
        raise ValueError(
            "No valid weighted wind/solar or total AC load features found for representative period clustering.",
        )
    if len(feature_t_full.index) < 2:
        raise ValueError("Not enough snapshots to cluster representative periods.")

    timestep_hours = (feature_t_full.index[1] - feature_t_full.index[0]).total_seconds() / 3600.0
    if timestep_hours <= 0:
        raise ValueError("Invalid timestep spacing detected.")

    period_steps = _get_period_steps(period_hours, timestep_hours, "period length", "source")
    source_period_rows = _build_contiguous_source_period_rows(source_index_full, period_steps, timestep_hours)
    if number > len(source_period_rows):
        raise ValueError(
            "representative_periods.number exceeds the number of source periods available for clustering.",
        )

    # tsam slices its periods off a single contiguous frame, so the candidate rows
    # (which never straddle a weather-year gap) are concatenated and relabelled.
    cluster_rows = np.concatenate(source_period_rows)
    feature_cluster = feature_t_full.iloc[cluster_rows].copy()
    feature_cluster.index = _build_contiguous_timestep_index(
        feature_t_full.index[0], len(feature_cluster), timestep_hours,
    )

    tsam_kwargs = dict(
        timeSeries=feature_cluster,
        hoursPerPeriod=period_hours,
        noTypicalPeriods=number,
        clusterMethod="hierarchical",
    )
    add_mean_min, add_mean_max = build_extreme_period_selector_columns(feature_cluster, extreme_selectors)
    if add_mean_min or add_mean_max:
        tsam_kwargs.update(
            extremePeriodMethod="new_cluster_center",
            addMeanMin=add_mean_min,
            addMeanMax=add_mean_max,
        )

    agg = tsam.TimeSeriesAggregation(**tsam_kwargs)
    agg.createTypicalPeriods()

    matching = agg.indexMatching()
    matching_idx = pd.MultiIndex.from_arrays(
        [matching["PeriodNum"].to_numpy(), matching["TimeStep"].to_numpy()], names=["PeriodNum", "TimeStep"],
    )
    period_ids = sorted(int(label) for label in matching["PeriodNum"].unique())
    source_periods = _build_representative_period_mapping(agg, period_ids)
    extreme_period_ids = _get_extreme_period_ids(agg, period_ids)

    # One unit of weight per source snapshot, summed per (period, step) by the tsam
    # matching: every period's weight is the number of source periods it represents.
    sw_source = pd.DataFrame(1.0, index=feature_cluster.index, columns=["objective", "stores", "generators"])
    sw_local = _build_representative_snapshot_weightings(
        sw_source,
        matching_idx,
        pd.MultiIndex.from_product([period_ids, range(period_steps)], names=["PeriodNum", "TimeStep"]),
    )

    period_entries = []
    for period_id in period_ids:
        positions = source_period_rows[source_periods[period_id]]
        local_weights = sw_local.loc[pd.IndexSlice[period_id, :]].copy()
        local_weights.index = pd.RangeIndex(len(local_weights))
        period_entries.append(
            _build_period_entry(
                period_id,
                "extreme" if period_id in extreme_period_ids else "representative",
                period_steps,
                source_index_full[positions],
                local_weights,
            ),
        )

    logger.info(
        "Clustered %s source periods into %s periods (%s extreme): %s.",
        len(source_period_rows),
        len(period_entries),
        len(extreme_period_ids),
        ", ".join(
            f"P{entry['period_id']}"
            f"{'(extreme)' if entry['period_id'] in extreme_period_ids else ''}"
            f"={entry['weightings']['objective'].iloc[0]:g}x {entry['source_snapshots'][0]:%Y-%m-%d}"
            for entry in period_entries
        ),
    )
    return period_entries


def select_representative_snapshots(
    feature_t_full,
    source_index_full,
    representative_periods_cfg,
    investment_periods,
):
    """
    Run representative-period selection once and tile the result across investment horizons.

    The source weather-year timeline (and therefore the selected representative
    hours) is identical for every planning horizon, so tsam only needs to run
    once; the result is then relabeled with each horizon's ``period`` value.

    Returns
    -------
    snapshots : pandas.MultiIndex
        ``["period", "timestep"]`` snapshots, ready to assign to
        ``n.snapshots``.
    snapshot_weightings : pandas.DataFrame
        Matching ``n.snapshot_weightings``-shaped frame.
    period_entries_by_horizon : dict[int, list[dict]]
        Per-horizon period entries (each with ``source_snapshots`` for reading raw
        source data, and ``snapshots`` for the representative label index).
    metadata : dict
        JSON-safe representative-period metadata for ``n.meta``.
    """
    timestep_hours = (feature_t_full.index[1] - feature_t_full.index[0]).total_seconds() / 3600.0
    base_entries = select_period_entries(
        feature_t_full, source_index_full, representative_periods_cfg,
    )

    period_entries_by_horizon = {}
    snapshots_parts = []
    weightings_parts = []
    for horizon in investment_periods:
        horizon = int(horizon)
        entries, horizon_snapshots = _assign_period_entry_snapshots(
            horizon, source_index_full[0], timestep_hours, base_entries,
        )
        period_entries_by_horizon[horizon] = entries
        snapshots_parts.append(horizon_snapshots)
        weightings_parts.append(_build_period_snapshot_weightings(entries))

    if not snapshots_parts:
        raise ValueError("No investment periods supplied for representative-period selection.")

    snapshots = snapshots_parts[0].append(snapshots_parts[1:])
    snapshot_weightings = pd.concat(weightings_parts)
    metadata = serialize_representative_period_metadata(period_entries_by_horizon)
    return snapshots, snapshot_weightings, period_entries_by_horizon, metadata


# --------------------------------------------------------------------------- #
# National aggregate clustering features, built straight from the raw sources.
# --------------------------------------------------------------------------- #


def read_reeds_national_capacity_factor(carriers, reeds_vre_dir, weather_years):
    """
    Build the national wind/solar clustering features from the raw ReEDS files.

    The feature is the capacity-weighted mean capacity factor,
    ``sum_i(capacity_i * cf_i(t)) / sum_i(capacity_i)``, taken over **every** site
    in the ReEDS supply curve -- no spatial aggregation, no region filter.
    Selection runs before ``build_renewable_profiles``, so there are no per-bus
    profile files to read yet, and this needs nothing but the supply curve and the
    CF table.

    Carriers in the same feature group are combined by capacity, so numerator and
    denominator accumulate separately and are divided once at the end. Carriers
    sharing a supply curve are read once: ``offwind`` and ``offwind_floating`` are
    the same ReEDS sites split by water depth downstream, so counting both would
    double the offshore contribution.

    The solar inverter loading ratio is applied per site before aggregating,
    matching the downstream clip at 1.0.

    Returns
    -------
    dict[str, pandas.Series | None]
        ``{"wind": ..., "solar": ...}`` mean capacity factor over the full
        weather-year timeline; ``None`` for a group with no configured carriers.
    """
    # Imported lazily: build_reeds_renewable_profiles imports this module for
    # read_representative_snapshots, so a top-level import would be circular.
    from build_reeds_renewable_profiles import (
        REEDS_TECH,
        SOLAR_INVERTER_LOADING_RATIO,
        national_available_generation,
    )

    def solar_transform(block):
        return np.clip(block * SOLAR_INVERTER_LOADING_RATIO, None, 1.0)

    carriers = set(carriers)
    profiles = {}
    for group, group_carriers in REEDS_FEATURE_GROUPS.items():
        numerator = None
        total_capacity = 0.0
        seen_supply_curves = set()
        for carrier in group_carriers:
            if carrier not in carriers or carrier not in REEDS_TECH:
                continue
            cfg = REEDS_TECH[carrier]
            if cfg["sc"] in seen_supply_curves:
                logger.info(
                    "Skipping %s: it shares ReEDS supply curve %s already counted for this feature.",
                    carrier,
                    cfg["sc"],
                )
                continue
            seen_supply_curves.add(cfg["sc"])

            supply_curve = pd.read_csv(f"{reeds_vre_dir}/{cfg['sc']}")
            capacities = supply_curve.groupby("sc_point_gid").capacity.sum()
            capacities = capacities[capacities > 0]
            if capacities.empty:
                logger.warning("No positive %s site capacity in %s. Skipping it.", carrier, cfg["sc"])
                continue

            transform = solar_transform if carrier in SOLAR_CARRIERS else None
            cf_path = f"{reeds_vre_dir}/{cfg['cf']}"
            parts = []
            for weather_year in weather_years:
                values = national_available_generation(
                    cf_path,
                    capacities.index.tolist(),
                    capacities,
                    int(weather_year),
                    transform=transform,
                )
                index = get_weather_year_snapshots([int(weather_year)], drop_leap_day=True)
                if len(values) != len(index):
                    raise ValueError(
                        f"ReEDS {carrier} {weather_year} returned {len(values)} hours; "
                        f"expected {len(index)}.",
                    )
                parts.append(pd.Series(values, index=index, dtype="float64"))

            series = pd.concat(parts)
            logger.info(
                "Aggregated %s ReEDS %s sites (%.0f MW nameplate); mean CF %.4f.",
                len(capacities),
                carrier,
                float(capacities.sum()),
                float(series.sum() / (len(series) * float(capacities.sum()))),
            )
            if numerator is None:
                numerator = series
            elif not series.index.equals(numerator.index):
                raise ValueError(
                    f"Carriers contributing to the '{group}' feature do not share an identical "
                    "time index; all must be built on the same weather-year calendar.",
                )
            else:
                numerator = numerator + series
            total_capacity += float(capacities.sum())

        if numerator is None or total_capacity <= 0:
            profiles[group] = None
            continue
        profiles[group] = numerator / total_capacity
        logger.info(
            "Combined %s feature: mean CF %.4f over %s hours (%.0f MW nameplate).",
            group,
            float(profiles[group].mean()),
            len(profiles[group]),
            total_capacity,
        )
    return profiles


def build_national_feature_frame(carrier_profiles, demand_total):
    """
    Assemble the clustering feature frame from the national aggregate series.

    Parameters
    ----------
    carrier_profiles : dict[str, pandas.Series | None]
        ``{"wind": ..., "solar": ...}`` capacity-weighted mean capacity factor,
        from ``read_reeds_national_capacity_factor``.
    demand_total : pandas.Series
        Total national AC demand for the planning horizon, indexed by the raw
        source timestamps. **Must already be converted to UTC** -- build it with
        ``build_eer_demand.ReadEer``, which applies the CST->UTC roll. Passing a
        CST series here silently shifts the load feature six hours out of phase
        with the renewable features.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, pandas.Series | None]]
        The feature frame (wind, solar, AC demand) and the individual aggregate
        profiles keyed ``"wind"`` / ``"solar"`` / ``"load"``, which the diagnostic
        plots reuse. Extreme periods are ranked on the feature columns themselves
        (see ``build_extreme_period_selector_columns``).
    """
    load = None
    if demand_total is not None and len(demand_total):
        load = pd.Series(pd.to_numeric(demand_total, errors="coerce"), dtype="float64")
        load.index = pd.DatetimeIndex(load.index)

    profile_map = {
        "wind": carrier_profiles.get("wind"),
        "solar": carrier_profiles.get("solar"),
        "load": load,
    }

    # Hard alignment check: this is where a CST/UTC mix-up would otherwise slip
    # through and quietly corrupt both the clustering and the extreme selection.
    available = {name: profile for name, profile in profile_map.items() if profile is not None}
    if not available:
        raise ValueError(
            "No wind/solar generation or AC demand available for representative-period clustering.",
        )
    reference_name, reference = next(iter(available.items()))
    for name, profile in available.items():
        if not profile.index.equals(reference.index):
            raise ValueError(
                f"The '{name}' feature index does not match the '{reference_name}' feature index. "
                "ReEDS profiles are UTC while EER demand is CST (UTC-06:00); build the demand "
                "series with build_eer_demand.ReadEer so it is rolled to UTC first.",
            )

    feature_map = {}
    for carrier_name, column in (("wind", WIND_FEATURE), ("solar", SOLAR_FEATURE)):
        if profile_map[carrier_name] is not None:
            feature_map[column] = profile_map[carrier_name]
    if load is not None:
        feature_map[LOAD_FEATURE] = load

    features = pd.DataFrame(feature_map, index=reference.index)
    features.columns = pd.MultiIndex.from_tuples(features.columns)
    features = features.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    logger.info(
        "Built national clustering features %s over %s source snapshots.",
        [tuple(column) for column in features.columns],
        len(features),
    )
    return features, profile_map


# --------------------------------------------------------------------------- #
# Diagnostic plots, built from the national feature profiles.
# --------------------------------------------------------------------------- #

_PROFILE_PLOT_LABELS = (("wind", "Wind"), ("solar", "Solar"), ("load", "Load"))
_PROFILE_PLOT_COLORS = {"Wind": "#2a6f97", "Solar": "#dd8a24", "Load": "#b23a48"}


def _normalize_profile_for_plot(data):
    """Normalize plotted profile data by a shared maximum value."""
    max_value = float(np.nanmax(data.to_numpy())) if data.size else 0.0
    if max_value <= 0:
        return data * 0.0
    return data.astype(float) / max_value


def _format_representative_period_timestamp(timestamp):
    """Format a representative-period timestamp for plot labels."""
    if timestamp is None or pd.isna(timestamp):
        return "N/A"
    return pd.Timestamp(timestamp).strftime("%m-%d")


def _format_representative_period_range(start, end):
    """Format a representative-period source time range for plot labels."""
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return "N/A"
    return (
        f"{_format_representative_period_timestamp(start)}"
        f" to {_format_representative_period_timestamp(end)}"
    )


def _format_weather_year_label(years):
    """
    Format the real weather year(s) a representative period is sourced from.

    Built from the *real* source hours (not the synthetic labels and not the
    duplicated source index), so the label always names an actual weather year.
    A period that wraps the end of the timeline touches two years and is
    labelled ``"<first>/<second>"``.
    """
    years = [int(year) for year in years]
    if not years:
        return "N/A"
    return "/".join(str(year) for year in years)


def _format_representative_mean_value(value):
    """Format representative-period summary values for the plot sidebar."""
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _build_period_profile_matrix(profile, period_entries):
    """
    Reshape a national source profile into a (time-step x period) matrix.

    Periods shorter than the longest one are NaN-padded, so representative and
    extreme blocks of differing length can share one matrix.
    """
    if profile is None or profile.empty:
        return None

    columns = {}
    for entry in period_entries:
        source_snapshots = pd.DatetimeIndex(entry["source_snapshots"])
        values = profile.reindex(source_snapshots).to_numpy(dtype=float)
        columns[int(entry["period_id"])] = pd.Series(values, index=np.arange(len(values)))

    if not columns:
        return None

    matrix = pd.DataFrame(columns)
    matrix.index.name = "TimeStep"
    matrix.columns.name = "PeriodNum"
    return matrix


def build_plot_data(profile_map, period_entries_by_label):
    """Build the per-label plot payload consumed by ``plot_representative_period_profiles``."""
    plot_data = {}
    for label, entries in period_entries_by_label.items():
        plot_profiles = {}
        period_means = {}
        for key, display_name in _PROFILE_PLOT_LABELS:
            matrix = _build_period_profile_matrix(profile_map.get(key), entries)
            if matrix is None:
                continue
            period_means[display_name] = matrix.mean(axis=0)
            plot_profiles[display_name] = _normalize_profile_for_plot(matrix)

        if plot_profiles:
            # Panels are labelled from the source hours, never from the synthetic
            # snapshot labels (see the module docstring).
            plot_data[label] = {
                "period_ids": [int(entry["period_id"]) for entry in entries],
                "period_ranges": {
                    int(entry["period_id"]): _get_source_bounds(entry) for entry in entries
                },
                "period_weather_years": {
                    int(entry["period_id"]): _format_weather_year_label(_get_period_weather_years(entry))
                    for entry in entries
                },
                "period_kinds": {
                    int(entry["period_id"]): str(entry.get("kind", "representative"))
                    for entry in entries
                },
                "profiles": plot_profiles,
                "period_means": period_means,
            }
    return plot_data


def plot_representative_period_profiles(plot_data, period_label, output_path):
    """
    Plot every representative period into one figure at ``output_path``.

    One stacked panel per period (normalized national wind/solar/load), written to
    the single file the rule declares as its output. Earlier versions wrote one
    file per period next to ``output_path``, which left the declared output
    missing and failed the rule.

    Each panel is titled with the real weather year and month-day range the block
    was drawn from -- not the planning horizon and not the synthetic snapshot
    labels, which are contiguous placeholders (see the module docstring).
    """
    if not plot_data or output_path is None:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available. Skipping representative-period plot export.")
        return

    panels = [
        (label, period_id)
        for label in plot_data
        for period_id in plot_data[label]["period_ids"]
        if any(period_id in frame.columns for frame in plot_data[label]["profiles"].values())
    ]
    if not panels:
        logger.warning("No representative-period profiles available to plot.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3.4 * len(panels)), squeeze=False)
    for axis, (label, period_id) in zip(axes[:, 0], panels):
        label_data = plot_data[label]
        for metric, profile_df in label_data["profiles"].items():
            if period_id not in profile_df.columns:
                continue
            series = profile_df[period_id].dropna()
            axis.plot(
                np.arange(1, len(series) + 1),
                series.to_numpy(),
                color=_PROFILE_PLOT_COLORS[metric],
                linestyle="-",
                linewidth=1.8,
                label=metric,
            )

        period_ranges = label_data.get("period_ranges", {})
        period_kinds = label_data.get("period_kinds", {})
        period_range = _format_representative_period_range(*period_ranges.get(period_id, (None, None)))
        kind = period_kinds.get(period_id, "")
        kind_suffix = f", {kind}" if kind else ""
        weather_year = label_data.get("period_weather_years", {}).get(period_id, "N/A")
        axis.set_title(
            f"Weather year {weather_year} {period_label} P{period_id + 1} ({period_range}{kind_suffix})",
            fontsize=11,
        )
        axis.set_xlabel("Step within period")
        axis.set_ylabel("Normalized value")
        axis.set_ylim(-0.02, 1.05)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper left", title="Metric", fontsize=8)

        summary_lines = [f"P{period_id + 1} means"]
        period_means = label_data.get("period_means", {})
        for short_label, metric_name in (("W", "Wind"), ("S", "Solar"), ("L", "Load")):
            values = period_means.get(metric_name)
            if values is None or period_id not in values.index:
                continue
            summary_lines.append(f"{short_label}: {_format_representative_mean_value(values.loc[period_id])}")
        if len(summary_lines) > 1:
            axis.text(
                1.01,
                0.98,
                "\n".join(summary_lines),
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                family="monospace",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
            )

    fig.tight_layout(rect=(0, 0, 0.85, 1))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s representative-period panels to %s.", len(panels), output_path)


# ---------------------------------------------------------------------------
# Reading the snapshot definition back (used by downstream rules)
# ---------------------------------------------------------------------------


def read_representative_snapshots(path: str):
    """
    Read the snapshot definition written by ``select_representative_periods``.

    Returns
    -------
    snapshots : pandas.MultiIndex
        ``["period", "timestep"]`` index ready to assign to ``n.snapshots``.
        ``timestep`` holds the synthetic contiguous labels, not real weather
        hours.
    snapshot_weightings : pandas.DataFrame
        ``objective`` / ``stores`` / ``generators`` columns indexed by
        ``snapshots``, already scaled so ``objective`` sums to 8760 per period.
    source_timesteps : pandas.DatetimeIndex
        The real weather hour behind each snapshot, positionally aligned with
        ``snapshots``. Use it to slice raw time series before relabelling.
    """
    table = pd.read_csv(path, parse_dates=["timestep", "source_timestep"])
    snapshots = pd.MultiIndex.from_arrays(
        [table.period.astype(int), pd.DatetimeIndex(table.timestep)],
        names=["period", "timestep"],
    )
    snapshot_weightings = table.loc[:, ["objective", "stores", "generators"]].astype(float)
    snapshot_weightings.index = snapshots
    return snapshots, snapshot_weightings, pd.DatetimeIndex(table.source_timestep)


def reindex_source_timeseries_to_snapshots(n, df, source_timesteps):
    """
    Slice a raw source time series down to the representative hours.

    Parameters
    ----------
    n : pypsa.Network
        Network whose ``snapshots`` the result is labelled with.
    df : pandas.DataFrame | pandas.Series
        Time series indexed by real weather timestamps (the same calendar the
        ``profile_{tech}.nc`` files use).
    source_timesteps : pandas.DatetimeIndex
        Real weather hour per representative snapshot, from
        ``read_representative_snapshots``.

    Notes
    -----
    ``source_timesteps`` may repeat an hour when two periods overlap, which is
    why this selects with ``.loc`` rather than reindexing: repeated labels in the
    selector are fine as long as ``df``'s own index is unique.
    """
    source_timesteps = pd.DatetimeIndex(source_timesteps)
    if len(source_timesteps) != len(n.snapshots):
        raise ValueError(
            f"Got {len(source_timesteps)} source hours for {len(n.snapshots)} network snapshots.",
        )

    df = df.copy()
    df.index = pd.DatetimeIndex(df.index)
    if not df.index.is_unique:
        raise ValueError("Source time series index must be unique to slice representative hours.")

    missing = source_timesteps.difference(df.index)
    if not missing.empty:
        raise ValueError(
            f"Source time series is missing {len(missing)} representative hours "
            f"(first: {missing[0]}). Check that it uses the same weather-year calendar "
            "as get_weather_year_snapshots(..., drop_leap_day=True).",
        )

    sliced = df.loc[source_timesteps]
    sliced.index = n.snapshots
    return sliced


def _calendar_hour_key(index: pd.DatetimeIndex) -> pd.Index:
    """Month/day/hour identity of a timestamp, ignoring its year."""
    return pd.Index(index.month * 10000 + index.day * 100 + index.hour)


def reindex_calendar_timeseries_to_snapshots(n, df, source_timesteps):
    """
    Slice a fixed-calendar time series down to the representative hours.

    Some inputs (the Breakthrough hydro profiles) are published on a single
    canonical year rather than on the real weather years, so they carry no
    weather-hour identity and cannot be sliced by timestamp like
    ``reindex_source_timeseries_to_snapshots`` does. Match on (month, day, hour)
    instead, which is the same year-agnostic reuse the full-timeline path gets
    from ``broadcast_investment_horizons_index``.
    """
    source_timesteps = pd.DatetimeIndex(source_timesteps)
    if len(source_timesteps) != len(n.snapshots):
        raise ValueError(
            f"Got {len(source_timesteps)} source hours for {len(n.snapshots)} network snapshots.",
        )

    df = df.copy()
    df.index = _calendar_hour_key(pd.DatetimeIndex(df.index))
    if not df.index.is_unique:
        raise ValueError(
            "Calendar time series must cover a single year: found repeated (month, day, hour) "
            "entries, so representative hours cannot be resolved unambiguously.",
        )

    selector = _calendar_hour_key(source_timesteps)
    missing = selector.difference(df.index)
    if not missing.empty:
        first = missing[0]
        raise ValueError(
            f"Calendar time series is missing {len(missing)} representative hours "
            f"(first: month {first // 10000:02d}, day {first % 10000 // 100:02d}, hour {first % 100:02d}). "
            "Check that it covers a full leap-day-free year.",
        )

    sliced = df.loc[selector]
    sliced.index = n.snapshots
    return sliced


# ---------------------------------------------------------------------------
# Snakemake entry point
# ---------------------------------------------------------------------------


SNAPSHOT_TABLE_COLUMNS = [
    "period",
    "timestep",
    "source_timestep",
    "period_id",
    "kind",
    "objective",
    "stores",
    "generators",
]


def read_total_national_demand(
    demand_path: str,
    planning_horizon: int,
    weather_years,
) -> pd.Series:
    """
    Read total national AC demand for one planning horizon, rolled to UTC.

    Distribution losses are deliberately not applied: they are a uniform
    multiplier and therefore affect neither the clustering (tsam
    normalizes each feature) nor the extreme-period ranking.
    """
    # Imported lazily: build_eer_demand imports this module for
    # read_representative_snapshots, so a top-level import would be circular.
    from build_eer_demand import ReadEer

    demand = ReadEer(demand_path, [planning_horizon], weather_years).read()
    demand = demand.loc[planning_horizon]
    total = demand.sum(axis=1).astype("float64")
    total.index = pd.DatetimeIndex(total.index)
    logger.info(
        "Read total national AC demand for %s: %s snapshots, mean %.1f MW.",
        planning_horizon,
        len(total),
        float(total.mean()),
    )
    return total


def build_snapshot_table(period_entries_by_horizon, snapshot_weightings, snapshots) -> pd.DataFrame:
    """
    Flatten the selection result into the snapshot-definition table.

    ``timestep`` is the synthetic contiguous label carried by the network, while
    ``source_timestep`` is the real weather hour the values must be read from.
    """
    frames = []
    for entries in period_entries_by_horizon.values():
        for entry in entries:
            entry_snapshots = entry["snapshots"]
            source = pd.DatetimeIndex(entry["source_snapshots"])
            if len(entry_snapshots) != len(source):
                raise ValueError(
                    f"Representative period {entry['period_id']} has {len(entry_snapshots)} snapshots "
                    f"but {len(source)} source hours.",
                )
            frames.append(
                pd.DataFrame(
                    {
                        "source_timestep": source,
                        "period_id": int(entry["period_id"]),
                        "kind": str(entry["kind"]),
                    },
                    index=entry_snapshots,
                ),
            )

    table = pd.concat(frames)
    if not table.index.equals(snapshots):
        raise ValueError("Snapshot table order does not match the selected snapshots.")

    table = table.join(snapshot_weightings)
    missing = [column for column in ("objective", "stores", "generators") if column not in table.columns]
    if missing:
        raise ValueError(f"Snapshot weightings are missing required columns {missing}.")

    table = table.reset_index()
    return table.loc[:, SNAPSHOT_TABLE_COLUMNS]


def validate_period_counts(period_entries_by_horizon, representative_periods) -> None:
    """
    Guard against tsam silently returning the wrong periods.

    The representative count must come out exactly; the extreme count may fall
    short, because tsam skips an extreme period that clustering already picked as
    a cluster center instead of taking the next-most-extreme candidate. That is
    reported as a warning, not an error.
    """
    number = int(representative_periods.get("number", 4))
    extreme_selectors = get_extreme_period_selectors(representative_periods)
    period_hours = get_period_hours(representative_periods)

    for horizon, entries in period_entries_by_horizon.items():
        representative_entries = [entry for entry in entries if entry.get("kind") != "extreme"]
        extreme_entries = [entry for entry in entries if entry.get("kind") == "extreme"]
        if len(representative_entries) != number:
            raise ValueError(
                f"Representative selection produced {len(representative_entries)} representative periods "
                f"for {horizon}; expected {number}.",
            )
        if len(extreme_entries) != len(extreme_selectors):
            logger.warning(
                "Selection produced %s extreme periods for %s instead of the %s requested (%s): tsam drops "
                "an extreme period that is already a cluster center.",
                len(extreme_entries),
                horizon,
                len(extreme_selectors),
                extreme_selectors,
            )

        expected_steps = len(entries) * period_hours
        total_steps = sum(int(entry["steps"]) for entry in entries)
        if total_steps != int(expected_steps):
            raise ValueError(
                f"Representative selection produced {total_steps} snapshots for {horizon}; "
                f"expected {int(expected_steps)}.",
            )


def main(snakemake) -> None:
    params = snakemake.params
    representative_periods = params.representative_periods or {}

    if not representative_periods.get("enable", False):
        raise ValueError(
            "select_representative_periods ran with clustering.temporal.representative_periods.enable "
            "set to false. The rule should not be part of the DAG in that case.",
        )

    planning_horizons = [int(horizon) for horizon in params.planning_horizons]
    if len(planning_horizons) != 1:
        raise ValueError(
            "Representative-period selection supports exactly one planning horizon; "
            f"received {planning_horizons}.",
        )
    planning_horizon = planning_horizons[0]

    carrier_profiles = read_reeds_national_capacity_factor(
        params.renewable_carriers,
        params.reeds_vre_dir,
        params.renewable_weather_years,
    )

    demand_total = read_total_national_demand(
        snakemake.input.electricity_demand,
        planning_horizon,
        params.renewable_weather_years,
    )

    feature_t_full, profile_map = build_national_feature_frame(carrier_profiles, demand_total)

    snapshots, snapshot_weightings, period_entries_by_horizon, metadata = select_representative_snapshots(
        feature_t_full,
        feature_t_full.index,
        representative_periods,
        [planning_horizon],
    )

    validate_period_counts(period_entries_by_horizon, representative_periods)

    table = build_snapshot_table(period_entries_by_horizon, snapshot_weightings, snapshots)
    table.to_csv(snakemake.output.snapshots, index=False)
    logger.info(
        "Wrote %s representative snapshots (%s periods) to %s.",
        len(table),
        table.period_id.nunique(),
        snakemake.output.snapshots,
    )

    with open(snakemake.output.metadata, "w") as stream:
        json.dump(metadata, stream, indent=2)

    representative_days = get_period_hours(representative_periods) / 24.0
    plot_representative_period_profiles(
        build_plot_data(profile_map, period_entries_by_horizon),
        f"{representative_days:g}-day",
        snakemake.output.plot,
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("select_representative_periods", demand_level="High")
    configure_logging(snakemake)
    main(snakemake)
