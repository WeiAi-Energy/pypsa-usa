"""
Representative-period selection: everything in one place.

This module owns the whole representative-period story:

* the tsam hierarchical clustering itself, run on three national feature series -- the
  ``p_nom_max``-weighted onshore wind capacity factor, the ``p_nom_max``-weighted solar
  capacity factor and the AC demand -- computed straight from the raw ReEDS supply curves
  / CF tables and the EER demand h5;
* the snapshot definition it writes out (``snapshots.csv``, ``metadata.json``,
  ``profiles.png``) and the readers downstream rules use to consume it;
* the diagnostic profile plots.

It runs at the very top of the electricity workflow -- *before*
``build_renewable_profiles`` -- so the selected hours are known before any
per-bus profile or network exists. ``build_renewable_profiles`` then builds only
those hours, and every downstream rule attaches time series for those hours only,
so the full 15-weather-year x 8760 h timeline is never materialised anywhere.

Clustering features and column weights
--------------------------------------
The wind and solar features are ``sum_i(capacity_i * cf_i(t)) / sum_i(capacity_i)`` over
**every** site in the ReEDS supply curve. The supply-curve capacity summed per bus is
exactly the ``p_nom_max`` that ``build_reeds_renewable_profiles`` writes, so each feature
is the national capacity factor weighted by where the model can actually build. The load
feature is the national AC demand, the sum of the EER per-state columns. None of this
needs a bus region or any other spatial mapping, so the step depends on nothing but
static data files -- which matters, because it runs before any network exists.

Every column is compared on a per-unit-of-its-maximum scale (``x / max``). tsam always
min-max normalizes each column internally, and ward only sees differences, so scaling a
column's ``weightDict`` entry by ``(max - min) / max`` turns tsam's min-max scale into
``x / max`` (see ``max_normalized_weights``). For a capacity factor, whose minimum is
~0, this changes nothing; for load, whose minimum is roughly half its peak, it halves
the range, so a load swing is no longer counted as if it spanned 0 to peak. With the
build the model ends up with (onshore wind and solar nameplate each about national peak
load), one GW of deviation then weighs about the same in all three families, where
min-max normalization weighted a GW of load 3-4x a GW of wind or solar. Beyond that
scale the three columns carry equal weight.

The ``wind`` feature is onshore wind only; ``offwind`` / ``offwind_floating`` are left
out of the clustering even when they are configured carriers. Offshore is 42% of
national supply-curve nameplate, so merging it in would put roughly half the wind
feature on offshore shapes while the model builds almost nothing offshore, and the
selected periods would under-represent the onshore resource the capacity expansion
actually responds to. Offshore profiles are still built downstream for whichever periods
are selected; they just do not steer the selection, and the wind-lull extreme is ranked
on onshore wind.

The older in-network "force one spring + one fall representative period" seasonal
constraint is intentionally gone. Representative periods are real historical periods and
their values are never rescaled -- every weather hour stays physically consistent with
its temperature (``build_region_temperature`` reads the same hours). Only which member
represents a cluster and the period *weights* are chosen here, see "Representative
periods" and "Period weights".

Representative periods
----------------------
tsam only partitions the timeline into clusters, on the full hourly profiles of the three
features; the member that represents each cluster is picked by ``match_period_means``:
the period whose national means (onshore wind CF, solar CF, load -- relative to their
timeline means) are closest to its cluster's. Only the means are matched, not the
intra-period shape: the shape is what tsam clustered on, so the members of a cluster
already share one. tsam's own medoid -- the member with the smallest summed distance to
the rest over every hour of every column -- is a multivariate median, and daily wind is
right-skewed, so the most central day is a calm one and the medoids under-shoot the wind
mean. A frame without weighted columns falls back to tsam's medoid.

Extreme periods and weighting
-----------------------------
``include_extreme`` is a plain on/off switch. When it is true, three extreme periods
are requested -- one single-resource stress case per clustering feature:

* **demand max**: the period with the highest mean national AC demand;
* **wind min**: the period with the lowest mean ``p_nom_max``-weighted national onshore
  wind capacity factor;
* **solar min**: the period with the lowest mean ``p_nom_max``-weighted national solar
  capacity factor.

Each is ranked on one raw feature, so the three are the three physical stress cases the
system has to survive on their own terms -- the peak-load block, the wind lull and the
solar lull -- rather than one blended metric that can only ever return whichever stress
the year happens to be worst at. tsam's ``addMeanMax`` / ``addMeanMin`` rank on the
weighted, normalized profiles, and a positive constant scale is order-preserving, so
each picks exactly the period its raw series would. A feature that is missing -- no wind
carrier configured, say -- or degenerate (flat over the whole timeline, so it carries no
ranking signal) is dropped with a warning rather than failing the run.

tsam adds each extreme period with ``extremePeriodMethod="append"``: its only member is
its own source period, so it constrains the stress case without absorbing the periods
around it. (tsam's ``new_cluster_center`` would hand it every period closer to it than to
its own medoid; on onshore wind that let the wind lull absorb 15-40 days of a 365-day
year and dragged the annual mean CF down by several percent.)

Period weights
--------------
The weights are the cluster membership counts, rescaled so ``objective`` sums to 8760 h
per planning horizon (``finalize_period_weights``). One source period out of 15 weather
years is worth well under an hour per snapshot, so every extreme snapshot is raised to
``MIN_SNAPSHOT_WEIGHT_HOURS`` (1 h); the representative periods share the rest of the
year in proportion to their counts.

The weights are deliberately not refitted to the timeline means. With a handful of
periods, a least-squares refit bought a closer mean by pushing whole clusters to the
floor -- a 578-day cluster down to 1 h per snapshot -- and so traded away the
distribution the clusters stand for. What is left of the mean mismatch is what
``match_period_means`` leaves; it is logged.

Because both kinds of period come out of one tsam run over one period partition,
extreme and representative periods necessarily share a length: the single
``period_length`` config key (see ``get_period_hours``). Note also that tsam
*drops* a requested extreme period that is already a cluster center -- or that
another selector has already claimed -- rather than falling back to the
next-most-extreme candidate, so a run can legitimately return fewer periods than
``number + 3``; ``validate_period_counts`` warns about it.

Timezone
--------
ReEDS capacity factors are UTC and EER demand is fixed US Central Standard Time
(UTC-06:00). ``build_eer_demand.ReadEer`` converts EER's published timestamps
to UTC, while the VRE reader uses each file's published ``time_index_<year>``.
The clustering frame retains only their exact shared UTC hours rather than
assuming that 8760 positional rows imply the same calendar.

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
from _helpers import configure_logging

logger = logging.getLogger(__name__)

# Renewable carriers aggregated into each clustering feature. Carriers in the same
# group are combined by capacity into one mean-capacity-factor series. Offshore wind is
# deliberately absent: it would dominate the wind feature (see the module docstring).
WIND_CARRIERS = ("onwind",)
SOLAR_CARRIERS = ("solar",)
REEDS_FEATURE_GROUPS = {"wind": WIND_CARRIERS, "solar": SOLAR_CARRIERS}

WIND_FEATURE = ("Generator", "p_max_pu", "wind")
SOLAR_FEATURE = ("Generator", "p_max_pu", "solar")
LOAD_FEATURE = ("Load", "p_set", "ac_load")

# The three clustering columns, keyed by feature family. Flat 3-level tuples, so tsam can
# key a weightDict and addMeanMax/addMeanMin off them.
_NATIONAL_FEATURES = {"wind": WIND_FEATURE, "solar": SOLAR_FEATURE, "load": LOAD_FEATURE}

# Every extreme snapshot keeps at least this many hours of the 8760 h year once the
# period weights are finalized (see "Period weights").
MIN_SNAPSHOT_WEIGHT_HOURS = 1.0

# The three extreme periods, one per clustering feature: the peak-load block, the
# p_nom_max-weighted onshore wind lull and the p_nom_max-weighted solar lull. Each is
# ranked on the period mean of one raw national feature. ``direction`` picks the tsam
# argument the feature is passed to -- "max" -> ``addMeanMax``, "min" -> ``addMeanMin``.
EXTREME_SELECTORS = (
    {"name": "demand_max", "feature": LOAD_FEATURE, "direction": "max"},
    {"name": "wind_min", "feature": WIND_FEATURE, "direction": "min"},
    {"name": "solar_min", "feature": SOLAR_FEATURE, "direction": "min"},
)


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


def get_include_extreme(representative_periods):
    """
    Return the boolean ``representative_periods.include_extreme`` switch.

    The key used to take a *configurable* list of per-feature selectors
    (``demand_max``, ``solar_min``, ...). That is gone -- the three selectors in
    ``EXTREME_SELECTORS`` are now fixed -- so the key is a plain on/off switch and a
    leftover list is rejected rather than silently reinterpreted.
    """
    include_extreme = representative_periods.get("include_extreme", False)
    if include_extreme is None:
        return False
    if isinstance(include_extreme, bool):
        return include_extreme
    raise ValueError(
        "representative_periods.include_extreme must be true or false; configurable per-feature "
        f"selector lists are no longer supported (got {include_extreme!r}). true selects the three "
        "fixed extremes: the period with the highest mean demand, the one with the lowest mean onshore "
        "wind capacity factor, and the one with the lowest mean solar capacity factor.",
    )


def resolve_extreme_selectors(feature_t):
    """
    Split ``EXTREME_SELECTORS`` into the tsam ``addMeanMax`` / ``addMeanMin`` column lists.

    Every selector ranks on a raw clustering feature, so the columns tsam is asked to
    rank extreme periods on are columns of the frame it already clusters -- nothing is
    appended to ``feature_t`` and no column weight is touched, which is why the
    representative-period clustering is bit-for-bit the same whether extremes are
    requested or not.

    A selector whose feature is missing from the frame -- no wind carrier configured,
    say -- is dropped with a warning rather than failing the run, and so is one whose
    feature is degenerate: a flat series has no highest or lowest period, so whichever
    block tsam's ``idxmax``/``idxmin`` happens to land on carries no meaning.

    Parameters
    ----------
    feature_t : pandas.DataFrame
        The frame handed to tsam.

    Returns
    -------
    tuple[list[tuple], list[tuple]]
        The feature columns to pass as ``addMeanMax`` and as ``addMeanMin``
        respectively; both are empty when no selector could be resolved.
    """
    mean_max_columns, mean_min_columns = [], []
    for selector in EXTREME_SELECTORS:
        column = selector["feature"]
        if column not in feature_t.columns:
            logger.warning(
                "Skipping the %s extreme period: clustering feature %s is not available.",
                selector["name"],
                column,
            )
            continue

        values = feature_t[column].astype(float)
        span = float(values.max() - values.min())
        if not np.isfinite(span) or span <= 0:
            logger.warning(
                "Skipping the %s extreme period: clustering feature %s is degenerate, so no "
                "period is more extreme than any other.",
                selector["name"],
                column,
            )
            continue

        if selector["direction"] == "max":
            mean_max_columns.append(column)
        else:
            mean_min_columns.append(column)
        logger.info(
            "Ranking the %s extreme period on the %s period mean of %s.",
            selector["name"],
            selector["direction"],
            column,
        )
    return mean_max_columns, mean_min_columns


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
        if current - previous != expected_delta:
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


def _extreme_period_sources(agg):
    """
    Return ``{period label: source period}`` for every extreme period tsam appended.

    ``extremePeriods`` records each extreme's source period (``stepNo``) and
    ``extremeClusterIdx`` its label; ``append`` fills both in the same order.
    """
    extreme_periods = list((getattr(agg, "extremePeriods", None) or {}).values())
    labels = list(getattr(agg, "extremeClusterIdx", None) or [])
    return {
        int(label): int(info["stepNo"])
        for label, info in zip(labels, extreme_periods)
        if info.get("stepNo") is not None
    }


def _build_representative_period_mapping(agg, period_ids, representatives=None):
    """
    Map each period label tsam returned to the source period it was taken from.

    Typical cluster ``i`` is represented by ``representatives[i]`` when given (see
    ``match_period_means``), else by tsam's medoid at position ``i`` of
    ``clusterCenterIndices``; the extreme periods come from ``_extreme_period_sources``.
    """
    source_periods = {
        int(label): int(center_idx)
        for label, center_idx in enumerate(getattr(agg, "clusterCenterIndices", None) or [])
    }
    if representatives:
        source_periods.update(
            {int(label): int(index) for label, index in representatives.items() if int(label) in source_periods},
        )
    source_periods.update(_extreme_period_sources(agg))

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
    return sorted(label for label in _extreme_period_sources(agg) if label in period_id_set)


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
    """
    Concatenate per-period snapshot weightings and optionally rescale annually.

    ``finalize_period_weights`` already makes them sum to 8760 h, so the rescale is only
    a guard against rounding.
    """
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


def max_normalized_weights(weight_dict, frame):
    """
    Rescale tsam column weights so the clustering compares every column as ``x / max``.

    tsam min-max normalizes each column to ``(x - min) / (max - min)`` and ward only sees
    differences, so multiplying a column's weight by ``(max - min) / max`` makes the
    distance exactly the one of ``x / max``. A column whose maximum is not positive
    carries no signal and gets weight 0 (tsam lifts it to its own floor). Columns the
    frame does not carry are dropped -- tsam raises on an unknown key.
    """
    weights = {}
    for column, weight in weight_dict.items():
        if column not in frame.columns:
            continue
        peak = float(frame[column].max())
        span = peak - float(frame[column].min())
        weights[column] = float(weight) * (span / peak if peak > 0 else 0.0)
    return weights


def _match_weights(weight_dict, columns):
    """Weight of every clustering column in ``columns`` in the representative match.

    ``weight ** 2``: ward minimises squared distance, so that is the column's pull on the
    clustering, and the match weighs the columns the same way.
    """
    return pd.Series(
        {column: float(weight) ** 2 for column, weight in (weight_dict or {}).items() if column in columns},
        dtype="float64",
    )


def match_period_means(period_means, cluster_order, target_means, column_weights):
    """
    Pick, for every cluster, the member whose period means are closest to the cluster's.

    Each period is described by its mean of every weighted column, as the relative
    deviation from ``target_means`` scaled by ``sqrt(column_weights)``; the member with
    the smallest squared distance to its cluster's average of that vector is chosen.
    Columns with a non-positive target or weight are ignored.

    Parameters
    ----------
    period_means : pandas.DataFrame
        Mean of every column over each candidate period, in tsam's candidate order.
    cluster_order : array-like of int
        Cluster label of every candidate period (``agg.clusterOrder``).
    target_means : pandas.Series
        Mean of every column over the whole source timeline.
    column_weights : pandas.Series
        Non-negative weight of every column.

    Returns
    -------
    dict[int, int] or None
        ``{cluster label: candidate period index}``; None when no column is usable.
    """
    target = pd.Series(target_means, dtype="float64")
    weights = pd.Series(column_weights, dtype="float64")
    columns = [
        column for column in weights.index
        if weights[column] > 0 and column in period_means.columns
        and np.isfinite(target.get(column, np.nan)) and target[column] > 0
    ]
    if not columns:
        return None
    scaled = (period_means[columns].to_numpy() / target[columns].to_numpy() - 1.0) * np.sqrt(
        weights[columns].to_numpy(),
    )
    cluster_order = np.asarray(cluster_order)
    representatives = {}
    for label in np.unique(cluster_order):
        members = np.flatnonzero(cluster_order == label)
        member_values = scaled[members]
        distance = ((member_values - member_values.mean(axis=0)) ** 2).sum(axis=1)
        representatives[int(label)] = int(members[np.argmin(distance)])
    return representatives


def finalize_period_weights(period_entries, period_means, target_means):
    """
    Turn cluster-count weights into the final hours per snapshot (see "Period weights").

    The counts are rescaled to 8760 h, every extreme snapshot is raised to
    ``MIN_SNAPSHOT_WEIGHT_HOURS``, and the representative periods share the remaining
    hours in proportion to their counts. ``period_means`` holds each entry's feature
    means, row for row; it and ``target_means`` only feed the logged mean error. Returns
    new entries whose ``weightings`` columns all carry the final hours.
    """
    steps = np.array([int(entry["steps"]) for entry in period_entries], dtype="float64")
    counts = np.array([float(entry["weightings"]["objective"].mean()) for entry in period_entries])
    hours = counts * 8760.0 / float(np.sum(steps * counts))
    is_extreme = np.array([entry["kind"] == "extreme" for entry in period_entries])

    hours[is_extreme] = np.maximum(hours[is_extreme], MIN_SNAPSHOT_WEIGHT_HOURS)
    budget = 8760.0 - float(np.sum(steps[is_extreme] * hours[is_extreme]))
    representative_share = float(np.sum(steps[~is_extreme] * hours[~is_extreme]))
    if budget <= 0 or representative_share <= 0:
        raise ValueError(
            f"The extreme periods at {MIN_SNAPSHOT_WEIGHT_HOURS:g} h per snapshot leave no hours of the "
            "8760 h year for the representative periods.",
        )
    hours[~is_extreme] *= budget / representative_share

    columns = [column for column in _NATIONAL_FEATURES.values() if column in period_means.columns]
    errors = (period_means[columns].T @ (steps * hours) / 8760.0) / target_means[columns] - 1.0
    logger.info(
        "National mean error of the period weights: %s.",
        ", ".join(f"{column[-1]} {100 * error:+.2f}%" for column, error in errors.items()) or "n/a",
    )

    finalized = []
    for entry, entry_hours in zip(period_entries, hours):
        updated = dict(entry)
        updated["weightings"] = pd.DataFrame(
            float(entry_hours), index=entry["weightings"].index, columns=entry["weightings"].columns,
        )
        finalized.append(updated)
    return finalized


def select_period_entries(feature_t_full, source_index_full, representative_periods_cfg, weight_dict=None):
    """
    Select representative + extreme period entries for one source weather-year timeline.

    One tsam run partitions the timeline: hierarchical clustering on the max-normalized
    features (``max_normalized_weights``) forms ``number`` clusters, and -- when
    ``include_extreme`` is true -- the three single-feature extreme periods (peak demand,
    wind lull, solar lull) are appended with only their own period as member. Each
    cluster is represented by the real period whose national means match the cluster's
    best (``match_period_means``), and ``finalize_period_weights`` turns the cluster
    membership counts into the final hours per snapshot (see the module docstring).

    Parameters
    ----------
    feature_t_full : pandas.DataFrame
        Clustering features (national ``p_nom_max``-weighted onshore wind and solar CF and
        national AC load), indexed by ``source_index_full``.
    source_index_full : pandas.DatetimeIndex
        Full source timeline (e.g. concatenated 15 weather years).
    representative_periods_cfg : dict
        The ``clustering.temporal.representative_periods`` config block.
    weight_dict : dict[tuple, float], optional
        Per-column weight from ``build_feature_frame``. It weights the clustering (after
        ``max_normalized_weights``) and, as ``weight ** 2``, the representative match;
        entries naming a column the frame does not carry are dropped.

    Returns
    -------
    list[dict]
        Period entries (without a "snapshots" label yet -- see
        ``_assign_period_entry_snapshots``), each carrying a ``source_snapshots``
        DatetimeIndex that indexes directly into the raw source data and a
        ``weightings`` frame holding that period's final hours per snapshot.
    """
    import tsam.timeseriesaggregation as tsam

    number = int(representative_periods_cfg.get("number", 4))
    if number <= 0:
        raise ValueError("representative_periods.number must be a positive integer.")

    include_extreme = get_include_extreme(representative_periods_cfg)
    period_hours = get_period_hours(representative_periods_cfg)

    if feature_t_full.empty:
        raise ValueError(
            "No wind/solar capacity factor or AC load features found for representative period clustering.",
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

    mean_max_columns, mean_min_columns = [], []
    if include_extreme:
        mean_max_columns, mean_min_columns = resolve_extreme_selectors(feature_cluster)

    tsam_kwargs = dict(
        timeSeries=feature_cluster,
        hoursPerPeriod=period_hours,
        noTypicalPeriods=number,
        clusterMethod="hierarchical",
        # Only the partition is read out of tsam (cluster membership, extreme labels, and
        # the medoids as fallback); every value downstream is sliced from the raw source data
        # by ``source_snapshots``. Rescaling only rewrites ``agg.typicalPeriods``, which
        # nothing here touches, so it is switched off rather than left to spend time --
        # and warn about its convergence -- on a series that is never used.
        rescaleClusterPeriods=False,
    )
    if weight_dict:
        tsam_kwargs["weightDict"] = max_normalized_weights(weight_dict, feature_cluster)
    if mean_max_columns or mean_min_columns:
        tsam_kwargs.update(
            extremePeriodMethod="append",
            addMeanMax=mean_max_columns,
            addMeanMin=mean_min_columns,
        )

    agg = tsam.TimeSeriesAggregation(**tsam_kwargs)
    agg.createTypicalPeriods()

    matching = agg.indexMatching()
    matching_idx = pd.MultiIndex.from_arrays(
        [matching["PeriodNum"].to_numpy(), matching["TimeStep"].to_numpy()], names=["PeriodNum", "TimeStep"],
    )
    period_ids = sorted(int(label) for label in matching["PeriodNum"].unique())

    # Candidate periods are consecutive blocks of ``period_steps`` rows of feature_cluster,
    # in the order tsam labels them in ``clusterOrder``.
    candidate_means = pd.DataFrame(
        feature_cluster.to_numpy().reshape(len(source_period_rows), period_steps, -1).mean(axis=1),
        columns=feature_cluster.columns,
    )
    target_means = feature_cluster.mean()
    representatives = match_period_means(
        candidate_means,
        agg.clusterOrder,
        target_means,
        _match_weights(weight_dict, feature_cluster.columns),
    )
    if representatives is None:
        logger.info("No weighted columns to match period means on; representing clusters by tsam's medoids.")
    source_periods = _build_representative_period_mapping(agg, period_ids, representatives)
    extreme_period_ids = _get_extreme_period_ids(agg, period_ids)

    # One unit of weight per source snapshot, summed per (period, step) by the tsam
    # matching: every period's weight is the number of source periods it represents.
    sw_source = pd.DataFrame(1.0, index=feature_cluster.index, columns=["objective", "stores", "generators"])
    sw_local = _build_representative_snapshot_weightings(
        sw_source,
        matching_idx,
        pd.MultiIndex.from_product([period_ids, range(period_steps)], names=["PeriodNum", "TimeStep"]),
    )

    period_entries, period_means = [], []
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
        period_means.append(feature_t_full.iloc[positions].mean())

    # The target is the mean over the same candidate periods the counts partition.
    period_entries = finalize_period_weights(period_entries, pd.DataFrame(period_means), target_means)

    logger.info(
        "Clustered %s source periods into %s periods (%s extreme): %s.",
        len(source_period_rows),
        len(period_entries),
        len(extreme_period_ids),
        ", ".join(
            f"P{entry['period_id']}"
            f"{'(extreme)' if entry['period_id'] in extreme_period_ids else ''}"
            f"={entry['weightings']['objective'].iloc[0]:.3g} h {entry['source_snapshots'][0]:%Y-%m-%d}"
            for entry in period_entries
        ),
    )
    return period_entries


def select_representative_snapshots(
    feature_t_full,
    source_index_full,
    representative_periods_cfg,
    investment_periods,
    weight_dict=None,
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
        feature_t_full, source_index_full, representative_periods_cfg, weight_dict=weight_dict,
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
# National clustering features, built straight from the raw sources.
# --------------------------------------------------------------------------- #


def read_reeds_national_capacity_factor(carriers, reeds_vre_dir, weather_years):
    """
    Build the national wind/solar clustering features from the raw ReEDS files.

    The feature is the ``p_nom_max``-weighted mean capacity factor,
    ``sum_i(capacity_i * cf_i(t)) / sum_i(capacity_i)``, taken over every site in the
    ReEDS supply curve. Selection runs before ``build_renewable_profiles``, so there are
    no per-bus profile files to read yet, and this needs nothing but the supply curve and
    the CF table.

    Carriers in the same feature group are combined by capacity, so numerator and
    denominator accumulate separately and are divided once at the end. Carriers
    sharing a supply curve are read once, so two carriers split from the same ReEDS
    sites downstream are never counted twice. Configured carriers outside
    ``REEDS_FEATURE_GROUPS`` -- offshore wind among them -- are ignored.

    The solar inverter loading ratio is applied per site before aggregating,
    matching the downstream clip at 1.0.

    Returns
    -------
    dict[str, pandas.Series | None]
        ``{"wind": ..., "solar": ...}``: the national mean capacity factor over the full
        weather-year timeline, or ``None`` for a group with no configured carriers.
    """
    # Imported lazily: build_reeds_renewable_profiles imports this module for
    # read_representative_snapshots, so a top-level import would be circular.
    from build_reeds_renewable_profiles import (
        REEDS_TECH,
        SOLAR_INVERTER_LOADING_RATIO,
        national_available_generation,
        read_cf_time_indexes,
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
            # One open for every weather year: these CF tables are multi-gigabyte.
            cf_time_indexes = read_cf_time_indexes(cf_path, weather_years)
            parts = []
            for weather_year in weather_years:
                values = national_available_generation(
                    cf_path,
                    capacities.index.tolist(),
                    capacities,
                    int(weather_year),
                    transform=transform,
                )
                index = cf_time_indexes[int(weather_year)]
                if len(values) != len(index):
                    raise ValueError(
                        f"ReEDS {carrier} {weather_year} returned {len(values)} hours; "
                        f"expected {len(index)}.",
                    )
                parts.append(pd.Series(values, index=index))

            series = pd.concat(parts)
            capacity = float(capacities.sum())
            logger.info(
                "Aggregated %s ReEDS %s sites (%.0f MW nameplate); mean CF %.4f.",
                len(capacities),
                carrier,
                capacity,
                float(series.mean() / capacity),
            )
            if numerator is None:
                numerator = series
            elif not series.index.equals(numerator.index):
                raise ValueError(
                    f"Carriers contributing to the '{group}' feature do not share an identical "
                    "time index; all must be built on the same weather-year calendar.",
                )
            else:
                # Several carriers in one group: numerator and denominator both
                # accumulate, so the division at the end is the combined
                # capacity-weighted mean, not a mean of means.
                numerator = numerator + series
            total_capacity += capacity

        if numerator is None or total_capacity <= 0:
            profiles[group] = None
            continue

        profiles[group] = numerator / total_capacity
        logger.info(
            "Combined %s feature: national mean CF %.4f over %s hours (%.0f MW nameplate).",
            group,
            float(profiles[group].mean()),
            len(profiles[group]),
            total_capacity,
        )
    return profiles


def build_feature_frame(capacity_factors, national_demand):
    """
    Assemble the clustering feature frame and its tsam column weights.

    The frame carries one column per feature family -- national ``p_nom_max``-weighted
    onshore wind CF, solar CF and AC demand -- restricted to the UTC hours every feature
    shares. The three columns weigh equally; ``max_normalized_weights`` then puts them on
    a per-unit-of-maximum scale (see the module docstring).

    Parameters
    ----------
    capacity_factors : dict[str, pandas.Series | None]
        ``{"wind": ..., "solar": ...}`` from ``read_reeds_national_capacity_factor``.
    national_demand : pandas.Series | None
        National AC demand for the planning horizon, indexed by actual UTC source
        timestamps. Build it with ``read_national_demand``.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, pandas.Series | None], dict[tuple, float]]
        The feature frame, the national profiles keyed ``"wind"`` / ``"solar"`` /
        ``"load"`` that the diagnostic plots reuse, and the tsam ``weightDict``.
    """
    profiles = {name: capacity_factors.get(name) for name in ("wind", "solar")}
    profiles["load"] = None
    if national_demand is not None and len(national_demand):
        demand = pd.to_numeric(pd.Series(national_demand), errors="coerce").astype("float64")
        demand.index = pd.DatetimeIndex(demand.index)
        profiles["load"] = demand

    available = {name: profile for name, profile in profiles.items() if profile is not None and len(profile)}
    if not available:
        raise ValueError(
            "No wind/solar generation or AC demand available for representative-period clustering.",
        )

    # EER is published in fixed CST and VRE in UTC.  Both omit Dec. 31 in leap
    # years, so row counts cannot establish physical alignment.  Use only exact
    # timestamp overlap, which also makes any unpaired boundary hours explicit.
    common_index = None
    for profile in available.values():
        index = pd.DatetimeIndex(profile.index)
        common_index = index if common_index is None else common_index.intersection(index)
    common_index = pd.DatetimeIndex(common_index).sort_values()
    if common_index.empty:
        raise ValueError("Demand and renewable profiles have no shared UTC timestamps.")
    for name, profile in available.items():
        excluded = len(pd.DatetimeIndex(profile.index).difference(common_index))
        if excluded:
            logger.info(
                "Excluding %s %s source hours without a physical UTC match in every clustering feature.",
                excluded,
                name,
            )

    features = pd.DataFrame(
        {_NATIONAL_FEATURES[name]: profile.reindex(common_index) for name, profile in available.items()},
        index=common_index,
    )
    features.columns = pd.MultiIndex.from_tuples(features.columns)
    features = features.replace([np.inf, -np.inf], np.nan)

    incomplete = features.columns[features.isna().any()]
    if len(incomplete):
        logger.warning(
            "Dropping %s clustering column(s) with missing hours: %s.",
            len(incomplete),
            ", ".join(str(column[-1]) for column in incomplete),
        )
        features = features.drop(columns=incomplete)
    if features.empty or not len(features.columns):
        raise ValueError("No clustering feature survived the shared-timestamp and completeness checks.")

    weight_dict = {column: 1.0 for column in features.columns}
    logger.info(
        "Built clustering features over %s source snapshots: %s.",
        len(features),
        ", ".join(str(column[-1]) for column in features.columns),
    )
    return features, profiles, weight_dict


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
            f"(first: {missing[0]}). Check that it uses the same published UTC "
            "timestamps as the representative-period selection.",
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


def read_national_demand(
    demand_path: str,
    planning_horizon: int,
    weather_years,
) -> pd.Series:
    """
    Read national AC demand for one planning horizon, rolled to UTC.

    EER publishes one column per state; the national feature is their row sum.

    Distribution losses are deliberately not applied: they are a uniform
    multiplier and therefore affect neither the clustering (tsam
    normalizes each feature) nor the extreme-period ranking.
    """
    # Imported lazily: build_eer_demand imports this module for
    # read_representative_snapshots, so a top-level import would be circular.
    from build_eer_demand import ReadEer

    demand = ReadEer(demand_path, [planning_horizon], weather_years).read()
    demand = demand.loc[planning_horizon].astype("float64")
    demand.index = pd.DatetimeIndex(demand.index)
    national = demand.sum(axis="columns")
    logger.info(
        "Read AC demand for %s across %s states: %s snapshots, national mean %.1f MW.",
        planning_horizon,
        demand.shape[1],
        len(national),
        float(national.mean()),
    )
    return national


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
    expected_extremes = len(EXTREME_SELECTORS) if get_include_extreme(representative_periods) else 0
    period_hours = get_period_hours(representative_periods)

    for horizon, entries in period_entries_by_horizon.items():
        representative_entries = [entry for entry in entries if entry.get("kind") != "extreme"]
        extreme_entries = [entry for entry in entries if entry.get("kind") == "extreme"]
        if len(representative_entries) != number:
            raise ValueError(
                f"Representative selection produced {len(representative_entries)} representative periods "
                f"for {horizon}; expected {number}.",
            )
        if len(extreme_entries) != expected_extremes:
            logger.warning(
                "Selection produced %s extreme periods for %s instead of the %s requested: tsam drops an "
                "extreme period that is already a cluster center, and a selector is skipped when its "
                "clustering feature is missing or degenerate.",
                len(extreme_entries),
                horizon,
                expected_extremes,
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

    capacity_factors = read_reeds_national_capacity_factor(
        params.renewable_carriers,
        params.reeds_vre_dir,
        params.renewable_weather_years,
    )

    national_demand = read_national_demand(
        snakemake.input.electricity_demand,
        planning_horizon,
        params.renewable_weather_years,
    )

    feature_t_full, profile_map, weight_dict = build_feature_frame(capacity_factors, national_demand)

    snapshots, snapshot_weightings, period_entries_by_horizon, metadata = select_representative_snapshots(
        feature_t_full,
        feature_t_full.index,
        representative_periods,
        [planning_horizon],
        weight_dict=weight_dict,
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

        snakemake = mock_snakemake("select_representative_periods", case="test")
    configure_logging(snakemake)
    main(snakemake)
