"""
Representative-period selection: everything in one place.

This module owns the whole representative-period story:

* the tsam hierarchical-clustering selection itself, run on a per-state feature frame
  (capacity-weighted mean wind CF, capacity-weighted mean solar CF, AC demand -- one
  column per state) computed straight from the raw ReEDS supply curves / CF tables and
  the EER demand h5, alongside the three national aggregates the extreme periods are
  ranked on;
* the snapshot definition it writes out (``snapshots.csv``, ``metadata.json``,
  ``profiles.png``) and the readers downstream rules use to consume it;
* the diagnostic profile plots.

It runs at the very top of the electricity workflow -- *before*
``build_renewable_profiles`` -- so the selected hours are known before any
per-bus profile or network exists. ``build_renewable_profiles`` then builds only
those hours, and every downstream rule attaches time series for those hours only,
so the full 15-weather-year x 8760 h timeline is never materialised anywhere.

Spatial resolution and column weights
-------------------------------------
The wind/solar features are aggregated over **every** site in the ReEDS supply curve,
grouped by the state the site sits in. A site's state comes from the county FIPS the
interconnection tables already carry per ``sc_point_gid``, so no spatial join and no bus
region is needed and this step still depends on nothing but static data files -- which
matters, because it runs before any network exists. Demand needs no mapping at all: the
EER h5 is published per state.

Grouping is free on disk. ``grouped_available_generation`` already reads every site
column of each row block, so the only change is the collapse at the end
(``block @ W`` instead of ``block.dot(weights)``), and the national aggregate falls out
as the row sum of the same pass.

tsam min-max normalizes every column, which would give Rhode Island's wind profile the
same pull as Texas'. Columns are therefore weighted by maximum capacity potential --
supply-curve nameplate for wind and solar, mean AC demand for load. Note the **square
root**: ``weightDict`` scales a column linearly but ``clusterMethod="hierarchical"`` is
ward, which minimises *squared* Euclidean distance, so a column's pull on the clustering
goes as ``weight**2``. Weights are ``sqrt(share)``, which makes the pull proportional to
share and makes each feature family's weights satisfy ``sum(w**2) == 1`` -- so wind,
solar and load contribute equally in total, exactly as the three unweighted national
columns used to.

``onwind`` and ``offwind`` stay merged into one capacity-weighted ``wind`` feature per
state. Offshore is 42% of national supply-curve nameplate and dominates the wind column
of 16 coastal states, so roughly half the wind weight rides on offshore shapes; that is
deliberate, but it is also why the wind feature is the one to revisit first if the
selected periods look wrong.

Two pieces of the older in-network implementation are intentionally gone: the
"force one spring + one fall representative period" seasonal constraint, and the
post-hoc rescaling of non-extreme snapshots to match true annual energy totals.
Representative periods are used exactly as tsam selects them.

Extreme periods and weighting
-----------------------------
``include_extreme`` is a plain on/off switch. When it is true, three extreme periods
are requested -- one single-resource stress case per clustering feature:

* **demand max**: the period with the highest mean *national* AC demand;
* **wind min**: the period with the lowest mean *national* wind capacity factor;
* **solar min**: the period with the lowest mean *national* solar capacity factor.

Each is ranked on one raw national aggregate, so the three are the three physical
stress cases the system has to survive on their own terms -- the peak-load block, the
wind lull and the solar lull -- rather than one blended metric that can only ever
return whichever stress the year happens to be worst at. They stay national even though
the clustering is per state: a stress case is a system-wide event, and ranking on one
state's lull would pick whichever state happens to be calmest, not the worst hour for
the system.

tsam's ``addMeanMax`` / ``addMeanMin`` can only rank on columns of the frame it is
handed, so the three national aggregates ride along *in* the clustering frame. They are
given ``MIN_WEIGHT`` (1e-6) in ``weightDict`` -- four orders below the smallest state
weight -- so they are invisible to the clustering distance while remaining available to
rank on. That is safe because tsam ranks extremes on the weighted, normalized profiles
and a positive constant scale is order-preserving, so a down-weighted national column
picks exactly the period its raw counterpart would. A feature that is missing -- no wind
carrier configured, say -- or degenerate (flat over the whole timeline, so it carries no
ranking signal at all) is dropped with a warning rather than failing the run.

tsam picks the period with the highest (or lowest) period mean per ranking feature, adds it as
an *additional cluster center*, and re-checks every other period against it -- a
period closer to the extreme than to its own medoid joins the extreme's cluster.
Weights therefore fall out of the cluster membership counts for representative
and extreme periods alike: an extreme week represents all the weeks that look
like it, not only its own hours. Snapshot weightings are finally rescaled so
``objective`` sums to 8760 h per planning horizon.

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
# group are combined by capacity into one mean-capacity-factor series -- onshore and
# offshore wind therefore share a single "wind" feature per state (see the module
# docstring on what that costs in coastal states).
WIND_CARRIERS = ("onwind", "offwind", "offwind_floating")
SOLAR_CARRIERS = ("solar",)
REEDS_FEATURE_GROUPS = {"wind": WIND_CARRIERS, "solar": SOLAR_CARRIERS}

WIND_FEATURE = ("Generator", "p_max_pu", "wind")
SOLAR_FEATURE = ("Generator", "p_max_pu", "solar")
LOAD_FEATURE = ("Load", "p_set", "ac_load")

# Per-state columns reuse the national tuples with the state code appended to the last
# level, so the frame keeps one flat 3-level MultiIndex that tsam can key a weightDict
# and addMeanMax/addMeanMin off.
_NATIONAL_FEATURES = {"wind": WIND_FEATURE, "solar": SOLAR_FEATURE, "load": LOAD_FEATURE}


def state_feature(kind, state):
    """Return the frame column for one state's ``kind`` ("wind"/"solar"/"load") feature."""
    prefix = _NATIONAL_FEATURES[kind]
    return (*prefix[:-1], f"{prefix[-1]}_{state}")


# tsam clips any weightDict entry below this and prints a notice, so the national
# ranking columns are pinned exactly at the floor rather than below it.
NATIONAL_COLUMN_WEIGHT = 1e-6

# The three extreme periods, one per clustering feature: the peak-load block, the wind
# lull and the solar lull. Each is ranked on the *raw* feature, so each extreme is the
# period that stresses one resource hardest rather than the one that scores worst on a
# blended metric. ``direction`` picks the tsam argument the feature is passed to --
# "max" -> ``addMeanMax``, "min" -> ``addMeanMin`` -- and both rank on the period mean.
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
        "fixed extremes: the period with the highest mean demand, the one with the lowest mean wind "
        "capacity factor, and the one with the lowest mean solar capacity factor.",
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


def select_period_entries(feature_t_full, source_index_full, representative_periods_cfg, weight_dict=None):
    """
    Select representative + extreme period entries for one source weather-year timeline.

    One tsam run does everything: hierarchical clustering picks ``number``
    representative periods (represented by the real historical period closest to
    each cluster centroid, tsam's ``medoidRepresentation``), and -- when
    ``include_extreme`` is true -- tsam's ``new_cluster_center`` method adds the
    three single-feature extreme periods (peak demand, wind lull, solar lull) as
    further cluster centers and reassigns whatever is closer to them. Period weights
    are the resulting cluster membership counts (see the module docstring).

    Parameters
    ----------
    feature_t_full : pandas.DataFrame
        Clustering features (per-state weighted wind/solar CF and AC load, plus the
        three national aggregates the extremes rank on), indexed by
        ``source_index_full``.
    source_index_full : pandas.DatetimeIndex
        Full source timeline (e.g. concatenated 15 weather years).
    representative_periods_cfg : dict
        The ``clustering.temporal.representative_periods`` config block.
    weight_dict : dict[tuple, float], optional
        Per-column tsam ``weightDict``. Build it with ``build_clustering_weights``;
        entries naming a column the frame does not carry are dropped.

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

    include_extreme = get_include_extreme(representative_periods_cfg)
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

    mean_max_columns, mean_min_columns = [], []
    if include_extreme:
        mean_max_columns, mean_min_columns = resolve_extreme_selectors(feature_cluster)

    tsam_kwargs = dict(
        timeSeries=feature_cluster,
        hoursPerPeriod=period_hours,
        noTypicalPeriods=number,
        clusterMethod="hierarchical",
        # Only the period *mapping* is read out of tsam (medoid indices, extreme labels,
        # cluster membership); every value downstream is sliced from the raw source data
        # by ``source_snapshots``. Rescaling only rewrites ``agg.typicalPeriods``, which
        # nothing here touches, so it is switched off rather than left to spend time --
        # and warn about its convergence -- on a series that is never used.
        rescaleClusterPeriods=False,
    )
    if weight_dict:
        # Only columns the frame actually carries: tsam raises on an unknown key, and
        # build_feature_frame drops a column whose series never materialised.
        tsam_kwargs["weightDict"] = {
            column: float(weight) for column, weight in weight_dict.items() if column in feature_cluster.columns
        }
    if mean_max_columns or mean_min_columns:
        tsam_kwargs.update(
            extremePeriodMethod="new_cluster_center",
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
# Per-state clustering features, built straight from the raw sources.
# --------------------------------------------------------------------------- #


def read_site_states(interconnection_dir, counties_path, interconnection_file):
    """
    Map every ``sc_point_gid`` in one interconnection table onto its USPS state code.

    The interconnection tables already carry a 5-digit county FIPS per site -- it is
    what ``build_reeds_renewable_profiles.load_sites`` joins on -- so the state comes
    out of a plain lookup rather than a spatial join against bus regions that do not
    exist yet at this point in the workflow. It also gives offshore sites a state,
    which a point-in-polygon join against onshore regions cannot.

    Returns
    -------
    pandas.Series
        USPS state code indexed by ``sc_point_gid``; sites whose FIPS matches no
        county are absent, and the caller reports their capacity rather than
        silently folding them into another state.
    """
    import geopandas as gpd
    import tables

    # Attributes only: the geometry is irrelevant here and parsing 3,234 county
    # polygons to read two columns would be pure waste.
    counties = gpd.read_file(counties_path, columns=["GEOID", "STUSPS"], ignore_geometry=True)
    fips_to_state = pd.Series(counties.STUSPS.to_numpy(), index=counties.GEOID.astype(str))
    fips_to_state = fips_to_state[~fips_to_state.index.duplicated()]

    with tables.open_file(f"{interconnection_dir}/{interconnection_file}", "r") as h5:
        gids = h5.get_node("/data/sc_point_gid")[:]
        fips = [value.decode() if isinstance(value, bytes) else str(value) for value in h5.get_node("/data/FIPS")[:]]
    site_fips = pd.Series(fips, index=pd.Index(gids, name="sc_point_gid"))
    site_fips = site_fips.groupby(level=0).first()
    return site_fips.map(fips_to_state).dropna()


def build_clustering_weights(capacity_by_state):
    """
    Turn one feature family's per-state capacity potential into tsam column weights.

    ``weightDict`` scales a column linearly, but ``clusterMethod="hierarchical"`` is
    ward, which minimises *squared* Euclidean distance: a column's pull on the
    clustering goes as ``weight**2``. The weight is therefore ``sqrt(share)``, which
    makes the pull proportional to the state's share of the family's national total
    and makes ``sum(w**2)`` come out at 1 -- so every family contributes the same
    total pull no matter how many states it spans, matching what the three unweighted
    national columns used to do.

    Parameters
    ----------
    capacity_by_state : pandas.Series
        Maximum capacity potential per state: supply-curve nameplate for wind and
        solar, mean AC demand for load. Non-positive and missing entries drop out.

    Returns
    -------
    pandas.Series
        ``sqrt(share)`` indexed by state code; empty when nothing positive is left.
    """
    capacity = pd.Series(capacity_by_state, dtype="float64").dropna()
    capacity = capacity[capacity > 0]
    if capacity.empty:
        return pd.Series(dtype="float64")
    return np.sqrt(capacity / capacity.sum())


def read_reeds_state_capacity_factor(
    carriers,
    reeds_vre_dir,
    interconnection_dir,
    counties_path,
    weather_years,
):
    """
    Build the per-state wind/solar clustering features from the raw ReEDS files.

    The feature is the capacity-weighted mean capacity factor,
    ``sum_i(capacity_i * cf_i(t)) / sum_i(capacity_i)``, taken over every site in the
    ReEDS supply curve and grouped by the state the site sits in. Selection runs before
    ``build_renewable_profiles``, so there are no per-bus profile files to read yet, and
    this needs nothing but the supply curve, the CF table, and the county FIPS the
    interconnection table already carries.

    The national aggregate the extreme periods rank on is the row sum of the same
    grouped numerator over the summed denominator, so it costs no extra pass over the
    multi-gigabyte CF tables and stays exactly consistent with the state columns.

    Carriers in the same feature group are combined by capacity, so numerator and
    denominator accumulate separately and are divided once at the end. Carriers
    sharing a supply curve are read once: ``offwind`` and ``offwind_floating`` are
    the same ReEDS sites split by water depth downstream, so counting both would
    double the offshore contribution.

    The solar inverter loading ratio is applied per site before aggregating,
    matching the downstream clip at 1.0.

    Returns
    -------
    dict[str, dict | None]
        ``{"wind": ..., "solar": ...}``; ``None`` for a group with no configured
        carriers, else ``{"states": DataFrame, "national": Series, "capacity":
        Series}`` -- mean capacity factor per state over the full weather-year
        timeline, the national mean, and the supply-curve nameplate per state that
        ``build_clustering_weights`` turns into column weights.
    """
    # Imported lazily: build_reeds_renewable_profiles imports this module for
    # read_representative_snapshots, so a top-level import would be circular.
    from build_reeds_renewable_profiles import (
        REEDS_TECH,
        SOLAR_INVERTER_LOADING_RATIO,
        grouped_available_generation,
        read_cf_time_indexes,
    )

    def solar_transform(block):
        return np.clip(block * SOLAR_INVERTER_LOADING_RATIO, None, 1.0)

    carriers = set(carriers)
    site_state_cache = {}
    profiles = {}
    for group, group_carriers in REEDS_FEATURE_GROUPS.items():
        numerator = None
        capacity_by_state = None
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

            if cfg["interconnection"] not in site_state_cache:
                site_state_cache[cfg["interconnection"]] = read_site_states(
                    interconnection_dir, counties_path, cfg["interconnection"],
                )
            site_states = capacities.index.to_series().map(site_state_cache[cfg["interconnection"]])
            unassigned = float(capacities[site_states.isna().to_numpy()].sum())
            if unassigned > 0:
                logger.warning(
                    "%.0f MW of %s nameplate (%.2f%%) sits at sites whose county FIPS matches no "
                    "state and is excluded from the clustering features.",
                    unassigned,
                    carrier,
                    100 * unassigned / float(capacities.sum()),
                )
            carrier_capacity = capacities.groupby(site_states.to_numpy()).sum()
            states = sorted(carrier_capacity.index)
            if not states:
                logger.warning("No %s site could be placed in a state. Skipping it.", carrier)
                continue

            transform = solar_transform if carrier in SOLAR_CARRIERS else None
            cf_path = f"{reeds_vre_dir}/{cfg['cf']}"
            # One open for every weather year: these CF tables are multi-gigabyte.
            cf_time_indexes = read_cf_time_indexes(cf_path, weather_years)
            parts = []
            for weather_year in weather_years:
                values = grouped_available_generation(
                    cf_path,
                    capacities.index.tolist(),
                    capacities,
                    int(weather_year),
                    groups=site_states.to_numpy(),
                    group_labels=states,
                    transform=transform,
                )
                index = cf_time_indexes[int(weather_year)]
                if len(values) != len(index):
                    raise ValueError(
                        f"ReEDS {carrier} {weather_year} returned {len(values)} hours; "
                        f"expected {len(index)}.",
                    )
                values.index = index
                parts.append(values)

            frame = pd.concat(parts)
            logger.info(
                "Aggregated %s ReEDS %s sites across %s states (%.0f MW nameplate); mean CF %.4f.",
                len(capacities),
                carrier,
                len(states),
                float(carrier_capacity.sum()),
                float(frame.to_numpy().sum() / (len(frame) * float(carrier_capacity.sum()))),
            )
            if numerator is None:
                numerator, capacity_by_state = frame, carrier_capacity
            elif not frame.index.equals(numerator.index):
                raise ValueError(
                    f"Carriers contributing to the '{group}' feature do not share an identical "
                    "time index; all must be built on the same weather-year calendar.",
                )
            else:
                # Merged onshore + offshore: numerator and denominator both accumulate
                # per state, so the division at the end is the combined capacity-weighted
                # mean, not a mean of means.
                numerator = numerator.add(frame, fill_value=0.0)
                capacity_by_state = capacity_by_state.add(carrier_capacity, fill_value=0.0)

        if numerator is None or float(capacity_by_state.sum()) <= 0:
            profiles[group] = None
            continue

        capacity_by_state = capacity_by_state.reindex(numerator.columns).fillna(0.0)
        state_cf = numerator.divide(capacity_by_state.replace(0.0, np.nan), axis="columns")
        national_cf = numerator.sum(axis="columns") / float(capacity_by_state.sum())
        profiles[group] = {
            "states": state_cf,
            "national": national_cf,
            "capacity": capacity_by_state,
        }
        logger.info(
            "Combined %s feature: %s state columns, national mean CF %.4f over %s hours "
            "(%.0f MW nameplate).",
            group,
            state_cf.shape[1],
            float(national_cf.mean()),
            len(national_cf),
            float(capacity_by_state.sum()),
        )
    return profiles


def build_feature_frame(carrier_profiles, state_demand):
    """
    Assemble the clustering feature frame and its tsam column weights.

    The frame carries one column per state and feature family -- wind, solar, AC
    demand -- plus the three national aggregates the extreme selectors rank on. State
    columns are weighted by ``sqrt(share)`` of maximum capacity potential; the national
    columns are pinned at ``NATIONAL_COLUMN_WEIGHT`` so they cannot pull on the
    clustering (see the module docstring on why the square root, and why the national
    columns have to be in the frame at all).

    Parameters
    ----------
    carrier_profiles : dict[str, dict | None]
        ``{"wind": ..., "solar": ...}`` from ``read_reeds_state_capacity_factor``;
        each value holds ``"states"``, ``"national"`` and ``"capacity"``.
    state_demand : pandas.DataFrame
        AC demand per state for the planning horizon, indexed by actual UTC source
        timestamps. Build it with ``read_state_demand``.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, pandas.Series | None], dict[tuple, float]]
        The feature frame, the national profiles keyed ``"wind"`` / ``"solar"`` /
        ``"load"`` that the diagnostic plots and the extreme selectors reuse, and the
        tsam ``weightDict``.
    """
    state_frames, national_profiles, capacity_potential = {}, {}, {}
    for name in ("wind", "solar"):
        group = carrier_profiles.get(name)
        if not group:
            national_profiles[name] = None
            continue
        state_frames[name] = group["states"]
        national_profiles[name] = group["national"]
        capacity_potential[name] = group["capacity"]

    national_profiles["load"] = None
    if state_demand is not None and len(state_demand.columns) and len(state_demand):
        demand = state_demand.apply(pd.to_numeric, errors="coerce").astype("float64")
        demand.index = pd.DatetimeIndex(demand.index)
        state_frames["load"] = demand
        national_profiles["load"] = demand.sum(axis="columns")
        # Mean AC demand is the load-side analogue of nameplate potential: the state's
        # standing share of the system, not the one hour it happens to peak in.
        capacity_potential["load"] = demand.mean(axis="rows")

    # EER is published in fixed CST and VRE in UTC.  Both omit Dec. 31 in leap
    # years, so row counts cannot establish physical alignment.  Use only exact
    # timestamp overlap, which also makes any unpaired boundary hours explicit.
    if not state_frames:
        raise ValueError(
            "No wind/solar generation or AC demand available for representative-period clustering.",
        )
    common_index = None
    for frame in state_frames.values():
        common_index = frame.index if common_index is None else common_index.intersection(frame.index)
    common_index = pd.DatetimeIndex(common_index).sort_values()
    if common_index.empty:
        raise ValueError("Demand and renewable profiles have no shared UTC timestamps.")
    for name, frame in state_frames.items():
        excluded = len(frame.index.difference(common_index))
        if excluded:
            logger.info(
                "Excluding %s %s source hours without a physical UTC match in every clustering feature.",
                excluded,
                name,
            )

    feature_map, weight_dict = {}, {}
    for name, frame in state_frames.items():
        weights = build_clustering_weights(capacity_potential[name])
        for state in frame.columns:
            if state not in weights.index:
                logger.info(
                    "Dropping the %s %s clustering column: no positive capacity potential.",
                    state,
                    name,
                )
                continue
            column = state_feature(name, state)
            feature_map[column] = frame[state].reindex(common_index)
            weight_dict[column] = float(weights[state])

    # The national aggregates ride along only so the extreme selectors have something
    # to rank on; NATIONAL_COLUMN_WEIGHT keeps them out of the clustering distance.
    for name, profile in national_profiles.items():
        if profile is None:
            continue
        column = _NATIONAL_FEATURES[name]
        feature_map[column] = profile.reindex(common_index)
        weight_dict[column] = NATIONAL_COLUMN_WEIGHT

    features = pd.DataFrame(feature_map, index=common_index)
    features.columns = pd.MultiIndex.from_tuples(features.columns)
    features = features.replace([np.inf, -np.inf], np.nan)

    # A column that never materialised is dropped, but loudly: at three national
    # columns this was a no-op, at ~145 it silently decides what gets clustered.
    incomplete = features.columns[features.isna().any()]
    if len(incomplete):
        logger.warning(
            "Dropping %s clustering column(s) with missing hours: %s.",
            len(incomplete),
            ", ".join(str(column[-1]) for column in incomplete),
        )
        features = features.drop(columns=incomplete)
    weight_dict = {column: weight for column, weight in weight_dict.items() if column in features.columns}
    if features.empty or not len(features.columns):
        raise ValueError("No clustering feature survived the shared-timestamp and completeness checks.")

    state_columns = [column for column in features.columns if tuple(column) not in _NATIONAL_FEATURES.values()]
    logger.info(
        "Built clustering features over %s source snapshots: %s state columns + %s national "
        "ranking columns.",
        len(features),
        len(state_columns),
        len(features.columns) - len(state_columns),
    )
    for name in state_frames:
        family = {
            column: weight
            for column, weight in weight_dict.items()
            if column != _NATIONAL_FEATURES[name] and column[-1].startswith(f"{_NATIONAL_FEATURES[name][-1]}_")
        }
        if family:
            logger.info(
                "  %s: %s columns, weight %.4f-%.4f, sum(w^2)=%.3f.",
                name,
                len(family),
                min(family.values()),
                max(family.values()),
                float(np.square(list(family.values())).sum()),
            )
    return features, national_profiles, weight_dict


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


def read_state_demand(
    demand_path: str,
    planning_horizon: int,
    weather_years,
) -> pd.DataFrame:
    """
    Read per-state AC demand for one planning horizon, rolled to UTC.

    EER publishes one column per state, so the clustering resolution needs no mapping
    here at all -- unlike the VRE side, which has to place supply-curve sites first.
    The national total the extreme selectors rank on is the row sum, taken in
    ``build_feature_frame``.

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
    demand = demand.reindex(sorted(demand.columns), axis="columns")
    logger.info(
        "Read AC demand for %s across %s states: %s snapshots, national mean %.1f MW.",
        planning_horizon,
        demand.shape[1],
        len(demand),
        float(demand.sum(axis="columns").mean()),
    )
    return demand


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
    a cluster center -- or that an earlier selector already claimed -- instead of
    taking the next-most-extreme candidate. That is reported as a warning, not an error.
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
                "extreme period that is already a cluster center or that another selector already claimed, "
                "and a selector is skipped when its clustering feature is missing or degenerate.",
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

    carrier_profiles = read_reeds_state_capacity_factor(
        params.renewable_carriers,
        params.reeds_vre_dir,
        params.reeds_interconnection_dir,
        snakemake.input.counties,
        params.renewable_weather_years,
    )

    state_demand = read_state_demand(
        snakemake.input.electricity_demand,
        planning_horizon,
        params.renewable_weather_years,
    )

    feature_t_full, profile_map, weight_dict = build_feature_frame(carrier_profiles, state_demand)

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
