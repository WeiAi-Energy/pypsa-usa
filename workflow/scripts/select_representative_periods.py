"""
Representative periods for electricity-only networks.

``prepare_network`` calls :func:`apply_representative_periods` when
``clustering.temporal.representative_periods.enable`` is true. Per investment
period, it keeps three extreme periods and ``number`` typical periods of the
network's own time series and drops every other snapshot.

Features
    Three national series per snapshot: the onshore wind and the solar capacity
    factor, each averaged over the extendable candidates active in the investment
    period with their ``p_nom_max`` as weights, and total demand (the sum of every
    Load's ``p_set``).

Source periods
    Consecutive, non-overlapping blocks of ``period_length`` hours counted from the
    first snapshot of the investment period. A trailing remainder too short for a
    whole block is never selected, but its hours still count toward the weights.

Extreme periods
    The block with the highest mean demand, the one with the lowest mean wind
    capacity factor and the one with the lowest mean solar capacity factor, picked in
    that order. A block one selector has already taken passes to the next-ranked
    block, so there are always three distinct extremes. Each represents only itself.

Typical periods
    Ward hierarchical clustering of the remaining blocks on their profiles, each
    feature divided by its maximum over the investment period. A cluster is
    represented by the member whose mean wind capacity factor, solar capacity factor
    and demand (each relative to its mean over the investment period) are closest to
    the cluster's.

Weights
    ``objective`` and ``generators`` are multiplied by the block's weight: 1 for an
    extreme period, and for a typical period its cluster size scaled so that the
    weightings of each investment period sum to what they did before (8760 h for a
    full year). ``stores`` keeps the length of one timestep: PyPSA reads it as the
    hours elapsed between snapshots in the storage balances.

Snapshots keep their real timestamps and stay in chronological order. Which block
each snapshot belongs to is stored in ``n.meta[META_KEY]`` (see
:func:`apply_representative_periods`); ``opts/representative_periods.py`` reads it
to cycle storage within each block and to drop the ramp limits between blocks.
"""

import logging

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

logger = logging.getLogger(__name__)

META_KEY = "representative_periods"

WIND_CARRIERS = ("onwind",)
SOLAR_CARRIERS = ("solar",)

# (name, feature, direction): the three extreme periods, picked in this order.
EXTREME_SELECTORS = (
    ("demand_max", "load", "max"),
    ("wind_min", "wind", "min"),
    ("solar_min", "solar", "min"),
)
FEATURES = ("wind", "solar", "load")


def representative_periods_enabled(rep_cfg) -> bool:
    """Whether the ``clustering.temporal.representative_periods`` block switches the feature on."""
    return bool((rep_cfg or {}).get("enable", False))


def _capacity_factor(n, period, carriers) -> pd.Series:
    """National capacity factor of ``carriers`` in ``period``, weighted by candidate ``p_nom_max``."""
    gens = n.generators
    active = n.get_active_assets("Generator", period).reindex(gens.index, fill_value=False)
    candidates = gens.index[
        gens.carrier.isin(carriers)
        & gens.p_nom_extendable
        & active
        & np.isfinite(gens.p_nom_max)
        & (gens.p_nom_max > 0)
    ]
    sns = n.snapshots[n.snapshots.get_level_values("period") == period]
    if candidates.empty:
        logger.warning(
            "No extendable %s candidates with a finite p_nom_max in %s; the feature is left out.",
            "/".join(carriers),
            period,
        )
        return pd.Series(np.nan, index=sns.get_level_values("timestep"))

    cf = n.get_switchable_as_dense("Generator", "p_max_pu", snapshots=sns, inds=candidates)
    weights = gens.loc[candidates, "p_nom_max"]
    series = cf.mul(weights, axis=1).sum(axis=1) / weights.sum()
    return pd.Series(series.to_numpy(), index=sns.get_level_values("timestep"))


def national_features(n, period) -> pd.DataFrame:
    """Wind CF, solar CF and total demand of one investment period, indexed by timestep."""
    sns = n.snapshots[n.snapshots.get_level_values("period") == period]
    load = n.get_switchable_as_dense("Load", "p_set", snapshots=sns).sum(axis=1)
    return pd.DataFrame(
        {
            "wind": _capacity_factor(n, period, WIND_CARRIERS),
            "solar": _capacity_factor(n, period, SOLAR_CARRIERS),
            "load": pd.Series(load.to_numpy(), index=sns.get_level_values("timestep")),
        },
    )


def _usable_features(features: pd.DataFrame) -> list[str]:
    """Features that carry a ranking signal: present, positive somewhere and not flat."""
    usable = []
    for name in FEATURES:
        series = features[name]
        if series.isna().any() or series.max() <= 0 or np.isclose(series.max(), series.min()):
            logger.warning("Feature %r is missing or flat; it does not steer the period selection.", name)
            continue
        usable.append(name)
    if not usable:
        raise ValueError("No usable feature (wind CF, solar CF, demand) to select representative periods from.")
    return usable


def select_periods(features: pd.DataFrame, hours: pd.Series, steps: int, number: int) -> list[dict]:
    """
    Choose the extreme and typical blocks of one investment period.

    Parameters
    ----------
    features : pd.DataFrame
        One row per snapshot, columns ``wind``, ``solar`` and ``load``.
    hours : pd.Series
        Snapshot weighting (hours) of each row, positionally aligned with ``features``.
    steps : int
        Snapshots per block.
    number : int
        Typical periods to choose.

    Returns
    -------
    list of dict
        One entry per selected block, in chronological order: ``block`` (its position
        among the period's blocks), ``kind`` (``extreme`` or ``typical``), ``label``
        (the extreme selector, or ``typical``), ``members`` (the source blocks it
        stands for) and ``weight`` (the factor on its snapshots' weightings).
    """
    n_blocks = len(features) // steps
    if n_blocks < len(EXTREME_SELECTORS) + 1:
        raise ValueError(
            f"Only {n_blocks} whole periods fit in the investment period; representative periods "
            f"need at least {len(EXTREME_SELECTORS) + 1}.",
        )
    used = n_blocks * steps
    if used < len(features):
        logger.info(
            "%d trailing snapshot(s) do not fill a whole period and are never selected; "
            "their hours still count toward the weights.",
            len(features) - used,
        )

    usable = _usable_features(features)
    # (block, step, feature)
    profiles = features[list(FEATURES)].to_numpy()[:used].reshape(n_blocks, steps, len(FEATURES))
    block_means = pd.DataFrame(profiles.mean(axis=1), columns=list(FEATURES))
    block_hours = hours.to_numpy()[:used].reshape(n_blocks, steps).sum(axis=1)

    entries = {}
    for label, feature, direction in EXTREME_SELECTORS:
        if feature not in usable:
            continue
        ranked = block_means[feature].sort_values(ascending=direction == "min", kind="stable").index
        block = next(b for b in ranked if b not in entries)
        entries[block] = {"kind": "extreme", "label": label, "members": 1, "weight": 1.0}

    rest = np.array([b for b in range(n_blocks) if b not in entries])
    if number > len(rest):
        raise ValueError(
            f"representative_periods.number ({number}) exceeds the {len(rest)} periods left after the extremes.",
        )

    columns = [FEATURES.index(f) for f in usable]
    scale = features[usable].max().to_numpy()
    x = (profiles[rest][:, :, columns] / scale).reshape(len(rest), -1)
    if number == len(rest):
        clusters = np.arange(1, len(rest) + 1)
    elif number == 1:
        clusters = np.ones(len(rest), dtype=int)
    else:
        clusters = fcluster(linkage(x, method="ward"), t=number, criterion="maxclust")

    # pick each cluster's member closest to the cluster in mean wind CF, solar CF and demand
    relative_means = (block_means[usable] / features[usable].mean()).to_numpy()
    for cluster in np.unique(clusters):
        members = rest[clusters == cluster]
        target = relative_means[members].mean(axis=0)
        distance = np.linalg.norm(relative_means[members] - target, axis=1)
        block = int(members[np.argmin(distance)])
        entries[block] = {"kind": "typical", "label": "typical", "members": len(members), "weight": np.nan}
    if len(np.unique(clusters)) < number:
        logger.warning(
            "Clustering returned %d typical periods instead of the requested %d.",
            len(np.unique(clusters)),
            number,
        )

    # typical weights: cluster sizes, scaled so the weightings keep their total
    total = hours.sum()
    extreme_hours = sum(block_hours[b] for b, e in entries.items() if e["kind"] == "extreme")
    typical_hours = sum(e["members"] * block_hours[b] for b, e in entries.items() if e["kind"] == "typical")
    scale = (total - extreme_hours) / typical_hours
    for b, e in entries.items():
        if e["kind"] == "typical":
            e["weight"] = e["members"] * scale

    return [{"block": int(b), **entries[b]} for b in sorted(entries)]


def apply_representative_periods(n, rep_cfg) -> dict:
    """
    Reduce ``n`` to its representative periods in place.

    Returns the metadata to store in ``n.meta[META_KEY]``::

        {"period_hours": 24.0,
         "periods": {"2030": [{"start": "2030-01-05 00:00:00", "end": "2030-01-05 23:00:00",
                               "kind": "typical", "label": "typical", "members": 37,
                               "weight": 37.0}, ...]}}

    ``periods`` maps each investment period (as a string) to its selected blocks in
    chronological order; a block holds the snapshots from ``start`` to ``end``, both
    included.
    """
    period_hours = float(rep_cfg.get("period_length", 24))
    number = int(rep_cfg.get("number", 8))
    if period_hours <= 0:
        raise ValueError("representative_periods.period_length must be positive.")
    if number < 1:
        raise ValueError("representative_periods.number must be at least 1.")

    weightings = n.snapshot_weightings
    keep = []
    factors = []
    metadata = {"period_hours": period_hours, "periods": {}}

    for period in n.investment_periods:
        sns = n.snapshots[n.snapshots.get_level_values("period") == period]
        hours = weightings.loc[sns, "stores"]
        if not np.allclose(hours, hours.iloc[0]):
            raise ValueError(
                "Representative periods need evenly spaced snapshots; they cannot follow tsam segmentation.",
            )
        step_hours = float(hours.iloc[0])
        steps = period_hours / step_hours
        if not np.isclose(steps, round(steps)) or round(steps) < 1:
            raise ValueError(
                f"representative_periods.period_length ({period_hours:g} h) is not a whole number of "
                f"{step_hours:g} h snapshots.",
            )
        steps = round(steps)

        features = national_features(n, period)
        objective = weightings.loc[sns, "objective"].reset_index(drop=True)
        entries = select_periods(features, objective, steps, number)

        timesteps = sns.get_level_values("timestep")
        period_meta = []
        for entry in entries:
            first = entry["block"] * steps
            keep.append(sns[first : first + steps])
            factors.append(np.full(steps, entry["weight"]))
            period_meta.append(
                {
                    "start": str(timesteps[first]),
                    "end": str(timesteps[first + steps - 1]),
                    "kind": entry["kind"],
                    "label": entry["label"],
                    "members": int(entry["members"]),
                    "weight": float(entry["weight"]),
                },
            )
        metadata["periods"][str(period)] = period_meta

        selected = sum(e["weight"] * steps * step_hours for e in entries)
        logger.info(
            "Representative periods for %s: %d extreme + %d typical periods of %g h; weightings sum to %g h (was %g h).",
            period,
            sum(e["kind"] == "extreme" for e in entries),
            sum(e["kind"] == "typical" for e in entries),
            period_hours,
            selected,
            objective.sum(),
        )
        for meta in period_meta:
            logger.info(
                "  %s  %-10s weight %6.2f  (%d source periods)",
                meta["start"],
                meta["label"],
                meta["weight"],
                meta["members"],
            )

    snapshots = keep[0].append(keep[1:])
    factor = pd.Series(np.concatenate(factors), index=snapshots)
    new_weightings = weightings.loc[snapshots].copy()
    for column in ("objective", "generators"):
        new_weightings[column] = new_weightings[column] * factor

    n.set_snapshots(snapshots)
    n.snapshot_weightings = new_weightings
    return metadata
