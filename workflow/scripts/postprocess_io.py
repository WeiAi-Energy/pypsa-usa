"""Read solved networks for post-processing without changing plot semantics.

The workflow writes representative-period networks with a two-level snapshot
index and non-uniform snapshot weights.  Keeping the small amount of input
normalisation here prevents individual plotting scripts from re-implementing
path/index assumptions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pypsa


def load_postprocess_network(path: str | Path) -> pypsa.Network:
    """Load a solved network and validate the common post-processing contract."""
    n = pypsa.Network(path)

    if not isinstance(n.snapshots, pd.MultiIndex) or n.snapshots.nlevels != 2:
        raise ValueError(
            "Post-processing expects snapshots indexed by "
            "(investment period, representative timestep).",
        )

    if tuple(n.snapshots.names) != ("period", "timestep"):
        n.snapshots = n.snapshots.set_names(["period", "timestep"])

    if not n.snapshot_weightings.index.equals(n.snapshots):
        n.snapshot_weightings = n.snapshot_weightings.reindex(n.snapshots)
    if n.snapshot_weightings.isna().any().any():
        raise ValueError("Solved-network snapshot weights do not cover every snapshot.")

    meta = getattr(n, "meta", {})
    if meta is None:
        n.meta = {}
    elif not isinstance(meta, dict):
        raise TypeError("Solved-network metadata must be a dictionary.")

    return n


def snapshot_weight_series(
    n: pypsa.Network,
    preferred: str = "objective",
) -> pd.Series:
    """Return one numeric snapshot-weight column aligned to ``n.snapshots``."""
    candidates = (preferred, "generators", "stores", "objective")
    for column in dict.fromkeys(candidates):
        if column in n.snapshot_weightings.columns:
            weights = pd.to_numeric(
                n.snapshot_weightings[column],
                errors="coerce",
            ).reindex(n.snapshots)
            if weights.notna().all():
                return weights.astype(float)
    raise ValueError(
        "Solved network has no complete objective/generator/store snapshot weights.",
    )


def representative_period_metadata(n: pypsa.Network) -> dict:
    """Return serialized representative-period metadata, or an empty mapping."""
    data = n.meta.get("representative_periods_plot_metadata", {})
    return data if isinstance(data, dict) else {}

