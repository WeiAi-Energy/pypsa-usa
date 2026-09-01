import os
import sys

import numpy as np
import pandas as pd
import pypsa

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from opts.representative_periods import _get_representative_blocks, storage_elapsed_hours


def test_get_representative_blocks_uses_metadata_steps_for_mixed_lengths():
    timesteps = pd.date_range("2030-01-01 00:00", periods=9, freq="h")
    snapshots = pd.MultiIndex.from_arrays(
        [np.repeat(2030, len(timesteps)), timesteps],
        names=["period", "timestep"],
    )
    metadata = {
        "2030": {
            "periods": [
                {"period_id": 0, "kind": "representative", "steps": 4},
                {"period_id": 1, "kind": "extreme", "steps": 2},
                {"period_id": 2, "kind": "representative", "steps": 3},
            ],
        },
    }

    blocks = _get_representative_blocks(
        snapshots,
        base_hours=24.0,
        metadata=metadata,
    )

    assert [entry["steps"] for entry in blocks[2030]] == [4, 2, 3]
    assert [entry["kind"] for entry in blocks[2030]] == [
        "representative",
        "extreme",
        "representative",
    ]


def _hourly_network(weighting):
    """Two-snapshot network whose weighting is deliberately not its timestep length."""
    timesteps = pd.date_range("2030-01-01 00:00", periods=2, freq="h")
    snapshots = pd.MultiIndex.from_arrays(
        [np.repeat(2030, len(timesteps)), timesteps],
        names=["period", "timestep"],
    )
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.set_investment_periods(periods=[2030])
    n.snapshot_weightings.loc[:, :] = float(weighting)
    return n


def test_storage_elapsed_hours_falls_back_to_the_weighting_without_representative_periods():
    n = _hourly_network(weighting=6.0)

    elapsed = storage_elapsed_hours(n, n.snapshots)

    assert list(elapsed) == [6.0, 6.0]
    assert elapsed.index.equals(n.snapshots)


def test_storage_elapsed_hours_uses_the_timestep_under_representative_periods():
    n = _hourly_network(weighting=6.0)
    n.meta = {
        "representative_periods_plot_metadata": {
            "2030": {"periods": [{"period_id": 0, "kind": "representative", "steps": 2}]},
        },
    }

    from_metadata = storage_elapsed_hours(n, n.snapshots)
    from_config = storage_elapsed_hours(
        _hourly_network(weighting=6.0),
        n.snapshots,
        {"clustering": {"temporal": {"representative_periods": {"enable": True}}}},
    )

    assert list(from_metadata) == [1.0, 1.0]
    assert list(from_config) == [1.0, 1.0]
    assert from_metadata.index.equals(n.snapshots)
