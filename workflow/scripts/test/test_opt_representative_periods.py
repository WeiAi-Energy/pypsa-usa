import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from opts.representative_periods import _get_representative_blocks


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
