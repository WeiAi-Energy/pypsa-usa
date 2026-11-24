"""
Test the block reader of build_dynamic_line_rating.

``read_rating_blocks`` parses the rating CSV with pyarrow in blocks cut at line
ends; it must give exactly what ``pandas.read_csv(...).to_numpy(np.float32)``
gives, whatever the block size, line ending or missing values.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_dynamic_line_rating import read_header, read_rating_blocks


def write_rating_csv(path, rows=30, lines=7, newline="\n", seed=0):
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2012-01-01", periods=rows, freq="h", tz="UTC")
    values = rng.uniform(0.3, 2.3, (rows, lines)).round(3)
    values[3, 2] = np.nan  # an empty field
    values[5] = 1.0  # integer-looking text
    df = pd.DataFrame(values, index=pd.Index(hours.astype(str), name="hour"), columns=[str(i) for i in range(lines)])
    with open(path, "w", newline="") as fh:
        fh.write(df.to_csv(lineterminator=newline))
    return path


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("block_bytes", [40, 150, 1 << 20])  # a line spans blocks, several lines, everything
def test_blocks_equal_pandas(tmp_path, newline, block_bytes):
    path = write_rating_csv(tmp_path / "rating.csv", newline=newline)
    expected = pd.read_csv(path, index_col=0)

    stamps, parts = [], []
    for block_stamps, values in read_rating_blocks(str(path), block_bytes=block_bytes):
        stamps.extend(block_stamps)
        parts.append(values)
    values = np.vstack(parts)

    assert read_header(str(path)) == ["hour", *expected.columns]
    assert stamps == list(expected.index)
    assert values.dtype == np.float32
    np.testing.assert_array_equal(values, expected.to_numpy(np.float32))
    assert np.isnan(values[3, 2])
    if block_bytes == 40:
        assert len(parts) > 1


def test_no_trailing_newline(tmp_path):
    path = write_rating_csv(tmp_path / "rating.csv", rows=8)
    with open(path, "rb+") as fh:
        fh.seek(-1, os.SEEK_END)
        fh.truncate()
    blocks = list(read_rating_blocks(str(path), block_bytes=50))
    assert sum(len(stamps) for stamps, _ in blocks) == 8
    np.testing.assert_array_equal(
        np.vstack([v for _, v in blocks]),
        pd.read_csv(path, index_col=0).to_numpy(np.float32),
    )
