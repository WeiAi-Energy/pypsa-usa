import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import get_weather_year_snapshots
from build_eer_demand import ReadEer


class _FakeEerGroup:
    columns = np.array(["datetime", "CA"])

    def __init__(self, values):
        self.CA = values


def test_weather_snapshots_are_naive_8760_hour_years_without_february_29():
    snapshots = get_weather_year_snapshots([2008, 2009], drop_leap_day=True)

    assert len(snapshots) == 2 * 8760
    assert snapshots.tz is None
    assert not ((snapshots.month == 2) & (snapshots.day == 29)).any()
    assert snapshots[8759] == pd.Timestamp("2008-12-31 23:00")
    assert snapshots[8760] == pd.Timestamp("2009-01-01 00:00")


def test_eer_utc_roll_is_applied_inside_each_weather_year_block():
    values = np.arange(15 * 8760, dtype=float)
    reader = ReadEer("unused.h5", [2050], list(ReadEer.WEATHER_YEARS))
    segment = reader._read_segment(_FakeEerGroup(values), 2008)

    block_start = 8760
    expected = values[block_start + 8754 : block_start + 8760]
    np.testing.assert_array_equal(segment.CA.iloc[:6].to_numpy(), expected)
    assert segment.index.tz is None
    assert not ((segment.index.month == 2) & (segment.index.day == 29)).any()
