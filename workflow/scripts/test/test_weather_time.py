import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_eer_demand import ReadEer


class _FakeEerGroup:
    columns = np.array(["datetime", "CA"])

    def __init__(self, values, timestamps, tz="Etc/GMT+6"):
        self.CA = values
        stamps = (
            timestamps
            if tz is None
            else timestamps.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
        )
        fmt = "%Y-%m-%d %H:%M:%S" if tz is None else "%Y-%m-%d %H:%M:%S%z"
        self.datetime = np.asarray(stamps.strftime(fmt), dtype="S30")


def _eer_cst_weather_timestamps(years):
    """Published EER convention: leap years omit Dec. 31, not Feb. 29."""
    parts = []
    for year in years:
        hours = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", inclusive="left", freq="h")
        if hours.is_leap_year.any():
            hours = hours[~((hours.month == 12) & (hours.day == 31))]
        parts.append(hours)
    return parts[0].append(parts[1:])


def test_eer_uses_published_timestamps_and_converts_fixed_cst_to_utc():
    values = np.arange(15 * 8760, dtype=float)
    reader = ReadEer("unused.h5", [2050], list(ReadEer.WEATHER_YEARS))
    segment = reader._read_segment(_FakeEerGroup(values, _eer_cst_weather_timestamps(reader.WEATHER_YEARS)), 2008)

    block_start = 8760
    np.testing.assert_array_equal(segment.CA.to_numpy(), values[block_start : block_start + 8760])
    assert segment.index.tz is None
    assert segment.index[0] == pd.Timestamp("2008-01-01 06:00")
    assert segment.index[-1] == pd.Timestamp("2008-12-31 05:00")
    assert ((segment.index.month == 2) & (segment.index.day == 29)).any()


def test_eer_rejects_datetimes_carrying_daylight_saving():
    """A US/Central file would shift by five hours in summer; fixed CST never does."""
    import pytest

    values = np.arange(15 * 8760, dtype=float)
    reader = ReadEer("unused.h5", [2050], list(ReadEer.WEATHER_YEARS))
    stamps = _eer_cst_weather_timestamps(reader.WEATHER_YEARS)
    group = _FakeEerGroup(values, stamps, tz="US/Central")

    with pytest.raises(ValueError, match="fixed CST"):
        reader._read_segment(group, 2008)
