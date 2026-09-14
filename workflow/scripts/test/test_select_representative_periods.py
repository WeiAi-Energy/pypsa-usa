import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from build_reeds_renewable_profiles import (
    CF_SCALE,
    build_profile_dataset,
    national_available_generation,
    read_cf_for_sites,
    read_cf_time_index,
    source_rows_by_weather_year,
)
from select_representative_periods import (
    LOAD_FEATURE,
    SOLAR_FEATURE,
    WIND_FEATURE,
    _build_contiguous_source_period_rows,
    _build_period_entry,
    _build_representative_period_mapping,
    _build_representative_snapshot_weightings,
    _get_extreme_period_ids,
    _scale_snapshot_weightings_to_total_hours,
    build_national_feature_frame,
    build_plot_data,
    get_include_extreme,
    get_period_hours,
    resolve_extreme_selectors,
    select_period_entries,
    serialize_representative_period_metadata,
    validate_period_counts,
)


def test_contiguous_source_periods_cross_consecutive_years_but_not_weather_gap():
    first = pd.date_range("2012-12-31 00:00", periods=48, freq="h")
    second = pd.date_range("2016-01-01 00:00", periods=24, freq="h")
    source_index = first.append(second)

    rows = _build_contiguous_source_period_rows(source_index, steps_per_period=36, timestep_hours=1.0)

    assert rows[0].tolist() == list(range(36))
    assert rows[0][0] < 24 <= rows[0][-1]
    assert all(not ({47, 48} <= set(row)) for row in rows)


def test_representative_snapshot_weightings_sum_only_complete_periods():
    hours = pd.date_range("2030-01-01 00:00", periods=8, freq="h")
    weights = pd.DataFrame({"objective": 1.0}, index=hours)
    matching_idx = pd.MultiIndex.from_arrays(
        [np.zeros(8, dtype=int), np.tile(np.arange(4, dtype=int), 2)],
        names=["PeriodNum", "TimeStep"],
    )
    typical_idx = pd.MultiIndex.from_product([[0], range(4)], names=["PeriodNum", "TimeStep"])

    representative_weights = _build_representative_snapshot_weightings(weights, matching_idx, typical_idx)

    assert representative_weights["objective"].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert representative_weights["objective"].sum() == 8.0


def test_scale_snapshot_weightings_to_total_hours():
    sw = pd.DataFrame(
        {
            "objective": [2.0, 2.0, 2.0, 2.0],
            "stores": [1.0, 1.0, 1.0, 1.0],
            "generators": [3.0, 3.0, 3.0, 3.0],
        },
    )

    scaled = _scale_snapshot_weightings_to_total_hours(sw, target_hours=8760.0)

    assert scaled["objective"].sum() == pytest.approx(8760.0)
    assert scaled["stores"].sum() == pytest.approx(4380.0)
    assert scaled["generators"].sum() == pytest.approx(13140.0)


def test_scale_snapshot_weightings_to_total_hours_rejects_non_positive_sum():
    sw = pd.DataFrame({"objective": [0.0, 0.0], "stores": [1.0, 1.0]})

    with pytest.raises(ValueError, match="non-positive"):
        _scale_snapshot_weightings_to_total_hours(sw, target_hours=8760.0)


def test_period_hours_reads_the_single_period_length_key():
    """One tsam run means one period length for representative and extreme alike."""
    assert get_period_hours({"period_length": 5}) == 120.0

    with pytest.raises(ValueError, match="period_length must be configured"):
        get_period_hours({"number": 4})
    with pytest.raises(ValueError, match="period_length must be positive"):
        get_period_hours({"period_length": 0})


@pytest.mark.parametrize("retired_key", ["representative_period_length", "extreme_period_length"])
def test_period_hours_rejects_the_retired_length_keys(retired_key):
    """A stale config must fail loudly instead of falling back to a default."""
    with pytest.raises(ValueError, match="Retired config key"):
        get_period_hours({"period_length": 5, retired_key: 5})


def test_get_include_extreme_is_a_plain_switch():
    """include_extreme is on/off; an absent or null key means off."""
    assert get_include_extreme({"include_extreme": True}) is True
    assert get_include_extreme({"include_extreme": False}) is False
    assert get_include_extreme({"include_extreme": None}) is False
    assert get_include_extreme({}) is False


def test_get_include_extreme_rejects_the_retired_selector_lists():
    """A config still carrying per-feature selectors must fail loudly."""
    with pytest.raises(ValueError, match="must be true or false"):
        get_include_extreme({"include_extreme": ["solar_min", "demand_max"]})
    with pytest.raises(ValueError, match="must be true or false"):
        get_include_extreme({"include_extreme": "demand_max"})


def _feature_frame(columns, periods=3):
    """Build a feature frame carrying the requested feature columns."""
    snapshots = pd.date_range("2030-01-01", periods=periods, freq="h")
    frame = pd.DataFrame({column: np.arange(periods, dtype=float) for column in columns}, index=snapshots)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


def test_extreme_selectors_split_demand_max_from_the_two_capacity_factor_minima():
    """Demand ranks on addMeanMax; both capacity factors rank on addMeanMin."""
    feature_t = _feature_frame([WIND_FEATURE, SOLAR_FEATURE, LOAD_FEATURE], periods=4)

    mean_max, mean_min = resolve_extreme_selectors(feature_t)

    assert mean_max == [LOAD_FEATURE]
    # EXTREME_SELECTORS order: wind before solar.
    assert mean_min == [WIND_FEATURE, SOLAR_FEATURE]


def test_extreme_selectors_skip_a_missing_feature():
    """A carrier that is not modelled simply loses its extreme period."""
    feature_t = _feature_frame([SOLAR_FEATURE, LOAD_FEATURE], periods=4)

    mean_max, mean_min = resolve_extreme_selectors(feature_t)

    assert mean_max == [LOAD_FEATURE]
    assert mean_min == [SOLAR_FEATURE]


def test_extreme_selectors_skip_a_degenerate_feature():
    """A flat series has no most-extreme period, so it carries no ranking signal."""
    feature_t = _feature_frame([WIND_FEATURE, SOLAR_FEATURE, LOAD_FEATURE], periods=4)
    feature_t[WIND_FEATURE] = 0.4
    feature_t[LOAD_FEATURE] = 100.0

    mean_max, mean_min = resolve_extreme_selectors(feature_t)

    assert mean_max == []
    assert mean_min == [SOLAR_FEATURE]


def test_extreme_selectors_resolve_nothing_for_an_all_flat_frame():
    """With no feature carrying any variation there is no extreme period at all."""
    feature_t = _feature_frame([WIND_FEATURE, SOLAR_FEATURE, LOAD_FEATURE], periods=4)
    feature_t.loc[:, :] = 1.0

    assert resolve_extreme_selectors(feature_t) == ([], [])


def test_scale_snapshot_weightings_to_total_hours_preserves_column_ratios():
    """Snapshot weightings should be normalized to the requested annual hours."""
    snapshot_weightings = pd.DataFrame(
        {
            "objective": [10.0, 20.0, 30.0],
            "generators": [5.0, 10.0, 15.0],
            "stores": [2.0, 4.0, 6.0],
        },
    )

    scaled = _scale_snapshot_weightings_to_total_hours(snapshot_weightings, target_hours=8760.0)

    assert scaled["objective"].sum() == pytest.approx(8760.0)
    assert (scaled["generators"] / scaled["objective"]).tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert (scaled["stores"] / scaled["objective"]).tolist() == pytest.approx([0.2, 0.2, 0.2])


def test_select_period_entries_rejects_a_retired_period_length_key():
    with pytest.raises(ValueError, match="Retired config key"):
        select_period_entries(
            pd.DataFrame(),
            pd.DatetimeIndex([]),
            {
                "period_length": 5,
                "extreme_period_length": 2,
                "include_extreme": True,
            },
        )


PEAK_LOAD_DAY = pd.Timestamp("2030-07-20")
WIND_LULL_DAY = pd.Timestamp("2030-01-15")
SOLAR_LULL_DAY = pd.Timestamp("2030-10-08")


def _extreme_feature_frame():
    """
    A year of hourly features with one planted day per extreme selector.

    Every other day of the year carries the same normal profile, so each planted day
    is the unique winner of exactly one selector: July 20 the annual peak load, January
    15 a full day of zero wind, October 8 a full day of zero sun. None of them wins two,
    which is what lets the run return all three (tsam drops a selector's candidate once
    another selector has already claimed it).
    """
    hours = pd.date_range("2030-01-01 00:00", "2030-12-31 23:00", freq="h")
    day = hours.normalize()
    is_peak_load = np.asarray(day == PEAK_LOAD_DAY)
    is_wind_lull = np.asarray(day == WIND_LULL_DAY)
    is_solar_lull = np.asarray(day == SOLAR_LULL_DAY)

    hour_of_day = np.asarray(hours.hour)
    shape = 1.0 + 0.2 * np.sin(2 * np.pi * hour_of_day / 24.0)
    sun = np.clip(np.sin(np.pi * (hour_of_day - 6) / 12.0), 0.0, None)

    frame = pd.DataFrame(
        {
            LOAD_FEATURE: 10_000.0 * np.where(is_peak_load, 1.0, 0.5) * shape,
            SOLAR_FEATURE: np.where(is_solar_lull, 0.0, 0.8) * sun,
            WIND_FEATURE: np.where(is_wind_lull, 0.0, 0.5) * shape,
        },
        index=hours,
    )
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


def test_select_period_entries_returns_one_extreme_per_selector():
    """A live tsam run must return the peak-load day, the wind lull and the solar lull.

    ``number`` is 1 on purpose: the planted extremes are the most distinctive days in
    the frame, so a larger cluster count makes them medoids in their own right and tsam
    then drops them as extremes (see ``validate_period_counts``).
    """
    frame = _extreme_feature_frame()

    entries = select_period_entries(
        frame,
        frame.index,
        {"number": 1, "period_length": 1, "include_extreme": True},
    )

    representative = [entry for entry in entries if entry["kind"] == "representative"]
    extreme = [entry for entry in entries if entry["kind"] == "extreme"]
    assert len(representative) == 1
    assert {entry["source_start"].normalize() for entry in extreme} == {
        PEAK_LOAD_DAY,
        WIND_LULL_DAY,
        SOLAR_LULL_DAY,
    }
    # Every source period is accounted for: 1 typical day + the three extreme days.
    assert sum(entry["weightings"]["objective"].iloc[0] for entry in entries) == len(frame) / 24


def test_select_period_entries_skips_extremes_when_the_switch_is_off():
    """With include_extreme false the run is pure clustering: no extra periods."""
    frame = _extreme_feature_frame()

    entries = select_period_entries(
        frame,
        frame.index,
        {"number": 3, "period_length": 1, "include_extreme": False},
    )

    assert len(entries) == 3
    assert all(entry["kind"] == "representative" for entry in entries)


def test_get_extreme_period_ids_reads_the_labels_tsam_appended():
    """Extreme periods are the clusters tsam created via new_cluster_center."""
    agg = SimpleNamespace(
        _clusterOrder=np.array([0, 1, 2]),
        extremePeriods={
            "wind": {"newClusterNo": 2, "stepNo": 7},
            # A label tsam dropped from the final matching must not be reported.
            "solar": {"newClusterNo": 9, "stepNo": 3},
        },
    )

    assert _get_extreme_period_ids(agg, period_ids=[0, 1, 2]) == [2]
    assert _get_extreme_period_ids(SimpleNamespace(), period_ids=[0, 1]) == []


def test_representative_period_mapping_covers_medoids_and_extremes():
    """Cluster i's medoid sits at position i; extremes carry their own stepNo."""
    agg = SimpleNamespace(
        clusterCenterIndices=[11, 22],
        extremePeriods={"load daily max.": {"newClusterNo": 2, "stepNo": 33}},
    )

    assert _build_representative_period_mapping(agg, [0, 1, 2]) == {0: 11, 1: 22, 2: 33}

    with pytest.raises(ValueError, match="Unable to map"):
        _build_representative_period_mapping(agg, [0, 1, 2, 3])


def test_validate_period_counts_warns_when_tsam_drops_an_extreme(caplog):
    """A dropped extreme is a warning; a missing representative is an error."""
    weights = pd.DataFrame({"objective": [1.0, 1.0]})
    hours = pd.date_range("2030-01-01 00:00", periods=2, freq="h")
    entries = [
        _build_period_entry(0, "representative", 2, hours, weights),
        _build_period_entry(1, "extreme", 2, hours, weights),
    ]
    cfg = {
        "number": 1,
        "period_length": 2 / 24,
        "include_extreme": True,
    }

    with caplog.at_level("WARNING"):
        validate_period_counts({2030: entries}, cfg)
    assert "instead of the 3 requested" in caplog.text

    with pytest.raises(ValueError, match="expected 2"):
        validate_period_counts({2030: entries}, {**cfg, "number": 2})


def test_serialize_representative_period_metadata_preserves_source_ranges():
    timestamps = pd.date_range("2030-01-01 00:00", periods=4, freq="h")
    snapshots = pd.MultiIndex.from_arrays([np.repeat(2030, 4), timestamps], names=["period", "timestep"])
    entries = [
        {
            "period_id": 0,
            "kind": "representative",
            "steps": 2,
            "source_start": timestamps[0],
            "source_end": timestamps[1],
            "source_snapshots": timestamps[:2],
            "snapshots": snapshots[:2],
        },
        {
            "period_id": 1,
            "kind": "extreme",
            "steps": 2,
            "source_start": timestamps[2],
            "source_end": timestamps[3],
            "source_snapshots": timestamps[2:],
            "snapshots": snapshots[2:],
        },
    ]

    metadata = serialize_representative_period_metadata({2030: entries})

    assert metadata["2030"]["periods"] == [
        {
            "period_id": 0,
            "kind": "representative",
            "steps": 2,
            "start": "2030-01-01T00:00:00",
            "end": "2030-01-01T01:00:00",
            "weather_years": [2030],
            "snapshot_start": "2030-01-01T00:00:00",
            "snapshot_end": "2030-01-01T01:00:00",
        },
        {
            "period_id": 1,
            "kind": "extreme",
            "steps": 2,
            "start": "2030-01-01T02:00:00",
            "end": "2030-01-01T03:00:00",
            "weather_years": [2030],
            "snapshot_start": "2030-01-01T02:00:00",
            "snapshot_end": "2030-01-01T03:00:00",
        },
    ]


def test_serialize_representative_period_metadata_reports_source_hours_not_labels():
    """Serialized dates come from the weather hours, never from the snapshot labels."""
    source_hours = pd.DatetimeIndex([pd.Timestamp("2012-12-31 23:00"), pd.Timestamp("2013-01-01 00:00")])
    snapshots = pd.MultiIndex.from_arrays(
        [np.repeat(2050, 2), pd.date_range("1998-01-01 00:00", periods=2, freq="h")],
        names=["period", "timestep"],
    )
    weights = pd.DataFrame({"objective": [1.0, 1.0]})
    entry = _build_period_entry(0, "extreme", 2, source_hours, weights)
    entry["snapshots"] = snapshots

    period = serialize_representative_period_metadata({2050: [entry]})["2050"]["periods"][0]

    assert period["start"] == "2012-12-31T23:00:00"
    assert period["end"] == "2013-01-01T00:00:00"
    assert period["weather_years"] == [2012, 2013]
    assert period["snapshot_start"] == "1998-01-01T00:00:00"


def test_serialize_representative_period_metadata_falls_back_to_bounds():
    """Entries built without source hours still serialize their known bounds."""
    entries = [
        {
            "period_id": 0,
            "kind": "representative",
            "steps": 2,
            "source_start": pd.Timestamp("2030-01-01 00:00"),
            "source_end": pd.Timestamp("2030-01-01 01:00"),
        },
    ]

    period = serialize_representative_period_metadata({2030: entries})["2030"]["periods"][0]

    assert period["start"] == "2030-01-01T00:00:00"
    assert period["weather_years"] == [2030]
    assert period["snapshot_start"] is None


def _write_cf_h5(path, gids, cf_by_gid, weather_year=2007):
    """Write a minimal ReEDS CF table with its published time index."""
    import tables

    values = np.column_stack([np.asarray(cf_by_gid[gid], dtype=float) for gid in gids])
    with tables.open_file(str(path), "w") as h5:
        h5.create_array("/", "columns", np.asarray(gids, dtype=np.int64))
        h5.create_array("/", f"cf_profile_{weather_year}", np.rint(values / CF_SCALE).astype(np.int32))
        # Match the shipped ReEDS convention: tz-aware UTC strings.
        timestamps = pd.date_range(f"{weather_year}-01-01", periods=len(values), freq="h", tz="UTC")
        h5.create_array("/", f"time_index_{weather_year}", np.asarray(timestamps.strftime("%Y-%m-%d %H:%M:%S%z"), dtype="S25"))
    return str(path)


def test_national_available_generation_sums_capacity_times_cf(tmp_path):
    """The wind/solar feature numerator is sum_i(capacity_i * cf_i)."""
    path = _write_cf_h5(
        tmp_path / "cf.h5",
        [11, 22],
        {11: [1.0, 0.5, 0.0], 22: [0.0, 0.5, 1.0]},
    )
    capacities = pd.Series({11: 100.0, 22: 300.0})

    values = national_available_generation(path, [11, 22], capacities, 2007)

    assert values.tolist() == pytest.approx([100.0, 200.0, 300.0])


def test_national_available_generation_applies_transform_per_site(tmp_path):
    """Solar's inverter loading ratio must clip per site, not after aggregation."""
    path = _write_cf_h5(
        tmp_path / "cf.h5",
        [11, 22],
        {11: [0.9], 22: [0.1]},
    )
    capacities = pd.Series({11: 100.0, 22: 100.0})

    def transform(block):
        return np.clip(block * 1.34, None, 1.0)

    values = national_available_generation(path, [11, 22], capacities, 2007, transform=transform)

    # Site 11 clips at 1.0 (0.9 * 1.34 = 1.206); site 22 does not (0.1 * 1.34 = 0.134).
    assert values.tolist() == pytest.approx([100.0 * 1.0 + 100.0 * 0.134])
    # Aggregating first would have given a strictly larger, wrong answer.
    assert values[0] < (0.5 * (0.9 + 0.1)) * 1.34 * 200.0


def test_read_cf_for_sites_row_subset_matches_full_read(tmp_path):
    """Row subsetting is how representative runs skip most of the year."""
    path = _write_cf_h5(
        tmp_path / "cf.h5",
        [11, 22],
        {11: [0.1, 0.2, 0.3, 0.4], 22: [0.5, 0.6, 0.7, 0.8]},
    )

    full = read_cf_for_sites(path, [11, 22], 2007)
    subset = read_cf_for_sites(path, [11, 22], 2007, rows=[1, 3])

    assert full.index.tolist() == [0, 1, 2, 3]
    assert subset.index.tolist() == [1, 3]
    pd.testing.assert_frame_equal(subset, full.loc[[1, 3]])

    timestamps = read_cf_time_index(path, 2007)
    assert timestamps.tz is None
    assert timestamps.equals(pd.date_range("2007-01-01", periods=4, freq="h"))


def test_read_cf_time_index_rejects_a_non_utc_published_offset(tmp_path):
    """A file published in local time must not be read as if it were UTC."""
    import tables

    path = str(tmp_path / "local.h5")
    with tables.open_file(path, "w") as h5:
        h5.create_array("/", "columns", np.asarray([1], dtype=np.int64))
        h5.create_array("/", "cf_profile_2007", np.zeros((4, 1), dtype=np.int32))
        stamps = pd.date_range("2007-01-01", periods=4, freq="h", tz="Etc/GMT+6")
        h5.create_array("/", "time_index_2007", np.asarray(stamps.strftime("%Y-%m-%d %H:%M:%S%z"), dtype="S25"))

    with pytest.raises(ValueError, match="not published in UTC"):
        read_cf_time_index(path, 2007)


def test_source_rows_by_weather_year_skips_untouched_years():
    """Only the weather years the windows land in should be opened."""
    year_hours = pd.date_range("2008-01-01", "2009-01-01", inclusive="left", freq="h")
    year_hours = year_hours[~((year_hours.month == 12) & (year_hours.day == 31))]
    source = pd.DatetimeIndex([year_hours[0], year_hours[5], year_hours[8759]])

    mapping = source_rows_by_weather_year(source, {2007: pd.date_range("2007-01-01", periods=2, freq="h"), 2008: year_hours, 2016: pd.date_range("2016-01-01", periods=2, freq="h")})

    assert set(mapping) == {2008}
    rows, wanted = mapping[2008]
    assert rows.tolist() == [0, 5, 8759]
    assert wanted.equals(pd.DatetimeIndex([year_hours[0], year_hours[5], year_hours[8759]]))


def test_source_rows_by_weather_year_uses_the_published_leap_day_and_rejects_missing_december_31():
    hours = pd.date_range("2008-01-01", "2009-01-01", inclusive="left", freq="h")
    hours = hours[~((hours.month == 12) & (hours.day == 31))]
    mapping = source_rows_by_weather_year(pd.DatetimeIndex(["2008-02-29 00:00"]), {2008: hours})
    assert mapping[2008][0].tolist() == [1416]

    with pytest.raises(ValueError, match="published ReEDS UTC indexes"):
        source_rows_by_weather_year(pd.DatetimeIndex(["2008-12-31 00:00"]), {2008: hours})


def test_build_profile_dataset_keeps_time_indexed_for_named_source_hours():
    """Representative hours arrive named 'source_timestep'; time must stay indexed.

    xarray names a coordinate's dimension after the pandas index, so a named
    index used to create a stray dimension and leave ``time`` without an index.
    ``to_pandas()`` then returned positional rows that matched no weather hour.
    """
    hours = pd.DatetimeIndex(
        ["2009-12-07 06:00", "2009-12-07 07:00"],
        name="source_timestep",
    )
    buses = pd.Index(["b0", "b1"], name="bus")
    profile_df = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], index=hours, columns=buses)
    series = pd.Series([1.0, 2.0], index=buses)

    dataset = build_profile_dataset("solar", profile_df, buses, series, series, series)

    assert set(dataset.sizes) == {"time", "bus"}
    assert "time" in dataset.indexes
    recovered = dataset["profile"].transpose("time", "bus").to_pandas()
    assert pd.DatetimeIndex(recovered.index).equals(pd.DatetimeIndex(hours, name="time"))


def test_build_national_feature_frame_assembles_three_features():
    hours = pd.date_range("2030-01-01 00:00", periods=4, freq="h")
    wind = pd.Series([0.5, 0.5, 0.5, 0.5], index=hours)
    solar = pd.Series([0.0, 0.5, 1.0, 0.5], index=hours)
    demand = pd.Series([10.0, 20.0, 30.0, 40.0], index=hours)

    features, profiles = build_national_feature_frame({"wind": wind, "solar": solar}, demand)

    assert list(features.columns) == [
        ("Generator", "p_max_pu", "wind"),
        ("Generator", "p_max_pu", "solar"),
        ("Load", "p_set", "ac_load"),
    ]
    assert features[("Generator", "p_max_pu", "wind")].tolist() == [0.5, 0.5, 0.5, 0.5]
    assert features[("Generator", "p_max_pu", "solar")].tolist() == [0.0, 0.5, 1.0, 0.5]
    assert features[("Load", "p_set", "ac_load")].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert profiles["load"].tolist() == [10.0, 20.0, 30.0, 40.0]


def test_build_national_feature_frame_drops_absent_carrier_groups():
    hours = pd.date_range("2030-01-01 00:00", periods=3, freq="h")
    demand = pd.Series([1.0, 2.0, 3.0], index=hours)

    features, profiles = build_national_feature_frame({"wind": None, "solar": None}, demand)

    assert list(features.columns) == [("Load", "p_set", "ac_load")]
    assert profiles["wind"] is None


def test_build_national_feature_frame_keeps_only_exact_physical_timestamp_overlap():
    """Naive but offset source indices are never aligned positionally."""
    hours = pd.date_range("2030-01-01 00:00", periods=8, freq="h")
    solar = pd.Series(np.linspace(0.0, 1.0, 8), index=hours)
    cst_demand = pd.Series(np.arange(8, dtype=float), index=hours - pd.Timedelta(hours=6))

    features, _ = build_national_feature_frame({"solar": solar}, cst_demand)

    assert features.index.equals(hours[:2])
    assert features[LOAD_FEATURE].tolist() == [6.0, 7.0]


def test_build_plot_data_pads_periods_of_differing_length():
    hours = pd.date_range("2030-01-01 00:00", periods=6, freq="h")
    load = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=hours)
    snapshots = pd.MultiIndex.from_arrays([np.repeat(2030, 6), hours], names=["period", "timestep"])
    weights = pd.DataFrame({"objective": 1.0, "stores": 1.0, "generators": 1.0}, index=range(4))

    representative = _build_period_entry(0, "representative", 4, hours[:4], weights)
    representative["snapshots"] = snapshots[:4]
    extreme = _build_period_entry(1, "extreme", 2, hours[4:], weights.iloc[:2])
    extreme["snapshots"] = snapshots[4:]

    plot_data = build_plot_data({"wind": None, "solar": None, "load": load}, {2030: [representative, extreme]})

    matrix = plot_data[2030]["profiles"]["Load"]
    assert matrix[0].tolist() == [1 / 6, 2 / 6, 3 / 6, 4 / 6]
    assert matrix[1].dropna().tolist() == [5 / 6, 6 / 6]
    assert plot_data[2030]["period_means"]["Load"].loc[1] == pytest.approx(5.5)


def test_build_plot_data_labels_periods_with_real_weather_years():
    """Panel labels must name the real weather year, not the planning horizon."""
    hours = pd.date_range("2012-12-31 22:00", periods=4, freq="h")
    load = pd.Series([1.0, 2.0, 3.0, 4.0], index=hours)
    snapshots = pd.MultiIndex.from_arrays([np.repeat(2050, 4), hours], names=["period", "timestep"])
    weights = pd.DataFrame({"objective": 1.0, "stores": 1.0, "generators": 1.0}, index=range(2))

    # The synthetic snapshot labels say 2050; the source hours say 2012/2013.
    within_year = _build_period_entry(0, "representative", 2, hours[:2], weights)
    within_year["snapshots"] = snapshots[:2]
    across_years = _build_period_entry(
        1,
        "extreme",
        2,
        pd.DatetimeIndex([pd.Timestamp("2012-12-31 23:00"), pd.Timestamp("2013-01-01 00:00")]),
        weights,
    )
    across_years["snapshots"] = snapshots[2:]

    plot_data = build_plot_data({"wind": None, "solar": None, "load": load}, {2050: [within_year, across_years]})

    weather_years = plot_data[2050]["period_weather_years"]
    assert weather_years[0] == "2012"
    assert weather_years[1] == "2012/2013"
    assert plot_data[2050]["period_ranges"][0] == (hours[0], hours[1])
