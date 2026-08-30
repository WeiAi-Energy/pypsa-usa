import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from plot_statistics import (
    _format_representative_period_title,
    _get_representative_period_snapshot_groups,
    _calculate_line_x_capex,
    build_statistics_overview_table,
    build_statistics_summary_table,
    _sort_area_plot_columns_by_variability,
    _split_mixed_sign_area_series,
    get_carrier_cost_breakdown,
    get_sssc_capacity_by_nerc_region,
    normalize_line_x_statistics_columns,
)


def test_normalize_line_x_statistics_columns_backfills_line_cost_and_default_sssc_nom():
    n = SimpleNamespace(
        line_xs=pd.DataFrame(
            {
                "capital_cost_line": [12.0],
                "capital_cost_sssc": [4.0],
                "sssc_nom_opt": [3.5],
            },
            index=["lx1"],
        ),
    )

    normalize_line_x_statistics_columns(n)

    assert n.line_xs.at["lx1", "capital_cost"] == 12.0
    assert n.line_xs.at["lx1", "capital_cost_sssc"] == 4.0
    assert n.line_xs.at["lx1", "sssc_nom_opt"] == 3.5
    assert n.line_xs.at["lx1", "sssc_nom"] == 0.0


def test_calculate_line_x_capex_uses_sssc_nom_fields():
    n = SimpleNamespace(
        line_xs=pd.DataFrame(
            {
                "carrier": ["AC"],
                "s_nom_extendable": [True],
                "s_nom": [100.0],
                "s_nom_line_opt": [130.0],
                "capital_cost_line": [2.0],
                "sssc_nom": [5.0],
                "sssc_nom_opt": [9.0],
                "capital_cost_sssc": [3.0],
            },
            index=["lx1"],
        ),
        carriers=pd.DataFrame(
            {
                "nice_name": ["Ac"],
                "color": ["#123456"],
            },
            index=["AC"],
        ),
    )

    line_capex, sssc_capex = _calculate_line_x_capex(n)

    assert line_capex.to_dict() == {"Ac": 60.0}
    assert sssc_capex.to_dict() == {"Ac SSSC": 12.0}


def test_calculate_line_x_capex_counts_sssc_when_line_is_not_extendable():
    n = SimpleNamespace(
        line_xs=pd.DataFrame(
            {
                "carrier": ["AC"],
                "s_nom_extendable": [False],
                "s_nom": [100.0],
                "s_nom_line_opt": [130.0],
                "capital_cost_line": [2.0],
                "sssc_nom": [1.0],
                "sssc_nom_opt": [6.0],
                "capital_cost_sssc": [4.0],
            },
            index=["lx1"],
        ),
        carriers=pd.DataFrame(
            {
                "nice_name": ["Ac"],
                "color": ["#123456"],
            },
            index=["AC"],
        ),
    )

    line_capex, sssc_capex = _calculate_line_x_capex(n)

    assert line_capex.empty
    assert sssc_capex.to_dict() == {"Ac SSSC": 20.0}


def test_get_sssc_capacity_by_nerc_region_splits_cross_region_lines_evenly():
    n = SimpleNamespace(
        buses=pd.DataFrame(
            {
                "nerc_reg": ["PJM", "PJM", "PJM", "NPCC_NE", "ERCOT"],
            },
            index=["pjm1", "pjm2", "pjm3", "ne1", "ercot1"],
        ),
        line_xs=pd.DataFrame(
            {
                "bus0": ["pjm1", "pjm3"],
                "bus1": ["pjm2", "ne1"],
                "sssc_nom_opt": [10.0, 6.0],
            },
            index=["lx1", "lx2"],
        ),
    )

    result = get_sssc_capacity_by_nerc_region(n)

    assert list(result.index) == ["ERCOT", "NPCC_NE", "PJM"]
    assert result["sssc_capacity_mw"].to_dict() == {"ERCOT": 0.0, "NPCC_NE": 3.0, "PJM": 13.0}
    assert result.loc["PJM", "pct_of_national_total"] == pytest.approx(81.25)
    assert result.loc["NPCC_NE", "pct_of_national_total"] == pytest.approx(18.75)
    assert result.loc["ERCOT", "pct_of_national_total"] == pytest.approx(0.0)


def test_get_sssc_capacity_by_nerc_region_handles_missing_line_xs():
    n = SimpleNamespace(
        buses=pd.DataFrame({"nerc_reg": ["PJM", "ERCOT"]}, index=["pjm1", "ercot1"]),
        line_xs=pd.DataFrame(),
    )

    result = get_sssc_capacity_by_nerc_region(n)

    assert list(result.index) == ["ERCOT", "PJM"]
    assert (result["sssc_capacity_mw"] == 0.0).all()
    assert (result["pct_of_national_total"] == 0.0).all()


def test_get_carrier_cost_breakdown_merges_opex_entries_with_same_nice_name():
    class StatisticsStub:
        @staticmethod
        def opex():
            index = pd.MultiIndex.from_tuples(
                [("Generator", "ccgt_existing"), ("Generator", "ccgt_new")],
                names=["component", "carrier"],
            )
            return pd.Series([10.0, 5.0], index=index)

    n = SimpleNamespace(
        statistics=StatisticsStub(),
        carriers=pd.DataFrame(
            {
                "nice_name": ["Combined-Cycle Gas", "Combined-Cycle Gas"],
                "color": ["#123456", "#123456"],
            },
            index=["ccgt_existing", "ccgt_new"],
        ),
        generators=pd.DataFrame(),
        storage_units=pd.DataFrame(),
        links=pd.DataFrame(),
        stores=pd.DataFrame(),
        lines=pd.DataFrame(),
        line_xs=pd.DataFrame(),
    )

    costs = get_carrier_cost_breakdown(n)

    assert costs.index.tolist() == ["Combined-Cycle Gas"]
    assert costs.at["Combined-Cycle Gas", "OPEX"] == 15.0
    assert costs.at["Combined-Cycle Gas", "CAPEX"] == 0.0


def test_get_carrier_cost_breakdown_existing_retirement_yields_positive_capex():
    class StatisticsStub:
        @staticmethod
        def opex():
            return pd.Series(dtype=float)

    n = SimpleNamespace(
        statistics=StatisticsStub(),
        carriers=pd.DataFrame({"nice_name": ["Nuclear"], "color": ["#123456"]}, index=["nuclear"]),
        generators=pd.DataFrame(
            {
                "carrier": ["nuclear", "nuclear"],
                "p_nom": [1000.0, 0.0],
                "p_nom_opt": [0.0, 500.0],
                "p_nom_extendable": [True, True],
                "capital_cost": [175000.0, 175000.0],
            },
            index=["gen1 nuclear existing", "gen2 nuclear_2050"],
        ),
        storage_units=pd.DataFrame(),
        links=pd.DataFrame(),
        stores=pd.DataFrame(),
        lines=pd.DataFrame(),
        line_xs=pd.DataFrame(),
    )

    costs = get_carrier_cost_breakdown(n)

    # "existing" unit retires (p_nom_opt < p_nom): capex uses capital_cost * p_nom, not the
    # negative (p_nom_opt - p_nom) delta, so retirement no longer shows up as negative CAPEX.
    # The non-"existing" build option keeps the original capital_cost * (p_nom_opt - p_nom).
    assert costs.at["Nuclear", "CAPEX"] == 175000.0 * 1000.0 + 175000.0 * 500.0


def test_build_statistics_summary_table_merges_line_x_line_and_splits_sssc():
    class StatisticsStub:
        @staticmethod
        def __call__(nice_names=False):
            assert nice_names is False
            index = pd.MultiIndex.from_tuples(
                [
                    ("Line", "AC"),
                    ("Link", "DC"),
                    ("LineX", "AC SSSC"),
                    ("Generator", "solar"),
                ],
                names=["component", "carrier"],
            )
            return pd.DataFrame(
                {
                    "Optimal Capacity": [230.0, 350.0, 20.0, 50.0],
                    "Installed Capacity": [210.0, 300.0, 10.0, 40.0],
                    "Expanded Capacity": [20.0, 50.0, 10.0, 10.0],
                    "Curtailment": [5.0, 0.0, 0.0, 30.0],
                    "Supply": [25.0, 0.0, 0.0, 70.0],
                    "Withdrawal": [5.0, 0.0, 0.0, 0.0],
                    "Transmission": [230.0, 350.0, 0.0, 0.0],
                    "Capacity Factor": [0.8, 0.9, 1.95, 0.3],
                    "Capital Expenditure": [460.0, 700.0, 60.0, 500.0],
                    "Operational Expenditure": [28.0, 35.0, 0.0, 30.0],
                    "Revenue": [59.0, 0.0, 0.0, 100.0],
                    "Market Value": [2.5, 0.0, 0.0, 4.0],
                },
                index=index,
            )

    n = SimpleNamespace(
        statistics=StatisticsStub(),
        carriers=pd.DataFrame(
            {
                "nice_name": ["Ac", "Dc", "Solar"],
                "color": ["#123456", "#654321", "#abcdef"],
            },
            index=["AC", "DC", "solar"],
        ),
        lines=pd.DataFrame(
            {
                "carrier": ["AC"],
                "length": [100.0],
                "s_nom": [210.0],
                "s_nom_opt": [230.0],
            },
            index=["line_1"],
        ),
        links=pd.DataFrame(
            {
                "carrier": ["DC", "battery"],
                "length": [50.0, 200.0],
                "p_nom": [300.0, 999.0],
                "p_nom_opt": [350.0, 999.0],
            },
            index=["dc_link", "battery_link"],
        ),
        line_xs=pd.DataFrame(
            {
                "carrier": ["AC"],
                "length": [10.0],
                "s_nom": [10.0],
                "s_nom_line_opt": [20.0],
            },
            index=["line_x_1"],
        ),
    )

    summary = build_statistics_summary_table(n)
    overview = build_statistics_overview_table(n)

    assert list(summary.columns) == [
        "Installed Capacity",
        "Optimal Capacity",
        "Installed Volume",
        "Optimal Volume",
        "Capacity Factor",
        "Curtailment",
        "Capital Expenditure",
    ]
    assert summary.loc["Ac", "Installed Capacity"] == 210.0
    assert summary.loc["Ac", "Optimal Capacity"] == 230.0
    assert summary.loc["Ac", "Installed Volume"] == 100.0 * 210.0 + 10.0 * 10.0
    assert summary.loc["Ac", "Optimal Volume"] == 100.0 * 230.0 + 10.0 * 20.0
    assert summary.loc["Ac", "Capacity Factor"] == 0.8
    assert summary.loc["Ac", "Curtailment"] == 5.0 / 30.0
    assert summary.loc["Ac", "Capital Expenditure"] == 460.0
    assert summary.loc["Ac SSSC", "Installed Capacity"] == 10.0
    assert summary.loc["Ac SSSC", "Optimal Capacity"] == 20.0
    assert summary.loc["Ac SSSC", "Installed Volume"] == 0.0
    assert summary.loc["Ac SSSC", "Optimal Volume"] == 0.0
    assert summary.loc["Ac SSSC", "Capacity Factor"] == 1.95
    assert summary.loc["Ac SSSC", "Curtailment"] == 0.0
    assert summary.loc["Ac SSSC", "Capital Expenditure"] == 60.0
    assert summary.loc["Dc", "Installed Capacity"] == 300.0
    assert summary.loc["Dc", "Optimal Capacity"] == 350.0
    assert summary.loc["Dc", "Installed Volume"] == 50.0 * 300.0
    assert summary.loc["Dc", "Optimal Volume"] == 50.0 * 350.0
    assert summary.loc["Dc", "Capacity Factor"] == 0.9
    assert summary.loc["Dc", "Curtailment"] == 0.0
    assert summary.loc["Dc", "Capital Expenditure"] == 700.0
    assert summary.loc["Solar", "Installed Capacity"] == 40.0
    assert summary.loc["Solar", "Optimal Capacity"] == 50.0
    assert summary.loc["Solar", "Installed Volume"] == 0.0
    assert summary.loc["Solar", "Optimal Volume"] == 0.0
    assert summary.loc["Solar", "Curtailment"] == 30.0 / 100.0
    assert overview.loc["Ac", "Installed Capacity"] == 210.0
    assert overview.loc["Ac", "Expanded Capacity"] == 20.0
    assert overview.loc["Ac", "Curtailment"] == 5.0
    assert overview.loc["Ac", "Supply"] == 25.0
    assert overview.loc["Ac", "Withdrawal"] == 5.0
    assert overview.loc["Ac", "Transmission"] == 230.0
    assert overview.loc["Ac", "Operational Expenditure"] == 28.0
    assert overview.loc["Ac", "Revenue"] == 59.0
    assert overview.loc["Ac", "Market Value"] == 2.5
    assert "Ac SSSC" not in overview.index or overview.loc["Ac SSSC", ["Supply", "Withdrawal", "Transmission"]].sum() == 0.0


def test_build_statistics_summary_table_aligns_volume_to_component_and_carrier_index():
    class StatisticsStub:
        @staticmethod
        def __call__(nice_names=False):
            assert nice_names is False
            index = pd.MultiIndex.from_tuples(
                [
                    ("Line", "AC"),
                    ("LineX", "AC"),
                    ("Load", "AC"),
                    ("Link", "DC"),
                    ("Generator", "solar"),
                ],
                names=["component", "technology"],
            )
            return pd.DataFrame(
                {
                    "Optimal Capacity": [230.0, 25.0, 0.0, 350.0, 50.0],
                    "Installed Capacity": [210.0, 20.0, 0.0, 300.0, 40.0],
                    "Capacity Factor": [0.8, 0.9, 0.0, 0.7, 0.3],
                    "Curtailment": [0.0, 0.0, 0.0, 0.0, 30.0],
                    "Supply": [0.0, 0.0, 0.0, 0.0, 70.0],
                    "Capital Expenditure": [460.0, 50.0, 1.0, 700.0, 500.0],
                },
                index=index,
            )

    n = SimpleNamespace(
        statistics=StatisticsStub(),
        carriers=pd.DataFrame(
            {
                "nice_name": ["Ac", "Dc", "Solar"],
                "color": ["#123456", "#654321", "#abcdef"],
            },
            index=["AC", "DC", "solar"],
        ),
        lines=pd.DataFrame(
            {
                "carrier": ["AC"],
                "length": [100.0],
                "s_nom": [210.0],
                "s_nom_opt": [230.0],
            },
            index=["line_1"],
        ),
        links=pd.DataFrame(
            {
                "carrier": ["DC"],
                "length": [50.0],
                "p_nom": [300.0],
                "p_nom_opt": [350.0],
            },
            index=["dc_link"],
        ),
        line_xs=pd.DataFrame(
            {
                "carrier": ["AC"],
                "length": [10.0],
                "s_nom": [20.0],
                "s_nom_line_opt": [25.0],
            },
            index=["line_x_1"],
        ),
    )

    summary = build_statistics_summary_table(n)

    assert summary.loc[("Line", "AC"), "Installed Volume"] == 100.0 * 210.0
    assert summary.loc[("Line", "AC"), "Optimal Volume"] == 100.0 * 230.0
    assert summary.loc[("LineX", "AC"), "Installed Volume"] == 10.0 * 20.0
    assert summary.loc[("LineX", "AC"), "Optimal Volume"] == 10.0 * 25.0
    assert summary.loc[("Load", "AC"), "Installed Volume"] == 0.0
    assert summary.loc[("Load", "AC"), "Optimal Volume"] == 0.0
    assert summary.loc[("Link", "DC"), "Installed Volume"] == 50.0 * 300.0
    assert summary.loc[("Link", "DC"), "Optimal Volume"] == 50.0 * 350.0
    assert summary.loc[("Generator", "solar"), "Installed Volume"] == 0.0
    assert summary.loc[("Generator", "solar"), "Optimal Volume"] == 0.0


def test_split_mixed_sign_area_series_splits_battery_storage_charge_and_discharge():
    df = pd.DataFrame(
        {
            "Battery Storage": [2.0, -3.0, 0.0, 4.0],
            "Solar": [1.0, 2.0, 3.0, 4.0],
            "exports": [-1.0, -2.0, -1.5, -0.5],
        },
    )

    prepared, source_labels = _split_mixed_sign_area_series(df)

    assert prepared.columns.tolist() == [
        "Battery Storage Discharge",
        "Battery Storage Charge",
        "Solar",
        "exports",
    ]
    assert prepared["Battery Storage Discharge"].tolist() == [2.0, 0.0, 0.0, 4.0]
    assert prepared["Battery Storage Charge"].tolist() == [0.0, -3.0, 0.0, 0.0]
    assert prepared["Solar"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert prepared["exports"].tolist() == [-1.0, -2.0, -1.5, -0.5]
    assert source_labels == {
        "Battery Storage Discharge": "Battery Storage",
        "Battery Storage Charge": "Battery Storage",
        "Solar": "Solar",
        "exports": "exports",
    }


def test_sort_area_plot_columns_by_variability_puts_lower_score_nearer_axis():
    df = pd.DataFrame(
        {
            "stable": [10.0, 10.2, 9.8, 10.1],
            "medium": [5.0, 7.0, 3.0, 5.0],
            "volatile": [1.0, 9.0, -7.0, 8.0],
        },
    )

    ordered = _sort_area_plot_columns_by_variability(df)

    assert ordered.columns.tolist() == ["stable", "medium", "volatile"]


def test_get_representative_period_snapshot_groups_splits_each_block():
    timesteps = pd.date_range("2030-01-01 00:00", periods=48, freq="h")
    snapshots = pd.MultiIndex.from_arrays(
        [np.repeat(2030, len(timesteps)), timesteps],
        names=["period", "timestep"],
    )

    groups = _get_representative_period_snapshot_groups(
        snapshots,
        {"enable": True, "period_length": 1},
    )

    assert [group["suffix"] for group in groups] == ["-rp01", "-rp02"]
    assert len(groups[0]["snapshots_by_period"][2030]) == 24
    assert len(groups[1]["snapshots_by_period"][2030]) == 24


def test_get_representative_period_snapshot_groups_uses_metadata_for_mixed_lengths():
    timesteps = pd.date_range("2030-01-01 00:00", periods=9, freq="h")
    snapshots = pd.MultiIndex.from_arrays(
        [np.repeat(2030, len(timesteps)), timesteps],
        names=["period", "timestep"],
    )
    representative_metadata = {
        "2030": {
            "periods": [
                {"period_id": 0, "kind": "representative", "steps": 4},
                {"period_id": 1, "kind": "extreme", "steps": 2},
                {"period_id": 2, "kind": "representative", "steps": 3},
            ],
        },
    }

    groups = _get_representative_period_snapshot_groups(
        snapshots,
        {"enable": True, "period_length": 1},
        representative_metadata,
    )

    assert [len(group["snapshots_by_period"][2030]) for group in groups] == [4, 2, 3]


def test_format_representative_period_title_names_the_weather_year():
    entry = {
        "period_id": 0,
        "kind": "representative",
        "steps": 24,
        "start": "1998-06-16T16:00:00",
        "end": "1998-06-17T15:00:00",
        "weather_years": [1998],
    }

    assert _format_representative_period_title(entry, "2050") == "Weather year 1998 (06-16 to 06-17)"


def test_format_representative_period_title_handles_blocks_spanning_two_years():
    entry = {
        "start": "2012-12-31T23:00:00",
        "end": "1998-01-01T00:00:00",
        "weather_years": [1998, 2012],
    }

    assert _format_representative_period_title(entry, "2050") == "Weather year 1998/2012 (12-31 to 01-01)"


def test_format_representative_period_title_derives_year_from_legacy_metadata():
    """Metadata written before ``weather_years`` existed still gets a weather-year title."""
    entry = {"start": "2007-03-04T00:00:00", "end": "2007-03-05T23:00:00"}

    assert _format_representative_period_title(entry, "2050") == "Weather year 2007 (03-04 to 03-05)"


def test_format_representative_period_title_falls_back_to_investment_period():
    assert _format_representative_period_title({"steps": 24}, "2050") == "2050"
    assert _format_representative_period_title(None, "2050") == "2050"
