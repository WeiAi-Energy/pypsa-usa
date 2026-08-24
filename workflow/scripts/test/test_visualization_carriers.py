import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from visualization_carriers import (
    aggregate_carrier_columns,
    build_visualization_palette,
    get_visualization_label,
)


def make_carriers():
    return pd.DataFrame(
        {
            "nice_name": {
                "CCGT": "Combined-Cycle Gas",
                "OCGT": "Open-Cycle Gas",
                "hydro": "Hydro",
                "PHS": "PHS",
                "battery": "Battery",
                "4hr_battery_storage": "4Hr Battery Storage",
                "irb": "Iron-Air Battery",
                "offwind": "Offshore Wind Fixed",
                "offwind_floating": "Offshore Wind Floating",
                "DC": "DC",
                "AC": "AC",
            },
            "color": {
                "CCGT": "#111111",
                "OCGT": "#222222",
                "hydro": "#333333",
                "PHS": "#444444",
                "battery": "#555555",
                "4hr_battery_storage": "#666666",
                "irb": "#5a8f4e",
                "offwind": "#777777",
                "offwind_floating": "#888888",
                "DC": "#999999",
                "AC": "#aaaaaa",
            },
        },
    )


def test_get_visualization_label_merges_requested_carriers():
    carriers = make_carriers()

    assert get_visualization_label("CCGT", carriers) == "Gas w/o CCS"
    assert get_visualization_label("OCGT", carriers) == "Gas w/o CCS"
    assert get_visualization_label("hydro", carriers) == "Hydro & PHS"
    assert get_visualization_label("PHS", carriers) == "Hydro & PHS"
    assert get_visualization_label("battery", carriers) == "Battery Storage"
    assert get_visualization_label("4hr_battery_storage", carriers) == "Battery Storage"
    assert get_visualization_label("4Hr Battery Storage", carriers) == "Battery Storage"
    assert get_visualization_label("battery_discharger", carriers) == "Battery Storage"
    assert get_visualization_label("irb", carriers) == "Iron-Air Battery"
    assert get_visualization_label("Combined-Cycle Gas", carriers) == "Gas w/o CCS"
    assert get_visualization_label("offwind", carriers) == "Offshore Wind"
    assert get_visualization_label("offwind_floating", carriers) == "Offshore Wind"


def test_aggregate_carrier_columns_merges_duplicate_visualization_labels():
    carriers = make_carriers()
    df = pd.DataFrame(
        {
            "CCGT": [1.0, 2.0],
            "OCGT": [3.0, 4.0],
            "hydro": [5.0, 6.0],
            "PHS": [7.0, 8.0],
            "battery_discharger": [9.0, 10.0],
            "4hr_battery_storage_charger": [-2.0, -3.0],
            "offwind": [11.0, 12.0],
            "offwind_floating": [13.0, 14.0],
        },
    )

    aggregated = aggregate_carrier_columns(df, carriers, collapse_storage_directions=True)

    assert aggregated.columns.tolist() == [
        "Gas w/o CCS",
        "Hydro & PHS",
        "Battery Storage",
        "Offshore Wind",
    ]
    assert aggregated["Gas w/o CCS"].tolist() == [4.0, 6.0]
    assert aggregated["Hydro & PHS"].tolist() == [12.0, 14.0]
    assert aggregated["Battery Storage"].tolist() == [7.0, 7.0]
    assert aggregated["Offshore Wind"].tolist() == [24.0, 26.0]


def test_build_visualization_palette_uses_requested_source_colors():
    carriers = make_carriers()

    palette = build_visualization_palette(carriers)

    assert palette["Gas w/o CCS"] == "#111111"
    assert palette["Hydro & PHS"] == "#333333"
    assert palette["Battery Storage"] == "#555555"
    assert palette["Offshore Wind"] == "#777777"
