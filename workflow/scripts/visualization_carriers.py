import re

import pandas as pd

MERGED_VISUALIZATION_LABELS = {
    "Gas w/o CCS": {"color_sources": ["CCGT"]},
    "Gas w/ CCS": {"color_sources": ["CCGT-95CCS"]},
    "Hydro & PHS": {"color_sources": ["hydro"]},
    "Battery Storage": {"color_sources": ["battery"]},
    "Offshore Wind": {"color_sources": ["offwind", "offwind_floating"]},
}


def _format_carrier_label(carrier: str) -> str:
    return carrier.replace("_", " ").strip().title()


def _canonicalize_carrier_name(carrier: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(carrier).strip().lower())


def strip_dispatch_suffix(carrier: str) -> str:
    carrier = str(carrier)
    for suffix in ("_charger", "_discharger", " charger", " discharger"):
        if carrier.endswith(suffix):
            return carrier[: -len(suffix)]
    return carrier


def _is_battery_storage_carrier(carrier: str) -> bool:
    carrier_key = _canonicalize_carrier_name(carrier)
    return carrier_key in {"battery", "batterystorage"} or bool(re.match(r"^\d+hrbatterystorage$", carrier_key))


def _is_hydro_phs_carrier(carrier: str) -> bool:
    carrier_key = _canonicalize_carrier_name(carrier)
    return carrier_key == "hydro" or "phs" in carrier_key


def get_merged_visualization_label(carrier: str) -> str | None:
    carrier = strip_dispatch_suffix(carrier)
    carrier_key = _canonicalize_carrier_name(carrier)
    if carrier_key in {"ocgt", "ccgt", "opencyclegas", "combinedcyclegas"}:
        return "Gas w/o CCS"
    if carrier_key in {"ccgt95ccs", "ccgt97ccs", "ccgt95cc", "ccgt97cc"}:
        return "Gas w/ CCS"
    if _is_hydro_phs_carrier(carrier):
        return "Hydro & PHS"
    if _is_battery_storage_carrier(carrier):
        return "Battery Storage"
    if carrier_key in {"offwind", "offwindfloating", "offshorewindfixed", "offshorewindfloating"}:
        return "Offshore Wind"
    return None


def get_visualization_label(
    carrier: str,
    carriers: pd.DataFrame,
    collapse_storage_directions: bool = False,
) -> str:
    carrier = str(carrier)
    base_carrier = strip_dispatch_suffix(carrier)
    merged_label = get_merged_visualization_label(base_carrier)
    if merged_label:
        return merged_label

    if collapse_storage_directions and carrier != base_carrier:
        carrier = base_carrier

    if carrier in carriers.index:
        nice_name = carriers.at[carrier, "nice_name"] if "nice_name" in carriers.columns else None
        if pd.notna(nice_name) and str(nice_name).strip():
            return str(nice_name)

    if base_carrier in carriers.index:
        nice_name = carriers.at[base_carrier, "nice_name"] if "nice_name" in carriers.columns else None
        if pd.notna(nice_name) and str(nice_name).strip():
            return str(nice_name)

    return _format_carrier_label(carrier)


def get_visualization_color(
    label: str,
    carriers: pd.DataFrame,
    fallback: str = "#000000",
) -> str:
    if label in MERGED_VISUALIZATION_LABELS:
        source_carriers = MERGED_VISUALIZATION_LABELS[label]["color_sources"]
        if "color" in carriers.columns:
            for source_carrier in source_carriers:
                if source_carrier in carriers.index:
                    color = carriers.at[source_carrier, "color"]
                    if pd.notna(color) and str(color).strip():
                        return str(color)

    if "nice_name" in carriers.columns and "color" in carriers.columns:
        label_match = carriers.index[carriers["nice_name"] == label]
        if len(label_match):
            color = carriers.at[label_match[0], "color"]
            if pd.notna(color) and str(color).strip():
                return str(color)

    if label in carriers.index and "color" in carriers.columns:
        color = carriers.at[label, "color"]
        if pd.notna(color) and str(color).strip():
            return str(color)

    return fallback


def build_visualization_carriers(
    carriers: pd.DataFrame,
    collapse_storage_directions: bool = False,
) -> pd.DataFrame:
    records = []
    for carrier in carriers.index:
        label = get_visualization_label(
            carrier,
            carriers,
            collapse_storage_directions=collapse_storage_directions,
        )
        color = get_visualization_color(label, carriers)
        records.append({"label": label, "color": color})

    visualization_carriers = pd.DataFrame.from_records(records).drop_duplicates(subset="label", keep="first")
    if visualization_carriers.empty:
        return pd.DataFrame(columns=["color"])
    return visualization_carriers.set_index("label").sort_index()


def build_visualization_palette(
    carriers: pd.DataFrame,
    labels: list[str] | pd.Index | None = None,
    collapse_storage_directions: bool = False,
    extra_colors: dict[str, str] | None = None,
) -> dict[str, str]:
    palette = build_visualization_carriers(
        carriers,
        collapse_storage_directions=collapse_storage_directions,
    )["color"].to_dict()

    if labels is not None:
        for label in labels:
            palette.setdefault(label, get_visualization_color(str(label), carriers))

    if extra_colors:
        palette.update(extra_colors)

    return palette


def aggregate_carrier_columns(
    df: pd.DataFrame,
    carriers: pd.DataFrame,
    collapse_storage_directions: bool = False,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    labels = pd.Index(
        [
            get_visualization_label(
                column,
                carriers,
                collapse_storage_directions=collapse_storage_directions,
            )
            for column in df.columns
        ],
        name=df.columns.name,
    )
    return df.T.groupby(labels, sort=False).sum().T


def aggregate_carrier_series(
    series: pd.Series,
    carriers: pd.DataFrame,
    collapse_storage_directions: bool = False,
) -> pd.Series:
    if series.empty:
        return series.copy()

    labels = pd.Index(
        [
            get_visualization_label(
                index,
                carriers,
                collapse_storage_directions=collapse_storage_directions,
            )
            for index in series.index
        ],
        name=series.index.name,
    )
    return series.groupby(labels, sort=False).sum()
