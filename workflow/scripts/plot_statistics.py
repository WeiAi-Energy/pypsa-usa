"""
Plots static and interactive charts to analyze system results.

**Inputs**

A solved network

**Outputs**

System level charts for:
    - Hourly production
    - Generator costs
    - Generator capacity

    .. image:: _static/plots/production-area.png
        :scale: 33 %

    .. image:: _static/plots/costs-bar.png
        :scale: 33 %

    .. image:: _static/plots/capacity-bar.png
        :scale: 33 %

Emission charts for:
    - Accumulated emissions

    .. image:: _static/plots/emissions-area.png
        :scale: 33 %
"""

import logging
import math
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pypsa
from postprocess_io import (
    load_postprocess_network,
    representative_period_metadata,
    snapshot_weight_series,
)
import seaborn as sns
from _helpers import configure_logging, set_case_config
from add_electricity import sanitize_carriers
from plot_network_maps import get_color_palette
from summary import (
    get_demand_timeseries,
    get_energy_timeseries,
    get_fuel_costs,
    get_generator_marginal_costs,
    get_node_emissions_timeseries,
    get_tech_emissions_timeseries,
)
from visualization_carriers import (
    aggregate_carrier_columns,
    build_visualization_carriers,
    build_visualization_palette,
    get_visualization_label,
)

logger = logging.getLogger(__name__)

# Global Plotting Settings
TITLE_SIZE = 16
ONE_BILLION = 1e9
DEFAULT_COST_FALLBACK_COLORS = {
    "Ac SSSC": "#9D755D",
}


def normalize_line_x_statistics_columns(n: pypsa.Network) -> None:
    """Backfill LineX columns expected by PyPSA statistics helpers."""
    line_xs = getattr(n, "line_xs", pd.DataFrame())
    if line_xs.empty:
        return

    if "capital_cost" not in line_xs.columns and "capital_cost_line" in line_xs.columns:
        n.line_xs["capital_cost"] = line_xs["capital_cost_line"]
    if "sssc_nom" not in line_xs.columns:
        n.line_xs["sssc_nom"] = line_xs.get("sssc_nom", pd.Series(0.0, index=line_xs.index, dtype=float)).fillna(0.0)


def get_export_lines_table(n: pypsa.Network) -> pd.DataFrame:
    """Return a line-like export table with LineX assets appended."""
    export_lines = n.lines.copy()
    export_lines["component"] = "Line"

    line_xs = getattr(n, "line_xs", pd.DataFrame())
    if line_xs.empty:
        return export_lines

    export_line_xs = line_xs.copy()
    export_line_xs["component"] = "LineX"

    export_columns = export_lines.columns.union(export_line_xs.columns)
    export_lines = export_lines.reindex(columns=export_columns)
    export_line_xs = export_line_xs.reindex(columns=export_columns)
    return pd.concat([export_lines, export_line_xs], axis=0)


def _sum_statistics_by_carrier(statistics: pd.DataFrame | pd.Series) -> pd.Series:
    """Collapse PyPSA statistics output to a single value per carrier."""
    if isinstance(statistics, pd.Series):
        totals = statistics.astype(float).fillna(0.0)
    else:
        totals = statistics.sum(axis=1, min_count=1).astype(float).fillna(0.0)

    if isinstance(totals.index, pd.MultiIndex):
        carrier_level = totals.index.names.index("carrier") if "carrier" in totals.index.names else -1
        totals = totals.groupby(level=carrier_level).sum()

    return totals.sort_index()


def _get_carrier_display_series(
    raw_carriers: pd.Series,
    n: pypsa.Network,
    suffix: str = "",
    use_visualization_labels: bool = False,
) -> pd.Series:
    """Map raw carrier names to display labels, preferring explicit carrier nice names."""

    def _to_display_label(carrier: object) -> str:
        carrier_str = str(carrier).strip()

        if carrier_str.endswith(" SSSC"):
            base_carrier = carrier_str.removesuffix(" SSSC").strip()
            if use_visualization_labels:
                return f"{str(get_visualization_label(base_carrier, n.carriers)).strip()} SSSC"
            if base_carrier in n.carriers.index and "nice_name" in n.carriers.columns:
                nice_name = n.carriers.at[base_carrier, "nice_name"]
                if pd.notna(nice_name) and str(nice_name).strip():
                    return f"{str(nice_name).strip()} SSSC"
            return f"{str(get_visualization_label(base_carrier, n.carriers)).strip()} SSSC"

        if use_visualization_labels:
            return str(get_visualization_label(carrier_str, n.carriers)).strip()

        if carrier_str in n.carriers.index and "nice_name" in n.carriers.columns:
            nice_name = n.carriers.at[carrier_str, "nice_name"]
            if pd.notna(nice_name) and str(nice_name).strip():
                return str(nice_name).strip()

        return str(get_visualization_label(carrier_str, n.carriers)).strip()

    display = raw_carriers.map(_to_display_label).fillna("Unknown").astype(str)
    if suffix:
        display = display + suffix
    return display


def _aggregate_costs_by_display_label(series: pd.Series, n: pypsa.Network) -> pd.Series:
    """Group a cost series by plotted carrier label."""
    if series.empty:
        return series.copy()

    labels = _get_carrier_display_series(
        pd.Series(series.index, index=series.index, dtype=object),
        n,
        use_visualization_labels=True,
    )
    return series.groupby(labels).sum().sort_index()


def _calculate_component_capex(
    df: pd.DataFrame,
    n: pypsa.Network,
    nominal_attr: str,
    extendable_col: str | None = None,
    optimal_col: str | None = None,
    capital_cost_col: str = "capital_cost",
) -> pd.Series:
    """Calculate CAPEX for extendable assets using capital_cost * (nom_opt - nom), grouped by carrier."""
    if df.empty:
        return pd.Series(dtype=float)

    extendable_col = extendable_col or f"{nominal_attr}_extendable"
    optimal_col = optimal_col or f"{nominal_attr}_opt"
    required_columns = {nominal_attr, optimal_col, capital_cost_col, "carrier"}
    if not required_columns.issubset(df.columns):
        return pd.Series(dtype=float)

    if extendable_col in df.columns:
        extendable_mask = df[extendable_col].fillna(False).astype(bool)
    else:
        extendable_mask = pd.Series(True, index=df.index, dtype=bool)

    expandable = df.loc[extendable_mask, [capital_cost_col, nominal_attr, optimal_col, "carrier"]].copy()
    if expandable.empty:
        return pd.Series(dtype=float)

    capex = (
        expandable[capital_cost_col].fillna(0.0)
        * (
        expandable[optimal_col].fillna(expandable[nominal_attr].fillna(0.0)) - expandable[nominal_attr].fillna(0.0)
    )
    )
    carrier_labels = _get_carrier_display_series(
        expandable["carrier"],
        n,
        use_visualization_labels=True,
    )
    return capex.groupby(carrier_labels).sum().sort_index()


def _calculate_line_x_capex(n: pypsa.Network) -> tuple[pd.Series, pd.Series]:
    """Return LineX line-part and SSSC-part CAPEX separately, grouped by carrier."""
    line_xs = getattr(n, "line_xs", pd.DataFrame())
    if line_xs.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    line_capex = pd.Series(dtype=float)
    if "s_nom_extendable" in line_xs.columns:
        extendable_mask = line_xs["s_nom_extendable"].fillna(False).astype(bool)
    else:
        extendable_mask = pd.Series(True, index=line_xs.index, dtype=bool)

    expandable = line_xs.loc[extendable_mask].copy()
    if not expandable.empty:
        s_nom = expandable.get("s_nom", pd.Series(0.0, index=expandable.index, dtype=float)).fillna(0.0)
        s_nom_line_opt = expandable.get(
            "s_nom_line_opt",
            expandable.get("s_nom_opt", pd.Series(0.0, index=expandable.index, dtype=float)),
        ).fillna(0.0)
        capital_cost_line = expandable.get(
            "capital_cost_line",
            expandable.get("capital_cost", pd.Series(0.0, index=expandable.index, dtype=float)),
        ).fillna(0.0)

        line_capex = capital_cost_line * (s_nom_line_opt - s_nom)
        carrier_labels = _get_carrier_display_series(
            expandable["carrier"],
            n,
            use_visualization_labels=True,
        )
        line_capex = line_capex.groupby(carrier_labels).sum().sort_index()

    sssc_nom = line_xs.get("sssc_nom", pd.Series(0.0, index=line_xs.index, dtype=float)).fillna(0.0)
    sssc_nom_opt = line_xs.get("sssc_nom_opt", pd.Series(0.0, index=line_xs.index, dtype=float)).fillna(0.0)
    capital_cost_sssc = line_xs.get("capital_cost_sssc", pd.Series(0.0, index=line_xs.index, dtype=float)).fillna(
        0.0,
    )
    sssc_capex = capital_cost_sssc * (sssc_nom_opt - sssc_nom)
    sssc_labels = _get_carrier_display_series(
        line_xs["carrier"],
        n,
        suffix=" SSSC",
        use_visualization_labels=True,
    )
    sssc_capex = sssc_capex.groupby(sssc_labels).sum().sort_index()

    return line_capex, sssc_capex


def build_statistics_overview_table(n: pypsa.Network) -> pd.DataFrame:
    """Build a carrier-level statistics table directly from PyPSA's statistics output."""
    base = n.statistics(nice_names=False)
    if base.empty:
        return pd.DataFrame()

    if not isinstance(base.index, pd.MultiIndex) or "carrier" not in base.index.names:
        overview = base.copy()
        overview = overview.fillna(0.0)
        overview = overview.loc[overview.abs().sum(axis=1) > 0.0].sort_index()
        return overview

    carrier_labels = pd.Series(base.index.get_level_values("carrier"), index=base.index, dtype=object)
    display_labels = _get_carrier_display_series(carrier_labels, n)
    overview = base.groupby(display_labels).sum().fillna(0.0)
    overview = overview.loc[overview.abs().sum(axis=1) > 0.0].sort_index()
    return overview


def _get_transmission_volume_table(n: pypsa.Network) -> pd.DataFrame:
    """Return installed and optimal length*capacity totals for AC/DC assets."""
    carrier_filter = {"AC", "DC"}
    parts = []

    for component, df, installed_col, optimal_cols in (
        ("Line", getattr(n, "lines", pd.DataFrame()), "s_nom", ["s_nom_opt"]),
        ("Link", getattr(n, "links", pd.DataFrame()), "p_nom", ["p_nom_opt"]),
        ("LineX", getattr(n, "line_xs", pd.DataFrame()), "s_nom", ["s_nom_line_opt", "s_nom_opt"]),
    ):
        if df.empty or not {"carrier", "length"}.issubset(df.columns):
            continue

        transmission = df[df["carrier"].isin(carrier_filter)].copy()
        if transmission.empty:
            continue

        installed_capacity = transmission.get(
            installed_col,
            pd.Series(0.0, index=transmission.index, dtype=float),
        ).fillna(0.0)
        optimal_capacity = installed_capacity.copy()
        for col in optimal_cols:
            if col in transmission.columns:
                optimal_capacity = transmission[col].fillna(optimal_capacity)
                break

        part = pd.DataFrame(
            {
                "Installed Volume": (transmission["length"].fillna(0.0) * installed_capacity).to_numpy(),
                "Optimal Volume": (transmission["length"].fillna(0.0) * optimal_capacity).to_numpy(),
            },
            index=pd.MultiIndex.from_arrays(
                [
                    pd.Index([component] * len(transmission), dtype=object),
                    transmission["carrier"].astype(str),
                ],
                names=["component", "carrier"],
            ),
        )
        parts.append(part)

    if not parts:
        return pd.DataFrame(columns=["Installed Volume", "Optimal Volume"])

    return pd.concat(parts).groupby(level=[0, 1]).sum().sort_index()


def build_statistics_summary_table(n: pypsa.Network) -> pd.DataFrame:
    """Build the exported statistics.csv table from the merged carrier-level overview."""
    summary_columns = [
        "Installed Capacity",
        "Optimal Capacity",
        "Capacity Factor",
        "Curtailment",
        "Capital Expenditure",
    ]
    overview = build_statistics_overview_table(n)
    available_columns = [column for column in summary_columns if column in overview.columns]
    summary = overview.loc[:, available_columns].copy()
    volume_columns = pd.DataFrame(0.0, index=summary.index, columns=["Installed Volume", "Optimal Volume"])
    volume_table = _get_transmission_volume_table(n)

    if not volume_table.empty and not summary.empty:
        carrier_lookup = {str(carrier): str(carrier) for carrier in n.carriers.index}
        if "nice_name" in n.carriers.columns:
            for carrier, nice_name in n.carriers["nice_name"].dropna().items():
                carrier_lookup[str(nice_name).strip()] = str(carrier)

        if isinstance(summary.index, pd.MultiIndex):
            index_names = list(summary.index.names)
            component_level = index_names.index("component") if "component" in index_names else 0
            carrier_level = index_names.index("carrier") if "carrier" in index_names else len(index_names) - 1
            aligned_index = pd.MultiIndex.from_arrays(
                [
                    summary.index.get_level_values(component_level).astype(str),
                    pd.Index(
                        [
                            carrier_lookup.get(str(carrier).strip(), str(carrier).strip())
                            for carrier in summary.index.get_level_values(carrier_level)
                        ],
                        dtype=object,
                    ),
                ],
                names=volume_table.index.names,
            )
            volume_columns = volume_table.reindex(aligned_index).fillna(0.0)
            volume_columns.index = summary.index
        else:
            carrier_index = pd.Index(
                [carrier_lookup.get(str(carrier).strip(), str(carrier).strip()) for carrier in summary.index],
                name="carrier",
                dtype=object,
            )
            volume_columns = volume_table.groupby(level="carrier").sum().reindex(carrier_index).fillna(0.0)
            volume_columns.index = summary.index

    summary["Installed Volume"] = volume_columns["Installed Volume"]
    summary["Optimal Volume"] = volume_columns["Optimal Volume"]
    ordered_columns = [
        "Installed Capacity",
        "Optimal Capacity",
        "Installed Volume",
        "Optimal Volume",
        "Capacity Factor",
        "Curtailment",
        "Capital Expenditure",
    ]
    summary = summary[[column for column in ordered_columns if column in summary.columns]]

    if "Curtailment" in summary.columns and "Supply" in overview.columns:
        curtailment = overview["Curtailment"].fillna(0.0)
        supply = overview["Supply"].fillna(0.0)
        denominator = curtailment + supply
        summary["Curtailment"] = curtailment.div(denominator.where(denominator != 0.0)).fillna(0.0)

    return summary


def get_carrier_cost_breakdown(n: pypsa.Network) -> pd.DataFrame:
    """Return system OPEX and CAPEX aggregated by carrier."""
    opex = _aggregate_costs_by_display_label(_sum_statistics_by_carrier(n.statistics.opex()), n)

    capex_parts = [
        _calculate_component_capex(n.generators, n, "p_nom"),
        _calculate_component_capex(n.storage_units, n, "p_nom"),
        _calculate_component_capex(n.links, n, "p_nom"),
        _calculate_component_capex(n.stores, n, "e_nom"),
        _calculate_component_capex(n.lines, n, "s_nom"),
    ]
    capex = pd.concat(capex_parts).groupby(level=0).sum() if capex_parts else pd.Series(dtype=float)

    line_x_line_capex, line_x_sssc_capex = _calculate_line_x_capex(n)
    if not line_x_line_capex.empty:
        capex = pd.concat([capex, line_x_line_capex]).groupby(level=0).sum()
    if not line_x_sssc_capex.empty:
        capex = pd.concat([capex, line_x_sssc_capex]).groupby(level=0).sum()

    costs = pd.concat([opex.rename("OPEX"), capex.rename("CAPEX")], axis=1).fillna(0.0)
    costs = costs.loc[(costs.abs().sum(axis=1) > 0.0)].sort_values(
        by=["OPEX", "CAPEX"],
        ascending=False,
    )
    return costs


def _adjust_color(color: str, factor: float = 0.85) -> str:
    """Darken or lighten a color by multiplying RGB values."""
    rgb = np.array(mcolors.to_rgb(color))
    adjusted = np.clip(rgb * factor, 0.0, 1.0)
    return mcolors.to_hex(adjusted)


def _resolve_plot_color(color, fallback) -> str:
    """Return a valid matplotlib color, falling back when the input is missing or invalid."""
    if pd.notna(color) and mcolors.is_color_like(color):
        return color
    return fallback


def _get_cost_carrier_palette(n: pypsa.Network, labels: list[str]) -> dict[str, str]:
    """Build a color palette for carrier-based cost charts."""
    nice_name_map = n.carriers.get("nice_name", pd.Series(dtype=object))
    base_palette = {}
    for carrier, nice_name in nice_name_map.items():
        if pd.isna(nice_name):
            continue
        color = n.carriers.at[carrier, "color"] if "color" in n.carriers.columns else None
        if pd.notna(color) and mcolors.is_color_like(color):
            base_palette[str(nice_name)] = color

    palette = {}
    for i, label in enumerate(labels):
        fallback = DEFAULT_COST_FALLBACK_COLORS.get(label, plt.get_cmap("tab20")(i % 20))
        if label.endswith(" SSSC"):
            base_label = label.removesuffix(" SSSC")
            base_color = _resolve_plot_color(base_palette.get(base_label), fallback)
            palette[label] = _adjust_color(base_color, factor=0.75)
        else:
            palette[label] = _resolve_plot_color(base_palette.get(label), fallback)
    return palette


def _spread_label_positions(y_values: list[float], min_gap: float) -> list[float]:
    """Spread annotation labels vertically to reduce overlaps."""
    if not y_values:
        return []

    adjusted = [y_values[0]]
    for y in y_values[1:]:
        adjusted.append(max(y, adjusted[-1] + min_gap))
    return adjusted


def _annotate_stacked_cost_bar(
    ax: plt.Axes,
    x_position: float,
    segments: list[dict[str, float | str]],
    chart_max: float,
) -> None:
    """Annotate each stacked-bar segment with a leader line and cost label."""
    min_gap = max(chart_max * 0.05, 0.15)
    label_offset = 0.6
    stem_offset = 0.28

    for sign, ha in ((1, "left"), (-1, "right")):
        group = [
            segment
            for segment in segments
            if np.sign(float(segment["value"])) == sign
            and float(segment["value"]) != 0.0
        ]
        if not group:
            continue

        group = sorted(group, key=lambda segment: float(segment["center"]))
        label_positions = _spread_label_positions(
            [float(segment["center"]) for segment in group],
            min_gap=min_gap,
        )
        x_text = x_position + label_offset if sign > 0 else x_position - label_offset
        x_stem = x_position + stem_offset if sign > 0 else x_position - stem_offset

        for segment, y_text in zip(group, label_positions):
            label = str(segment["component"])
            value = float(segment["value"])
            ax.annotate(
                f"{label}: {value:.2f}",
                xy=(x_stem, float(segment["center"])),
                xytext=(x_text, y_text),
                ha=ha,
                va="center",
                fontsize=9,
                arrowprops={"arrowstyle": "-", "linewidth": 0.8, "color": "#444444"},
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9},
            )


def plot_component_cost_breakdown(
    n: pypsa.Network,
    save: str,
    **wildcards,
) -> None:
    """Plot stacked OPEX and CAPEX bars aggregated by carrier."""
    costs = get_carrier_cost_breakdown(n)
    if costs.empty:
        logger.warning("No component costs available for plotting.")
        return

    costs_billion = costs / ONE_BILLION
    fig, axes = plt.subplots(1, 2, figsize=(20, 14))
    present_labels = costs_billion.index.tolist()
    palette = _get_cost_carrier_palette(n, present_labels)
    combined_total = costs_billion[["OPEX", "CAPEX"]].to_numpy().sum()

    for ax, column in zip(axes, ["OPEX", "CAPEX"]):
        series = costs_billion[column].sort_values(ascending=False)
        positive_total = 0.0
        negative_total = 0.0
        segments: list[dict[str, float | str]] = []

        for label, value in series.items():
            if np.isclose(value, 0.0):
                continue

            bottom = positive_total if value >= 0 else negative_total
            ax.bar(
                0,
                value,
                bottom=bottom,
                width=0.65,
                color=palette[label],
            )
            segments.append(
                {
                    "component": label,
                    "value": float(value),
                    "center": float(bottom + value / 2),
                },
            )

            if value >= 0:
                positive_total += value
            else:
                negative_total += value

        chart_max = max(abs(positive_total), abs(negative_total), abs(series.sum()), 1.0)
        top_padding = max(chart_max * 0.2, 0.5)
        bottom_padding = max(chart_max * 0.15, 0.5)
        ax.set_ylim(negative_total - bottom_padding, positive_total + top_padding)
        ax.set_xlim(-1.6, 1.6)

        _annotate_stacked_cost_bar(ax, 0, segments, chart_max=chart_max)

        total_cost = series.sum()
        ax.text(
            0,
            positive_total + top_padding * 0.45,
            f"Total: {total_cost:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks([0])
        ax.set_xticklabels([column], fontsize=11)
        ax.set_title(f"{column} by Carrier", fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Cost [billion USD]")
    fig.suptitle(
        create_title("System Cost Breakdown by Carrier", **wildcards),
        fontsize=TITLE_SIZE,
        y=0.985,
    )
    fig.text(
        0.5,
        0.925,
        f"CAPEX + OPEX: {combined_total:.2f} billion USD",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[label])
        for label in present_labels
    ]
    fig.legend(
        legend_handles,
        present_labels,
        loc="lower center",
        ncol=min(4, len(present_labels)),
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.85])
    fig.savefig(save, bbox_inches="tight")
    plt.close()


def create_title(title: str, **wildcards) -> str:
    """
    Standardizes wildcard writing in titles.

    Arguments:
        title: str
            Title of chart to plot
        **wildcards
            any wildcards to add to title
    """
    w = []
    for wildcard, value in wildcards.items():
        if wildcard == "interconnect":
            w.append(f"interconnect = {value}")
        elif wildcard == "ll":
            w.append(f"ll = {value}")
        elif wildcard == "opts":
            w.append(f"opts = {value}")
        elif wildcard == "sector":
            w.append(f"sectors = {value}")
    wildcards_joined = " | ".join(w)
    return f"{title} \n ({wildcards_joined})"


def stacked_bar_horizons(
    stats,
    variable,
    variable_units,
    carriers,
):
    if "nice_name" in carriers.columns:
        carriers = build_visualization_carriers(carriers)
    else:
        carriers = carriers.copy()
    technologies = pd.Index(
        sorted(
            {
                str(technology)
                for df in stats.values()
                for technology in df.index.unique()
            },
        ),
    )
    carriers = carriers.reindex(carriers.index.union(technologies))
    for technology in technologies:
        carriers.loc[technology, "color"] = carriers.loc[technology, "color"] if pd.notna(
            carriers.loc[technology, "color"],
        ) else "#000000"
    colors_ = carriers["color"]
    carriers_legend = carriers  # to track which carriers have non-zero values
    # Create subplots
    planning_horizons = stats[next(iter(stats.keys()))].columns
    fig, axes = plt.subplots(
        nrows=len(planning_horizons),
        ncols=1,
        figsize=(8, 1.2 * len(planning_horizons)),
        sharex=True,
    )

    # Ensure axes is always iterable (even if there's only one planning horizon)
    if len(planning_horizons) == 1:
        axes = [axes]

    # Loop through each planning horizon
    for ax, horizon in zip(axes, planning_horizons):
        y_positions = np.arange(len(stats))  # One position for each scenario
        for j, (scenario, df) in enumerate(stats.items()):
            bottoms = np.zeros(
                len(df.columns),
            )  # Initialize the bottom positions for stacking
            # Stack the technologies for each scenario
            for i, technology in enumerate(df.index.unique()):
                values = df.loc[technology, horizon]
                values = values / (1e3) if "GW" in variable_units else values
                ax.barh(
                    y_positions[j],
                    values,
                    left=bottoms[j],
                    color=colors_[technology],
                    label=technology if j == 0 else "",
                )
                bottoms[j] += values
                carriers_legend.loc[technology, "value"] = values

        # Set the title for each subplot
        ax.text(
            1.01,
            0.5,
            f"{horizon}",
            transform=ax.transAxes,
            va="center",
            rotation="vertical",
        )
        ax.set_yticks(y_positions)  # Positioning scenarios on the y-axis
        ax.set_yticklabels(stats.keys())  # Labeling y-axis with scenario names
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Create legend handles and labels from the carriers DataFrame
    carriers_legend = carriers_legend[carriers_legend["value"] > 0.01]
    colors_ = carriers_legend["color"]
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors_[tech]) for tech in carriers_legend.index]
    # fig.legend(handles=legend_handles, labels=carriers.index.tolist(), loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=4, title='Technologies')
    ax.legend(
        handles=legend_handles,
        labels=carriers_legend.index.tolist(),
        loc="upper center",
        bbox_to_anchor=(0.5, -1.3),
        ncol=4,
        title="Technologies",
    )

    fig.subplots_adjust(hspace=0, bottom=0.5)
    fig.suptitle(f"{variable}", fontsize=12, fontweight="bold")
    plt.xlabel(f"{variable} {variable_units}")
    # fig.tight_layout()
    # plt.show(block=True)
    return fig


#### Bar Plots ####
def plot_capacity_additions_bar(
    n: pypsa.Network,
    carriers_2_plot: list[str],
    save: str,
    **wildcards,
) -> None:
    """Plots base capacity vs optimal capacity as a bar chart."""
    existing_capacity = n.generators.groupby("carrier").p_nom.sum().round(0)
    existing_capacity = existing_capacity.to_frame(name="Existing Capacity")
    storage_units = n.storage_units.groupby("carrier").p_nom.sum().round(0)
    storage_units = storage_units.to_frame(name="Existing Capacity")
    existing_capacity = pd.concat([existing_capacity, storage_units])
    existing_capacity.index = existing_capacity.index.map(lambda carrier: get_visualization_label(carrier, n.carriers))
    existing_capacity = existing_capacity.groupby(level=0).sum()

    optimal_capacity = n.statistics.optimal_capacity()
    optimal_capacity = optimal_capacity[optimal_capacity.index.get_level_values(0).isin(["Generator", "StorageUnit"])]
    optimal_capacity.index = optimal_capacity.index.droplevel(0)
    optimal_capacity = optimal_capacity.reset_index()
    optimal_capacity = optimal_capacity.rename(columns={"index": "carrier"})

    optimal_capacity = optimal_capacity.set_index("carrier")
    optimal_capacity.index = optimal_capacity.index.map(lambda carrier: get_visualization_label(carrier, n.carriers))
    optimal_capacity = optimal_capacity.groupby(level=0).sum()
    optimal_capacity.insert(0, "Existing", existing_capacity["Existing Capacity"])
    optimal_capacity = optimal_capacity.fillna(0)

    stats = {"": optimal_capacity}
    variable = "Optimal Capacity"
    variable_units = " GW"
    fig_ = stacked_bar_horizons(stats, variable, variable_units, build_visualization_carriers(n.carriers))
    fig_.savefig(save)
    plt.close()


def plot_production_bar(
    n: pypsa.Network,
    carriers_2_plot: list[str],
    save: str,
    **wildcards,
) -> None:
    """Plot diaptch per carrier."""
    energy_mix = n.statistics.supply().round(0)
    energy_mix = energy_mix[
        energy_mix.index.get_level_values("component").isin(
            ["Generator", "StorageUnit"],
        )
    ]
    energy_mix.index = energy_mix.index.droplevel(0)
    energy_mix = energy_mix.fillna(0)
    energy_mix.index = energy_mix.index.map(lambda carrier: get_visualization_label(carrier, n.carriers))
    energy_mix = energy_mix.groupby(level=0).sum()
    stats = {"": energy_mix}
    variable = "Energy Mix"
    variable_units = " GWh"

    fig_ = stacked_bar_horizons(stats, variable, variable_units, build_visualization_carriers(n.carriers))
    fig_.savefig(save)
    plt.close()


def plot_global_constraint_shadow_prices(
    n: pypsa.Network,
    save: str,
    **wildcards,
) -> None:
    """Plots shadow prices on global constraints."""
    shadow_prices = n.global_constraints.mu.round(3).reset_index()

    # plot data
    fig, ax = plt.subplots(figsize=(10, 10))

    sns.barplot(
        y=shadow_prices.GlobalConstraint,
        x=shadow_prices.mu,
        data=shadow_prices,
        color="purple",
        ax=ax,
    )

    ax.set_title(create_title("Shadow Prices on Constraints", **wildcards))
    ax.set_ylabel("")
    ax.set_xlabel("Shadow Price [$/MWh]")
    fig.tight_layout()
    fig.savefig(save)
    plt.close()


def get_currently_installed_capacity(n: pypsa.Network) -> pd.DataFrame:
    """Returns a DataFrame with the currently installed capacity for each carrier and nerc region."""
    n.generators["nerc_reg"] = n.generators.bus.map(n.buses.nerc_reg)
    existing_capacity = n.generators.groupby(["nerc_reg", "carrier"]).p_nom.sum().round(0)
    existing_capacity = existing_capacity.to_frame(name="Existing")
    n.storage_units["nerc_reg"] = n.storage_units.bus.map(n.buses.nerc_reg)
    storage_units = n.storage_units.groupby(["nerc_reg", "carrier"]).p_nom.sum().round(0)
    storage_units = storage_units.to_frame(name="Existing")
    existing_capacity = pd.concat([existing_capacity, storage_units])

    # Groupby regions and carriers, then fix indexing
    existing_capacity = existing_capacity.groupby(existing_capacity.index).sum()
    existing_capacity = existing_capacity.reset_index()
    existing_capacity[["Region", "Carrier"]] = pd.DataFrame(
        existing_capacity["index"].tolist(),
        index=existing_capacity.index,
    )
    existing_capacity = existing_capacity.drop(columns="index")
    existing_capacity = existing_capacity.set_index(["Region", "Carrier"])

    nn_carriers = existing_capacity.index.get_level_values(1).map(
        lambda carrier: get_visualization_label(carrier, n.carriers),
    )
    existing_capacity = existing_capacity.droplevel(1)
    existing_capacity = existing_capacity.set_index(nn_carriers, append=True)
    return existing_capacity.groupby(level=[0, 1]).sum()


def get_statistics(n, column_name):
    """
    Prepare the statistics data for plotting by extracting and grouping by region and carrier.

    Parameters
    ----------
    - n: pypsa.Network
    - column_name: str, the name of the column to extract from statistics (e.g., 'Optimal Capacity', 'Supply')

    Returns
    -------
    - pd.DataFrame: Prepared and grouped data
    """
    groupers = n.statistics.groupers
    df = n.statistics(groupby=groupers.get_name_bus_and_carrier).round(3)
    df = df.loc[["Generator", "StorageUnit"]]

    # Add nerc_region data
    gens = df.loc["Generator"].index.get_level_values(0)
    gens_reg = gens.map(n.generators.bus.map(n.buses.nerc_reg)).to_series()
    su = df.loc["StorageUnit"].index.get_level_values(0)
    su_reg = su.map(n.storage_units.bus.map(n.buses.nerc_reg)).to_series()
    nerc_reg = pd.concat([gens_reg, su_reg])

    df = df.set_index(nerc_reg, append=True)
    df = df.droplevel([0, 1, 2])
    df = df.reset_index()
    df = df.rename(columns={"level_0": "carrier", "level_1": "region"})
    df = df.set_index(["region", "carrier"])

    df_selected = df[column_name]
    df_selected = df_selected.groupby(df_selected.index).sum()
    df_selected = df_selected.reset_index()
    df_selected[["Region", "Carrier"]] = pd.DataFrame(
        df_selected["index"].tolist(),
        index=df_selected.index,
    )
    df_selected = df_selected.drop(columns="index")
    df_selected = df_selected.set_index(["Region", "Carrier"])
    carriers = df_selected.index.get_level_values("Carrier").map(
        lambda carrier: get_visualization_label(carrier, n.carriers),
    )
    df_selected = df_selected.droplevel("Carrier")
    df_selected = df_selected.set_index(carriers, append=True)
    df_selected = df_selected.groupby(level=[0, 1]).sum()

    return df_selected


def plot_bar(data, n, save, title, ylabel, is_capacity=False):
    """
    Plot the data in a bar chart with subplots by region and carrier.

    Parameters
    ----------
    - data: pd.DataFrame, data to plot
    - n: pypsa.Network
    - save: str, file path to save the plot
    - title: str, plot title
    - ylabel: str, y-axis label
    - is_capacity: bool, whether to add extra processing for capacities
    """
    if is_capacity:
        existing_cap = get_currently_installed_capacity(n)
        combined_index = data.index.union(existing_cap.index)
        data = data.reindex(combined_index)
        data["Existing"] = existing_cap["Existing"].reindex(combined_index).fillna(0.0)
        data = data.fillna(0.0)
        data = data[["Existing"] + [col for col in data.columns if col != "Existing"]]
        retirements = data.diff(axis=1).clip(upper=0)
        retirements = retirements[(retirements < -0.001).any(axis=1)]
        retirements = retirements.fillna(0)
        data = pd.concat([data, retirements])

    data = data / 1e3  # Convert to GW

    palette = build_visualization_palette(n.carriers, labels=data.index.get_level_values(1).unique())
    regions = data.index.get_level_values(0).unique()

    # Determine grid layout for subplots
    num_regions = len(regions)
    columns = min(5, num_regions)  # Limit to 5 columns
    rows = math.ceil(num_regions / columns)

    # Set up the figure and axes
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2.5, rows * 5),
        sharex=True,
        sharey=True,
    )

    # Ensure axes is a flattened array for consistent indexing
    if num_regions == 1:
        axes = [axes]  # Wrap single Axes object in a list
    else:
        axes = axes.flatten()

    for i, region in enumerate(regions):
        region_data = data.loc[region]
        region_data.T.plot(
            kind="bar",
            stacked=True,
            ax=axes[i],
            color=[palette.get(carrier) for carrier in region_data.index.get_level_values(0)],
            legend=False,
        )
        axes[i].axhline(0, color="black", linewidth=0.8)
        axes[i].set_title(region)
        axes[i].set_ylabel(ylabel)
        axes[i].set_xlabel("")

    # Remove unused subplots
    if num_regions > 1:
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

    # Create legend
    handles, labels = [], []
    for carrier in data.reset_index().Carrier.unique():
        handle = plt.Rectangle((0, 0), 1, 1, color=palette[carrier])
        handles.append(handle)
        labels.append(f"{carrier}")
    fig.legend(handles, labels, title="Carrier", loc="lower center", ncol=columns)

    plt.tight_layout(rect=[0, 0.3, 1, 1])
    plt.subplots_adjust(wspace=0.4)
    fig.suptitle(title)
    fig.savefig(save)
    plt.close()


def plot_regional_capacity_additions_bar(n, save):
    """Plot capacity evolution by NERC region in a stacked bar plot."""
    data = get_statistics(n, "Optimal Capacity")
    data.to_csv(f"{Path(save).parent.parent}/statistics/bar_regional_capacity.csv")
    plot_bar(data, n, save, "", "Capacity (GW)", is_capacity=True)


def plot_regional_production_bar(n, save):
    """Plot production evolution by NERC region in a stacked bar plot."""
    data = get_statistics(n, "Supply")
    data.to_csv(f"{Path(save).parent.parent}/statistics/bar_regional_production.csv")
    plot_bar(data, n, save, "", "Production (GWh)")


def plot_regional_emissions_bar(
    n: pypsa.Network,
    save: str,
) -> None:
    """PLOT OF CO2 EMISSIONS BY NERC REGION AND INVESTMENT PERIOD."""
    regional_emisssions_ts = get_node_emissions_timeseries(n).T.groupby(n.buses.nerc_reg).sum().T / 1e6
    regional_emissions = (
        regional_emisssions_ts.groupby(regional_emisssions_ts.index.get_level_values(0)).sum().round(3).T
    )

    # Determine grid layout for subplots
    regions = regional_emissions.index.get_level_values(0).unique()
    num_regions = len(regions)
    columns = min(5, num_regions)  # Limit to 5 columns
    rows = math.ceil(num_regions / columns)

    # Set up the figure and axes
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2.5, rows * 5),
        sharex=True,
        sharey=True,
    )

    # Ensure axes is a flattened array for consistent indexing
    if num_regions == 1:
        axes = [axes]  # Wrap single Axes object in a list
    else:
        axes = axes.flatten()

    for i, region in enumerate(regions):
        region_data = regional_emissions.loc[region]
        region_data.T.plot(
            kind="bar",
            stacked=True,
            ax=axes[i],
            legend=False,
        )
        axes[i].axhline(0, color="black", linewidth=0.8)
        axes[i].set_title(region)
        axes[i].set_ylabel("MMtCo2")
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.3, 1, 1])
    plt.subplots_adjust(wspace=0.4)

    plt.xlabel("")
    plt.ylabel("MMtCO2")

    plt.tight_layout()
    plt.savefig(save)
    plt.close()


def plot_emissions_bar(
    n: pypsa.Network,
    save: str,
) -> None:
    """PLOT OF CO2 EMISSIONS BY INVESTMENT PERIOD."""
    emisssions_ts = get_node_emissions_timeseries(n).T.sum().T / 1e6
    emissions = emisssions_ts.groupby(emisssions_ts.index.get_level_values(0)).sum().round(3).T

    # Set up the figure and axes
    _, ax = plt.subplots(figsize=(7, 4))
    emissions.T.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        legend=False,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("MMtCo2")
    ax.set_xlabel("")

    plt.tight_layout(rect=[0, 0.3, 1, 1])
    plt.subplots_adjust(wspace=0.4)

    plt.xlabel("CO2 Emissions [MMtCO2]")
    plt.ylabel("MMtCO2")

    plt.tight_layout()
    plt.savefig(save)
    plt.close()


#### Temporal Plots ####


def _get_special_link_power_timeseries(n: pypsa.Network) -> pd.DataFrame:
    """Return plot-ready link power for components that need explicit sign handling."""
    special_link_power = pd.DataFrame(index=n.snapshots)

    if n.links.empty:
        return special_link_power

    tes_links = n.links[n.links.carrier == "tes"]
    if not tes_links.empty:
        tes_charge_links = tes_links[tes_links.index.str.endswith(" tes charger")].index
        tes_discharge_links = tes_links[tes_links.index.str.endswith(" tes discharger")].index

        if len(tes_charge_links):
            special_link_power["TES_charge"] = n.links_t.p0[tes_charge_links].sum(axis=1).mul(-1)

        if len(tes_discharge_links):
            # Use electrical output at bus1 so discharge is reported in electric power units.
            special_link_power["TES_discharge"] = n.links_t.p1[tes_discharge_links].sum(axis=1).mul(-1)

    flexible_electrolysis_links = n.links[
        (n.links.carrier == "electrolysis") & (n.links.index.str.contains("flexible electrolysis"))
    ].index
    if len(flexible_electrolysis_links):
        special_link_power["flexible_electrolysis"] = (
            n.links_t.p0[flexible_electrolysis_links].sum(axis=1).mul(-1)
        )

    if special_link_power.empty:
        return special_link_power

    return special_link_power.loc[:, (special_link_power.abs() > 1e-6).any(axis=0)]


def _get_area_plot_split_labels(label: str) -> tuple[str, str]:
    """Return positive/negative legend labels for mixed-sign area-plot series."""
    normalized = "".join(ch for ch in str(label).lower() if ch.isalnum())
    if any(token in normalized for token in ("battery", "storage", "tes", "phs")):
        return f"{label} Discharge", f"{label} Charge"
    return f"{label} Positive", f"{label} Negative"


def _split_mixed_sign_area_series(
    df: pd.DataFrame,
    tolerance: float = 1e-9,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Split mixed-sign columns so each stacked area series is sign-consistent."""
    if df.empty:
        return df.copy(), {}

    prepared = pd.DataFrame(index=df.index)
    source_labels: dict[str, str] = {}

    for column in df.columns:
        series = df[column].fillna(0.0)
        has_positive = (series > tolerance).any()
        has_negative = (series < -tolerance).any()

        if has_positive and has_negative:
            positive_label, negative_label = _get_area_plot_split_labels(str(column))
            positive = series.clip(lower=0)
            negative = series.clip(upper=0)

            if (positive.abs() > tolerance).any():
                prepared[positive_label] = positive
                source_labels[positive_label] = str(column)
            if (negative.abs() > tolerance).any():
                prepared[negative_label] = negative
                source_labels[negative_label] = str(column)
        else:
            prepared[column] = series
            source_labels[str(column)] = str(column)

    return prepared, source_labels


def _sort_area_plot_columns_by_variability(df: pd.DataFrame) -> pd.DataFrame:
    """Place lower variance-to-mean series closer to the area-chart baseline."""
    if df.empty:
        return df.copy()

    def score(column: str) -> tuple[float, str]:
        series = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        mean_abs = abs(series.mean())
        variance_abs = abs(series.var(ddof=0))
        if mean_abs <= 1e-12:
            metric = 0.0 if variance_abs <= 1e-12 else math.inf
        else:
            metric = variance_abs / mean_abs
        return metric, str(column)

    ordered_columns = sorted(df.columns, key=score)
    return df.loc[:, ordered_columns]


def _get_period_hours(rep_cfg: dict) -> float | None:
    """Return the representative/extreme period length in hours from plotting config."""
    if not rep_cfg or not rep_cfg.get("enable", False):
        return None

    period_length = rep_cfg.get("period_length")
    if period_length is None:
        return None
    days = float(period_length)
    if days <= 0:
        return None
    return days * 24.0


def _get_representative_period_snapshot_groups(
    snapshots: pd.Index,
    rep_cfg: dict,
    representative_metadata: dict | None = None,
) -> list[dict[str, object]]:
    """Group optimization snapshots into representative-period blocks."""
    if not isinstance(snapshots, pd.MultiIndex):
        return []

    names = snapshots.names or []
    period_level = "period" if "period" in names else 0
    timestep_level = "timestep" if "timestep" in names else 1
    grouped_by_period = {}
    max_groups = 0

    for period in snapshots.get_level_values(period_level).unique():
        period_snapshots = snapshots[snapshots.get_level_values(period_level) == period]
        year_meta = (representative_metadata or {}).get(str(period), {})
        period_entries = year_meta.get("periods", []) if isinstance(year_meta, dict) else []

        if period_entries:
            offset = 0
            period_groups = []
            for entry in period_entries:
                steps = int(entry.get("steps", 0))
                if steps <= 0:
                    continue
                group = period_snapshots[offset : offset + steps]
                if len(group) != steps:
                    break
                period_groups.append(group)
                offset += steps
        else:
            period_hours = _get_period_hours(rep_cfg)
            if period_hours is None:
                continue

            timesteps = pd.DatetimeIndex(period_snapshots.get_level_values(timestep_level))
            if len(timesteps) < 2:
                continue

            timestep_hours = (
                timesteps.to_series(index=period_snapshots)
                .diff()
                .dt.total_seconds()
                .div(3600)
                .dropna()
            )
            timestep_hours = timestep_hours[timestep_hours > 0]
            if timestep_hours.empty:
                continue

            steps_per_group = max(1, int(round(period_hours / float(timestep_hours.iloc[0]))))
            period_groups = [
                period_snapshots[i : i + steps_per_group]
                for i in range(0, len(period_snapshots), steps_per_group)
                if len(period_snapshots[i : i + steps_per_group]) == steps_per_group
            ]

        if not period_groups:
            continue

        grouped_by_period[period] = period_groups
        max_groups = max(max_groups, len(period_groups))

    groups = []
    for idx in range(max_groups):
        snapshots_by_period = {
            period: period_groups[idx]
            for period, period_groups in grouped_by_period.items()
            if idx < len(period_groups)
        }
        if snapshots_by_period:
            groups.append(
                {
                    "suffix": f"-rp{idx + 1:02d}",
                    "label": f"Representative Period {idx + 1}",
                    "group_index": idx,
                    "snapshots_by_period": snapshots_by_period,
                },
            )
    return groups


def _get_representative_period_plot_metadata(n: pypsa.Network) -> dict:
    """Read representative-period source date metadata from the solved network."""
    return representative_period_metadata(n)


def _format_representative_period_source_range(period_entry: dict | None) -> str | None:
    """Format serialized representative-period source dates for subplot titles."""
    if not period_entry:
        return None

    start = period_entry.get("start")
    end = period_entry.get("end")
    if not start or not end:
        return None

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return f"{start_ts:%m-%d} to {end_ts:%m-%d}"


def _format_representative_period_weather_years(period_entry: dict | None) -> str | None:
    """
    Format the real weather year(s) a representative period was drawn from.

    ``weather_years`` is written by ``select_representative_periods``; the
    ``start`` / ``end`` fallback covers metadata written before that field
    existed. A block that wraps the end of the weather timeline touches two
    years and is labelled ``"<first>/<second>"``.
    """
    if not period_entry:
        return None

    weather_years = period_entry.get("weather_years")
    if not weather_years:
        weather_years = sorted(
            {
                pd.Timestamp(timestamp).year
                for timestamp in (period_entry.get("start"), period_entry.get("end"))
                if timestamp
            },
        )
    if not weather_years:
        return None
    return "/".join(str(int(year)) for year in weather_years)


def _format_representative_period_title(period_entry: dict | None, fallback: str) -> str:
    """
    Title a representative-period subplot with its real weather year.

    Representative snapshots carry synthetic contiguous labels, so the planning
    horizon says nothing about which weather the block came from -- name the
    weather year instead, and fall back to ``fallback`` (the investment period)
    only when the metadata carries no source dates at all.
    """
    weather_years = _format_representative_period_weather_years(period_entry)
    source_range = _format_representative_period_source_range(period_entry)
    if weather_years and source_range:
        return f"Weather year {weather_years} ({source_range})"
    if weather_years:
        return f"Weather year {weather_years}"
    return fallback


def plot_production_area(
    n: pypsa.Network,
    carriers_2_plot: list[str],
    save: str,
    **wildcards,
) -> None:
    """
    Plot timeseries production.

    Will plot an image for the entire time horizon, in addition to
    seperate monthly generation curves
    """
    # get data
    energy_mix = get_energy_timeseries(n).mul(1e-3)  # MW -> GW
    demand = get_demand_timeseries(n).mul(1e-3)  # MW -> GW

    energy_mix = aggregate_carrier_columns(
        energy_mix,
        n.carriers,
        collapse_storage_directions=True,
    )

    special_link_power = _get_special_link_power_timeseries(n).mul(1e-3)  # MW -> GW
    if not special_link_power.empty:
        energy_mix = energy_mix.drop(columns=[col for col in ["tes", "electrolysis"] if col in energy_mix.columns], errors="ignore")
        energy_mix = pd.concat([energy_mix, special_link_power], axis=1)

    carriers_2_plot = [
        get_visualization_label(carrier, n.carriers, collapse_storage_directions=True)
        for carrier in carriers_2_plot
    ]
    carriers_2_plot.extend([col for col in special_link_power.columns if col in energy_mix.columns])
    carriers_2_plot = list(dict.fromkeys(carriers_2_plot))
    energy_mix = energy_mix[[x for x in carriers_2_plot if x in energy_mix]]
    custom_display_names = {
        "TES_charge": "TES Charge",
        "TES_discharge": "TES Discharge",
        "flexible_electrolysis": "Flexible Electrolysis",
    }
    energy_mix = energy_mix.rename(columns=custom_display_names)
    energy_mix = energy_mix.T.groupby(level=0, sort=False).sum().T
    energy_mix, area_series_sources = _split_mixed_sign_area_series(energy_mix)
    energy_mix = _sort_area_plot_columns_by_variability(energy_mix)

    color_palette = get_color_palette(
        n,
        collapse_storage_directions=True,
        extra_colors={
            "TES Charge": get_color_palette(n).get("tes", "#8c564b"),
            "TES Discharge": get_color_palette(n).get("tes", "#8c564b"),
            "Flexible Electrolysis": get_color_palette(n).get("Electrolysis", "#4c9f70"),
        },
    )
    if "tes" in n.carriers.index:
        tes_color = n.carriers.at["tes", "color"]
        if pd.notna(tes_color):
            color_palette.update(
                {
                    "TES Charge": tes_color,
                    "TES Discharge": tes_color,
                },
            )
    if "electrolysis" in n.carriers.index:
        electrolysis_color = n.carriers.at["electrolysis", "color"]
        if pd.notna(electrolysis_color):
            color_palette["Flexible Electrolysis"] = electrolysis_color
    fallback_colors = {
        "TES Charge": "#8c564b",
        "TES Discharge": "#8c564b",
        "Flexible Electrolysis": "#4c9f70",
    }
    for label, source_label in area_series_sources.items():
        if label not in color_palette and source_label in color_palette:
            color_palette[label] = color_palette[source_label]
    missing_colors = [label for label in energy_mix.columns if label not in color_palette]
    for label in missing_colors:
        color_palette[label] = fallback_colors.get(label, "#000000")
    if missing_colors:
        logger.warning(
            "Missing production plot colors for carriers: %s. Using fallback colors.",
            ", ".join(missing_colors),
        )
    representative_periods_cfg = (
        getattr(n, "meta", {})
        .get("clustering", {})
        .get("temporal", {})
        .get("representative_periods", {})
    )
    representative_metadata = _get_representative_period_plot_metadata(n)
    representative_groups = _get_representative_period_snapshot_groups(
        n.snapshots,
        representative_periods_cfg,
        representative_metadata,
    )
    months = n.snapshots.get_level_values(1).month.unique()
    num_periods = len(n.investment_periods)
    base_plot_size = 4
    plot_groups = [
        {
            "suffix": "",
            "label": None,
            "group_index": None,
            "snapshots_by_period": {
                investment_period: n.snapshots[n.snapshots.get_level_values(0) == investment_period]
                for investment_period in n.investment_periods
            },
        },
    ]
    if representative_groups:
        plot_groups.extend(representative_groups)
    else:
        plot_groups.extend(
            [
                {
                    "suffix": "-" + datetime.strptime(str(month), "%m").strftime("%b"),
                    "label": datetime.strptime(str(month), "%m").strftime("%b"),
                    "group_index": None,
                    "snapshots_by_period": {
                        investment_period: n.snapshots[
                            (n.snapshots.get_level_values(0) == investment_period)
                            & (n.snapshots.get_level_values(1).month == month)
                        ]
                        for investment_period in n.investment_periods
                    },
                }
                for month in months.to_list()
            ],
        )

    for plot_group in plot_groups:
        figsize = (14, (base_plot_size * num_periods))
        fig, axs = plt.subplots(figsize=figsize, ncols=1, nrows=num_periods)
        if not isinstance(axs, np.ndarray):  # only one horizon
            axs = np.array([axs])
        visible_axes = []
        for i, investment_period in enumerate(n.investment_periods):
            sns = plot_group["snapshots_by_period"].get(investment_period, n.snapshots[:0])
            if len(sns) == 0:
                axs[i].set_visible(False)
                continue

            visible_axes.append(axs[i])
            energy_mix.loc[sns].droplevel("period").round(2).plot.area(
                ax=axs[i],
                alpha=0.7,
                color=color_palette,
                linewidth=0,
            )
            demand.loc[sns].droplevel("period").round(2).plot.line(
                ax=axs[i],
                ls="-",
                color="darkblue",
            )

            # Remove auto-generated legend from each subplot
            if axs[i].get_legend():
                axs[i].get_legend().remove()
            title = f"{investment_period}"
            if plot_group["group_index"] is not None:
                year_meta = representative_metadata.get(str(investment_period), {})
                periods = year_meta.get("periods", []) if isinstance(year_meta, dict) else []
                if plot_group["group_index"] < len(periods):
                    title = _format_representative_period_title(
                        periods[plot_group["group_index"]],
                        title,
                    )
            axs[i].set_title(title)
            axs[i].set_ylabel("Power [GW]")
            axs[i].set_xlabel("")

        if not visible_axes:
            plt.close(fig)
            continue

        # Create single shared legend outside the plot area (centered vertically)
        handles, labels = visible_axes[0].get_legend_handles_labels()
        fig.tight_layout(rect=[0, 0, 0.78, 0.95])  # Leave space on right for legend
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.78, 0.5), frameon=False)
        title = "Production [GW]"
        if plot_group["label"]:
            title = f"{title} - {plot_group['label']}"
        fig.suptitle(create_title(title, **wildcards))
        save = Path(save)
        fig.savefig(save.parent / (save.stem + plot_group["suffix"] + save.suffix))
        plt.close()


def plot_hourly_emissions(n: pypsa.Network, save: str, **wildcards) -> None:
    """Plots snapshot emissions by technology."""
    # get data
    emissions = get_tech_emissions_timeseries(n).mul(1e-6)  # T -> MT
    zeros = emissions.columns[(np.abs(emissions) < 1e-7).all()]
    emissions = emissions.drop(columns=zeros)
    emissions = aggregate_carrier_columns(emissions, n.carriers)

    # plot
    color_palette = get_color_palette(n)

    fig, ax = plt.subplots(figsize=(14, 4))
    if not emissions.empty:
        emissions.plot.area(
            ax=ax,
            alpha=0.7,
            legend="reverse",
            color=color_palette,
        )

    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_title(create_title("Technology Emissions", **wildcards))
    ax.set_ylabel("Emissions [MT]")
    fig.tight_layout()

    fig.savefig(save)
    plt.close()


def plot_accumulated_emissions_tech(n: pypsa.Network, save: str, **wildcards) -> None:
    """Creates area plot of accumulated emissions by technology."""
    # get data

    emissions = get_tech_emissions_timeseries(n).cumsum().mul(1e-6)  # T -> MT
    zeros = emissions.columns[(np.abs(emissions) < 1e-7).all()]
    emissions = emissions.drop(columns=zeros)
    emissions = aggregate_carrier_columns(emissions, n.carriers)

    # plot

    color_palette = get_color_palette(n)

    fig, ax = plt.subplots(figsize=(14, 4))
    if not emissions.empty:
        emissions.plot.area(
            ax=ax,
            alpha=0.7,
            legend="reverse",
            color=color_palette,
        )

    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_title(create_title("Technology Accumulated Emissions", **wildcards))
    ax.set_ylabel("Emissions [MT]")
    fig.tight_layout()

    fig.savefig(save)
    plt.close()


def plot_accumulated_emissions(n: pypsa.Network, save: str, **wildcards) -> None:
    """Plots accumulated emissions."""
    # get data

    emissions = get_tech_emissions_timeseries(n).mul(1e-6).sum(axis=1)  # T -> MT
    emissions = emissions.cumsum().to_frame("co2")

    # plot

    color_palette = get_color_palette(n)

    fig, ax = plt.subplots(figsize=(14, 4))

    if not emissions.empty:
        emissions.plot.area(
            ax=ax,
            alpha=0.7,
            legend="reverse",
            color=color_palette,
            stacked=False,
        )

    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_title(create_title("Accumulated Emissions", **wildcards))
    ax.set_ylabel("Emissions [MT]")
    fig.tight_layout()
    fig.savefig(save)
    plt.close()


def plot_capacity_factor_heatmap(n: pypsa.Network, save: str, **wildcards) -> None:
    """HEATMAP OF RENEWABLE CAPACITY FACTORS BY CARRIER."""
    df_long = n.generators_t.p.loc[n.investment_periods[0]].melt(
        var_name="bus",
        value_name="p",
        ignore_index=False,
    )
    df_long["region"] = df_long["bus"].map(n.generators.bus.map(n.buses.country))
    df_long["carrier"] = df_long["bus"].map(n.generators.carrier)
    df_long["hour"] = df_long.index.hour
    df_long["month"] = df_long.index.month
    df_long = df_long.drop(columns="bus")
    df_long = df_long.drop(columns="region").groupby(["carrier", "month", "hour"]).mean().reset_index()

    # Get unique months for separate panels
    unique_months = df_long["month"].unique()

    # Prepare figure and axes
    _, axs = plt.subplots(
        len(unique_months),
        1,
        figsize=(12, len(unique_months) * 4),
        sharex=True,
    )

    # Iterate over each month to create a panel
    for idx, month in enumerate(sorted(unique_months)):
        month_data = df_long[df_long["month"] == month]
        pivot_data = month_data.pivot(index="hour", columns="carrier", values="p")

        ax = axs[idx] if len(unique_months) > 1 else axs
        pivot_data.plot.area(ax=ax, title=f"Month: {month}", alpha=0.7)
        ax.set_ylabel("Mean Power (p)")
        ax.set_xlabel("Hour of the Day")
        ax.legend(title="Carrier", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.suptitle("Heatmap of Renewable Capacity Factors by by Carrier")
    plt.tight_layout()

    plt.savefig(save)
    plt.close()


#### Panel / Mixed Plots ####


def plot_generator_data_panel(
    n: pypsa.Network,
    save: str,
    **wildcards,
):
    df_capex_expand = n.generators.loc[
        n.generators.p_nom_extendable & ~n.generators.index.str.contains("existing"),
        :,
    ]

    df_storage_units = n.storage_units.loc[n.storage_units.p_nom_extendable, :].copy()
    df_storage_units.loc[:, "efficiency"] = df_storage_units.efficiency_dispatch
    df_capex_expand = pd.concat([df_capex_expand, df_storage_units])

    df_efficiency = n.generators.loc[
        ~n.generators.carrier.isin(
            ["solar", "onwind", "offwind", "offwind_floating", "hydro", "load"],
        ),
        :,
    ]
    # Create a figure and subplots with 2 rows and 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    # Plot on each subplot
    sns.lineplot(
        data=get_generator_marginal_costs(n),
        x="timestep",
        y="Value",
        hue="Carrier",
        ax=axes[0, 0],
    )
    sns.barplot(data=df_capex_expand, x="carrier", y="capital_cost", ax=axes[0, 1])
    sns.boxplot(data=df_efficiency, x="carrier", y="efficiency", ax=axes[1, 0])

    # Create line plot of declining capital costs
    sns.lineplot(
        data=df_capex_expand[df_capex_expand.build_year > 0],
        x="build_year",
        y="capital_cost",
        hue="carrier",
        ax=axes[2, 0],
    )

    cf_profiles = n.get_switchable_as_dense("Generator", "p_max_pu")
    fuel_costs = n.generators.marginal_cost * cf_profiles.sum()
    n.generators["lcoe"] = (n.generators.capital_cost + fuel_costs) / cf_profiles.sum()
    n.generators["cf"] = cf_profiles.mean()
    lcoe_plot_df = n.generators.loc[
        n.generators.p_nom_extendable & ~n.generators.index.str.contains("existing"),
        :,
    ]

    sns.boxplot(
        data=n.generators,
        x="cf",
        y="carrier",
        ax=axes[1, 1],
    )

    sns.boxplot(
        data=lcoe_plot_df,
        x="lcoe",
        y="carrier",
        ax=axes[2, 1],
    )

    # Set titles for each subplot
    axes[0, 0].set_title("Generator Marginal Costs")
    axes[0, 1].set_title("Extendable Capital Costs")
    axes[1, 0].set_title("Plant Efficiency")
    axes[1, 1].set_title("Capacity Factors by Carrier")
    axes[2, 0].set_title("Expansion Capital Costs by Carrier")
    axes[2, 1].set_title("LCOE by Carrier")

    # Set labels for each subplot
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("$ / MWh")
    # axes[0, 0].set_ylim(0, 200)
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("$ / MW-yr")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("MWh_primary / MWh_elec")
    axes[1, 1].set_xlabel("p.u.")
    axes[1, 1].set_ylabel("")
    axes[2, 0].set_xlabel("Year")
    axes[2, 0].set_ylabel("$ / MW-yr")
    axes[2, 1].set_xlabel("$ / MWh")
    axes[2, 1].set_ylabel("")

    # Rotate x-axis labels for each subplot
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=25)

    # Lay legend out horizontally
    axes[0, 0].legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        ncol=1,
        fontsize="xx-small",
    )
    axes[2, 0].legend(fontsize="xx-small")

    fig.tight_layout()
    fig.savefig(save)
    plt.close()


def plot_region_lmps(
    n: pypsa.Network,
    save: str,
    **wildcards,
) -> None:
    """Plots a box plot of LMPs for each region."""
    df_lmp = n.buses_t.marginal_price
    df_long = pd.melt(
        df_lmp.reset_index(),
        id_vars=["timestep"],
        var_name="bus",
        value_name="lmp",
    )
    df_long["season"] = df_long["timestep"].dt.quarter
    df_long["hour"] = df_long["timestep"].dt.hour
    df_long = df_long.drop(columns="timestep")
    df_long["region"] = df_long.bus.map(n.buses.country)

    plt.figure(figsize=(10, 10))

    sns.boxplot(
        df_long,
        x="lmp",
        y="region",
        width=0.5,
        fliersize=0.5,
        linewidth=1,
    )

    plt.title(create_title("LMPs by Region", **wildcards))
    plt.xlabel("LMP [$/MWh]")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(save)
    plt.close()


# Fuel costs


def plot_fuel_costs(
    n: pypsa.Network,
    save: str,
    **wildcards,
) -> None:
    fuel_costs = get_fuel_costs(n)

    fuels = set(fuel_costs.index.get_level_values("carrier"))

    fig, axs = plt.subplots(len(fuels) + 1, 1, figsize=(20, 40))

    color_palette = n.carriers.color.to_dict()

    # plot error plot of all fuels
    df = fuel_costs.droplevel(["bus", "Generator"]).T.resample("d").mean().reset_index().melt(id_vars="timestep")
    sns.lineplot(
        data=df,
        x="timestep",
        y="value",
        hue="carrier",
        ax=axs[0],
        legend=True,
        palette=color_palette,
    )
    (axs[0].set_title("Daily Average Fuel Costs [$/MWh]"),)
    (axs[0].set_xlabel(""),)
    (axs[0].set_ylabel("$/MWh"),)

    # plot bus fuel prices for each fuel
    for i, fuel in enumerate(fuels):
        nice_name = n.carriers.at[fuel, "nice_name"]
        df = fuel_costs.loc[fuel, :, :].droplevel("Generator").T.resample("d").mean().T.groupby(level=0).mean().T
        sns.lineplot(
            data=df,
            legend=False,
            palette="muted",
            dashes=False,
            ax=axs[i + 1],
        )
        (axs[i + 1].set_title(f"Daily Average {nice_name} Fuel Costs per Bus [$/MWh]"),)
        (axs[i + 1].set_xlabel(""),)
        (axs[i + 1].set_ylabel("$/MWh"),)

    fig.savefig(save)
    plt.close()


#### Climate Analysis Plots ####
def plot_renewable_capacity_factors(
    n: pypsa.Network,
    save: str,
    **wildcards,
) -> None:
    """Multi-panel capacity factor analysis for renewable technologies."""
    # Get renewable carriers
    renewable_carriers = ["solar", "onwind", "offwind", "offwind_floating"]
    renewable_gens = n.generators[n.generators.carrier.isin(renewable_carriers)]

    if renewable_gens.empty:
        logger.warning("No renewable generators found for capacity factor plot")
        return

    # Filter to carriers that actually exist
    carriers_present = [c for c in renewable_carriers if c in renewable_gens.carrier.values]

    if not carriers_present:
        logger.warning("No renewable carriers found in generators")
        return

    num_carriers = len(carriers_present)
    num_periods = len(n.investment_periods)
    representative_periods_enabled = bool(
        (globals().get("snakemake").config if globals().get("snakemake") else {})
        .get("clustering", {})
        .get("temporal", {})
        .get("representative_periods", {})
        .get("enable", False),
    )
    month_index = pd.Index(range(1, 13), name="month")
    period_month_counts = {
        period: n.snapshots[n.snapshots.get_level_values(0) == period].get_level_values(1).month.nunique()
        for period in n.investment_periods
    }
    monthly_plots_supported = (
        not representative_periods_enabled
        and all(month_count == len(month_index) for month_count in period_month_counts.values())
    )

    # Create figure: rows = carriers, cols = [Duration Curve, Monthly Pattern]
    fig, axs = plt.subplots(
        nrows=num_carriers,
        ncols=2,
        figsize=(14, 4 * num_carriers),
        squeeze=False,
    )

    # Line styles for different periods
    line_styles = ["-", "--", ":", "-."]

    for row, carrier in enumerate(carriers_present):
        carrier_gens = renewable_gens[renewable_gens.carrier == carrier].index
        nice_name = n.carriers.at[carrier, "nice_name"]
        carrier_color = n.carriers.at[carrier, "color"]  # Use carrier's characteristic color

        ax_duration = axs[row, 0]
        ax_monthly = axs[row, 1]

        monthly_data = []

        for period_idx, period in enumerate(n.investment_periods):
            period_sns = n.snapshots[n.snapshots.get_level_values(0) == period]
            p_max_pu = n.generators_t.p_max_pu.loc[period_sns]

            # Filter to generators that exist
            valid_gens = [g for g in carrier_gens if g in p_max_pu.columns]
            if not valid_gens:
                continue

            # Average capacity factor across all generators of this carrier
            cf_series = p_max_pu[valid_gens].mean(axis=1)

            # Duration curve, sort values descending
            cf_sorted = cf_series.sort_values(ascending=False).reset_index(drop=True)
            cf_sorted.index = cf_sorted.index / len(cf_sorted) * 100

            ax_duration.plot(
                cf_sorted.index,
                cf_sorted.values,
                color=carrier_color,
                linestyle=line_styles[period_idx % len(line_styles)],
                label=str(period),
                linewidth=2.5 - (period_idx * 0.3),
                alpha=0.9 - (period_idx * 0.1),
            )

            if monthly_plots_supported:
                # Monthly averages only make sense when all months are represented.
                cf_monthly = (
                    cf_series.groupby(cf_series.index.get_level_values(1).month).mean().reindex(month_index)
                )
                monthly_data.append(
                    {
                        "period": period,
                        "monthly_cf": cf_monthly,
                        "color": carrier_color,
                    },
                )

        # Format duration curve plot
        ax_duration.set_xlim(0, 100)
        ax_duration.set_ylim(0, 1)
        ax_duration.set_xlabel("% of Time")
        ax_duration.set_ylabel("Capacity Factor")
        ax_duration.set_title(f"{nice_name} - Duration Curve", fontweight="bold")
        ax_duration.legend(title="Period", loc="upper right")
        ax_duration.grid(True, alpha=0.3)
        ax_duration.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

        # Format monthly plot
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        if not monthly_plots_supported:
            ax_monthly.text(
                0.5,
                0.5,
                "Monthly plot skipped\n(representative periods)",
                ha="center",
                va="center",
                transform=ax_monthly.transAxes,
                fontsize=10,
            )
            ax_monthly.set_title(f"{nice_name} - Monthly Pattern", fontweight="bold")
            ax_monthly.set_xticks([])
            ax_monthly.set_yticks([])
            ax_monthly.grid(False)
        else:
            x_positions = np.arange(12)
            bar_width = 0.8 / num_periods
            hatch_patterns = ["", "//", "\\\\", "xx", ".."]  # Different patterns for periods

            for i, data in enumerate(monthly_data):
                offset = (i - num_periods / 2 + 0.5) * bar_width
                ax_monthly.bar(
                    x_positions + offset,
                    data["monthly_cf"].values,
                    bar_width,
                    color=data["color"],
                    label=str(data["period"]),
                    alpha=0.9 - (i * 0.15),  # Slightly lighter for later periods
                    edgecolor="black",
                    linewidth=0.5,
                    hatch=hatch_patterns[i % len(hatch_patterns)],
                )

            ax_monthly.set_xticks(x_positions)
            ax_monthly.set_xticklabels(month_names, rotation=45, ha="right")
            ax_monthly.set_ylim(0, 1)
            ax_monthly.set_ylabel("Avg Capacity Factor")
            ax_monthly.set_title(f"{nice_name} - Monthly Pattern", fontweight="bold")
            ax_monthly.legend(title="Period", loc="upper right")
            ax_monthly.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        create_title("Renewable Capacity Factor Analysis", **wildcards),
        fontsize=TITLE_SIZE,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save, bbox_inches="tight")
    plt.close()


def plot_seasonal_generation(
    n: pypsa.Network,
    save: str,
    **wildcards,
) -> None:
    """Multi-panel generation analysis showing total monthly energy by technology."""
    from plot_network_maps import get_color_palette

    # Get energy timeseries (power in GW)
    energy_mix = get_energy_timeseries(n).mul(1e-3)  # convert MW to GW
    energy_mix = aggregate_carrier_columns(
        energy_mix,
        n.carriers,
        collapse_storage_directions=True,
    )
    energy_positive = energy_mix.clip(lower=0)

    # Get snapshot weighting (hours per snapshot) for energy calculation
    # Use the first weighting column (typically 'generators' or 'objective')
    hours_per_snapshot = snapshot_weight_series(n, preferred="generators")

    # Get top technologies by total generation
    total_gen = energy_positive.sum()
    top_techs = total_gen[total_gen > total_gen.sum() * 0.02].sort_values(ascending=False).index.tolist()

    if not top_techs:
        logger.warning("No significant generation technologies found")
        return

    num_techs = min(len(top_techs), 8)  # Limit to top 8 technologies
    top_techs = top_techs[:num_techs]
    num_periods = len(n.investment_periods)
    month_index = pd.Index(range(1, 13), name="month")
    period_month_counts = {
        period: n.snapshots[n.snapshots.get_level_values(0) == period].get_level_values(1).month.nunique()
        for period in n.investment_periods
    }
    monthly_plots_supported = all(month_count == len(month_index) for month_count in period_month_counts.values())

    color_palette = get_color_palette(n, collapse_storage_directions=True)

    # Create figure: rows = technologies, cols = [Monthly Energy, Period Change]
    fig, axs = plt.subplots(
        nrows=num_techs,
        ncols=2,
        figsize=(14, 3 * num_techs),
        squeeze=False,
    )

    # Line styles for different periods (use tech color, vary line style)
    line_styles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    for row, tech in enumerate(top_techs):
        ax_monthly = axs[row, 0]
        ax_change = axs[row, 1]

        tech_color = color_palette.get(tech, "gray")
        monthly_by_period = {}
        baseline_monthly = None

        if not monthly_plots_supported:
            ax_monthly.text(
                0.5,
                0.5,
                "Monthly plot skipped\n(representative periods)",
                ha="center",
                va="center",
                transform=ax_monthly.transAxes,
                fontsize=10,
            )
            ax_monthly.set_title(f"{tech}", fontsize=11, fontweight="bold")
            ax_monthly.set_xticks([])
            ax_monthly.set_yticks([])
            ax_monthly.grid(False)

            ax_change.text(
                0.5,
                0.5,
                "Period change skipped\n(representative periods)",
                ha="center",
                va="center",
                transform=ax_change.transAxes,
                fontsize=10,
            )
            ax_change.set_title(f"Change from {n.investment_periods[0]}", fontsize=10)
            ax_change.set_xticks([])
            ax_change.set_yticks([])
            ax_change.grid(False)
            continue

        for period_idx, period in enumerate(n.investment_periods):
            period_sns = n.snapshots[n.snapshots.get_level_values(0) == period]

            if tech not in energy_positive.columns:
                continue

            gen_series = energy_positive.loc[period_sns, tech]

            # Calculate total monthly energy: sum of power * hours, converted to TWh
            monthly_energy = (
                gen_series.mul(hours_per_snapshot.reindex(period_sns))
                .groupby(gen_series.index.get_level_values(1).month)
                .sum()
                / 1000  # convert to TWh
            )
            monthly_energy = monthly_energy.reindex(month_index, fill_value=0.0)
            monthly_by_period[period] = monthly_energy

            if baseline_monthly is None:
                baseline_monthly = monthly_energy

            # Plot monthly energy - use tech color with different line styles per period
            ax_monthly.plot(
                range(1, 13),
                monthly_energy.values,
                color=tech_color,
                linestyle=line_styles[period_idx % len(line_styles)],
                label=str(period),
                linewidth=2.5 - (period_idx * 0.3),
                marker=markers[period_idx % len(markers)],
                markersize=6,
                alpha=0.9 - (period_idx * 0.1),
            )

        # Format monthly plot
        ax_monthly.set_xticks(range(1, 13))
        ax_monthly.set_xticklabels(month_names, rotation=45, ha="right", fontsize=8)
        ax_monthly.set_ylabel("Energy [TWh]")
        ax_monthly.set_title(f"{tech}", fontsize=11, fontweight="bold")
        ax_monthly.legend(title="Period", loc="upper right", fontsize=8)
        ax_monthly.grid(True, alpha=0.3)
        ax_monthly.set_xlim(0.5, 12.5)

        # Plot period-to-period changes (% change from baseline)
        if baseline_monthly is not None and len(monthly_by_period) > 1:
            x_positions = np.arange(12)
            bar_width = 0.8 / (num_periods - 1) if num_periods > 1 else 0.8

            for period_idx, (period, monthly_energy) in enumerate(monthly_by_period.items()):
                if period == n.investment_periods[0]:
                    continue  # Skip baseline

                pct_change = ((monthly_energy - baseline_monthly) / baseline_monthly.replace(0, np.nan) * 100).fillna(0)

                offset = (period_idx - 1 - (num_periods - 2) / 2) * bar_width
                colors = ["#1a7f37" if v >= 0 else "#a3200d" for v in pct_change.values]

                ax_change.bar(
                    x_positions + offset,
                    pct_change.values,
                    bar_width,
                    color=colors,
                    alpha=0.7,
                    edgecolor="white",
                    linewidth=0.5,
                    label=f"Δ {n.investment_periods[0]}→{period}",
                )

            ax_change.axhline(y=0, color="black", linewidth=0.8)
            ax_change.set_xticks(x_positions)
            ax_change.set_xticklabels(month_names, rotation=45, ha="right", fontsize=8)
            ax_change.set_ylabel("% Change")
            ax_change.set_title(f"Change from {n.investment_periods[0]}", fontsize=10)
            ax_change.legend(loc="upper right", fontsize=8)
            ax_change.grid(True, axis="y", alpha=0.3)
        else:
            ax_change.text(
                0.5,
                0.5,
                "Single period\n(no comparison)",
                ha="center",
                va="center",
                transform=ax_change.transAxes,
                fontsize=10,
            )
            ax_change.set_xticks([])
            ax_change.set_yticks([])

    fig.suptitle(
        create_title("Monthly Energy Production by Technology [TWh]", **wildcards),
        fontsize=TITLE_SIZE,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_statistics",
            case="2050_MidDmd_v1.30_NoDCNet_SSSC_InfGVA",
            ll="v1.30",
            opts="RPS-ERM-3h",
            sector="E",
        )
    configure_logging(snakemake)
    set_case_config(snakemake)

    # extract shared plotting files
    n = load_postprocess_network(snakemake.input.network)
    onshore_regions = gpd.read_file(snakemake.input.regions_onshore)
    retirement_method = snakemake.params.retirement

    normalize_line_x_statistics_columns(n)
    sanitize_carriers(n, snakemake.config)

    # mappers
    generating_link_carrier_map = {"fuel cell": "H2", "battery discharger": "battery"}

    # carriers to plot
    carriers = (
        snakemake.params.electricity["conventional_carriers"]
        + snakemake.params.electricity["renewable_carriers"]
        + snakemake.params.electricity["extendable_carriers"]["Generator"]
        + snakemake.params.electricity["extendable_carriers"]["StorageUnit"]
        + snakemake.params.electricity["extendable_carriers"]["Store"]
        + snakemake.params.electricity["extendable_carriers"]["Link"]
        + ["battery_charger", "battery_discharger"]
    )
    irb_option = snakemake.params.electricity.get("iron_air_battery", False)
    irb_enabled = irb_option.get("enable", False) if isinstance(irb_option, dict) else bool(irb_option)
    if irb_enabled:
        carriers.append("irb")

    if not n.storage_units.empty and "carrier" in n.storage_units.columns:
        carriers.extend(n.storage_units.carrier.dropna().unique().tolist())

    carriers = list(set(carriers))  # remove any duplicates

    # Export Statistics Tables
    groupers = n.statistics.groupers
    n.statistics(groupby=groupers.get_name_bus_and_carrier).round(3).to_csv(
        snakemake.output.statistics_dissaggregated,
    )
    build_statistics_summary_table(n).round(4).to_csv(snakemake.output.statistics_summary)
    get_carrier_cost_breakdown(n).round(4).to_csv(snakemake.output.cost_breakdown)
    n.generators.to_csv(snakemake.output.generators)
    n.storage_units.to_csv(snakemake.output.storage_units)
    n.links.to_csv(snakemake.output.links)
    get_export_lines_table(n).to_csv(snakemake.output.lines)
    n.buses.to_csv(snakemake.output.buses)
    n.stores.to_csv(snakemake.output.stores)
    n.global_constraints.to_csv(snakemake.output.global_constraints)

    # Panel Plots
    plot_generator_data_panel(
        n,
        snakemake.output["generator_data_panel.pdf"],
        **snakemake.wildcards,
    )

    # Bar Plots
    plot_capacity_additions_bar(
        n,
        carriers,
        snakemake.output["capacity_additions_bar.pdf"],
        **snakemake.wildcards,
    )
    plot_production_bar(
        n,
        carriers,
        snakemake.output["production_bar.pdf"],
        **snakemake.wildcards,
    )
    plot_global_constraint_shadow_prices(
        n,
        snakemake.output["global_constraint_shadow_prices.pdf"],
        **snakemake.wildcards,
    )
    plot_component_cost_breakdown(
        n,
        snakemake.output["cost_breakdown_bar.pdf"],
        **snakemake.wildcards,
    )
    plot_regional_capacity_additions_bar(
        n,
        snakemake.output["bar_regional_capacity_additions.pdf"],
    )
    plot_regional_production_bar(
        n,
        snakemake.output["bar_regional_production.pdf"],
    )
    plot_regional_emissions_bar(
        n,
        snakemake.output["bar_regional_emissions.pdf"],
    )
    plot_emissions_bar(
        n,
        snakemake.output["bar_emissions.pdf"],
    )

    # Time Series Plots
    plot_production_area(
        n,
        carriers,
        snakemake.output["production_area.pdf"],
        **snakemake.wildcards,
    )
    plot_hourly_emissions(
        n,
        snakemake.output["emissions_area.pdf"],
        **snakemake.wildcards,
    )
    plot_accumulated_emissions_tech(
        n,
        snakemake.output["emissions_accumulated_tech.pdf"],
        **snakemake.wildcards,
    )
    plot_accumulated_emissions(
        n,
        snakemake.output["emissions_accumulated.pdf"],
        **snakemake.wildcards,
    )
    plot_fuel_costs(
        n,
        snakemake.output["fuel_costs.pdf"],
        **snakemake.wildcards,
    )

    # Box Plot
    plot_region_lmps(
        n,
        snakemake.output["region_lmps.pdf"],
        **snakemake.wildcards,
    )

    # Renewable Capacity Factor and Seasonal Generation Plots
    plot_renewable_capacity_factors(
        n,
        snakemake.output["renewable_capacity_factors.pdf"],
        **snakemake.wildcards,
    )
    plot_seasonal_generation(
        n,
        snakemake.output["seasonal_generation.pdf"],
        **snakemake.wildcards,
    )
