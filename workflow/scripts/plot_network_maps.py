"""
Plots static and interactive charts to analyze system results.

**Inputs**

A solved network

**Outputs**

Capacity maps for:
    - Base capacity
    - New capacity
    - Optimal capacity (does not show existing unused capacity)
    - Optimal browfield capacity

    .. image:: _static/plots/capacity-map.png
        :scale: 33 %
"""

import logging
from decimal import Decimal, ROUND_HALF_UP

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from postprocess_io import load_postprocess_network
import seaborn as sns
from _helpers import configure_logging, set_case_config
from add_electricity import sanitize_carriers
from cartopy import crs as ccrs
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyBboxPatch, Patch
from summary import get_demand_base
from visualization_carriers import (
    build_visualization_carriers,
    build_visualization_palette,
    get_visualization_label,
)

logger = logging.getLogger(__name__)

DEFAULT_EPSG = 4326

# Global Plotting Settings
PLOT_FONT_FAMILY = "Times New Roman"
PLOT_FONT_RC_PARAMS = {
    "font.family": PLOT_FONT_FAMILY,
    "mathtext.fontset": "custom",
    "mathtext.rm": PLOT_FONT_FAMILY,
    "mathtext.it": f"{PLOT_FONT_FAMILY}:italic",
    "mathtext.bf": f"{PLOT_FONT_FAMILY}:bold",
}
TITLE_SIZE = 16
MAP_BACKGROUND_COLOR = "#EEEEEE"
MAP_REGION_BOUNDARY_COLOR = "#ffffff"
MAP_REGION_BOUNDARY_ALPHA = 1.0
MAP_REGION_BOUNDARY_WIDTH = 1.0
MAP_REGION_MERGE_TOLERANCE_M = 3_000
MAP_REGION_MIN_AREA_M2 = 2_000_000
MAP_MIN_LON_SHIFT_DEGREES = 2.0
MAP_MAX_LON_SHIFT_DEGREES = 1.0
CAPACITY_MAP_NO_PIE_MAX_LON_SHIFT_DEGREES = 0.5
MAP_MAX_LAT_SHRINK_DEGREES = 0
MAP_FIGSIZE = (12, 12)
MAP_LAYOUT_LEFT_MARGIN = 0.0
MAP_LAYOUT_RIGHT_MARGIN = 1.0
MAP_LAYOUT_BOTTOM_MARGIN = 0.03
MAP_LAYOUT_TOP_MARGIN = 0.95
TECH_LEGEND_ANCHOR_X = 1.0
TECH_LEGEND_ANCHOR_Y = 0.65
TECH_LEGEND_SECTION_GAP = 0.05
BOTTOM_LEGEND_LEFT_X = 0.05
BOTTOM_LEGEND_ANCHOR_Y = 0.28
BOTTOM_LEGEND_GROUP_GAP = 0.025
LEGEND_TITLE_SIZE = 12
LEGEND_FONT_SIZE = 11
NO_PIE_LEGEND_FONT_SIZE_INCREMENT = 5
NO_PIE_TRANSMISSION_LEGEND_Y_OFFSET = 0.05
CAPACITY_LEGEND_LABEL_SPACING_FACTOR = 1.5
CAPACITY_LEGEND_TITLE_GAP = 0.02
CAPACITY_LEGEND_LABEL_GAP = 0.012
CAPACITY_LEGEND_INDICATOR_LINE_EXTRA = 0.015
LEGEND_MONOCHROME_COLOR = "#374151"
LEGEND_TEXT_COLOR = LEGEND_MONOCHROME_COLOR
LEGEND_LINE_COLOR = LEGEND_MONOCHROME_COLOR
CAPACITY_MAP_BUS_ALPHA = 0.6
BOXED_LEGEND_STYLE = {
    "frameon": True,
    "facecolor": "white",
    "edgecolor": "none",
    "framealpha": 0.9,
    "borderaxespad": 0.4,
}
TRANSPARENT_LEGEND_STYLE = {
    "frameon": False,
    "facecolor": "none",
    "edgecolor": "none",
    "framealpha": 0.0,
    "borderaxespad": 0.0,
}
CAPACITY_CIRCLE_LEGEND_VALUES_MW = [10000, 30000, 100000, 250000, 500000]
TRANSMISSION_CAPACITY_LEGEND_VALUES_MW = [1000, 3000, 10000, 30000]
SSSC_MARKER_COLOR = LEGEND_MONOCHROME_COLOR
SSSC_CAPACITY_LEGEND_VALUES_MW = [100, 300, 1000, 3000, 10000]
SSSC_MARKER_AREA_SCALE = 0.1
CAPACITY_MAP_EXCLUDED_CARRIERS = {
    "H2",
    "hydrogen storage",
    "H2 electrolysis",
}
CAPACITY_MAP_EXCLUDED_LINK_CARRIERS = CAPACITY_MAP_EXCLUDED_CARRIERS | {"AC", "DC"}
CAPACITY_MAP_LINK_CARRIER_MAP = {
    "battery discharger": "battery",
}


def get_color_palette(
    n: pypsa.Network,
    collapse_storage_directions: bool = False,
    extra_colors: dict[str, str] | None = None,
) -> dict[str, str]:
    """Returns colors keyed by visualization display label."""
    palette = build_visualization_palette(
        n.carriers,
        collapse_storage_directions=collapse_storage_directions,
        extra_colors=extra_colors,
    )
    palette.setdefault("co2", "k")
    return palette


def get_carrier_color(n: pypsa.Network, carrier: str, fallback: str) -> str:
    if carrier in n.carriers.index and "color" in n.carriers.columns:
        color = n.carriers.at[carrier, "color"]
        if pd.notna(color) and str(color).strip():
            return str(color)
    return fallback


def get_bus_scale(interconnect: str) -> float:
    """Scales lines based on interconnect size."""
    if interconnect != "usa":
        return 1e5
    else:
        return 1e5


def get_line_scale(interconnect: str) -> float:
    """Scales lines based on interconnect size."""
    if interconnect != "usa":
        return 2e3
    else:
        return 3e3


def get_plot_interconnect(
    config: dict | None = None,
    wildcards: dict | None = None,
) -> str | None:
    """Resolve plotting interconnect from case config first, then wildcards."""
    scenario = (config or {}).get("scenario", {})
    interconnect = scenario.get("interconnect")

    if isinstance(interconnect, (list, tuple)):
        interconnect = interconnect[0] if interconnect else None
    if interconnect is not None and str(interconnect).strip():
        return str(interconnect)

    if wildcards is None:
        return None

    if hasattr(wildcards, "get"):
        interconnect = wildcards.get("interconnect", None)
    elif isinstance(wildcards, dict):
        interconnect = wildcards.get("interconnect", None)
    else:
        interconnect = getattr(wildcards, "interconnect", None)

    if isinstance(interconnect, (list, tuple)):
        interconnect = interconnect[0] if interconnect else None
    if interconnect is not None and str(interconnect).strip():
        return str(interconnect)
    return None


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
    return f"{title}\n{wildcards_joined}" if wildcards_joined else title


def get_map_boundaries(
    regions: gpd.GeoDataFrame,
    shrink: bool = True,
) -> tuple[float, float, float, float]:
    """Return plotting bounds in the order expected by ``n.plot(..., boundaries=...)``."""
    bounds = regions.total_bounds.astype(float).copy()
    if shrink:
        bounds[0] = min(bounds[2], bounds[0] + MAP_MIN_LON_SHIFT_DEGREES)
        bounds[2] = max(bounds[0], bounds[2] - MAP_MAX_LON_SHIFT_DEGREES)
        bounds[3] = max(bounds[2], bounds[3] - MAP_MAX_LAT_SHRINK_DEGREES)
    return tuple(bounds[[0, 2, 1, 3]])


def get_capacity_map_boundaries(
    regions: gpd.GeoDataFrame,
    show_capacity_pie: bool = True,
) -> tuple[float, float, float, float]:
    """Return capacity-map bounds, tightening the east edge when pies are hidden."""
    min_lon, max_lon, min_lat, max_lat = get_map_boundaries(regions, shrink=True)
    if not show_capacity_pie:
        max_lon = max(min_lon, max_lon - CAPACITY_MAP_NO_PIE_MAX_LON_SHIFT_DEGREES)
    return min_lon, max_lon, min_lat, max_lat


def remove_sector_buses(df: pd.DataFrame) -> pd.DataFrame:
    """Removes buses for sector coupling."""
    num_levels = df.index.nlevels

    if num_levels > 1:
        condition = (df.index.get_level_values("bus").str.endswith(" gas")) | (
            df.index.get_level_values("bus").str.endswith(" gas storage")
        )
    else:
        condition = (
            (df.index.str.endswith(" gas"))
            | (df.index.str.endswith(" gas storage"))
            | (df.index.str.endswith(" gas import"))
            | (df.index.str.endswith(" gas export"))
        )
    return df.loc[~condition].copy()


def get_model_region_background(regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a cleaned, dissolved model-area polygon for map background fills."""
    background = _clean_model_regions(regions)[["geometry"]].copy()
    background = background.dissolve()
    background = background.explode(index_parts=False).reset_index(drop=True)
    return background


def get_zone_region_boundaries(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Dissolve model regions into ReEDS zone boundaries for an overlay."""
    grouped = regions.copy()
    region_names = (
        grouped["name"].astype(str)
        if "name" in grouped.columns
        else pd.Series(grouped.index.astype(str), index=grouped.index)
    )
    grouped["reeds_zone"] = region_names.map(n.buses.reeds_zone)
    dissolved = grouped.dissolve(by="reeds_zone").reset_index()
    if hasattr(regions, "crs"):
        dissolved.crs = regions.crs
    return _clean_model_regions(dissolved)


def _apply_region_merge_tolerance(
    geometry: gpd.GeoSeries,
    tolerance_m: float,
) -> gpd.GeoSeries:
    """Apply a small close/open buffer in meters to remove slivers and pinholes."""
    cleaned = geometry.buffer(0)
    if tolerance_m <= 0:
        return cleaned
    return cleaned.buffer(tolerance_m).buffer(-tolerance_m).buffer(0)


def _clean_model_regions(regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clean region polygons for map display in a projected CRS."""
    cleaned = regions[["geometry"]].copy()
    cleaned = cleaned.loc[cleaned.geometry.notna()].copy()
    if cleaned.empty:
        return cleaned

    if cleaned.crs is None:
        cleaned["geometry"] = cleaned.geometry.buffer(0)
        return cleaned.explode(index_parts=False).reset_index(drop=True)

    original_crs = cleaned.crs
    cleaned = cleaned.to_crs(epsg=3857)
    cleaned["geometry"] = _apply_region_merge_tolerance(cleaned.geometry, MAP_REGION_MERGE_TOLERANCE_M)
    cleaned = cleaned.explode(index_parts=False).reset_index(drop=True)
    cleaned = cleaned.loc[cleaned.geometry.notna()].copy()
    if not cleaned.empty:
        cleaned = cleaned.loc[cleaned.geometry.area >= MAP_REGION_MIN_AREA_M2].copy()
    if cleaned.empty:
        return gpd.GeoDataFrame(geometry=[], crs=original_crs)

    cleaned["geometry"] = cleaned.geometry.buffer(0)
    return cleaned.to_crs(original_crs).reset_index(drop=True)


def draw_model_region_background(
    ax,
    regions: gpd.GeoDataFrame,
    n: pypsa.Network,
) -> None:
    """Draw filled model extent plus internal ReEDS zone boundaries."""
    region_background = get_model_region_background(regions)
    region_boundaries = get_zone_region_boundaries(n, regions)

    region_background.plot(
        ax=ax,
        facecolor=MAP_BACKGROUND_COLOR,
        edgecolor="none",
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=MAP_REGION_BOUNDARY_WIDTH,
        alpha=MAP_REGION_BOUNDARY_ALPHA,
        zorder=0,
    )
    region_boundaries.plot(
        ax=ax,
        facecolor="none",
        edgecolor=MAP_REGION_BOUNDARY_COLOR,
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=MAP_REGION_BOUNDARY_WIDTH,
        alpha=MAP_REGION_BOUNDARY_ALPHA,
        zorder=1,
    )


def _empty_capacity_series() -> pd.Series:
    return pd.Series(
        dtype=float,
        index=pd.MultiIndex.from_arrays([[], []], names=["bus", "carrier"]),
    )


def _combine_capacity_series(series_list: list[pd.Series]) -> pd.Series:
    non_empty = [series for series in series_list if series is not None and not series.empty]
    if not non_empty:
        return _empty_capacity_series()
    return pd.concat(non_empty).groupby(level=["bus", "carrier"]).sum()


def _get_line_xs(n: pypsa.Network) -> pd.DataFrame:
    return getattr(n, "line_xs", pd.DataFrame())


def get_plot_branch_components(n: pypsa.Network) -> list[str]:
    """Return branch components supported by the local PyPSA plotting backend."""
    components = ["Line", "Link", "Transformer"]
    if not _get_line_xs(n).empty:
        components.insert(1, "LineX")
    return components


def get_line_plot_values(
    n: pypsa.Network,
    line_attr: str | None = None,
    line_x_attr: str | None = None,
    fill_value: float = 0.0,
) -> pd.Series:
    """Return line-like plotting values for native Line and LineX plotting."""
    if line_attr is None:
        line_values = pd.Series(fill_value, index=n.lines.index, dtype=float)
    else:
        line_values = n.lines.get(
            line_attr,
            pd.Series(fill_value, index=n.lines.index, dtype=float),
        ).fillna(fill_value)

    line_xs = _get_line_xs(n)
    if line_xs.empty:
        return line_values

    attr = line_x_attr or line_attr
    if attr is None:
        line_x_values = pd.Series(fill_value, index=line_xs.index, dtype=float)
    else:
        line_x_values = line_xs.get(
            attr,
            pd.Series(fill_value, index=line_xs.index, dtype=float),
        ).fillna(fill_value)

    return pd.concat([line_values, line_x_values])


def append_line_x_values(
    n: pypsa.Network,
    line_values: pd.Series,
    line_x_values: pd.Series,
) -> pd.Series:
    """Append synthetic LineX values to a line-like series."""
    if _get_line_xs(n).empty:
        return line_values
    return pd.concat([line_values, line_x_values])


def get_transmission_link_values(
    n: pypsa.Network,
    attr: str,
    fill_value: float = 0.0,
) -> pd.Series:
    """Return plotting values for transmission links only."""
    if n.links.empty:
        return pd.Series(dtype=float)

    transmission_links = n.links.carrier.isin(["AC", "DC"])
    if not transmission_links.any():
        return pd.Series(dtype=float)

    return n.links.loc[transmission_links, attr].fillna(fill_value)


def get_transmission_link_colors(n: pypsa.Network) -> pd.Series:
    """Return colors for AC/DC transmission links."""
    if n.links.empty:
        return pd.Series(dtype=object)

    transmission_links = n.links.carrier.isin(["AC", "DC"])
    if not transmission_links.any():
        return pd.Series(dtype=object)

    ac_color = get_carrier_color(n, "AC", "teal")
    dc_color = get_carrier_color(n, "DC", "#cf1dab")
    return n.links.loc[transmission_links, "carrier"].map({"AC": ac_color, "DC": dc_color})


def get_line_x_sssc_plot_values(
    n: pypsa.Network,
    attr: str = "sssc_nom_opt",
    fill_value: float = 0.0,
) -> pd.Series:
    """Return LineX SSSC capacities used for marker plotting."""
    line_xs = _get_line_xs(n)
    if line_xs.empty:
        return pd.Series(dtype=float)
    if attr in line_xs.columns:
        return line_xs[attr].fillna(fill_value)
    return pd.Series(fill_value, index=line_xs.index, dtype=float)


def get_line_x_sssc_new_plot_values(
    n: pypsa.Network,
    base_attr: str = "sssc_nom",
    opt_attr: str = "sssc_nom_opt",
) -> pd.Series:
    """Return LineX newly installed SSSC capacities (opt - base), clipped at zero."""
    line_xs = _get_line_xs(n)
    if line_xs.empty:
        return pd.Series(dtype=float)
    base = line_xs.get(base_attr, pd.Series(0.0, index=line_xs.index, dtype=float)).fillna(0.0)
    opt = line_xs.get(opt_attr, pd.Series(0.0, index=line_xs.index, dtype=float)).fillna(0.0)
    return (opt - base).clip(lower=0.0)


def get_line_x_midpoint_values(
    n: pypsa.Network,
    values: pd.Series,
    min_value: float = 0.0,
) -> pd.DataFrame:
    """Return LineX midpoint coordinates and values for positive-capacity assets."""
    line_xs = _get_line_xs(n)
    if line_xs.empty or values.empty:
        return pd.DataFrame(columns=["x", "y", "value"])
    if not {"bus0", "bus1"}.issubset(line_xs.columns):
        return pd.DataFrame(columns=["x", "y", "value"])
    if not {"x", "y"}.issubset(n.buses.columns):
        return pd.DataFrame(columns=["x", "y", "value"])

    line_x_values = values.reindex(line_xs.index).fillna(0.0)
    active = line_x_values[line_x_values > min_value]
    if active.empty:
        return pd.DataFrame(columns=["x", "y", "value"])

    buses = n.buses[["x", "y"]]
    line_x_active = line_xs.loc[active.index, ["bus0", "bus1"]].copy()
    line_x_active = (
        line_x_active.join(buses, on="bus0")
        .rename(columns={"x": "x0", "y": "y0"})
        .join(buses, on="bus1")
        .rename(columns={"x": "x1", "y": "y1"})
    )
    line_x_active["value"] = active
    line_x_active["x"] = (line_x_active["x0"] + line_x_active["x1"]) / 2
    line_x_active["y"] = (line_x_active["y0"] + line_x_active["y1"]) / 2
    line_x_active = line_x_active.dropna(subset=["x", "y", "value"])
    return line_x_active[["x", "y", "value"]]


def _scale_sssc_marker_area(values: pd.Series) -> pd.Series:
    """Scale capacities to scatter marker area (pt^2), proportional to MW."""
    if values.empty:
        return values
    scaled = values.astype(float) * SSSC_MARKER_AREA_SCALE
    return scaled.clip(lower=0.0)


def get_capacity_map_legend_entries(
    n: pypsa.Network,
    bus_values: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Build color and label series for the capacity-map technology legend."""
    if isinstance(bus_values.index, pd.MultiIndex) and "carrier" in bus_values.index.names:
        legend_carriers = (
            bus_values.groupby(level="carrier")
            .sum()
            .sort_values(ascending=False)
            .index
        )
    else:
        legend_carriers = pd.Index([])

    palette = build_visualization_palette(n.carriers, labels=legend_carriers)
    bus_colors = pd.Series({carrier: palette.get(carrier, "#000000") for carrier in legend_carriers})
    nice_names = pd.Series(legend_carriers, index=legend_carriers)
    return bus_colors, nice_names


def _get_max_series_value(values: pd.Series | None) -> float:
    if values is None or values.empty:
        return 0.0
    return float(values.fillna(0.0).clip(lower=0.0).max())


def get_capacity_size_max_value(bus_values: pd.Series) -> float:
    """Return the largest plotted pie size, aggregating carrier slices by bus."""
    if bus_values.empty:
        return 0.0

    if isinstance(bus_values.index, pd.MultiIndex):
        level = "bus" if "bus" in bus_values.index.names else 0
        grouped = bus_values.groupby(level=level).sum()
        return _get_max_series_value(grouped)

    return _get_max_series_value(bus_values)


def get_adaptive_legend_values(default_values: list[float], max_value: float) -> list[float]:
    """
    Select legend breakpoints from default tiers and the actual plotted maximum.

    If x lies between tier[n] and tier[n+1]:
      - show 1..n and x when x > 1.5 * tier[n]
      - show 1..n-1 and x when tier[n] < x <= 1.5 * tier[n]
    """
    if max_value <= 0:
        return []

    defaults = sorted(float(value) for value in default_values)
    if not defaults:
        return []

    if max_value < defaults[0]:
        return [float(max_value)]

    for idx, value in enumerate(defaults):
        if np.isclose(max_value, value):
            return defaults[: idx + 1]

    lower_idx = np.searchsorted(defaults, max_value, side="right") - 1
    if lower_idx < 0:
        return [float(max_value)]

    if lower_idx >= len(defaults) - 1 or max_value > defaults[lower_idx] * 1.5:
        prefix = defaults[: lower_idx + 1]
    else:
        prefix = defaults[:lower_idx]

    return [*prefix, float(max_value)]


def _format_significant_number(value: float, significant_digits: int = 3) -> str:
    if np.isclose(value, 0.0):
        return "0"

    decimal_value = Decimal(str(float(value)))
    exponent = decimal_value.adjusted() - significant_digits + 1
    rounded = decimal_value.quantize(Decimal(f"1e{exponent}"), rounding=ROUND_HALF_UP)
    return format(rounded, "f")


def _format_legend_value(
    value: float,
    small_unit: str,
    large_unit: str,
    significant_digits: int | None = None,
) -> str:
    display_value = value / 1000 if value >= 1000 else value
    unit = large_unit if value >= 1000 else small_unit

    if significant_digits is not None:
        number = _format_significant_number(display_value, significant_digits)
    elif float(display_value).is_integer():
        number = str(int(display_value))
    else:
        number = f"{display_value:g}"

    return f"{number} {unit}"


def _build_legend_labels(values: list[float], small_unit: str, large_unit: str) -> list[str]:
    if not values:
        return []

    max_index = len(values) - 1
    return [
        _format_legend_value(
            value,
            small_unit,
            large_unit,
            significant_digits=3 if idx == max_index else None,
        )
        for idx, value in enumerate(values)
    ]


def _measure_artist_size_in_axes(ax: plt.Axes, artist) -> tuple[float, float]:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artist_bbox = artist.get_window_extent(renderer=renderer)
    ax_bbox = ax.get_window_extent(renderer=renderer)
    if ax_bbox.width == 0 or ax_bbox.height == 0:
        return 0.0, 0.0
    return artist_bbox.width / ax_bbox.width, artist_bbox.height / ax_bbox.height


def _apply_plot_font_theme() -> None:
    plt.rcParams.update(PLOT_FONT_RC_PARAMS)


def _create_right_legend_panel(fig: plt.Figure) -> plt.Axes:
    """Create a frameless right-side legend axis for tests and ad hoc layout."""
    ax = fig.add_axes([0.73, 0.06, 0.25, 0.88], frameon=False)
    ax.set_axis_off()
    return ax


def _get_circle_radii_in_axes(
    ax: plt.Axes,
    values: list[float],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert bus values to legend radii using the same scaling as map buses."""
    if not values:
        empty = np.array([], dtype=float)
        return empty, empty

    if scale <= 0:
        msg = "Legend circle scale must be positive."
        raise ValueError(msg)

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer=renderer)
    if ax_bbox.width == 0 or ax_bbox.height == 0:
        empty = np.array([], dtype=float)
        return empty, empty

    sizes = np.asarray(values, dtype=float) / float(scale)
    if hasattr(ax, "projection"):
        sizes = sizes * _get_projected_area_factor(ax, DEFAULT_EPSG) ** 2

    raw_radii = np.sqrt(np.clip(sizes, a_min=0.0, a_max=None))
    unit_x, unit_y = np.diff(ax.transData.transform([(0, 0), (1, 1)]), axis=0)[0]
    unit_pixels = min(abs(unit_x), abs(unit_y))
    display_radii = raw_radii * unit_pixels

    rx = display_radii / ax_bbox.width
    ry = display_radii / ax_bbox.height
    return rx, ry


def _get_projected_area_factor(ax: plt.Axes, original_crs: int = DEFAULT_EPSG) -> float:
    """Return the cartopy area correction used for projected map circles.

    This mirrors the PyPSA helper but stays local to avoid version-specific
    imports such as ``pypsa.constants`` / ``pypsa.geo``.
    """
    if not hasattr(ax, "projection"):
        return 1.0

    x1, x2, y1, y2 = ax.get_extent()
    if original_crs != DEFAULT_EPSG:
        msg = f"Unsupported CRS fallback: {original_crs}"
        raise ValueError(msg)

    pbounds = ccrs.PlateCarree().transform_points(
        ax.projection,
        np.array([x1, x2]),
        np.array([y1, y2]),
    )
    numerator = abs((x2 - x1) * (y2 - y1))
    denominator = abs((pbounds[0] - pbounds[1])[:2].prod())
    if np.isclose(denominator, 0.0):
        return 1.0
    return float(np.sqrt(numerator / denominator))


def _add_legend_block(
    ax: plt.Axes,
    handles: list,
    title: str,
    anchor: tuple[float, float],
    loc: str,
    legend_style: dict | None = None,
    **legend_kwargs,
) -> Legend | None:
    if not handles:
        return None

    legend_style = legend_style or BOXED_LEGEND_STYLE
    legend_kwargs.setdefault("fontsize", LEGEND_FONT_SIZE)
    legend_kwargs.setdefault("title_fontsize", LEGEND_TITLE_SIZE)
    legend = ax.legend(
        handles=handles,
        title=title,
        bbox_to_anchor=anchor,
        loc=loc,
        bbox_transform=ax.transAxes,
        labelspacing=0.45,
        handletextpad=0.7,
        handlelength=2.2,
        **legend_style,
        **legend_kwargs,
    )
    _style_legend_text(ax, legend)
    ax.add_artist(legend)
    return legend


def _get_no_pie_legend_font_kwargs(show_capacity_pie: bool) -> dict[str, float]:
    if show_capacity_pie:
        return {}
    return {
        "fontsize": LEGEND_FONT_SIZE + NO_PIE_LEGEND_FONT_SIZE_INCREMENT,
        "title_fontsize": LEGEND_TITLE_SIZE + NO_PIE_LEGEND_FONT_SIZE_INCREMENT,
    }


def _get_transmission_legend_anchor(show_capacity_pie: bool) -> tuple[float, float]:
    if show_capacity_pie:
        return (BOTTOM_LEGEND_LEFT_X, BOTTOM_LEGEND_ANCHOR_Y)
    return (
        BOTTOM_LEGEND_LEFT_X,
        BOTTOM_LEGEND_ANCHOR_Y + NO_PIE_TRANSMISSION_LEGEND_Y_OFFSET,
    )


def _get_latest_legend(ax: plt.Axes) -> Legend | None:
    legends = [artist for artist in ax.figure.artists if isinstance(artist, Legend)]
    legends.extend(artist for artist in ax.get_children() if isinstance(artist, Legend))
    return legends[-1] if legends else None


def _style_legend_text(ax: plt.Axes, legend: Legend | None = None) -> None:
    legend = legend or _get_latest_legend(ax)
    if legend is None:
        return
    if hasattr(legend, "_legend_box"):
        legend._legend_box.align = "left"
    legend.get_title().set_color(LEGEND_TEXT_COLOR)
    legend.get_title().set_fontweight("bold")
    legend.get_title().set_ha("left")
    legend.get_title().set_multialignment("left")
    for text in legend.get_texts():
        text.set_color(LEGEND_TEXT_COLOR)



def _add_circle_legend(
    ax: plt.Axes,
    legend_values: list[float],
    scale: float,
    title: str,
    anchor: tuple[float, float],
    loc: str,
    small_unit: str,
    large_unit: str,
    legend_style: dict | None = None,
) -> FancyBboxPatch | None:
    """Backward-compatible wrapper for the map-matched capacity legend."""
    return _add_circle_legend_matching_map(
        ax=ax,
        legend_values=legend_values,
        scale=scale,
        title=title,
        anchor=anchor,
        loc=loc,
        small_unit=small_unit,
        large_unit=large_unit,
        legend_style=legend_style,
    )

def _add_circle_legend_matching_map(
    ax: plt.Axes,
    legend_values: list[float],
    scale: float,
    title: str,
    anchor: tuple[float, float],
    loc: str,
    small_unit: str,
    large_unit: str,
    legend_style: dict | None = None,
) -> FancyBboxPatch | None:
    """Draw a nested-circles legend whose size-capacity mapping matches the map."""
    if not legend_values:
        return None

    legend_style = legend_style or BOXED_LEGEND_STYLE

    values_sorted = sorted(legend_values)
    labels = _build_legend_labels(values_sorted, small_unit, large_unit)
    n = len(values_sorted)

    fig = ax.figure
    fig_w_in, fig_h_in = fig.get_size_inches()
    ax_pos = ax.get_position()
    ar = (ax_pos.width * fig_w_in) / (ax_pos.height * fig_h_in)
    rx, ry = _get_circle_radii_in_axes(ax, values_sorted, scale)
    if len(rx) == 0 or len(ry) == 0:
        return None

    x_anc, y_anc = anchor
    side_pad = 0.010
    top_pad = 0.010
    title_h = 0.022
    line_gap = CAPACITY_LEGEND_LABEL_GAP / ar

    max_chars = max(len(lbl) for lbl in labels) if labels else 5
    char_w_ax = LEGEND_FONT_SIZE / 72.0 / (ax_pos.width * fig_w_in)
    char_h_ax = LEGEND_FONT_SIZE / 72.0 / (ax_pos.height * fig_h_in)
    text_w = max_chars * char_w_ax * 0.60

    x_text_right = x_anc - side_pad
    x_text_left = x_text_right - text_w
    x_line_end = x_text_left - line_gap
    x_ctr = x_line_end - rx[-1] - (CAPACITY_LEGEND_INDICATOR_LINE_EXTRA / ar)

    y_circles_top = y_anc - title_h - top_pad - CAPACITY_LEGEND_TITLE_GAP
    x_box_left = x_ctr - rx[-1] - side_pad

    y_top_lbl = y_circles_top - ry[0]
    y_bot_lbl = y_circles_top - ry[-1]
    if n > 1:
        step = ((y_top_lbl - y_bot_lbl) / (n - 1)) * CAPACITY_LEGEND_LABEL_SPACING_FACTOR
        y_label = [y_top_lbl - i * step for i in range(n)]
    else:
        y_label = [y_bot_lbl]

    circle_box_bot = y_circles_top - 2.0 * ry[-1] - top_pad
    label_box_bot = min(y_label) - 0.7 * char_h_ax - top_pad
    y_box_bot = min(circle_box_bot, label_box_bot)

    box_w = x_anc - x_box_left
    box_h = y_anc - y_box_bot
    bg = FancyBboxPatch(
        (x_box_left, y_box_bot),
        box_w,
        box_h,
        boxstyle="square,pad=0",
        facecolor=legend_style.get("facecolor", "white") if legend_style.get("frameon", True) else "none",
        edgecolor=legend_style.get("edgecolor", "none"),
        alpha=legend_style.get("framealpha", 0.9),
        transform=ax.transAxes,
        zorder=4,
        clip_on=False,
    )
    ax.add_patch(bg)

    ax.text(
        (x_box_left + x_anc) / 2.0,
        y_anc - top_pad,
        title,
        ha="center",
        va="top",
        fontsize=LEGEND_TITLE_SIZE,
        fontweight="bold",
        color=LEGEND_TEXT_COLOR,
        transform=ax.transAxes,
        zorder=5,
        clip_on=False,
    )

    for i in range(n - 1, -1, -1):
        y_ctr_i = y_circles_top - ry[i]

        ax.add_patch(
            Ellipse(
                (x_ctr, y_ctr_i),
                width=2.0 * rx[i],
                height=2.0 * ry[i],
                facecolor="none",
                edgecolor=LEGEND_LINE_COLOR,
                linewidth=1.2,
                transform=ax.transAxes,
                zorder=5,
                clip_on=False,
            ),
        )

        ax.plot(
            [x_ctr + rx[i], x_line_end],
            [y_ctr_i, y_label[i]],
            color=LEGEND_LINE_COLOR,
            linewidth=0.8,
            transform=ax.transAxes,
            zorder=5,
            clip_on=False,
        )

        ax.text(
            x_text_left,
            y_label[i],
            labels[i],
            ha="left",
            va="center",
            fontsize=LEGEND_FONT_SIZE,
            color=LEGEND_TEXT_COLOR,
            transform=ax.transAxes,
            zorder=5,
            clip_on=False,
        )

    return bg


def _build_transmission_capacity_handles(
    n: pypsa.Network,
    line_scale: float,
    legend_values_mw: list[float],
) -> list[Line2D]:
    labels = _build_legend_labels(legend_values_mw, "MW", "GW")
    return [
        Line2D(
            [0],
            [0],
            color=LEGEND_MONOCHROME_COLOR,
            lw=max(value / line_scale, 0.3),
            label=label,
        )
        for value, label in zip(legend_values_mw, labels)
    ]


def _build_transmission_type_handles(n: pypsa.Network, show_dc_link: bool) -> list[Line2D]:
    handles = [Line2D([0], [0], color=get_carrier_color(n, "AC", "teal"), lw=3, label="AC line")]
    if show_dc_link:
        handles.append(Line2D([0], [0], color=get_carrier_color(n, "DC", "#cf1dab"), lw=3, label="DC line"))
    return handles


def _build_sssc_handles(legend_values_mw: list[float]) -> list[Line2D]:
    areas = _scale_sssc_marker_area(pd.Series(legend_values_mw, dtype=float))
    labels = _build_legend_labels(legend_values_mw, "MVAr", "GVAr")
    return [
        Line2D(
            [0],
            [0],
            marker="+",
            color=SSSC_MARKER_COLOR,
            linestyle="None",
            markeredgewidth=1.4,
            markersize=np.sqrt(area),
            label=label,
        )
        for area, label in zip(areas, labels)
    ]


def _build_carrier_patch_handles(bus_colors: pd.Series, nice_names: pd.Series) -> list[Patch]:
    return [
        Patch(
            facecolor=color,
            edgecolor="none",
            alpha=CAPACITY_MAP_BUS_ALPHA,
            label=str(nice_names[label]),
        )
        for label, color in bus_colors.items()
    ]


def _add_technology_block(
    ax: plt.Axes,
    bus_colors: pd.Series,
    nice_names: pd.Series,
    bus_legend_values: list[float],
    bus_scale: float,
    panel_title: str | None = None,
    value_title: str | None = None,
) -> None:
    anchor_x = TECH_LEGEND_ANCHOR_X
    anchor_y = TECH_LEGEND_ANCHOR_Y

    capacity_legend = None
    if bus_legend_values:
        capacity_legend = _add_circle_legend_matching_map(
            ax,
            bus_legend_values,
            scale=bus_scale,
            title=panel_title or "",
            anchor=(anchor_x, anchor_y),
            loc="upper right",
            small_unit="MW",
            large_unit="GW",
            legend_style=TRANSPARENT_LEGEND_STYLE,
        )

    next_anchor_y = anchor_y
    if capacity_legend is not None:
        _, capacity_height = _measure_artist_size_in_axes(ax, capacity_legend)
        next_anchor_y -= capacity_height + TECH_LEGEND_SECTION_GAP

    if not bus_colors.empty:
        _add_legend_block(
            ax,
            _build_carrier_patch_handles(bus_colors, nice_names),
            title="",
            anchor=(anchor_x, next_anchor_y),
            loc="upper right",
            legend_style=TRANSPARENT_LEGEND_STYLE,
        )


def add_capacity_map_legends(
    ax: plt.Axes,
    n: pypsa.Network,
    bus_values: pd.Series,
    bus_scale: float,
    bus_colors: pd.Series,
    nice_names: pd.Series,
    line_values: pd.Series,
    link_values: pd.Series,
    line_scale: float,
    show_dc_link: bool,
    sssc_values: pd.Series | None = None,
    show_capacity_pie: bool = True,
) -> None:
    """Lay out capacity-map legends inside the map axes."""
    if show_capacity_pie:
        bus_legend_values = get_adaptive_legend_values(
            CAPACITY_CIRCLE_LEGEND_VALUES_MW,
            get_capacity_size_max_value(bus_values),
        )
        _add_technology_block(
            ax,
            bus_colors,
            nice_names,
            bus_legend_values,
            bus_scale,
            panel_title="Technology",
        )

    transmission_values = get_adaptive_legend_values(
        TRANSMISSION_CAPACITY_LEGEND_VALUES_MW,
        max(_get_max_series_value(line_values), _get_max_series_value(link_values)),
    )
    legend_font_kwargs = _get_no_pie_legend_font_kwargs(show_capacity_pie)
    transmission_handles = _build_transmission_capacity_handles(n, line_scale, transmission_values)
    transmission_handles.extend(_build_transmission_type_handles(n, show_dc_link))
    transmission_legend = _add_legend_block(
        ax,
        transmission_handles,
        title="Transmission",
        anchor=_get_transmission_legend_anchor(show_capacity_pie),
        loc="upper left",
        **legend_font_kwargs,
    )

    sssc_legend_values = get_adaptive_legend_values(SSSC_CAPACITY_LEGEND_VALUES_MW, _get_max_series_value(sssc_values))
    if sssc_legend_values:
        transmission_width = 0.0
        if transmission_legend is not None:
            transmission_width, _ = _measure_artist_size_in_axes(ax, transmission_legend)

        _add_legend_block(
            ax,
            _build_sssc_handles(sssc_legend_values),
            title="SSSC",
            anchor=(
                BOTTOM_LEGEND_LEFT_X + transmission_width + BOTTOM_LEGEND_GROUP_GAP,
                BOTTOM_LEGEND_ANCHOR_Y,
            ),
            loc="upper left",
            **legend_font_kwargs,
        )


def add_demand_map_legends(
    ax: plt.Axes,
    n: pypsa.Network,
    bus_values: pd.Series,
    bus_scale: float,
    line_values: pd.Series,
    link_values: pd.Series,
    line_scale: float,
) -> None:
    carrier_frame = build_visualization_carriers(n.carriers)
    bus_colors = carrier_frame.color.fillna("#000000")
    nice_names = carrier_frame.index.to_series().reindex(carrier_frame.index)
    _add_technology_block(
        ax,
        bus_colors,
        nice_names,
        get_adaptive_legend_values(CAPACITY_CIRCLE_LEGEND_VALUES_MW, _get_max_series_value(bus_values)),
        bus_scale,
        panel_title="Technology",
    )
    transmission_values = get_adaptive_legend_values(
        TRANSMISSION_CAPACITY_LEGEND_VALUES_MW,
        max(_get_max_series_value(line_values), _get_max_series_value(link_values)),
    )
    transmission_handles = _build_transmission_capacity_handles(n, line_scale, transmission_values)
    transmission_handles.extend(_build_transmission_type_handles(n, not n.links.empty and n.links.carrier.eq("DC").any()))
    _add_legend_block(
        ax,
        transmission_handles,
        title="Transmission",
        anchor=(BOTTOM_LEGEND_LEFT_X, BOTTOM_LEGEND_ANCHOR_Y),
        loc="lower left",
    )


def plot_line_x_sssc_markers(
    ax: plt.Axes,
    n: pypsa.Network,
    sssc_values: pd.Series,
) -> pd.DataFrame:
    """Plot LineX SSSC capacities as plus markers at line midpoints."""
    midpoint_values = get_line_x_midpoint_values(n, sssc_values, min_value=0.0)
    if midpoint_values.empty:
        return midpoint_values

    marker_sizes = _scale_sssc_marker_area(midpoint_values["value"])
    ax.scatter(
        midpoint_values["x"],
        midpoint_values["y"],
        s=marker_sizes,
        marker="+",
        c=SSSC_MARKER_COLOR,
        linewidths=1.4,
        alpha=0.95,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )

    return midpoint_values
def get_capacity_map_carriers(n: pypsa.Network, configured_carriers: list[str]) -> list[str]:
    """Get capacity-map carriers from the solved network instead of config-only lists."""
    carriers = []

    for component in (n.generators, n.storage_units):
        if not component.empty and "carrier" in component.columns:
            carriers.extend(
                component.carrier.dropna().map(lambda carrier: get_visualization_label(carrier, n.carriers)).tolist(),
            )

    if not n.links.empty:
        link_mask = (
            n.links.bus0.map(n.buses.carrier).eq("AC") | n.links.bus1.map(n.buses.carrier).eq("AC")
        ) & ~n.links.carrier.isin(CAPACITY_MAP_EXCLUDED_LINK_CARRIERS)
        tes_discharge_mask = n.links.carrier.eq("tes") & n.links.index.to_series().str.contains(
            "discharge",
            case=False,
            na=False,
        )
        link_mask &= (~n.links.carrier.eq("tes")) | tes_discharge_mask
        output_link_carriers = n.links.loc[link_mask, "carrier"].dropna().map(
            lambda carrier: CAPACITY_MAP_LINK_CARRIER_MAP.get(carrier, carrier),
        )
        carriers.extend(output_link_carriers.map(lambda carrier: get_visualization_label(carrier, n.carriers)).tolist())

    actual_carriers = list(dict.fromkeys(carriers))
    actual_carrier_set = set(actual_carriers)
    configured_labels = [
        get_visualization_label(carrier, n.carriers)
        for carrier in configured_carriers
        if carrier not in CAPACITY_MAP_EXCLUDED_CARRIERS
    ]
    ordered = [carrier for carrier in configured_labels if carrier in actual_carrier_set]
    ordered.extend(carrier for carrier in actual_carriers if carrier not in ordered)
    return [carrier for carrier in ordered if carrier not in CAPACITY_MAP_EXCLUDED_CARRIERS]


def get_capacity_map_bus_values(
    n: pypsa.Network,
    carriers: list[str],
    capacity_attr: str,
) -> pd.Series:
    """Aggregate map capacities in MW for generators, storage units, and selected links."""
    carrier_filter = set(carriers) - CAPACITY_MAP_EXCLUDED_CARRIERS
    components = []

    for component in (n.generators, n.storage_units):
        component_attr = capacity_attr if capacity_attr in component.columns else None
        if component_attr is None and "p_nom_opt" in component.columns:
            component_attr = "p_nom_opt"
        if component_attr is None:
            continue
        if component.empty or component_attr not in component.columns:
            continue
        values = component.loc[:, ["bus", "carrier", component_attr]].copy()
        if values.empty:
            continue
        values["carrier"] = values["carrier"].map(lambda carrier: get_visualization_label(carrier, n.carriers))
        values = values[values.carrier.isin(carrier_filter)]
        if values.empty:
            continue
        values["capacity_mw"] = values[component_attr].fillna(0)
        grouped = values.groupby(["bus", "carrier"])["capacity_mw"].sum()
        components.append(grouped)

    capacity_column = capacity_attr if capacity_attr in n.links.columns else None
    if capacity_column is None and "p_nom_opt" in n.links.columns:
        capacity_column = "p_nom_opt"

    link_columns = ["bus0", "bus1", "carrier", "efficiency"]
    if capacity_column is not None:
        link_columns.insert(3, capacity_column)

    if not n.links.empty and capacity_column is not None:
        link_values = n.links.loc[
            ~n.links.carrier.isin(CAPACITY_MAP_EXCLUDED_LINK_CARRIERS),
            link_columns,
        ].copy()
        if not link_values.empty:
            bus0_is_ac = link_values["bus0"].map(n.buses.carrier).eq("AC")
            bus1_is_ac = link_values["bus1"].map(n.buses.carrier).eq("AC")
            link_values = link_values.loc[bus0_is_ac | bus1_is_ac].copy()
            if not link_values.empty:
                tes_discharge_mask = link_values["carrier"].eq("tes") & link_values.index.to_series().str.contains(
                    "discharge",
                    case=False,
                    na=False,
                )
                electrolysis_mask = link_values["carrier"].eq("electrolysis")
                default_mask = ~link_values["carrier"].eq("tes") & ~electrolysis_mask

                selected_links = []

                if tes_discharge_mask.any():
                    tes_values = link_values.loc[tes_discharge_mask].copy()
                    tes_values["bus"] = np.where(
                        bus1_is_ac.reindex(tes_values.index).fillna(False),
                        tes_values["bus1"],
                        tes_values["bus0"],
                    )
                    tes_values["capacity_mw"] = (
                        tes_values[capacity_column].fillna(0) * tes_values["efficiency"].fillna(1.0)
                    )
                    selected_links.append(tes_values)

                if electrolysis_mask.any():
                    electrolysis_values = link_values.loc[electrolysis_mask].copy()
                    electrolysis_values["bus"] = np.where(
                        bus0_is_ac.reindex(electrolysis_values.index).fillna(False),
                        electrolysis_values["bus0"],
                        electrolysis_values["bus1"],
                    )
                    electrolysis_values["capacity_mw"] = electrolysis_values[capacity_column].fillna(0)
                    selected_links.append(electrolysis_values)

                if default_mask.any():
                    default_values = link_values.loc[default_mask].copy()
                    default_capacity = default_values[capacity_column].fillna(0)
                    default_values["bus"] = np.where(
                        bus1_is_ac.reindex(default_values.index).fillna(False),
                        default_values["bus1"],
                        default_values["bus0"],
                    )
                    default_values["capacity_mw"] = np.where(
                        bus1_is_ac.reindex(default_values.index).fillna(False),
                        default_capacity * default_values["efficiency"].fillna(1.0),
                        default_capacity,
                    )
                    selected_links.append(default_values)

                if selected_links:
                    link_values = pd.concat(selected_links)
                else:
                    link_values = pd.DataFrame(columns=["bus", "carrier", "capacity_mw"])

                link_values["carrier"] = link_values["carrier"].map(
                    lambda carrier: CAPACITY_MAP_LINK_CARRIER_MAP.get(carrier, carrier),
                )
                link_values["carrier"] = link_values["carrier"].map(
                    lambda carrier: get_visualization_label(carrier, n.carriers),
                )
                link_values = link_values[link_values.carrier.isin(carrier_filter)]
                if not link_values.empty:
                    grouped = link_values.groupby(["bus", "carrier"])["capacity_mw"].sum()
                    components.append(grouped)

    bus_values = _combine_capacity_series(components)
    if bus_values.empty:
        return bus_values

    bus_values = remove_sector_buses(bus_values)
    bus_values = bus_values.groupby(level=["bus", "carrier"]).sum()
    return bus_values[bus_values > 0]


def _zone_pair_labels(zone_map: pd.Series, bus0: pd.Series, bus1: pd.Series) -> pd.Series:
    """Return a canonical 'zoneA~zoneB' label per branch, dropping unmapped/intra-zone branches."""
    zone0 = bus0.map(zone_map)
    zone1 = bus1.map(zone_map)
    valid = zone0.notna() & zone1.notna() & (zone0 != zone1)
    lo = zone0.where(zone0 <= zone1, zone1)
    hi = zone1.where(zone0 <= zone1, zone0)
    labels = lo.astype(str) + "~" + hi.astype(str)
    return labels[valid]


def aggregate_bus_values_to_zone(n: pypsa.Network, bus_values: pd.Series) -> pd.Series:
    """Re-key (bus, carrier)-indexed capacities to (reeds_zone, carrier)."""
    if bus_values.empty:
        return bus_values
    zone_map = n.buses["reeds_zone"]
    frame = bus_values.rename("value").reset_index()
    frame["bus"] = frame["bus"].map(zone_map)
    frame = frame.dropna(subset=["bus"])
    if frame.empty:
        return _empty_capacity_series()
    return frame.groupby(["bus", "carrier"])["value"].sum()


def build_reeds_zone_capacity_network(
    n: pypsa.Network,
    line_values: pd.Series,
    link_values: pd.Series,
) -> tuple[pypsa.Network, pd.Series, pd.Series]:
    """Aggregate AC/DC transmission (Line + LineX + AC/DC Link) between ReEDS zones.

    Buses become one node per zone (positioned at the mean of member-bus
    coordinates); branches between the same zone are dropped and branches
    between different zones are summed, keyed to match the returned network's
    Line/Link indices for direct use as ``line_widths``/``link_widths``.
    """
    zone_map = n.buses["reeds_zone"]
    zones = zone_map.dropna().unique().tolist()

    zone_n = pypsa.Network()
    if zones:
        zone_positions = n.buses.groupby(zone_map)[["x", "y"]].mean().reindex(zones)
        zone_n.madd("Bus", zones, x=zone_positions["x"].to_numpy(), y=zone_positions["y"].to_numpy(), carrier="AC")
    zone_n.carriers = n.carriers.copy()

    branch_bus0 = n.lines["bus0"] if not n.lines.empty else pd.Series(dtype=object)
    branch_bus1 = n.lines["bus1"] if not n.lines.empty else pd.Series(dtype=object)
    line_xs = _get_line_xs(n)
    if not line_xs.empty:
        branch_bus0 = pd.concat([branch_bus0, line_xs["bus0"]])
        branch_bus1 = pd.concat([branch_bus1, line_xs["bus1"]])
    branch_pairs = _zone_pair_labels(zone_map, branch_bus0, branch_bus1)

    zone_line_values = pd.Series(dtype=float)
    if not branch_pairs.empty and not line_values.empty:
        aligned = line_values.reindex(branch_pairs.index).fillna(0.0)
        grouped = aligned.groupby(branch_pairs).sum()
        if not grouped.empty:
            zone_pairs = [label.split("~", 1) for label in grouped.index]
            zone_n.madd(
                "Line",
                grouped.index,
                bus0=[pair[0] for pair in zone_pairs],
                bus1=[pair[1] for pair in zone_pairs],
            )
            zone_line_values = grouped

    zone_link_values = pd.Series(dtype=float)
    if not n.links.empty and not link_values.empty:
        links = n.links.reindex(link_values.index)
        link_pairs = _zone_pair_labels(zone_map, links["bus0"], links["bus1"])
        if not link_pairs.empty:
            aligned_links = link_values.reindex(link_pairs.index).fillna(0.0)
            carriers = links["carrier"].reindex(link_pairs.index)
            grouped_links = aligned_links.groupby([carriers, link_pairs]).sum()
            if not grouped_links.empty:
                labels = [f"{carrier}::{pair}" for carrier, pair in grouped_links.index]
                zone_pairs = [pair.split("~", 1) for _, pair in grouped_links.index]
                zone_n.madd(
                    "Link",
                    labels,
                    bus0=[pair[0] for pair in zone_pairs],
                    bus1=[pair[1] for pair in zone_pairs],
                    carrier=[carrier for carrier, _ in grouped_links.index],
                )
                zone_link_values = pd.Series(grouped_links.to_numpy(), index=labels)

    return zone_n, zone_line_values, zone_link_values


def resolve_capacity_map_plot_inputs(
    n: pypsa.Network,
    bus_values: pd.Series,
    line_values: pd.Series,
    link_values: pd.Series,
    sssc_values: pd.Series | None,
    show_capacity_pie: bool,
) -> tuple[pypsa.Network, pd.Series, pd.Series, pd.Series, pd.Series | None, pypsa.Network | None]:
    """Pick the network/values actually drawn for a capacity map variant.

    When showing the capacity pie, the whole network is reduced to
    ``reeds_zone`` level (and SSSC is never drawn, per the caller's
    convention); otherwise the full-resolution network and values are used
    unchanged, including any SSSC markers.
    """
    if not show_capacity_pie:
        return n, bus_values, line_values, link_values, sssc_values, None

    zone_bus_values = aggregate_bus_values_to_zone(n, bus_values)
    zone_n, zone_line_values, zone_link_values = build_reeds_zone_capacity_network(n, line_values, link_values)
    return zone_n, zone_bus_values, zone_line_values, zone_link_values, None, n


def plot_capacity_map(
    n: pypsa.Network,
    bus_values: pd.DataFrame,
    line_values: pd.DataFrame,
    link_values: pd.DataFrame,
    regions: gpd.GeoDataFrame,
    bus_scale=1,
    line_scale=1,
    title=None,
    flow=None,
    line_colors="teal",
    link_colors="green",
    line_cmap="viridis",
    line_norm=None,
    sssc_values: pd.Series | None = None,
    show_capacity_pie: bool = True,
    original_n: pypsa.Network | None = None,
) -> tuple[plt.figure, plt.axes]:
    """Generic network plotting function for capacity pie charts at each node.

    ``n`` is the network actually drawn (buses/lines/links define the plotted
    topology) — for the zone-aggregated pie view this is the reduced
    zone-level network from ``build_reeds_zone_capacity_network``, not the
    full-resolution network. ``original_n`` (defaults to ``n``) is the
    full-resolution network used for the region/zone-boundary background and
    carrier color lookups, which need per-bus ``reeds_zone`` data.
    """
    original_n = original_n if original_n is not None else n
    fig, ax = plt.subplots(
        figsize=MAP_FIGSIZE,
        subplot_kw={"projection": ccrs.EqualEarth(original_n.buses.x.mean())},
    )
    fig.subplots_adjust(
        left=MAP_LAYOUT_LEFT_MARGIN,
        right=MAP_LAYOUT_RIGHT_MARGIN,
        bottom=MAP_LAYOUT_BOTTOM_MARGIN,
        top=MAP_LAYOUT_TOP_MARGIN,
    )
    map_boundaries = get_capacity_map_boundaries(regions, show_capacity_pie=show_capacity_pie)
    bus_colors, _ = get_capacity_map_legend_entries(original_n, bus_values)

    if isinstance(line_colors, str) and line_colors == "teal":
        line_colors = get_carrier_color(original_n, "AC", "teal")
    if isinstance(link_colors, str) and link_colors == "green":
        link_colors = get_transmission_link_colors(n)

    line_width = line_values / line_scale
    link_width = link_values / line_scale

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    draw_model_region_background(ax, regions, n=original_n)

    with plt.rc_context({"patch.linewidth": 0.1}):
        n.plot(
            bus_sizes=bus_values / bus_scale if show_capacity_pie else 0,
            bus_colors=bus_colors,
            bus_alpha=CAPACITY_MAP_BUS_ALPHA,
            line_widths=line_width,
            link_widths=0 if link_width.empty else link_width,
            line_colors=line_colors,
            link_colors=link_colors,
            ax=ax,
            margin=0.05,
            boundaries=map_boundaries,
            color_geomap=False,
            flow=flow,
            line_cmap=line_cmap,
            line_norm=line_norm,
            branch_components=get_plot_branch_components(n),
        )
    if not show_capacity_pie and sssc_values is not None and not sssc_values.empty:
        plot_line_x_sssc_markers(
            ax=ax,
            n=n,
            sssc_values=sssc_values,
        )

    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, pad=20)

    return fig, ax


def plot_demand_map(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    carriers: list[str],
    save: str,
    interconnect: str | None = None,
    **wildcards,
) -> None:
    """Plots map of network nodal demand."""
    # get data

    bus_values = get_demand_base(n).mul(1e-3)
    line_values = get_line_plot_values(n, "s_nom")
    link_values = get_transmission_link_values(n, "p_nom")

    # plot data
    title = create_title("Network Demand")
    bus_scale = get_bus_scale(interconnect) if interconnect else 1
    line_scale = get_line_scale(interconnect) if interconnect else 1

    fig, ax = plt.subplots(
        figsize=MAP_FIGSIZE,
        subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())},
    )
    fig.subplots_adjust(
        left=MAP_LAYOUT_LEFT_MARGIN,
        right=MAP_LAYOUT_RIGHT_MARGIN,
        bottom=MAP_LAYOUT_BOTTOM_MARGIN,
        top=MAP_LAYOUT_TOP_MARGIN,
    )
    map_boundaries = get_map_boundaries(regions, shrink=True)
    line_width = line_values / line_scale
    link_width = link_values / line_scale
    line_colors = get_carrier_color(n, "AC", "teal")
    link_colors = get_transmission_link_colors(n)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    draw_model_region_background(ax, regions, n=n)

    with plt.rc_context({"patch.linewidth": 0.1}):
        n.plot(
            bus_sizes=bus_values / bus_scale,
            # bus_colors=None,
            bus_alpha=CAPACITY_MAP_BUS_ALPHA,
            line_widths=line_width,
            link_widths=0 if link_width.empty else link_width,
            line_colors=line_colors,
            link_colors=link_colors,
            ax=ax,
            margin=0.05,
            boundaries=map_boundaries,
            color_geomap=False,
            branch_components=get_plot_branch_components(n),
        )

    add_demand_map_legends(
        ax=ax,
        n=n,
        bus_values=bus_values,
        bus_scale=bus_scale,
        line_values=line_values,
        link_values=link_values,
        line_scale=line_scale,
    )
    if not title:
        ax.set_title("Total Annual Demand (MW)", fontsize=TITLE_SIZE, pad=20)
    else:
        ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
    fig.savefig(save, dpi=600, bbox_inches='tight')
    plt.close()


def plot_base_capacity_map(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    carriers: list[str],
    save: str,
    interconnect: str | None = None,
    show_capacity_pie: bool = True,
    **wildcards,
) -> None:
    """Plots map of base network capacities."""

    bus_values = get_capacity_map_bus_values(n, carriers, "p_nom")
    line_values = get_line_plot_values(n, "s_nom")
    link_values = get_transmission_link_values(n, "p_nom")

    # plot data
    bus_scale = get_bus_scale(interconnect) if interconnect else 1
    line_scale = get_line_scale(interconnect) if interconnect else 1
    bus_colors, nice_names = get_capacity_map_legend_entries(n, bus_values)
    plot_n, plot_bus_values, plot_line_values, plot_link_values, plot_sssc_values, original_n = (
        resolve_capacity_map_plot_inputs(n, bus_values, line_values, link_values, None, show_capacity_pie)
    )

    fig, ax = plot_capacity_map(
        n=plot_n,
        bus_values=plot_bus_values,
        line_values=plot_line_values,
        link_values=plot_link_values,
        regions=regions,
        line_scale=line_scale,
        bus_scale=bus_scale,
        title=None,
        sssc_values=plot_sssc_values,
        show_capacity_pie=show_capacity_pie,
        original_n=original_n,
    )
    add_capacity_map_legends(
        ax=ax,
        n=plot_n,
        bus_values=plot_bus_values,
        bus_scale=bus_scale,
        bus_colors=bus_colors,
        nice_names=nice_names,
        line_values=plot_line_values,
        link_values=plot_link_values,
        line_scale=line_scale,
        show_dc_link=not n.links.empty and n.links.carrier.eq("DC").any(),
        sssc_values=plot_sssc_values,
        show_capacity_pie=show_capacity_pie,
    )
    fig.savefig(save, dpi=600, bbox_inches='tight')
    plt.close()


def plot_opt_capacity_map(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    carriers: list[str],
    save: str,
    interconnect: str | None = None,
    show_capacity_pie: bool = True,
    **wildcards,
) -> None:
    """Plots map of optimal network capacities."""
    bus_values = get_capacity_map_bus_values(n, carriers, "p_nom_opt")
    line_values = get_line_plot_values(n, "s_nom_opt")
    link_values = get_transmission_link_values(n, "p_nom_opt")

    # plot data
    bus_scale = get_bus_scale(interconnect) if interconnect else 1
    line_scale = get_line_scale(interconnect) if interconnect else 1
    bus_colors, nice_names = get_capacity_map_legend_entries(n, bus_values)
    sssc_values = get_line_x_sssc_plot_values(n, "sssc_nom_opt")
    plot_n, plot_bus_values, plot_line_values, plot_link_values, plot_sssc_values, original_n = (
        resolve_capacity_map_plot_inputs(n, bus_values, line_values, link_values, sssc_values, show_capacity_pie)
    )

    fig, ax = plot_capacity_map(
        n=plot_n,
        bus_values=plot_bus_values,
        line_values=plot_line_values,
        link_values=plot_link_values,
        regions=regions,
        line_scale=line_scale,
        bus_scale=bus_scale,
        title=None,
        sssc_values=plot_sssc_values,
        show_capacity_pie=show_capacity_pie,
        original_n=original_n,
    )
    add_capacity_map_legends(
        ax=ax,
        n=plot_n,
        bus_values=plot_bus_values,
        bus_scale=bus_scale,
        bus_colors=bus_colors,
        nice_names=nice_names,
        line_values=plot_line_values,
        link_values=plot_link_values,
        line_scale=line_scale,
        show_dc_link=not n.links.empty and n.links.carrier.eq("DC").any(),
        sssc_values=plot_sssc_values,
        show_capacity_pie=show_capacity_pie,
    )
    fig.savefig(save, dpi=600, bbox_inches='tight')
    plt.close()


def plot_new_capacity_map(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    carriers: list[str],
    save: str,
    interconnect: str | None = None,
    show_capacity_pie: bool = True,
    **wildcards,
) -> None:
    """Plots map of new capacity."""
    bus_pnom = get_capacity_map_bus_values(n, carriers, "p_nom")
    bus_pnom_opt = get_capacity_map_bus_values(n, carriers, "p_nom_opt")
    bus_values = bus_pnom_opt.sub(bus_pnom, fill_value=0)
    bus_values = bus_values[bus_values > 0]

    line_snom = get_line_plot_values(n, "s_nom")
    line_snom_opt = get_line_plot_values(n, "s_nom_opt")
    line_values = line_snom_opt - line_snom

    link_pnom = get_transmission_link_values(n, "p_nom")
    link_pnom_opt = get_transmission_link_values(n, "p_nom_opt")
    link_values = (link_pnom_opt - link_pnom).replace(to_replace={pd.NA: 0})

    # plot data
    bus_scale = get_bus_scale(interconnect) if interconnect else 1
    line_scale = get_line_scale(interconnect) if interconnect else 1
    bus_colors, nice_names = get_capacity_map_legend_entries(n, bus_values)
    sssc_values = get_line_x_sssc_new_plot_values(n)
    plot_n, plot_bus_values, plot_line_values, plot_link_values, plot_sssc_values, original_n = (
        resolve_capacity_map_plot_inputs(n, bus_values, line_values, link_values, sssc_values, show_capacity_pie)
    )

    fig, ax = plot_capacity_map(
        n=plot_n,
        bus_values=plot_bus_values,
        line_values=plot_line_values,
        link_values=plot_link_values,
        regions=regions,
        line_scale=line_scale,
        bus_scale=bus_scale,
        title=None,
        sssc_values=plot_sssc_values,
        show_capacity_pie=show_capacity_pie,
        original_n=original_n,
    )
    add_capacity_map_legends(
        ax=ax,
        n=plot_n,
        bus_values=plot_bus_values,
        bus_scale=bus_scale,
        bus_colors=bus_colors,
        nice_names=nice_names,
        line_values=plot_line_values,
        link_values=plot_link_values,
        line_scale=line_scale,
        show_dc_link=not n.links.empty and n.links.carrier.eq("DC").any(),
        sssc_values=plot_sssc_values,
        show_capacity_pie=show_capacity_pie,
    )
    fig.savefig(save, dpi=600, bbox_inches='tight')
    plt.close()


def plot_lmp_map(network: pypsa.Network, save: str, **wildcards):
    _, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(8, 8),
    )

    lmps = network.buses_t.marginal_price.mean()

    plt.hexbin(
        network.buses.x,
        network.buses.y,
        gridsize=40,
        C=lmps,
        cmap=plt.cm.bwr,
        zorder=3,
    )
    network.plot(
        ax=ax,
        line_widths=get_line_plot_values(network, fill_value=0.5),
        bus_sizes=0,
        branch_components=get_plot_branch_components(network),
    )

    cb = plt.colorbar(
        location="bottom",
        pad=0.01,
    )  # Adjust the pad value to move the color bar closer
    cb.set_label("LMP ($/MWh)")
    plt.title(create_title("Locational Marginal Price [$/MWh]", **wildcards))
    plt.tight_layout(
        rect=[0, 0, 1, 0.95],
    )  # Adjust the rect values to make the layout tighter
    plt.savefig(save, dpi=600)
    plt.close()


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_network_maps",
            case="2050_MidDmd_v1.30_DCNet_NoSSSC",
            ll="v1.30",
            opts="TCT-RPS-ERM-3h",
            sector="E",
        )
    configure_logging(snakemake)
    set_case_config(snakemake)

    # extract shared plotting files
    n = load_postprocess_network(snakemake.input.network)
    # if not n.links.empty and "p_nom_opt" in n.links.columns:
    #     drop_mask = n.links.carrier.eq("DC") & n.links.p_nom_opt.fillna(0).lt(1.0)
    #     if drop_mask.any():
    #         n.links.drop(index=n.links.index[drop_mask], inplace=True)
    onshore_regions = gpd.read_file(snakemake.input.regions_onshore)

    sanitize_carriers(n, snakemake.config)

    # mappers
    # carriers to plot
    configured_carriers = (
        snakemake.params.electricity["conventional_carriers"]
        + snakemake.params.electricity["renewable_carriers"]
        + snakemake.params.electricity["extendable_carriers"]["Generator"]
        + snakemake.params.electricity["extendable_carriers"]["StorageUnit"]
        + snakemake.params.electricity["extendable_carriers"]["Store"]
        + snakemake.params.electricity["extendable_carriers"]["Link"]
    )
    carriers = get_capacity_map_carriers(n, list(dict.fromkeys(configured_carriers)))
    interconnect = get_plot_interconnect(config=snakemake.config, wildcards=snakemake.wildcards)
    plot_wildcards = dict(snakemake.wildcards)
    plot_wildcards.pop("interconnect", None)

    # plotting theme
    sns.set_theme("paper", style="darkgrid")
    _apply_plot_font_theme()

    # create plots: each capacity map is rendered twice — once as a
    # reeds_zone-aggregated pie map, once as a full-resolution transmission map.
    for show_capacity_pie, output_suffix in ((True, "pie"), (False, "transmission")):
        plot_base_capacity_map(
            n,
            onshore_regions,
            carriers,
            snakemake.output[f"capacity_map_base_{output_suffix}.png"],
            interconnect=interconnect,
            show_capacity_pie=show_capacity_pie,
            **plot_wildcards,
        )
        plot_opt_capacity_map(
            n,
            onshore_regions,
            carriers,
            interconnect=interconnect,
            show_capacity_pie=show_capacity_pie,
            **plot_wildcards,
            save=snakemake.output[f"capacity_map_optimized_{output_suffix}.png"],
        )
        plot_new_capacity_map(
            n,
            onshore_regions,
            carriers,
            interconnect=interconnect,
            show_capacity_pie=show_capacity_pie,
            **plot_wildcards,
            save=snakemake.output[f"capacity_map_new_{output_suffix}.png"],
        )
    plot_demand_map(
        n,
        onshore_regions,
        carriers,
        snakemake.output["demand_map.pdf"],
        interconnect=interconnect,
        **plot_wildcards,
    )
    # plot_lmp_map(n, snakemake.output["lmp_map.pdf"], **snakemake.wildcards)
