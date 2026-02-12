import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from cartopy import crs as ccrs
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# Global Plotting Settings
TITLE_SIZE = 14
PROJECTION_CORRECTION = 1  # 0.48

# Interconnection definitions
INTERCONNECTIONS = {
    "Western": ["AZ", "CA", "CO", "ID", "MT", "NM", "NV", "OR", "UT", "WA", "WY"],
    "Texas": ["TX"],
    "Eastern": [
        "CT",
        "MA",
        "ME",
        "NH",
        "NY",
        "RI",
        "VT",
        "AL",
        "AR",
        "FL",
        "GA",
        "LA",
        "MS",
        "MO",
        "NC",
        "SC",
        "TN",
        "IL",
        "KY",
        "VA",
        "IA",
        "KS",
        "MN",
        "ND",
        "NE",
        "OK",
        "SD",
        "WI",
        "DE",
        "IN",
        "PA",
        "MD",
        "WV",
        "MI",
        "NJ",
        "OH",
    ],
}

interconnection_colors = {
    "Western": "#000000",  # Blue
    "Texas": "#000000",  # Orange
    "Eastern": "#000000",  # Green
}


def get_smart_legend_sizes(max_value, base_sizes):
    """
    Generate smart legend sizes based on the maximum value in the data.

    Parameters
    ----------
    max_value : float
        Maximum value in the data
    base_sizes : list
        Base size values to use as reference

    Returns
    -------
    list
        Adjusted size values for legend
    """
    if max_value <= 1:
        return base_sizes[:1]  # Return minimal legend if no significant values

    # Find the closest smaller size
    smaller_sizes = [s for s in base_sizes if s < max_value]

    if not smaller_sizes:
        # Max value is smaller than all base sizes, use only the smallest base size and max_value
        return [base_sizes[0], max_value]

    # Get the closest smaller size
    closest_smaller = smaller_sizes[-1]

    # Check if we should remove the closest smaller size
    if max_value < 1.5 * closest_smaller:
        # Remove the closest smaller size
        result_sizes = [s for s in base_sizes if s < closest_smaller]
        result_sizes.append(max_value)
    else:
        # Keep the closest smaller size
        result_sizes = smaller_sizes.copy()
        result_sizes.append(max_value)

    return result_sizes


def normalize_flow_by_capacity(flow_mean, capacities):
    """
    Normalize flow values by their corresponding capacities to get utilization rates.

    Parameters
    ----------
    flow_mean : pd.Series
        Mean flow values with MultiIndex (component, name)
    capacities : pd.Series
        Capacity values indexed by name

    Returns
    -------
    pd.Series
        Normalized flow values (utilization rates)
    """
    normalized_flow = flow_mean.copy()

    for idx in flow_mean.index:
        component, name = idx
        if name in capacities.index and capacities[name] > 0:
            # Normalize by capacity
            normalized_flow[idx] = flow_mean[idx] / capacities[name]
        else:
            # If no capacity or zero capacity, set to zero
            normalized_flow[idx] = 0

    return normalized_flow


def plot_elec_cap_network(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    save: str,
    title: str = "Electricity Network Capacities",
    line_max_extension: float = 20000.0,
    **wildcards,
) -> None:
    """
    Plots network map showing power generation, battery storage,
    and fuel cell capacities, with transmission connections.
    Links (HVDC) and lines (HVAC) are colored by expansion capacity (p_nom_opt - p_nom)
    using different color schemes to distinguish transmission types.
    Uses non-linear color mapping (fast then slow) starting from light colors.

    Calculates net flow for bidirectional transmission links to avoid
    drawing two arrows between the same pair of nodes.

    Flow values are normalized by capacity (s_nom_opt or p_nom_opt) to show utilization rates.

    For capacity display, only _fwd links are shown to avoid double-counting.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    # Get bus-level capacity data (Multi-indexed Series with (bus, carrier))
    bus_values = get_elec_capacities(n)

    # Get transmission link capacities - ONLY _fwd links to avoid double-counting
    trans_links = n.links[n.links.carrier.isin(["AC", "DC"])]
    trans_links_fwd = trans_links[trans_links.index.str.endswith("_fwd")]
    link_values = trans_links_fwd.p_nom_opt

    # Get transmission line capacities (lines are not bidirectional pairs)
    trans_lines = n.lines.copy()
    line_values = trans_lines.s_nom_opt

    # Calculate expansion capacities (new capacity built) - only for _fwd links
    link_expansion = trans_links_fwd.p_nom_opt - trans_links_fwd.p_nom
    line_expansion = trans_lines.s_nom_opt - trans_lines.s_nom

    # Log combined summary
    total_link_capacity = link_values.sum()
    total_line_capacity = line_values.sum()
    total_capacity = total_link_capacity + total_line_capacity
    total_link_expansion = link_expansion.sum()
    total_line_expansion = line_expansion.sum()

    logger.info(f"Total transmission capacity: {total_capacity:,.0f} MW")
    logger.info(f"  Links (HVDC) capacity: {total_link_capacity:,.0f} MW")
    logger.info(f"  Lines (HVAC) capacity: {total_line_capacity:,.0f} MW")
    logger.info(f"Total transmission expansion: {total_link_expansion + total_line_expansion:,.0f} MW")
    logger.info(f"  Links (HVDC) expansion: {total_link_expansion:,.0f} MW")
    logger.info(f"  Lines (HVAC) expansion: {total_line_expansion:,.0f} MW")

    # Calculate and log transmission losses
    # AC Lines (traditional AC transmission lines)
    if not n.lines_t.p0.empty:
        ac_line_total_loss = n.lines_t.loss.sum().sum()
        ac_line_total_flow = abs(n.lines_t.p0).sum().sum()
        if ac_line_total_flow > 0:
            ac_line_avg_loss = ac_line_total_loss / ac_line_total_flow
            logger.info(f"AC lines average loss: {ac_line_avg_loss * 100:.2f}%")
        else:
            logger.info("AC lines average loss: N/A (no flow)")

    # DC Transmission (DC links)
    dc_links = trans_links[trans_links.carrier == "DC"]
    if not dc_links.empty and not n.links_t.p0.empty:
        dc_link_indices = dc_links.index
        dc_link_p0 = n.links_t.p0[dc_link_indices]
        dc_link_p1 = n.links_t.p1[dc_link_indices]
        # Loss = p0 + p1 (since p1 is negative at receiving end)
        dc_link_loss = (dc_link_p0 + dc_link_p1).sum().sum()
        dc_link_total_flow = abs(dc_link_p0).sum().sum()
        if dc_link_total_flow > 0:
            dc_link_avg_loss = dc_link_loss / dc_link_total_flow
            logger.info(f"DC transmission average loss: {dc_link_avg_loss * 100:.2f}%")
        else:
            logger.info("DC transmission average loss: N/A (no flow)")

    # AC Transmission (AC links, if any)
    ac_links = trans_links[trans_links.carrier == "AC"]
    if not ac_links.empty and not n.links_t.p0.empty:
        ac_link_indices = ac_links.index
        ac_link_p0 = n.links_t.p0[ac_link_indices]
        ac_link_p1 = n.links_t.p1[ac_link_indices]
        ac_link_loss = (ac_link_p0 + ac_link_p1).sum().sum()
        ac_link_total_flow = abs(ac_link_p0).sum().sum()
        if ac_link_total_flow > 0:
            ac_link_avg_loss = ac_link_loss / ac_link_total_flow
            logger.info(f"AC links average loss: {ac_link_avg_loss * 100:.2f}%")
        else:
            logger.info("AC links average loss: N/A (no flow)")

    # Generate the plot using PyPSA method
    fig, ax = plt.subplots(
        figsize=(12, 10),
        subplot_kw={"projection": ccrs.EqualEarth(central_longitude=-96)},
    )

    # Plot settings
    interconnect = wildcards.get("interconnect", "usa")
    bus_scale = get_bus_scale(interconnect) * 1.5
    line_scale = get_line_scale(interconnect)

    # Calculate flow for transmission links and lines
    line_flow = n.lines_t.p0.copy()
    link_flow = n.links_t.p0[trans_links.index].copy()

    # Process line flows (lines are not bidirectional in the same way)
    line_flow_mean = line_flow.mean(axis=0)

    # Calculate net flow for bidirectional transmission links
    link_flow_mean = link_flow.mean(axis=0)

    # Calculate net flow by pairing forward and reverse links
    net_link_flow = {}
    processed_pairs = set()

    for link_idx in trans_links.index:
        # Skip if already processed as part of a pair
        if link_idx in processed_pairs:
            continue

        # Identify forward and reverse links
        if link_idx.endswith("_fwd"):
            # This is a forward link
            fwd_link = link_idx
            rev_link = link_idx.replace("_fwd", "_rev")
        elif link_idx.endswith("_rev"):
            # This is a reverse link
            rev_link = link_idx
            fwd_link = link_idx.replace("_rev", "_fwd")
        else:
            # Handle links without _fwd/_rev naming (if any)
            logger.warning(f"Transmission link {link_idx} doesn't follow expected naming convention")
            # Use the link as-is
            net_link_flow[link_idx] = link_flow_mean.get(link_idx, 0)
            processed_pairs.add(link_idx)
            continue

        # Calculate net flow (positive means flow in forward direction)
        if fwd_link in link_flow_mean.index and rev_link in link_flow_mean.index:
            net_flow_value = link_flow_mean[fwd_link] - link_flow_mean[rev_link]
            # Store net flow only for forward link
            net_link_flow[fwd_link] = net_flow_value
            # Mark both as processed
            processed_pairs.add(fwd_link)
            processed_pairs.add(rev_link)
        elif fwd_link in link_flow_mean.index:
            # Only forward link exists
            net_link_flow[fwd_link] = link_flow_mean[fwd_link]
            processed_pairs.add(fwd_link)
        elif rev_link in link_flow_mean.index:
            # Only reverse link exists - treat as negative flow in forward direction
            net_link_flow[rev_link] = -link_flow_mean[rev_link]
            processed_pairs.add(rev_link)

    # Convert net link flows to Series
    if net_link_flow:
        link_flow_series = pd.Series(net_link_flow)
    else:
        link_flow_series = pd.Series(dtype=float)

    # ===== NORMALIZE FLOWS BY CAPACITY =====
    # Normalize line flows by s_nom_opt
    line_flow_normalized = line_flow_mean.copy()
    for line_idx in line_flow_mean.index:
        if line_idx in line_values.index and line_values[line_idx] > 0:
            line_flow_normalized[line_idx] = line_flow_mean[line_idx] / line_values[line_idx]
        else:
            line_flow_normalized[line_idx] = 0

    # Normalize link flows by p_nom_opt
    link_flow_normalized = link_flow_series.copy()
    for link_idx in link_flow_series.index:
        if link_idx in link_values.index and link_values[link_idx] > 0:
            link_flow_normalized[link_idx] = link_flow_series[link_idx] / link_values[link_idx]
        else:
            link_flow_normalized[link_idx] = 0

    # Create MultiIndex for PyPSA plotting
    line_flow_normalized.index = pd.MultiIndex.from_product(
        [["Line"], line_flow_normalized.index],
        names=["component", "name"],
    )

    link_flow_normalized.index = pd.MultiIndex.from_product(
        [["Link"], link_flow_normalized.index],
        names=["component", "name"],
    )

    # Combine normalized flows
    flow_mean = pd.concat([line_flow_normalized, link_flow_normalized])

    # ========== New color scheme: start from light colors with non-linear gradient (fast then slow) ==========
    # HVAC (lines): Light Blue → Medium Blue → Deep Blue
    hvac_colors = ["#B3D9FF", "#5B9BD5", "#2E5C8A", "#1A3A5C"]  # Light Blue → Deep Blue
    cmap_hvac = LinearSegmentedColormap.from_list("hvac_expansion", hvac_colors, N=256)

    # HVDC (links): Light Red → Medium Red → Deep Red
    hvdc_colors = ["#FFB3B3", "#FF6B6B", "#CC0000", "#800000"]  # Light Red → Deep Red
    cmap_hvdc = LinearSegmentedColormap.from_list("hvdc_expansion", hvdc_colors, N=256)

    # Non-linear normalization: Use PowerNorm with gamma=0.5 for "fast then slow" effect
    # gamma < 1: Fast change at low values, slow change at high values
    line_exp_max = max(trans_lines.s_nom_opt - trans_lines.s_nom) if not trans_lines.empty else 0
    link_exp_max = max(trans_links.p_nom_opt - trans_links.p_nom)
    vmax = min(line_max_extension, max(line_exp_max, link_exp_max))
    vmin = 0
    norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)

    # Create color mapping for HVDC links (_fwd links only)
    link_color_dict = {}
    for idx, exp in link_expansion.items():
        link_color_dict[idx] = cmap_hvdc(norm(exp))

    # Create color mapping for HVAC lines
    line_color_dict = {}
    for idx, exp in line_expansion.items():
        line_color_dict[idx] = cmap_hvac(norm(exp))

    # Convert to Series for PyPSA plotting
    link_colors_series = pd.Series(link_color_dict)
    line_colors_series = pd.Series(line_color_dict)

    with plt.rc_context({"patch.linewidth": 0.0}):
        n.plot(
            bus_sizes=(bus_values / bus_scale) ** 1.0,
            bus_colors=get_color_palette(),
            bus_alpha=0.8,
            link_widths=(link_values / line_scale) ** 1.0 if not link_values.empty else 0,
            link_colors=link_colors_series,
            line_widths=(line_values / line_scale) ** 1.0 if not line_values.empty else 0,
            line_colors=line_colors_series,
            flow=(flow_mean.abs() * 500 / 0.7) ** 1.0 * np.sign(flow_mean),
            ax=ax,
            margin=0.2,
            color_geomap=True,
        )

    # Plot regions background
    regions.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="white",
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=1.0,
        alpha=1.0,
    )
    # ax.set_extent(regions.total_bounds[[0, 2, 1, 3]])
    bounds = regions.total_bounds  # [minx, miny, maxx, maxy]
    ax.set_extent([bounds[0] + 1.0, bounds[2] - 1.0, bounds[1], bounds[3] - 3.0])

    regions_states = regions.copy()
    regions_states["name"] = regions_states["name"].str.upper()

    # Draw interconnection boundaries instead of NERC region boundaries
    for interconnection_name, state_list in INTERCONNECTIONS.items():
        interconnection_gdf = regions_states[regions_states["name"].isin([s.upper() for s in state_list])]
        if interconnection_gdf.empty:
            continue

        merged_geom = unary_union(interconnection_gdf.geometry).buffer(0.0001).buffer(-0.0001)

        merged_gdf = gpd.GeoDataFrame(
            {"interconnection": [interconnection_name], "geometry": [merged_geom]},
            crs=regions_states.crs,
        )

        merged_gdf.boundary.plot(
            ax=ax,
            facecolor="none",
            edgecolor=interconnection_colors.get(interconnection_name, "black"),
            linewidth=1.0,
            aspect="equal",
            transform=ccrs.PlateCarree(),
            alpha=0.5,
            label=interconnection_name,
        )

    # ========== Add dual colorbars with non-linear scale ==========
    if line_expansion.max() > 1.0 or link_expansion.max() > 1.0:
        # Calculate positions for side-by-side colorbars
        # Original width: 0.775, split into two with gap
        colorbar_width = 0.375  # Each colorbar width (half of original minus gap)
        colorbar_height = 0.012
        left_margin = 0.125
        y_position = 0.2  # Single y position for both colorbars
        gap = 0.025  # Gap between two colorbars

        # HVAC colorbar (blue scheme) - Left side
        cbar_hvac_ax = fig.add_axes([left_margin, y_position, colorbar_width, colorbar_height])
        cb_hvac = ColorbarBase(cbar_hvac_ax, cmap=cmap_hvac, norm=norm, orientation="horizontal")
        cb_hvac.set_label("HVAC Expansion (GW)", fontsize=9, color="#2E5C8A")
        # Manually set ticks to reflect non-linear mapping
        tick_values = [0, vmax * 0.1, vmax * 0.3, vmax * 0.5, vmax * 0.7, vmax]
        cb_hvac.set_ticks(tick_values)
        cb_hvac.ax.set_xticklabels([f"{tick / 1000:.0f}" for tick in tick_values], fontsize=8)
        cb_hvac.ax.tick_params(colors="#2E5C8A")

        # HVDC colorbar (red scheme) - Right side
        cbar_hvdc_ax = fig.add_axes([left_margin + colorbar_width + gap, y_position, colorbar_width, colorbar_height])
        cb_hvdc = ColorbarBase(cbar_hvdc_ax, cmap=cmap_hvdc, norm=norm, orientation="horizontal")
        cb_hvdc.set_label("HVDC Expansion (GW)", fontsize=9, color="#CC0000")
        cb_hvdc.set_ticks(tick_values)
        cb_hvdc.ax.set_xticklabels([f"{tick / 1000:.0f}" for tick in tick_values], fontsize=8)
        cb_hvdc.ax.tick_params(colors="#CC0000")

    # Add legends
    add_elec_capacity_legends(ax, bus_scale, line_scale, n, bus_values, link_values, line_values)

    # Set title and layout
    # ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
    fig.savefig(save, dpi=600, bbox_inches="tight")
    plt.close()


def plot_gas_cap_network(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    save: str,
    title: str = "Gas Network Capacities",
    **wildcards,
) -> None:
    """
    Plots network map showing gas production capacities and gas pipeline capacities
    using PyPSA's native plotting method.

    Flow values are normalized by capacity (p_nom_opt) to show utilization rates.
    """
    fig, ax = plt.subplots(
        figsize=(12, 10),
        subplot_kw={"projection": ccrs.EqualEarth(central_longitude=-96)},
    )

    # Get gas production capacities
    bus_values = get_gas_capacities(n)

    # Get gas pipeline capacities
    link_values = get_gas_link_capacities(n)

    # Plot settings
    interconnect = wildcards.get("interconnect", "usa")
    bus_scale = get_bus_scale(interconnect)
    line_scale = get_line_scale(interconnect)

    link_flow = n.links_t.p0.loc[:, n.links[n.links.carrier == "gas pipeline"].index].copy()
    flow_mean = link_flow.mean(axis=0)

    # ===== NORMALIZE FLOWS BY CAPACITY =====
    flow_normalized = flow_mean.copy()
    for link_idx in flow_mean.index:
        if link_idx in link_values.index and link_values[link_idx] > 0:
            flow_normalized[link_idx] = flow_mean[link_idx] / link_values[link_idx]
        else:
            flow_normalized[link_idx] = 0

    # Create MultiIndex for PyPSA plotting
    flow_normalized.index = pd.MultiIndex.from_product(
        [["Link"], flow_normalized.index],
        names=["component", "name"],
    )

    # Use PyPSA's native plot method
    with plt.rc_context({"patch.linewidth": 0.0}):
        # Create a version of bus_values with absolute values for sizing
        bus_values_for_sizing = bus_values.abs()
        bus_values_for_sizing = bus_values_for_sizing[bus_values_for_sizing > 0]
        n.plot(
            bus_sizes=(bus_values_for_sizing / bus_scale) ** 1.0,
            bus_colors=get_gas_color_palette(),
            bus_alpha=0.8,
            link_widths=(link_values / line_scale) ** 1.0 if not link_values.empty else 0,
            line_widths=0,
            link_colors="#8B4513",
            flow=(flow_normalized.abs() * 500) ** 1.0 * np.sign(flow_normalized),
            ax=ax,
            margin=0.2,
            color_geomap=True,
        )

    # Plot regions background
    regions.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="white",
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=1.0,
        alpha=1.0,
    )
    # ax.set_extent(regions.total_bounds[[0, 2, 1, 3]])
    bounds = regions.total_bounds  # [minx, miny, maxx, maxy]
    ax.set_extent([bounds[0] + 1.0, bounds[2] - 1.0, bounds[1], bounds[3] - 3.0])

    regions_states = regions.copy()
    regions_states["name"] = regions_states["name"].str.upper()

    # Draw interconnection boundaries instead of NERC region boundaries
    for interconnection_name, state_list in INTERCONNECTIONS.items():
        interconnection_gdf = regions_states[regions_states["name"].isin([s.upper() for s in state_list])]
        if interconnection_gdf.empty:
            continue

        merged_geom = unary_union(interconnection_gdf.geometry).buffer(0.0001).buffer(-0.0001)

        merged_gdf = gpd.GeoDataFrame(
            {"interconnection": [interconnection_name], "geometry": [merged_geom]},
            crs=regions_states.crs,
        )

        merged_gdf.boundary.plot(
            ax=ax,
            facecolor="none",
            edgecolor=interconnection_colors.get(interconnection_name, "black"),
            linewidth=1.0,
            aspect="equal",
            transform=ccrs.PlateCarree(),
            alpha=0.5,
            label=interconnection_name,
        )

    # Add legends
    add_gas_capacity_legends(ax, bus_scale, line_scale, bus_values, link_values)

    # Set title and layout
    # ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
    fig.savefig(save, dpi=600, bbox_inches="tight")
    plt.close()


def plot_h2_cap_network(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    save: str,
    title: str = "Hydrogen Network Capacities",
    **wildcards,
) -> None:
    """
    Plots network map showing hydrogen production capacities (electrolysis, SMR, bioH2)
    and hydrogen pipeline capacities using PyPSA's native plotting method.

    This function calculates net flow between states to avoid drawing two arrows
    for bidirectional H2 pipelines.

    Flow values are normalized by capacity (p_nom_opt) to show utilization rates.

    For capacity display, only _fwd links are shown to avoid double-counting.
    """
    # Get hydrogen production capacities
    bus_values = get_h2_capacities(n)

    # Get hydrogen pipeline capacities - ONLY _fwd links to avoid double-counting
    h2_pipeline_carriers = ["h2 pipeline retrofit", "h2 pipeline new"]
    h2_pipeline_links_all = n.links[n.links.carrier.isin(h2_pipeline_carriers)]
    h2_pipeline_links_fwd = h2_pipeline_links_all[h2_pipeline_links_all.index.str.endswith("_fwd")]
    link_values = h2_pipeline_links_fwd.p_nom_opt

    # Check if there's any hydrogen data to plot
    has_production = not bus_values.empty and bus_values.sum() > 1
    has_pipelines = not link_values.empty and link_values.sum() > 1

    if not has_production and not has_pipelines:
        logger.warning("No hydrogen network data found - creating empty plot with message")
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.text(
            0.5,
            0.5,
            "No Hydrogen Network Data Available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=16,
            color="gray",
        )
        # ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
        ax.axis("off")
        fig.savefig(save, dpi=600, bbox_inches="tight")
        plt.close()
        return

    fig, ax = plt.subplots(
        figsize=(12, 10),
        subplot_kw={"projection": ccrs.EqualEarth(central_longitude=-96)},
    )

    # Get hydrogen pipeline color mapping - only for _fwd links
    link_color_mapping = {}
    h2_color = "#2ca02c"
    for idx in h2_pipeline_links_fwd.index:
        link_color_mapping[idx] = h2_color

    # Plot settings
    interconnect = wildcards.get("interconnect", "usa")
    bus_scale = get_bus_scale(interconnect)
    line_scale = get_line_scale(interconnect)

    # Calculate net flow for H2 pipelines only if pipelines exist
    if has_pipelines:
        h2_pipeline_links = h2_pipeline_links_all  # Use all links for flow calculation

        # Get all flows
        link_flow = n.links_t.p0.loc[:, h2_pipeline_links.index].copy()
        flow_mean = link_flow.mean(axis=0)

        # Calculate net flow by pairing forward and reverse links
        net_flow = {}
        processed_pairs = set()

        for link_idx in h2_pipeline_links.index:
            # Skip if already processed as part of a pair
            if link_idx in processed_pairs:
                continue

            # Identify forward and reverse links
            if link_idx.endswith("_fwd"):
                fwd_link = link_idx
                rev_link = link_idx.replace("_fwd", "_rev")
            elif link_idx.endswith("_rev"):
                rev_link = link_idx
                fwd_link = link_idx.replace("_rev", "_fwd")
            else:
                logger.warning(f"H2 pipeline link {link_idx} doesn't follow expected naming convention")
                continue

            # Calculate net flow
            if fwd_link in flow_mean.index and rev_link in flow_mean.index:
                net_flow_value = flow_mean[fwd_link] - flow_mean[rev_link]
                net_flow[fwd_link] = net_flow_value
                processed_pairs.add(fwd_link)
                processed_pairs.add(rev_link)
            elif fwd_link in flow_mean.index:
                net_flow[fwd_link] = flow_mean[fwd_link]
                processed_pairs.add(fwd_link)
            elif rev_link in flow_mean.index:
                net_flow[rev_link] = -flow_mean[rev_link]
                processed_pairs.add(rev_link)

        # Convert to Series
        if net_flow:
            flow_series = pd.Series(net_flow)
        else:
            flow_series = pd.Series(dtype=float)

        # Normalize flows by capacity
        flow_normalized = flow_series.copy()
        for link_idx in flow_series.index:
            if link_idx in link_values.index and link_values[link_idx] > 0:
                flow_normalized[link_idx] = flow_series[link_idx] / link_values[link_idx]
            else:
                flow_normalized[link_idx] = 0

        # Create MultiIndex for PyPSA plotting
        flow_normalized.index = pd.MultiIndex.from_product(
            [["Link"], flow_normalized.index],
            names=["component", "name"],
        )

        # Log summary
        total_capacity = link_values.sum()
        logger.info(f"Total hydrogen pipeline capacity: {total_capacity:,.0f} MW")
    else:
        flow_normalized = pd.Series(dtype=float)
        flow_normalized.index = pd.MultiIndex.from_tuples([], names=["component", "name"])

    # Use PyPSA's native plot method
    with plt.rc_context({"patch.linewidth": 0.0}):
        n.plot(
            bus_sizes=(bus_values / bus_scale) ** 1.0 if has_production else 0,
            bus_colors=get_h2_color_palette() if has_production else {},
            bus_alpha=0.8,
            link_widths=(link_values / line_scale) ** 1.0 if has_pipelines else 0,
            line_widths=0,
            link_colors=link_color_mapping if has_pipelines else {},
            flow=(flow_normalized.abs() * 500) ** 1.0 * np.sign(flow_normalized) if has_pipelines else 0,
            ax=ax,
            margin=0.2,
            color_geomap=True,
        )

    # Plot regions background
    regions.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="white",
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=1.0,
        alpha=1.0,
    )
    bounds = regions.total_bounds
    ax.set_extent([bounds[0] + 1.0, bounds[2] - 1.0, bounds[1], bounds[3] - 3.0])

    regions_states = regions.copy()
    regions_states["name"] = regions_states["name"].str.upper()

    # Draw interconnection boundaries instead of NERC region boundaries
    for interconnection_name, state_list in INTERCONNECTIONS.items():
        interconnection_gdf = regions_states[regions_states["name"].isin([s.upper() for s in state_list])]
        if interconnection_gdf.empty:
            continue

        merged_geom = unary_union(interconnection_gdf.geometry).buffer(0.0001).buffer(-0.0001)

        merged_gdf = gpd.GeoDataFrame(
            {"interconnection": [interconnection_name], "geometry": [merged_geom]},
            crs=regions_states.crs,
        )

        merged_gdf.boundary.plot(
            ax=ax,
            facecolor="none",
            edgecolor=interconnection_colors.get(interconnection_name, "black"),
            linewidth=1.0,
            aspect="equal",
            transform=ccrs.PlateCarree(),
            alpha=0.5,
            label=interconnection_name,
        )

    # Add legends only if there's data
    if has_production or has_pipelines:
        add_h2_capacity_legends(ax, bus_scale, line_scale, bus_values, link_values)

    # Set title and layout
    # ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
    fig.savefig(save, dpi=600, bbox_inches="tight")
    plt.close()


def plot_large_storage_cap_network(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    save: str,
    title: str = "Large Storage Energy Capacities",
    storage_eng_retro_factor_h2: float = 0.2551,
    storage_eng_retro_factor_acaes: float = 0.0093,
    **wildcards,
) -> None:
    """
    Plots network map showing storage energy capacities (MWh) for battery, PHS, gas storage, H2 storage, and LDES.
    Now includes detailed gas storage breakdown with retrofit/non-retrofit distinction.
    """
    fig, ax = plt.subplots(
        figsize=(12, 10),
        subplot_kw={"projection": ccrs.EqualEarth(central_longitude=-96)},
    )

    # Get storage energy capacities (MWh) with detailed breakdown
    bus_values = get_storage_capacities(n, storage_eng_retro_factor_h2, storage_eng_retro_factor_acaes)
    values_to_drop = ["acaes new", "acaes retrofit", "battery", "phs", "h2 storage retrofit", "tes"]
    bus_values = bus_values[~bus_values.index.get_level_values("carrier").isin(values_to_drop)]

    # Plot settings
    interconnect = wildcards.get("interconnect", "usa")
    bus_scale = get_bus_scale(interconnect) * 500  # Scale up since using energy capacity (MWh)

    # Use PyPSA's native plot method with absolute values for sizing
    with plt.rc_context({"patch.linewidth": 0.0}):
        # Create a version of bus_values with absolute values for sizing
        bus_values_for_sizing = bus_values.abs()

        n.plot(
            bus_sizes=(bus_values_for_sizing / bus_scale) ** 1.0,
            bus_colors=get_storage_color_palette(),
            bus_alpha=0.8,
            link_widths=0,  # No links for storage plot
            line_widths=0,  # No lines for storage plot
            line_colors="gray",
            ax=ax,
            margin=0.2,
            color_geomap=True,
        )

    # Plot regions background
    regions.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="white",
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=1.0,
        alpha=1.0,
    )
    # ax.set_extent(regions.total_bounds[[0, 2, 1, 3]])
    bounds = regions.total_bounds  # [minx, miny, maxx, maxy]
    ax.set_extent([bounds[0] + 1.0, bounds[2] - 1.0, bounds[1], bounds[3] - 3.0])

    regions_states = regions.copy()
    regions_states["name"] = regions_states["name"].str.upper()

    # Draw interconnection boundaries instead of NERC region boundaries
    for interconnection_name, state_list in INTERCONNECTIONS.items():
        interconnection_gdf = regions_states[regions_states["name"].isin([s.upper() for s in state_list])]
        if interconnection_gdf.empty:
            continue

        merged_geom = unary_union(interconnection_gdf.geometry).buffer(0.0001).buffer(-0.0001)

        merged_gdf = gpd.GeoDataFrame(
            {"interconnection": [interconnection_name], "geometry": [merged_geom]},
            crs=regions_states.crs,
        )

        merged_gdf.boundary.plot(
            ax=ax,
            facecolor="none",
            edgecolor=interconnection_colors.get(interconnection_name, "black"),
            linewidth=1.0,
            aspect="equal",
            transform=ccrs.PlateCarree(),
            alpha=0.5,
            label=interconnection_name,
        )

    # Add legends - pass the actual data
    add_storage_capacity_legends(ax, bus_scale, bus_values, "large")

    # Set title and layout
    # ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
    fig.savefig(save, dpi=600, bbox_inches="tight")
    plt.close()


def plot_small_storage_cap_network(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    save: str,
    title: str = "Small Storage Energy Capacities",
    storage_eng_retro_factor_h2: float = 0.2551,
    storage_eng_retro_factor_acaes: float = 0.0093,
    **wildcards,
) -> None:
    """
    Plots network map showing storage energy capacities (MWh) for battery, PHS, gas storage, H2 storage, and LDES.
    Now includes detailed gas storage breakdown with retrofit/non-retrofit distinction.
    """
    fig, ax = plt.subplots(
        figsize=(12, 10),
        subplot_kw={"projection": ccrs.EqualEarth(central_longitude=-96)},
    )

    # Get storage energy capacities (MWh) with detailed breakdown
    bus_values = get_storage_capacities(n, storage_eng_retro_factor_h2, storage_eng_retro_factor_acaes)
    values_small = ["acaes new", "acaes retrofit", "battery", "phs", "tes"]
    bus_values = bus_values[bus_values.index.get_level_values("carrier").isin(values_small)]

    # Plot settings
    interconnect = wildcards.get("interconnect", "usa")
    bus_scale = get_bus_scale(interconnect) / 2

    # Use PyPSA's native plot method with absolute values for sizing
    with plt.rc_context({"patch.linewidth": 0.0}):
        # Create a version of bus_values with absolute values for sizing
        bus_values_for_sizing = bus_values.abs()

        n.plot(
            bus_sizes=(bus_values_for_sizing / bus_scale) ** 1.0,
            bus_colors=get_storage_color_palette(),
            bus_alpha=0.8,
            link_widths=0,  # No links for storage plot
            line_widths=0,  # No lines for storage plot
            line_colors="gray",
            ax=ax,
            margin=0.2,
            color_geomap=True,
        )

    # Plot regions background
    regions.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="white",
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=1.0,
        alpha=1.0,
    )
    # ax.set_extent(regions.total_bounds[[0, 2, 1, 3]])
    bounds = regions.total_bounds  # [minx, miny, maxx, maxy]
    ax.set_extent([bounds[0] + 1.0, bounds[2] - 1.0, bounds[1], bounds[3] - 3.0])

    regions_states = regions.copy()
    regions_states["name"] = regions_states["name"].str.upper()

    # Draw interconnection boundaries instead of NERC region boundaries
    for interconnection_name, state_list in INTERCONNECTIONS.items():
        interconnection_gdf = regions_states[regions_states["name"].isin([s.upper() for s in state_list])]
        if interconnection_gdf.empty:
            continue

        merged_geom = unary_union(interconnection_gdf.geometry).buffer(0.0001).buffer(-0.0001)

        merged_gdf = gpd.GeoDataFrame(
            {"interconnection": [interconnection_name], "geometry": [merged_geom]},
            crs=regions_states.crs,
        )

        merged_gdf.boundary.plot(
            ax=ax,
            facecolor="none",
            edgecolor=interconnection_colors.get(interconnection_name, "black"),
            linewidth=1.0,
            aspect="equal",
            transform=ccrs.PlateCarree(),
            alpha=0.5,
            label=interconnection_name,
        )

    # Add legends - pass the actual data
    add_storage_capacity_legends(ax, bus_scale, bus_values, "small")

    # Set title and layout
    # ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
    fig.savefig(save, dpi=600, bbox_inches="tight")
    plt.close()


def plot_demand(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    save: str,
    title: str = "Annual Energy Demand by State",
    **wildcards,
) -> None:
    """
    Plots network map showing annual average electricity, gas, and hydrogen demand by state.
    Uses PyPSA's native plotting with pie charts showing different demand types.
    """
    fig, ax = plt.subplots(
        figsize=(12, 10),
        subplot_kw={"projection": ccrs.EqualEarth(central_longitude=-96)},
    )

    # Get demand data in PyPSA format
    bus_values = get_demand_capacities(n)

    if bus_values.empty:
        logger.warning("No demand data found")
        return

    # Plot settings
    interconnect = wildcards.get("interconnect", "usa")
    demand_scale = get_demand_scale(interconnect)

    # Use PyPSA's native plot method
    with plt.rc_context({"patch.linewidth": 0.0}):
        n.plot(
            bus_sizes=(bus_values / demand_scale) ** 1.0,
            bus_colors=get_demand_color_palette(),
            bus_alpha=0.8,
            link_widths=0,  # No links for demand plot
            line_widths=0,
            link_colors="gray",
            ax=ax,
            margin=0.2,
            color_geomap=True,
        )

    # Plot regions background
    regions.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="white",
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=1.0,
        alpha=1.0,
    )
    # ax.set_extent(regions.total_bounds[[0, 2, 1, 3]])
    bounds = regions.total_bounds  # [minx, miny, maxx, maxy]
    ax.set_extent([bounds[0] + 1.0, bounds[2] - 1.0, bounds[1], bounds[3] - 3.0])

    regions_states = regions.copy()
    regions_states["name"] = regions_states["name"].str.upper()

    # Draw interconnection boundaries instead of NERC region boundaries
    for interconnection_name, state_list in INTERCONNECTIONS.items():
        interconnection_gdf = regions_states[regions_states["name"].isin([s.upper() for s in state_list])]
        if interconnection_gdf.empty:
            continue

        merged_geom = unary_union(interconnection_gdf.geometry).buffer(0.0001).buffer(-0.0001)

        merged_gdf = gpd.GeoDataFrame(
            {"interconnection": [interconnection_name], "geometry": [merged_geom]},
            crs=regions_states.crs,
        )

        merged_gdf.boundary.plot(
            ax=ax,
            facecolor="none",
            edgecolor=interconnection_colors.get(interconnection_name, "black"),
            linewidth=1.0,
            aspect="equal",
            transform=ccrs.PlateCarree(),
            alpha=0.5,
            label=interconnection_name,
        )

    # Add legends
    add_demand_capacity_legends(ax, demand_scale, bus_values)

    # Set title and layout
    # ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
    fig.savefig(save, dpi=600, bbox_inches="tight")
    plt.close()


def get_bus_scale(interconnect: str) -> float:
    """Scales buses based on interconnect size."""
    if interconnect != "usa":
        return 1e5
    else:
        return 1e5


def get_line_scale(interconnect: str) -> float:
    """Scales lines based on interconnect size."""
    if interconnect != "usa":
        return 1e4
    else:
        return 6e3


def get_demand_scale(interconnect: str) -> float:
    """Get demand scaling factor based on interconnect size."""
    if interconnect != "usa":
        return 1e5
    else:
        return 3e5


def get_elec_capacities(n: pypsa.Network) -> pd.Series:
    """
    Extract electrical capacities in the format expected by PyPSA plot method.
    Returns Multi-indexed Series with (bus, carrier) as index.
    """
    capacities = []

    # 1. Generators (renewables and remaining conventional)
    gen_caps = n.generators.groupby(["bus", "carrier"])["p_nom_opt"].sum()

    for (bus, carrier), capacity in gen_caps.items():
        if capacity > 1.0:  # Filter small capacities
            # Merge wind technologies
            if carrier in ["onwind", "offwind_floating"]:
                carrier_name = "wind"
            else:
                carrier_name = carrier

            capacities.append(((bus, carrier_name), capacity))

    # 2. Links + CCS (excluding LDES since it needs special handling)
    power_carriers = {
        "OCGT",
        "CCGT",
        "CCGT-95CCS",
        "CCGT-97CCS",
        "coal",
        "coal-95CCS",
        "coal-99CCS",
        "biomass",
        "biomass-CCS",
        "h2 fuel cell",
        "h2 turbine",
        "dac",
    }

    power_links = n.links[n.links.carrier.isin(power_carriers)]

    if not power_links.empty:
        link_caps = power_links.groupby(
            [power_links["bus0"].map(n.buses["STATE"]), "carrier"],
        )["p_nom_opt"].sum()

        for (bus, carrier), capacity in link_caps.items():
            if capacity > 1.0:
                # Simplify carrier names
                if ("CCGT" in carrier or "OCGT" in carrier) and "CCS" not in carrier:
                    carrier_name = "gas"
                elif ("CCGT" in carrier or "OCGT" in carrier) and "CCS" in carrier:
                    carrier_name = "gas-cc"
                elif "coal" in carrier and "CCS" not in carrier:
                    carrier_name = "coal"
                elif "coal" in carrier and "CCS" in carrier:
                    carrier_name = "coal-cc"
                elif "biomass" in carrier and "CCS" not in carrier:
                    carrier_name = "biomass"
                elif "biomass" in carrier and "CCS" in carrier:
                    carrier_name = "biomass-cc"
                elif "fuel cell" in carrier:
                    carrier_name = "fuel cell"
                else:
                    carrier_name = carrier

                # Apply efficiency to get electrical output capacity
                efficiency = (
                    power_links[(power_links.bus1 == bus) & (power_links.carrier == carrier)]["efficiency"].iloc[0]
                    if carrier != "dac"
                    else 1.0
                )

                power_capacity = capacity * efficiency
                capacities.append(((bus, carrier_name), power_capacity))

    # 3. LDES discharge links specifically (these connect ldes bus to electrical bus)
    ldes_carriers = ["tes", "acaes retrofit", "acaes new"]
    for ldes_carrier in ldes_carriers:
        ldes_discharge_links = n.links[
            (n.links.carrier == ldes_carrier) & (n.links.index.str.contains("discharge", case=False, na=False))
        ]
        if not ldes_discharge_links.empty:
            # Group by state of the electrical bus (bus1 for discharge links)
            ldes_caps = ldes_discharge_links.groupby(
                ldes_discharge_links["bus1"].map(n.buses["STATE"]),
            )["p_nom_opt"].sum()

            for state, capacity in ldes_caps.items():
                if capacity > 1.0 and pd.notna(state):
                    # Apply efficiency to get electrical output capacity
                    efficiency = ldes_discharge_links[ldes_discharge_links["bus1"].map(n.buses["STATE"]) == state][
                        "efficiency"
                    ].iloc[0]
                    power_capacity = capacity * efficiency
                    capacities.append(((state, ldes_carrier), power_capacity))

    # 4. Storage Units
    storage_caps = n.storage_units.groupby(["bus", "carrier"])["p_nom_opt"].sum()

    for (bus, carrier), capacity in storage_caps.items():
        if capacity > 1.0:
            # Simplify storage naming
            if "battery" in carrier.lower():
                carrier_name = "battery"
            elif carrier == "PHS":
                carrier_name = "phs"
            else:
                carrier_name = carrier

            capacities.append(((bus, carrier_name), capacity))

    # Convert to Multi-indexed Series
    if capacities:
        capacity_dict = {}
        for (bus, carrier), capacity in capacities:
            key = (bus, carrier)
            if key in capacity_dict:
                capacity_dict[key] += capacity
            else:
                capacity_dict[key] = capacity

        bus_values = pd.Series(capacity_dict)
        bus_values.index = pd.MultiIndex.from_tuples(bus_values.index, names=["bus", "carrier"])
    else:
        bus_values = pd.Series(dtype=float)
        bus_values.index = pd.MultiIndex.from_tuples([], names=["bus", "carrier"])

    # Log summary by technology
    if not bus_values.empty:
        tech_summary = bus_values.groupby("carrier").sum().sort_values(ascending=False)
        logger.info("Electrical technology capacities (MW):")
        for tech, cap in tech_summary.items():
            logger.info(f"  {tech}: {cap:,.0f} MW")

    return bus_values


def get_gas_capacities(n: pypsa.Network) -> pd.Series:
    """
    Extract gas production capacities in PyPSA format, distinguishing
    between retained and retired capacities, plus bio-cc and methanation.
    Returns Multi-indexed Series with (bus, carrier) as index.
    """
    capacities = []

    # Extract gas production links (traditional)
    gas_production_carriers = {"gas production"}
    gas_production_links = n.links[n.links.carrier.isin(gas_production_carriers)]

    for idx, link in gas_production_links.iterrows():
        bus = n.buses.at[link["bus0"], "STATE"]
        retained_capacity = link["p_nom_opt"]
        retired_capacity = link["p_nom"] - link["p_nom_opt"]

        if retained_capacity > 1.0:
            capacities.append(((bus, "gas production retained"), retained_capacity))

        if retired_capacity > 1.0:
            capacities.append(((bus, "gas production retired"), -retired_capacity))

    # Extract gas bio-cc production links
    gas_biocc_carriers = {"gas bio-cc"}
    gas_biocc_links = n.links[n.links.carrier.isin(gas_biocc_carriers)]

    for idx, link in gas_biocc_links.iterrows():
        bus = n.buses.at[link["bus1"], "STATE"]  # Output bus (gas bus)
        capacity = link["p_nom_opt"]

        if capacity > 1.0:
            # Apply efficiency to get gas output capacity
            efficiency = link["efficiency"]
            gas_capacity = capacity * efficiency
            capacities.append(((bus, "gas bio-cc"), gas_capacity))

    # Extract gas methanation links
    gas_methanation_carriers = {"gas methanation"}
    gas_methanation_links = n.links[n.links.carrier.isin(gas_methanation_carriers)]

    for idx, link in gas_methanation_links.iterrows():
        bus = n.buses.at[link["bus1"], "STATE"]  # Output bus (gas bus)
        capacity = link["p_nom_opt"]

        if capacity > 1.0:
            # Apply efficiency to get gas output capacity
            efficiency = link["efficiency"]
            gas_capacity = capacity * efficiency
            capacities.append(((bus, "gas methanation"), gas_capacity))

    # Convert to Multi-indexed Series
    if capacities:
        capacity_dict = {}
        for (bus, carrier), capacity in capacities:
            key = (bus, carrier)
            if key in capacity_dict:
                capacity_dict[key] += capacity
            else:
                capacity_dict[key] = capacity

        bus_values = pd.Series(capacity_dict)
        bus_values.index = pd.MultiIndex.from_tuples(bus_values.index, names=["bus", "carrier"])
    else:
        bus_values = pd.Series(dtype=float)
        bus_values.index = pd.MultiIndex.from_tuples([], names=["bus", "carrier"])

    # Log summary by technology
    if not bus_values.empty:
        tech_summary = bus_values.groupby("carrier").sum().sort_values(ascending=False)
        logger.info("Gas production capacities (MW):")
        for tech, cap in tech_summary.items():
            logger.info(f"  {tech}: {cap:,.0f} MW")

    return bus_values


def get_h2_capacities(n: pypsa.Network) -> pd.Series:
    """
    Extract hydrogen production capacities in PyPSA format.
    Returns Multi-indexed Series with (bus, carrier) as index.
    This is essentially the same as the original get_nonelec_capacities function.
    """
    capacities = []

    # Extract hydrogen production links
    h2_production = {"h2 electrolysis", "h2 smr", "h2 smr-cc", "h2 bio", "h2 bio-cc"}
    h2_production_links = n.links[n.links.carrier.isin(h2_production)]

    for idx, link in h2_production_links.iterrows():
        bus = n.buses.at[link["bus0"], "STATE"]  # Input bus for hydrogen production
        carrier = link["carrier"]
        capacity = link["p_nom_opt"]

        if capacity > 1.0:
            # Apply efficiency to get hydrogen output capacity
            efficiency = link["efficiency"]
            h2_capacity = capacity * efficiency
            capacities.append(((bus, carrier), h2_capacity))

    # Convert to Multi-indexed Series
    if capacities:
        capacity_dict = {}
        for (bus, carrier), capacity in capacities:
            key = (bus, carrier)
            if key in capacity_dict:
                capacity_dict[key] += capacity
            else:
                capacity_dict[key] = capacity

        bus_values = pd.Series(capacity_dict)
        bus_values.index = pd.MultiIndex.from_tuples(bus_values.index, names=["bus", "carrier"])
    else:
        bus_values = pd.Series(dtype=float)
        bus_values.index = pd.MultiIndex.from_tuples([], names=["bus", "carrier"])

    # Log summary by technology
    if not bus_values.empty:
        tech_summary = bus_values.groupby("carrier").sum().sort_values(ascending=False)
        logger.info("Hydrogen production capacities (MW):")
        for tech, cap in tech_summary.items():
            logger.info(f"  {tech}: {cap:,.0f} MW")

    return bus_values


def get_gas_link_capacities(n: pypsa.Network) -> pd.Series:
    """
    Extract gas pipeline capacities.
    Returns Series indexed by link names.
    """
    gas_pipeline_carriers = ["gas pipeline"]
    gas_pipeline_links = n.links[n.links.carrier.isin(gas_pipeline_carriers)]

    link_values = gas_pipeline_links["p_nom_opt"] / 2

    # Log summary
    total_capacity = link_values.sum()
    logger.info(f"Total gas pipeline capacity: {total_capacity:,.0f} MW")

    return link_values


def get_h2_link_capacities(n: pypsa.Network) -> pd.Series:
    """
    Extract hydrogen pipeline capacities.
    Returns Series indexed by link names.
    """
    h2_pipeline_carriers = ["h2 pipeline retrofit", "h2 pipeline new"]
    h2_pipeline_links = n.links[n.links.carrier.isin(h2_pipeline_carriers)]

    link_values = h2_pipeline_links["p_nom_opt"] / 2

    # Log summary with detailed carrier breakdown
    total_capacity = link_values.sum()
    logger.info(f"Total hydrogen pipeline capacity: {total_capacity:,.0f} MW")

    # Summary by original carrier (detailed for logging)
    carrier_summary = h2_pipeline_links.groupby("carrier")["p_nom_opt"].sum() / 2
    logger.info("Hydrogen pipeline capacities by detailed type:")
    for carrier, cap in carrier_summary.items():
        logger.info(f"  {carrier}: {cap:,.0f} MW")

    return link_values


def get_storage_capacities(
    n: pypsa.Network, storage_eng_retro_factor_h2: float = 0.2551, storage_eng_retro_factor_acaes: float = 0.0093
) -> pd.Series:
    """
    Extract storage capacities in PyPSA format with detailed gas and H2 storage breakdown.
    Returns Multi-indexed Series with (state, carrier) as index containing energy capacities for plotting.

    For storage_units (battery, PHS):
    - Energy capacity = sum of (duration * discharge_capacity / discharge_efficiency)

    For stores (gas storage, h2 storage, ldes):
    - Energy capacity = sum of e_nom_opt * (e_max_pu - e_min_pu)
    - Gas storage is broken down by retrofit/non-retrofit and retained/retired/retrofitted status.
    - H2 storage is broken down by new and retrofit, with field type information.
    """
    capacities = []
    storage_unit_capacities = []
    store_capacities = []

    # 1. Process battery and PHS (storage_units)
    storage_units = n.storage_units.copy()
    if not storage_units.empty:
        storage_units["state"] = storage_units["bus"].map(n.buses["reeds_state"])
        units_grouped = storage_units.groupby(["state", "carrier"])

        for (state, carrier), group in units_grouped:
            if "battery" in carrier.lower() or "PHS" in carrier.upper():
                if "battery" in carrier.lower():
                    carrier_name = "battery"
                elif "PHS" in carrier.upper():
                    carrier_name = "phs"

                total_energy_capacity = 0
                detailed_info = []

                for idx, unit in group.iterrows():
                    if carrier_name == "battery":
                        duration = unit.get("max_hours", 1)
                        efficiency = unit.get("efficiency_dispatch", 0.9)
                    elif carrier_name == "phs":
                        duration = unit.get("max_hours", 10)
                        efficiency = unit.get("efficiency_dispatch", 0.8)
                    else:
                        continue

                    p_nom_opt = unit["p_nom_opt"]
                    energy_capacity = duration * p_nom_opt / efficiency
                    total_energy_capacity += energy_capacity

                    detailed_info.append(
                        {
                            "p_nom_opt": p_nom_opt,
                            "duration": duration,
                            "efficiency": efficiency,
                            "energy_capacity_mwh": energy_capacity,
                        }
                    )

                if total_energy_capacity > 1.0:
                    capacities.append(((state, carrier_name), total_energy_capacity))
                    storage_unit_capacities.append(
                        {
                            "state": state,
                            "carrier": carrier_name,
                            "energy_mwh": total_energy_capacity,
                            "type": "storage_unit",
                        }
                    )

    # 2. Process H2 storage separately with field type breakdown
    h2_stores = n.stores[n.stores.carrier.str.contains("h2 storage", case=False, na=False)]
    h2_retrofit_by_state = {}  # Track H2 retrofit storage for gas calculations

    if not h2_stores.empty:
        # Separate H2 storage by retrofit and new
        h2_retrofit_stores = h2_stores[h2_stores.carrier.str.contains("retrofit", case=False, na=False)]
        h2_new_stores = h2_stores[
            h2_stores.carrier.str.contains("new", case=False, na=False)
            | (~h2_stores.carrier.str.contains("retrofit", case=False, na=False))
        ]

        # Process H2 retrofit storage with field type details
        if not h2_retrofit_stores.empty:
            h2_retrofit_stores["state"] = h2_retrofit_stores["bus"].map(n.buses["reeds_state"])

            # Group by state and field type
            h2_retrofit_grouped = h2_retrofit_stores.groupby("state")

            for state, group in h2_retrofit_grouped:
                total_energy_capacity = 0

                for idx, store in group.iterrows():
                    e_nom = store["e_nom_opt"]
                    e_max_pu = store["e_max_pu"]
                    e_min_pu = store["e_min_pu"]
                    energy_contrib = e_nom * (e_max_pu - e_min_pu)
                    total_energy_capacity += energy_contrib

                if total_energy_capacity > 1.0:
                    # Use detailed carrier name for H2 retrofit
                    carrier_name = "h2 storage retrofit"
                    capacities.append(((state, carrier_name), total_energy_capacity))
                    store_capacities.append(
                        {
                            "state": state,
                            "carrier": carrier_name,
                            "energy_mwh": total_energy_capacity,
                            "type": "store",
                        }
                    )

                    # Track total H2 retrofit storage by state for gas calculations
                    if state not in h2_retrofit_by_state:
                        h2_retrofit_by_state[state] = 0
                    h2_retrofit_by_state[state] += total_energy_capacity

        # Process H2 new storage
        if not h2_new_stores.empty:
            h2_new_stores["state"] = h2_new_stores["bus"].map(n.buses["reeds_state"])

            h2_new_grouped = h2_new_stores.groupby("state")

            for state, group in h2_new_grouped:
                total_energy_capacity = 0

                for idx, store in group.iterrows():
                    e_nom = store["e_nom_opt"]
                    e_max_pu = store["e_max_pu"]
                    e_min_pu = store["e_min_pu"]
                    energy_contrib = e_nom * (e_max_pu - e_min_pu)
                    total_energy_capacity += energy_contrib

                if total_energy_capacity > 1.0:
                    carrier_name = "H2 Storage-New"

                    capacities.append(((state, carrier_name), total_energy_capacity))
                    store_capacities.append(
                        {
                            "state": state,
                            "carrier": carrier_name,
                            "energy_mwh": total_energy_capacity,
                            "type": "store",
                        }
                    )

    # 3. Process LDES separately
    ldes_carriers = ["tes", "acaes retrofit", "acaes new"]
    acaes_retrofit_by_state = {}  # Track ACAES retrofit storage for gas calculations
    for ldes_carrier in ldes_carriers:
        ldes_stores = n.stores[n.stores.carrier == ldes_carrier]

        if not ldes_stores.empty:
            ldes_stores["state"] = ldes_stores["bus"].map(n.buses["reeds_state"])
            ldes_grouped = ldes_stores.groupby("state")

            for state, group in ldes_grouped:
                total_energy_capacity = 0

                for idx, store in group.iterrows():
                    e_nom = store["e_nom_opt"]
                    e_max_pu = store["e_max_pu"]
                    e_min_pu = store["e_min_pu"]
                    energy_contrib = e_nom * (e_max_pu - e_min_pu)
                    total_energy_capacity += energy_contrib

                if total_energy_capacity > 1.0:
                    capacities.append(((state, ldes_carrier), total_energy_capacity))
                    store_capacities.append(
                        {
                            "state": state,
                            "carrier": ldes_carrier,
                            "energy_mwh": total_energy_capacity,
                            "type": "store",
                        }
                    )

                    # Track total ACAES retrofit storage by state for gas calculations
                    if ldes_carrier == "acaes retrofit":
                        if state not in acaes_retrofit_by_state:
                            acaes_retrofit_by_state[state] = 0
                        acaes_retrofit_by_state[state] += total_energy_capacity

    # 4. Process gas storage with detailed breakdown
    gas_stores = n.stores[n.stores.carrier.str.contains("gas storage", case=False, na=False)]

    if not gas_stores.empty:
        # Separate retrofit and non-retrofit stores
        retrofit_stores = gas_stores[
            gas_stores.carrier.str.contains("retrofit", case=False, na=False)
            & ~gas_stores.carrier.str.contains("nonretrofit", case=False, na=False)
        ]
        nonretrofit_stores = gas_stores[gas_stores.carrier.str.contains("nonretrofit", case=False, na=False)]

        # Process retrofitable storage
        if not retrofit_stores.empty:
            retrofit_stores["state"] = retrofit_stores["bus"].map(n.buses["reeds_state"])
            retrofit_grouped = retrofit_stores.groupby("state")

            for state, group in retrofit_grouped:
                gas_baseline_retrofit = 0
                gas_retained_retrofit = 0

                for idx, store in group.iterrows():
                    e_nom = store["e_nom"]
                    e_nom_opt = store["e_nom_opt"]
                    e_max_pu = store["e_max_pu"]
                    e_min_pu = store["e_min_pu"]

                    baseline_effective = e_nom * (e_max_pu - e_min_pu)
                    retained_effective = e_nom_opt * (e_max_pu - e_min_pu)

                    gas_baseline_retrofit += baseline_effective
                    gas_retained_retrofit += retained_effective

                # Calculate retrofitted capacity based on H2 storage and ACAES
                h2_retrofit_capacity = h2_retrofit_by_state.get(state, 0)
                acaes_retrofit_capacity = acaes_retrofit_by_state.get(state, 0)
                gas_retrofitted_to_h2 = h2_retrofit_capacity / storage_eng_retro_factor_h2
                gas_retrofitted_to_acaes = acaes_retrofit_capacity / storage_eng_retro_factor_acaes

                # Calculate retired capacity
                gas_retired_retrofit = (
                    gas_baseline_retrofit - gas_retained_retrofit - gas_retrofitted_to_h2 - gas_retrofitted_to_acaes
                )

                # Add to capacities with merged categories
                if gas_retained_retrofit > 1.0:
                    capacities.append(((state, "Gas Storage-Retained"), gas_retained_retrofit))
                if gas_retrofitted_to_h2 > 1.0:
                    capacities.append(((state, "Gas Storage-Retrofitted"), -gas_retrofitted_to_h2))
                if gas_retrofitted_to_acaes > 1.0:
                    capacities.append(((state, "gas storage (retrofitted to acaes)"), -gas_retrofitted_to_acaes))
                if gas_retired_retrofit > 1.0:
                    capacities.append(((state, "Gas Storage-Retired"), -gas_retired_retrofit))

        # Process non-retrofitable storage
        if not nonretrofit_stores.empty:
            nonretrofit_stores["state"] = nonretrofit_stores["bus"].map(n.buses["reeds_state"])
            nonretrofit_grouped = nonretrofit_stores.groupby("state")

            for state, group in nonretrofit_grouped:
                gas_baseline_nonretrofit = 0
                gas_retained_nonretrofit = 0

                for idx, store in group.iterrows():
                    e_nom = store["e_nom"]
                    e_nom_opt = store["e_nom_opt"]
                    e_max_pu = store["e_max_pu"]
                    e_min_pu = store["e_min_pu"]

                    baseline_effective = e_nom * (e_max_pu - e_min_pu)
                    retained_effective = e_nom_opt * (e_max_pu - e_min_pu)

                    gas_baseline_nonretrofit += baseline_effective
                    gas_retained_nonretrofit += retained_effective

                gas_retired_nonretrofit = gas_baseline_nonretrofit - gas_retained_nonretrofit

                # Add to capacities with merged categories
                if gas_retired_nonretrofit > 1.0:
                    capacities.append(((state, "Gas Storage-Retired"), -gas_retired_nonretrofit))
                if gas_retained_nonretrofit > 1.0:
                    capacities.append(((state, "Gas Storage-Retained"), gas_retained_nonretrofit))

    # Convert to Multi-indexed Series for plotting (energy capacities only)
    # Accumulate capacities for the same (state, carrier) key
    capacity_dict = {}
    for (state, carrier), capacity in capacities:
        key = (state, carrier)
        if key in capacity_dict:
            capacity_dict[key] += capacity  # Add to existing value
        else:
            capacity_dict[key] = capacity

    bus_values = pd.Series(capacity_dict)

    # Add this check to avoid errors if bus_values is empty
    if not bus_values.empty:
        bus_values.index = pd.MultiIndex.from_tuples(
            bus_values.index,
            names=["state", "carrier"],
        )

    return bus_values


def get_demand_capacities(n: pypsa.Network) -> pd.Series:
    """
    Extract demand data in PyPSA format for plotting.
    Returns Multi-indexed Series with (state, carrier) as index containing annual demand in GWh.
    """
    capacities = []

    # Process loads
    if not n.loads.empty and not n.loads_t.p.empty:
        for load_idx, load in n.loads.iterrows():
            if load_idx not in n.loads_t.p.columns:
                continue

            # Get the time series data for this load
            demand_series = n.loads_t.p[load_idx]

            # Calculate annual total energy
            annual_energy = demand_series.mean() * 8.76  # GWh

            # Get state from bus
            bus = load["bus"]
            if bus not in n.buses.index:
                continue

            state = n.buses.at[bus, "STATE"]
            carrier = load["carrier"]

            # Map carrier types to display names
            if carrier == "AC":
                carrier_name = "electricity"
            elif carrier == "gas":
                carrier_name = "gas"
            elif carrier == "h2":
                carrier_name = "hydrogen"
            elif carrier == "biomass":
                carrier_name = "biomass"
            else:
                continue  # Skip unknown carriers

            capacities.append(((state, carrier_name), annual_energy))

    # Convert to Multi-indexed Series
    if capacities:
        capacity_dict = {}
        for (state, carrier), capacity in capacities:
            key = (state, carrier)
            if key in capacity_dict:
                capacity_dict[key] += capacity
            else:
                capacity_dict[key] = capacity

        bus_values = pd.Series(capacity_dict)
        bus_values.index = pd.MultiIndex.from_tuples(bus_values.index, names=["state", "carrier"])
    else:
        bus_values = pd.Series(dtype=float)
        bus_values.index = pd.MultiIndex.from_tuples([], names=["state", "carrier"])

    return bus_values


def get_color_palette() -> dict:
    """Get color palette for electrical technologies."""
    return {
        "wind": "#c9e3c1",
        "solar": "#ffd700",
        "gas": "#8B4513",
        "gas-cc": "#D2B48C",
        "coal": "#8c564b",
        "coal-cc": "#5a5a5a",
        "biomass": "#ff7f0e",
        "biomass-cc": "#9e9518",
        "hydro": "#6BC5E6",
        "nuclear": "#b5260d",
        "battery": "#1f77b4",
        "phs": "#aec7e8",
        "fuel cell": "#ea048a",
        "h2 turbine": "#f5bfe7",
        "dac": "#9370DB",
        "tes": "#FF0001",
        "acaes retrofit": "#ff69b4",
        "acaes new": "#ff1493",
    }


def get_gas_color_palette() -> dict:
    """Get color palette for gas technologies."""
    return {
        "gas production retained": "#ff7f0e",
        "gas production retired": "#808080",
        "gas bio-cc": "#2d5016",  # Deep green for gas bio-cc
        "gas methanation": "#98fb98",  # Light green for gas methanation
    }


def get_h2_color_palette() -> dict:
    """Get color palette for hydrogen technologies."""
    return {
        "h2 electrolysis": "#7FD12C",
        "h2 smr": "#0000ff",
        "h2 smr-cc": "#00BFFF",
        "h2 bio": "#228B22",
        "h2 bio-cc": "#006400",
    }


def get_demand_color_palette() -> dict:
    """Get color palette for demand types."""
    return {
        "electricity": "#1f77b4",  # Blue
        "gas": "#ff7f0e",  # Orange
        "hydrogen": "#f5bfe7",  # Pink
        "biomass": "#006400",  # Green
    }


def get_h2_pipeline_color_mapping(n: pypsa.Network) -> dict:
    """
    Create color mapping for hydrogen pipeline links.
    Returns dict mapping link indices to colors.
    """
    h2_pipeline_carriers = ["h2 pipeline retrofit", "h2 pipeline new"]
    h2_pipeline_links = n.links[n.links.carrier.isin(h2_pipeline_carriers)]

    # Color for hydrogen pipelines (same for both retrofit and new)
    h2_color = "#2ca02c"

    # Create mapping from link index to color
    link_colors = {}
    for idx, link in h2_pipeline_links.iterrows():
        link_colors[idx] = h2_color

    return link_colors


def get_storage_color_palette() -> dict:
    """
    Returns color palette for storage technologies with merged gas storage categories.
    """
    return {
        "battery": "#1f77b4",
        "phs": "#aec7e8",
        "H2 Storage-New": "#98df8a",
        "h2 storage retrofit": "#2ca02c",
        "tes": "#FF0001",
        "acaes new": "#ff69b4",
        "acaes retrofit": "#ff1493",
        "Gas Storage-Retained": "#ff7f0e",
        "Gas Storage-Retired": "#808080",
        "Gas Storage-Retrofitted": "#ff9896",
        "gas storage (retrofitted to acaes)": "#DA70D6",
    }


def add_elec_capacity_legends(ax, bus_scale, line_scale, n, bus_values, link_values, line_values):
    """Add legends for electrical capacity plot with smart sizing - only show existing technologies."""
    legend_kwargs = {"loc": "upper left", "frameon": False}
    bus_scale *= PROJECTION_CORRECTION

    # Bus size legend with smart sizing
    if not bus_values.empty and bus_values.sum() > 1:
        max_bus_value = bus_values.groupby("bus").sum().max()
        base_bus_sizes = [10000, 30000, 100000, 300000, 1000000]
        bus_sizes = get_smart_legend_sizes(max_bus_value, base_bus_sizes)

        add_legend_circles(
            ax,
            [(s / bus_scale) ** 1.0 for s in bus_sizes],
            [f"{s / 1000:.0f} GW" for s in bus_sizes],
            legend_kw={"bbox_to_anchor": (1, 1), **legend_kwargs},
        )
        legend_y_offset = 0.75
    else:
        legend_y_offset = 1.0

    # Transmission size legend with smart sizing - only show if transmission exists
    total_transmission_capacity = link_values.sum() + line_values.sum()
    if total_transmission_capacity > 1:
        max_line_value = max(
            link_values.max() if not link_values.empty else 0,
            line_values.max() if not line_values.empty else 0,
        )
        base_line_sizes = [1000, 3000, 10000, 30000, 100000]
        line_sizes = get_smart_legend_sizes(max_line_value, base_line_sizes)

        add_legend_lines(
            ax,
            [(s / line_scale) ** 1.0 for s in line_sizes],
            [f"{s / 1000:.0f} GW" for s in line_sizes],
            legend_kw={"bbox_to_anchor": (1, legend_y_offset), **legend_kwargs},
        )
        legend_y_offset -= 0.25

    # Technology color legend - only show existing technologies
    if not bus_values.empty:
        colors = get_color_palette()
        existing_carriers = bus_values.groupby("carrier").sum()
        existing_carriers = existing_carriers[existing_carriers > 1].index.tolist()

        # Filter colors to only include existing technologies
        existing_colors = [colors[carrier] for carrier in existing_carriers if carrier in colors]
        existing_labels = [carrier for carrier in existing_carriers if carrier in colors]

        if existing_colors:
            add_legend_patches(
                ax,
                existing_colors,
                existing_labels,
                legend_kw={"bbox_to_anchor": (1, legend_y_offset), **legend_kwargs},
            )


def add_gas_capacity_legends(ax, bus_scale, line_scale, bus_values, link_values):
    """Add legends for gas capacity plot with smart sizing - only show existing technologies."""
    legend_kwargs = {"loc": "upper left", "frameon": False}
    bus_scale *= PROJECTION_CORRECTION

    # Bus size legend with smart sizing - only show if gas production exists
    if not bus_values.empty and bus_values.abs().sum() > 1:
        max_bus_value = bus_values.abs().groupby("bus").sum().max()
        base_bus_sizes = [3000, 10000, 30000, 100000, 300000]
        bus_sizes = get_smart_legend_sizes(max_bus_value, base_bus_sizes)

        add_legend_circles(
            ax,
            [(s / bus_scale) ** 1.0 for s in bus_sizes],
            [f"{s / 1000:.0f} GW" for s in bus_sizes],
            legend_kw={"bbox_to_anchor": (1, 1), **legend_kwargs},
        )
        legend_y_offset = 0.75
    else:
        legend_y_offset = 1.0

    # Line size legend with smart sizing - only show if gas pipelines exist
    if not link_values.empty and link_values.sum() > 1:
        max_line_value = link_values.max()
        base_line_sizes = [1000, 3000, 10000, 30000, 100000]
        line_sizes = get_smart_legend_sizes(max_line_value, base_line_sizes)

        add_legend_lines(
            ax,
            [(s / line_scale) ** 1.0 for s in line_sizes],
            [f"{s / 1000:.0f} GW" for s in line_sizes],
            legend_kw={"bbox_to_anchor": (1, legend_y_offset), **legend_kwargs},
        )
        legend_y_offset -= 0.25

    # Technology color legend for gas production - only existing types
    if not bus_values.empty:
        gas_colors = get_gas_color_palette()
        existing_carriers = bus_values.groupby("carrier").sum()
        existing_carriers = existing_carriers[existing_carriers.abs() > 1].index.tolist()

        existing_gas_colors = [gas_colors[carrier] for carrier in existing_carriers if carrier in gas_colors]
        existing_gas_labels = [carrier for carrier in existing_carriers if carrier in gas_colors]

        if existing_gas_colors:
            add_legend_patches(
                ax,
                existing_gas_colors,
                existing_gas_labels,
                legend_kw={"bbox_to_anchor": (1, legend_y_offset), **legend_kwargs},
            )


def add_h2_capacity_legends(ax, bus_scale, line_scale, bus_values, link_values):
    """Add legends for hydrogen capacity plot with smart sizing - only show existing technologies."""
    legend_kwargs = {"loc": "upper left", "frameon": False}
    bus_scale *= PROJECTION_CORRECTION

    # Bus size legend with smart sizing - only show if hydrogen production exists
    if not bus_values.empty and bus_values.sum() > 1:
        max_bus_value = bus_values.groupby("bus").sum().max()
        base_bus_sizes = [3000, 10000, 30000, 100000, 300000]
        bus_sizes = get_smart_legend_sizes(max_bus_value, base_bus_sizes)

        add_legend_circles(
            ax,
            [(s / bus_scale) ** 1.0 for s in bus_sizes],
            [f"{s / 1000:.0f} GW" for s in bus_sizes],
            legend_kw={"bbox_to_anchor": (1, 1), **legend_kwargs},
        )
        legend_y_offset = 0.75
    else:
        legend_y_offset = 1.0

    # Line size legend with smart sizing - only show if hydrogen pipelines exist
    if not link_values.empty and link_values.sum() > 1:
        max_line_value = link_values.max()
        base_line_sizes = [1000, 3000, 10000, 30000, 100000]
        line_sizes = get_smart_legend_sizes(max_line_value, base_line_sizes)

        add_legend_lines(
            ax,
            [(s / line_scale) ** 1.0 for s in line_sizes],
            [f"{s / 1000:.0f} GW" for s in line_sizes],
            legend_kw={"bbox_to_anchor": (1, legend_y_offset), **legend_kwargs},
        )
        legend_y_offset -= 0.25

    # Technology color legend for hydrogen production - only existing types
    if not bus_values.empty:
        h2_colors = get_h2_color_palette()
        existing_carriers = bus_values.groupby("carrier").sum()
        existing_carriers = existing_carriers[existing_carriers > 1].index.tolist()

        existing_h2_colors = [h2_colors[carrier] for carrier in existing_carriers if carrier in h2_colors]
        existing_h2_labels = [carrier for carrier in existing_carriers if carrier in h2_colors]

        if existing_h2_colors:
            add_legend_patches(
                ax,
                existing_h2_colors,
                existing_h2_labels,
                legend_kw={"bbox_to_anchor": (1, legend_y_offset), **legend_kwargs},
            )


def add_storage_capacity_legends(ax, bus_scale, bus_values, size):
    """Add legends for storage energy capacity plot with smart sizing - only show existing technologies."""
    legend_kwargs = {"loc": "upper left", "frameon": False}
    bus_scale *= PROJECTION_CORRECTION

    # Bus size legend with smart sizing - only show if storage exists
    if not bus_values.empty and bus_values.abs().sum() > 1:
        max_bus_value = bus_values.abs().groupby("state").sum().max()

        if size == "large":
            base_bus_sizes = [1000000, 3000000, 10000000, 30000000, 100000000, 300000000]
            bus_sizes = get_smart_legend_sizes(max_bus_value, base_bus_sizes)
            add_legend_circles(
                ax,
                [(s / bus_scale) ** 1.0 for s in bus_sizes],
                [f"{s / 1000000:.0f} TWh" for s in bus_sizes],
                legend_kw={"bbox_to_anchor": (1, 1), **legend_kwargs},
            )
        else:
            base_bus_sizes = [1000, 3000, 10000, 30000, 100000, 300000]
            bus_sizes = get_smart_legend_sizes(max_bus_value, base_bus_sizes)
            add_legend_circles(
                ax,
                [(s / bus_scale) ** 1.0 for s in bus_sizes],
                [f"{s / 1000:.0f} GWh" for s in bus_sizes],
                legend_kw={"bbox_to_anchor": (1, 1), **legend_kwargs},
            )

    # Technology color legend - only show existing storage technologies
    colors = get_storage_color_palette()
    existing_carriers = bus_values.groupby("carrier").sum()
    existing_carriers = existing_carriers[existing_carriers.abs() > 1].index.tolist()

    # Create legend handles with appropriate styling
    legend_handles = []
    legend_labels = []

    from matplotlib.patches import Rectangle

    for carrier in existing_carriers:
        if carrier in colors:
            # Create a patch for the legend
            handle = Rectangle(
                (0, 0),
                1,
                1,
                facecolor=colors[carrier],
                edgecolor="black",
                linewidth=0.5,
            )
            legend_handles.append(handle)

            # Format label for better readability
            label = carrier.replace("gas storage ", "Gas Storage ").replace("(", "\n(")
            legend_labels.append(label)

    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(1.0, 0.65),
            frameon=False,
        )


def add_demand_capacity_legends(ax, bus_scale, bus_values):
    """Add legends for demand capacity plot with smart sizing - only show existing demand types."""
    legend_kwargs = {"loc": "upper left", "frameon": False}
    bus_scale *= PROJECTION_CORRECTION

    # Bus size legend with smart sizing - only show if demand exists
    if not bus_values.empty and bus_values.sum() > 1:
        max_bus_value = bus_values.groupby("state").sum().max()
        base_bus_sizes = [10000, 30000, 100000, 300000, 1000000]
        bus_sizes = get_smart_legend_sizes(max_bus_value, base_bus_sizes)

        add_legend_circles(
            ax,
            [(s / bus_scale) ** 1.0 for s in bus_sizes],
            [f"{s / 1000:.0f} TWh" for s in bus_sizes],
            legend_kw={"bbox_to_anchor": (1, 1), **legend_kwargs},
        )

    # Technology color legend - only show existing demand types
    if not bus_values.empty:
        colors = get_demand_color_palette()
        existing_carriers = bus_values.groupby("carrier").sum()
        existing_carriers = existing_carriers[existing_carriers > 1].index.tolist()

        # Filter colors to only include existing demand types
        existing_colors = [colors[carrier] for carrier in existing_carriers if carrier in colors]
        existing_labels = [carrier.title() for carrier in existing_carriers if carrier in colors]

        if existing_colors:
            add_legend_patches(
                ax,
                existing_colors,
                existing_labels,
                legend_kw={"bbox_to_anchor": (1, 0.7), **legend_kwargs},
            )


def get_existing_pipeline_types(n, link_values):
    """Helper function to get existing pipeline types from network."""
    if link_values.empty:
        return []

    pipeline_carriers = ["gas pipeline", "h2 pipeline retrofit", "h2 pipeline new"]
    pipeline_links = n.links[n.links.carrier.isin(pipeline_carriers)]

    existing_types = set()
    for idx, link in pipeline_links.iterrows():
        if link["p_nom_opt"] > 1:  # Only count significant capacities
            if link["carrier"] == "gas pipeline":
                existing_types.add("gas pipeline")
            elif link["carrier"] in ["h2 pipeline retrofit", "h2 pipeline new"]:
                existing_types.add("h2 pipeline")

    return list(existing_types)


def create_title(base_title: str, **wildcards) -> str:
    """Create title with wildcard information."""
    if wildcards:
        wildcard_str = "_".join([f"{k}={v}" for k, v in wildcards.items() if v])
        if wildcard_str:
            return f"{base_title} ({wildcard_str})"
    return base_title


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_simpsec_network",
            case="HighE_new_h2storage_tes",
            transmission_network="reeds",
        )

    # Import additional required modules for logging
    from _helpers import configure_logging
    from add_electricity import sanitize_carriers

    configure_logging(snakemake)

    # Load network and regions
    n = pypsa.Network(snakemake.input.network)
    regions_onshore = gpd.read_file(snakemake.input.regions_onshore)

    # Sanitize carriers to ensure consistent naming
    sanitize_carriers(n, snakemake.config)

    # Extract wildcards for title
    wildcards = dict(snakemake.wildcards)

    # Get natural gas options for storage retrofit factor
    ng_options = snakemake.params.sector["natural_gas"]
    storage_eng_retro_factor_h2 = ng_options.get("storage_eng_retro_factor_h2", 0.2551)
    storage_eng_retro_factor_acaes = ng_options.get("storage_eng_retro_factor_acaes", 0.0093)

    # Create the electrical capacity plot
    plot_elec_cap_network(
        n=n,
        regions=regions_onshore,
        save=snakemake.output.elec_cap_network,
        title="Electricity Network Capacities",
        line_max_extension=float(snakemake.params.line_max_extension),
        **wildcards,
    )

    # Create the gas capacity plot
    plot_gas_cap_network(
        n=n,
        regions=regions_onshore,
        save=snakemake.output.gas_cap_network,
        title="Gas Network Capacities",
        **wildcards,
    )

    # Create the hydrogen capacity plot
    plot_h2_cap_network(
        n=n,
        regions=regions_onshore,
        save=snakemake.output.h2_cap_network,
        title="Hydrogen Network Capacities",
        **wildcards,
    )

    # Create the storage capacity plot with enhanced breakdown
    plot_large_storage_cap_network(
        n=n,
        regions=regions_onshore,
        save=snakemake.output.large_storage_cap_network,
        title="Underground Energy Storage Capacities",
        storage_eng_retro_factor_h2=storage_eng_retro_factor_h2,
        storage_eng_retro_factor_acaes=storage_eng_retro_factor_acaes,
        **wildcards,
    )

    plot_small_storage_cap_network(
        n=n,
        regions=regions_onshore,
        save=snakemake.output.small_storage_cap_network,
        title="Small Storage Energy Capacities",
        storage_eng_retro_factor_h2=storage_eng_retro_factor_h2,
        storage_eng_retro_factor_acaes=storage_eng_retro_factor_acaes,
        **wildcards,
    )

    # Create the demand plot
    plot_demand(
        n=n,
        regions=regions_onshore,
        save=snakemake.output.demand_cap_network,
        title="Annual Energy Demand by State",
        **wildcards,
    )
