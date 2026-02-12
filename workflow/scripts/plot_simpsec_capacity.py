"""
Plots capacity analysis for SimpSec scenarios.

Creates stacked bar charts for:
1. Power generation and hydrogen production capacities (GW)
2. Transmission capacities × distance (GW×1000km)
3. Storage energy capacities (TWh)
4. Storage charge/discharge power capacities (GW)
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging
from plot_simpsec_network import (
    get_color_palette,
    get_elec_capacities,
    get_gas_color_palette,
    get_h2_capacities,
    get_h2_color_palette,
    get_storage_capacities,
    get_storage_color_palette,
)

logger = logging.getLogger(__name__)

# Global plotting settings
TITLE_SIZE = 16
FIG_WIDTH = 12
FIG_HEIGHT = 8


def get_transport_capacities(n: pypsa.Network, pipeline_retro_factor) -> dict:
    """
    Extract transport capacities and distances for different infrastructure types.
    Returns dict with capacity × distance data in GW×1000km.
    """
    transport_data = {}

    # Add AC/DC links for electrical transport
    trans_links = n.links[n.links.carrier.isin(["AC", "DC"])]
    link_existing = (trans_links["p_nom"] * trans_links["length"]).sum() / 1e6 / 2
    link_new = ((trans_links["p_nom_opt"] - trans_links["p_nom"]) * trans_links["length"]).sum() / 1e6 / 2

    # Add AC lines for electrical transport
    trans_lines = n.lines.copy()
    line_existing = (trans_lines["s_nom"] * trans_lines["length"]).sum() / 1e6
    line_new = ((trans_lines["s_nom_opt"] - trans_lines["s_nom"]) * trans_lines["length"]).sum() / 1e6

    # Combine links and lines
    transport_data["Electrical (Existing)"] = link_existing + line_existing
    transport_data["Electrical (New)"] = link_new + line_new

    # Hydrogen pipelines
    h2_pipeline_carriers = ["h2 pipeline retrofit", "h2 pipeline new"]
    h2_links = n.links[n.links.carrier.isin(h2_pipeline_carriers)]

    retrofit_links = h2_links[h2_links.carrier == "h2 pipeline retrofit"]
    new_links = h2_links[h2_links.carrier == "h2 pipeline new"]
    h2_retrofit = (retrofit_links["p_nom_opt"] * retrofit_links["length"]).sum() / 1e6 / 2
    h2_new = (new_links["p_nom_opt"] * new_links["length"]).sum() / 1e6 / 2

    transport_data["Hydrogen (Retrofit)"] = h2_retrofit
    transport_data["Hydrogen (New)"] = h2_new

    # Natural gas pipelines
    gas_pipeline_carriers = ["gas pipeline"]
    gas_links = n.links[n.links.carrier.isin(gas_pipeline_carriers)]

    # Calculate baseline capacity (original capacity)
    gas_baseline = (gas_links["p_nom"] * gas_links["length"]).sum() / 1e6 / 2

    # Calculate retained capacity (optimized capacity)
    gas_retained = (gas_links["p_nom_opt"] * gas_links["length"]).sum() / 1e6 / 2
    transport_data["Gas (Retained)"] = gas_retained

    gas_retrofitted = -h2_retrofit / pipeline_retro_factor
    transport_data["Gas (Retrofitted to H2)"] = gas_retrofitted

    transport_data["Gas (Retired)"] = -(gas_baseline - gas_retained + gas_retrofitted)

    return transport_data


def get_storage_energy_capacities(
    n: pypsa.Network, storage_eng_retro_factor_h2, storage_eng_retro_factor_acaes
) -> dict:
    """
    Extract storage energy capacities with retrofit/non-retrofit distinction.
    Returns dict with energy capacity data in TWh.
    """
    # Get detailed storage capacities from network function
    storage_caps = get_storage_capacities(n, storage_eng_retro_factor_h2, storage_eng_retro_factor_acaes)

    if storage_caps.empty:
        return {}

    # Convert to dictionary format for plotting, aggregate by carrier and convert to TWh
    storage_data = {}

    # Group by carrier and sum capacities, convert MWh to TWh
    carrier_totals = storage_caps.groupby("carrier").sum() / 1e6

    for carrier, capacity in carrier_totals.items():
        if abs(capacity) > 0.001:  # Only include significant capacities (0.001 TWh = 1 GWh threshold)
            # Map carrier names to display names for consistency with plotting
            if carrier == "battery":
                display_name = "Battery"
            elif carrier == "phs":
                display_name = "PHS"
            elif carrier == "h2 storage retrofit":
                display_name = "H2 Storage (Retrofitted from Gas)"
            elif carrier == "h2 storage new":
                display_name = "H2 Storage (New)"
            elif carrier == "tes":
                display_name = "TES"
            elif carrier == "acaes retrofit":
                display_name = "ACAES Retrofit"
            elif carrier == "acaes new":
                display_name = "ACAES New"
            elif carrier == "gas storage retrofitable (retained)":
                display_name = "Gas Storage Retrofitable (Retained)"
            elif carrier == "gas storage nonretrofitable (retained)":
                display_name = "Gas Storage Nonretrofitable (Retained)"
            elif carrier == "gas storage retrofitable (retrofitted to h2)":
                display_name = "Gas Storage Retrofitable (Retrofitted to H2)"
            elif carrier == "gas storage retrofitable (retrofitted to acaes)":
                display_name = "Gas Storage Retrofitable (Retrofitted to ACAES)"
            elif carrier == "gas storage retrofitable (retired)":
                display_name = "Gas Storage Retrofitable (Retired)"
            elif carrier == "gas storage nonretrofitable (retired)":
                display_name = "Gas Storage Nonretrofitable (Retired)"
            else:
                # Fallback for any other carriers
                display_name = carrier.replace("_", " ").title()

            storage_data[display_name] = capacity

    # Log summary
    logger.info("Storage energy capacity summary for plotting (TWh):")
    for tech, cap in storage_data.items():
        sign = "+" if cap >= 0 else ""
        logger.info(f"  {tech}: {sign}{cap:.3f} TWh")

    return storage_data


def get_storage_power_capacities(n: pypsa.Network, storage_pow_retro_factor_h2) -> tuple[dict, dict]:
    """
    Extract storage charge and discharge power capacities with gas storage breakdown by retrofit/non-retrofit types.
    Returns tuple of (charge_data, discharge_data) in GW.
    """
    charge_data = {}
    discharge_data = {}

    # Storage units (battery, PHS) - unchanged
    if not n.storage_units.empty:
        storage_units_grouped = n.storage_units.groupby("carrier")

        for carrier, group in storage_units_grouped:
            if "battery" in carrier.lower():
                carrier_name = "Battery"
            elif carrier.upper() == "PHS":
                carrier_name = "PHS"
            else:
                continue

            total_capacity = group["p_nom_opt"].sum() / 1000  # Convert MW to GW
            if total_capacity > 0.001:
                charge_data[carrier_name] = charge_data.get(carrier_name, 0) + total_capacity
                discharge_data[carrier_name] = discharge_data.get(carrier_name, 0) + total_capacity

    # 2. Hydrogen Storage (new/retrofit) - Refactored Logic
    h2_stores = n.stores[n.stores.carrier.str.contains("h2 storage", case=False, na=False)]
    if not h2_stores.empty:
        # Separate into 'new' and 'retrofit' based on keywords in the carrier name
        new_h2_stores = h2_stores[h2_stores.carrier.str.contains("new", case=False, na=False)]
        retrofit_h2_stores = h2_stores[h2_stores.carrier.str.contains("retrofit", case=False, na=False)]

        # Process New H2 Storage
        if not new_h2_stores.empty:
            total_charge = 0
            total_discharge = 0
            for _, store in new_h2_stores.iterrows():
                charge_links = n.links[
                    (n.links["bus1"] == store["bus"]) & n.links.index.str.contains("charge", case=False, na=False)
                ]
                total_charge += charge_links["p_nom_opt"].sum()
                discharge_links = n.links[
                    (n.links["bus0"] == store["bus"])
                    & n.links.index.str.contains(
                        "discharge",
                        case=False,
                        na=False,
                    )
                ]
                total_discharge += discharge_links["p_nom_opt"].sum()

            if total_charge > 1000:  # in MW
                charge_data["H2 Storage New"] = total_charge / 1000
            if total_discharge > 1000:
                discharge_data["H2 Storage New"] = total_discharge / 1000

        # Process Retrofit H2 Storage
        if not retrofit_h2_stores.empty:
            total_charge = 0
            total_discharge = 0
            for _, store in retrofit_h2_stores.iterrows():
                charge_links = n.links[
                    (n.links["bus1"] == store["bus"]) & n.links.index.str.contains("charge", case=False, na=False)
                ]
                total_charge += charge_links["p_nom_opt"].sum()
                discharge_links = n.links[
                    (n.links["bus0"] == store["bus"])
                    & n.links.index.str.contains(
                        "discharge",
                        case=False,
                        na=False,
                    )
                ]
                total_discharge += discharge_links["p_nom_opt"].sum()

            if total_charge > 1000:  # in MW
                charge_data["H2 Storage Retrofit"] = total_charge / 1000
            if total_discharge > 1000:
                discharge_data["H2 Storage Retrofit"] = total_discharge / 1000

    # 3. LDES (Long-Duration Energy Storage) - Separated Logic
    ldes_carriers = ["tes", "acaes retrofit", "acaes new"]
    for ldes_carrier in ldes_carriers:
        ldes_stores = n.stores[n.stores.carrier == ldes_carrier]
        if not ldes_stores.empty:
            total_charge = 0
            total_discharge = 0
            for _, store in ldes_stores.iterrows():
                charge_links = n.links[
                    (n.links["bus1"] == store["bus"]) & n.links.index.str.contains("charge", case=False, na=False)
                ]
                total_charge += charge_links["p_nom_opt"].sum()
                discharge_links = n.links[
                    (n.links["bus0"] == store["bus"]) & n.links.index.str.contains("discharge", case=False, na=False)
                ]
                total_discharge += discharge_links["p_nom_opt"].sum()

            if ldes_carrier == "tes":
                nice_name = "TES"
            elif ldes_carrier == "acaes retrofit":
                nice_name = "ACAES Retrofit"
            elif ldes_carrier == "acaes new":
                nice_name = "ACAES New"
            if total_charge > 1000:  # in MW
                charge_data[nice_name] = total_charge / 1000
            if total_discharge > 1000:
                discharge_data[nice_name] = total_discharge / 1000

    # Handle gas storage with retrofit/non-retrofit breakdown
    gas_stores = n.stores[n.stores.carrier.str.contains("gas storage", case=False, na=False)]

    if not gas_stores.empty:
        # Separate retrofit and non-retrofit stores
        retrofit_stores = gas_stores[
            gas_stores.carrier.str.contains("retrofit", case=False, na=False)
            & ~gas_stores.carrier.str.contains("nonretrofit", case=False, na=False)
        ]
        nonretrofit_stores = gas_stores[gas_stores.carrier.str.contains("nonretrofit", case=False, na=False)]

        # Calculate capacities for retrofitable storage
        gas_baseline_charge_retrofit = 0
        gas_baseline_discharge_retrofit = 0
        gas_retained_charge_retrofit = 0
        gas_retained_discharge_retrofit = 0

        for idx, store in retrofit_stores.iterrows():
            store_bus = store["bus"]

            charge_links = n.links[
                (n.links["bus1"] == store_bus) & n.links.index.str.contains("charge", case=False, na=False)
            ]
            discharge_links = n.links[
                (n.links["bus0"] == store_bus) & n.links.index.str.contains("discharge", case=False, na=False)
            ]

            gas_baseline_charge_retrofit += charge_links["p_nom"].sum() / 1000
            gas_baseline_discharge_retrofit += discharge_links["p_nom"].sum() / 1000
            gas_retained_charge_retrofit += charge_links["p_nom_opt"].sum() / 1000
            gas_retained_discharge_retrofit += discharge_links["p_nom_opt"].sum() / 1000

        # Calculate capacities for non-retrofitable storage
        gas_retained_charge_nonretrofit = 0
        gas_retained_discharge_nonretrofit = 0

        for idx, store in nonretrofit_stores.iterrows():
            store_bus = store["bus"]

            charge_links = n.links[
                (n.links["bus1"] == store_bus) & n.links.index.str.contains("charge", case=False, na=False)
            ]
            discharge_links = n.links[
                (n.links["bus0"] == store_bus) & n.links.index.str.contains("discharge", case=False, na=False)
            ]

            gas_retained_charge_nonretrofit += charge_links["p_nom_opt"].sum() / 1000
            gas_retained_discharge_nonretrofit += discharge_links["p_nom_opt"].sum() / 1000

        # Add to data dictionaries with separate categories
        if gas_retained_charge_retrofit > 1:
            charge_data["Gas Storage Retrofitable"] = gas_retained_charge_retrofit
            discharge_data["Gas Storage Retrofitable"] = gas_retained_discharge_retrofit
        if gas_retained_charge_nonretrofit > 1:
            charge_data["Gas Storage Nonretrofitable"] = gas_retained_charge_nonretrofit
            discharge_data["Gas Storage Nonretrofitable"] = gas_retained_discharge_nonretrofit

    return charge_data, discharge_data


def plot_generation_capacities(n: pypsa.Network, save_path: str, **wildcards):
    """Plot stacked bar chart of power generation and production capacities with improved legend."""
    # Get electrical generation capacities
    elec_caps = get_elec_capacities(n)
    if not elec_caps.empty:
        if isinstance(elec_caps.index, pd.MultiIndex):
            elec_total = elec_caps.groupby("carrier").sum() / 1e3  # Convert MW to GW
        else:
            elec_total = elec_caps / 1e3
    else:
        elec_total = pd.Series(dtype=float, name="capacity")

    # # Get gas production capacities (now includes bio-cc and methanation)
    # gas_caps = get_gas_capacities(n)
    # if not gas_caps.empty:
    #     if isinstance(gas_caps.index, pd.MultiIndex):
    #         gas_total = gas_caps.groupby('carrier').sum() / 1e3  # Convert MW to GW
    #     else:
    #         gas_total = gas_caps / 1e3
    # else:
    #     gas_total = pd.Series(dtype=float, name='capacity')
    #
    # # Filter out retired gas production (negative values) for capacity plotting
    # gas_total = gas_total[gas_total > 0]

    # Get hydrogen production capacities
    h2_caps = get_h2_capacities(n)
    if not h2_caps.empty:
        if isinstance(h2_caps.index, pd.MultiIndex):
            h2_total = h2_caps.groupby("carrier").sum() / 1e3  # Convert MW to GW
        else:
            h2_total = h2_caps / 1e3
    else:
        h2_total = pd.Series(dtype=float, name="capacity")

    # Create combined dataset for plotting
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Get colors
    elec_colors = get_color_palette()
    gas_colors = get_gas_color_palette()
    h2_colors = get_h2_color_palette()
    all_colors = {**elec_colors, **gas_colors, **h2_colors}

    # Prepare data for plotting - now with 3 categories
    categories = []
    category_data = []

    categories.append("Power Generation")
    category_data.append(elec_total)

    # categories.append('Gas Production')
    # category_data.append(gas_total)

    categories.append("Hydrogen Production")
    category_data.append(h2_total)

    x_pos = np.arange(len(categories))
    width = 0.6

    # Track legend items in stacking order for each bar
    all_legend_items = []  # Will store (handle, label) tuples in stacking order
    legend_handles_dict = {}  # To avoid duplicates

    # Plot stacked bars for each category
    for i, (category, data) in enumerate(zip(categories, category_data)):
        bottom = 0
        current_bar_items = []  # Track items for this specific bar

        # Calculate total for this category to determine significant segments
        total_capacity = data.sum()

        for carrier, capacity in data.items():
            if capacity > 0:
                color = all_colors.get(carrier, "#D3D3D3")

                # Create bar
                handle = ax.bar(
                    i,
                    capacity,
                    bottom=bottom,
                    color=color,
                    width=width,
                )

                # Track for legend (avoid duplicates across bars)
                if carrier not in legend_handles_dict:
                    legend_handles_dict[carrier] = handle[0]
                    current_bar_items.append((handle[0], carrier))

                # Add individual segment label if significant (>0.1% of category total)
                if total_capacity > 0 and capacity > total_capacity * 0.001:
                    ax.text(
                        i,
                        bottom + capacity / 2,
                        f"{capacity:.1f}",
                        ha="center",
                        va="center",
                        fontweight="bold",
                        fontsize=9,
                    )

                bottom += capacity

        # Add current bar items to overall legend items (in stacking order)
        all_legend_items.extend(current_bar_items)

        # Add total capacity annotation
        if bottom > 0:
            ax.text(
                i,
                bottom + bottom * 0.02,
                f"{bottom:.1f} GW",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    # Customize plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Capacity (GW)", fontsize=12)
    ax.set_title("Production Capacity", fontsize=TITLE_SIZE)

    ax.grid(axis="y", alpha=0.3)

    # Create legend in REVERSE stacking order (visual top to bottom)
    if legend_handles_dict:
        # Get unique carriers in the order they appear (first bar takes precedence)
        seen_carriers = []
        for category, data in zip(categories, category_data):
            for carrier in data.index:
                if carrier in legend_handles_dict and carrier not in seen_carriers:
                    seen_carriers.append(carrier)

        # Reverse for top-to-bottom visual order
        reversed_carriers = list(reversed(seen_carriers))
        handles = [legend_handles_dict[carrier] for carrier in reversed_carriers]
        labels = reversed_carriers

        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Generation capacities plot saved to {save_path}")
    logger.info(f"  Total electrical capacity: {elec_total.sum():.1f} GW")
    # logger.info(f"  Total gas production capacity: {gas_total.sum():.1f} GW")
    logger.info(f"  Total hydrogen capacity: {h2_total.sum():.1f} GW")


def plot_transport_capacities(n: pypsa.Network, pipeline_retro_factor, save_path: str, **wildcards):
    """Plot stacked bar chart of transport capacities × distance with corrected legend order."""
    transport_data = get_transport_capacities(n, pipeline_retro_factor)

    # Check if we have any meaningful data
    total_capacity = sum(abs(v) for v in transport_data.values())

    # Reorder data to put gas retired/retained near axis
    ordered_keys = [
        "Gas (Retired)",
        "Gas (Retrofitted to H2)",
        "Gas (Retained)",
        "Electrical (Existing)",
        "Electrical (New)",
        "Hydrogen (Retrofit)",
        "Hydrogen (New)",
    ]

    # Separate positive and negative values in the desired order
    positive_data = [(k, transport_data[k]) for k in ordered_keys if k in transport_data and transport_data[k] > 0]
    negative_data = [(k, transport_data[k]) for k in ordered_keys if k in transport_data and transport_data[k] < 0]

    # Create plot
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Define colors for different infrastructure types
    colors = {
        "Electrical (Existing)": "#1f77b4",
        "Electrical (New)": "#aec7e8",
        "Hydrogen (Retrofit)": "#2ca02c",
        "Hydrogen (New)": "#98df8a",
        "Gas (Retained)": "#ff7f0e",
        "Gas (Retired)": "#808080",
        "Gas (Retrofitted to H2)": "#ff9896",
    }

    width = 0.6

    # Track legend items in visual stacking order (bottom to top for positive, top to bottom for negative)
    legend_items_order = []

    # Plot positive values (retained, new, retrofit) - start from gas retained at bottom
    bottom = 0
    for label, value in positive_data:
        if value > 0:
            color = colors.get(label, "#D3D3D3")
            handle = ax.bar(0, value, bottom=bottom, color=color, width=width)

            # Add to legend order (will be reversed later)
            legend_items_order.append((handle[0], label))

            # Add value label for significant segments
            if value > total_capacity * 0.001:  # Only label if >0.1% of total
                ax.text(
                    0,
                    bottom + value / 2,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    fontsize=9,
                )

            bottom += value

    # Plot negative values (retired/retrofitted gas) - start from gas retired at top
    neg_bottom = 0
    negative_legend_items = []
    for label, value in negative_data:
        if value < 0:
            color = colors.get(label, "#D3D3D3")
            handle = ax.bar(0, value, bottom=neg_bottom, color=color, width=width)

            # Add to negative legend items
            negative_legend_items.append((handle[0], label))

            # Add value label for significant segments
            if abs(value) > total_capacity * 0.001:
                ax.text(
                    0,
                    neg_bottom + value / 2,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    fontsize=9,
                )

            neg_bottom += value

    # Customize plot
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylabel("Capacity × Distance (GW × 1000km)", fontsize=12)

    ax.set_title("Transport Infrastructure Capacity × Distance", fontsize=TITLE_SIZE)

    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks([])

    # Create legend in visual order (top to bottom)
    all_legend_items = []

    # Add positive items in reverse order (visual top to bottom)
    all_legend_items.extend(reversed(legend_items_order))
    # Add negative items in their order (already top to bottom for negative section)
    all_legend_items.extend(negative_legend_items)

    if all_legend_items:
        handles, labels = zip(*all_legend_items)
        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add horizontal line at zero for reference
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

    # Add summary annotations
    total_positive = sum(v for v in transport_data.values() if v > 0)
    total_negative = sum(v for v in transport_data.values() if v < 0)

    if total_positive > 0:
        ax.text(
            0.25,
            total_positive + abs(total_positive) * 0.05,
            f"Total Positive:\n{total_positive:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
        )

    if total_negative < 0:
        ax.text(
            -0.25,
            total_negative - abs(total_negative) * 0.05,
            f"Total Negative:\n{total_negative:.1f}",
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Transport capacities plot saved to {save_path}")
    logger.info(f"  Total positive capacity: {total_positive:.1f} GW×1000km")
    if total_negative < 0:
        logger.info(f"  Total negative capacity: {total_negative:.1f} GW×1000km")

    # Log individual components
    for category, value in transport_data.items():
        if value != 0:
            logger.info(f"    {category}: {value:.1f} GW×1000km")


def plot_storage_energy_capacities(
    n: pypsa.Network, storage_eng_retro_factor_h2, storage_eng_retro_factor_acaes, save_path: str, **wildcards
):
    """Plot stacked bar chart of storage energy capacities with retrofit/non-retrofit distinction."""
    # Get detailed storage capacities
    storage_data = get_storage_energy_capacities(n, storage_eng_retro_factor_h2, storage_eng_retro_factor_acaes)

    # Define ordered stacking sequence
    ordered_keys = [
        "Gas Storage Nonretrofitable (Retained)",
        "Gas Storage Nonretrofitable (Retired)",
        "Gas Storage Retrofitable (Retired)",
        "Gas Storage Retrofitable (Retrofitted to ACAES)",
        "Gas Storage Retrofitable (Retrofitted to H2)",
        "Gas Storage Retrofitable (Retained)",
        "H2 Storage (Retrofitted from Gas)",
        "H2 Storage (New)",
        "ACAES Retrofit",
        "ACAES New",
        "TES",
        "PHS",
        "Battery",
    ]

    # Separate positive and negative values in the desired order
    positive_data = [(k, storage_data[k]) for k in ordered_keys if k in storage_data and storage_data[k] > 0]
    negative_data = [(k, storage_data[k]) for k in ordered_keys if k in storage_data and storage_data[k] < 0]

    # Create plot
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Get colors - use same color for retrofit/non-retrofit pairs
    storage_colors = get_storage_color_palette()
    colors = {
        "Battery": storage_colors.get("battery", "#1f77b4"),
        "PHS": storage_colors.get("phs", "#aec7e8"),
        "H2 Storage (New)": storage_colors.get("h2 storage new", "#98df8a"),
        "H2 Storage (Retrofitted from Gas)": storage_colors.get("h2 storage retrofit", "#2ca02c"),
        "TES": storage_colors.get("tes", "#FF0001"),
        "ACAES New": storage_colors.get("acaes new", "#ff69b4"),
        "ACAES Retrofit": storage_colors.get("acaes retrofit", "#ff1493"),
        "Gas Storage Retrofitable (Retained)": "#ff7f0e",
        "Gas Storage Retrofitable (Retired)": "#808080",
        "Gas Storage Nonretrofitable (Retained)": "#A0522D",
        "Gas Storage Nonretrofitable (Retired)": "#4D4D4D",
        "Gas Storage Retrofitable (Retrofitted to H2)": "#ff9896",
        "Gas Storage Retrofitable (Retrofitted to ACAES)": "#DA70D6",
    }

    width = 0.6
    legend_items_positive = []
    legend_items_negative = []

    # Calculate total absolute capacity for significance threshold
    total_abs = sum(abs(v) for v in storage_data.values())

    # Plot positive stacked bars with hatching for non-retrofitable
    positive_bottom = 0
    for name, value in positive_data:
        hatch = "//" if "Nonretrofitable" in name else None  # Add diagonal lines for non-retrofitable
        bar = ax.bar(
            [0],
            [value],
            width,
            bottom=positive_bottom,
            label=name,
            color=colors[name],
            hatch=hatch,
            edgecolor="black",
            linewidth=0.5,
        )
        positive_bottom += value
        legend_items_positive.append(bar)

        # Add individual segment label if significant (>1% of total absolute capacity)
        if value > total_abs * 0.01:
            ax.text(
                0,
                positive_bottom - value / 2,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=10,
            )

    # Plot negative stacked bars with hatching for non-retrofitable
    negative_bottom = 0
    for name, value in negative_data:
        hatch = "//" if "Nonretrofitable" in name else None  # Add diagonal lines for non-retrofitable
        bar = ax.bar(
            [0],
            [value],
            width,
            bottom=negative_bottom,
            label=name,
            color=colors[name],
            hatch=hatch,
            edgecolor="black",
            linewidth=0.5,
        )
        negative_bottom += value
        legend_items_negative.append(bar)

        # Add individual segment label if significant
        if abs(value) > total_abs * 0.01:
            ax.text(
                0,
                negative_bottom - value / 2,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=10,
            )

    # Format plot
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylabel("Energy Capacity (TWh)", fontsize=12)

    ax.set_title("Storage Energy Capacities", fontsize=TITLE_SIZE)

    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks([])

    # Add horizontal line at zero for reference
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

    # Create legend in visual stacking order
    all_legend_items = legend_items_positive[::-1] + legend_items_negative
    labels = [item.get_label() for item in all_legend_items]
    ax.legend(all_legend_items, labels, loc="upper left", bbox_to_anchor=(1.05, 1), frameon=False)

    # Add total annotations
    total_positive = sum(v for v in storage_data.values() if v > 0) if any(v > 0 for v in storage_data.values()) else 0
    total_negative = sum(v for v in storage_data.values() if v < 0) if any(v < 0 for v in storage_data.values()) else 0

    if total_positive > 0:
        ax.text(
            0.25,
            total_positive + total_positive * 0.05,
            f"Total: {total_positive:.2f} TWh",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
        )

    if total_negative < 0:
        ax.text(
            -0.25,
            total_negative - abs(total_negative) * 0.05,
            f"Total: {total_negative:.2f} TWh",
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Storage energy capacity plot saved to {save_path}")


def plot_storage_power_capacities(n: pypsa.Network, storage_pow_retro_factor_h2, save_path: str, **wildcards):
    """Plot stacked bar chart of storage charge/discharge power capacities with retrofit/non-retrofit distinction."""
    # Get storage power data
    charge_data, discharge_data = get_storage_power_capacities(n, storage_pow_retro_factor_h2)

    if not charge_data and not discharge_data:
        logger.warning("No storage power capacity data found")
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
        ax.text(
            0.5,
            0.5,
            "No Storage Power Capacity Data Available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title("Storage Charge/Discharge Power Capacities", fontsize=TITLE_SIZE)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Define colors - use same color for retrofit/non-retrofit pairs
    colors = {
        "Battery": "#1f77b4",
        "PHS": "#aec7e8",
        "H2 Storage New": "#98df8a",
        "H2 Storage Retrofit": "#2ca02c",
        "TES": "#FF0001",
        "ACAES New": "#ff69b4",
        "ACAES Retrofit": "#ff1493",
        "Gas Storage Retrofitable": "#ff7f0e",
        "Gas Storage Nonretrofitable": "#A0522D",
    }

    # Define stacking order
    ordered_keys = [
        "Gas Storage Nonretrofitable",
        "Gas Storage Retrofitable",
        "H2 Storage Retrofit",
        "H2 Storage New",
        "ACAES Retrofit",
        "ACAES New",
        "TES",
        "PHS",
        "Battery",
    ]

    categories = ["Charge", "Discharge"]
    category_data = [charge_data, discharge_data]
    x_pos = np.arange(len(categories))
    width = 0.6

    # Collect all unique storage types that exist in data
    all_storage_types = set(charge_data.keys()).union(discharge_data.keys())

    # Filter ordered_keys to only include existing storage types
    existing_ordered_keys = [k for k in ordered_keys if k in all_storage_types]

    # Collect legend items
    legend_items_dict = {}

    # Plot stacked bars for each category
    for i, (category, data) in enumerate(zip(categories, category_data)):
        positive_bottom = 0

        # Calculate total capacity for this category to determine significant segments
        total_capacity = sum(abs(v) for v in data.values())

        for storage_type in existing_ordered_keys:
            if storage_type in data:
                value = data[storage_type]
                hatch = "//" if "Nonretrofitable" in storage_type else None

                bar = ax.bar(
                    x_pos[i],
                    value,
                    width,
                    bottom=positive_bottom,
                    color=colors[storage_type],
                    hatch=hatch,
                    edgecolor="black",
                    linewidth=0.5,
                    label=storage_type,
                )
                if total_capacity > 0 and value > total_capacity * 0.01:
                    ax.text(
                        x_pos[i],
                        positive_bottom + value / 2,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontweight="bold",
                        fontsize=9,
                    )
                positive_bottom += value

                # 保存 legend bar
                if storage_type not in legend_items_dict:
                    legend_items_dict[storage_type] = bar

        # Add total capacity annotations
        ax.text(
            x_pos[i],
            positive_bottom + positive_bottom * 0.02,
            f"{positive_bottom:.1f} GW",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Format plot
    ax.set_ylabel("Storage Power Capacity (GW)", fontsize=12)
    ax.set_title("Storage Charge/Discharge Power Capacities", fontsize=TITLE_SIZE)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    legend_items_positive = []

    for storage_type in existing_ordered_keys:
        if storage_type in legend_items_dict:
            legend_items_positive.append(legend_items_dict[storage_type])

    all_legend_items = legend_items_positive[::-1]
    labels = [item.get_label() for item in all_legend_items]

    ax.legend(all_legend_items, labels, loc="upper left", bbox_to_anchor=(1.05, 1), frameon=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Storage power capacity plot saved to {save_path}")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_simpsec_capacity",
            case="HighE_new_h2storage_tes",
            transmission_network="reeds",
        )

    configure_logging(snakemake)

    # Load network
    n = pypsa.Network(snakemake.input.network)

    ng_options = snakemake.params.sector["natural_gas"]
    # Extract wildcards for titles
    wildcards = dict(snakemake.wildcards)

    # Create plots
    plot_generation_capacities(n, snakemake.output.generation_capacities, **wildcards)
    plot_transport_capacities(
        n,
        ng_options["pipeline_retro_factor"],
        snakemake.output.transport_capacities,
        **wildcards,
    )
    plot_storage_energy_capacities(
        n,
        ng_options["storage_eng_retro_factor_h2"],
        ng_options["storage_eng_retro_factor_acaes"],
        snakemake.output.storage_energy_capacities,
        **wildcards,
    )
    plot_storage_power_capacities(
        n,
        ng_options["storage_pow_retro_factor_h2"],
        snakemake.output.storage_power_capacities,
        **wildcards,
    )

    logger.info("All capacity plots completed successfully")
