"""
Plots long-term flexibility contribution analysis for SimpSec scenarios.

Creates stacked bar charts showing how different technologies contribute to
meeting peak demand (top month) compared to annual average.

Two plots are generated:
1. Based on electricity net load peaks (electricity load - wind - solar)
2. Based on total system load peaks (all loads - wind - solar)

Modified to merge transmission import/export and charge/discharge pairs.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging

# Import dispatch functions from plot_simpsec_dispatch
from plot_simpsec_dispatch import (
    get_electricity_dispatch_data,
    get_gas_dispatch_data,
    get_hydrogen_dispatch_data,
)

logger = logging.getLogger(__name__)

# Global plotting settings
TITLE_SIZE = 18
FIG_WIDTH = 18
FIG_HEIGHT = 10

# NERC Region definitions
NERC_REGIONS = {
    "NPCC": ["CT", "MA", "ME", "NH", "NY", "RI", "VT"],
    "SERC": ["AL", "AR", "FL", "GA", "LA", "MS", "MO", "NC", "SC", "TN", "IL", "KY", "VA"],
    "WECC": ["AZ", "CA", "CO", "ID", "MT", "NM", "NV", "OR", "UT", "WA", "WY"],
    "MRO": ["IA", "KS", "MN", "ND", "NE", "OK", "SD", "WI"],
    "RF": ["DE", "IN", "PA", "MD", "WV", "MI", "NJ", "OH"],
    "TRE": ["TX"],
}

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


def get_region_states(region: str) -> list:
    """
    Get states for a given region name (NERC Region or Interconnection).

    Args:
        region: Region name

    Returns
    -------
        List of state abbreviations
    """
    if region in NERC_REGIONS:
        return NERC_REGIONS[region]
    elif region in INTERCONNECTIONS:
        return INTERCONNECTIONS[region]
    else:
        logger.warning(f"Unknown region: {region}")
        return []


# Technology abbreviation mapping
def get_tech_abbreviation(tech_name: str) -> str:
    """Get abbreviated name for technology."""
    abbreviations = {
        "transmission": "trans",
        "compression": "comp",
        "electrolysis": "electro",
        "h2 turbine": "h2 turb",
    }
    return abbreviations.get(tech_name, tech_name)


def normalize_index_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame index to simple DatetimeIndex.
    Handles MultiIndex by extracting datetime level.

    Args:
        df: DataFrame with potentially MultiIndex

    Returns
    -------
        DataFrame with DatetimeIndex
    """
    if df.empty:
        return df

    df = df.copy()

    if isinstance(df.index, pd.MultiIndex):
        # Extract datetime level (usually level 1 or -1)
        datetime_values = df.index.get_level_values(-1)
        if not isinstance(datetime_values, pd.DatetimeIndex):
            datetime_values = pd.to_datetime(datetime_values)
        df.index = pd.DatetimeIndex(datetime_values, name="datetime")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df


def get_regional_dispatch_all_carriers(n: pypsa.Network, region: str) -> dict:
    """
    Get dispatch data for all carriers in a specific region.
    Normalizes all indices to DatetimeIndex.

    Args:
        n: PyPSA network
        region: Region name (e.g., 'NPCC', 'WECC')

    Returns
    -------
        Dictionary with structure:
        {
            'electricity': {'production': DataFrame, 'consumption': DataFrame},
            'gas': {'production': DataFrame, 'consumption': DataFrame},
            'hydrogen': {'production': DataFrame, 'consumption': DataFrame}
        }
    """
    # Get dispatch data for each carrier (native time resolution)
    elec_prod, elec_cons = get_electricity_dispatch_data(n, time_resolution="native", region=region)
    gas_prod, gas_cons = get_gas_dispatch_data(n, time_resolution="native", region=region)
    h2_prod, h2_cons = get_hydrogen_dispatch_data(n, time_resolution="native", region=region)

    # Normalize all indices to DatetimeIndex
    elec_prod = normalize_index_to_datetime(elec_prod)
    elec_cons = normalize_index_to_datetime(elec_cons)
    gas_prod = normalize_index_to_datetime(gas_prod)
    gas_cons = normalize_index_to_datetime(gas_cons)
    h2_prod = normalize_index_to_datetime(h2_prod)
    h2_cons = normalize_index_to_datetime(h2_cons)

    return {
        "electricity": {"production": elec_prod, "consumption": elec_cons},
        "gas": {"production": gas_prod, "consumption": gas_cons},
        "hydrogen": {"production": h2_prod, "consumption": h2_cons},
    }


def calculate_net_electricity_load(dispatch_data: dict) -> pd.Series:
    """
    Calculate net electricity load timeseries.
    Net load = electricity load - wind generation - solar generation

    Args:
        dispatch_data: Dictionary from get_regional_dispatch_all_carriers

    Returns
    -------
        Series of net load values (positive = net demand)
    """
    elec_prod = dispatch_data["electricity"]["production"]
    elec_cons = dispatch_data["electricity"]["consumption"]

    # Use production index as the reference (ensure consistent index)
    if not elec_prod.empty:
        reference_index = elec_prod.index
    elif not elec_cons.empty:
        reference_index = elec_cons.index
    else:
        return pd.Series(dtype=float)

    # Load (in consumption, negative values)
    if "load" in elec_cons.columns:
        load = elec_cons["load"]
        # Align to reference index if needed
        if not load.index.equals(reference_index):
            load = load.reindex(reference_index, fill_value=0)
    else:
        load = pd.Series(0, index=reference_index)

    # Wind and solar (in production, positive values)
    if "wind" in elec_prod.columns:
        wind = elec_prod["wind"]
    else:
        wind = pd.Series(0, index=reference_index)

    if "solar" in elec_prod.columns:
        solar = elec_prod["solar"]
    else:
        solar = pd.Series(0, index=reference_index)

    # Net load = -load - wind - solar
    # (load is negative, so -load converts to positive demand)
    net_load = -load - wind - solar

    return net_load


def find_peak_period(timeseries: pd.Series, window_hours: int = 720) -> pd.Index:
    """
    Find the continuous period with highest average values.
    Uses a rolling window to ensure temporal continuity.
    Treats the time series as circular (end connects to beginning).

    Args:
        timeseries: Time series data with DatetimeIndex
        window_hours: Length of the window in hours (24 for day, 168 for week, 720 for month)

    Returns
    -------
        Index of the continuous peak period
    """
    # Detect time step from the index
    time_delta = timeseries.index[1] - timeseries.index[0]
    timestep_hours = time_delta.total_seconds() / 3600

    # Calculate number of points needed to represent the window
    window_size = int(window_hours / timestep_hours)

    # Ensure window size doesn't exceed timeseries length
    if window_size > len(timeseries):
        logger.warning(
            f"Window size {window_size} exceeds timeseries length {len(timeseries)}, returning all points",
        )
        return timeseries.index

    # Create circular time series by concatenating original with itself
    # This allows rolling window to wrap around from end to beginning
    extended_series = pd.concat([timeseries, timeseries])

    # Calculate rolling mean on extended series
    rolling_mean = extended_series.rolling(window=window_size, min_periods=window_size).mean()

    # Find the index where rolling mean is maximum in the first cycle
    # Only consider windows that start in the first cycle to avoid duplicates
    first_cycle_rolling = rolling_mean.iloc[: len(timeseries)]
    peak_end_idx = first_cycle_rolling.idxmax()

    # Get the position of this index in the original series
    peak_end_pos = timeseries.index.get_loc(peak_end_idx)

    # Calculate the start position (the window ends at peak_end_pos)
    peak_start_pos = peak_end_pos - window_size + 1

    # Handle wrap-around case: peak period spans from end to beginning
    if peak_start_pos < 0:
        # Peak wraps around: take elements from end + elements from beginning
        n_from_end = abs(peak_start_pos)
        n_from_start = window_size - n_from_end

        # Get indices from end and beginning
        end_indices = timeseries.index[-n_from_end:]
        start_indices = timeseries.index[:n_from_start]

        # Combine them
        peak_hours = end_indices.append(start_indices)

        logger.info(f"Peak period ({window_hours}h) wraps around year boundary:")
        logger.info(f"  End portion: {end_indices[0]} to {end_indices[-1]}")
        logger.info(f"  Start portion: {start_indices[0]} to {start_indices[-1]}")
        logger.info(f"  Average: {timeseries.loc[peak_hours].mean():.2f} GW")
    else:
        # Normal case: continuous window within the year
        peak_hours = timeseries.index[peak_start_pos : peak_end_pos + 1]

        logger.info(
            f"Peak period ({window_hours}h): {peak_hours[0]} to {peak_hours[-1]} "
            f"(average: {timeseries.loc[peak_hours].mean():.2f} GW)",
        )

    return peak_hours


def merge_technology_contributions(carrier_results: dict) -> dict:
    """
    Merge related technology contributions (transmission import/export, charge/discharge pairs).

    Merging rules:
    - transmission import + transmission export -> transmission
    - battery + battery charge -> battery
    - phs + phs charge -> phs
    - h2 storage discharge + h2 storage charge -> h2 storage
    - gas storage discharge + gas storage charge -> gas storage
    - tes + tes charge -> tes
    - acaes + acaes charge -> acaes

    Args:
        carrier_results: Dictionary with technology contributions

    Returns
    -------
        Dictionary with merged technology contributions
    """
    merged_results = {}

    # Define merging rules: (target_name, [source_names])
    merge_rules = {
        "transmission": ["transmission import", "transmission export", "transmission loss"],
        "battery": ["battery", "battery charge"],
        "phs": ["phs", "phs charge"],
        "h2 storage": ["h2 storage discharge", "h2 storage charge"],
        "gas storage": ["gas storage discharge", "gas storage charge"],
        "tes": ["tes", "tes charge"],
        "acaes": ["acaes", "acaes charge"],
    }

    # Track which technologies have been merged
    merged_techs = set()

    # Apply merging rules
    for target_name, source_names in merge_rules.items():
        total_contribution = 0
        found_any = False

        for source in source_names:
            if source in carrier_results:
                total_contribution += carrier_results[source]
                merged_techs.add(source)
                found_any = True

        # Only add merged technology if at least one source was found and contribution is significant
        if found_any and abs(total_contribution) > 0.001:
            merged_results[target_name] = total_contribution

    # Add technologies that were not merged
    for tech, contribution in carrier_results.items():
        if tech not in merged_techs and abs(contribution) > 0.001:
            merged_results[tech] = contribution

    return merged_results


def calculate_flexibility_contribution(
    n: pypsa.Network,
    region_dict: dict,
    window_hours: int = 720,
) -> dict:
    """
    Calculate how much each technology contributes to meeting peak electricity demand
    compared to annual average.

    Only considers electricity-based peak periods (net electricity load).
    Flexibility contribution = (average during peak period) - (annual average)

    Args:
        n: PyPSA network
        region_dict: Dictionary of regions (NERC_REGIONS or INTERCONNECTIONS)
        window_hours: Window size in hours for peak period

    Returns
    -------
        Dictionary with structure:
        {
            'NPCC': {
                'electricity': {'wind': value, 'solar': value, ...},
                'gas': {...},
                'hydrogen': {...}
            },
            ...
        }
    """
    results = {}

    for region_name in region_dict.keys():
        logger.info(f"Calculating flexibility contribution for region: {region_name}")

        # Get dispatch data for this region
        dispatch_data = get_regional_dispatch_all_carriers(n, region_name)

        # Check if region has data
        has_data = False
        for carrier in ["electricity", "gas", "hydrogen"]:
            if not dispatch_data[carrier]["production"].empty or not dispatch_data[carrier]["consumption"].empty:
                has_data = True
                break

        if not has_data:
            logger.warning(f"  No dispatch data found for region {region_name}, skipping")
            results[region_name] = {"electricity": {}, "gas": {}, "hydrogen": {}}
            continue

        # Calculate net electricity load for peak period identification
        load_timeseries = calculate_net_electricity_load(dispatch_data)

        # Find peak period
        peak_hours = find_peak_period(load_timeseries, window_hours=window_hours)

        # Calculate flexibility for each carrier
        region_results = {}
        for carrier in ["electricity", "gas", "hydrogen"]:
            carrier_results = {}

            # Production technologies
            prod_df = dispatch_data[carrier]["production"]
            if not prod_df.empty:
                # Ensure peak_hours exist in prod_df index
                valid_peak_hours = peak_hours.intersection(prod_df.index)

                for tech in prod_df.columns:
                    annual_mean = prod_df[tech].mean()
                    if len(valid_peak_hours) > 0:
                        peak_mean = prod_df.loc[valid_peak_hours, tech].mean()
                    else:
                        peak_mean = annual_mean
                    contribution = peak_mean - annual_mean

                    # Only store significant contributions
                    if abs(contribution) > 0.0003:
                        carrier_results[tech] = contribution

            # Consumption technologies
            cons_df = dispatch_data[carrier]["consumption"]
            if not cons_df.empty:
                # Ensure peak_hours exist in cons_df index
                valid_peak_hours = peak_hours.intersection(cons_df.index)

                for tech in cons_df.columns:
                    annual_mean = cons_df[tech].mean()
                    if len(valid_peak_hours) > 0:
                        peak_mean = cons_df.loc[valid_peak_hours, tech].mean()
                    else:
                        peak_mean = annual_mean
                    contribution = peak_mean - annual_mean

                    # Only store significant contributions
                    if abs(contribution) > 0.0003:
                        carrier_results[tech] = contribution

            # Merge related technologies
            carrier_results = merge_technology_contributions(carrier_results)

            region_results[carrier] = carrier_results

            # Log summary for this carrier
            total_positive = sum(v for v in carrier_results.values() if v > 0)
            total_negative = sum(v for v in carrier_results.values() if v < 0)
            logger.info(f"  {carrier}: +{total_positive:.1f} GW, {total_negative:.1f} GW")

        results[region_name] = region_results

    return results


def get_comprehensive_color_palette() -> dict:
    """Get color palette for all technologies."""
    return {
        # Electricity production
        "wind": "#c9e3c1",
        "solar": "#ffd700",
        "hydro": "#6BC5E6",
        "nuclear": "#b5260d",
        "gas": "#8B4513",
        "gas-cc": "#D2B48C",
        "coal": "#8c564b",
        "coal-cc": "#5a5a5a",
        "biomass": "#ff7f0e",
        "biomass-cc": "#9e9518",
        "fuel cell": "#ea048a",
        "h2 turbine": "#f5bfe7",
        "battery": "#1f77b4",
        "phs": "#aec7e8",
        "tes": "#FF0001",
        "acaes": "#ff69b4",
        # Electricity consumption
        "load": "#2f2f2f",
        "electrolysis": "#7FD12C",
        "dac": "#9370DB",
        "compression": "#D3D3D3",
        "smr": "#0000ff",
        "smr-cc": "#00BFFF",
        # Gas
        "gas production": "#cd853f",
        "gas storage": "#ff8c00",
        "gas power plants": "#8B4513",
        # Hydrogen
        "bio h2-cc": "#006400",
        "h2 storage": "#2ca02c",
        # Transmission (merged)
        "transmission": "#7f8c8d",
    }


def plot_flexibility_contribution(
    flexibility_data: dict,
    save_path: str,
    title: str,
    window_label: str = "Month",
) -> None:
    """
    Plot stacked bar chart of flexibility contributions.
    Includes a table showing percentage contributions for electricity (excluding wind, solar, load).
    Filters contributions < 1% per region.

    Args:
        flexibility_data: Dictionary from calculate_flexibility_contribution
        save_path: Path to save the figure
        title: Plot title
        window_label: Label for time window (Day, Week, Month)
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    regions = list(flexibility_data.keys())
    carriers = ["electricity", "gas", "hydrogen"]
    carrier_labels = ["Electricity", "Gas", "Hydrogen"]

    n_regions = len(regions)
    n_carriers = len(carriers)

    # Set up x positions
    x = np.arange(n_regions)
    width = 0.25

    # Get color palette
    colors = get_comprehensive_color_palette()

    # Track technologies actually plotted (> 1% contribution in at least one region)
    techs_plotted = set()
    tech_carriers = {}  # Map tech to its carrier for legend grouping
    legend_handles = {}

    # First pass: filter contributions < 1% per region and collect plotted techs
    filtered_data = {}
    for region in regions:
        filtered_data[region] = {}
        for carrier in carriers:
            region_data = flexibility_data[region][carrier]

            if not region_data:
                filtered_data[region][carrier] = {}
                continue

            # Calculate total absolute contribution for this region-carrier
            total_abs = sum(abs(v) for v in region_data.values())

            # Filter out contributions < 1%
            filtered_region_carrier = {}
            for tech, value in region_data.items():
                if total_abs > 0 and abs(value) / total_abs >= 0.01:  # >= 1%
                    filtered_region_carrier[tech] = value
                    techs_plotted.add(tech)
                    tech_carriers[tech] = carrier

            filtered_data[region][carrier] = filtered_region_carrier

    # Calculate percentage contributions for electricity carrier
    elec_percentages = {}
    national_data = {"electricity": {}, "gas": {}, "hydrogen": {}}

    for region in regions:
        region_data = filtered_data[region]["electricity"]

        # Filter positive contributions, excluding wind, solar, load
        positive_contributions = {
            tech: value for tech, value in region_data.items() if value > 0 and tech not in ["wind", "solar", "load"]
        }

        total_positive = sum(positive_contributions.values())

        if total_positive > 0:
            region_percentages = {
                tech: (value / total_positive * 100) for tech, value in positive_contributions.items()
            }
            elec_percentages[region] = region_percentages
        else:
            elec_percentages[region] = {}

    # Calculate national totals - use original flexibility_data, not filtered
    national_data = {"electricity": {}, "gas": {}, "hydrogen": {}}

    for region in regions:
        for carrier in carriers:
            # Use original flexibility_data, not filtered_data
            for tech, value in flexibility_data[region][carrier].items():
                if tech not in national_data[carrier]:
                    national_data[carrier][tech] = 0
                national_data[carrier][tech] += value

    # Calculate percentage contributions for electricity carrier (per region)
    elec_percentages = {}

    for region in regions:
        region_data = filtered_data[region]["electricity"]  # Use filtered for display

        # Filter positive contributions, excluding wind, solar, load
        positive_contributions = {
            tech: value for tech, value in region_data.items() if value > 0 and tech not in ["wind", "solar", "load"]
        }

        total_positive = sum(positive_contributions.values())

        if total_positive > 0:
            region_percentages = {
                tech: (value / total_positive * 100) for tech, value in positive_contributions.items()
            }
            elec_percentages[region] = region_percentages
        else:
            elec_percentages[region] = {}

    # Calculate national percentages for electricity - use unfiltered national_data
    national_positive = {
        tech: value
        for tech, value in national_data["electricity"].items()
        if value > 0 and tech not in ["wind", "solar", "load"]
    }
    national_total = sum(national_positive.values())
    if national_total > 0:
        national_percentages = {tech: (value / national_total * 100) for tech, value in national_positive.items()}
    else:
        national_percentages = {}

    # Plot bars for each carrier using filtered data
    for carrier_idx, (carrier, carrier_label) in enumerate(zip(carriers, carrier_labels)):
        offset = (carrier_idx - 1) * width

        for region_idx, region in enumerate(regions):
            region_data = filtered_data[region][carrier]

            if not region_data:
                continue

            # Separate positive and negative contributions
            positive_techs = {k: v for k, v in region_data.items() if v > 0}
            negative_techs = {k: v for k, v in region_data.items() if v < 0}

            # Sort by absolute value
            positive_techs = dict(sorted(positive_techs.items(), key=lambda x: x[1], reverse=True))
            negative_techs = dict(sorted(negative_techs.items(), key=lambda x: x[1]))

            x_pos = x[region_idx] + offset

            # Plot positive contributions (stacked upward)
            bottom_pos = 0
            for tech, value in positive_techs.items():
                color = colors.get(tech, "#808080")
                bar = ax.bar(
                    x_pos,
                    value,
                    width,
                    bottom=bottom_pos,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.9,
                )

                if tech not in legend_handles:
                    legend_handles[tech] = bar[0]

                bottom_pos += value

            # Plot negative contributions (stacked downward)
            bottom_neg = 0
            for tech, value in negative_techs.items():
                color = colors.get(tech, "#808080")
                bar = ax.bar(
                    x_pos,
                    value,
                    width,
                    bottom=bottom_neg,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.9,
                )

                if tech not in legend_handles:
                    legend_handles[tech] = bar[0]

                bottom_neg += value

    # Add vertical dashed lines between regions
    for region_idx in range(1, n_regions):
        x_line = x[region_idx] - 0.5
        ax.axvline(x=x_line, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)

    # Adjust y-axis range to prevent table from covering bars
    y_min_data = ax.get_ylim()[0]
    y_max_data = ax.get_ylim()[1]
    y_range = y_max_data - y_min_data

    # Calculate required space for table (table occupies 0.76-0.99 of plot height)
    # This means table covers top 23% of plot area
    # We need to extend y_max to ensure bars don't reach into table area
    table_coverage = 0.23  # Table height in normalized coordinates
    required_padding = table_coverage / (1 - table_coverage)  # Convert to data space
    bottom_padding_factor = 0.15  # Keep some space at bottom

    ax.set_ylim(
        y_min_data - y_range * bottom_padding_factor,
        y_max_data + y_range * required_padding,  # Ensure table doesn't overlap bars
    )

    # Customize plot
    ax.set_ylabel("Flexibility Contribution (GW)", fontsize=18)
    ax.set_xlabel("NERC Region", fontsize=18)
    full_title = f"{title} ({window_label})"
    ax.set_title(full_title, fontsize=TITLE_SIZE + 2, pad=20)

    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.0)
    ax.grid(axis="y", alpha=0.3)

    # Add carrier labels
    y_label_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    for carrier_idx, carrier_label in enumerate(carrier_labels):
        offset = (carrier_idx - 1) * width
        for region_idx in range(n_regions):
            x_pos = x[region_idx] + offset
            ax.text(
                x_pos,
                ax.get_ylim()[0] + y_label_offset,
                carrier_label[0],
                ha="center",
                va="top",
                fontsize=14,
                color="gray",
                weight="bold",
                transform=ax.transData,
            )

    # Create single-column legend in top-right corner outside plot area
    if legend_handles:
        # Group technologies by carrier
        elec_techs = []
        gas_techs = []
        h2_techs = []

        for tech in techs_plotted:
            carrier = tech_carriers.get(tech, "electricity")
            if carrier == "electricity":
                elec_techs.append(tech)
            elif carrier == "gas":
                gas_techs.append(tech)
            else:  # hydrogen
                h2_techs.append(tech)

        # Sort by total contribution
        def get_total_contribution(tech):
            total = 0
            for region_data in filtered_data.values():
                for carrier_data in region_data.values():
                    total += abs(carrier_data.get(tech, 0))
            return total

        elec_techs.sort(key=get_total_contribution, reverse=True)
        gas_techs.sort(key=get_total_contribution, reverse=True)
        h2_techs.sort(key=get_total_contribution, reverse=True)

        # Build handles and labels organized by carrier (single column)
        from matplotlib.patches import Patch

        all_handles = []
        all_labels = []

        # Extract transmission and load from their respective lists
        special_techs = ["transmission", "load"]
        has_special = False

        for special_tech in special_techs:
            if special_tech in elec_techs:
                all_handles.append(legend_handles[special_tech])
                all_labels.append(special_tech)
                elec_techs.remove(special_tech)
                has_special = True
            elif special_tech in gas_techs:
                all_handles.append(legend_handles[special_tech])
                all_labels.append(special_tech)
                gas_techs.remove(special_tech)
                has_special = True
            elif special_tech in h2_techs:
                all_handles.append(legend_handles[special_tech])
                all_labels.append(special_tech)
                h2_techs.remove(special_tech)
                has_special = True

        # Add spacer after special techs if they exist
        if has_special:
            all_handles.append(Patch(facecolor="none", edgecolor="none"))
            all_labels.append(" ")

        # Add Electricity section
        if elec_techs:
            all_handles.append(Patch(facecolor="none", edgecolor="none"))
            all_labels.append("Electricity")
            for tech in elec_techs:
                all_handles.append(legend_handles[tech])
                all_labels.append(tech)

        # Add Gas section
        if gas_techs:
            # Add spacer before Gas section
            if elec_techs or has_special:
                all_handles.append(Patch(facecolor="none", edgecolor="none"))
                all_labels.append(" ")

            all_handles.append(Patch(facecolor="none", edgecolor="none"))
            all_labels.append("Gas")
            for tech in gas_techs:
                all_handles.append(legend_handles[tech])
                all_labels.append(tech)

        # Add Hydrogen section
        if h2_techs:
            # Add spacer before Hydrogen section
            if elec_techs or gas_techs or has_special:
                all_handles.append(Patch(facecolor="none", edgecolor="none"))
                all_labels.append(" ")

            all_handles.append(Patch(facecolor="none", edgecolor="none"))
            all_labels.append("Hydrogen")
            for tech in h2_techs:
                all_handles.append(legend_handles[tech])
                all_labels.append(tech)

        # Create legend in top-right corner outside plot area as single column
        legend = ax.legend(
            all_handles,
            all_labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            ncol=1,
            fontsize=14,
            frameon=True,
            fancybox=False,
            shadow=False,
            framealpha=1.0,
            edgecolor="black",
            facecolor="white",
            handlelength=1.5,
            handleheight=1.2,
        )

        # Set legend zorder to appear on top
        legend.set_zorder(100)

        # Make carrier headers bold
        for i, label in enumerate(legend.get_texts()):
            if label.get_text() in ["Electricity", "Gas", "Hydrogen"]:
                label.set_weight("bold")

    # Add percentage table with national column
    table_data = []

    # Build table header with region names + national
    header = regions + ["National"]
    table_data.append(header)

    # For each region, get top 5 technologies sorted by percentage
    region_top5 = {}
    for region in regions:
        region_percentages = elec_percentages.get(region, {})
        if region_percentages:
            sorted_techs = sorted(region_percentages.items(), key=lambda x: x[1], reverse=True)
            region_top5[region] = sorted_techs[:5]
        else:
            region_top5[region] = []

    # Get national top 5
    if national_percentages:
        national_sorted = sorted(national_percentages.items(), key=lambda x: x[1], reverse=True)
        region_top5["National"] = national_sorted[:5]
    else:
        region_top5["National"] = []

    # Build rows for top 5 positions
    for rank in range(5):
        row = []
        for region in regions + ["National"]:
            if rank < len(region_top5[region]):
                tech, pct = region_top5[region][rank]
                abbr = get_tech_abbreviation(tech)
                row.append(f"{abbr}: {pct:.1f}%")
            else:
                row.append("-")
        table_data.append(row)

    # Add table to plot if we have data
    if len(table_data) > 1:
        n_cols = len(header)

        # Adaptive table width based on number of columns
        # Increase base column width for better readability
        base_col_width = 0.12  # Increased from 0.065 to 0.12
        total_table_width = min(base_col_width * n_cols, 0.95)  # Cap at 0.95 (increased from 0.85)
        col_width = total_table_width / n_cols

        # Position table in upper left corner with adaptive width
        table = ax.table(
            cellText=table_data,
            cellLoc="center",
            loc="upper left",
            bbox=[0.01, 0.76, total_table_width, 0.23],  # Adaptive width
            colWidths=[col_width] * n_cols,  # Uniform adaptive column width
            zorder=10,
        )

        table.auto_set_font_size(False)
        table.set_fontsize(14)

        # Style the header row
        for i in range(len(header)):
            cell = table[(0, i)]
            cell.set_facecolor("#4a90e2")
            cell.set_text_props(weight="bold", fontsize=14, color="white")
            cell.set_edgecolor("white")

        # Style data cells
        for i in range(1, len(table_data)):
            for j in range(len(header)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor("#f0f0f0")
                else:
                    cell.set_facecolor("white")
                cell.set_alpha(1.0)
                cell.set_fontsize(14)
                cell.set_edgecolor("#d0d0d0")

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()

    logger.info(f"Flexibility contribution plot saved to {save_path}")
    logger.info(f"  Technologies plotted (>1% in at least one region): {len(techs_plotted)}")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_simpsec_flexibility",
            case="HighE_new_h2storage",
            transmission_network="tamu",
        )

    configure_logging(snakemake)

    # Load network
    n = pypsa.Network(snakemake.input.network)

    logger.info("Starting long-term flexibility contribution analysis (electricity-based peak only)")

    # Define time windows with their corresponding output attributes
    time_windows = [
        (168, "Week", "week"),  # 7 * 24 hours
        (720, "Month", "month"),  # 30 * 24 hours
        (2190, "Season", "season"),  # 91.25 * 24 hours
    ]

    # Calculate and plot flexibility for each time window
    for window_hours, window_label, output_suffix in time_windows:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing {window_label} analysis ({window_hours} hours)")
        logger.info(f"{'=' * 60}")

        # # ========== NERC REGION FLEXIBILITY ==========
        # flex_nerc = calculate_flexibility_contribution(n, NERC_REGIONS, window_hours=window_hours)
        #
        # # Get output path from snakemake outputs
        # output_nerc = getattr(snakemake.output, f"flexibility_nerc_{output_suffix}")
        #
        # plot_flexibility_contribution(
        #     flex_nerc,
        #     output_nerc,
        #     f"Long-term Flexibility Contribution - NERC Regions (Net Electricity Load Peak)",
        #     window_label=window_label
        # )

        # ========== INTERCONNECTION FLEXIBILITY ==========
        flex_interconnection = calculate_flexibility_contribution(n, INTERCONNECTIONS, window_hours=window_hours)

        # Get output path from snakemake outputs
        output_interconnection = getattr(snakemake.output, f"flexibility_interconnection_{output_suffix}")

        plot_flexibility_contribution(
            flex_interconnection,
            output_interconnection,
            "Long-term Flexibility Contribution - Interconnections (Net Electricity Load Peak)",
            window_label=window_label,
        )

    logger.info("\n" + "=" * 60)
    logger.info("Flexibility analysis completed successfully for all time windows")
    logger.info("=" * 60)
