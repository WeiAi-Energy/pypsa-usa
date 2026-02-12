"""
Plots dispatch analysis for SimpSec scenarios.

Creates time-series plots for:
1. Electricity dispatch (production vs consumption) with stacked area charts
2. Gas dispatch (production vs consumption) with stacked area charts
3. Hydrogen dispatch (production vs consumption) with stacked area charts
4. Storage state-of-charge curves for all storage types (excluding battery)

Additionally creates regional dispatch plots for each NERC region.
"""

import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging
from plot_simpsec_network import (
    get_color_palette,
    get_gas_color_palette,
    get_h2_color_palette,
    get_storage_color_palette,
)

logger = logging.getLogger(__name__)

# Global plotting settings
TITLE_SIZE = 16
FIG_WIDTH = 14
FIG_HEIGHT = 10

# NERC Region definitions - for granular analysis
NERC_REGIONS = {
    "NPCC": ["CT", "MA", "ME", "NH", "NY", "RI", "VT"],
    "RF": ["DE", "IN", "PA", "MD", "WV", "MI", "NJ", "OH"],
    "SERC": ["AL", "AR", "FL", "GA", "LA", "MS", "MO", "NC", "SC", "TN", "IL", "KY", "VA"],
    "MRO": ["IA", "KS", "MN", "ND", "NE", "OK", "SD", "WI"],
    "TRE": ["TX"],
    "WECC": ["AZ", "CA", "CO", "ID", "MT", "NM", "NV", "OR", "UT", "WA", "WY"],
}

# Interconnection definitions - for high-level analysis
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

# Create reverse mappings
STATE_TO_NERC = {}
for region, states in NERC_REGIONS.items():
    for state in states:
        STATE_TO_NERC[state] = region

STATE_TO_INTERCONNECTION = {}
for interconnection, states in INTERCONNECTIONS.items():
    for state in states:
        STATE_TO_INTERCONNECTION[state] = interconnection


def get_comprehensive_color_palette() -> dict:
    """Get comprehensive color palette for all possible dispatch categories."""
    # Base colors from existing functions
    base_colors = get_color_palette()
    gas_colors = get_gas_color_palette()
    h2_colors = get_h2_color_palette()
    storage_colors = get_storage_color_palette()

    # Comprehensive color palette
    comprehensive_colors = {
        # Electricity production
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
        "fuel cell": "#ea048a",
        "h2 turbine": "#f5bfe7",
        "acaes": "#ff1493",
        "tes": "#FF0001",
        "battery": "#1f77b4",
        "phs": "#aec7e8",
        # Electricity consumption
        "load": "#2f2f2f",
        "electrolysis": "#7FD12C",
        "dac": "#9370DB",
        "acaes charge": "#ff69b4",
        "tes charge": "#FF0001",
        "battery charge": "#4169e1",
        "phs charge": "#87ceeb",
        "smr": "#0000ff",
        "smr-cc": "#00BFFF",
        "compression": "#D3D3D3",
        # Gas production and consumption
        "gas production": "#ff7f0e",
        "gas bio-cc": "#2d5016",  # Deep green for gas bio-cc
        "gas methanation": "#98fb98",  # Light green for gas methanation
        "gas storage discharge": "#ffa500",
        "gas power plants": "#8B4513",
        "gas storage charge": "#ffa500",
        "other gas consumption": "#a0522d",
        # Hydrogen production and consumption
        "bio h2-cc": "#006400",
        "h2 storage discharge": "#2ca02c",
        "h2 storage charge": "#2ca02c",
        # Storage (for SOC plots)
        "gas storage": "#ff7f0e",
        "h2 storage": "#2ca02c",
        # Transmission (new)
        "transmission import": "#7f8c8d",
        "transmission export": "#7f8c8d",
        "loss": "#7f8c8d",
    }

    return comprehensive_colors


def get_dynamic_color_palette(data_columns: list) -> dict:
    """
    Get color palette only for technologies that actually exist in the data.
    Add default colors for any missing technologies.
    """
    comprehensive_colors = get_comprehensive_color_palette()

    # Default colors for missing technologies
    default_colors = [
        "#808080",
        "#a0a0a0",
        "#606060",
        "#c0c0c0",
        "#404040",
        "#e0e0e0",
        "#909090",
        "#b0b0b0",
        "#707070",
        "#d0d0d0",
    ]

    used_colors = {}
    default_idx = 0

    for col in data_columns:
        if col in comprehensive_colors:
            used_colors[col] = comprehensive_colors[col]
        else:
            # Assign default color for missing technology
            if default_idx < len(default_colors):
                used_colors[col] = default_colors[default_idx]
                default_idx += 1
            else:
                used_colors[col] = "#808080"  # Fallback gray

            logger.warning(f"Missing color definition for '{col}', using default color {used_colors[col]}")

    return used_colors


def calculate_regional_transmission(
    n: pypsa.Network,
    region: str,
    carrier_type: str = "electricity",
) -> tuple:
    """
    Calculate transmission import/export for a region.

    Correctly handles transmission losses:
    - p0: power at bus0 (positive = flowing out of bus0, negative = flowing into bus0)
    - p1: power at bus1 (positive = flowing into bus1, negative = flowing out of bus1)
    - Loss = p0 + p1

    For regional accounting, use the power at the regional bus (which includes losses).

    Algorithm:
    1. Calculate net flow (positive = net import, negative = net export)
    2. Split into import (positive part) and export (negative part)

    Args:
        n: PyPSA network
        region: Region name (e.g., 'NPCC', 'WECC')
        carrier_type: 'electricity', 'gas', or 'h2'

    Returns
    -------
        Tuple of (import_series, export_series) in MW/GW
    """
    region_states = get_region_states(region)
    snapshots = n.snapshots

    # Initialize time series
    if isinstance(snapshots, pd.MultiIndex):
        datetime_values = snapshots.get_level_values(1)
        datetime_index = pd.DatetimeIndex(datetime_values, name="datetime")
    else:
        datetime_index = pd.DatetimeIndex(snapshots, name="datetime")

    # Initialize net flow (positive = import, negative = export)
    net_flow = pd.Series(0.0, index=datetime_index)
    loss = pd.Series(0.0, index=datetime_index)
    # Process based on carrier type
    if carrier_type == "electricity":
        # 1. Process AC lines
        for line_idx, line in n.lines.iterrows():
            bus0 = line["bus0"]
            bus1 = line["bus1"]

            bus0_state = n.buses.at[bus0, "STATE"]
            bus1_state = n.buses.at[bus1, "STATE"]

            # Check if line crosses region boundary
            bus0_in_region = bus0_state in region_states
            bus1_in_region = bus1_state in region_states

            if bus0_in_region and not bus1_in_region:
                # bus0 is in region, use p0
                # p0 > 0: flowing out → net export → subtract from net_flow
                # p0 < 0: flowing in → net import → subtract from net_flow (double negative)
                net_flow -= n.lines_t.p0[line_idx].values

            elif bus1_in_region and not bus0_in_region:
                # bus1 is in region, use p1
                # p1 > 0: flowing in → net import → add to net_flow
                # p1 < 0: flowing out → net export → add to net_flow (becomes negative)
                net_flow -= n.lines_t.p1[line_idx].values

            elif bus0_in_region and bus1_in_region:
                loss += n.lines_t.p0[line_idx].values + n.lines_t.p1[line_idx].values

        # 2. Process AC/DC links
        ac_dc_links = n.links[n.links.carrier.isin(["AC", "DC"])]
        for link_idx, link in ac_dc_links.iterrows():
            bus0 = link["bus0"]
            bus1 = link["bus1"]

            bus0_state = n.buses.at[bus0, "STATE"]
            bus1_state = n.buses.at[bus1, "STATE"]

            bus0_in_region = bus0_state in region_states
            bus1_in_region = bus1_state in region_states

            if bus0_in_region and not bus1_in_region:
                # bus0 is in region, use p0
                net_flow -= n.links_t.p0[link_idx].values

            elif bus1_in_region and not bus0_in_region:
                # bus1 is in region, use p1
                net_flow -= n.links_t.p1[link_idx].values

            elif bus0_in_region and bus1_in_region:
                loss += n.links_t.p0[link_idx].values + n.links_t.p1[link_idx].values

    elif carrier_type == "gas":
        # Process gas pipeline links
        gas_links = n.links[n.links.carrier.str.contains("gas pipeline", case=False, na=False)]
        for link_idx, link in gas_links.iterrows():
            bus0 = link["bus0"]
            bus1 = link["bus1"]

            if bus0 not in n.buses.index or bus1 not in n.buses.index:
                continue

            bus0_state = n.buses.at[bus0, "STATE"]
            bus1_state = n.buses.at[bus1, "STATE"]

            bus0_in_region = bus0_state in region_states
            bus1_in_region = bus1_state in region_states

            if bus0_in_region and not bus1_in_region:
                # bus0 is in region, use p0
                net_flow -= n.links_t.p0[link_idx].values

            elif bus1_in_region and not bus0_in_region:
                # bus1 is in region, use p1
                net_flow -= n.links_t.p1[link_idx].values

            elif bus0_in_region and bus1_in_region:
                loss += n.links_t.p0[link_idx].values + n.links_t.p1[link_idx].values

    elif carrier_type == "h2":
        # Process hydrogen pipeline links
        h2_links = n.links[n.links.carrier.str.contains("h2 pipeline", case=False, na=False)]
        for link_idx, link in h2_links.iterrows():
            bus0 = link["bus0"]
            bus1 = link["bus1"]

            if bus0 not in n.buses.index or bus1 not in n.buses.index:
                continue

            bus0_state = n.buses.at[bus0, "STATE"]
            bus1_state = n.buses.at[bus1, "STATE"]

            bus0_in_region = bus0_state in region_states
            bus1_in_region = bus1_state in region_states

            if bus0_in_region and not bus1_in_region:
                # bus0 is in region, use p0
                net_flow -= n.links_t.p0[link_idx].values

            elif bus1_in_region and not bus0_in_region:
                # bus1 is in region, use p1
                net_flow -= n.links_t.p1[link_idx].values

            elif bus0_in_region and bus1_in_region:
                loss += n.links_t.p0[link_idx].values + n.links_t.p1[link_idx].values

    # Split net flow into import and export
    import_flow = pd.Series(np.maximum(net_flow.values, 0), index=datetime_index)
    export_flow = pd.Series(np.maximum(-net_flow.values, 0), index=datetime_index)

    return import_flow, export_flow, loss


def calculate_peak_period_windows(production_df: pd.DataFrame, consumption_df: pd.DataFrame) -> dict:
    """
    Calculate peak periods for different time windows (week, month, season).
    Returns the start and end times for each peak period.

    Args:
        production_df: Production DataFrame with wind and solar columns
        consumption_df: Consumption DataFrame with load column

    Returns
    -------
        Dictionary with structure:
        {
            'week': (start_time, end_time),
            'month': (start_time, end_time),
            'season': (start_time, end_time)
        }
    """
    from plot_simpsec_flexibility import calculate_net_electricity_load, find_peak_period

    # Prepare dispatch data in the format expected by calculate_net_electricity_load
    dispatch_data = {
        "electricity": {
            "production": production_df,
            "consumption": consumption_df,
        },
    }

    # Calculate net electricity load
    net_load = calculate_net_electricity_load(dispatch_data)

    # Define time windows in hours
    time_windows = {
        "week": 168,  # 7 * 24 hours
        "month": 720,  # 30 * 24 hours
        "season": 2190,  # 91.25 * 24 hours
    }

    peak_periods = {}

    for window_name, window_hours in time_windows.items():
        # Find peak period for this window
        peak_index = find_peak_period(net_load, window_hours=window_hours)

        if len(peak_index) > 0:
            peak_periods[window_name] = (peak_index[0], peak_index[-1])
        else:
            peak_periods[window_name] = (None, None)

    return peak_periods


def resample_timeseries(df: pd.DataFrame, time_resolution: str = "1H") -> pd.DataFrame:
    """
    Resample time series data to specified resolution.

    Args:
        df: DataFrame with time index (may be MultiIndex)
        time_resolution: Target resolution (e.g., "1H", "1D", "1W")

    Returns
    -------
        Resampled DataFrame
    """
    if time_resolution == "native":
        return df

    try:
        # Handle MultiIndex by temporarily using datetime level for resampling
        if isinstance(df.index, pd.MultiIndex):
            # Extract datetime level (assuming it's the second level)
            datetime_index = df.index.get_level_values(1)
            if not isinstance(datetime_index, pd.DatetimeIndex):
                datetime_index = pd.to_datetime(datetime_index)

            # Create temporary DataFrame with simple datetime index
            temp_df = df.copy()
            temp_df.index = datetime_index

            # Resample and return
            resampled = temp_df.resample(time_resolution).mean()
            return resampled
        else:
            # Use mean for resampling to handle different time steps properly
            return df.resample(time_resolution).mean()
    except Exception as e:
        logger.warning(f"Failed to resample to {time_resolution}: {e}")
        return df


def get_region_states(region: str) -> list:
    """
    Get states for a given region name (NERC Region or Interconnection).

    Args:
        region: Region name (could be NERC Region or Interconnection)

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


def get_electricity_dispatch_data(
    n: pypsa.Network,
    time_resolution: str = "1H",
    region: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract electricity production and consumption data.
    Excludes transmission links from consumption.

    Args:
        n: PyPSA network
        time_resolution: Time resolution for resampling
        region: Optional region name to filter data

    Returns
    -------
        Tuple of (production_df, consumption_df)
    """
    # Get all snapshots from the network
    snapshots = n.snapshots

    # Get region states if region is specified
    region_states = get_region_states(region) if region else None

    # Production data
    production_data = {}

    # 1. Generators
    if not n.generators_t.p.empty:
        gen_p = n.generators_t.p
        for gen_idx in gen_p.columns:
            if gen_idx in n.generators.index:
                # Filter by region if specified
                if region_states:
                    bus = n.generators.at[gen_idx, "bus"]
                    if bus not in n.buses.index:
                        continue
                    state = n.buses.at[bus, "STATE"]
                    if state not in region_states:
                        continue

                carrier = n.generators.at[gen_idx, "carrier"]
                # Merge wind technologies
                if carrier in ["onwind", "offwind_floating"]:
                    carrier = "wind"

                if carrier not in production_data:
                    production_data[carrier] = gen_p[gen_idx].copy()
                else:
                    production_data[carrier] += gen_p[gen_idx]

    # 2. Links producing electricity (bus1 is electrical)
    link_p1 = n.links_t.p1
    for link_idx in link_p1.columns:
        link = n.links.loc[link_idx]
        bus1 = link["bus1"]
        bus1_carrier = n.buses.at[bus1, "carrier"]
        if bus1_carrier == "AC":  # Electrical buses
            # Filter by region if specified
            if region_states:
                if bus1 not in n.buses.index:
                    continue
                state = n.buses.at[bus1, "STATE"]
                if state not in region_states:
                    continue

            carrier = link["carrier"]
            if carrier in ["AC", "DC"]:
                continue

            # Simplify carrier names
            if "CCGT" in carrier or "OCGT" in carrier:
                carrier = "gas-cc" if "CCS" in carrier else "gas"
            elif "coal" in carrier:
                carrier = "coal-cc" if "CCS" in carrier else "coal"
            elif "biomass" in carrier:
                carrier = "biomass-cc" if "CCS" in carrier else "biomass"
            elif "fuel cell" in carrier:
                carrier = "fuel cell"
            elif "h2 turbine" in carrier:
                carrier = "h2 turbine"
            elif "tes" in carrier and "discharge" in link_idx:
                carrier = "tes"
            elif "acaes" in carrier and "discharge" in link_idx:
                carrier = "acaes"

            power_flow = -link_p1[link_idx]  # Power injection is negative

            if carrier not in production_data:
                production_data[carrier] = power_flow.copy()
            else:
                production_data[carrier] += power_flow

    # 3. Storage units discharging
    storage_p = n.storage_units_t.p
    for storage_idx in storage_p.columns:
        # Filter by region if specified
        if region_states:
            bus = n.storage_units.at[storage_idx, "bus"]
            if bus not in n.buses.index:
                continue
            state = n.buses.at[bus, "STATE"]
            if state not in region_states:
                continue

        carrier = n.storage_units.at[storage_idx, "carrier"]
        if "battery" in carrier.lower():
            carrier = "battery"
        elif carrier == "PHS":
            carrier = "phs"

        discharge_flow = storage_p[storage_idx].clip(lower=0).abs()  # Power generation is positive

        if carrier not in production_data:
            production_data[carrier] = discharge_flow.copy()
        else:
            production_data[carrier] += discharge_flow

    # Handle MultiIndex issues properly
    if isinstance(snapshots, pd.MultiIndex):
        datetime_values = snapshots.get_level_values(1)
        datetime_index = pd.DatetimeIndex(datetime_values, name="datetime")
    else:
        datetime_index = pd.DatetimeIndex(snapshots, name="datetime")

    # Create production DataFrame
    production_series_list = []
    production_columns = []

    for carrier, series in production_data.items():
        series_values = series.values
        production_series_list.append(series_values)
        production_columns.append(carrier)

    if production_series_list:
        production_df = pd.DataFrame(
            dict(zip(production_columns, production_series_list)),
            index=datetime_index,
        )
        production_df = production_df.fillna(0).astype(float)
    else:
        production_df = pd.DataFrame(index=datetime_index)

    # Consumption data
    consumption_data = {}

    # 1. Loads
    loads_p = n.loads_t.p
    for load_idx in loads_p.columns:
        carrier = n.loads.at[load_idx, "carrier"]
        if carrier == "AC":
            # Filter by region if specified
            if region_states:
                bus = n.loads.at[load_idx, "bus"]
                if bus not in n.buses.index:
                    continue
                state = n.buses.at[bus, "STATE"]
                if state not in region_states:
                    continue

            if "load" not in consumption_data:
                consumption_data["load"] = loads_p[load_idx].copy()
            else:
                consumption_data["load"] += loads_p[load_idx]

    # 2. Links consuming electricity (bus0 is electrical OR bus2 is electrical with negative efficiency2), excluding transmission
    link_p0 = n.links_t.p0
    for link_idx in link_p0.columns:
        link = n.links.loc[link_idx]
        bus0 = link["bus0"]
        bus2 = link["bus2"]
        carrier = link["carrier"]

        # Exclude transmission links (AC, DC)
        if carrier in ["AC", "DC"]:
            continue

        # Check if this link consumes electricity from either bus0 or bus2
        consumes_electricity = False
        power_flow = None
        filter_bus = None

        # Case 1: bus0 is electrical (original logic)
        bus0_carrier = n.buses.at[bus0, "carrier"]
        if bus0_carrier == "AC":
            consumes_electricity = True
            power_flow = link_p0[link_idx]
            filter_bus = bus0

        # Case 2: bus2 is electrical with negative efficiency2
        if not consumes_electricity and bus2:
            bus2_carrier = n.buses.at[bus2, "carrier"]
            efficiency2 = link.get("efficiency2", 0)

            if bus2_carrier == "AC" and efficiency2 < 0:
                consumes_electricity = True
                power_flow = n.links_t.p2[link_idx]
                filter_bus = bus2

        # Filter by region if specified
        if consumes_electricity and region_states and filter_bus:
            if filter_bus not in n.buses.index:
                continue
            state = n.buses.at[filter_bus, "STATE"]
            if state not in region_states:
                continue

        # If this link consumes electricity, categorize and add it
        if consumes_electricity and power_flow is not None:
            # Group by major categories
            if "electrolysis" in carrier:
                carrier = "electrolysis"
            elif "dac" in carrier:
                carrier = "dac"
            elif "tes" in carrier and "charge" in link_idx:
                carrier = "tes charge"
            elif "acaes" in carrier and "charge" in link_idx:
                carrier = "acaes charge"
            elif "smr" in carrier and "cc" not in carrier:
                carrier = "smr"
            elif "smr" in carrier and "cc" in carrier:
                carrier = "smr-cc"
            else:
                carrier = "compression"

            if carrier not in consumption_data:
                consumption_data[carrier] = power_flow.copy()
            else:
                consumption_data[carrier] += power_flow

    # 3. Storage units charging
    storage_p = n.storage_units_t.p
    for storage_idx in storage_p.columns:
        # Filter by region if specified
        if region_states:
            bus = n.storage_units.at[storage_idx, "bus"]
            if bus not in n.buses.index:
                continue
            state = n.buses.at[bus, "STATE"]
            if state not in region_states:
                continue

        carrier = n.storage_units.at[storage_idx, "carrier"]
        if "battery" in carrier.lower():
            carrier = "battery charge"
        elif carrier == "PHS":
            carrier = "phs charge"

        charge_flow = -storage_p[storage_idx].clip(upper=0)  # Power consumption is negative

        if carrier not in consumption_data:
            consumption_data[carrier] = charge_flow.copy()
        else:
            consumption_data[carrier] += charge_flow

    # Create consumption DataFrame
    consumption_series_list = []
    consumption_columns = []

    for carrier, series in consumption_data.items():
        series_values = series.values
        consumption_series_list.append(series_values)
        consumption_columns.append(carrier)

    if consumption_series_list:
        consumption_df = pd.DataFrame(
            dict(zip(consumption_columns, consumption_series_list)),
            index=datetime_index,
        )
        consumption_df = consumption_df.fillna(0).astype(float) * -1
    else:
        consumption_df = pd.DataFrame(index=datetime_index)

    # Add transmission (import/export for regional, loss for all cases)
    import_flow, export_flow, loss = calculate_regional_transmission(n, region, "electricity")

    if region:
        # For regional case, add import/export
        if import_flow.sum() > 0:
            production_df["transmission import"] = import_flow.values
        if export_flow.sum() > 0:
            consumption_df["transmission export"] = -export_flow.values
    consumption_df["transmission loss"] = -loss.values

    # Resample if requested
    production_df = resample_timeseries(production_df, time_resolution) / 1000
    consumption_df = resample_timeseries(consumption_df, time_resolution) / 1000

    production_mean = production_df.mean()
    significant_prod_cols = production_mean[production_mean >= 3e-4 * production_mean.sum()].index
    production_df = production_df[significant_prod_cols]

    consumption_mean = consumption_df.mean()
    significant_cons_cols = consumption_mean[consumption_mean <= 3e-4 * consumption_mean.sum()].index
    consumption_df = consumption_df[significant_cons_cols]

    return production_df, consumption_df


def get_gas_dispatch_data(
    n: pypsa.Network,
    time_resolution: str = "1H",
    region: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract gas production and consumption data.

    Args:
        n: PyPSA network
        time_resolution: Time resolution for resampling
        region: Optional region name to filter data

    Returns
    -------
        Tuple of (production_df, consumption_df)
    """
    snapshots = n.snapshots

    # Get region states if region is specified
    region_states = get_region_states(region) if region else None

    # Production data
    production_data = {}

    # Gas production from stores and links
    link_p1 = n.links_t.p1
    for link_idx in link_p1.columns:
        link = n.links.loc[link_idx]
        bus1 = link["bus1"]
        bus1_carrier = n.buses.at[bus1, "carrier"]
        if bus1_carrier == "gas":
            # Filter by region if specified
            if region_states:
                if bus1 not in n.buses.index:
                    continue
                state = n.buses.at[bus1, "STATE"]
                if state not in region_states:
                    continue

            carrier = link["carrier"]
            if "production" in carrier:
                carrier = "gas production"
            elif "bio-cc" in carrier:
                carrier = "gas bio-cc"
            elif "methanation" in carrier:
                carrier = "gas methanation"
            elif "storage" in carrier and "discharge" in link_idx:
                carrier = "gas storage discharge"
            elif "pipeline" in carrier:
                continue

            power_flow = -link_p1[link_idx].clip(upper=0)

            if carrier not in production_data:
                production_data[carrier] = power_flow.copy()
            else:
                production_data[carrier] += power_flow

    # Consumption data
    consumption_data = {}

    # 1. Loads
    loads_p = n.loads_t.p
    for load_idx in loads_p.columns:
        carrier = n.loads.at[load_idx, "carrier"]
        if carrier == "gas":
            # Filter by region if specified
            if region_states:
                bus = n.loads.at[load_idx, "bus"]
                if bus not in n.buses.index:
                    continue
                state = n.buses.at[bus, "STATE"]
                if state not in region_states:
                    continue

            if "load" not in consumption_data:
                consumption_data["load"] = loads_p[load_idx].copy()
            else:
                consumption_data["load"] += loads_p[load_idx]

    # 2. Gas consumption by links
    link_p0 = n.links_t.p0
    for link_idx in link_p0.columns:
        link = n.links.loc[link_idx]
        bus0 = link["bus0"]
        bus0_carrier = n.buses.at[bus0, "carrier"]
        if bus0_carrier == "gas":
            # Filter by region if specified
            if region_states:
                if bus0 not in n.buses.index:
                    continue
                state = n.buses.at[bus0, "STATE"]
                if state not in region_states:
                    continue

            carrier = link["carrier"]

            if "CCGT" in carrier or "OCGT" in carrier:
                carrier = "gas power plants"
            elif "smr" in carrier and "cc" not in carrier:
                carrier = "smr"
            elif "smr" in carrier and "cc" in carrier:
                carrier = "smr-cc"
            elif "storage" in carrier and "charge" in link_idx:
                carrier = "gas storage charge"
            elif "pipeline" in carrier:
                continue
            else:
                carrier = "other gas consumption"

            power_flow = link_p0[link_idx].clip(lower=0)

            if carrier not in consumption_data:
                consumption_data[carrier] = power_flow.copy()
            else:
                consumption_data[carrier] += power_flow

    # Handle MultiIndex
    if isinstance(snapshots, pd.MultiIndex):
        datetime_values = snapshots.get_level_values(1)
        datetime_index = pd.DatetimeIndex(datetime_values, name="datetime")
    else:
        datetime_index = pd.DatetimeIndex(snapshots, name="datetime")

    # Convert to DataFrames
    production_df = pd.DataFrame(production_data, index=snapshots).fillna(0)
    consumption_df = pd.DataFrame(consumption_data, index=snapshots).fillna(0) * -1

    # Add transmission (import/export for regional, loss for all cases)
    import_flow, export_flow, loss = calculate_regional_transmission(n, region, "gas")

    if region:
        # For regional case, add import/export
        if import_flow.sum() > 0:
            production_df["transmission import"] = import_flow.values
        if export_flow.sum() > 0:
            consumption_df["transmission export"] = -export_flow.values

        # Add transport loss to consumption (all cases)
        consumption_df["transmission loss"] = -loss.values

    # Resample if requested
    production_df = resample_timeseries(production_df, time_resolution) / 1000
    consumption_df = resample_timeseries(consumption_df, time_resolution) / 1000

    production_mean = production_df.mean()
    significant_prod_cols = production_mean[production_mean >= 3e-4 * production_mean.sum()].index
    production_df = production_df[significant_prod_cols]

    consumption_mean = consumption_df.mean()
    significant_cons_cols = consumption_mean[consumption_mean <= 3e-4 * consumption_mean.sum()].index
    consumption_df = consumption_df[significant_cons_cols]

    return production_df, consumption_df


def get_hydrogen_dispatch_data(
    n: pypsa.Network,
    time_resolution: str = "1H",
    region: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract hydrogen production and consumption data.

    Args:
        n: PyPSA network
        time_resolution: Time resolution for resampling
        region: Optional region name to filter data

    Returns
    -------
        Tuple of (production_df, consumption_df)
    """
    snapshots = n.snapshots

    # Get region states if region is specified
    region_states = get_region_states(region) if region else None

    # Production data
    production_data = {}

    # H2 production from links
    link_p1 = n.links_t.p1
    for link_idx in link_p1.columns:
        link = n.links.loc[link_idx]
        bus1 = link["bus1"]
        bus1_carrier = n.buses.at[bus1, "carrier"]
        if bus1_carrier == "h2":
            # Filter by region if specified
            if region_states:
                if bus1 not in n.buses.index:
                    continue
                state = n.buses.at[bus1, "STATE"]
                if state not in region_states:
                    continue

            carrier = link["carrier"]

            if "electrolysis" in carrier:
                carrier = "electrolysis"
            elif "smr" in carrier and "cc" not in carrier:
                carrier = "smr"
            elif "smr" in carrier and "cc" in carrier:
                carrier = "smr-cc"
            elif "bio" in carrier:
                carrier = "bio h2-cc"
            elif "storage" in carrier and "discharge" in link_idx:
                carrier = "h2 storage discharge"
            elif "pipeline" in carrier:
                continue

            power_flow = -link_p1[link_idx].clip(upper=0)

            if carrier not in production_data:
                production_data[carrier] = power_flow.copy()
            else:
                production_data[carrier] += power_flow

    # Consumption data
    consumption_data = {}

    # 1. Loads
    loads_p = n.loads_t.p
    for load_idx in loads_p.columns:
        carrier = n.loads.at[load_idx, "carrier"]
        if carrier == "h2":
            # Filter by region if specified
            if region_states:
                bus = n.loads.at[load_idx, "bus"]
                if bus not in n.buses.index:
                    continue
                state = n.buses.at[bus, "STATE"]
                if state not in region_states:
                    continue

            if "load" not in consumption_data:
                consumption_data["load"] = loads_p[load_idx].copy()
            else:
                consumption_data["load"] += loads_p[load_idx]

    # 2. H2 consumption by links
    link_p0 = n.links_t.p0
    for link_idx in link_p0.columns:
        link = n.links.loc[link_idx]
        bus0 = link["bus0"]
        bus0_carrier = n.buses.at[bus0, "carrier"]
        if bus0_carrier == "h2":
            # Filter by region if specified
            if region_states:
                if bus0 not in n.buses.index:
                    continue
                state = n.buses.at[bus0, "STATE"]
                if state not in region_states:
                    continue

            carrier = link["carrier"]

            if "fuel cell" in carrier:
                carrier = "fuel cell"
            elif "turbine" in carrier:
                carrier = "h2 turbine"
            elif "storage" in carrier and "charge" in link_idx:
                carrier = "h2 storage charge"
            elif "pipeline" in carrier:
                continue

            power_flow = link_p0[link_idx].clip(lower=0)

            if carrier not in consumption_data:
                consumption_data[carrier] = power_flow.copy()
            else:
                consumption_data[carrier] += power_flow

    # Handle MultiIndex
    if isinstance(snapshots, pd.MultiIndex):
        datetime_values = snapshots.get_level_values(1)
        datetime_index = pd.DatetimeIndex(datetime_values, name="datetime")
    else:
        datetime_index = pd.DatetimeIndex(snapshots, name="datetime")

    # Convert to DataFrames
    production_df = pd.DataFrame(production_data, index=snapshots).fillna(0)
    consumption_df = pd.DataFrame(consumption_data, index=snapshots).fillna(0) * -1

    # Add transmission (import/export for regional, loss for all cases)
    import_flow, export_flow, loss = calculate_regional_transmission(n, region, "h2")

    if region:
        # For regional case, add import/export
        if import_flow.sum() > 0:
            production_df["transmission import"] = import_flow.values
        if export_flow.sum() > 0:
            consumption_df["transmission export"] = -export_flow.values

        # Add transport loss to consumption (all cases)
        consumption_df["transmission loss"] = -loss.values

    # Resample if requested
    production_df = resample_timeseries(production_df, time_resolution) / 1000
    consumption_df = resample_timeseries(consumption_df, time_resolution) / 1000

    production_mean = production_df.mean()
    significant_prod_cols = production_mean[production_mean >= 3e-4 * production_mean.sum()].index
    production_df = production_df[significant_prod_cols]

    consumption_mean = consumption_df.mean()
    significant_cons_cols = consumption_mean[consumption_mean <= 3e-4 * consumption_mean.sum()].index
    consumption_df = consumption_df[significant_cons_cols]

    return production_df, consumption_df


def get_storage_soc_data(n: pypsa.Network, time_resolution: str = "1H", region: str = None) -> pd.DataFrame:
    """
    Extract storage state-of-charge data (excluding battery),
    optionally filtered by NERC region.

    Args:
        n : pypsa.Network
            Network object.
        time_resolution : str
            Target time resolution (e.g., "1H", "1D", "1W").
        region : str, optional
            NERC region name (e.g., "WECC", "SERC").
            If None, aggregate all regions (national).

    Returns
    -------
        pd.DataFrame
            SOC data (TWh) by storage carrier over time.
    """
    snapshots = n.snapshots
    soc_data = {}

    region_states = get_region_states(region) if region else None

    # --- Stores SOC (gas, H2, TES, etc.) ---
    store_e = n.stores_t.e
    for store_idx in store_e.columns:
        carrier = n.stores.at[store_idx, "carrier"]

        if region_states:
            bus = n.stores.at[store_idx, "bus"]
            if bus not in n.buses.index:
                continue
            state = n.buses.at[bus, "STATE"]
            if state not in region_states:
                continue

        if "gas storage" in carrier:
            carrier = "gas storage"
        elif "h2 storage" in carrier:
            carrier = "h2 storage"
        elif "tes" in carrier:
            carrier = "tes"
        elif "caes" in carrier:
            continue  # skip compressed air
        else:
            continue

        if carrier not in soc_data:
            soc_data[carrier] = store_e[store_idx].copy()
        else:
            soc_data[carrier] += store_e[store_idx]

    if isinstance(snapshots, pd.MultiIndex):
        datetime_values = snapshots.get_level_values(-1)
        datetime_index = pd.DatetimeIndex(datetime_values, name="datetime")
    else:
        datetime_index = pd.DatetimeIndex(snapshots, name="datetime")

    if soc_data:
        soc_df = pd.concat(soc_data, axis=1)
        soc_df.index = datetime_index
        soc_df = soc_df.fillna(0).astype(float)
    else:
        soc_df = pd.DataFrame(index=datetime_index)

    soc_df = soc_df / 1e6  # TWh

    soc_df = resample_timeseries(soc_df, time_resolution)

    soc_mean = soc_df.mean()
    significant_cols = soc_mean[soc_mean >= 0.001].index
    soc_df = soc_df[significant_cols]

    return soc_df


def plot_dispatch(
    production_df: pd.DataFrame,
    consumption_df: pd.DataFrame,
    title: str,
    save_path: str,
    peak_period_windows: dict = None,
) -> None:
    """
    Plot stacked area chart for production and consumption with improved ordering and legend with percentages.
    For electricity dispatch, nuclear is always at bottom, others sorted by contribution.
    For gas dispatch, smr/smr-cc is always placed closest to the axis in consumption.

    Args:
        production_df: Production data DataFrame
        consumption_df: Consumption data DataFrame
        title: Plot title
        save_path: Path to save the figure
        peak_period_windows: Optional dict with peak period start/end times for week, month, season
                           Format: {'week': (start, end), 'month': (start, end), 'season': (start, end)}
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Handle MultiIndex - extract datetime part for plotting
    plot_production = production_df.copy()
    plot_consumption = consumption_df.copy()

    if isinstance(production_df.index, pd.MultiIndex):
        plot_index = production_df.index.get_level_values(1)
        if not isinstance(plot_index, pd.DatetimeIndex):
            plot_index = pd.to_datetime(plot_index)
    else:
        plot_index = production_df.index
        if not isinstance(plot_index, pd.DatetimeIndex):
            plot_index = pd.to_datetime(plot_index)

    # Update indices for both dataframes
    plot_production.index = plot_index
    if not plot_consumption.empty:
        plot_consumption.index = plot_index

    # Ensure data is float type
    plot_production = plot_production.astype(float)
    if not plot_consumption.empty:
        plot_consumption = plot_consumption.astype(float)

    # Calculate total production and consumption for percentage calculation
    total_production = plot_production.mean().sum() if not plot_production.empty else 0
    total_consumption = plot_consumption.mean().sum() if not plot_consumption.empty else 0
    total_consumption_abs = abs(total_consumption)

    # Get color palette for existing data only
    all_columns = list(plot_production.columns) + list(plot_consumption.columns)
    color_palette = get_dynamic_color_palette(all_columns)

    # Track legend items for consistent ordering
    legend_items = []

    is_electricity_dispatch = "Electricity" in title
    is_hydrogen_dispatch = "Hydrogen" in title
    is_gas_dispatch = "Gas" in title

    # -------------------- PRODUCTION --------------------
    if not plot_production.empty and plot_production.sum().sum() > 0:
        production_means = plot_production.mean()

        if is_electricity_dispatch:
            ordered_columns = []
            if "biomass-cc" in production_means.index:
                ordered_columns.append("biomass-cc")
            if "nuclear" in production_means.index:
                ordered_columns.append("nuclear")
            if "gas-cc" in production_means.index:
                ordered_columns.append("gas-cc")
            other_techs = [c for c in production_means.index if c not in ordered_columns]
            ordered_columns.extend(production_means[other_techs].sort_values(ascending=False).index)
            production_sorted = plot_production[ordered_columns]

        elif is_hydrogen_dispatch:
            ordered_columns = []
            if "bio h2-cc" in production_means.index:
                ordered_columns.append("bio h2-cc")
            if "smr-cc" in production_means.index:
                ordered_columns.append("smr-cc")
            other_techs = [c for c in production_means.index if c not in ordered_columns]
            ordered_columns.extend(production_means[other_techs].sort_values(ascending=False).index)
            production_sorted = plot_production[ordered_columns]

        else:
            production_means_sorted = production_means.sort_values(ascending=False)
            production_sorted = plot_production[production_means_sorted.index]

        bottom = np.zeros(len(production_sorted))
        for col in production_sorted.columns:
            values = production_sorted[col].values
            color = color_palette.get(col, "#808080")
            mean_val = production_means[col]
            if values.max() > 0:
                handle = ax.fill_between(
                    production_sorted.index,
                    bottom,
                    bottom + values,
                    alpha=0.8,
                    color=color,
                    linewidth=0,
                )
                pct = (mean_val / total_production * 100) if total_production > 0 else 0
                label = f"{col}: {pct:.2f}%"
                legend_items.append((handle, label, mean_val, True))
                bottom += values

    # -------------------- CONSUMPTION --------------------
    if not plot_consumption.empty and plot_consumption.sum().sum() < 0:
        consumption_means = plot_consumption.mean()
        consumption_abs_means = consumption_means.abs().sort_values(ascending=False)

        ordered_cols = list(consumption_abs_means.index)

        if is_gas_dispatch:
            smr_cols = [c for c in ordered_cols if "smr-cc" in c]
            load_cols = [c for c in ordered_cols if c == "load"]
            other_cols = [c for c in ordered_cols if c not in smr_cols + load_cols]
            ordered_cols = smr_cols + load_cols + other_cols

        consumption_sorted = plot_consumption[ordered_cols]

        bottom_cons = np.zeros(len(plot_consumption))
        neg_items_ordered = []

        for col in consumption_sorted.columns:
            values = consumption_sorted[col].values
            color = color_palette.get(col, "#333333")
            mean_val = consumption_means[col]
            abs_mean_val = abs(mean_val)
            if values.min() < 0:
                handle = ax.fill_between(
                    plot_consumption.index,
                    bottom_cons,
                    bottom_cons + values,
                    alpha=0.8,
                    color=color,
                    linewidth=0,
                )
                pct = (abs_mean_val / total_consumption_abs * 100) if total_consumption_abs > 0 else 0
                label = f"{col}: {pct:.2f}%"
                neg_items_ordered.append((handle, label, abs_mean_val, False))
                bottom_cons += values

    # -------------------- PEAK PERIOD OVERLAY (for electricity dispatch only) --------------------
    peak_line_handles = []
    if is_electricity_dispatch and peak_period_windows is not None:
        # Define colors and labels for different peak periods
        peak_styles = {
            "week": {"color": "#FF0000", "label": "Peak Week", "linestyle": "--", "alpha": 0.7},
            "month": {"color": "#FF8C00", "label": "Peak Month", "linestyle": "--", "alpha": 0.7},
            "season": {"color": "#9370DB", "label": "Peak Season", "linestyle": "--", "alpha": 0.7},
        }

        # Plot vertical lines for each peak period
        for period_name in ["week", "month", "season"]:
            if period_name in peak_period_windows:
                start_time, end_time = peak_period_windows[period_name]
                if start_time is not None and end_time is not None:
                    style = peak_styles[period_name]

                    # Draw vertical lines at start and end
                    ax.axvline(
                        x=start_time,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=2,
                        alpha=style["alpha"],
                    )
                    ax.axvline(
                        x=end_time,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=2,
                        alpha=style["alpha"],
                    )

                    # Create a dummy handle for legend (only need one per period)
                    handle = plt.Line2D(
                        [],
                        [],
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=2,
                        alpha=style["alpha"],
                    )
                    peak_line_handles.append((handle, style["label"]))

    # -------------------- FORMATTING --------------------
    y_max = plot_production.sum(axis=1).max() if not plot_production.empty else 0
    y_min = plot_consumption.sum(axis=1).min() if not plot_consumption.empty else 0

    if y_max > 0 or y_min < 0:
        margin = abs(y_max - y_min) * 0.05
        ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_ylabel("Power (GW)", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

    if len(plot_index) > 100:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.xticks(rotation=45, ha="right")

    # -------------------- LEGEND --------------------
    # Combine production and consumption legend entries
    prod_handles, prod_labels = [], []
    cons_handles, cons_labels = [], []

    # Production entries were added in stacking order (bottom → top)
    for h, label, _, is_prod in legend_items:
        if is_prod:
            prod_handles.append(h)
            prod_labels.append(label)

    # Consumption entries were added in stacking order (bottom → top)
    if "neg_items_ordered" in locals():
        for h, label, _, is_prod in neg_items_ordered:
            if not is_prod:
                cons_handles.append(h)
                cons_labels.append(label)

    # Construct grouped legend layout
    header_style = dict(color="none", linestyle="", marker="", linewidth=0)

    grouped_handles = []
    grouped_labels = []

    # --- Production group ---
    grouped_handles.append(plt.Line2D([], [], **header_style))
    grouped_labels.append("Production")
    grouped_handles.extend(prod_handles)
    grouped_labels.extend(prod_labels)

    # Add blank line (visual spacing)
    grouped_handles.append(plt.Line2D([], [], **header_style))
    grouped_labels.append(" ")

    # --- Consumption group ---
    grouped_handles.append(plt.Line2D([], [], **header_style))
    grouped_labels.append("Consumption")
    grouped_handles.extend(cons_handles)
    grouped_labels.extend(cons_labels)

    # Add peak periods to legend if present (for electricity dispatch only)
    if is_electricity_dispatch and peak_line_handles:
        grouped_handles.append(plt.Line2D([], [], **header_style))
        grouped_labels.append(" ")
        grouped_handles.append(plt.Line2D([], [], **header_style))
        grouped_labels.append("Peak Periods")

        for handle, label in peak_line_handles:
            grouped_handles.append(handle)
            grouped_labels.append(label)

    # Final legend placement
    ax.legend(
        grouped_handles,
        grouped_labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=True,
        fontsize=10,
        ncol=1,
        handlelength=1.5,
        labelspacing=0.8,
        borderaxespad=0.5,
    )

    # -------------------- FINAL FORMATTING --------------------
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_storage_soc(soc_df: pd.DataFrame, title: str, save_path: str) -> None:
    """Plot storage state-of-charge curves for all storage types (including regional)."""
    if soc_df.empty:
        logger.warning("No storage SOC data to plot")
        return

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    color_palette = get_dynamic_color_palette(soc_df.columns)

    for col in soc_df.columns:
        ax.plot(
            soc_df.index,
            soc_df[col],
            label=col,
            color=color_palette.get(col, "#808080"),
            linewidth=2,
            alpha=0.8,
        )

    ax.set_ylabel("State of Charge (TWh)", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)

    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.grid(True, alpha=0.3)

    if len(soc_df.index) > 100:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.xticks(rotation=45, ha="right")

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=True,
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_simpsec_dispatch",
            case="HighE_reeds_new_h2storage_tes",
            transmission_network="reeds",
        )

    configure_logging(snakemake)

    # Load and filter network
    n = pypsa.Network(snakemake.input.network)
    time_resolution = snakemake.config.get("plotting", {}).get("time_resolution", "1D")

    wildcards = dict(snakemake.wildcards)

    # Create output directories
    output_dir = Path(snakemake.output.electricity_dispatch).parent
    nerc_dir = output_dir / "NERC_Region"
    interconnection_dir = output_dir / "Interconnection"
    nerc_dir.mkdir(exist_ok=True, parents=True)
    interconnection_dir.mkdir(exist_ok=True, parents=True)

    # ============ NATIONAL DISPATCH ============
    logger.info("Generating national dispatch plots...")

    elec_prod, elec_cons = get_electricity_dispatch_data(n, time_resolution)
    gas_prod, gas_cons = get_gas_dispatch_data(n, time_resolution)
    h2_prod, h2_cons = get_hydrogen_dispatch_data(n, time_resolution)
    storage_soc = get_storage_soc_data(n, time_resolution)

    # Calculate peak period windows for national electricity dispatch
    national_peak_windows = calculate_peak_period_windows(elec_prod, elec_cons)
    logger.info("National peak periods calculated:")
    for period, (start, end) in national_peak_windows.items():
        if start is not None:
            logger.info(f"  {period}: {start} to {end}")

    # Create national plots
    plot_dispatch(
        elec_prod,
        elec_cons,
        "Electricity Dispatch - National",
        snakemake.output.electricity_dispatch,
        peak_period_windows=national_peak_windows,
    )

    plot_dispatch(
        gas_prod,
        gas_cons,
        "Natural Gas Dispatch - National",
        snakemake.output.gas_dispatch,
    )

    plot_dispatch(
        h2_prod,
        h2_cons,
        "Hydrogen Dispatch - National",
        snakemake.output.hydrogen_dispatch,
    )

    plot_storage_soc(
        storage_soc,
        "Underground Storage SOC - National",
        snakemake.output.storage_soc,
    )

    # ============ NERC REGION DISPATCH ============
    logger.info("Generating NERC Region dispatch plots...")

    for region_name in NERC_REGIONS.keys():
        logger.info(f"Processing NERC Region: {region_name}")

        # Get regional dispatch data
        elec_prod_reg, elec_cons_reg = get_electricity_dispatch_data(n, time_resolution, region=region_name)
        gas_prod_reg, gas_cons_reg = get_gas_dispatch_data(n, time_resolution, region=region_name)
        h2_prod_reg, h2_cons_reg = get_hydrogen_dispatch_data(n, time_resolution, region=region_name)
        storage_soc_reg = get_storage_soc_data(n, time_resolution, region=region_name)

        # Skip if no data
        if elec_prod_reg.empty and elec_cons_reg.empty:
            logger.warning(f"No electricity dispatch data for region {region_name}, skipping...")
            continue

        # Calculate peak period windows for regional electricity dispatch
        regional_peak_windows = calculate_peak_period_windows(elec_prod_reg, elec_cons_reg)

        # Create regional plots
        plot_dispatch(
            elec_prod_reg,
            elec_cons_reg,
            f"Electricity Dispatch - {region_name}",
            str(nerc_dir / f"electricity_dispatch_{region_name}.png"),
            peak_period_windows=regional_peak_windows,
        )

        if not gas_prod_reg.empty or not gas_cons_reg.empty:
            plot_dispatch(
                gas_prod_reg,
                gas_cons_reg,
                f"Natural Gas Dispatch - {region_name}",
                str(nerc_dir / f"gas_dispatch_{region_name}.png"),
            )

        if not h2_prod_reg.empty or not h2_cons_reg.empty:
            plot_dispatch(
                h2_prod_reg,
                h2_cons_reg,
                f"Hydrogen Dispatch - {region_name}",
                str(nerc_dir / f"hydrogen_dispatch_{region_name}.png"),
            )

        if not storage_soc_reg.empty:
            plot_storage_soc(
                storage_soc_reg,
                f"Underground Storage SOC - {region_name}",
                str(nerc_dir / f"storage_soc_{region_name}.png"),
            )

    # ============ INTERCONNECTION DISPATCH ============
    logger.info("Generating Interconnection dispatch plots...")

    for interconnection_name in INTERCONNECTIONS.keys():
        logger.info(f"Processing Interconnection: {interconnection_name}")

        # Get interconnection dispatch data
        elec_prod_int, elec_cons_int = get_electricity_dispatch_data(n, time_resolution, region=interconnection_name)
        gas_prod_int, gas_cons_int = get_gas_dispatch_data(n, time_resolution, region=interconnection_name)
        h2_prod_int, h2_cons_int = get_hydrogen_dispatch_data(n, time_resolution, region=interconnection_name)
        storage_soc_int = get_storage_soc_data(n, time_resolution, region=interconnection_name)

        # Skip if no data
        if elec_prod_int.empty and elec_cons_int.empty:
            logger.warning(f"No electricity dispatch data for interconnection {interconnection_name}, skipping...")
            continue

        # Calculate peak period windows for interconnection electricity dispatch
        interconnection_peak_windows = calculate_peak_period_windows(elec_prod_int, elec_cons_int)

        # Create interconnection plots
        plot_dispatch(
            elec_prod_int,
            elec_cons_int,
            f"Electricity Dispatch - {interconnection_name} Interconnection",
            str(interconnection_dir / f"electricity_dispatch_{interconnection_name}.png"),
            peak_period_windows=interconnection_peak_windows,
        )

        if not gas_prod_int.empty or not gas_cons_int.empty:
            plot_dispatch(
                gas_prod_int,
                gas_cons_int,
                f"Natural Gas Dispatch - {interconnection_name} Interconnection",
                str(interconnection_dir / f"gas_dispatch_{interconnection_name}.png"),
            )

        if not h2_prod_int.empty or not h2_cons_int.empty:
            plot_dispatch(
                h2_prod_int,
                h2_cons_int,
                f"Hydrogen Dispatch - {interconnection_name} Interconnection",
                str(interconnection_dir / f"hydrogen_dispatch_{interconnection_name}.png"),
            )

        if not storage_soc_int.empty:
            plot_storage_soc(
                storage_soc_int,
                f"Underground Storage SOC - {interconnection_name} Interconnection",
                str(interconnection_dir / f"storage_soc_{interconnection_name}.png"),
            )

    logger.info("All dispatch plots (national, NERC regions, and interconnections) completed successfully")
