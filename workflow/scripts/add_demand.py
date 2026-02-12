"""
Adds demand to the network.

Depending on study, the load will all be aggregated to a single load
type, or distributed to different sectors and end use fuels.
"""

import logging
from pathlib import Path

import pandas as pd
import pypsa
from _helpers import configure_logging, get_multiindex_snapshots, mock_snakemake
from add_electricity import add_missing_carriers
from constants_sector import (
    AirTransport,
    BoatTransport,
    RailTransport,
    RoadTransport,
    SecCarriers,
    SecNames,
    Transport,
)

logger = logging.getLogger(__name__)

# TODO: replace with just constants through naming in build demand
VEHICLE_MAPPER = {
    "bus": f"{Transport.ROAD.value}-{RoadTransport.BUS.value}",
    "heavy_duty": f"{Transport.ROAD.value}-{RoadTransport.HEAVY.value}",
    "light_duty": f"{Transport.ROAD.value}-{RoadTransport.LIGHT.value}",
    "med_duty": f"{Transport.ROAD.value}-{RoadTransport.MEDIUM.value}",
    "air": f"{Transport.AIR.value}-{AirTransport.PASSENGER.value}",
    "rail_shipping": f"{Transport.RAIL.value}-{RailTransport.SHIPPING.value}",
    "rail_passenger": f"{Transport.RAIL.value}-{RailTransport.PASSENGER.value}",
    "boat_shipping": f"{Transport.BOAT.value}-{BoatTransport.SHIPPING.value}",
}


def parse_adj_scenario(adj_scenario: str) -> tuple:
    """
    Parse adj_scenario string into components.

    Format: x-L-y where:
    - x: fraction of hydrogen demand to adjust (float)
    - L: letter indicating adjustment type (E/G/O)
    - y: conversion factor (float)

    Returns
    -------
    tuple: (fraction, adjustment_type, conversion_factor)
        Returns (0.0, None, 0.0) if adj_scenario is empty or invalid

    Examples
    --------
    - "0.5-E-0.75" -> (0.5, 'E', 0.75)
    - "1.0-G-0.5" -> (1.0, 'G', 0.5)
    - "0.7-O-1.0" -> (0.7, 'O', 1.0)
    """
    if not adj_scenario or adj_scenario == "":
        return (0.0, None, 0.0)

    try:
        parts = adj_scenario.split("-")
        if len(parts) != 3:
            logger.warning(f"Invalid adj_scenario format: {adj_scenario}. Expected format: x-L-y")
            return (0.0, None, 0.0)

        # Parse components
        fraction = float(parts[0])
        adj_type = parts[1].upper()
        conversion_factor = float(parts[2])

        # Validate adjustment type
        if adj_type not in ["E", "G", "O"]:
            logger.warning(f"Invalid adjustment type: {adj_type}. Must be E, G, or O")
            return (0.0, None, 0.0)

        # Validate fraction range
        if not (0.0 <= fraction <= 1.0):
            logger.warning(f"Fraction {fraction} out of range [0.0, 1.0]")
            return (0.0, None, 0.0)

        # Validate conversion factor is positive
        if conversion_factor < 0.0:
            logger.warning(f"Conversion factor {conversion_factor} must be non-negative")
            return (0.0, None, 0.0)

        logger.info(f"Parsed adj_scenario: fraction={fraction}, type={adj_type}, factor={conversion_factor}")
        return (fraction, adj_type, conversion_factor)

    except (ValueError, IndexError) as e:
        logger.warning(f"Error parsing adj_scenario '{adj_scenario}': {e}")
        return (0.0, None, 0.0)


def attach_demand(n: pypsa.Network, df: pd.DataFrame, carrier: str, suffix: str):
    """
    Add demand to network from specified configuration setting.

    Returns network with demand added.
    """
    df.index = pd.to_datetime(df.index)
    assert len(df.index) == len(
        n.snapshots,
    ), "Demand time series length does not match network snapshots"
    df.index = n.snapshots
    n.madd(
        "Load",
        df.columns,
        suffix=suffix,
        bus=df.columns,
        p_set=df,
        carrier=carrier,
    )


def process_simpsec_data(demand_files, planning_horizons):
    """
    Process raw SimpSec demand files and organize by carrier.

    Returns
    -------
    dict: {carrier: DataFrame} mapping
    """
    carrier_data = {}

    for demand_file in demand_files:
        filename = str(demand_file)

        # Extract carrier and year from filename
        # Expected format: repo_data/MY_simpsec_demand/{scenario}/{carrier}_{year}.csv
        if "electricity" in filename:
            carrier = "electricity"
        elif "natural_gas" in filename:
            carrier = "natural_gas"
        elif "hydrogen" in filename:
            carrier = "hydrogen"
        elif "biomass" in filename:
            carrier = "biomass"
        else:
            logger.warning(f"Unknown carrier in filename {filename}")
            continue

        # Read the data
        try:
            df = pd.read_csv(demand_file, index_col=0)
            logger.info(f"Loaded raw {carrier} data from {demand_file}")

            if carrier not in carrier_data:
                carrier_data[carrier] = df
            else:
                # If multiple years, concatenate (though typically we'd have one file per year)
                carrier_data[carrier] = pd.concat([carrier_data[carrier], df])

        except Exception as e:
            logger.error(f"Error reading raw data file {demand_file}: {e}")
            raise

    return carrier_data


if __name__ == "__main__":
    if "snakemake" not in globals():
        # snakemake = mock_snakemake("add_demand", interconnect="usa")
        snakemake = mock_snakemake("add_demand", interconnect="usa", simpl="53", clusters="53")
    configure_logging(snakemake)

    demand_files = snakemake.input.demand
    n = pypsa.Network(snakemake.input.network)

    sectors = snakemake.params.sectors

    # add snapshots
    sns_config = snakemake.params.snapshots
    planning_horizons = snakemake.params.planning_horizons

    if list(n.snapshots) == ["now"]:
        # add snapshots
        n.snapshots = get_multiindex_snapshots(sns_config, planning_horizons)
        n.set_investment_periods(periods=planning_horizons)

    if isinstance(demand_files, str):
        demand_files = [demand_files]

    if sectors == "E" or sectors == "":  # electricity only
        assert len(demand_files) == 1

        suffix = ""
        carrier = "AC"

        df = pd.read_csv(demand_files[0], index_col=0)
        attach_demand(n, df, carrier, suffix)
        logger.info("Electricity demand added to network")

    elif sectors == "SimpSec":
        logger.info("Processing SimpSec demand")
        # Process raw data files directly
        carrier_data = process_simpsec_data(demand_files, planning_horizons)
        # Parse adj_scenario
        adj_scenario = snakemake.params.get("adj_scenario", "")
        fraction, adj_type, conversion_factor = parse_adj_scenario(adj_scenario)
        # Calculate hydrogen demand adjustments if needed
        total_h2_reduction_mwh = 0.0
        emission_cap_reduction = 0.0

        if fraction > 0.0 and adj_type is not None:
            logger.info(
                f"Applying adj_scenario: {adj_scenario} (fraction={fraction}, type={adj_type}, factor={conversion_factor})",
            )
            # Get original hydrogen demand
            h2_original = carrier_data["hydrogen"].copy()
            # Calculate total annual hydrogen reduction (across all states and timesteps)
            total_h2_reduction_mwh = h2_original.sum().sum() * fraction
            logger.info(f"Total hydrogen demand reduction: {total_h2_reduction_mwh:.2f} MWh/year")
            # Reduce hydrogen demand proportionally across all states and timesteps
            carrier_data["hydrogen"] = h2_original * (1.0 - fraction)
            # Calculate the reduction for each state and timestep (proportional)
            h2_reduction = h2_original * fraction

            if adj_type == "E":
                # Replace with electricity demand
                logger.info(f"Converting {fraction * 100}% H2 to electricity with factor {conversion_factor}")
                carrier_data["electricity"] = carrier_data["electricity"] + h2_reduction * conversion_factor
            elif adj_type == "G":
                # Replace with natural gas demand
                logger.info(f"Converting {fraction * 100}% H2 to natural gas")
                carrier_data["natural_gas"] = carrier_data["natural_gas"] + h2_reduction
                # Calculate emission cap reduction: MWh * 0.2002 ton/MWh * y
                emission_cap_reduction = total_h2_reduction_mwh * (0.2002 * conversion_factor + 0.0282)
                logger.info(f"Emission cap reduction (G): {emission_cap_reduction:.2f} ton CO2")
            elif adj_type == "O":
                # Just reduce hydrogen, no replacement
                logger.info(f"Reducing {fraction * 100}% H2 demand without replacement")
                # Calculate emission cap reduction: MWh * 0.27 ton/MWh * y
                emission_cap_reduction = total_h2_reduction_mwh * 0.27 * conversion_factor
                logger.info(f"Emission cap reduction (O): {emission_cap_reduction:.2f} ton CO2")

        # Store emission cap reduction in network attributes for use in solve_network
        n.adj_scenario_emission_reduction = emission_cap_reduction

        # Define carrier mappings and attach demand
        carrier_mappings = {
            "electricity": "AC",
            "natural_gas": "gas",
            "hydrogen": "h2",
        }

        for carrier_key, carrier_name in carrier_mappings.items():
            suffix = f" {carrier_name}"

            df = carrier_data[carrier_key]

            attach_demand(n, df, carrier_name, suffix)

        add_missing_carriers(n, ["gas", "h2"])

    else:  # detailed sector files
        for demand_file in demand_files:
            parsed_name = Path(demand_file).name.split("_")
            parsed_name[-1] = parsed_name[-1].split(".pkl")[0]

            if len(parsed_name) == 2:
                sector = parsed_name[0].upper()
                end_use = parsed_name[1].upper().replace("-", "_")

                sec_name = SecNames[sector].value
                sec_car = SecCarriers[end_use].value

                carrier = f"{sec_name}-{sec_car}"

                log_statement = f"{sector} {end_use} demand added to network"

            elif len(parsed_name) == 3:
                sector = parsed_name[0].upper()
                subsector = parsed_name[1].replace("-", "_")
                end_use = parsed_name[2].upper()  # lpg | elec

                sec_name = SecNames[sector].value
                sec_car = SecCarriers[end_use].value

                carrier = f"{sec_name}-{sec_car}-{VEHICLE_MAPPER[subsector]}"

                log_statement = f"{sector} {subsector} {end_use} demand added to network"

            else:
                raise NotImplementedError

            suffix = f"-{carrier}"

            df = pd.read_pickle(demand_file)
            attach_demand(n, df, carrier, suffix)
            logger.info(log_statement)

    n.export_to_netcdf(snakemake.output.network)
