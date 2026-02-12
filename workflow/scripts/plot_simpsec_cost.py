"""
Plots cost analysis for SimpSec scenarios.

Creates a single stacked bar chart showing:
- Electricity system costs
- Natural gas system costs
- Hydrogen system costs

Each bar shows capital costs + marginal costs by technology.
Modified to calculate capital cost only for extendable components.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns
from _helpers import configure_logging

logger = logging.getLogger(__name__)

# Global plotting settings
TITLE_SIZE = 16
FIG_WIDTH = 12
FIG_HEIGHT = 8

# Set style for better visualization
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")


def calculate_capital_cost_extendable(
    component_df,
    cost_column="capital_cost",
    nom_column="p_nom",
    nom_opt_column="p_nom_opt",
    extendable_column="p_nom_extendable",
):
    """
    Calculate capital cost only for extendable components using the specified logic.

    Parameters
    ----------
    - component_df: DataFrame containing component data
    - cost_column: column name for capital cost
    - nom_column: column name for initial capacity (p_nom or e_nom)
    - nom_opt_column: column name for optimized capacity (p_nom_opt or e_nom_opt)
    - extendable_column: column name for extendable flag

    Returns
    -------
    - Series with calculated capital costs
    """
    # Filter only extendable components
    extendable_mask = component_df.get(extendable_column, False)
    extendable_components = component_df[extendable_mask]

    if extendable_components.empty:
        return pd.Series(dtype=float)

    capital_costs = []

    for idx, comp in extendable_components.iterrows():
        capital_cost = comp.get(cost_column, 0)
        nom = comp.get(nom_column, 0)
        nom_opt = comp.get(nom_opt_column, 0)

        if (
            "exist" not in idx
            and "gas production" not in idx
            and "gas pipeline" not in idx
            and "gas storage" not in idx
        ):
            # Calculate cost for new capacity only: capital_cost * (p_nom_opt - p_nom) or capital_cost * (e_nom_opt - e_nom)
            calculated_cost = capital_cost * (nom_opt - nom)
        else:
            # Calculate cost for optimized capacity: capital_cost * p_nom_opt or capital_cost * e_nom_opt
            calculated_cost = capital_cost * nom_opt

        capital_costs.append(calculated_cost)

    return pd.Series(capital_costs, index=extendable_components.index)


def calculate_storage_based_marginal_cost(n: pypsa.Network, carrier: str, link_idx: str) -> float:
    """
    Calculate storage-based marginal cost for coal and biomass power plants.

    For coal carriers: uses time-varying coal storage marginal cost
    For biomass carriers: uses weighted average biomass storage marginal cost based on annual consumption

    Args:
        n: PyPSA network
        carrier: Link carrier type (e.g., 'biomass', 'coal', 'biomass-CCS')
        link_idx: Link index

    Returns
    -------
        Total marginal cost based on storage consumption
    """
    # Get the link's time-series power data (p0 is the input power)
    if link_idx not in n.links_t.p0.columns:
        return 0.0

    link_power_timeseries = n.links_t.p0[link_idx]

    if link_power_timeseries.abs().sum() == 0:
        return 0.0

    # Determine storage type based on carrier
    if "coal" in carrier.lower():
        storage_carrier_pattern = "coal"
    elif "biomass" in carrier.lower():
        storage_carrier_pattern = "biomass"
    else:
        return 0.0

    # Find relevant storage stores
    relevant_stores = n.stores[n.stores.carrier.str.contains(storage_carrier_pattern, case=False, na=False)]

    if relevant_stores.empty:
        return 0.0

    # For biomass links, filter stores to only include those from the same state as the link
    if "biomass" in carrier.lower():
        # Extract state from the link index
        link_state = None
        if link_idx in n.links.index:
            link = n.links.loc[link_idx]
            bus0_name = link["bus0"]
            if " biomass" in bus0_name:
                link_state = bus0_name.replace(" biomass", "")

        # Filter stores to only include those from the same state as the link
        if link_state:
            state_specific_stores = relevant_stores[relevant_stores.index.str.startswith(f"{link_state} biomass")]
            if not state_specific_stores.empty:
                relevant_stores = state_specific_stores

    total_marginal_cost = 0.0

    # Handle coal storage (time-varying marginal cost)
    if "coal" in storage_carrier_pattern:
        for store_idx in relevant_stores.index:
            store = relevant_stores.loc[store_idx]

            if (
                hasattr(
                    n,
                    "stores_t",
                )
                and "marginal_cost" in n.stores_t
                and store_idx in n.stores_t.marginal_cost.columns
            ):
                # Time-varying marginal cost
                time_varying_cost = n.stores_t.marginal_cost[store_idx]
                cost_per_timestep = abs(link_power_timeseries) * time_varying_cost
                store_marginal_cost = cost_per_timestep.mean() * 8760
            else:
                # Fallback to constant marginal cost if no time series
                constant_marginal_cost = store.get("marginal_cost", 0)
                store_marginal_cost = abs(link_power_timeseries).mean() * constant_marginal_cost * 8760

            total_marginal_cost += store_marginal_cost

    # Handle biomass storage (weighted average marginal cost based on annual consumption)
    else:  # biomass case
        # First, calculate weighted average marginal cost based on annual consumption from all relevant stores
        total_consumption = 0.0
        weighted_marginal_cost_sum = 0.0

        for store_idx in relevant_stores.index:
            store = relevant_stores.loc[store_idx]
            constant_marginal_cost = store.get("marginal_cost", 0)

            if hasattr(n, "stores_t") and "e" in n.stores_t and store_idx in n.stores_t.e.columns:
                # Get the energy time series for this store
                store_energy_timeseries = n.stores_t.e[store_idx]

                # Calculate annual energy consumption from this store
                initial_energy = 0  # Biomass stores typically start at 0
                final_energy = store_energy_timeseries.iloc[-1]  # Last time step
                annual_consumption = abs(initial_energy - final_energy)  # Annual consumption from this store

                # Add to totals for weighted average calculation
                if annual_consumption > 0:  # Only include stores that are actually consumed
                    total_consumption += annual_consumption
                    weighted_marginal_cost_sum += annual_consumption * constant_marginal_cost

        # Calculate weighted average marginal cost
        if total_consumption > 0:
            weighted_average_marginal_cost = weighted_marginal_cost_sum / total_consumption
            # Calculate marginal cost for this specific link: mean_power * weighted_avg_marginal_cost * 8760
            total_marginal_cost = abs(link_power_timeseries).mean() * weighted_average_marginal_cost * 8760

    return total_marginal_cost


def get_electricity_costs(n: pypsa.Network) -> pd.Series:
    """
    Extract electricity system costs including generators, power plants, and storage.
    Returns Series with technology names and their total costs.
    Modified to calculate capital cost only for extendable components.
    Modified to use storage-based marginal costs for specific coal and biomass carriers.
    """
    costs_data = {}

    # 1. Generators (renewables and remaining conventional)
    annual_generation = n.generators_t.p.mean() * 8760

    # Calculate capital costs only for extendable generators
    gen_capital_costs = calculate_capital_cost_extendable(
        n.generators,
        "capital_cost",
        "p_nom",
        "p_nom_opt",
        "p_nom_extendable",
    )

    for idx, gen in n.generators.iterrows():
        carrier = gen["carrier"]
        # Merge wind technologies
        if carrier in ["onwind", "offwind_floating"]:
            carrier = "wind"

        # Use calculated capital cost for extendable components, 0 for others
        capital_cost = gen_capital_costs.get(idx, 0)
        marginal_cost = gen.get("marginal_cost", 0) * annual_generation.get(idx, 0)
        total_cost = capital_cost + marginal_cost

        if carrier in costs_data:
            costs_data[carrier] += total_cost
        else:
            costs_data[carrier] = total_cost

    # 2. Power plant links (CCGT, OCGT, coal, biomass, fuel cell, H2 turbine, LDES)
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
        "tes",
        "acaes retrofit",
        "acaes new",
    }

    power_links = n.links[n.links.carrier.isin(power_carriers)]
    annual_operation = n.links_t.p0[power_links.index].mean() * 8760

    # Calculate capital costs only for extendable links
    link_capital_costs = calculate_capital_cost_extendable(
        power_links,
        "capital_cost",
        "p_nom",
        "p_nom_opt",
        "p_nom_extendable",
    )

    # Define carriers that should use storage-based marginal cost calculation
    storage_based_carriers = {"coal", "coal-95CCS", "coal-99CCS", "biomass", "biomass-CCS"}

    for idx, link in power_links.iterrows():
        carrier = link["carrier"]
        # Simplify carrier names
        if ("CCGT" in carrier or "OCGT" in carrier) and "CCS" not in carrier:
            tech_name = "gas"
        elif ("CCGT" in carrier or "OCGT" in carrier) and "CCS" in carrier:
            tech_name = "gas-cc"
        elif "coal" in carrier and "CCS" not in carrier:
            tech_name = "coal"
        elif "coal" in carrier and "CCS" in carrier:
            tech_name = "coal-cc"
        elif "biomass" in carrier and "CCS" not in carrier:
            tech_name = "biomass"
        elif "biomass" in carrier and "CCS" in carrier:
            tech_name = "biomass-cc"
        elif "fuel cell" in carrier:
            tech_name = "fuel cell"
        elif "h2 turbine" in carrier:
            tech_name = "h2 turbine"
        elif "tes" in carrier:
            tech_name = "tes"
        elif "acaes" in carrier:
            tech_name = "acaes"
        else:
            tech_name = carrier

        # Use calculated capital cost for extendable components, 0 for others
        capital_cost = link_capital_costs.get(idx, 0)

        # Modified marginal cost calculation for specific carriers
        marginal_cost = link.get("marginal_cost", 0) * abs(annual_operation.get(idx, 0))
        if carrier in storage_based_carriers:
            # Add storage-based marginal cost calculation
            marginal_cost += calculate_storage_based_marginal_cost(n, carrier, idx)

        total_cost = capital_cost + marginal_cost

        if tech_name in costs_data:
            costs_data[tech_name] += total_cost
        else:
            costs_data[tech_name] = total_cost

    # 3. Storage units (battery, PHS)
    annual_storage_operation = n.storage_units_t.p.abs().mean() * 8760

    # Calculate capital costs only for extendable storage units
    storage_capital_costs = calculate_capital_cost_extendable(
        n.storage_units,
        "capital_cost",
        "p_nom",
        "p_nom_opt",
        "p_nom_extendable",
    )

    for idx, storage in n.storage_units.iterrows():
        carrier = storage["carrier"]
        if "battery" in carrier.lower():
            tech_name = "battery"
        elif carrier == "PHS":
            tech_name = "phs"
        else:
            tech_name = carrier

        # Use calculated capital cost for extendable components, 0 for others
        capital_cost = storage_capital_costs.get(idx, 0)
        marginal_cost = storage.get("marginal_cost", 0) * annual_storage_operation.get(idx, 0)
        total_cost = capital_cost + marginal_cost

        if tech_name in costs_data:
            costs_data[tech_name] += total_cost
        else:
            costs_data[tech_name] = total_cost

    # 4. Transmission infrastructure (both links and lines)
    # Transmission links (DC/AC links)
    trans_links = n.links[n.links.carrier.isin(["DC", "AC"])]
    trans_link_capital_costs = calculate_capital_cost_extendable(
        trans_links,
        "capital_cost",
        "p_nom",
        "p_nom_opt",
        "p_nom_extendable",
    )

    link_transmission_cost = 0
    for idx, link in trans_links.iterrows():
        capital_cost = trans_link_capital_costs.get(idx, 0)
        link_transmission_cost += capital_cost

    # Transmission lines (AC lines)
    trans_lines = n.lines.copy()
    trans_line_capital_costs = calculate_capital_cost_extendable(
        trans_lines,
        "capital_cost",
        "s_nom",
        "s_nom_opt",
        "s_nom_extendable",
    )

    line_transmission_cost = 0
    for idx, line in trans_lines.iterrows():
        capital_cost = trans_line_capital_costs.get(idx, 0)
        line_transmission_cost += capital_cost

    # Combine transmission costs
    total_transmission_cost = link_transmission_cost + line_transmission_cost

    tech_name = "transmission"
    if total_transmission_cost > 0:
        costs_data[tech_name] = total_transmission_cost

    return pd.Series(costs_data) / 1e9  # Convert to billion USD


def get_gas_costs(n: pypsa.Network) -> pd.Series:
    """
    Extract natural gas system costs.
    Modified to calculate capital cost only for extendable components.
    """
    costs_data = {}

    # 1. Gas production links
    gas_production_links = n.links[n.links.carrier == "gas production"]
    annual_operation = n.links_t.p0[gas_production_links.index].mean() * 8760

    # Calculate capital costs only for extendable gas production links
    prod_capital_costs = calculate_capital_cost_extendable(
        gas_production_links,
        "capital_cost",
        "p_nom",
        "p_nom_opt",
        "p_nom_extendable",
    )

    for idx, link in gas_production_links.iterrows():
        capital_cost = prod_capital_costs.get(idx, 0)
        marginal_cost = link.get("marginal_cost", 0) * annual_operation.get(idx, 0)
        total_cost = capital_cost + marginal_cost

        if "gas production" in costs_data:
            costs_data["gas production"] += total_cost
        else:
            costs_data["gas production"] = total_cost

    # 2. Gas bio-cc and methanation production links
    gas_biocc_carriers = {"gas bio-cc"}
    gas_methanation_carriers = {"gas methanation"}

    # Gas bio-cc
    gas_biocc_links = n.links[n.links.carrier.isin(gas_biocc_carriers)]
    if not gas_biocc_links.empty:
        annual_operation = n.links_t.p0[gas_biocc_links.index].mean() * 8760

        # Calculate capital costs only for extendable bio-cc links
        biocc_capital_costs = calculate_capital_cost_extendable(
            gas_biocc_links,
            "capital_cost",
            "p_nom",
            "p_nom_opt",
            "p_nom_extendable",
        )

        for idx, link in gas_biocc_links.iterrows():
            capital_cost = biocc_capital_costs.get(idx, 0)
            marginal_cost = link.get("marginal_cost", 0) * annual_operation.get(idx, 0)

            # Add storage-based marginal cost for biomass consumption
            marginal_cost += calculate_storage_based_marginal_cost(n, "biomass-CCS", idx)

            total_cost = capital_cost + marginal_cost

            if "gas bio-cc" in costs_data:
                costs_data["gas bio-cc"] += total_cost
            else:
                costs_data["gas bio-cc"] = total_cost

    # Gas methanation
    gas_methanation_links = n.links[n.links.carrier.isin(gas_methanation_carriers)]
    if not gas_methanation_links.empty:
        annual_operation = n.links_t.p0[gas_methanation_links.index].mean() * 8760

        # Calculate capital costs only for extendable methanation links
        methanation_capital_costs = calculate_capital_cost_extendable(
            gas_methanation_links,
            "capital_cost",
            "p_nom",
            "p_nom_opt",
            "p_nom_extendable",
        )

        for idx, link in gas_methanation_links.iterrows():
            capital_cost = methanation_capital_costs.get(idx, 0)
            marginal_cost = link.get("marginal_cost", 0) * annual_operation.get(idx, 0)

            total_cost = capital_cost + marginal_cost

            if "gas methanation" in costs_data:
                costs_data["gas methanation"] += total_cost
            else:
                costs_data["gas methanation"] = total_cost

    # 3. Gas pipelines
    gas_pipeline_links = n.links[n.links.carrier == "gas pipeline"]

    # Calculate capital costs only for extendable gas pipelines
    pipe_capital_costs = calculate_capital_cost_extendable(
        gas_pipeline_links,
        "capital_cost",
        "p_nom",
        "p_nom_opt",
        "p_nom_extendable",
    )

    for idx, link in gas_pipeline_links.iterrows():
        capital_cost = pipe_capital_costs.get(idx, 0)

        if "gas pipeline" in costs_data:
            costs_data["gas pipeline"] += capital_cost
        else:
            costs_data["gas pipeline"] = capital_cost

    # 4. Gas storage (stores + links)
    gas_storage_stores = n.stores[n.stores.carrier.str.contains("gas storage", case=False, na=False)]
    gas_storage_links = n.links[n.links.carrier.str.contains("gas storage", case=False, na=False)]

    # Storage stores - calculate capital costs only for extendable stores
    store_capital_costs = calculate_capital_cost_extendable(
        gas_storage_stores,
        "capital_cost",
        "e_nom",
        "e_nom_opt",
        "e_nom_extendable",
    )

    for idx, store in gas_storage_stores.iterrows():
        capital_cost = store_capital_costs.get(idx, 0)
        marginal_cost = store.get("marginal_cost", 0) * store.get("e_nom_opt", 0)
        total_cost = capital_cost + marginal_cost

        if "gas storage" in costs_data:
            costs_data["gas storage"] += total_cost
        else:
            costs_data["gas storage"] = total_cost

    # Storage operation links - calculate capital costs only for extendable links
    storage_link_capital_costs = calculate_capital_cost_extendable(
        gas_storage_links,
        "capital_cost",
        "p_nom",
        "p_nom_opt",
        "p_nom_extendable",
    )

    for idx, link in gas_storage_links.iterrows():
        capital_cost = storage_link_capital_costs.get(idx, 0)

        if "gas storage" in costs_data:
            costs_data["gas storage"] += capital_cost
        else:
            costs_data["gas storage"] = capital_cost

    return pd.Series(costs_data) / 1e9  # Convert to billion USD


def get_hydrogen_costs(n: pypsa.Network) -> pd.Series:
    """
    Extract hydrogen system costs.
    Modified to calculate capital cost only for extendable components.
    """
    costs_data = {}

    # 1. Hydrogen production links
    h2_production_carriers = {"h2 electrolysis", "h2 smr", "h2 smr-cc", "h2 bio", "h2 bio-cc"}
    h2_production_links = n.links[n.links.carrier.isin(h2_production_carriers)]

    annual_operation = n.links_t.p0[h2_production_links.index].mean() * 8760

    # Calculate capital costs only for extendable H2 production links
    h2_prod_capital_costs = calculate_capital_cost_extendable(
        h2_production_links,
        "capital_cost",
        "p_nom",
        "p_nom_opt",
        "p_nom_extendable",
    )

    # Define carriers that should use storage-based marginal cost calculation
    # For hydrogen production: h2 bio and h2 bio-cc use biomass storage
    storage_based_h2_carriers = {"h2 bio", "h2 bio-cc"}

    for idx, link in h2_production_links.iterrows():
        carrier = link["carrier"]
        capital_cost = h2_prod_capital_costs.get(idx, 0)

        # Regular marginal cost calculation
        marginal_cost = link.get("marginal_cost", 0) * annual_operation.get(idx, 0)

        # Add storage-based marginal cost for biomass-based hydrogen production
        if carrier in storage_based_h2_carriers:
            # Map hydrogen carrier to corresponding biomass storage type
            if "bio-cc" in carrier:
                storage_carrier = "biomass-CCS"  # For h2 bio-cc, use biomass-CCS storage pattern
            else:
                storage_carrier = "biomass"  # For h2 bio, use biomass storage pattern

            marginal_cost += calculate_storage_based_marginal_cost(n, storage_carrier, idx)

        total_cost = capital_cost + marginal_cost

        tech_name = link["carrier"]
        if tech_name in costs_data:
            costs_data[tech_name] += total_cost
        else:
            costs_data[tech_name] = total_cost

    # 2. Hydrogen pipelines
    h2_pipeline_carriers = ["h2 pipeline retrofit", "h2 pipeline new"]
    h2_pipeline_links = n.links[n.links.carrier.isin(h2_pipeline_carriers)]

    # Calculate capital costs only for extendable H2 pipelines
    h2_pipe_capital_costs = calculate_capital_cost_extendable(
        h2_pipeline_links,
        "capital_cost",
        "p_nom",
        "p_nom_opt",
        "p_nom_extendable",
    )

    for idx, link in h2_pipeline_links.iterrows():
        capital_cost = h2_pipe_capital_costs.get(idx, 0)

        tech_name = "h2 pipeline"  # Combine both types
        if tech_name in costs_data:
            costs_data[tech_name] += capital_cost
        else:
            costs_data[tech_name] = capital_cost

    # 3. Hydrogen storage (stores + links)
    h2_storage_stores = n.stores[n.stores.carrier.str.contains("h2 storage", case=False, na=False)]
    h2_storage_links = n.links[n.links.carrier.str.contains("h2 storage", case=False, na=False)]

    # Storage stores - calculate capital costs only for extendable stores
    h2_store_capital_costs = calculate_capital_cost_extendable(
        h2_storage_stores,
        "capital_cost",
        "e_nom",
        "e_nom_opt",
        "e_nom_extendable",
    )

    for idx, store in h2_storage_stores.iterrows():
        capital_cost = h2_store_capital_costs.get(idx, 0)
        marginal_cost = store.get("marginal_cost", 0) * store.get("e_nom_opt", 0)
        total_cost = capital_cost + marginal_cost

        if "h2 storage" in costs_data:
            costs_data["h2 storage"] += total_cost
        else:
            costs_data["h2 storage"] = total_cost

    # Storage operation links - calculate capital costs only for extendable links
    h2_storage_link_capital_costs = calculate_capital_cost_extendable(
        h2_storage_links,
        "capital_cost",
        "p_nom",
        "p_nom_opt",
        "p_nom_extendable",
    )

    for idx, link in h2_storage_links.iterrows():
        capital_cost = h2_storage_link_capital_costs.get(idx, 0)

        if "h2 storage" in costs_data:
            costs_data["h2 storage"] += capital_cost
        else:
            costs_data["h2 storage"] = capital_cost

    return pd.Series(costs_data) / 1e9  # Convert to billion USD


def get_co2_costs(n: pypsa.Network) -> pd.Series:
    """
    Extract CO2 system costs including DAC, CO2 sequestration, and CO2 pipelines.
    Each item considers both capital cost and marginal cost.
    """
    costs_data = {}

    # 1. DAC (Direct Air Capture) links
    dac_links = n.links[n.links.carrier == "dac"]
    if not dac_links.empty:
        annual_operation = n.links_t.p0[dac_links.index].mean() * 8760

        # Calculate capital costs only for extendable DAC links
        dac_capital_costs = calculate_capital_cost_extendable(
            dac_links,
            "capital_cost",
            "p_nom",
            "p_nom_opt",
            "p_nom_extendable",
        )

        for idx, link in dac_links.iterrows():
            capital_cost = dac_capital_costs.get(idx, 0)
            marginal_cost = link.get("marginal_cost", 0) * abs(annual_operation.get(idx, 0))
            total_cost = capital_cost + marginal_cost

            if "dac" in costs_data:
                costs_data["dac"] += total_cost
            else:
                costs_data["dac"] = total_cost

    # 2. CO2 sequestration stores
    co2_seq_stores = n.stores[n.stores.carrier.str.contains("co2 sequestration", case=False, na=False)]
    if not co2_seq_stores.empty:
        # Calculate capital costs only for extendable sequestration stores
        seq_capital_costs = calculate_capital_cost_extendable(
            co2_seq_stores,
            "capital_cost",
            "e_nom",
            "e_nom_opt",
            "e_nom_extendable",
        )

        for idx, store in co2_seq_stores.iterrows():
            capital_cost = seq_capital_costs.get(idx, 0)
            marginal_cost = store.get("marginal_cost", 0) * store.get("e_nom_opt", 0)
            total_cost = capital_cost + marginal_cost

            if "co2 sequestration" in costs_data:
                costs_data["co2 sequestration"] += total_cost
            else:
                costs_data["co2 sequestration"] = total_cost

    # 3. CO2 storage tank
    co2_capture_stores = n.stores[n.stores.carrier.str.contains("co2 capture", case=False, na=False)]
    if not co2_capture_stores.empty:
        # Calculate capital costs only for extendable sequestration stores
        seq_capital_costs = calculate_capital_cost_extendable(
            co2_capture_stores,
            "capital_cost",
            "e_nom",
            "e_nom_opt",
            "e_nom_extendable",
        )

        for idx, store in co2_capture_stores.iterrows():
            capital_cost = seq_capital_costs.get(idx, 0)
            marginal_cost = store.get("marginal_cost", 0) * store.get("e_nom_opt", 0)
            total_cost = capital_cost + marginal_cost

            if "co2 tank" in costs_data:
                costs_data["co2 tank"] += total_cost
            else:
                costs_data["co2 tank"] = total_cost

    # 4. CO2 pipeline links
    co2_pipeline_links = n.links[n.links.carrier.str.contains("co2 pipeline", case=False, na=False)]
    if not co2_pipeline_links.empty:
        # Calculate capital costs only for extendable CO2 pipelines
        co2_pipe_capital_costs = calculate_capital_cost_extendable(
            co2_pipeline_links,
            "capital_cost",
            "p_nom",
            "p_nom_opt",
            "p_nom_extendable",
        )

        for idx, link in co2_pipeline_links.iterrows():
            capital_cost = co2_pipe_capital_costs.get(idx, 0)
            # CO2 pipelines may also have marginal costs
            marginal_cost = link.get("marginal_cost", 0) * abs(n.links_t.p0.get(idx, pd.Series(0)).mean() * 8760)
            total_cost = capital_cost + marginal_cost

            if "co2 pipeline" in costs_data:
                costs_data["co2 pipeline"] += total_cost
            else:
                costs_data["co2 pipeline"] = total_cost

    return pd.Series(costs_data) / 1e9  # Convert to billion USD


def get_color_palette_cost() -> dict:
    """Get comprehensive color palette for cost plotting - consistent with other plot scripts."""
    return {
        # Renewables
        "solar": "#ffd700",  # Gold - consistent with other scripts
        "wind": "#c9e3c1",  # Green - consistent with other scripts
        "onwind": "#c9e3c1",  # Green - same as wind
        "offwind": "#c9e3c1",  # Green - same as wind
        "offwind_floating": "#c9e3c1",  # Green - same as wind
        "hydro": "#6BC5E6",  # Cyan - consistent with other scripts
        # Fossil fuels
        "gas": "#8B4513",  # SaddleBrown - consistent with other scripts
        "gas-cc": "#D2B48C",  # Tan - consistent with other scripts
        "coal": "#8c564b",  # Brown - consistent with other scripts
        "coal-cc": "#5a5a5a",  # DimGray - consistent with other scripts
        # Bioenergy
        "biomass": "#ff7f0e",  # Orange - consistent with other scripts
        "biomass-cc": "#9e9518",  # OrangeRed - consistent with other scripts
        # Nuclear
        "nuclear": "#b5260d",  # Red - consistent with other scripts
        # Storage
        "battery": "#1f77b4",  # Blue - consistent with other scripts
        "phs": "#aec7e8",  # LightSkyBlue - consistent with other scripts
        "tes": "#FF0001",
        "acaes": "#ff69b4",  # HotPink - consistent with other scripts
        # Transmission
        "transmission": "#7f8c8d",
        # Fuel cells and H2
        "fuel cell": "#ea048a",
        "h2 turbine": "#f5bfe7",
        # H2 production
        "h2 electrolysis": "#7FD12C",  # Lime - consistent with other scripts
        "h2 smr": "#0000ff",  # Blue - consistent with other scripts
        "h2 smr-cc": "#00BFFF",  # Navy - consistent with other scripts
        "h2 bio": "#228B22",  # SaddleBrown - consistent with other scripts
        "h2 bio-cc": "#006400",  # DarkOliveGreen - consistent with other scripts
        # H2 infrastructure
        "h2 pipeline": "#2ca02c",  # ForestGreen - consistent with other scripts
        "h2 storage": "#98df8a",  # ForestGreen - consistent with other scripts
        # Gas infrastructure
        "gas production": "#8B4513",  # SaddleBrown - consistent with other scripts
        "gas pipeline": "#ff8c00",
        "gas storage": "#ff7f0e",  # Orange - consistent with other scripts
        "gas bio-cc": "#2d5016",  # Deep green for gas bio-cc
        "gas methanation": "#98fb98",  # Light green for gas methanation
        # CO2 infrastructure
        "dac": "#9370DB",  # Medium purple for DAC
        "co2 sequestration": "#8B008B",  # Dark magenta for sequestration
        "co2 tank": "#DDA0DD",  # Plum for CO2 capture/storage
        "co2 pipeline": "#BA55D3",  # Medium orchid for CO2 pipeline
    }


def plot_system_costs(n: pypsa.Network, save_path: str, **kwargs) -> None:
    """
    Create system costs plot showing electricity, gas, and hydrogen costs by technology.
    Modified to only display costs and legends for items > 0.01% of total cost.
    Title is updated to reflect the sum of carrier costs.
    Legend is annotated with costs.
    Y-axis ticks are set to specific intervals.
    """
    logger.info("Creating system costs plot")

    # Extract cost data
    elec_costs = get_electricity_costs(n)
    gas_costs = get_gas_costs(n)
    h2_costs = get_hydrogen_costs(n)
    co2_costs = get_co2_costs(n)

    # Combine all costs to get a grand total and per-technology totals
    all_costs = pd.concat([elec_costs, gas_costs, h2_costs, co2_costs])
    tech_total_costs = all_costs.groupby(all_costs.index).sum()
    grand_total_cost = tech_total_costs.sum()

    # Define the threshold for displaying costs (0.01% of grand total)
    cost_threshold = grand_total_cost * 0.0001

    # Identify technologies that meet the threshold
    plottable_techs = tech_total_costs[tech_total_costs > cost_threshold].index.tolist()

    logger.info(f"Grand total cost: {grand_total_cost:.2f} billion. Plotting threshold: {cost_threshold:.4f} billion.")
    logger.info(f"Technologies to be plotted: {plottable_techs}")

    # Get color palette
    all_colors = get_color_palette_cost()

    # Prepare data for plotting
    left_categories = ["Electricity"] if not elec_costs.empty else []
    left_data = [elec_costs] if not elec_costs.empty else []

    right_categories = []
    right_data = []
    if not gas_costs.empty:
        right_categories.append("Gas")
        right_data.append(gas_costs)
    if not h2_costs.empty:
        right_categories.append("Hydrogen")
        right_data.append(h2_costs)
    if not co2_costs.empty:
        right_categories.append("CO2")
        right_data.append(co2_costs)

    if not left_categories and not right_categories:
        logger.warning("No cost data available for plotting")
        return

    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax2 = ax1.twinx()

    # Track legend elements
    legend_handles_dict = {}

    # Plot left axis data (Electricity)
    if left_categories:
        x_pos_left = [0]
        width = 0.6

        for i, (category, data) in enumerate(zip(left_categories, left_data)):
            bottom = 0
            total_cost = data.sum()
            sorted_data = data.sort_values(ascending=False)

            for tech, cost in sorted_data.items():
                if cost > 0 and tech in plottable_techs:
                    color = all_colors.get(tech, "#D3D3D3")
                    handle = ax1.bar(
                        x_pos_left[i],
                        cost,
                        bottom=bottom,
                        color=color,
                        width=width,
                        edgecolor="black",
                        linewidth=0.5,
                        alpha=1.0,
                    )

                    if tech not in legend_handles_dict:
                        legend_handles_dict[tech] = handle[0]

                    if total_cost > 0 and cost > total_cost * 0.01:
                        ax1.text(
                            x_pos_left[i],
                            bottom + cost / 2,
                            f"{cost:.1f}",
                            ha="center",
                            va="center",
                            fontweight="bold",
                            fontsize=9,
                        )
                    bottom += cost

            if bottom > 0:
                ax1.text(
                    x_pos_left[i],
                    bottom + bottom * 0.02,
                    f"${bottom:.1f}B",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=12,
                )

    # Plot right axis data (Gas and Hydrogen)
    if right_categories:
        start_pos = 1 if left_categories else 0
        x_pos_right = list(range(start_pos, start_pos + len(right_categories)))
        width = 0.6

        for i, (category, data) in enumerate(zip(right_categories, right_data)):
            bottom = 0
            total_cost = data.sum()
            sorted_data = data.sort_values(ascending=False)

            for tech, cost in sorted_data.items():
                if cost > 0 and tech in plottable_techs:
                    color = all_colors.get(tech, "#D3D3D3")
                    handle = ax2.bar(
                        x_pos_right[i],
                        cost,
                        bottom=bottom,
                        color=color,
                        width=width,
                        edgecolor="black",
                        linewidth=0.5,
                        alpha=1.0,
                    )

                    if tech not in legend_handles_dict:
                        legend_handles_dict[tech] = handle[0]

                    if total_cost > 0 and cost > total_cost * 0.01:
                        ax2.text(
                            x_pos_right[i],
                            bottom + cost / 2,
                            f"{cost:.1f}",
                            ha="center",
                            va="center",
                            fontweight="bold",
                            fontsize=9,
                        )
                    bottom += cost

            if bottom > 0:
                ax2.text(
                    x_pos_right[i],
                    bottom + bottom * 0.02,
                    f"${bottom:.1f}B",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=12,
                )

    # Customize axes
    all_categories = left_categories + right_categories
    all_x_pos = (x_pos_left if left_categories else []) + (x_pos_right if right_categories else [])

    ax1.set_xticks(all_x_pos)
    ax1.set_xticklabels(all_categories)

    ax1.set_ylim(0, 600)
    ax1.set_yticks(np.arange(0, 601, 100))  # Set left y-axis ticks
    ax1.set_ylabel("Electricity System Cost (Billion USD)", fontsize=12, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2.set_ylim(0, 150)
    ax2.set_yticks(np.arange(0, 151, 25))  # Set right y-axis ticks
    ax2.set_ylabel("Gas & Hydrogen & CO2 System Cost (Billion USD)", fontsize=12, color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Title
    total_cost_str = f"${grand_total_cost:.1f}B"
    ax1.set_title(
        f"System Costs by Sector (Total Cost: {total_cost_str})",
        fontsize=TITLE_SIZE,
    )

    # Grid
    ax1.grid(axis="y", alpha=0.3, color="blue", linestyle="-")
    ax2.grid(axis="y", alpha=0.3, color="red", linestyle="--")

    # Create unified, annotated, and sorted legend
    if legend_handles_dict:
        plotted_labels = list(legend_handles_dict.keys())

        sorted_legend_labels = sorted(
            plotted_labels,
            key=lambda tech: tech_total_costs[tech],
            reverse=True,
        )

        annotated_labels = [f"{tech}: {tech_total_costs[tech]:.2f} billion" for tech in sorted_legend_labels]

        handles = [legend_handles_dict[tech] for tech in sorted_legend_labels]
        ax1.legend(handles, annotated_labels, bbox_to_anchor=(1.15, 1), loc="upper left", title="Technology Costs")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"System costs plot saved to {save_path}")

    # Log summary
    total_all_costs = 0
    if not elec_costs.empty:
        elec_total = elec_costs.sum()
        logger.info(f"  Total electricity system cost: ${elec_total:.1f} billion (Left axis)")
        total_all_costs += elec_total

    if not gas_costs.empty:
        gas_total = gas_costs.sum()
        logger.info(f"  Total gas system cost: ${gas_total:.1f} billion (Right axis)")
        total_all_costs += gas_total

    if not h2_costs.empty:
        h2_total = h2_costs.sum()
        logger.info(f"  Total hydrogen system cost: ${h2_total:.1f} billion (Right axis)")
        total_all_costs += h2_total

    if not co2_costs.empty:
        co2_total = co2_costs.sum()
        logger.info(f"  Total CO2 system cost: ${co2_total:.1f} billion (Right axis)")
        total_all_costs += co2_total

    logger.info(f"  Grand total system cost: ${total_all_costs:.1f} billion")
    logger.info("Note: Color scheme is consistent with other PyPSA-USA plotting scripts")


def plot_lmp(n: pypsa.Network, save: str) -> None:
    """
    Plot violin plots of LMP distributions for electricity, gas, and hydrogen by state.

    Args:
        n: PyPSA network object
        save: Path to save the figure
    """
    # Get marginal prices
    df_lmp = n.buses_t.marginal_price

    # Prepare data for each energy carrier
    energy_carriers = ["AC", "gas", "h2", "biomass"]
    carrier_data = {}

    for carrier in energy_carriers:
        # Filter buses for this carrier
        if carrier == "AC":
            # For electricity, bus names are just state codes
            carrier_buses = n.buses[n.buses.carrier == "AC"].index
        else:
            # For gas and h2, bus names are "STATE carrier"
            carrier_buses = n.buses[n.buses.carrier == carrier].index

        if len(carrier_buses) == 0:
            print(f"Warning: No {carrier} buses found")
            continue

        # Get LMP data for these buses
        carrier_lmp = df_lmp[carrier_buses]

        # Convert to long format
        # The index of carrier_lmp is n.snapshots. Get its level names.
        id_vars = list(carrier_lmp.index.names)

        # In case the index is unnamed, reset_index() creates a column 'index'
        if id_vars == [None]:
            id_vars = ["index"]

        df_long = pd.melt(
            carrier_lmp.reset_index(),
            id_vars=id_vars,
            var_name="bus",
            value_name="lmp",
        )

        # Extract state from bus name
        if carrier == "AC":
            # For AC buses, bus name IS the state name directly
            df_long["state"] = df_long["bus"]
        else:
            # For gas and h2, extract state from bus name (format: "STATE carrier")
            df_long["state"] = df_long["bus"].str.split().str[0]

        # Remove any NaN states
        df_long = df_long.dropna(subset=["state"])

        # Store processed data
        carrier_data[carrier] = df_long

    # Create figure with subplots (4 rows, 1 column)
    fig, axes = plt.subplots(4, 1, figsize=(24, 18))

    # Define nice names for carriers
    carrier_names = {
        "AC": "Electricity",
        "gas": "Natural Gas",
        "h2": "Hydrogen",
        "biomass": "Biomass",
    }

    # Define units
    carrier_units = {
        "AC": "$/MWh_e",
        "gas": "$/MWh_th",
        "h2": "$/MWh_H2",
        "biomass": "$/MWh_th",
    }

    # Plot each carrier
    for idx, carrier in enumerate(energy_carriers):
        ax = axes[idx]
        if carrier not in carrier_data:
            ax.text(
                0.5,
                0.5,
                f"No {carrier_names[carrier]} data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{carrier_names[carrier]} LMP Distribution")
            continue

        df = carrier_data[carrier]

        # Calculate statistics for each state
        state_stats = (
            df.groupby("state")["lmp"]
            .agg(
                [
                    "mean",
                    ("p995", lambda x: np.percentile(x, 99.5)),
                ]
            )
            .reset_index()
        )

        # Sort states by mean price (high to low)
        state_stats = state_stats.sort_values("mean", ascending=False)
        state_order = state_stats["state"].tolist()

        # Calculate national average
        national_avg = df["lmp"].mean()

        # Create violin plot
        sns.violinplot(
            data=df,
            x="state",
            y="lmp",
            order=state_order,
            ax=ax,
            inner=None,
            cut=0,
            linewidth=0.5,
        )

        # Add mean and 99.5th percentile markers
        for i, state in enumerate(state_order):
            state_data = state_stats[state_stats["state"] == state].iloc[0]
            mean_val = state_data["mean"]
            p995_val = state_data["p995"]

            # Add mean marker
            ax.scatter(
                i,
                mean_val,
                color="red",
                s=30,
                zorder=5,
                marker="o",
                label="Mean" if i == 0 else "",
            )

            # Add 99.5th percentile marker
            ax.scatter(
                i,
                p995_val,
                color="blue",
                s=30,
                zorder=5,
                marker="^",
                label="99.5th percentile" if i == 0 else "",
            )

            # Add text annotations
            ax.text(
                i,
                mean_val,
                f" {mean_val:.1f}",
                fontsize=8,
                va="center",
                ha="left",
            )
            ax.text(
                i,
                p995_val,
                f" {p995_val:.1f}",
                fontsize=8,
                va="center",
                ha="left",
            )

        # Add national average line
        ax.axhline(
            y=national_avg,
            color="green",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"National avg: {national_avg:.1f}",
        )

        # Set labels and title
        ax.set_ylabel(f"LMP [{carrier_units[carrier]}]")
        ax.set_xlabel("State" if idx == 2 else "")  # Only show x-label for the bottom plot
        ax.set_title(f"{carrier_names[carrier]} Marginal Price Distribution")
        ax.tick_params(axis="x", rotation=45)

        # Add legend
        ax.legend(loc="upper right", fontsize=9)

        # Set y-axis limits to better show the distribution, ensuring all markers are visible.
        # Determine the maximum value for the y-axis.
        # It should be the larger of the 99.5% quantile or the highest state-level 99.5th percentile.
        ylim_max_quantile = df["lmp"].quantile(0.995)
        max_p995_marker = state_stats["p995"].max()

        # Set the final upper limit with a 10% buffer for text labels
        final_ylim_max = max(ylim_max_quantile, max_p995_marker) * 1.1

        ax.set_ylim(0, final_ylim_max)

        # Add grid for better readability
        ax.grid(True, axis="y", alpha=0.3)

    # Add overall title
    title = "State-Level Energy Marginal Price Distributions"

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(save, dpi=600, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_simpsec_cost",
            case="HighE_new_h2storage_tes",
            transmission_network="reeds",
        )

    configure_logging(snakemake)

    # Load network
    n = pypsa.Network(snakemake.input.network)

    # Extract wildcards for titles
    wildcards = dict(snakemake.wildcards)

    # Create system costs plot
    plot_system_costs(n, snakemake.output.costs, **wildcards)

    # Plot LMP distributions by state
    plot_lmp(n, snakemake.output.lmps)

    logger.info("System costs plot completed successfully")
