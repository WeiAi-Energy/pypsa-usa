"""Module for adding the hydrogen sector.

This module will add hydrogen infrastructure to the model.
Specifically, it will do the following:

- Creates electrolyzers for hydrogen production from electricity
- Creates SMR (Steam Methane Reforming) facilities for hydrogen production from gas
- Creates bioH2 facilities for hydrogen production from biomass
- Creates hydrogen storage facilities (underground and aboveground)
- Creates hydrogen pipelines (new and retrofitted from gas)
"""

import logging

import numpy as np
import pandas as pd
import pypsa
from constants import POWER_MAX, HHV_to_LHV_H2, discount_rate
from pypsa.geo import haversine_pts

logger = logging.getLogger(__name__)
from _helpers import calculate_annuity, get_currency_conversion_factor

###
# UTILITY FUNCTIONS
###


def create_network_topology(
    n,
    prefix,
    carriers=["DC"],
    connector=" -> ",
    bidirectional=True,
):
    """
    Create a network topology from transmission lines and link carrier selection.
    """
    ln_attrs = ["bus0", "bus1", "length"]
    lk_attrs = ["bus0", "bus1", "length", "underwater_fraction"]
    lk_attrs = n.links.columns.intersection(lk_attrs)

    candidates = pd.concat(
        [n.lines[ln_attrs], n.links.loc[n.links.carrier.isin(carriers), lk_attrs]],
    ).fillna(0)

    # base network topology purely on location not carrier
    candidates["bus0"] = candidates.bus0.map(n.buses.location)
    candidates["bus1"] = candidates.bus1.map(n.buses.location)

    positive_order = candidates.bus0 < candidates.bus1
    candidates_p = candidates[positive_order]
    swap_buses = {"bus0": "bus1", "bus1": "bus0"}
    candidates_n = candidates[~positive_order].rename(columns=swap_buses)
    candidates = pd.concat([candidates_p, candidates_n])

    def make_index(c):
        return prefix + c.bus0 + connector + c.bus1

    topo = candidates.groupby(["bus0", "bus1"], as_index=False).mean()
    topo.index = topo.apply(make_index, axis=1)

    if not bidirectional:
        topo_reverse = topo.copy()
        topo_reverse.rename(columns=swap_buses, inplace=True)
        topo_reverse.index = topo_reverse.apply(make_index, axis=1)
        topo = pd.concat([topo, topo_reverse])

    return topo


###
# HYDROGEN INFRASTRUCTURE
###
def build_electrolysis(n: pypsa.Network, **kwargs) -> None:
    """Add electrolysis facilities to network."""
    df = pd.DataFrame(
        {
            "state": n.buses["reeds_state"].replace("", np.nan).dropna().unique(),
        }
    )

    addi_costs = kwargs.get("addi_costs", None)
    cost_factor = kwargs.get("cost_factor", 1.0)
    efficiency_adjustment = kwargs.get("efficiency_adjustment", 0.0)

    elec_input = addi_costs.loc["electrolysis", "elec-input"]  # kWh_e/kWh_H2
    investment = addi_costs.loc["electrolysis", "investment"]  # USD/kW_e
    FOM = addi_costs.loc["electrolysis", "FOM"]  # %
    lifetime = addi_costs.loc["electrolysis", "lifetime"]
    currency_year = addi_costs.loc["electrolysis", "currency_year"]

    efficiency = 1 / elec_input + efficiency_adjustment
    capital_cost = (
        (calculate_annuity(lifetime, discount_rate) + FOM * 0.01)
        * investment
        * get_currency_conversion_factor(currency_year, "USD")
        * 1e3
        * cost_factor
    )  # USD/MW_el

    if "h2 electrolysis" not in n.carriers.index:
        n.add("Carrier", "h2 electrolysis", color="#ff29d9", nice_name="H2 Electrolysis")

    n.madd(
        "Link",
        names=df.state,
        suffix=" h2 electrolysis",
        bus0=[f"{state}" for state in df.state],
        bus1=[f"{state} h2" for state in df.state],
        carrier="h2 electrolysis",
        p_nom_extendable=True,
        p_nom_max=POWER_MAX,
        efficiency=efficiency,
        capital_cost=capital_cost,
        lifetime=lifetime,
        build_year=n.investment_periods[0],
    )


def build_smr(n: pypsa.Network, **kwargs) -> None:
    """Add smr facilities to network."""
    df = pd.DataFrame(
        {
            "state": n.buses["reeds_state"].replace("", np.nan).dropna().unique(),
        }
    )
    addi_costs = kwargs.get("addi_costs", None)
    simplify_co2 = kwargs.get("simplify_co2", True)
    ts_cost = kwargs.get("ts_cost", 20)

    # SMR-CC
    efficiency = addi_costs.loc["SMR-CCS", "efficiency"]  # MW_H2/MW_gas
    investment_cc = addi_costs.loc["SMR-CCS", "investment"]  # EUR/MW_CH4
    FOM_cc = addi_costs.loc["SMR-CCS", "FOM"]
    lifetime_cc = addi_costs.loc["SMR-CCS", "lifetime"]
    currency_year_cc = addi_costs.loc["SMR-CCS", "currency_year"]

    gas2co2_atmosphere_cc = 0.2002 * 0.1  # tCO2/MWh_gas, 90% CC rate
    gas2co2_capture_cc = 0.2002 * 0.9  # tCO2/MWh_gas, 90% CC rate
    capital_cost_cc = (
        (calculate_annuity(lifetime_cc, discount_rate) + FOM_cc * 0.01)
        * investment_cc
        * get_currency_conversion_factor(currency_year_cc, "EUR")
    )  # USD/MW_gas

    if "h2 smr-cc" not in n.carriers.index:
        n.add("Carrier", "h2 smr-cc", color="#ff5733", nice_name="H2 Steam Methane Reforming-CC")

    if simplify_co2:
        n.madd(
            "Link",
            names=[f"{state} h2 smr-cc" for state in df.state],
            bus0=[f"{state} gas" for state in df.state],
            bus1=[f"{state} h2" for state in df.state],
            bus2=[f"{state} co2 atmosphere" for state in df.state],
            carrier="h2 smr-cc",
            p_nom_extendable=True,
            p_nom_max=POWER_MAX,
            efficiency=efficiency,
            efficiency2=gas2co2_atmosphere_cc,
            capital_cost=capital_cost_cc,
            marginal_cost=ts_cost * gas2co2_capture_cc,
            lifetime=lifetime_cc,
            build_year=n.investment_periods[0],
        )
    else:
        n.madd(
            "Link",
            names=[f"{state} h2 smr-cc" for state in df.state],
            bus0=[f"{state} gas" for state in df.state],
            bus1=[f"{state} h2" for state in df.state],
            bus2=[f"{state} co2 atmosphere" for state in df.state],
            bus3=[f"{state} co2 capture" for state in df.state],
            carrier="h2 smr-cc",
            p_nom_extendable=True,
            p_nom_max=POWER_MAX,
            efficiency=efficiency,
            efficiency2=gas2co2_atmosphere_cc,
            efficiency3=gas2co2_capture_cc,
            capital_cost=capital_cost_cc,
            lifetime=lifetime_cc,
            build_year=n.investment_periods[0],
        )


def build_bioH2(n: pypsa.Network, **kwargs) -> None:
    """Add biomass H2 facilities to network."""
    df = pd.DataFrame(
        {
            "state": n.buses["reeds_state"].replace("", np.nan).dropna().unique(),
        }
    )
    addi_costs = kwargs.get("addi_costs", None)
    simplify_co2 = kwargs.get("simplify_co2", True)
    ts_cost = kwargs.get("ts_cost", 20)

    # bioH2-CC
    bio_input_cc = (
        addi_costs.loc["H2 production biomass gasification CC", "wood-input"] * HHV_to_LHV_H2
    )  # MWh_wood_HHV/MWh_H2_HHV * MWh_H2_HHV/MWh_H2_LHV
    elec_input_cc = (
        addi_costs.loc["H2 production biomass gasification CC", "electricity-input"] * HHV_to_LHV_H2
    )  # MWh_el/MWh_H2_HHV * MWh_H2_HHV/MWh_H2_LHV
    investment_cc = (
        addi_costs.loc["H2 production biomass gasification CC", "investment"] * HHV_to_LHV_H2
    )  # USD/kW_H2_HHV * kW_H2_HHV/kW_H2_LHV
    FOM_cc = addi_costs.loc["H2 production biomass gasification CC", "FOM"]  # %/year
    VOM_cc = (
        addi_costs.loc["H2 production biomass gasification CC", "VOM"] * HHV_to_LHV_H2
    )  # USD/MWh_H2_HHV * MWh_H2_HHV/MWh_H2_LHV
    lifetime_cc = addi_costs.loc["H2 production biomass gasification CC", "lifetime"]
    currency_year = addi_costs.loc["H2 production biomass gasification CC", "currency_year"]

    bio2H2_cc = 1.0 / bio_input_cc  # MWh_H2/MWh_bio
    bio2elec_cc = -elec_input_cc / bio_input_cc  # MWh_el/MWh_bio
    bio2co2_capture_cc = 0.2734  # tCO2/MWh_bio, NZA
    capital_cost_cc = (
        (calculate_annuity(lifetime_cc, discount_rate) + FOM_cc * 0.01)
        * investment_cc
        * bio2H2_cc
        * 1e3
        * get_currency_conversion_factor(currency_year, "USD")
    )  # USD/MW_bio
    marginal_cost_cc = VOM_cc * bio2H2_cc * get_currency_conversion_factor(currency_year, "USD")  # USD/MWh_bio

    if "h2 bio-cc" not in n.carriers.index:
        n.add("Carrier", "h2 bio-cc", color="#654321", nice_name="H2 Biomass-CC")

    if simplify_co2:
        n.madd(
            "Link",
            names=[f"{state} h2 bio-cc" for state in df.state],
            bus0=[f"{state} biomass" for state in df.state],
            bus1=[f"{state} h2" for state in df.state],
            bus2=[f"{state}" for state in df.state],
            bus3=[f"{state} co2 atmosphere" for state in df.state],
            carrier="h2 bio-cc",
            p_nom_extendable=True,
            p_nom_max=POWER_MAX,
            efficiency=bio2H2_cc,
            efficiency2=bio2elec_cc,
            efficiency3=-bio2co2_capture_cc,
            capital_cost=capital_cost_cc,
            marginal_cost=marginal_cost_cc + ts_cost * bio2co2_capture_cc,
            lifetime=lifetime_cc,
            build_year=n.investment_periods[0],
        )
    else:
        n.madd(
            "Link",
            names=[f"{state} h2 bio-cc" for state in df.state],
            bus0=[f"{state} biomass" for state in df.state],
            bus1=[f"{state} h2" for state in df.state],
            bus2=[f"{state}" for state in df.state],
            bus3=[f"{state} co2 atmosphere" for state in df.state],
            bus4=[f"{state} co2 capture" for state in df.state],
            carrier="h2 bio-cc",
            p_nom_extendable=True,
            p_nom_max=POWER_MAX,
            efficiency=bio2H2_cc,
            efficiency2=bio2elec_cc,
            efficiency3=-bio2co2_capture_cc,
            efficiency4=bio2co2_capture_cc,
            capital_cost=capital_cost_cc,
            marginal_cost=marginal_cost_cc,
            lifetime=lifetime_cc,
            build_year=n.investment_periods[0],
        )


def build_h2p(n: pypsa.Network, **kwargs) -> None:
    """Add H2 to power to network."""
    df = pd.DataFrame(
        {
            "state": n.buses["reeds_state"].replace("", np.nan).dropna().unique(),
        }
    )

    costs = kwargs.get("costs", None)
    cost_factor = kwargs.get("cost_factor", 1.0)
    h2_options = kwargs.get("h2_options", {})
    retro_turbine = h2_options.get("retro_turbine", False)
    lifetime_extension = h2_options.get("lifetime_extension", 0)

    efficiency_ht = costs.loc["hydrogen_ct", "efficiency"] * HHV_to_LHV_H2
    investment_ht = costs.loc["hydrogen_ct", "annualized_capex_fom"] * cost_factor  # USD/MW_el
    lifetime_ht = costs.loc["hydrogen_ct", "lifetime"]
    VOM_ht = costs.loc["hydrogen_ct", "opex_variable_per_mwh"]

    if "h2 turbine" not in n.carriers.index:
        n.add("Carrier", "h2 turbine", color="#ea048a", nice_name="H2 Turbine")

    # Add new H2 turbines
    n.madd(
        "Link",
        names=df.state,
        suffix=" h2 turbine",
        bus0=[f"{state} h2" for state in df.state],
        bus1=[f"{state}" for state in df.state],
        carrier="h2 turbine",
        p_nom_extendable=True,
        p_nom_max=POWER_MAX,
        efficiency=efficiency_ht,
        capital_cost=investment_ht * efficiency_ht,
        marginal_cost=VOM_ht * efficiency_ht,
        lifetime=lifetime_ht,
        build_year=n.investment_periods[0],
    )

    # Add retrofit H2 turbines from existing CCGT/OCGT if enabled
    if retro_turbine:
        logger.info("Adding retrofit H2 turbines from existing CCGT/OCGT")

        # Find existing CCGT and OCGT links (without CCS) with build_year < 2050
        ccgt_links = n.links[
            (n.links.carrier == "CCGT")  # Only CCGT without CCS
            & (n.links.build_year < 2050)
        ]
        ocgt_links = n.links[(n.links.carrier == "OCGT") & (n.links.build_year < 2050)]

        # CCGT retrofit parameters
        ccgt_capex_base = 312662  # USD/MW
        ccgt_fom = 26700  # USD/MW/year
        ccgt_efficiency = 0.6144
        ccgt_vom = 1.79  # USD/MWh

        # OCGT retrofit parameters
        ocgt_capex_base = 345968  # USD/MW
        ocgt_fom = 22000  # USD/MW/year
        ocgt_efficiency = 0.3893
        ocgt_vom = 6.94  # USD/MWh

        discount_rate_retro = 0.054

        # Add CCGT retrofit links
        for link_idx in ccgt_links.index:
            link = ccgt_links.loc[link_idx]
            build_year = int(link.build_year)
            original_lifetime = int(link.lifetime)

            # Calculate remaining lifetime until 2050
            years_until_retire = build_year + original_lifetime - 2050 + lifetime_extension

            # Calculate capital cost
            annuity_factor = calculate_annuity(min(years_until_retire, 30), discount_rate_retro)
            capital_cost = (ccgt_capex_base * annuity_factor + ccgt_fom) * ccgt_efficiency
            marginal_cost = ccgt_vom * ccgt_efficiency

            # Create retrofit link name
            retrofit_name = f"{link_idx} retrofit"

            n.add(
                "Link",
                retrofit_name,
                bus0=link.bus0.replace(" gas", " h2"),  # Connect to H2 bus
                bus1=link.bus1,  # Same output bus as original
                carrier="h2 turbine",
                p_nom_extendable=True,
                p_nom=0,
                p_nom_min=0,
                p_nom_max=link.p_nom,  # Limited by original capacity
                efficiency=ccgt_efficiency,
                capital_cost=capital_cost,
                marginal_cost=marginal_cost,
                build_year=build_year,
            )

        # Add OCGT retrofit links
        for link_idx in ocgt_links.index:
            link = ocgt_links.loc[link_idx]
            build_year = int(link.build_year)
            original_lifetime = int(link.lifetime)

            # Calculate remaining lifetime until 2050
            years_until_retire = build_year + original_lifetime - 2050 + lifetime_extension

            # Calculate capital cost
            annuity_factor = calculate_annuity(min(years_until_retire, 30), discount_rate_retro)
            capital_cost = (ocgt_capex_base * annuity_factor + ocgt_fom) * ocgt_efficiency
            marginal_cost = ocgt_vom * ocgt_efficiency

            # Create retrofit link name
            retrofit_name = f"{link_idx} retrofit"

            n.add(
                "Link",
                retrofit_name,
                bus0=link.bus0.replace(" gas", " h2"),  # Connect to H2 bus
                bus1=link.bus1,  # Same output bus as original
                carrier="h2 turbine",
                p_nom_extendable=True,
                p_nom=0,
                p_nom_min=0,
                p_nom_max=link.p_nom,  # Limited by original capacity
                efficiency=ocgt_efficiency,
                capital_cost=capital_cost,
                marginal_cost=marginal_cost,
                build_year=build_year,
            )

        logger.info(f"Added {len(ccgt_links)} CCGT retrofit and {len(ocgt_links)} OCGT retrofit H2 turbines")


def build_new_h2storage(n: pypsa.Network, **kwargs) -> None:
    """
    Add new H2 storage (cavern) for each state with unlimited capacity.

    This function adds new hydrogen storage infrastructure including:
    - Storage buses for each state
    - Storage units with unlimited extendable capacity
    - Charge and discharge links connecting to H2 buses
    """
    # Create DataFrame with all states in the network
    df = pd.DataFrame(
        {
            "state": n.buses["reeds_state"].replace("", np.nan).dropna().unique(),
        }
    )
    df.index = df.state

    # Get additional costs data
    addi_costs = kwargs.get("addi_costs", None)
    cost_factor = kwargs.get("cost_factor", 1)

    # Read cost parameters from addi_costs
    comp_elec_input = addi_costs.loc["hydrogen storage compressor", "compression-electricity-input"]  # MWh_el/MWh_H2
    comp_well_investment = addi_costs.loc["hydrogen storage well and compressor new", "investment"]  # USD/kW_H2_dis
    comp_fom = addi_costs.loc["hydrogen storage compressor", "FOM"]  # %/year
    comp_well_currency_year = addi_costs.loc["hydrogen storage well and compressor new", "currency_year"]

    # Read storage investment cost (USD/kWh_H2)
    storage_investment = (
        addi_costs.loc["hydrogen storage underground cavern new", "investment"] * cost_factor
    )  # USD/kWh_H2
    storage_lifetime = addi_costs.loc["hydrogen storage underground", "lifetime"]  # years
    storage_currency_year = addi_costs.loc["hydrogen storage underground", "currency_year"]

    # Calculate capital costs
    # Charge link capital cost (USD/MW_H2_char)
    charge_link_capital_cost = (
        (calculate_annuity(storage_lifetime, discount_rate) + comp_fom * 0.01)
        * comp_well_investment
        * 1e3
        * get_currency_conversion_factor(comp_well_currency_year, "USD")
    )

    # Storage capital cost (USD/MWh_H2)
    storage_capital_cost = (
        calculate_annuity(storage_lifetime, discount_rate)
        * storage_investment
        * 1e3
        * get_currency_conversion_factor(storage_currency_year, "USD")
    )

    # Hard-coded parameters
    e_min_pu = 0.2932  # taken from existing cavern gas storage

    # Add carrier if not exists
    if "h2 storage new" not in n.carriers.index:
        n.add("Carrier", "h2 storage new", color="#ea048a", nice_name="H2 Storage New")

    # Add hydrogen storage buses for each state
    n.madd(
        "Bus",
        names=df.index,
        suffix=" h2 storage new",
        carrier="h2 storage new",
        unit="MWh_th",
        country=df.index,
        reeds_state=df.index,
    )

    # Add hydrogen storage stores with unlimited extendable capacity
    n.madd(
        "Store",
        names=df.index,
        suffix=" h2 storage new",
        bus=df.index + " h2 storage new",
        carrier="h2 storage new",
        e_nom_extendable=True,
        e_nom=0,
        e_nom_min=0,
        e_cyclic=True,
        e_min_pu=e_min_pu,
        e_max_pu=1,
        marginal_cost=0,
        capital_cost=storage_capital_cost,
        standing_loss=8.0e-6,  # 1%/month
        lifetime=storage_lifetime,
        build_year=n.investment_periods[0],
    )

    # Add charge links (from H2 bus to storage bus)
    n.madd(
        "Link",
        names=df.index,
        suffix=" charge h2 storage new",
        carrier="h2 storage new",
        bus0=df.index + " h2",  # From state H2 bus
        bus1=df.index + " h2 storage new",  # To storage bus
        bus2=df.index,  # Electricity consumption for compression
        efficiency=1,  # H2 storage efficiency
        efficiency2=-comp_elec_input,  # Electricity consumption (negative)
        p_nom_extendable=True,
        p_nom_max=POWER_MAX,
        p_nom=0,
        p_nom_min=0,
        p_min_pu=0,
        p_max_pu=1,
        marginal_cost=0,
        capital_cost=charge_link_capital_cost,
        lifetime=storage_lifetime,
        build_year=n.investment_periods[0],
    )

    # Add discharge links (from storage bus to H2 bus)
    n.madd(
        "Link",
        names=df.index,
        suffix=" discharge h2 storage new",
        carrier="h2 storage new",
        bus0=df.index + " h2 storage new",  # From storage bus
        bus1=df.index + " h2",  # To state H2 bus
        efficiency=1,  # H2 storage efficiency
        p_nom_extendable=True,
        p_nom_max=POWER_MAX,
        p_nom=0,
        p_nom_min=0,
        p_min_pu=0,
        p_max_pu=1,
        marginal_cost=0,
        capital_cost=0,
        lifetime=storage_lifetime,
        build_year=n.investment_periods[0],
    )


def _calculate_pipeline_distances(df: pd.DataFrame, n: pypsa.Network) -> pd.Series:
    """
    Calculate distances between state pairs for H2 pipelines.

    Args:
        df: Pipeline data with STATE_FROM and STATE_TO columns
        n: PyPSA network containing bus coordinates

    Returns
    -------
        Series of distances in kilometers indexed by pipeline names
    """
    # Get H2 bus coordinates from the network
    buses = n.buses[n.buses.index == n.buses.reeds_state]

    state_coords = buses.set_index("reeds_state")[["x", "y"]]

    # Calculate distances between state pairs using haversine formula
    distances = []
    for _, row in df.iterrows():
        state_from = row["STATE_FROM"]
        state_to = row["STATE_TO"]

        # Get coordinates for both states (longitude, latitude)
        lon1, lat1 = state_coords.loc[state_from, ["x", "y"]]
        lon2, lat2 = state_coords.loc[state_to, ["x", "y"]]

        # Calculate distance using haversine formula
        distance_km = haversine_pts([lon1, lat1], [lon2, lat2])
        distances.append(distance_km)

    return pd.Series(distances, index=df.index)


def _calculate_pipeline_capital_costs(
    df: pd.DataFrame,
    n: pypsa.Network,
    fom: float,
    investment: float,
    lifetime: float,
    length_factor: float,
    currency_year: int,
) -> pd.Series:
    """
    Calculate capital costs for H2 pipelines based on distance and cost parameters.
    Capital cost = distance × (annualized_investment + fom) × EUR_2_USD (in $/MW)
    """
    distances = _calculate_pipeline_distances(df, n) * length_factor

    # Calculate total annualized cost (investment + FOM)
    conversion_factor = get_currency_conversion_factor(currency_year, "EUR")
    total_annual_cost = (calculate_annuity(lifetime, discount_rate) + fom * 0.01) * investment * conversion_factor
    capital_costs = distances * total_annual_cost
    return capital_costs


def _calculate_efficiency(
    df: pd.DataFrame,
    n: pypsa.Network,
    elec_input: float,
    length_factor: float,
) -> pd.Series:
    """
    Calculate elec efficiency for H2 pipeline.
    """
    distances = _calculate_pipeline_distances(df, n) * length_factor

    # Leakage + Compression
    # Leakage rate: https://www.nature.com/articles/s41560-025-01752-6#Sec14
    # Compression drive efficiency: https://netl.doe.gov/sites/default/files/netl-file/Brun.pdf
    efficiency = 1 - 0.012 * distances / 1000 - elec_input * distances / 1000 / 0.325

    return efficiency


def build_new_h2pipeline(n: pypsa.Network, **kwargs) -> None:
    """
    Add new H2 pipelines between state pairs based on AC transmission lines.

    Args:
        n: PyPSA network
        **kwargs: Additional parameters including addi_costs, length_factor
    """
    # 1. Extract state pairs from AC links and gas pipeline links
    ac_links = n.links[n.links.carrier.isin(["AC"])]
    ac_lines = n.lines[n.lines.carrier.isin(["AC"])]
    gas_links = n.links[n.links.carrier == "gas pipeline"]

    # Combine both link types
    combined_links = pd.concat([ac_links, ac_lines, gas_links], ignore_index=True)

    state_pairs = set()
    for _, row in combined_links.iterrows():
        # Get states for bus0 and bus1
        bus0_state = n.buses.loc[row.bus0, "reeds_state"]
        bus1_state = n.buses.loc[row.bus1, "reeds_state"]

        if (
            pd.notna(bus0_state) and pd.notna(bus1_state) and bus0_state != bus1_state and bus0_state and bus1_state
        ):  # Check they're not empty strings
            # Ensure consistent ordering to avoid duplicates
            pair = tuple(sorted([bus0_state, bus1_state]))
            state_pairs.add(pair)

    logger.info(f"Found {len(state_pairs)} state pairs for new H2 pipelines from AC and gas pipeline topology")

    # 2. Create dataframe with state pairs
    df = pd.DataFrame(list(state_pairs), columns=["STATE_FROM", "STATE_TO"])
    df.index = df.STATE_FROM + " " + df.STATE_TO + " h2 pipeline new"

    # 3. Calculate distances and costs
    addi_costs = kwargs.get("addi_costs", None)
    length_factor = kwargs.get("length_factor", 1.25)

    h2_fom = addi_costs.loc["H2 (g) pipeline", "FOM"]
    h2_investment = addi_costs.loc["H2 (g) pipeline", "investment"]  # EUR/MW/km
    h2_lifetime = addi_costs.loc["H2 (g) pipeline", "lifetime"]
    h2_elec_input = addi_costs.loc["H2 (g) pipeline", "electricity-input"]  # MW_e/1000km/MW_H2
    currency_year = addi_costs.loc["H2 (g) pipeline", "currency_year"]

    # Calculate capital costs and electricity efficiency
    distances = _calculate_pipeline_distances(df, n) * length_factor
    capital_costs = _calculate_pipeline_capital_costs(
        df, n, h2_fom, h2_investment, h2_lifetime, length_factor, currency_year
    )
    efficiency = _calculate_efficiency(df, n, h2_elec_input, length_factor)

    # 4. Add carrier if not exists
    if "h2 pipeline new" not in n.carriers.index:
        n.add("Carrier", "h2 pipeline new", color="#ea048a", nice_name="H2 Pipeline New")

    # 5. Add H2 pipelines
    n.madd(
        "Link",
        names=df.index,
        suffix="_fwd",
        carrier="h2 pipeline new",
        bus0=df.STATE_FROM + " h2",
        bus1=df.STATE_TO + " h2",
        p_nom_extendable=True,
        p_nom=0,
        p_nom_min=0,
        p_nom_max=POWER_MAX,
        p_min_pu=0,
        p_max_pu=1,
        efficiency=efficiency,
        capital_cost=capital_costs / 2,
        length=distances,
        marginal_cost=0,
        lifetime=h2_lifetime,
        build_year=n.investment_periods[0],
    )

    n.madd(
        "Link",
        names=df.index,
        suffix="_rev",
        carrier="h2 pipeline new",
        bus0=df.STATE_TO + " h2",
        bus1=df.STATE_FROM + " h2",
        p_nom_extendable=True,
        p_nom=0,
        p_nom_min=0,
        p_nom_max=POWER_MAX,
        p_min_pu=0,
        p_max_pu=1,
        efficiency=efficiency,
        capital_cost=capital_costs / 2,
        length=distances,
        marginal_cost=0,
        lifetime=h2_lifetime,
        build_year=n.investment_periods[0],
    )


###
# MAIN FUNCTION
###


def build_hydrogen(
    n: pypsa.Network,
    h2_options: dict[str, any] | None = None,
    costs: pd.DataFrame | None = None,
    addi_costs: pd.DataFrame | None = None,
    simplify_co2: bool = True,
    ts_cost: float = 20,
) -> None:
    """
    Main function to build hydrogen infrastructure.
    """
    if not h2_options:
        h2_options = {}

    logger.info("Building hydrogen infrastructure")

    # 1. Add electrolysis
    logger.info("Adding hydrogen electrolysis")
    cost_factor = h2_options["electrolysis_cost_factor"]
    efficiency_adjustment = h2_options["electrolysis_efficiency_adjustment"]
    build_electrolysis(n, addi_costs=addi_costs, cost_factor=cost_factor, efficiency_adjustment=efficiency_adjustment)

    # 2. Add SMR-CC
    logger.info("Adding hydrogen SMR-CC")
    build_smr(n, addi_costs=addi_costs, simplify_co2=simplify_co2, ts_cost=ts_cost)

    # 3. Add bioH2-CC if enabled
    if h2_options.get("bioH2", True):
        logger.info("Adding biological hydrogen production")
        build_bioH2(n, addi_costs=addi_costs, simplify_co2=simplify_co2, ts_cost=ts_cost)

    # 4. Add H2 combustion turbine
    logger.info("Adding hydrogen turbine")
    cost_factor = h2_options["turbine_cost_factor"]
    build_h2p(n, costs=costs, addi_costs=addi_costs, cost_factor=cost_factor, h2_options=h2_options)

    # 5. Add new H2 storage
    if h2_options.get("new_storage", True):
        logger.info("Adding new H2 storage")
        new_storage_cost_factor = h2_options["new_storage_cost_factor"]
        build_new_h2storage(n, addi_costs=addi_costs, cost_factor=new_storage_cost_factor)

    # 6. Add new H2 pipeline
    if h2_options.get("new_pipeline", True):
        logger.info("Adding new H2 pipeline")
        build_new_h2pipeline(n, addi_costs=addi_costs)
