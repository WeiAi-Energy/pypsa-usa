"""
Part 1: Add sector foundation and natural gas infrastructure
"""

import logging

import pandas as pd
import pypsa
from _helpers import configure_logging, get_snapshots, load_costs
from add_simple_sectors import (
    add_sector_foundation,
    assign_bus_2_state,
    attach_phs_storageunits,
    convert_generators_2_links,
    handle_nuclear_expansion,
    split_loads_by_carrier,
)
from build_emission_tracking import build_co2_tracking
from build_natural_gas import StateGeometry, build_natural_gas
from constants import CODE_2_STATE, STATES_INTERCONNECT_MAPPER

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_simple_sectors_part1",
            case="HighE_new_h2storage_tes",
            transmission_network="reeds",
        )
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)

    eia_api = snakemake.params.api["eia"]

    # Map states to each clustered bus
    states_2_map = [x for x, y in STATES_INTERCONNECT_MAPPER.items() if y in ("western", "eastern", "texas")]
    assign_bus_2_state(n, snakemake.input.county, states_2_map, CODE_2_STATE)

    sns = get_snapshots(snakemake.params.snapshots)

    costs = load_costs(snakemake.input.costs)
    addi_costs = load_costs(snakemake.input.addi_costs)

    # 1. Handle PHS expansion (moved from add_extra_components)
    elec_config = snakemake.params.electricity
    if any("PHS" in s for s in elec_config["PHS_exp"]):
        logger.info("Adding PHS storage units from add_extra_components")
        attach_phs_storageunits(n, elec_config, costs)

    # 2. Handle nuclear expansion (moved from prepare_network)
    nuclear_exp = snakemake.params.electricity.get("Nuclear_exp", True)
    logger.info(f"Nuclear expansion setting: {nuclear_exp}")
    handle_nuclear_expansion(n, nuclear_exp)

    ###
    # Sector addition starts here
    ###

    # Add sector specific emission tracking
    simplify_co2 = snakemake.params.sector["co2"]["simplify"]
    ts_cost = snakemake.params.sector["co2"]["ts_cost"]
    co2_storage_file = snakemake.input.co2_storage_file
    build_co2_tracking(
        n,
        simpsec=True,
        simplify_co2=simplify_co2,
        co2_storage_file=co2_storage_file,
        addi_costs=addi_costs,
    )

    # Break out loads into sector specific buses
    split_loads_by_carrier(n)

    # Add exogenous CCGT/OCGT at 2035
    ng_options = snakemake.params.sector["natural_gas"]
    new_gas_2035 = ng_options["new_gas_2035"]

    if new_gas_2035:
        logger.info("Adding 60 GW CCGT and 160 GW OCGT in 2035 as generators based on state distribution...")

        # Read distribution file
        distribution_df = pd.read_csv(snakemake.input.gas_distribution, index_col="state")

        # Total capacity to distribute
        total_ngcc = 60000  # MW
        total_ngct = 160000  # MW

        # Add CCGT generators
        for state in distribution_df.index:
            if distribution_df.loc[state, "CCGT_percent"] == 0:
                continue

            # Calculate capacity for this state
            state_capacity = total_ngcc * distribution_df.loc[state, "CCGT_percent"] / 100

            # Get the state electricity bus
            state_bus = state

            # Read parameters from reference generator
            ref_params = n.generators.loc[f"{state} CCGT_2050"]
            efficiency = ref_params["efficiency"]
            marginal_cost = ref_params["marginal_cost"]
            lifetime = ref_params["lifetime"]

            # Add generator
            gen_name = f"{state} CCGT_2035"
            n.add(
                "Generator",
                gen_name,
                bus=state_bus,
                carrier="CCGT",
                p_nom=state_capacity,
                p_nom_extendable=True,
                p_nom_max=state_capacity,
                capital_cost=costs.at["CCGT", "opex_fixed_per_kw"] * 1.127,
                efficiency=efficiency,
                marginal_cost=marginal_cost,
                build_year=2035,
                lifetime=lifetime,
            )

        # Add OCGT generators
        for state in distribution_df.index:
            if distribution_df.loc[state, "OCGT_percent"] == 0:
                continue

            # Calculate capacity for this state
            state_capacity = total_ngct * distribution_df.loc[state, "OCGT_percent"] / 100

            # Get the state bus
            state_bus = state

            # Read parameters from reference generator
            ref_params = n.generators.loc[f"{state} OCGT_2050"]
            efficiency = ref_params["efficiency"]
            marginal_cost = ref_params["marginal_cost"]
            lifetime = ref_params["lifetime"]

            # Add generator
            gen_name = f"{state} OCGT_2035"
            n.add(
                "Generator",
                gen_name,
                bus=state_bus,
                carrier="OCGT",
                p_nom=state_capacity,
                p_nom_extendable=True,
                p_nom_max=state_capacity,
                capital_cost=costs.at["OCGT", "opex_fixed_per_kw"] * 1.1,
                efficiency=efficiency,
                marginal_cost=marginal_cost,
                build_year=2035,
                lifetime=lifetime,
            )

    # Add primary energy carriers for each state
    center_points = StateGeometry(snakemake.input.county).state_center_points.set_index("STATE")

    h2_options = snakemake.params.sector.get("hydrogen", {})
    for carrier in ("gas", "h2", "biomass"):
        add_supply = True if carrier in ["coal"] else False
        api = eia_api if carrier in ["coal"] else None
        add_sector_foundation(n, carrier, add_supply, costs, center_points, api)

    # Convert generators to links
    for carrier in ("OCGT", "CCGT", "CCGT-95CCS", "CCGT-97CCS"):
        convert_generators_2_links(n, carrier, " gas", costs, simplify_co2, ts_cost)

    # Remove coal generators
    n.mremove("Generator", n.generators.index[n.generators.carrier == "coal"])

    # Convert biomass generators to links
    for carrier in ("biomass", "biomass-CCS"):
        convert_generators_2_links(n, carrier, " biomass", costs)

    # Add natural gas infrastructure and data
    years = ng_options["years"]

    build_natural_gas(
        n=n,
        years=years,
        api=eia_api,
        interconnect="usa",
        county_path=snakemake.input.county,
        pipelines_path=snakemake.input.pipeline_capacity,
        pipeline_shape_path=snakemake.input.pipeline_shape,
        storage_plant_path=snakemake.input.get("gas_storage_plants", None),
        options=ng_options,
        addi_costs=addi_costs,
    )

    # Apply cost factor to CCGT links
    ngcc_cost_factor = ng_options["ngcc_cost_factor"]
    ccgt_links = n.links[n.links.carrier.isin(["CCGT-95CCS", "CCGT-97CCS", "CCGT"])].index
    if len(ccgt_links) > 0:
        n.links.loc[ccgt_links, "capital_cost"] *= ngcc_cost_factor
        logger.info(f"Applied cost factor {ngcc_cost_factor} to {len(ccgt_links)} CCGT links")

    # Add exogenous CCGT/OCGT at 2035
    new_gas_2035 = ng_options["new_gas_2035"]
    if new_gas_2035:
        logger.info("Adding 60 GW NGCC and 160 GW NGCT in 2035 based on state distribution...")

        # Read distribution file
        distribution_df = pd.read_csv(snakemake.input.gas_distribution, index_col="state")

        # Get 2050 costs (assuming 2050 is the last planning horizon)
        costs_2050_file = snakemake.input.costs.replace(
            snakemake.params.scenario["planning_horizons"][0],
            snakemake.params.scenario["planning_horizons"][-1],
        )
        costs_2050 = load_costs(costs_2050_file)

        # Total capacity to distribute
        total_ngcc = 60000  # MW
        total_ngct = 160000  # MW

        # Get state center points for bus assignment
        from build_natural_gas import StateGeometry

        state_geom = StateGeometry(snakemake.input.county)
        state_centers = state_geom.state_center_points.set_index("STATE")

    logger.info("Part 1 complete: Exporting intermediate network")
    n.export_to_netcdf(snakemake.output.network)
