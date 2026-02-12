"""
Part 2: Add hydrogen, biomass, DAC, CO2 sequestration, LDES, and apply derates
"""

import logging

import pypsa
from _helpers import configure_logging, load_costs
from add_electricity import sanitize_carriers
from add_simple_sectors import (
    apply_derates_to_new_components,
    build_biomass,
    build_co2_pipeline,
    build_co2_sequestration,
    build_dac,
    build_ldes,
    load_derates_from_csv,
)
from build_hydrogen import build_hydrogen

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_simple_sectors_part2",
            case="HighE_power_new_h2storage_tes",
            transmission_network="reeds",
        )
    configure_logging(snakemake)

    # Load the intermediate network from part 1
    n = pypsa.Network(snakemake.input.network)

    costs = load_costs(snakemake.input.costs)
    addi_costs = load_costs(snakemake.input.addi_costs)

    # Get parameters
    simplify_co2 = snakemake.params.sector["co2"]["simplify"]
    ts_cost = snakemake.params.sector["co2"]["ts_cost"]
    h2_options = snakemake.params.sector.get("hydrogen", {})

    # Biomass potential files
    agri_potential = snakemake.input.agri
    forestry_potential = snakemake.input.forestry
    wastes_potential = snakemake.input.wastes

    # Add hydrogen infrastructure and data
    build_hydrogen(
        n=n,
        h2_options=h2_options,
        costs=costs,
        addi_costs=addi_costs,
        simplify_co2=simplify_co2,
        ts_cost=ts_cost,
    )

    # Get reference network path for biomass (if exists)
    reference_network_path = getattr(snakemake.input, "reference_network", None)
    if isinstance(reference_network_path, list) and len(reference_network_path) == 0:
        reference_network_path = None

    # Add biomass production potential
    build_biomass(
        n=n,
        agri_potential=agri_potential,
        forestry_potential=forestry_potential,
        wastes_potential=wastes_potential,
        reference_network_path=reference_network_path,
    )

    # Add direct air capture
    build_dac(
        n,
        addi_costs=addi_costs,
        cost_multiplier=snakemake.params.sector["dac"]["cost_multiplier"],
        simplify_co2=simplify_co2,
        ts_cost=ts_cost,
    )

    # Add CO2 sequestration and pipeline if not simplified
    if not simplify_co2:
        build_co2_sequestration(n, addi_costs=addi_costs)
        build_co2_pipeline(n, addi_costs=addi_costs)

    # Add long-duration energy storage (LDES)
    ldes_options = snakemake.params.sector.get("ldes", {})
    if ldes_options["enable"]:
        build_ldes(n, addi_costs=addi_costs, ldes_options=ldes_options)

    # Handle only_power mode
    only_power_options = snakemake.params.sector.get("only_power", {})
    if only_power_options["enable"]:
        # Set gas loads to zero
        gas_loads = n.loads[n.loads.carrier == "gas"].index
        n.loads_t.p_set[gas_loads] = 0

        # Set H2 loads to zero
        h2_loads = n.loads[n.loads.carrier == "h2"].index
        n.loads_t.p_set[h2_loads] = 0

        # Remove SMR, bio-H2, methanation links
        smr_links = n.links[n.links.carrier.str.contains("smr", case=False, na=False)].index
        h2_bio_links = n.links[n.links.carrier.str.contains("h2 bio", case=False, na=False)].index
        methanation_links = n.links[n.links.carrier.str.contains("methanation", case=False, na=False)].index
        links_to_remove = smr_links.union(h2_bio_links).union(methanation_links)
        n.mremove("Link", links_to_remove)

    # Apply seasonal capacity derates to new generators and links
    logger.info("Applying seasonal capacity derates to new generators and links")
    state_derates, national_derates = load_derates_from_csv(
        snakemake.input.state_derates,
        snakemake.input.national_derates,
    )
    apply_derates_to_new_components(n, n.snapshots, state_derates, national_derates)

    # Sanitize carriers (needed as loads may be split off to urban/rural)
    sanitize_carriers(n, snakemake.config)

    logger.info("Part 2 complete: Exporting final network")
    n.export_to_netcdf(snakemake.output.network)
