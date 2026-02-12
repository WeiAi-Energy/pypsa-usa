def sector_input_files_part1(wildcards):
    """Input files for Part 1: sector foundation and natural gas"""
    network_path = (
        RESOURCES
        + f"{wildcards.transmission_network}/{wildcards.case}/prepared_network.nc"
    )
    costs_path = (
        RESOURCES + f"costs/costs_{config['scenario']['planning_horizons'][0]}.csv"
    )

    input_files = {
        "network": network_path,
        "costs": costs_path,
        "addi_costs": f"repo_data/costs/SimpSec_costs_{config['scenario']['planning_horizons'][0]}.csv",
        "county": DATA + "counties/cb_2020_us_county_500k.shp",
        "gas_distribution": RESOURCES + "gas_capacity_distribution.csv",
    }

    # PHS files (if needed for attach_phs_storageunits)
    phs_files = {
        f"phs_shp_{hour}": "repo_data/"
        + f"psh/40-100-dam-height-{hour}hr-no-croplands-no-ephemeral-no-highways.gpkg"
        for phs_tech in config["electricity"]["PHS_exp"]
        if "PHS" in phs_tech
        for hour in phs_tech.split("hr_")
        if hour.isdigit()
    }
    input_files.update(phs_files)

    # Regions onshore (needed for PHS)
    if phs_files:
        regions_onshore_files = {
            "regions_onshore": RESOURCES
            + "{transmission_network}/Geospatial/regions_onshore_clustered.geojson",
        }
        input_files.update(regions_onshore_files)

    # Natural gas files
    ng_files = {
        "pipeline_capacity": DATA + "natural_gas/EIA-StatetoStateCapacity_Feb2024.xlsx",
        "pipeline_shape": DATA + "natural_gas/pipelines.geojson",
        "gas_storage_plants": "repo_data/Natural gas storage plant.csv",
    }
    input_files.update(ng_files)

    # CO2 storage file (needed for build_co2_tracking)
    co2_storage_file = {
        "co2_storage_file": RESOURCES + "{transmission_network}/co2_storage.csv"
    }
    input_files.update(co2_storage_file)

    return input_files


def sector_input_files_part2(wildcards):
    """Input files for Part 2: hydrogen, biomass, DAC, CO2, LDES, derates"""
    case_config = config["cases"][wildcards.case]
    nza_scenario = case_config["scenario"]["nza_scenario"]

    input_files = {
        "network": RESOURCES
        + f"{wildcards.transmission_network}/{wildcards.case}/sector_network_part1.nc",
        "costs": RESOURCES
        + f"costs/costs_{config['scenario']['planning_horizons'][0]}.csv",
        "addi_costs": f"repo_data/costs/SimpSec_costs_{config['scenario']['planning_horizons'][0]}.csv",
    }

    # Biomass files based on nza_scenario
    if nza_scenario in ("E+", "E-", "E+RE-", "E+RE+"):
        bio_files = {
            "agri": "repo_data/biomass/billionton_23_agri_NT.csv",
            "forestry": "repo_data/biomass/billionton_23_forestry_NT.csv",
            "wastes": "repo_data/biomass/billionton_23_wastes_NT.csv",
        }
    else:
        bio_files = {
            "agri": "repo_data/biomass/billionton_23_agri_mm-med.csv",
            "forestry": "repo_data/biomass/billionton_23_forestry_mm-med.csv",
            "wastes": "repo_data/biomass/billionton_23_wastes_mm-med.csv",
        }
    input_files.update(bio_files)

    # CO2 storage file
    co2_storage_file = {
        "co2_storage_file": RESOURCES + "{transmission_network}/co2_storage.csv"
    }
    input_files.update(co2_storage_file)

    # Derates files
    derates_file = {
        "state_derates": RESOURCES + "state_derates.csv",
        "national_derates": RESOURCES + "national_derates.csv",
    }
    input_files.update(derates_file)

    return input_files


rule add_simple_sectors_part1:
    """
    First part: Add sector foundation and natural gas infrastructure
    """
    params:
        scenario=get_case_scenario,
        electricity=get_case_electricity,
        sector=get_case_sector,
        costs=case_config_provider("costs"),
        plotting=case_config_provider("plotting"),
        snapshots=case_config_provider("snapshots"),
        api=case_config_provider("api"),
    input:
        unpack(sector_input_files_part1),
    output:
        network=RESOURCES + "{transmission_network}/{case}/sector_network_part1.nc",
    log:
        LOGS + "{transmission_network}/{case}/add_simple_sectors_part1.log",
    group:
        "prepare"
    threads: 1
    resources:
        mem_mb=4000,
    script:
        "../scripts/add_simple_sectors_part1.py"


def get_reference_network_for_only_power(wildcards):
    """
    Get reference network path for only_power mode.
    Returns the solved network from the non-only_power case if only_power is enabled.
    The reference case name is derived by removing "_power" suffix from the case name.
    """
    case_cfg = get_case_config(wildcards)
    only_power_enabled = (
        case_cfg.get("sector", {}).get("only_power", {}).get("enable", False)
    )

    if only_power_enabled:
        # Remove "_power" suffix to get reference case name
        reference_case = wildcards.case.replace("_power", "")

        # Return the path to the reference network
        return (
            RESULTS
            + f"{wildcards.transmission_network}/{reference_case}/networks/solved_network.nc"
        )
    else:
        # Return empty list if only_power is not enabled
        return []


rule add_simple_sectors_part2:
    """
    Second part: Add hydrogen, biomass, DAC, CO2, LDES, and apply derates
    """
    params:
        scenario=get_case_scenario,
        electricity=get_case_electricity,
        sector=get_case_sector,
        costs=case_config_provider("costs"),
        plotting=case_config_provider("plotting"),
        snapshots=case_config_provider("snapshots"),
        api=case_config_provider("api"),
    input:
        unpack(sector_input_files_part2),
        reference_network=get_reference_network_for_only_power,
    output:
        network=RESOURCES + "{transmission_network}/{case}/sector_network.nc",
    log:
        LOGS + "{transmission_network}/{case}/add_simple_sectors.log",
    group:
        "prepare"
    threads: 1
    resources:
        mem_mb=4000,
    script:
        "../scripts/add_simple_sectors_part2.py"
