################# ----------- Rules to Build Network ---------- #################

from itertools import chain


rule build_shapes:
    params:
        source_offshore_shapes=config_provider("offshore_shape"),
        offwind_params=config_provider("renewable", "offwind"),
    input:
        zone=DATA + "breakthrough_network/base_grid/zone.csv",
        nerc_shapes="repo_data/geospatial/NERC_Regions/NERC_Regions_Subregions.shp",
        reeds_shapes="repo_data/geospatial/Reeds_Shapes/rb_and_ba_areas.shp",
        onshore_shapes="repo_data/geospatial/BA_shapes_new/Modified_BE_BA_Shapes.shp",
        offshore_shapes_ca_osw="repo_data/geospatial/BOEM_CA_OSW_GIS/CA_OSW_BOEM_CallAreas.shp",
        offshore_shapes_eez=DATA + "eez/conus_eez.shp",
        county_shapes=DATA + "counties/cb_2020_us_county_500k.shp",
    output:
        country_shapes=RESOURCES + "Geospatial/country_shapes.geojson",
        onshore_shapes=RESOURCES + "Geospatial/onshore_shapes.geojson",
        offshore_shapes=RESOURCES + "Geospatial/offshore_shapes.geojson",
        state_shapes=RESOURCES + "Geospatial/state_boundaries.geojson",
        reeds_shapes=RESOURCES + "Geospatial/reeds_shapes.geojson",
        county_shapes=RESOURCES + "Geospatial/county_shapes.geojson",
    log:
        LOGS + "build_shapes.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/build_shapes.py"


rule build_base_network:
    params:
        build_offshore_network=config_provider("offshore_network"),
        model_topology=config_provider("model_topology", "include"),
        length_factor=config["lines"]["length_factor"]
    input:
        buses=DATA + "breakthrough_network/base_grid/bus.csv",
        lines=DATA + "breakthrough_network/base_grid/branch.csv",
        links=DATA + "breakthrough_network/base_grid/dcline.csv",
        bus2sub=DATA + "breakthrough_network/base_grid/bus2sub.csv",
        sub=DATA + "breakthrough_network/base_grid/sub.csv",
        onshore_shapes=RESOURCES + "Geospatial/onshore_shapes.geojson",
        offshore_shapes=RESOURCES + "Geospatial/offshore_shapes.geojson",
        state_shapes=RESOURCES + "Geospatial/state_boundaries.geojson",
        reeds_shapes=RESOURCES + "Geospatial/reeds_shapes.geojson",
        county_shapes=RESOURCES + "Geospatial/county_shapes.geojson",
        reeds_memberships="repo_data/ReEDS_Constraints/membership.csv",
    output:
        bus2sub=RESOURCES + "bus2sub.csv",
        sub=RESOURCES + "sub.csv",
        bus_gis=RESOURCES + "bus_gis.csv",
        lines_gis=RESOURCES + "lines_gis.csv",
        network=RESOURCES + "base_network.nc",
    log:
        LOGS + "create_network.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/build_base_network.py"


rule build_bus_regions:
    params:
        topological_boundaries=config_provider(
            "model_topology", "topological_boundaries"
        ),
        focus_weights=config_provider("focus_weights"),
    input:
        country_shapes=RESOURCES + "Geospatial/country_shapes.geojson",
        county_shapes=RESOURCES + "Geospatial/county_shapes.geojson",
        state_shapes=RESOURCES + "Geospatial/state_boundaries.geojson",
        ba_region_shapes=RESOURCES + "Geospatial/onshore_shapes.geojson",
        reeds_shapes=RESOURCES + "Geospatial/reeds_shapes.geojson",
        offshore_shapes=RESOURCES + "Geospatial/offshore_shapes.geojson",
        base_network=RESOURCES + "base_network.nc",
        bus2sub=RESOURCES + "bus2sub.csv",
        sub=RESOURCES + "sub.csv",
    output:
        regions_onshore=RESOURCES + "Geospatial/regions_onshore.geojson",
        regions_offshore=RESOURCES + "Geospatial/regions_offshore.geojson",
    log:
        LOGS + "build_bus_regions.log",
    threads: 1
    resources:
        mem_mb=3000,
    script:
        "../scripts/build_bus_regions.py"


rule build_cost_data:
    params:
        costs=config_provider("costs"),
        pudl_path=config_provider("pudl_path"),
    input:
        efs_tech_costs="repo_data/costs/EFS_Technology_Data.xlsx",
        efs_icev_costs="repo_data/costs/efs_icev_costs.csv",
        eia_tech_costs="repo_data/costs/eia_tech_costs.csv",
        egs_costs="repo_data/costs/egs_costs.csv",
        additional_costs="repo_data/costs/additional_costs.csv",
    output:
        tech_costs=RESOURCES + "costs/costs_{year}.csv",
        sector_costs=RESOURCES + "costs/sector_costs_{year}.csv",
    log:
        LOGS + "costs_{year}.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/build_cost_data.py"


ATLITE_NPROCESSES = config["atlite"].get("nprocesses", 4)

if config["enable"].get("build_cutout", False):

    rule build_cutout:
        params:
            snapshots=config_provider("snapshots"),
            cutouts=config_provider("atlite", "cutouts"),
            interconnects=config_provider("atlite", "interconnects"),
        input:
            regions_onshore=RESOURCES
            + "Geospatial/country_shapes.geojson",
            regions_offshore=RESOURCES
            + "Geospatial/offshore_shapes.geojson",
        output:
            protected("cutouts/" + "{interconnect}_{cutout}.nc"),
        log:
            "logs/" + "build_cutout/{interconnect}_{cutout}.log",
        threads: ATLITE_NPROCESSES
        resources:
            mem_mb=ATLITE_NPROCESSES * 5000,
        script:
            "../scripts/build_cutout.py"


rule build_renewable_profiles:
    params:
        renewable=config_provider("renewable"),
        snapshots=config_provider("snapshots"),
    input:
        corine=ancient(
            DATA
            + "copernicus/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_USA_EPSG-4326.tif"
        ),
        natura=lambda w: (
            DATA + "natura.tiff" if config["renewable"][w.technology]["natura"] else []
        ),
        gebco=ancient(
            lambda w: (
                DATA + "gebco/gebco_2023_n55.0_s10.0_w-126.0_e-65.0.tif"
                if config["renewable"][w.technology].get("max_depth")
                else []
            )
        ),
        country_shapes=RESOURCES + "Geospatial/country_shapes.geojson",
        offshore_shapes=RESOURCES + "Geospatial/offshore_shapes.geojson",
        cec_onwind="repo_data/geospatial/CEC_GIS/CEC_Wind_BaseScreen_epsg3310.tif",
        cec_solar="repo_data/geospatial/CEC_GIS/CEC_Solar_BaseScreen_epsg3310.tif",
        boem_osw="repo_data/geospatial/boem_osw_planning_areas.tif",
        regions=lambda w: (
            RESOURCES + "Geospatial/regions_onshore.geojson"
            if w.technology in ("onwind", "solar")
            else RESOURCES + "Geospatial/regions_offshore.geojson"
        ),
        cutout=lambda wildcards: expand(
            "cutouts/"
            + "usa_"
            + config["renewable"][wildcards.technology]["cutout"]
            + "_{renewable_weather_year}"
            + ".nc",
            renewable_weather_year=config["renewable_weather_years"],
        ),
    output:
        profile=RESOURCES + "profile_{technology}.nc",
        availability=RESULTS + "land_use_availability_{technology}.png",
    log:
        LOGS + "build_renewable_profile_{technology}.log",
    threads: ATLITE_NPROCESSES
    retries: 3
    resources:
        mem_mb=lambda wildcards, input, attempt: (
            ATLITE_NPROCESSES * input.size // 3500000
        )
        * attempt
        * 1.5,
    wildcard_constraints:
        technology="(?!hydro|EGS).*",  # Any technology other than hydro
    script:
        "../scripts/build_renewable_profiles.py"


# eastern broken out just to aviod awful formatting issues
# texas in western due to spillover of interconnect
INTERCONNECT_2_STATE = {
    "eastern": ["AL", "AR", "CT", "DE", "FL", "GA", "IL", "IN", "IA", "KS", "KY", "LA"],
    "western": ["AZ", "CA", "CO", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY", "TX"],
    "texas": ["TX"],
}
INTERCONNECT_2_STATE["eastern"].extend(["ME", "MD", "MA", "MI", "MN", "MS", "MO", "NE"])
INTERCONNECT_2_STATE["eastern"].extend(["NH", "NJ", "NY", "NC", "ND", "OH", "OK", "PA"])
INTERCONNECT_2_STATE["eastern"].extend(["RI", "SC", "SD", "TN", "VT", "VA", "WV", "WI"])
INTERCONNECT_2_STATE["usa"] = sum(INTERCONNECT_2_STATE.values(), [])


def ba_gas_dynamic_fuel_price_files(wildcards):
    files = []
    files.append(DATA + "costs/caiso_ng_power_prices.csv")
    return files


rule build_fuel_prices:
    params:
        snapshots=config["snapshots"],
        api_eia=config["api"]["eia"],
        pudl_path=config_provider("pudl_path"),
    input:
        gas_balancing_area=ba_gas_dynamic_fuel_price_files,
    output:
        state_ng_fuel_prices=RESOURCES + "state_ng_power_prices.csv",
        state_coal_fuel_prices=RESOURCES + "state_coal_power_prices.csv",
        ba_ng_fuel_prices=RESOURCES + "ba_ng_power_prices.csv",
        pudl_fuel_costs=RESOURCES + "pudl_fuel_costs.csv",
    log:
        LOGS + "build_fuel_prices.log",
    threads: 1
    retries: 3
    resources:
        mem_mb=30000,
    script:
        "../scripts/build_fuel_prices.py"


def dynamic_fuel_price_files(wildcards):
    if config["conventional"]["dynamic_fuel_price"]["wholesale"]:
        return {
            "state_ng_fuel_prices": RESOURCES
            + "state_ng_power_prices.csv",
            "state_coal_fuel_prices": RESOURCES
            + "state_coal_power_prices.csv",
            "ba_ng_fuel_prices": RESOURCES + "ba_ng_power_prices.csv",
        }
    else:
        return {}


rule build_powerplants:
    params:
        pudl_path=config_provider("pudl_path"),
        renewable_weather_year=config_provider("renewable_weather_years"),
    input:
        wecc_ads="repo_data/WECC_ADS_public",
        eia_ads_generator_mapping="repo_data/WECC_ADS_public/eia_ads_generator_mapping_updated.csv",
        fuel_costs="repo_data/plants/fuelCost22.csv",
        cems="repo_data/plants/cems_heat_rates.xlsx",
        epa_crosswalk="repo_data/plants/epa_eia_crosswalk.csv",
    output:
        powerplants=RESOURCES + "powerplants.csv",
    log:
        LOGS + "build_powerplants.log",
    resources:
        mem_mb=30000,
    script:
        "../scripts/build_powerplants.py"


rule add_electricity:
    params:
        length_factor=config["lines"]["length_factor"],
        renewable=config["renewable"],
        renewable_carriers=config["electricity"]["renewable_carriers"],
        extendable_carriers=config["electricity"]["extendable_carriers"],
        conventional_carriers=config["electricity"]["conventional_carriers"],
        conventional=config["conventional"],
        costs=config["costs"],
        planning_horizons=config["scenario"]["planning_horizons"],
        snapshots=config["snapshots"],
        eia_api=config["api"]["eia"],
    input:
        unpack(dynamic_fuel_price_files),
        **{
            f"profile_{tech}": RESOURCES + f"profile_{tech}.nc"
            for tech in config["electricity"]["renewable_carriers"]
            if tech != "hydro"
        },
        **{
            f"conventional_{carrier}_{attr}": fn
            for carrier, d in config.get("conventional", {None: {}}).items()
            if carrier in config["electricity"]["conventional_carriers"]
            for attr, fn in d.items()
            if str(fn).startswith("data/")
        },
        # **{
        #     f"gen_cost_mult_{Path(x).stem}": f"repo_data/locational_multipliers/{Path(x).name}"
        #     for x in Path("repo_data/locational_multipliers/").glob("*")
        # },
        base_network=RESOURCES + "base_network.nc",
        tech_costs=RESOURCES
        + f"costs/costs_{config['scenario']['planning_horizons'][0]}.csv",
        # attach first horizon costs
        regions_onshore=RESOURCES + "Geospatial/regions_onshore.geojson",
        regions_offshore=RESOURCES + "Geospatial/regions_offshore.geojson",
        reeds_shapes=RESOURCES + "Geospatial/reeds_shapes.geojson",
        powerplants=RESOURCES + "powerplants.csv",
        plants_breakthrough=DATA + "breakthrough_network/base_grid/plant.csv",
        hydro_breakthrough=DATA + "breakthrough_network/base_grid/hydro.csv",
        bus2sub=RESOURCES + "bus2sub.csv",
        pudl_fuel_costs=RESOURCES + "pudl_fuel_costs.csv",
        specs_egs=(
            DATA + "EGS/specs_EGS.nc"
            if "EGS" in config["electricity"]["extendable_carriers"]["Generator"]
            else []
        ),
        profile_egs=(
            DATA + "EGS/profile_EGS.nc"
            if "EGS" in config["electricity"]["extendable_carriers"]["Generator"]
            else []
        ),
    output:
        network=RESOURCES + "add_elec_network.pkl",
        state_derates=RESOURCES + "state_derates.csv",
        national_derates=RESOURCES + "national_derates.csv",
    log:
        LOGS + "add_electricity.log",
    threads: 1
    resources:
        mem_mb=lambda wildcards, input, attempt: max(5000,
            sum(getattr(f, 'size', 0) for f in input if hasattr(f, 'size')) // 400000 * attempt * 2
        ),
    script:
        "../scripts/add_electricity.py"


################# ----------- Rules to Aggregate & Simplify Network ---------- #################
rule simplify_network:
    params:
        aggregation_strategies=config["clustering"].get("aggregation_strategies", {}),
        focus_weights=config_provider("focus_weights", default=False),
        simplify_network=config_provider("clustering", "simplify_network"),
        planning_horizons=config_provider("scenario", "planning_horizons"),
        topological_boundaries=config_provider(
            "model_topology", "topological_boundaries"
        ),
    input:
        bus2sub=RESOURCES + "bus2sub.csv",
        sub=RESOURCES + "sub.csv",
        network=RESOURCES + "add_elec_network.pkl",
        regions_onshore=RESOURCES + "Geospatial/regions_onshore.geojson",
        regions_offshore=RESOURCES + "Geospatial/regions_offshore.geojson",
    output:
        network=RESOURCES + "simplified_network.nc",
        regions_onshore=RESOURCES + "Geospatial/regions_onshore_simplified.geojson",
        regions_offshore=RESOURCES + "Geospatial/regions_offshore_simplified.geojson",
        gas_distribution=RESOURCES + "gas_capacity_distribution.csv",
    log:
        LOGS + "simplify_network.log",
    threads: 1
    resources:
        mem_mb=lambda wildcards, input, attempt: max(5000,
            sum(getattr(f,'size',0) for f in input if hasattr(f,'size')) // 100000 * attempt * 1.5
            ),
    script:
        "../scripts/simplify_network.py"


rule cluster_network:
    params:
        cluster_network=config_provider("clustering", "cluster_network"),
        conventional_carriers=config_provider("electricity", "conventional_carriers"),
        renewable_carriers=config_provider("electricity", "renewable_carriers"),
        aggregation_strategies=config_provider("clustering", "aggregation_strategies"),
        custom_busmap=config_provider("enable", "custom_busmap", default=False),
        focus_weights=config_provider("focus_weights", default=False),
        length_factor=config_provider("lines", "length_factor"),
        costs=config_provider("costs"),
        planning_horizons=config_provider("scenario", "planning_horizons"),
        transmission_network=get_transmission_network,
        topological_boundaries=config_provider(
            "model_topology", "topological_boundaries"
        ),
        topology_aggregation=config_provider("model_topology", "aggregate")
    input:
        network=RESOURCES + "simplified_network.nc",
        regions_onshore=RESOURCES + "Geospatial/regions_onshore_simplified.geojson",
        regions_offshore=RESOURCES + "Geospatial/regions_offshore_simplified.geojson",
        custom_busmap=(
            DATA + "custom_busmap.csv"
            if config["enable"].get("custom_busmap", False)
            else []
        ),
        tech_costs=RESOURCES
        + f"costs/costs_{config['scenario']['planning_horizons'][0]}.csv",
        itl_reeds_zone="repo_data/ReEDS_Constraints/transmission/transmission_capacity_init_AC_ba_NARIS2024.csv",
        itl_county="repo_data/ReEDS_Constraints/transmission/transmission_capacity_init_AC_county_NARIS2024.csv",
        itl_trans_grp="repo_data/ReEDS_Constraints/transmission/transmission_capacity_init_AC_transgrp_NARIS2024.csv",
        itl_costs_reeds_zone="repo_data/ReEDS_Constraints/transmission/transmission_distance_cost_500kVdc_ba.csv",
        itl_costs_county="repo_data/ReEDS_Constraints/transmission/transmission_distance_cost_500kVac_county.csv",
        itl_state="repo_data/ReEDS_Constraints/transmission/transmission_capacity_init_AC_state_NARIS2024.csv",
        itl_costs_state="repo_data/ReEDS_Constraints/transmission/transmission_distance_cost_500kVac_state.csv",
    output:
        network=RESOURCES + "{transmission_network}/clustered_network.nc",
        regions_onshore=RESOURCES + "{transmission_network}/Geospatial/regions_onshore_clustered.geojson",
        regions_offshore=RESOURCES + "{transmission_network}/Geospatial/regions_offshore_clustered.geojson",
        busmap=RESOURCES + "{transmission_network}/busmap_clustered.csv",
        linemap=RESOURCES + "{transmission_network}/linemap_clustered.csv",
    log:
        LOGS + "{transmission_network}/cluster_network.log",
    threads: 1
    resources:
        mem_mb=lambda wildcards, input, attempt: max(5000,
    sum(getattr(f, 'size', 0) for f in input if hasattr(f, 'size')) // 100000) * attempt * 2,
    script:
        "../scripts/cluster_network.py"


rule build_co2_storage:
    input:
        regions_onshore=RESOURCES + "{transmission_network}/Geospatial/regions_onshore_clustered.geojson",
        co2_storage="repo_data/geospatial/co2_storage/co2_storage.geojson",
    output:
        co2_storage=RESOURCES + "{transmission_network}/co2_storage.csv",
    log:
        LOGS + "{transmission_network}/build_co2_storage.log",
    resources:
        mem_mb=5000,
    script:
        "../scripts/build_co2_storage.py"


rule add_extra_components:
    input:
        existing_PHS="repo_data/costs/ReEDS_generator_database_final_EIA-NEMS.csv",
        network=RESOURCES + "{transmission_network}/clustered_network.nc",
        tech_costs=lambda wildcards: expand(
            RESOURCES + "costs/costs_{year}.csv",
            year=config["scenario"]["planning_horizons"],  # This is OK - used only for file path expansion
        ),
        regions_onshore=RESOURCES + "{transmission_network}/Geospatial/regions_onshore_clustered.geojson",
    params:
        retirement=case_config_provider("electricity", "retirement", default="technical"),
        demand_response=case_config_provider("electricity", "demand_response", default={}),
    output:
        RESOURCES + "{transmission_network}/extra_components_network.nc",
    log:
        LOGS + "{transmission_network}/add_extra_components.log",
    threads: 1
    resources:
        mem_mb=lambda wildcards, input, attempt: max(5000,
            sum(getattr(f,'size',0) for f in input if hasattr(f,'size')) // 100000) * attempt * 2,
    script:
        "../scripts/add_extra_components.py"


def demand_files(wildcards):
    """
    Generate demand file paths based on the case's nza_scenario.
    """
    # Use the unified config getter
    case_cfg = get_case_config(wildcards)
    nza_scenario = case_cfg["scenario"]["nza_scenario"]

    carriers = ["electricity", "natural_gas", "hydrogen"]
    years = case_cfg["scenario"]["planning_horizons"]

    return [
        f"repo_data/MY_simpsec_demand/{nza_scenario}/{carrier}_{year}.csv"
        for carrier in carriers
        for year in years
    ]


rule add_demand:
    params:
        sectors=case_config_provider("scenario", "sector"),
        planning_horizons=case_config_provider("scenario", "planning_horizons"),
        snapshots=case_config_provider("snapshots"),
        nza_scenario=case_config_provider("scenario", "nza_scenario"),
        adj_scenario=case_config_provider("scenario","adj_scenario"),
    input:
        network=RESOURCES + "{transmission_network}/extra_components_network.nc",
        demand=demand_files,
        busmap=RESOURCES + "{transmission_network}/busmap_clustered.csv",
    output:
        network=RESOURCES + "{transmission_network}/{case}/demand_network.nc",
    log:
        LOGS + "{transmission_network}/{case}/add_demand.log",
    resources:
        mem_mb=lambda wildcards, input, attempt: max(5000,
            sum(getattr(f,'size',0) for f in input if hasattr(f,'size')) // 70000 * attempt * 2
            ),
    script:
        "../scripts/add_demand.py"


rule prepare_network:
    params:
        time_resolution=case_config_provider(
            "clustering", "temporal", "resolution_elec"
        ),
        adjustments=case_config_provider("adjustments", default=False),
        links=case_config_provider("links"),
        lines=case_config_provider("lines"),
        co2base=case_config_provider("electricity", "co2base"),
        co2limit=case_config_provider("electricity", "co2limit"),
        co2limit_enable=case_config_provider("electricity", "co2limit_enable", default=False),
        gaslimit=case_config_provider("electricity", "gaslimit"),
        gaslimit_enable=case_config_provider("electricity", "gaslimit_enable", default=False),
        transmission_network=get_transmission_network,
        costs=case_config_provider("costs"),
        autarky=case_config_provider("electricity", "autarky"),
        ll=case_config_provider("scenario", "ll"),
    input:
        network=RESOURCES + "{transmission_network}/{case}/demand_network.nc",
        tech_costs=(
            RESOURCES + f"costs/costs_{config['scenario']['planning_horizons'][0]}.csv"
        ),
    output:
        RESOURCES + "{transmission_network}/{case}/prepared_network.nc",
    log:
        solver=LOGS + "{transmission_network}/{case}/prepare_network.log",
    threads: 1
    resources:
        mem_mb=lambda wildcards, input, attempt: max(5000,
            sum(getattr(f,'size',0) for f in input if hasattr(f,'size')) // 100000 * attempt * 2
        ),
    script:
        "../scripts/prepare_network.py"
