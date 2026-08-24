################# ----------- Rules to Build Network ---------- #################

from itertools import chain


rule build_shapes:
    params:
        interconnect=get_interconnect(),
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
        "logs/build_shapes.log",
    threads: 1
    resources:
        walltime=config_provider("walltime", "build_shapes", default="00:30:00"),
        mem_mb=5000,
    script:
        "../scripts/build_shapes.py"


rule build_base_network:
    params:
        interconnect=get_interconnect(),
        build_offshore_network=config_provider("offshore_network"),
        model_topology=config_provider("model_topology", "include"),
        topological_boundaries=config_provider(
            "model_topology", "topological_boundaries"
        ),
        length_factor=config["lines"]["length_factor"],
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
        network=RESOURCES + "elec_base_network.nc",
    log:
        "logs/create_network.log",
    threads: 1
    resources:
        mem_mb=5000,
        walltime=config_provider("walltime", "build_base_network", default="00:30:00"),
    script:
        "../scripts/build_base_network.py"


rule build_bus_regions:
    params:
        interconnect=get_interconnect(),
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
        base_network=RESOURCES + "elec_base_network.nc",
        bus2sub=RESOURCES + "bus2sub.csv",
        sub=RESOURCES + "sub.csv",
    output:
        regions_onshore=RESOURCES + "Geospatial/regions_onshore.geojson",
        regions_offshore=RESOURCES
        + "Geospatial/regions_offshore.geojson",
    log:
        "logs/build_bus_regions.log",
    threads: 1
    resources:
        mem_mb=3000,
        walltime=config_provider("walltime", "build_bus_regions", default="00:30:00"),
    script:
        "../scripts/build_bus_regions.py"


rule build_cost_data:
    params:
        costs=config_provider("costs"),
        pudl_path=config_provider("pudl_path"),
    output:
        tech_costs=RESOURCES + "costs/costs_{year}.csv",
    log:
        LOGS + "costs_{year}.log",
    threads: 1
    resources:
        mem_mb=5000,
        walltime=config_provider("walltime", "build_cost_data", default="00:30:00"),
    script:
        "../scripts/build_cost_data.py"


rule build_renewable_profiles:
    params:
        renewable_weather_years=config_provider("renewable_weather_years"),
        reeds_vre_dir=DATA + "ReEDS_VRE",
        reeds_interconnection_dir="repo_data/costs",
    input:
        unpack(representative_snapshots_input),
        reeds_vre_files=reeds_vre_files_for_technology,
        interconnection_land="repo_data/costs/interconnection_land.h5",
        interconnection_offshore="repo_data/costs/interconnection_offshore.h5",
        # Profiles are attached before topology reduction.  Their regions must
        # therefore be keyed by the raw buses; simplify/cluster aggregate the
        # resulting generators and capacity factors later.
        regions=lambda w: (
            RESOURCES + "Geospatial/regions_onshore.geojson"
            if w.technology in ("onwind", "solar")
            else RESOURCES + "Geospatial/regions_offshore.geojson"
        ),
        gebco=ancient(DATA + "gebco/gebco_2023_n55.0_s10.0_w-126.0_e-65.0.tif"),
    output:
        # Profiles remain keyed to raw buses.  They are attached once and
        # aggregated by the downstream simplify/cluster stages.
        profile=RESOURCES + "{demand_level}Dmd/profile_{technology}.nc",
    log:
        LOGS
        + "{demand_level}Dmd/build_renewable_profile_{technology}.log",
    benchmark:
        BENCHMARKS
        + "{demand_level}Dmd/build_renewable_profiles_{technology}"
    threads: 1
    resources:
        mem_mb=16000,
        walltime=config_provider(
            "walltime", "build_renewable_profiles", default="02:30:00"
        ),
    wildcard_constraints:
        technology="onwind|offwind|offwind_floating|solar",
        demand_level="Low|Mid|High",
    script:
        "../scripts/build_reeds_renewable_profiles.py"


EER_DEMAND_FILES = {
    "Low": "demand_EER2025_Baseline_AEO2023.h5",
    "Mid": "demand_EER2025_IRAlow.h5",
    "High": "demand_EER2025_100by2050.h5",
}


def eer_demand_file_for_wildcards(wildcards):
    """EER demand file for a rule that carries its own `demand_level` wildcard."""
    return DATA + "eer/" + EER_DEMAND_FILES[wildcards.demand_level]


validate_shared_representative_periods()


def representative_periods_reeds_files(wildcards):
    """
    Raw ReEDS supply curves and CF tables feeding the national clustering features.

    Selection runs before ``build_renewable_profiles``, so it reads the ReEDS data
    directly. ``offwind`` and ``offwind_floating`` share one ReEDS technology, so
    the set is deduplicated.
    """
    carriers = config["electricity"]["renewable_carriers"]
    techs = {
        "offwind" if tech.startswith("offwind") else tech
        for tech in carriers
        if tech not in ("hydro", "EGS")
    }
    return [
        DATA + f"ReEDS_VRE/{tech}/{file}"
        for tech in sorted(techs)
        for file in REEDS_VRE_DATAFILES[tech]
    ]


rule select_representative_periods:
    wildcard_constraints:
        demand_level="Low|Mid|High",
    params:
        representative_periods=lambda wildcards: representative_periods_config(),
        planning_horizons=lambda wildcards: representative_periods_planning_horizons(),
        renewable_weather_years=config_provider("renewable_weather_years"),
        renewable_carriers=lambda wildcards: config["electricity"]["renewable_carriers"],
        reeds_vre_dir=DATA + "ReEDS_VRE",
    input:
        reeds_vre_files=representative_periods_reeds_files,
        electricity_demand=eer_demand_file_for_wildcards,
    output:
        snapshots=RESOURCES
        + "{demand_level}Dmd/representative_periods/snapshots.csv",
        metadata=RESOURCES
        + "{demand_level}Dmd/representative_periods/metadata.json",
        plot=RESOURCES
        + "{demand_level}Dmd/representative_periods/profiles.png",
    log:
        LOGS + "{demand_level}Dmd/select_representative_periods.log",
    threads: 1
    resources:
        mem_mb=16000,
        walltime=config_provider("walltime", "select_representative_periods", default="04:00:00"),
    script:
        "../scripts/select_representative_periods.py"


rule build_electrical_demand:
    wildcard_constraints:
        end_use="power",  # added for consistency in build_demand.py
        demand_level="Low|Mid|High",
    params:
        planning_horizons=config_provider("scenario", "planning_horizons"),
        renewable_weather_years=config_provider("renewable_weather_years"),
    input:
        unpack(representative_snapshots_input),
        network=elec_network_path,
        electricity_demand=eer_demand_file_for_wildcards,
    output:
        elec_demand=RESOURCES
        + "{demand_level}Dmd/demand/{end_use}_electricity.pkl",
    log:
        LOGS + "{demand_level}Dmd/demand/{end_use}_build_demand.log",
    benchmark:
        BENCHMARKS + "{demand_level}Dmd/demand/{end_use}_build_demand"
    threads: 2
    resources:
        mem_mb=lambda wildcards, input, attempt: (input_size_bytes(input) // 100000) * attempt * 2,
        walltime=config_provider(
            "walltime", "build_electrical_demand", default="00:50:00"
        ),
    script:
        "../scripts/build_eer_demand.py"


rule add_demand:
    wildcard_constraints:
        demand_level="Low|Mid|High",
    params:
        planning_horizons=config_provider("scenario", "planning_horizons"),
        snapshots=config_provider("snapshots"),
    input:
        network=elec_network_path,
        demand=RESOURCES + "{demand_level}Dmd/demand/power_electricity.pkl",
    output:
        network=RESOURCES + "{demand_level}Dmd/elec_dem.nc",
    log:
        LOGS + "{demand_level}Dmd/demand/add_demand.log",
    benchmark:
        BENCHMARKS + "{demand_level}Dmd/demand/add_demand"
    resources:
        mem_mb=lambda wildcards, input, attempt: (input_size_bytes(input) // 70000) * attempt * 2,
        walltime=config_provider("walltime", "add_demand", default="00:50:00"),
    script:
        "../scripts/add_demand.py"


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
        "logs/build_powerplants.log",
    resources:
        mem_mb=30000,
        walltime=config_provider("walltime", "build_powerplants", default="00:30:00"),
    script:
        "../scripts/build_powerplants.py"


def add_electricity_input_network(wildcards):
    """Attach components to the raw topology before simplifying it."""
    return RESOURCES + "elec_base_network.nc"


def add_electricity_profile_inputs(wildcards):
    cfg = config_for_wildcards(wildcards)
    renewable_carriers = cfg["electricity"]["renewable_carriers"]
    return {
        f"profile_{tech}": RESOURCES
        + f"{wildcards.demand_level}Dmd/profile_{tech}.nc"
        for tech in renewable_carriers
        if tech != "hydro"
    }


def add_electricity_conventional_inputs(wildcards):
    cfg = config_for_wildcards(wildcards)
    return {
        f"conventional_{carrier}_{attr}": fn
        for carrier, d in cfg.get("conventional", {None: {}}).items()
        if carrier in cfg["electricity"]["conventional_carriers"]
        for attr, fn in d.items()
        if str(fn).startswith("data/")
    }


def add_electricity_egs_inputs(wildcards):
    """Only require the optional EGS supply curves when EGS is enabled."""
    cfg = config_for_wildcards(wildcards)
    carriers = set(cfg["electricity"].get("renewable_carriers", []))
    carriers.update(
        cfg["electricity"].get("extendable_carriers", {}).get("Generator", [])
    )
    if "EGS" not in carriers:
        return {}
    return {
        "egs_supply_curve_4500": "repo_data/EGS/egs_4500m_supply-curve.csv",
        "egs_supply_curve_5500": "repo_data/EGS/egs_5500m_supply-curve.csv",
        "egs_supply_curve_6500": "repo_data/EGS/egs_6500m_supply-curve.csv",
    }


rule add_electricity:
    wildcard_constraints:
        demand_level="Low|Mid|High",
    params:
        interconnect=get_interconnect(),
        length_factor=config_provider("lines", "length_factor"),
        snapshots=config_provider("snapshots"),
        renewable_carriers=config_provider("electricity", "renewable_carriers"),
        extendable_carriers=config_provider("electricity", "extendable_carriers"),
        conventional_carriers=config_provider("electricity", "conventional_carriers"),
        conventional=config_provider("conventional"),
        costs=config_provider("costs"),
        planning_horizons=config_provider("scenario", "planning_horizons"),
        renewable_weather_years=config_provider("renewable_weather_years"),
        add_existing_phs=config_provider("electricity", "add_existing_phs", default=False),
        cost_multipliers=config_provider("cost_multipliers", default={}),
    input:
        unpack(add_electricity_profile_inputs),
        unpack(add_electricity_conventional_inputs),
        unpack(add_electricity_egs_inputs),
        unpack(representative_snapshots_input),
        base_network=add_electricity_input_network,
        tech_costs=get_cost_file,
        existing_PHS="repo_data/costs/ReEDS_generator_database_final_EIA-NEMS.csv",
        # attach first horizon costs
        all_reeds_shapes="repo_data/geospatial/Reeds_Shapes/rb_and_ba_areas.shp",
        reeds_memberships="repo_data/ReEDS_Constraints/membership.csv",
        # Region/route-resolved 500 kV AC cost per county pair, and the voltage-class
        # and DC/AC cost relationships. Both in 2004 USD.
        transmission_distance_cost=(
            "repo_data/ReEDS_Constraints/transmission/"
            "transmission_distance_cost_500kVac_county.csv"
        ),
        transmission_basecost="repo_data/ReEDS_Constraints/transmission/rev_transmission_basecost.csv",
        regions_onshore=RESOURCES + "Geospatial/regions_onshore.geojson",
        regions_offshore=RESOURCES + "Geospatial/regions_offshore.geojson",
        county_shapes=RESOURCES + "Geospatial/county_shapes.geojson",
        reeds_shapes=RESOURCES + "Geospatial/reeds_shapes.geojson",
        powerplants=RESOURCES + "powerplants.csv",
        plants_breakthrough=DATA + "breakthrough_network/base_grid/plant.csv",
        hydro_breakthrough=DATA + "breakthrough_network/base_grid/hydro.csv",
        bus2sub=RESOURCES + "bus2sub.csv",
    output:
        RESOURCES + "{demand_level}Dmd/elec_pp.pkl",
    log:
        LOGS + "{demand_level}Dmd/add_electricity.log",
    benchmark:
        BENCHMARKS + "{demand_level}Dmd/add_electricity"
    threads: 1
    resources:
        mem_mb=lambda wildcards, input, attempt: (input_size_bytes(input) // 400000) * attempt * 2,
        walltime=config_provider("walltime", "add_electricity", default="01:00:00"),
    script:
        "../scripts/add_electricity.py"


################# ----------- Mandatory topology reduction ---------- #################
rule simplify_network:
    wildcard_constraints:
        demand_level="Low|Mid|High",
    params:
        aggregation_strategies=config["clustering"].get("aggregation_strategies", {}),
        electrical_distance=config_provider("clustering", "electrical_distance"),
        length_factor=config_provider("lines", "length_factor"),
        topological_boundaries=config_provider(
            "model_topology", "topological_boundaries"
        ),
    input:
        network=electricity_attached_network_path,
        regions_onshore=RESOURCES + "Geospatial/regions_onshore.geojson",
        regions_offshore=RESOURCES
        + "Geospatial/regions_offshore.geojson",
        state_boundaries=RESOURCES + "Geospatial/state_boundaries.geojson",
        reeds_memberships="repo_data/ReEDS_Constraints/membership.csv",
    output:
        network=RESOURCES + "{demand_level}Dmd/elec.nc",
        busmap=RESOURCES + "{demand_level}Dmd/busmap.csv",
        regions_onshore=RESOURCES
        + "{demand_level}Dmd/Geospatial/regions_onshore_elec.geojson",
        regions_offshore=RESOURCES
        + "{demand_level}Dmd/Geospatial/regions_offshore_elec.geojson",
        network_map_after_transformers=RESOURCES
        + "{demand_level}Dmd/elec_after_transformers_topology.png",
        network_map_after_substations=RESOURCES
        + "{demand_level}Dmd/elec_after_substations_topology.png",
        network_map_after_low_degree=RESOURCES
        + "{demand_level}Dmd/elec_after_low_degree_topology.png",
        network_map_after_electrical_distance=RESOURCES
        + "{demand_level}Dmd/elec_after_electrical_distance_topology.png",
        network_map=RESOURCES
        + "{demand_level}Dmd/elec_topology.png",
    log:
        LOGS + "{demand_level}Dmd/simplify_network/elec.log",
    threads: 1
    resources:
        mem_mb=lambda wildcards, input, attempt: (input_size_bytes(input) // 100000) * attempt * 2,
        walltime=config_provider("walltime", "simplify_network", default="01:30:00"),
    script:
        "../scripts/simplify_network.py"


def phs_shapes_for_extra_components(wildcards):
    cfg = config_for_wildcards(wildcards)
    return {
        f"phs_shp_{hour}": "repo_data/"
        + f"psh/40-100-dam-height-{hour}hr-no-croplands-no-ephemeral-no-highways.gpkg"
        for phs_tech in cfg["electricity"]["extendable_carriers"]["StorageUnit"]
        if "PHS" in phs_tech
        for hour in phs_tech.split("hr_")
        if hour.isdigit()
    }


def tech_costs_for_extra_components(wildcards):
    cfg = config_for_wildcards(wildcards)
    return get_cost_files_for_planning_horizons(cfg=cfg)


def prepare_network_input_network(wildcards):
    cfg = config_for_wildcards(wildcards)
    if cfg["custom_files"].get("activate", False):
        return cfg["custom_files"]["files_path"] + cfg["custom_files"]["network_name"]
    return case_resource_dir(wildcards) + "elec_ec.nc"


def prepare_network_input_costs(wildcards):
    cfg = config_for_wildcards(wildcards)
    if cfg["custom_files"].get("activate", False):
        return cfg["custom_files"]["files_path"] + "costs_2030.csv"
    return get_cost_file(cfg=cfg)


rule add_extra_components:
    input:
        unpack(phs_shapes_for_extra_components),
        network=elec_dem_path,
        tech_costs=tech_costs_for_extra_components,
        sector_costs="repo_data/costs/simple_sector_costs.csv",
        regions_onshore=lambda wildcards: clustered_region_path(wildcards, "onshore"),
        county_shapes=DATA + "counties/cb_2020_us_county_500k.shp",
        reg_cap_cost_diff=(
            "repo_data/locational_multipliers/reg_cap_cost_diff_default.csv"
            if config.get("cost_multipliers", {}).get("enable", False)
            else []
        ),
    params:
        retirement=config_provider("electricity", "retirement", default="technical"),
        trim_network=config_provider("model_topology", "trim", default=False),
        add_extendable_tes=config_provider("electricity", "add_extendable_tes", default=False),
        gas_fuel_price=config_provider("conventional", "gas_fuel_price", default=15),
        cost_multipliers=config_provider("cost_multipliers", default={}),
    output:
        CASE_RESOURCES + "elec_ec.nc",
    log:
        CASE_LOGS + "add_extra_components/elec_ec.log",
    threads: 1
    resources:
        mem_mb=7000,
        walltime=config_provider("walltime", "add_extra_components", default="00:30:00"),
    script:
        "../scripts/add_extra_components.py"


rule build_region_temperature:
    wildcard_constraints:
        demand_level="Low|Mid|High",
    params:
        interconnect=get_interconnect(),
        cutout_dir="cutouts/" + CDIR.rstrip("/") if CDIR else "cutouts",
    input:
        unpack(representative_snapshots_input),
        network=clustered_network_path,
    output:
        region_temperature=RESOURCES
        + "{demand_level}Dmd/region_temperature.csv",
    log:
        LOGS + "{demand_level}Dmd/build_region_temperature/region_temperature.log",
    threads: 4
    resources:
        mem_mb=8000,
        walltime=config_provider("walltime", "build_region_temperature", default="04:00:00"),
    script:
        "../scripts/build_region_temperature.py"


rule prepare_network:
    params:
        time_resolution=config_provider("clustering", "temporal", "resolution_elec"),
        adjustments=False,
        links=config_provider("links"),
        lines=config_provider("lines"),
        co2base=config_provider("electricity", "co2base"),
        co2limit=config_provider("electricity", "co2limit"),
        co2limit_enable=config_provider("electricity", "co2limit_enable", default=False),
        gaslimit=config_provider("electricity", "gaslimit"),
        gaslimit_enable=config_provider("electricity", "gaslimit_enable", default=False),
        transmission_network=config_provider("model_topology", "transmission_network"),
        costs=config_provider("costs"),
    input:
        unpack(representative_metadata_input),
        unpack(temperature_derate_input),
        network=prepare_network_input_network,
        tech_costs=prepare_network_input_costs,
    output:
        CASE_RESOURCES + "elec_ec_l{ll}_{opts}.nc",
    log:
        CASE_LOGS + "prepare_network/elec_ec_l{ll}_{opts}.log",
    threads: 1
    resources:
        walltime=config_provider("walltime", "prepare_network", default="00:30:00"),
        mem_mb=7000,
    script:
        "../scripts/prepare_network.py"
