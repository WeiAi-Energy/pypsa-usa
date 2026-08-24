# Rules to Retrieve Data

breakthrough_datafiles = [
    "bus.csv",
    "sub.csv",
    "bus2sub.csv",
    "branch.csv",
    "dcline.csv",
    "demand.csv",
    "plant.csv",
    "solar.csv",
    "wind.csv",
    "hydro.csv",
    "zone.csv",
]

pypsa_usa_datafiles = [
    "gebco/gebco_2023_tid_USA.nc",
    "gebco/gebco_2023_n55.0_s10.0_w-126.0_e-65.0.tif",
    "copernicus/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_USA_EPSG-4326.tif",
    "eez/conus_eez.shp",
    "natura.tiff",
    "counties/cb_2020_us_county_500k.shp",
]

REEDS_VRE_RECORDS = {
    "onwind": "18827074",
    "solar": "18826700",
    "offwind": "18826680",
}

REEDS_VRE_DATAFILES = {
    "onwind": [
        "cf_wind-ons_reference.h5",
        "sc_wind-ons_reference.csv",
        "sitemap.csv",
    ],
    "solar": [
        "cf_upv_reference.h5",
        "sc_upv_reference.csv",
        "sitemap.csv",
    ],
    "offwind": [
        "cf_wind-ofs_reference.h5",
        "sc_wind-ofs_reference.csv",
        "sitemap_offshore.csv",
    ],
}


def reeds_vre_files_for_technology(wildcards):
    tech = "offwind" if wildcards.technology.startswith("offwind") else wildcards.technology
    return expand(
        DATA + "ReEDS_VRE/{tech}/{file}",
        tech=tech,
        file=REEDS_VRE_DATAFILES[tech],
    )


def reeds_vre_url(wildcards):
    return (
        f"https://zenodo.org/records/{REEDS_VRE_RECORDS[wildcards.reeds_vre_tech]}/files/"
        f"{wildcards.reeds_vre_file}?download=1"
    )


def define_zenodo_databundles():
    return {
        "USATestSystem": "https://zenodo.org/record/4538590/files/USATestSystem.zip",
        "pypsa_usa_data": "https://zenodo.org/records/14219029/files/pypsa_usa_data.zip",
    }


rule retrieve_zenodo_databundles:
    params:
        define_zenodo_databundles(),
    output:
        expand(
            DATA + "breakthrough_network/base_grid/{file}", file=breakthrough_datafiles
        ),
        expand(DATA + "{file}", file=pypsa_usa_datafiles),
    resources:
        mem_mb=5000,
    log:
        "logs/retrieve/retrieve_databundles.log",
    script:
        "../scripts/retrieve_databundles.py"


rule retrieve_eer_demand_data:
    wildcard_constraints:
        eer_file="demand_EER2025_100by2050|demand_EER2025_Baseline_AEO2023|demand_EER2025_IRAlow",
    params:
        url=lambda wildcards: f"https://zenodo.org/records/18435264/files/{wildcards.eer_file}.h5?download=1",
    output:
        DATA + "eer/{eer_file}.h5",
    resources:
        mem_mb=5000,
    log:
        "logs/retrieve/retrieve_eer_{eer_file}.log",
    script:
        "../scripts/retrieve_http_file.py"


rule retrieve_reeds_vre_file:
    wildcard_constraints:
        reeds_vre_tech="onwind|solar|offwind",
        reeds_vre_file="[^/]+",
    params:
        url=reeds_vre_url,
    output:
        DATA + "ReEDS_VRE/{reeds_vre_tech}/{reeds_vre_file}",
    resources:
        mem_mb=5000,
    log:
        LOGS + "retrieve_reeds_vre/{reeds_vre_tech}/{reeds_vre_file}.log",
    retries: 2
    script:
        "../scripts/retrieve_http_file.py"


def egs_enabled_in_any_config():
    if "EGS" in config.get("electricity", {}).get("extendable_carriers", {}).get("Generator", []):
        return True

    for case_name in config.get("cases", {}):
        if "EGS" in case_config(case_name).get("electricity", {}).get("extendable_carriers", {}).get("Generator", []):
            return True

    return False


if egs_enabled_in_any_config():

    rule retrieve_egs:
        params:
            dispatch=config["renewable"]["EGS"]["dispatch"],
            subdir=DATA + "EGS",
        output:
            DATA + "EGS/specs_EGS.nc",
            DATA + "EGS/profile_EGS.nc",
        resources:
            walltime="00:30:00",
            mem_mb=5000,
        log:
            LOGS + "retrieve_EGS.log",
        script:
            "../scripts/retrieve_egs.py"
