"""Combines all time independent cost data sources into a standard format."""

import logging

import constants as const
import duckdb
import pandas as pd
from _helpers import calculate_annuity

logger = logging.getLogger(__name__)
CCS_CAPTURE_COST_USD_PER_TON = 20

# Natural gas emissions are split into two components [tCO2/MWh_thermal]:
#  - direct: stack CO2 from combustion, which CCS can capture
#  - indirect: upstream CO2e (extraction, processing, transport), which CCS
#    cannot capture and therefore stays on every gas MWh_th regardless of the
#    capture rate
GAS_DIRECT_CO2_PER_MWHTH = 0.1811
GAS_INDIRECT_CO2_PER_MWHTH = 0.0285
GAS_CO2_PER_MWHTH = GAS_DIRECT_CO2_PER_MWHTH + GAS_INDIRECT_CO2_PER_MWHTH

# Emission parameters written to the cost table, kept at higher precision than
# the other parameters (see the rounding step at the end of __main__).
EMISSION_PARAMETERS = [
    "co2_emissions",
    "co2_emissions_direct",
    "co2_emissions_indirect",
]

# Impute emissions factor data
# https://www.eia.gov/environment/emissions/co2_vol_mass.php
# Units: [tCO2/MWh_thermal]
EMISSIONS_DATA = [
    {"pypsa-name": "coal", "parameter": "co2_emissions", "value": 0.3453},
    {"pypsa-name": "oil", "parameter": "co2_emissions", "value": 0.34851},
    {"pypsa-name": "geothermal", "parameter": "co2_emissions", "value": 0.04029},
    {"pypsa-name": "waste", "parameter": "co2_emissions", "value": 0.1016},
    {"pypsa-name": "gas", "parameter": "co2_emissions", "value": GAS_CO2_PER_MWHTH},
    {
        "pypsa-name": "gas",
        "parameter": "co2_emissions_direct",
        "value": GAS_DIRECT_CO2_PER_MWHTH,
    },
    {
        "pypsa-name": "gas",
        "parameter": "co2_emissions_indirect",
        "value": GAS_INDIRECT_CO2_PER_MWHTH,
    },
    {"pypsa-name": "CCGT", "parameter": "co2_emissions", "value": GAS_CO2_PER_MWHTH},
    {
        "pypsa-name": "CCGT",
        "parameter": "co2_emissions_direct",
        "value": GAS_DIRECT_CO2_PER_MWHTH,
    },
    {
        "pypsa-name": "CCGT",
        "parameter": "co2_emissions_indirect",
        "value": GAS_INDIRECT_CO2_PER_MWHTH,
    },
    {"pypsa-name": "OCGT", "parameter": "co2_emissions", "value": GAS_CO2_PER_MWHTH},
    {
        "pypsa-name": "OCGT",
        "parameter": "co2_emissions_direct",
        "value": GAS_DIRECT_CO2_PER_MWHTH,
    },
    {
        "pypsa-name": "OCGT",
        "parameter": "co2_emissions_indirect",
        "value": GAS_INDIRECT_CO2_PER_MWHTH,
    },
    {
        "pypsa-name": "geothermal",
        "parameter": "heat_rate_mmbtu_per_mwh",
        "value": 8.881,
    },  # AEO 2023
]

LIFETIME_DATA = [
    {"pypsa-name": "coal", "parameter": "lifetime", "value": 70},
    {"pypsa-name": "oil", "parameter": "lifetime", "value": 55},  # using gas CT
    # Confirm with Jabs / NREL. 30 is way too small
    {"pypsa-name": "geothermal", "parameter": "lifetime", "value": 70},
    {"pypsa-name": "waste", "parameter": "lifetime", "value": 55},  # using gas CT
    {"pypsa-name": "CCGT", "parameter": "lifetime", "value": 40},
    {"pypsa-name": "OCGT", "parameter": "lifetime", "value": 40},
    {"pypsa-name": "CCGT-95CCS", "parameter": "lifetime", "value": 40},
    {"pypsa-name": "CCGT-97CCS", "parameter": "lifetime", "value": 40},
    {"pypsa-name": "coal-95CCS", "parameter": "lifetime", "value": 70},
    {"pypsa-name": "coal-99CCS", "parameter": "lifetime", "value": 70},
    {"pypsa-name": "SMR", "parameter": "lifetime", "value": 40},
    {"pypsa-name": "nuclear", "parameter": "lifetime", "value": 60},
    {"pypsa-name": "biomass", "parameter": "lifetime", "value": 30},
    {"pypsa-name": "offwind_floating", "parameter": "lifetime", "value": 30},
    {"pypsa-name": "offwind", "parameter": "lifetime", "value": 30},
    {"pypsa-name": "onwind", "parameter": "lifetime", "value": 30},
    {"pypsa-name": "solar", "parameter": "lifetime", "value": 30},
    {"pypsa-name": "EGS", "parameter": "lifetime", "value": 30},
    {
        "pypsa-name": "2hr_battery_storage",
        "parameter": "lifetime",
        "value": 20,
    },  # inquired with NREL on why they have CRP of 20 but lifetime of 15
    {"pypsa-name": "4hr_battery_storage", "parameter": "lifetime", "value": 20},
    {"pypsa-name": "6hr_battery_storage", "parameter": "lifetime", "value": 20},
    {"pypsa-name": "8hr_battery_storage", "parameter": "lifetime", "value": 20},
    {"pypsa-name": "10hr_battery_storage", "parameter": "lifetime", "value": 20},
]  # https://github.com/NREL/ReEDS-2.0/blob/e65ed5ed4ffff973071839481309f77d12d802cd/inputs/plant_characteristics/maxage.csv#L4


def create_duckdb_instance():
    """Set up DuckDB to read parquet files directly."""
    duckdb.connect(database=":memory:", read_only=False)
    # Install httpfs extension to access remote files if needed
    duckdb.query("INSTALL httpfs;")


def load_pudl_atb_data(parquet_path: str):
    """Loads ATB data directly from parquet files."""
    create_duckdb_instance()

    query = f"""
    WITH finance_cte AS (
        SELECT
            wacc_real,
            technology_description,
            model_case_nrelatb,
            scenario_atb,
            projection_year,
            cost_recovery_period_years,
            report_year
        FROM read_parquet('{parquet_path}/core_nrelatb__yearly_projected_financial_cases_by_scenario.parquet')
    )
    SELECT *
    FROM read_parquet('{parquet_path}/core_nrelatb__yearly_projected_cost_performance.parquet') atb
    LEFT JOIN finance_cte AS finance
        ON atb.technology_description = finance.technology_description
            AND atb.model_case_nrelatb = finance.model_case_nrelatb
            AND atb.scenario_atb = finance.scenario_atb
            AND atb.projection_year = finance.projection_year
            AND atb.cost_recovery_period_years = finance.cost_recovery_period_years
            AND atb.report_year = finance.report_year
    WHERE atb.report_year = 2024
    """
    return duckdb.query(query).to_df()


def load_pudl_aeo_data(parquet_path: str):
    """Loads AEO data directly from parquet files."""
    query = f"""
    SELECT *
    FROM read_parquet('{parquet_path}/core_eiaaeo__yearly_projected_fuel_cost_in_electric_sector_by_type.parquet') aeo
    WHERE aeo.report_year = 2023
    """
    return duckdb.query(query).to_df()


def match_technology(row, tech_dict):
    for key, value in tech_dict.items():
        # Match technology and techdetail
        if row["technology_description"] == value.get("technology") and row[
            "technology_description_detail_1"
        ] == value.get("techdetail"):
            return key
        # Match technology and techdetail2
        elif row["technology_description"] == value.get("technology") and row[
            "technology_description_detail_2"
        ] == value.get("techdetail2"):
            return key

    return None


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_cost_data", year=2030)
        rootpath = ".."
    else:
        rootpath = "."

    costs = snakemake.params.costs
    atb_params = costs.get("atb")
    aeo_params = costs.get("aeo")

    tech_year = snakemake.wildcards.year
    if int(tech_year) < 2024:
        logger.warning(
            "Minimum cost year supported is 2024, using 2024 expansion costs.",
        )
    years = range(2024, 2051)
    tech_year = min(years, key=lambda x: abs(x - int(tech_year)))

    emissions_data = EMISSIONS_DATA

    # Path to parquet files
    parquet_path = snakemake.params.pudl_path

    # Import PUDLs ATB data
    pudl_atb = load_pudl_atb_data(parquet_path)
    pudl_atb["pypsa-name"] = pudl_atb.apply(
        match_technology,
        axis=1,
        tech_dict=const.ATB_TECH_MAPPER,
    )
    pudl_atb = pudl_atb[pudl_atb["pypsa-name"].notnull()]

    # Group by pypsa-name and filter for correct cost recovery period
    pudl_atb = (
        pudl_atb.groupby("pypsa-name")[pudl_atb.columns]
        .apply(
            lambda x: x[x["cost_recovery_period_years"] == const.ATB_TECH_MAPPER[x.name].get("crp", 30)],
        )
        .reset_index(drop=True)
    )

    # Filter for the correct year, scenario, and model case
    pudl_atb_filt = pudl_atb[pudl_atb.projection_year == tech_year]
    if tech_year < 2030:
        logger.warning(
            "Using 2030 ATB data for offwind_floating; earlier data not available.",
        )
        pudl_atb_offwind_floating = pudl_atb[
            (pudl_atb["pypsa-name"] == "offwind_floating") & (pudl_atb.projection_year == 2030)
        ]
        pudl_atb = pd.concat(
            [pudl_atb_filt, pudl_atb_offwind_floating],
            ignore_index=True,
        )
    else:
        pudl_atb = pudl_atb_filt

    pudl_atb = pudl_atb[pudl_atb.scenario_atb == atb_params.get("scenario", "Moderate")]
    pudl_atb = pudl_atb[pudl_atb.model_case_nrelatb == atb_params.get("model_case", "Market")]

    pudl_premelt = pudl_atb.copy()
    # Pivot Data
    cols = [
        "cost_recovery_period_years",
        "capacity_factor",
        "capex_per_kw",
        "capex_overnight_per_kw",
        "capex_overnight_additional_per_kw",
        "capex_grid_connection_per_kw",
        "capex_construction_finance_factor",
        "fuel_cost_per_mwh",
        "heat_rate_mmbtu_per_mwh",
        "heat_rate_penalty",
        "levelized_cost_of_energy_per_mwh",
        "net_output_penalty",
        "opex_fixed_per_kw",
        "opex_variable_per_mwh",
        "wacc_real",
    ]
    # pivot such that cols all get moved to one column
    pudl_atb = pudl_atb.melt(
        id_vars="pypsa-name",
        value_vars=cols,
        var_name="parameter",
        value_name="value",
    )

    emissions_data = EMISSIONS_DATA

    # Impute Transmission Data
    # https://docs.nrel.gov/docs/fy21osti/78195.pdf
    # WACC & Lifetime: https://emp.lbl.gov/publications/improving-estimates-transmission
    # Subsea costs: Purvins et al. (2018): https://doi.org/10.1016/j.jclepro.2018.03.095
    # FOM assumed at 1% of capex per year for all transmission assets
    TRANSMISSION_FOM_PCT = 0.01
    hvdc_inverter_pair_capex_per_kw = 416  # MTEP24

    # AC line and DC line capex are no longer national scalars: they are resolved per
    # line from voltage class and route region in `add_electricity`, off the ReEDS
    # county-pair and base-cost tables. What is still needed here is the annualization
    # basis (recovery period, WACC) and the FOM percentage -- FOM is a share of capex,
    # so once capex varies by line the FOM has to follow it, which means exporting the
    # percentage rather than a pre-multiplied `opex_fixed_per_mw_km`.
    transmission_data = [
        {
            "pypsa-name": "HVAC overhead",
            "parameter": "opex_fixed_pct_of_capex",
            "value": TRANSMISSION_FOM_PCT,
        },
        {
            "pypsa-name": "HVAC overhead",
            "parameter": "cost_recovery_period_years",
            "value": 40,
        },
        {"pypsa-name": "HVAC overhead", "parameter": "wacc_real", "value": 0.036},
        {
            "pypsa-name": "HVDC overhead",
            "parameter": "opex_fixed_pct_of_capex",
            "value": TRANSMISSION_FOM_PCT,
        },
        {
            "pypsa-name": "HVDC overhead",
            "parameter": "cost_recovery_period_years",
            "value": 40,
        },
        {"pypsa-name": "HVDC overhead", "parameter": "wacc_real", "value": 0.036},
        {
            "pypsa-name": "HVDC inverter pair",
            "parameter": "capex_per_kw",
            "value": hvdc_inverter_pair_capex_per_kw,
        },
        {
            "pypsa-name": "HVDC inverter pair",
            "parameter": "opex_fixed_per_kw",
            "value": hvdc_inverter_pair_capex_per_kw * TRANSMISSION_FOM_PCT,
        },
        {
            "pypsa-name": "HVDC inverter pair",
            "parameter": "cost_recovery_period_years",
            "value": 40,
        },
        {"pypsa-name": "HVDC inverter pair", "parameter": "wacc_real", "value": 0.036},
    ]
    pudl_atb = pd.concat(
        [
            pudl_atb,
            pd.DataFrame(emissions_data),
            pd.DataFrame(transmission_data),
            pd.DataFrame(LIFETIME_DATA),
        ],
        ignore_index=True,
    )
    pudl_atb = pudl_atb.drop_duplicates(
        subset=["pypsa-name", "parameter"],
        keep="last",
    )

    # Load AEO Fuel Cost Data
    aeo = load_pudl_aeo_data(parquet_path)
    aeo = aeo[aeo.projection_year == tech_year]
    aeo = aeo[aeo.model_case_eiaaeo == aeo_params.get("scenario", "Reference")]
    cols = ["fuel_type_eiaaeo", "fuel_cost_real_per_mmbtu_eiaaeo"]
    aeo = aeo[cols]
    aeo = aeo.groupby("fuel_type_eiaaeo").mean()
    aeo["fuel_cost_real_per_mwhth"] = aeo["fuel_cost_real_per_mmbtu_eiaaeo"] * 3.412
    aeo = pd.melt(
        aeo.reset_index(),
        id_vars="fuel_type_eiaaeo",
        value_vars=["fuel_cost_real_per_mwhth"],
        var_name="parameter",
        value_name="value",
    )
    aeo = aeo.rename(columns={"fuel_type_eiaaeo": "pypsa-name"})

    addnl_fuels = pd.DataFrame(
        [
            {
                "pypsa-name": "nuclear",
                "parameter": "fuel_cost_real_per_mwhth",
                "value": 2.782,
            },
            {
                "pypsa-name": "EGS",
                "parameter": "fuel_cost_real_per_mwhth",
                "value": 0,
            },
        ],
    )
    aeo = pd.concat([aeo, addnl_fuels], ignore_index=True)

    tech_fuel_map = {
        "CCGT": "natural_gas",
        "OCGT": "natural_gas",
        "CCGT-95CCS": "natural_gas",
        "CCGT-97CCS": "natural_gas",
        "coal-95CCS": "coal",
        "coal-99CCS": "coal",
        "SMR": "nuclear",
    }
    tech_fuels = pd.DataFrame(
        [
            {
                "pypsa-name": new_name,
                "parameter": "fuel_cost_real_per_mwhth",
                "value": aeo.loc[aeo["pypsa-name"] == source_name, "value"].values[0],
            }
            for new_name, source_name in tech_fuel_map.items()
        ],
    )
    aeo = pd.concat([aeo, tech_fuels], ignore_index=True)
    pudl_atb = pd.concat([pudl_atb, aeo], ignore_index=True)

    # Calculate Annualized Costs and Marinal Costs
    # Apply: marginal_cost = opex_variable_per_mwh + fuel_cost_real_per_mwhth / efficiency
    pivot_atb = pudl_atb.pivot(
        index="pypsa-name",
        columns="parameter",
        values="value",
    ).reset_index()

    # Create Hydrogen Combustion Turbine from OCGT using assumptions per ReEDS
    # https://nrel.github.io/ReEDS-2.0/model_documentation.html#hydrogen
    hydrogen_ct = pivot_atb[pivot_atb["pypsa-name"] == "OCGT"].copy()
    hydrogen_ct["pypsa-name"] = "hydrogen_ct"
    hydrogen_ct["capex_overnight_per_kw"] *= 1.1
    hydrogen_ct["capex_per_kw"] = (hydrogen_ct["capex_overnight_per_kw"]
                            + hydrogen_ct["capex_grid_connection_per_kw"]
                            + hydrogen_ct["capex_construction_finance_factor"])
    hydrogen_ct["fuel_cost_real_per_mwhth"] = 20 * 3.412  # 20 USD/MMBtu * 3.412 MMBtu/MWh_th
    hydrogen_ct["co2_emissions"] = 0
    hydrogen_ct["co2_emissions_direct"] = 0
    hydrogen_ct["co2_emissions_indirect"] = 0
    pivot_atb = pd.concat([pivot_atb, hydrogen_ct], ignore_index=True)

    # Apply heat rate corrections
    heat_rate_corrections = {
        "hydrogen_ct": 1.076422072,
        "OCGT": 1.039104223,
        "CCGT": 1.076422072,
        "CCGT-95CCS": 1.076422072,
        "CCGT-97CCS": 1.076422072,
    }

    for tech, correction_factor in heat_rate_corrections.items():
        mask = pivot_atb["pypsa-name"] == tech
        if mask.any():
            pivot_atb.loc[mask, "heat_rate_mmbtu_per_mwh"] *= correction_factor

    pivot_atb["efficiency"] = 3.412 / pivot_atb["heat_rate_mmbtu_per_mwh"]
    egs_mask = pivot_atb["pypsa-name"] == "EGS"
    pivot_atb.loc[egs_mask, "efficiency"] = 1.0
    pivot_atb.loc[egs_mask, "opex_variable_per_mwh"] = 0.0

    # Only the direct (combustion) share of gas emissions can be captured; the
    # indirect (upstream) share passes through unabated.
    pivot_atb["capture_cost_per_mwh"] = 0.0
    for carrier in ("CCGT-95CCS", "CCGT-97CCS"):
        carrier_mask = pivot_atb["pypsa-name"] == carrier
        if not carrier_mask.any():
            continue
        capture_rate = int(carrier.split("-")[1].replace("CCS", "")) / 100
        captured_co2_per_mwh = GAS_DIRECT_CO2_PER_MWHTH / pivot_atb.loc[carrier_mask, "efficiency"] * capture_rate
        pivot_atb.loc[carrier_mask, "capture_cost_per_mwh"] = captured_co2_per_mwh * CCS_CAPTURE_COST_USD_PER_TON

        unabated_direct = (1 - capture_rate) * GAS_DIRECT_CO2_PER_MWHTH
        pivot_atb.loc[carrier_mask, "co2_emissions_direct"] = unabated_direct
        pivot_atb.loc[carrier_mask, "co2_emissions_indirect"] = GAS_INDIRECT_CO2_PER_MWHTH
        pivot_atb.loc[carrier_mask, "co2_emissions"] = unabated_direct + GAS_INDIRECT_CO2_PER_MWHTH

    pivot_atb["fuel_cost"] = pivot_atb["fuel_cost_real_per_mwhth"] / pivot_atb["efficiency"]
    pivot_atb["marginal_cost"] = (
        pivot_atb["opex_variable_per_mwh"] + pivot_atb["fuel_cost"] + pivot_atb["capture_cost_per_mwh"]
    )
    pivot_atb.loc[egs_mask, "fuel_cost"] = 0.0
    pivot_atb.loc[egs_mask, "marginal_cost"] = 0.0

    # Impute storage WACC from Utility Scale Solar. TODO: Revisit this assumption
    for x in [2, 4, 6, 8, 10]:
        pivot_atb.loc[
            pivot_atb["pypsa-name"] == f"{x}hr_battery_storage",
            "wacc_real",
        ] = pivot_atb.loc[
            pivot_atb["pypsa-name"] == "solar",
            "wacc_real",
        ].values[0]
        pivot_atb.loc[
            pivot_atb["pypsa-name"] == f"{x}hr_battery_storage",
            "efficiency",
        ] = 0.85  # 2023 ATB

    pivot_atb["annualized_capex_per_mw"] = (
        calculate_annuity(
            pivot_atb["cost_recovery_period_years"],
            pivot_atb["wacc_real"],
        )
        * pivot_atb["capex_per_kw"]
        * 1
        # change to nyears
    ) * 1e3

    # No `annualized_capex_per_mw_km` here: HVAC/HVDC overhead were the only
    # technologies that ever supplied `capex_per_mw_km`, and their per-km capex is now
    # resolved per line in `add_electricity` from `opex_fixed_pct_of_capex`,
    # `cost_recovery_period_years` and `wacc_real` instead.

    # Calculate grid interrconnection costs per MW-KM
    # All land-based resources assume 1 mile of spur line
    # All offshore resources assume 30 km of subsea cable
    pivot_atb["capex_grid_connection_per_kw_km"] = pivot_atb["capex_grid_connection_per_kw"] / 1.609
    pivot_atb.loc[
        pivot_atb["pypsa-name"].str.contains("offshore"),
        "capex_grid_connection_per_kw_km",
    ] = pivot_atb["capex_grid_connection_per_kw"] / 30

    pivot_atb["annualized_connection_capex_per_mw_km"] = (
        calculate_annuity(
            pivot_atb["cost_recovery_period_years"],
            pivot_atb["wacc_real"],
        )
        * pivot_atb["capex_grid_connection_per_kw_km"]
        * 1
        # change to nyears
    )

    pivot_atb["annualized_capex_fom"] = pivot_atb["annualized_capex_per_mw"] + (pivot_atb["opex_fixed_per_kw"] * 1e3)
    pudl_atb = pivot_atb.melt(
        id_vars=["pypsa-name"],
        value_vars=pivot_atb.columns.difference(["pypsa-name"]),
        var_name="parameter",
        value_name="value",
    )
    pudl_atb = pudl_atb.reset_index(drop=True)
    # Emission factors are small enough that the 3-decimal default would break
    # the direct + indirect = total bookkeeping (0.0285 rounds to 0.028, so the
    # implied direct share becomes 0.182 instead of 0.1811), so they keep more
    # precision.
    emission_rows = pudl_atb["parameter"].isin(EMISSION_PARAMETERS)
    pudl_atb["value"] = pudl_atb["value"].where(
        emission_rows,
        pudl_atb["value"].round(3),
    )
    pudl_atb.loc[emission_rows, "value"] = pudl_atb.loc[emission_rows, "value"].round(6)

    pudl_atb.to_csv(snakemake.output.tech_costs, index=False)
