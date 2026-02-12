"""
Generic module to add a new energy network.

Reads in the sector wildcard and will call corresponding scripts. In the
future, it would be good to integrate this logic into snakemake
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from _helpers import calculate_annuity, configure_logging, get_currency_conversion_factor, get_snapshots, load_costs
from add_electricity import add_missing_carriers, sanitize_carriers
from add_extra_components import add_co2_emissions
from build_emission_tracking import build_co2_tracking
from build_hydrogen import _calculate_pipeline_capital_costs, _calculate_pipeline_distances, build_hydrogen
from build_natural_gas import StateGeometry, build_natural_gas
from constants import (
    CODE_2_STATE,
    NG_MWH_2_MMCF,
    POWER_MAX,
    STATE_2_CODE,
    STATES_INTERCONNECT_MAPPER,
    HHV_to_LHV_CH4,
    HHV_to_LHV_H2,
    discount_rate,
    leakage_rate,
)
from eia import FuelCosts
from shapely.geometry import Point
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def load_derates_from_csv(state_derates_path: str, national_derates_path: str):
    """
    Load state and national derates from CSV files.

    Parameters
    ----------
    state_derates_path : str
        Path to state derates CSV file
    national_derates_path : str
        Path to national derates CSV file

    Returns
    -------
    tuple
        (state_derates, national_derates) as dictionaries
    """
    # Load CSV files
    state_derates_df = pd.read_csv(state_derates_path)
    national_derates_df = pd.read_csv(national_derates_path)

    # Convert to dictionary format expected by the code
    state_derates = {}
    for carrier in state_derates_df["carrier"].unique():
        carrier_data = state_derates_df[state_derates_df["carrier"] == carrier].copy()
        carrier_data = carrier_data.set_index("state")[["summer_derate", "winter_derate"]]
        state_derates[carrier] = carrier_data

    national_derates = {}
    for _, row in national_derates_df.iterrows():
        national_derates[row["carrier"]] = {
            "summer_derate": row["summer_derate"],
            "winter_derate": row["winter_derate"],
        }

    return state_derates, national_derates


def apply_derates_to_new_components(
    n: pypsa.Network,
    sns: pd.DatetimeIndex,
    state_derates: dict,
    national_derates: dict,
):
    """
    Apply seasonal capacity derates to new/extendable generators and links.
    Uses state-level average derates loaded from CSV resource files.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object
    sns : pd.DatetimeIndex
        Snapshot datetime index
    state_derates : dict
        Dictionary of state-level derates by carrier
    national_derates : dict
        Dictionary of national average derates by carrier
    """
    if not state_derates or not national_derates:
        logger.warning("No derate data provided. Skipping derate application to new components.")
        return

    sns_dt = sns.get_level_values(1) if isinstance(sns, pd.MultiIndex) else sns
    summer_sns = sns_dt[sns_dt.month.isin([6, 7, 8])]
    winter_sns = sns_dt[~sns_dt.month.isin([6, 7, 8])]

    # Map carriers that should use CCGT derates
    ccgt_like_carriers = ["CCGT", "CCGT-95CCS", "biomass-CCS", "h2 turbine", "tes"]

    # ========== Process Generators (nuclear) ==========
    new_gens = n.generators[
        (n.generators.carrier == "nuclear") & (~n.generators.index.str.contains("existing", case=False, na=False))
    ].copy()

    if not new_gens.empty:
        logger.info(f"Applying derates to {len(new_gens)} new generators.")

        # Create p_max_pu dataframe for new generators
        new_gen_p_max_pu = pd.DataFrame(1.0, index=sns_dt, columns=new_gens.index)

        gen_derates_applied = 0
        for gen_idx in new_gens.index:
            summer_derate, winter_derate = _get_derates_for_component(
                n,
                new_gens.loc[gen_idx],
                ccgt_like_carriers,
                state_derates,
                national_derates,
            )

            if summer_derate is not None and winter_derate is not None:
                new_gen_p_max_pu.loc[summer_sns, gen_idx] = summer_derate
                new_gen_p_max_pu.loc[winter_sns, gen_idx] = winter_derate
                gen_derates_applied += 1

        logger.info(f"Applied derates to {gen_derates_applied} generators.")

        # Broadcast across investment horizons if needed
        if len(n.investment_periods) > 1:
            from add_electricity import broadcast_investment_horizons_index

            new_gen_p_max_pu = broadcast_investment_horizons_index(n, new_gen_p_max_pu)
        else:
            new_gen_p_max_pu.index = sns

        # Add to network
        if hasattr(n.generators_t, "p_max_pu") and not n.generators_t.p_max_pu.empty:
            n.generators_t.p_max_pu = pd.concat(
                [n.generators_t.p_max_pu, new_gen_p_max_pu],
                axis=1,
            ).round(3)
        else:
            n.generators_t.p_max_pu = new_gen_p_max_pu.round(3)

    # ========== Process Links (e.g., NGCC-CCS, biomass-CCS, H2 turbine) ==========
    new_links = n.links[
        (n.links.carrier.isin(ccgt_like_carriers))
        & (~n.links.index.str.contains("existing", case=False, na=False))
        & (~n.links.index.str.contains(" charge", case=False, na=False))
    ].copy()

    if not new_links.empty:
        logger.info(f"Applying derates to {len(new_links)} new links.")

        # Create p_max_pu dataframe for new links
        new_link_p_max_pu = pd.DataFrame(1.0, index=sns_dt, columns=new_links.index)

        link_derates_applied = 0
        for link_idx in new_links.index:
            summer_derate, winter_derate = _get_derates_for_component(
                n,
                new_links.loc[link_idx],
                ccgt_like_carriers,
                state_derates,
                national_derates,
            )

            if summer_derate is not None and winter_derate is not None:
                new_link_p_max_pu.loc[summer_sns, link_idx] = summer_derate
                new_link_p_max_pu.loc[winter_sns, link_idx] = winter_derate
                link_derates_applied += 1

        logger.info(f"Applied derates to {link_derates_applied} links.")

        # Broadcast across investment horizons if needed
        if len(n.investment_periods) > 1:
            from add_electricity import broadcast_investment_horizons_index

            new_link_p_max_pu = broadcast_investment_horizons_index(n, new_link_p_max_pu)
        else:
            new_link_p_max_pu.index = sns

        # Add to network
        if hasattr(n.links_t, "p_max_pu") and not n.links_t.p_max_pu.empty:
            n.links_t.p_max_pu = pd.concat(
                [n.links_t.p_max_pu, new_link_p_max_pu],
                axis=1,
            ).round(3)
        else:
            n.links_t.p_max_pu = new_link_p_max_pu.round(3)


def _get_derates_for_component(n, component_row, ccgt_like_carriers, state_derates, national_derates):
    """
    Helper function to get summer and winter derates for a component.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object
    component_row : pd.Series
        Row from component dataframe
    ccgt_like_carriers : list
        List of carriers that should use CCGT derates
    state_derates : dict
        Dictionary of state-level derates by carrier
    national_derates : dict
        Dictionary of national derates by carrier

    Returns
    -------
    tuple
        (summer_derate, winter_derate)
    """
    carrier = component_row["carrier"]
    bus = component_row["bus"] if "bus" in component_row.index else component_row.get("bus0")
    state = n.buses.loc[bus, "STATE"] if bus in n.buses.index and "STATE" in n.buses.columns else None

    # Determine which carrier's derate to use
    # For CCGT-like technologies, use CCGT derates
    if carrier in ccgt_like_carriers:
        derate_carrier = "CCGT"
    else:
        derate_carrier = carrier

    # Get derates for this carrier and state
    summer_derate = None
    winter_derate = None

    if derate_carrier in state_derates:
        # Try to get state-specific derate
        if state and state in state_derates[derate_carrier].index:
            summer_derate = state_derates[derate_carrier].loc[state, "summer_derate"]
            winter_derate = state_derates[derate_carrier].loc[state, "winter_derate"]
        # Otherwise use national average
        elif derate_carrier in national_derates:
            summer_derate = national_derates[derate_carrier]["summer_derate"]
            winter_derate = national_derates[derate_carrier]["winter_derate"]

    return summer_derate, winter_derate


def attach_phs_storageunits(n: pypsa.Network, elec_opts, costs: pd.DataFrame):
    carriers = elec_opts["PHS_exp"]
    carriers = [k for k in carriers if "PHS" in k]

    for carrier in carriers:
        max_hours = int(carrier.split("hr_")[0])

        psh_resources = (
            gpd.read_file(snakemake.input[f"phs_shp_{max_hours}"])
            .to_crs(4326)
            .rename(
                columns={
                    "System Installed Capacity (Megawatts)": "potential_mw",
                    "System Energy Storage Capacity (Gigawatt hours)": "potential_gwh",
                    "System Cost (2020 US Dollars per Installed Kilowatt)": "cost_kw",
                    "Longitude": "longitude",
                    "Latitude": "latitude",
                },
            )
        )[
            [
                "longitude",
                "latitude",
                "potential_gwh",
                "potential_mw",
                "cost_kw",
                "geometry",
            ]
        ]

        # Round CAPEX to $1000 interval
        psh_resources["cost_kw_round"] = (psh_resources["cost_kw"] * 1.13 / 1000).round() * 1000

        # Join SC to PyPSA cluster
        region_onshore = gpd.read_file(snakemake.input.regions_onshore)
        region_onshore_psh = gpd.sjoin(
            region_onshore,
            psh_resources,
            how="inner",
        ).reset_index(drop=True)

        if region_onshore_psh.empty:
            continue

        region_onshore_psh_grp = (
            region_onshore_psh.groupby(["name", "cost_kw_round"])["potential_mw"].agg("sum").reset_index()
        )

        region_onshore_psh_grp["class"] = region_onshore_psh_grp.groupby(["name"]).cumcount() + 1
        region_onshore_psh_grp["class"] = "c" + region_onshore_psh_grp["class"].astype(
            str,
        )
        region_onshore_psh_grp["tech"] = carrier
        region_onshore_psh_grp["carrier"] = region_onshore_psh_grp[["tech", "class"]].agg("_".join, axis=1)
        region_onshore_psh_grp["Generator"] = region_onshore_psh_grp["name"] + " " + region_onshore_psh_grp["carrier"]
        region_onshore_psh_grp = region_onshore_psh_grp.set_index("Generator")

        # Updated annualize capital cost based on real location
        psh_lifetime = 100  # years
        psh_discount_rate = 0.055  # per unit
        psh_fom = 0.885  # %/year
        psh_vom = 0.54  # $/MWh_e

        region_onshore_psh_grp["capital_cost"] = (
            (calculate_annuity(psh_lifetime, psh_discount_rate) + psh_fom / 100)
            * region_onshore_psh_grp["cost_kw_round"]
            * 1e3
            * n.snapshot_weightings.objective.sum()
            / 8760.0
        )

        region_onshore_psh_grp["marginal_cost"] = psh_vom

        # Set RT efficiency = 0.8
        efficiency_store = 0.894427191  # 0.894427191^2 = 0.8
        efficiency_dispatch = 0.894427191  # 0.894427191^2 = 0.8

        costs.at["PHS", "efficiency"] = efficiency_store
        costs.at["PHS", "co2_emissions"] = 0
        add_missing_carriers(n, ["PHS"])
        add_co2_emissions(n, costs, ["PHS"])
        n.madd(
            "StorageUnit",
            region_onshore_psh_grp.index,
            bus=region_onshore_psh_grp.name,
            carrier="PHS",  # region_onshore_psh_grp.tech,
            p_nom_max=region_onshore_psh_grp.potential_mw,
            p_nom_extendable=True,
            capital_cost=region_onshore_psh_grp.capital_cost,
            marginal_cost=region_onshore_psh_grp.marginal_cost,
            efficiency_store=efficiency_store,
            efficiency_dispatch=efficiency_dispatch,
            max_hours=max_hours,
            cyclic_state_of_charge=True,
        )


def handle_nuclear_expansion(n: pypsa.Network, nuclear_exp: bool):
    """
    Handle nuclear expansion logic.
    Moved from prepare_network.py.
    """
    if not nuclear_exp:
        logger.info("Nuclear expansion disabled - removing nuclear generators with zero capacity")
        to_remove = n.generators.index[(n.generators.carrier == "nuclear") & (n.generators.p_nom == 0)]
        for gen in to_remove:
            logger.debug(f"Removing nuclear generator: {gen}")
            n.remove("Generator", gen)
        logger.info(f"Removed {len(to_remove)} nuclear generators with zero capacity")


def assign_bus_2_state(
    n: pypsa.Network,
    shp: str,
    states_2_include: list[str] | None = None,
    state_2_state_name: dict[str, str] | None = None,
) -> None:
    """
    Adds a state column to the network buses dataframe.

    The shapefile must be the counties shapefile
    """
    buses = n.buses[["x", "y"]].copy()
    buses["geometry"] = buses.apply(lambda x: Point(x.x, x.y), axis=1)
    buses = gpd.GeoDataFrame(buses, crs="EPSG:4269")

    states = gpd.read_file(shp).dissolve("STUSPS")["geometry"]
    states = gpd.GeoDataFrame(states)
    if states_2_include:
        states = states[states.index.isin(states_2_include)]

    # project to avoid CRS warning from geopandas
    buses_projected = buses.to_crs("EPSG:3857")
    states_projected = states.to_crs("EPSG:3857")
    gdf = gpd.sjoin_nearest(buses_projected, states_projected, how="left")

    n.buses["STATE"] = n.buses.index.map(gdf.STUSPS)

    if state_2_state_name:
        n.buses["STATE_NAME"] = n.buses.STATE.map(state_2_state_name)


def add_sector_foundation(
    n: pypsa.Network,
    carrier: str,
    add_supply: bool = True,
    costs: pd.DataFrame | None = pd.DataFrame(),
    center_points: pd.DataFrame | None = pd.DataFrame(),
    eia_api: str | None = None,
) -> None:
    """
    Adds carrier, state level bus and store for the energy carrier.

    If add_supply, the store to supply energy will be added. If false,
    only the bus is created and no energy supply will be added to the
    state level bus.
    """
    co2_carrier = carrier
    if carrier == "gas":
        carrier_kwargs = {"color": "#d35050", "nice_name": "Natural Gas"}
    elif carrier == "coal":
        carrier_kwargs = {"color": "#d35050", "nice_name": "Coal"}
    elif carrier == "h2":
        carrier_kwargs = {"color": "#d35050", "nice_name": "Hydrogen"}
    elif carrier == "oil":
        carrier_kwargs = {"color": "#d35050", "nice_name": "Oil"}
    elif carrier == "biomass":
        carrier_kwargs = {"color": "#d35050", "nice_name": "Biomass"}
    else:
        raise ValueError(f"Unknown carrier of {carrier}")

    try:
        carrier_kwargs["co2_emissions"] = costs.at[co2_carrier, "co2_emissions"]
    except KeyError:
        pass

    # make primary energy carriers

    if carrier not in n.carriers.index:
        n.add("Carrier", carrier, **carrier_kwargs)

    # make state level primary energy carrier buses

    states = n.buses.reeds_state.dropna().unique()

    zero_center_points = pd.DataFrame(
        index=states,
        columns=["x", "y"],
        dtype=float,
    ).fillna(0)
    zero_center_points.index.name = "STATE"

    if not center_points.empty:
        points = center_points.loc[states].copy()
        points = (
            pd.concat([points, zero_center_points])
            .reset_index(names=["STATE"])
            .drop_duplicates(keep="first", subset="STATE")
            .set_index("STATE")
        )
    else:
        points = zero_center_points.copy()

    points["name"] = points.index.map(CODE_2_STATE)
    points["interconnect"] = points.index.map(STATES_INTERCONNECT_MAPPER)

    buses_to_create = [f"{x} {carrier}" for x in points.index]
    existing = n.buses[n.buses.index.isin(buses_to_create)].STATE.dropna().unique()

    points = points[~points.index.isin(existing)]

    n.madd(
        "Bus",
        names=points.index,
        suffix=f" {carrier}",
        x=points.x,
        y=points.y,
        carrier=carrier,
        unit="MWh_th",
        interconnect=points.interconnect,
        country=points.index,  # for consistency
        STATE=points.index,
        STATE_NAME=points.name,
    )

    if eia_api:
        year = n.investment_periods[0]
        eia_carrier = "lpg" if carrier == "oil" else carrier
        dyanmic_cost = get_dynamic_marginal_costs(n, eia_carrier, eia_api, year, "power")
        dyanmic_cost = dyanmic_cost.set_index([dyanmic_cost.index.year, dyanmic_cost.index])

        if "USA" not in dyanmic_cost.columns:
            dyanmic_cost["USA"] = dyanmic_cost.mean(axis=1)

        marginal_cost = pd.DataFrame(index=dyanmic_cost.index)
        for state in n.buses.reeds_state.fillna(False).unique():
            if not state:
                continue
            try:
                marginal_cost[state] = dyanmic_cost[state]
            except KeyError:  # use USA average
                marginal_cost[state] = dyanmic_cost["USA"]
    else:
        marginal_cost = 0

    if add_supply:  # only coal
        gens = n.generators[n.generators.carrier == carrier].copy()
        gens["STATE"] = gens.bus.map(n.buses.STATE)
        # Drop any generators that couldn't be mapped to a state
        gens.dropna(subset=["STATE"], inplace=True)
        # Continue only if there are generators to process
        if not gens.empty:
            state_input_capacity = gens.groupby("STATE").apply(lambda df: (df.p_nom / df.efficiency).sum())
            # Filter out states with zero or negligible capacity
            state_input_capacity = state_input_capacity[state_input_capacity > 0]
            # Get the list of states that have generators
            states_with_gens = state_input_capacity.index
            if not states_with_gens.empty:
                e_nom_max_values = state_input_capacity * 8760 * 1.1
                marginal_cost = marginal_cost[states_with_gens]
                n.madd(
                    "Store",
                    names=states_with_gens,
                    suffix=f" {carrier}",
                    bus=[f"{x} {carrier}" for x in states_with_gens],
                    e_nom=0,
                    e_nom_extendable=True,
                    capital_cost=0,
                    e_nom_min=0,
                    e_nom_max=e_nom_max_values,
                    e_min_pu=-1,
                    e_max_pu=0,
                    e_cyclic_per_period=False,
                    carrier=carrier,
                    unit="MWh_th",
                    marginal_cost=marginal_cost,
                    lifetime=np.inf,
                    build_year=n.investment_periods[0],
                )
            # Remove buses for states without generators
            states_without_gens = points.index.difference(states_with_gens)
            if not states_without_gens.empty:
                buses_to_remove = [f"{state} {carrier}" for state in states_without_gens]
                n.mremove("Bus", buses_to_remove)


def get_dynamic_marginal_costs(
    n: pypsa.Network,
    fuel: str,
    eia: str,
    year: int,
    sector: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Gets end-use fuel costs at a state level."""
    sector_mapper = {
        "res": "residential",
        "com": "commercial",
        "pwr": "power",
        "ind": "industrial",
        "trn": "transport",
    }

    assert fuel in ("gas", "lpg", "coal", "heating_oil")

    if fuel == "gas":
        assert sector in ("res", "com", "ind", "pwr")
        if year < 2024:  # get actual monthly values
            raw = FuelCosts(fuel, year, eia, industry=sector_mapper[sector]).get_data(pivot=True)
            raw = raw * 1000 / NG_MWH_2_MMCF  # $/MCF -> $/MWh
        else:  # scale monthly values according to AEO
            act = FuelCosts(fuel, 2023, eia, industry=sector_mapper[sector]).get_data(pivot=True)
            proj = FuelCosts(fuel, year, eia, industry=sector_mapper[sector]).get_data(pivot=True)

            # actual comes in $/MCF, while projected comes in $/MMBTU
            # https://www.eia.gov/totalenergy/data/browser/index.php?tbl=TA4#/?f=A
            # 1.036 BTU / MCF
            act_mmbtu = act / 1.036

            if "USA" not in act.columns:
                act["USA"] = act.mean(axis=1)
                act_mmbtu["USA"] = act_mmbtu.mean(axis=1)

            actual_year_mean = act_mmbtu.mean().at["U.S."]
            proj_year_mean = proj.at[year, "USA"]
            scaler = proj_year_mean / actual_year_mean

            raw = act * scaler * 1000 / NG_MWH_2_MMCF  # $/MCF -> $/MWh
    elif fuel == "coal":
        # https://www.eia.gov/tools/faqs/faq.php?id=72&t=2
        # 19.18 MMBTU per short ton
        mmbtu_per_ston = 19.18
        wh_per_btu = 0.29307  # same as mwh_per_mmbtu

        # no industry = industrial, so use industry = power
        if year < 2024:  # get actual monthly values
            raw = FuelCosts(fuel, year, eia, industry="power").get_data(pivot=True)
            raw *= 1 / mmbtu_per_ston / wh_per_btu  # $/Ton -> $/MWh
        else:
            # idk why, but there is a weird issue from AEO actual costs (ie 2023) dont
            # seem to match actual reported value (or maybe more likely I am interpreteing
            # something wrong). I am taking the profile, then applying the value to the 2024
            # prices, and scaling from that.

            act = FuelCosts(fuel, 2023, eia, industry="power").get_data(pivot=True)
            proj_2024 = FuelCosts(fuel, 2024, eia, industry="power").get_data(pivot=True)
            proj = FuelCosts(fuel, year, eia, industry="power").get_data(pivot=True)

            act *= 1 / mmbtu_per_ston / wh_per_btu  # $/Ton -> $/MWh
            proj *= 1 / wh_per_btu  # $/MMBTU -> $/MWh
            proj_2024 *= 1 / wh_per_btu  # $/MMBTU -> $/MWh

            if "USA" not in act.columns:
                act["USA"] = act.mean(axis=1)

            present_day_scale = proj_2024.at[2024, "USA"] / act.mean().at["USA"]
            act_adjusted = act * present_day_scale

            proj_year_mean = proj.at[year, "USA"]
            scaler = proj_year_mean / act_adjusted.mean().at["USA"]

            raw = act_adjusted * scaler

            """
            act = FuelCosts(fuel, 2023, eia, industry="power").get_data(pivot=True)
            proj = FuelCosts(fuel, year, eia, industry="power").get_data(pivot=True)

            # actual comes in $/ton, while projected comes in $/MMBTU
            proj *= (1 / wh_per_btu) # $/MMBTU -> $/MWh
            act *= (1 / mmbtu_per_ston / wh_per_btu) # $/Ton -> $/MWh

            if "USA" not in act.columns:
                act["USA"] = act.mean(axis=1)

            # actual_year_mean = act.mean().at["USA"]
            proj_year_mean = proj.at[year, "USA"]
            scaler = proj_year_mean / actual_year_mean

            raw = act * scaler
            """

    elif fuel == "lpg":
        # https://www.eia.gov/energyexplained/units-and-calculators/
        btu_per_gallon = 120214
        wh_per_btu = 0.29307
        if year < 2024:
            raw = (
                FuelCosts(fuel, year, eia, grade="regular").get_data(pivot=True)
                * (1 / btu_per_gallon)
                * (1 / wh_per_btu)
                * (1000000)
            )  # $/gal -> $/MWh
        else:
            act = FuelCosts(fuel, 2023, eia, grade="regular").get_data(pivot=True)
            proj = FuelCosts(fuel, year, eia, grade="regular").get_data(pivot=True)

            # actual comes in $/gal, while projected comes in $/MMBTU
            proj *= btu_per_gallon / 1000000

            if "USA" not in act.columns:
                act["USA"] = act.mean(axis=1)

            actual_year_mean = act.mean().at["USA"]
            proj_year_mean = proj.at[year, "USA"]
            scaler = proj_year_mean / actual_year_mean

            # $/gal -> $/MWh
            raw = act * scaler * (1 / btu_per_gallon) * (1 / wh_per_btu) * (1000000)
    elif fuel == "heating_oil":
        # https://www.eia.gov/energyexplained/units-and-calculators/british-thermal-units.php
        btu_per_gallon = 138500
        wh_per_btu = 0.29307
        if year < 2024:
            raw = (
                FuelCosts("heating_oil", year, eia).get_data(pivot=True)
                * (1 / btu_per_gallon)
                * (1 / wh_per_btu)
                * (1000000)
            )  # $/gal -> $/MWh
        else:
            act = FuelCosts("heating_oil", 2023, eia).get_data(pivot=True)
            proj = FuelCosts("heating_oil", year, eia).get_data(pivot=True)

            # actual comes in $/gal, while projected comes in $/MMBTU
            proj *= btu_per_gallon / 1000000

            if "USA" not in act.columns:
                act["USA"] = act.mean(axis=1)

            actual_year_mean = act.mean().at["USA"]
            proj_year_mean = proj.at[year, "USA"]
            scaler = proj_year_mean / actual_year_mean

            # $/gal -> $/MWh
            raw = act * scaler * (1 / btu_per_gallon) * (1 / wh_per_btu) * (1000000)
    else:
        raise KeyError(f"{fuel} not recognized for dynamic fuel costs.")

    # may have to convert full state name to abbreviated state name
    # should probably change the EIA module to be consistent on what it returns...
    raw = raw.rename(columns=STATE_2_CODE)

    raw.index = pd.DatetimeIndex(raw.index)
    raw.index = raw.index.map(lambda x: x.replace(year=year))

    investment_year = n.investment_periods[0]

    hourly_index = pd.date_range(
        start=f"{year}-01-01",
        end=f"{year}-12-31 23:00:00",
        freq="H",
    )

    # need ffill and bfill as some data is not provided at the resolution or
    # timeframe required
    costs_hourly = raw.reindex(hourly_index)
    costs_hourly = costs_hourly.ffill().bfill()
    costs_hourly.index = costs_hourly.index.map(
        lambda x: x.replace(year=investment_year),
    )

    return costs_hourly[costs_hourly.index.isin(n.snapshots.get_level_values(1))]


def convert_generators_2_links(
    n: pypsa.Network,
    carrier: str,
    bus0_suffix: str,
    costs: pd.DataFrame = None,
    simplify_co2: bool = True,
    ts_cost: float = 20,
):
    """
    Replace Generators with a link connecting to a state level primary energy.

    NOTE: THIS WILL ACCOUNT EMISSIONS TOWARDS THE PWR SECTOR

    Links bus1 are the bus the generator is attached to. Links bus0 are state
    level followed by the suffix (ie. "WA gas" if " gas" is the bus0_suffix)

    CO2 emissions are automatically split between atmosphere (bus2) and capture (bus3).
    For gas technologies with CCS (e.g., "CCGT-95CCS"), the capture rate is parsed from
    the carrier name. For biomass and other carriers, CO2 values are taken directly from
    the costs dataframe.
    """
    plants = n.generators[n.generators.carrier == carrier].copy()

    if plants.empty:
        return

    plants["STATE"] = plants.bus.map(n.buses.STATE)

    if bus0_suffix == " gas":
        plants["efficiency"] = plants["efficiency"] * HHV_to_LHV_CH4

    # Calculate CO2 intensity and capture rate
    # Check if this is a gas technology that needs CCS parsing
    is_gas_with_ccs = ("CCGT" in carrier or "OCGT" in carrier) and "CCS" in carrier

    if is_gas_with_ccs:
        # For gas technologies with CCS, parse capture rate from carrier name
        base_carrier = "gas"
        base_co2_emissions = costs.at[base_carrier, "co2_emissions"]

        # Parse capture rate from carrier name (e.g., "95" from "CCGT-95CCS")
        import re

        match = re.search(r"(\d+)CCS", carrier)
        capture_rate = float(match.group(1)) / 100.0

        # Calculate emissions to atmosphere and capture
        co2_to_atmosphere = base_co2_emissions * (1 - capture_rate)
        co2_to_capture = base_co2_emissions * capture_rate
    else:
        # For other carriers (including biomass/biomass-CCS), use value directly from costs
        co2_emissions = costs.at[carrier, "co2_emissions"]
        if co2_emissions > 0:  # gas w/o ccs
            co2_to_atmosphere = co2_emissions
            co2_to_capture = 0
        else:  # biomass
            co2_to_atmosphere = co2_emissions
            co2_to_capture = -co2_emissions

    pnl = {}

    # copy over pnl parameters
    for c in n.iterate_components(["Generator"]):
        for param, df in c.pnl.items():
            # skip result vars
            if param not in (
                "p_min_pu",
                "p_max_pu",
                "p_set",
                "q_set",
                "marginal_cost",
                "marginal_cost_quadratic",
                "efficiency",
                "stand_by_cost",
            ):
                continue
            cols = [p for p in plants.index if p in df.columns]
            if cols:
                pnl[param] = df[cols]
    if simplify_co2:
        n.madd(
            "Link",
            names=plants.index,
            bus0=plants.STATE + bus0_suffix,
            bus1=plants.bus,
            bus2=plants.STATE + " co2 atmosphere",
            carrier=plants.carrier,
            p_nom_min=plants.p_nom_min / plants.efficiency,
            p_nom=plants.p_nom / plants.efficiency,  # links rated on input capacity
            p_nom_max=plants.p_nom_max / plants.efficiency,
            p_nom_extendable=plants.p_nom_extendable,
            ramp_limit_up=plants.ramp_limit_up,
            ramp_limit_down=plants.ramp_limit_down,
            efficiency=plants.efficiency,
            efficiency2=co2_to_atmosphere,
            marginal_cost=plants.vom_cost * plants.efficiency + ts_cost * co2_to_capture,
            capital_cost=plants.capital_cost * plants.efficiency,  # links rated on input capacity
            lifetime=plants.lifetime,
            build_year=plants.build_year,
        )
    else:
        n.madd(
            "Link",
            names=plants.index,
            bus0=plants.STATE + bus0_suffix,
            bus1=plants.bus,
            bus2=plants.STATE + " co2 atmosphere",
            bus3=plants.STATE + " co2 capture",
            carrier=plants.carrier,
            p_nom_min=plants.p_nom_min / plants.efficiency,
            p_nom=plants.p_nom / plants.efficiency,  # links rated on input capacity
            p_nom_max=plants.p_nom_max / plants.efficiency,
            p_nom_extendable=plants.p_nom_extendable,
            ramp_limit_up=plants.ramp_limit_up,
            ramp_limit_down=plants.ramp_limit_down,
            efficiency=plants.efficiency,
            efficiency2=co2_to_atmosphere,
            efficiency3=co2_to_capture,
            marginal_cost=plants.vom_cost * plants.efficiency,
            capital_cost=plants.capital_cost * plants.efficiency,  # links rated on input capacity
            lifetime=plants.lifetime,
            build_year=plants.build_year,
        )
        n.links["efficiency3"] = n.links.efficiency3.fillna(0)

    for param, df in pnl.items():
        n.links_t[param] = n.links_t[param].join(df, how="inner")

    n.mremove("Generator", plants.index)

    # existing links will give a 'nan in efficiency2/efficiency3' warning
    n.links["efficiency2"] = n.links.efficiency2.fillna(0)


def split_loads_by_carrier(n: pypsa.Network):
    """
    Splits loads by carrier.

    At this point, all loads (ie. com-elec, com-heat, com-cool) will be
    nested under the elec bus. This function will create a new bus-load
    pair for each energy carrier that is NOT electricity.

    Note: This will break the flow of energy in the model! You must add a
    new link between the new bus and old bus if you want to retain the flow
    """
    df_temp = n.loads.copy()
    df_temp.index = df_temp.apply(
        lambda row: row["bus"] if row["carrier"] == "AC" else row.name,
        axis=1,
    )
    for bus in n.buses.country.dropna().unique():
        df = df_temp[(df_temp.bus == bus) & (df_temp.carrier != "AC")][["bus", "carrier"]]
        n.madd(
            "Bus",
            df.index,
            v_nom=1,
            x=n.buses.at[bus, "x"],
            y=n.buses.at[bus, "y"],
            carrier=df.carrier,
            country=n.buses.at[bus, "country"],
            STATE=n.buses.at[bus, "STATE"],
            STATE_NAME=n.buses.at[bus, "STATE_NAME"],
        )

    n.loads["bus"] = df_temp.index


def build_biomass(
    n: pypsa.Network,
    agri_potential: str,
    forestry_potential: str,
    wastes_potential: str,
    reference_network_path: str,
    **kwargs,
) -> None:
    """
    Add biomass production potential to network with multi-platform supply curve approximation.

    This improved version uses K-means clustering to approximate each state's supply curve
    with three segments, providing better representation of biomass cost structure.

    Args:
        n: PyPSA network
        agri_potential: Path to agricultural biomass potential CSV
        forestry_potential: Path to forestry biomass potential CSV
        wastes_potential: Path to wastes biomass potential CSV
    """
    # 1. Read and process the three biomass data files
    agri_df = pd.read_csv(agri_potential)
    forestry_df = pd.read_csv(forestry_potential)
    wastes_df = pd.read_csv(wastes_potential)

    # Standardize column names for each dataset
    agri_df = agri_df.rename(
        columns={
            "Price Offered": "price_offered",
            "Production Energy Content": "production_energy_content",
            "Energy Content Unit": "energy_content_unit",
            "Production": "production",
            "State": "state",
        },
    )

    forestry_df = forestry_df.rename(
        columns={
            "Scenario Price Offered": "price_offered",
            "Production Energy Content": "production_energy_content",
            "Energy Content Unit": "energy_content_unit",
            "Production": "production",
            "State": "state",
        },
    )

    wastes_df = wastes_df.rename(
        columns={
            "Price Offered": "price_offered",
            "Production Energy Content": "production_energy_content",
            "Energy Content Unit": "energy_content_unit",
            "Production": "production",
            "Total Production Density per Sq Mile": "Production Density per Sq Mile",
            "State": "state",
        },
    )

    agri_df["price_offered"] = agri_df["price_offered"] + 40  # NZA
    forestry_df["price_offered"] = forestry_df["price_offered"] + 40

    # Convert state names to state codes for agri_df and forestry_df before combining
    agri_df["state"] = agri_df["state"].map(STATE_2_CODE).fillna(agri_df["state"])
    forestry_df["state"] = forestry_df["state"].map(STATE_2_CODE).fillna(forestry_df["state"])

    # 2. Combine all three datasets
    combined_df = pd.concat([agri_df, forestry_df, wastes_df], ignore_index=True)

    # 3. Apply filtering and adjustments for 'low' scenario
    # if 'nt' in agri_potential.lower():
    #     # Resources to exclude in low scenario
    #     resources_to_exclude = [
    #         'Miscanthus', 'Switchgrass', 'Biomass sorghum', 'Energy cane',
    #         'Plastics', 'Poplar', 'Willow', 'Hardwood, lowland whole trees',
    #         'Softwood, planted whole trees', 'Softwood, natural whole trees',
    #         'Other hardwood, planted whole trees', 'Eucalyptus'
    #     ]
    #
    #     # Drop rows with excluded resources
    #     combined_df = combined_df[~combined_df['Resource'].isin(resources_to_exclude)]

    # Drop rows of unusable waste
    resources_to_exclude = [
        "Rubber and leather",
        "Textiles",
        "Plastics",
        "Landfill gas",
        "Food waste, nonresidential",
        "Food waste, residential",
        "Manure, beef",
        "Manure, dairy",
        "Manure, swine",
        "Sludge",
        "FOG, animal fats",
        "FOG, brown grease",
        "FOG, yellow grease",
    ]
    combined_df = combined_df[~combined_df["Resource"].isin(resources_to_exclude)]

    # Adjust Corn stover production energy content
    corn_stover_mask = combined_df["Resource"] == "Corn stover"
    if corn_stover_mask.any():
        combined_df.loc[corn_stover_mask, "production_energy_content"] *= 0.62

    # Remove the lowest producing counties
    # if 'nt' in agri_potential.lower():
    #     threshold = combined_df['Production Density per Sq Mile'].quantile(0.32)
    # else:
    #     threshold = combined_df['Production Density per Sq Mile'].quantile(0.35)
    # combined_df = combined_df[combined_df['Production Density per Sq Mile'] > threshold]

    def convert_price_to_usd_per_mwh(df):
        btu_per_ton = df["production_energy_content"] / df["production"]
        # Convert BTU to MWh, 1 BTU = 2.93071e-7 MWh
        mwh_per_ton = btu_per_ton * 2.93071e-7
        df["price_offered"] = df["price_offered"] / mwh_per_ton
        return df

    combined_df = convert_price_to_usd_per_mwh(combined_df)

    # Keep only necessary columns
    combined_df = combined_df[["state", "price_offered", "production_energy_content"]]

    # Remove rows with missing values
    combined_df = combined_df.dropna(subset=["price_offered", "production_energy_content"])

    combined_df = combined_df[combined_df["price_offered"] <= 100]

    n_clusters = 3

    if reference_network_path:
        # Load the reference network
        n_ref = pypsa.Network(reference_network_path)

        # Calculate biomass used by h2 bio-cc links
        h2_bio_links = n_ref.links[n_ref.links.carrier == "h2 bio-cc"].index
        if not h2_bio_links.empty:
            # Get state mapping for h2 bio-cc links
            h2_bio_states = (
                n_ref.links.loc[h2_bio_links, "bus0"]
                .map(
                    lambda x: n_ref.buses.loc[x.replace(" biomass", ""), "STATE"] if " biomass" in x else None,
                )
                .dropna()
            )

            # Track total biomass removed across all states
            total_biomass_removed_mwh = 0

            # For each state with h2 bio-cc links, remove biomass from combined_df
            for state in h2_bio_states.unique():
                state_h2_bio_links = h2_bio_states[h2_bio_states == state].index
                biomass_to_remove = n_ref.links_t.p0[state_h2_bio_links].mean().sum() * 8760

                if biomass_to_remove > 0:
                    # Sort state data by price (low to high)
                    state_data = combined_df[combined_df["state"] == state].sort_values("price_offered")

                    # Remove biomass starting from lowest price
                    cumulative_energy = 0
                    rows_to_drop = []

                    for idx, row in state_data.iterrows():
                        if cumulative_energy >= biomass_to_remove:
                            break

                        energy_content = row["production_energy_content"] * 2.93071e-7  # Convert BTU to MWh

                        if cumulative_energy + energy_content <= biomass_to_remove:
                            # Remove entire row
                            rows_to_drop.append(idx)
                            cumulative_energy += energy_content
                        else:
                            # Partially reduce this row
                            remaining_to_remove = biomass_to_remove - cumulative_energy
                            reduction_fraction = remaining_to_remove / energy_content
                            combined_df.loc[idx, "production_energy_content"] *= 1 - reduction_fraction
                            cumulative_energy = biomass_to_remove

                    # Drop fully consumed rows
                    combined_df = combined_df.drop(rows_to_drop)

                    # Accumulate total removed biomass
                    total_biomass_removed_mwh += cumulative_energy

            # Log total biomass removed across all states
            logger.info(f"Total biomass removed from supply curve: {total_biomass_removed_mwh / 1e6:.2f} TWh")

    # 4. Apply K-means clustering for multi-platform approximation per state
    states_in_model = n.buses["reeds_state"].replace("", np.nan).dropna().unique()

    # Initialize lists to store results for all states
    all_platforms = []

    for state in states_in_model:
        state_data = combined_df[combined_df["state"] == state]

        if state_data.empty:
            logger.warning(f"No biomass data for state {state}, skipping")
            continue

        # Sort state data by price for K-means clustering
        state_data_sorted = state_data.sort_values("price_offered").reset_index(drop=True)

        # Prepare data for K-means: use price as feature, weighted by energy content
        prices = state_data_sorted["price_offered"].values.reshape(-1, 1)
        sample_weights = state_data_sorted["production_energy_content"].values

        # Adjust n_clusters based on both available samples AND unique price values
        n_samples = len(state_data_sorted)
        n_unique_prices = len(state_data_sorted["price_offered"].unique())
        effective_n_clusters = min(n_clusters, n_samples, n_unique_prices)

        if effective_n_clusters < n_clusters:
            logger.warning(
                f"State {state} has {n_samples} samples with {n_unique_prices} unique prices, using {effective_n_clusters} clusters instead of {n_clusters}",
            )

        # Apply K-means clustering using built-in sample_weight
        kmeans = KMeans(n_clusters=effective_n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(prices, sample_weight=sample_weights)

        state_data_sorted["cluster"] = clusters

        # Calculate platform characteristics for each cluster
        for cluster_id in sorted(state_data_sorted["cluster"].unique()):
            cluster_data = state_data_sorted[state_data_sorted["cluster"] == cluster_id]

            total_energy = cluster_data["production_energy_content"].sum()
            weighted_avg_price = np.average(
                cluster_data["price_offered"],
                weights=cluster_data["production_energy_content"],
            )

            all_platforms.append(
                {
                    "state": state,
                    "platform": cluster_id + 1,  # Platform IDs 1, 2, 3, ...
                    "total_energy_btu": total_energy,
                    "marginal_cost": weighted_avg_price,
                }
            )

    # Convert to DataFrame
    platforms_df = pd.DataFrame(all_platforms)

    if platforms_df.empty:
        logger.error("No biomass platforms created")
        return

    # Reorder platforms by marginal cost within each state
    platforms_df = platforms_df.sort_values(["state", "marginal_cost"]).reset_index(drop=True)
    platforms_df["platform"] = platforms_df.groupby("state").cumcount() + 1

    # Convert BTU to MWh (assuming the data is already in BTU units from the energy content conversion)
    platforms_df["total_energy_mwh"] = platforms_df["total_energy_btu"] * 2.93071e-7

    # 5. Add PyPSA components - Create stores for all platforms per state
    for _, platform in platforms_df.iterrows():
        state = platform["state"]
        platform_id = platform["platform"]
        # Create unique names for each platform
        store_name = f"{state} biomass-{platform_id}"
        bus_name = f"{state} biomass production-{platform_id}"

        # Add biomass production bus for this platform
        n.madd(
            "Bus",
            names=[bus_name],
            carrier="biomass production",
            unit="MWh_th",
            country=state,
            interconnect=n.buses[n.buses.reeds_state == state].interconnect.iloc[0],
            STATE=state,
        )

        # Add Link connecting biomass production bus to state biomass bus
        n.madd(
            "Link",
            names=[store_name],
            suffix=" production",
            carrier="biomass production",
            unit="MW",
            bus0=bus_name,
            bus1=f"{state} biomass",
            efficiency=1,
            p_nom_extendable=False,
            p_nom=POWER_MAX,
            p_min_pu=0,
            p_max_pu=1,
            capital_cost=0,
            marginal_cost=0,
            lifetime=np.inf,
            build_year=n.investment_periods[0],
        )

        # Add Store connected to biomass production bus (similar to gas production store)
        n.madd(
            "Store",
            names=[store_name],
            unit="MWh",
            bus=bus_name,  # Connect to production bus, not state biomass bus
            carrier="biomass production",
            capital_cost=0,
            marginal_cost=platform["marginal_cost"],
            e_cyclic=False,
            e_cyclic_per_period=False,
            e_nom=0,
            e_nom_max=platform["total_energy_mwh"],
            e_nom_extendable=True,
            e_min_pu=-1,
            e_max_pu=0,
            lifetime=np.inf,
            build_year=n.investment_periods[0],
        )

    logger.info("Successfully added multi-platform biomass supply to network")


def build_dac(n: pypsa.Network, **kwargs) -> None:
    """Add direct air capture to network."""
    logger.info("Adding direct air capture")

    df = pd.DataFrame(
        {
            "state": n.buses["reeds_state"].replace("", np.nan).dropna().unique(),
        }
    )

    addi_costs = kwargs.get("addi_costs", None)
    cost_multiplier = kwargs.get("cost_multiplier", 1.0)
    simplify_co2 = kwargs.get("simplify_co2", True)
    ts_cost = kwargs.get("ts_cost", 20)

    elec_intput = addi_costs.loc["direct air capture", "elec-input"]  # MWh/tCO2
    investment = addi_costs.loc["direct air capture", "investment"]  # USD/(tCO2/h)
    FOM = addi_costs.loc["direct air capture", "FOM"]  # %/year
    VOM = addi_costs.loc["direct air capture", "VOM"]  # USD/tCO2
    lifetime = addi_costs.loc["direct air capture", "lifetime"]
    currency_year = addi_costs.loc["direct air capture", "currency_year"]
    efficiency = 1 / elec_intput  # tCO2/MWh_el
    capital_cost = (
        (calculate_annuity(lifetime, discount_rate) + FOM * 0.01)
        * investment
        * get_currency_conversion_factor(currency_year, "USD")
        * efficiency
    )  # USD/MW_el

    if "dac" not in n.carriers.index:
        n.add("Carrier", "dac", color="#d35050", nice_name="Direct Air Capture")

    if simplify_co2:
        n.madd(
            "Link",
            names=df.state,
            suffix=" dac",
            bus0=[f"{state}" for state in df.state],
            bus1=[f"{state} co2 atmosphere" for state in df.state],
            carrier="dac",
            p_nom_extendable=True,
            p_nom_max=POWER_MAX,
            efficiency=-efficiency,
            capital_cost=capital_cost * cost_multiplier,
            marginal_cost=(ts_cost + VOM * get_currency_conversion_factor(currency_year, "USD")) * efficiency,
            lifetime=lifetime,
            build_year=n.investment_periods[0],
        )
    else:
        n.madd(
            "Link",
            names=df.state,
            suffix=" dac",
            bus0=[f"{state}" for state in df.state],
            bus1=[f"{state} co2 atmosphere" for state in df.state],
            bus2=[f"{state} co2 capture" for state in df.state],
            carrier="dac",
            p_nom_extendable=True,
            p_nom_max=POWER_MAX,
            efficiency=-efficiency,
            efficiency2=efficiency,
            capital_cost=capital_cost * cost_multiplier,
            lifetime=lifetime,
            build_year=n.investment_periods[0],
        )


def build_methanation(n: pypsa.Network, **kwargs) -> None:
    """Add methanation to network."""
    logger.info("Adding methanation")

    df = pd.DataFrame(
        {
            "state": n.buses["reeds_state"].replace("", np.nan).dropna().unique(),
        }
    )

    addi_costs = kwargs.get("addi_costs", None)
    simplify_co2 = kwargs.get("simplify_co2", True)
    hydrogen_input = (
        addi_costs.loc["methanation", "hydrogen-input"] * HHV_to_LHV_CH4 / HHV_to_LHV_H2
    )  # kWh_H2_HHV/kWh_gas_HHV / (MWh_H2_HHV/MWh_H2_LHV) * (MWh_gas_HHV/MWh_gas_LHV)
    investment = (
        addi_costs.loc["methanation", "investment"] * HHV_to_LHV_CH4
    )  # USD/kW_gas_HHV * MWh_gas_HHV/MWh_gas_LHV
    FOM = addi_costs.loc["methanation", "FOM"]  # %/year
    lifetime = addi_costs.loc["methanation", "lifetime"]
    currency_year = addi_costs.loc["methanation", "currency_year"]
    efficiency = 1 / hydrogen_input  # MWh_gas/MWh_h2
    capital_cost = (
        (calculate_annuity(lifetime, discount_rate) + FOM * 0.01)
        * investment
        * 1000
        * get_currency_conversion_factor(currency_year, "USD")
        * efficiency
    )  # USD/MW_h2

    if "gas methanation" not in n.carriers.index:
        n.add("Carrier", "gas methanation", color="#d35050", nice_name="Gas Methanation")

    if simplify_co2:
        n.madd(
            "Link",
            names=df.state,
            suffix=" gas methanation",
            bus0=[f"{state} h2" for state in df.state],
            bus1=[f"{state} gas" for state in df.state],
            bus2=[f"{state} co2 atmosphere" for state in df.state],
            carrier="gas methanation",
            p_nom_extendable=True,
            p_nom_max=POWER_MAX,
            efficiency=efficiency,
            efficiency2=leakage_rate
            * 0.072
            * 28
            * efficiency
            * 0.5,  # tCO2/MWh_H2 = tCO2/MWh_gas * MWh_gas/MWh_H2, https://pubs.acs.org/doi/10.1021/acs.est.1c05246?src=getftr&utm_source=scopus&getft_integrator=scopus
            capital_cost=capital_cost,
            lifetime=lifetime,
            build_year=n.investment_periods[0],
        )
    else:
        n.madd(
            "Link",
            names=df.state,
            suffix=" gas methanation",
            bus0=[f"{state} h2" for state in df.state],
            bus1=[f"{state} gas" for state in df.state],
            bus2=[f"{state} co2 capture" for state in df.state],
            bus3=[f"{state} co2 atmosphere" for state in df.state],
            carrier="gas methanation",
            p_nom_extendable=True,
            p_nom_max=POWER_MAX,
            efficiency=efficiency,
            efficiency2=-0.2002 * efficiency,  # tCO2/MWh_H2 = tCO2/MWh_gas * MWh_gas/MWh_H2
            efficiency3=leakage_rate
            * 0.072
            * 28
            * efficiency
            * 0.5,  # tCO2/MWh_H2 = tCO2/MWh_gas * MWh_gas/MWh_H2, https://pubs.acs.org/doi/10.1021/acs.est.1c05246?src=getftr&utm_source=scopus&getft_integrator=scopus
            capital_cost=capital_cost,
            lifetime=lifetime,
            build_year=n.investment_periods[0],
        )


def build_co2_sequestration(n: pypsa.Network, **kwargs) -> None:
    """Add co2 sequestration to network."""
    logger.info("Adding co2 sequestration")

    df = pd.DataFrame(
        {
            "state": n.buses["reeds_state"].replace("", np.nan).dropna().unique(),
        }
    )

    n.madd(
        "Link",
        names=df.state,
        suffix=" co2 sequestration",
        bus0=[f"{state} co2 capture" for state in df.state],
        bus1=[f"{state} co2 sequestration" for state in df.state],
        carrier="co2",
        p_nom_extendable=True,
        p_nom_max=POWER_MAX,
        efficiency=1,
        p_nom_min=0,
        build_year=n.investment_periods[0],
    )


def build_co2_pipeline(n: pypsa.Network, **kwargs) -> None:
    """
    Add co2 pipelines between state pairs based on AC transmission lines.

    Args:
        n: PyPSA network
        **kwargs: Additional parameters including addi_costs, length_factor
    """
    logger.info("Adding CO2 pipeline")

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

    logger.info(f"Found {len(state_pairs)} state pairs for co2 pipelines from AC and gas pipeline topology")

    # 2. Create dataframe with state pairs
    df = pd.DataFrame(list(state_pairs), columns=["STATE_FROM", "STATE_TO"])
    df.index = df.STATE_FROM + " " + df.STATE_TO + " co2 pipeline new"

    # 3. Calculate distances and costs
    addi_costs = kwargs.get("addi_costs", None)
    length_factor = kwargs.get("length_factor", 1.25)

    fom = addi_costs.loc["CO2 pipeline", "FOM"]
    investment = addi_costs.loc["CO2 pipeline", "investment"]  # EUR/(tCO2/h)/km
    lifetime = addi_costs.loc["CO2 pipeline", "lifetime"]
    currency_year = addi_costs.loc["CO2 pipeline", "currency_year"]

    # Calculate capital costs and electricity efficiency
    distances = _calculate_pipeline_distances(df, n) * length_factor
    capital_costs = _calculate_pipeline_capital_costs(
        df,
        n,
        fom,
        investment,
        lifetime,
        length_factor,
        currency_year,
    )  # USD/(tCO2/h)
    # 4. Add carrier if not exists
    if "co2 pipeline new" not in n.carriers.index:
        n.add("Carrier", "co2 pipeline new", color="#ea048a", nice_name="CO2 Pipeline New")

    # 5. Add CO2 pipelines
    n.madd(
        "Link",
        names=df.index,
        carrier="co2 pipeline new",
        bus0=df.STATE_FROM + " co2 capture",
        bus1=df.STATE_TO + " co2 capture",
        p_nom_extendable=True,
        p_nom=0,
        p_nom_min=0,
        p_nom_max=POWER_MAX,
        p_min_pu=-1,
        p_max_pu=1,
        capital_cost=capital_costs,
        length=distances,
        marginal_cost=0,
        lifetime=lifetime,
        build_year=n.investment_periods[0],
    )


def build_ldes(n: pypsa.Network, **kwargs) -> None:
    """Add LDES to the network using a generalized loop."""
    logger.info("Adding long-duration energy storage")

    # Get unique states/regions
    unique_states = n.buses["reeds_state"].replace("", np.nan).dropna().unique()
    df = pd.DataFrame(index=unique_states)
    df["state"] = unique_states

    # Get inputs from keyword arguments
    addi_costs = kwargs.get("addi_costs")
    ldes_options = kwargs.get("ldes_options", {})

    # Define configurations for different storage technologies
    tech_configs = {
        "TES": {
            "carrier_name": "tes",
            "nice_name": "Thermal Energy Storage",
            "unit": "MWh_th",
            "color": "#d35050",
        },
        "ACAES": {
            "carrier_name": "acaes new",
            "nice_name": "Advanced Compressed Air Energy Storage New",
            "unit": "MWh",
            "color": "#ff69b4",  # Different colors can be set for each technology
        },
    }

    # Iterate over all configured storage technologies
    for tech_key, config in tech_configs.items():
        # Check if this technology is enabled in ldes_options
        if not ldes_options.get(tech_key, False):
            continue

        logger.info(f"Adding LDES technology: {tech_key}")

        # 1. Extract technology-specific parameters from addi_costs
        params = addi_costs.loc[tech_key]
        lifetime = params["lifetime"]
        currency_year = params["currency_year"]
        FOM = params["FOM"]
        VOM = params["VOM"]

        # 2. Calculate annualized and marginal costs
        annuity = calculate_annuity(lifetime, discount_rate)
        fom_rate = FOM * 0.01
        inflation_adjustment = get_currency_conversion_factor(currency_year, "USD")
        if tech_key == "TES":
            store_capital_cost = (annuity + fom_rate) * params["energy_cost"] * inflation_adjustment
        else:
            store_capital_cost = (
                (annuity + fom_rate) * (params["air_storage_cost"] + params["tes_cost"]) * inflation_adjustment
            )
        charge_capital_cost = (annuity + fom_rate) * params["charge_cost"] * inflation_adjustment
        discharge_capital_cost = (annuity + fom_rate) * params["discharge_cost"] * inflation_adjustment
        discharge_marginal_cost = VOM * 0.01 * params["discharge_cost"]

        carrier = config["carrier_name"]

        # 3. Add a Carrier for the technology if it doesn't already exist
        if carrier not in n.carriers.index:
            n.add(
                "Carrier",
                carrier,
                color=config["color"],
                nice_name=config["nice_name"],
            )

        # 4. Use n.madd to add Bus, Store, and Link components
        bus_suffix = f" {carrier}"
        bus_names = df.index + bus_suffix

        n.madd(
            "Bus",
            names=df.index,
            suffix=bus_suffix,
            carrier=carrier,
            unit=config["unit"],
            country=df.index,
            reeds_state=df.index,
            STATE=df.index,
        )

        n.madd(
            "Store",
            names=df.index,
            suffix=bus_suffix,
            bus=bus_names,
            carrier=carrier,
            capital_cost=store_capital_cost,
            standing_loss=params["loss"],
            e_cyclic=True,
            e_nom_extendable=True,
            lifetime=lifetime,
            build_year=n.investment_periods[0],
        )

        n.madd(
            "Link",
            names=df.index,
            suffix=f"{bus_suffix} charge",
            bus0=df.index,
            bus1=bus_names,
            carrier=carrier,
            p_nom_extendable=True,
            p_nom_max=POWER_MAX,
            efficiency=params["charge_efficiency"],
            capital_cost=charge_capital_cost,
            lifetime=lifetime,
            build_year=n.investment_periods[0],
        )

        n.madd(
            "Link",
            names=df.index,
            suffix=f"{bus_suffix} discharge",
            bus0=bus_names,
            bus1=df.index,
            carrier=carrier,
            p_nom_extendable=True,
            p_nom_max=POWER_MAX,
            efficiency=params["discharge_efficiency"],
            capital_cost=discharge_capital_cost,
            marginal_cost=discharge_marginal_cost,
            lifetime=lifetime,
            build_year=n.investment_periods[0],
        )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_simple_sectors",
            case="HighE_new_h2storage_tes",
            transmission_network="reeds",
        )
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)

    eia_api = snakemake.params.api["eia"]
    # map states to each clustered bus

    # if snakemake.wildcards.interconnect == "usa":
    states_2_map = [x for x, y in STATES_INTERCONNECT_MAPPER.items() if y in ("western", "eastern", "texas")]
    # else:
    #     states_2_map = [x for x, y in STATES_INTERCONNECT_MAPPER.items() if y == snakemake.wildcards.interconnect]

    assign_bus_2_state(n, snakemake.input.county, states_2_map, CODE_2_STATE)

    sns = get_snapshots(snakemake.params.snapshots)

    costs = load_costs(snakemake.input.costs)
    addi_costs = load_costs(snakemake.input.addi_costs)

    # biomass potential
    agri_potential = snakemake.input.agri
    forestry_potential = snakemake.input.forestry
    wastes_potential = snakemake.input.wastes

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

    # add sector specific emission tracking
    simplify_co2 = snakemake.params.sector["co2"]["simplify"]
    ts_cost = snakemake.params.sector["co2"]["ts_cost"]
    co2_storage_file = snakemake.input.co2_storage_file
    build_co2_tracking(
        n, simpsec=True, simplify_co2=simplify_co2, co2_storage_file=co2_storage_file, addi_costs=addi_costs
    )

    # break out loads into sector specific buses
    split_loads_by_carrier(n)

    # add primary energy carriers for each state
    center_points = StateGeometry(snakemake.input.county).state_center_points.set_index(
        "STATE",
    )

    h2_options = snakemake.params.sector.get("hydrogen", {})
    for carrier in ("gas", "h2", "biomass"):
        # if (carrier == "biomass") and (not h2_options.get("bioH2", True)):
        #     continue
        add_supply = True if carrier in ["coal"] else False  # gas/H2/biomass added in build_ng/h2/biomass()
        api = eia_api if carrier in ["coal"] else None  # gas/H2 cost endogenously defined
        add_sector_foundation(n, carrier, add_supply, costs, center_points, api)

    for carrier in ("OCGT", "CCGT", "CCGT-95CCS", "CCGT-97CCS"):
        convert_generators_2_links(n, carrier, " gas", costs, simplify_co2, ts_cost)

    # for carrier in ("coal", "coal-95CCS", "coal-99CCS"):
    #     co2_intensity = get_pwr_co2_intensity(carrier, costs)
    #     convert_generators_2_links(n, carrier, " coal", co2_intensity)
    n.mremove("Generator", n.generators.index[n.generators.carrier == "coal"])
    # for carrier in ["oil"]:
    #     co2_intensity = get_pwr_co2_intensity(carrier, costs)
    #     convert_generators_2_links(n, carrier, " oil", co2_intensity)

    for carrier in ("biomass", "biomass-CCS"):
        convert_generators_2_links(n, carrier, " biomass", costs)

    ng_options = snakemake.params.sector["natural_gas"]
    years = ng_options["years"]

    # add natural gas infrastructure and data
    build_natural_gas(
        n=n,
        years=years,
        api=eia_api,
        interconnect="usa",  # snakemake.wildcards.interconnect
        county_path=snakemake.input.county,
        pipelines_path=snakemake.input.pipeline_capacity,
        pipeline_shape_path=snakemake.input.pipeline_shape,
        storage_plant_path=snakemake.input.get("gas_storage_plants", None),
        options=ng_options,
        addi_costs=addi_costs,
    )

    # add hydrogen infrastructure and data
    build_hydrogen(
        n=n,
        h2_options=h2_options,
        costs=costs,
        addi_costs=addi_costs,
        simplify_co2=simplify_co2,
        ts_cost=ts_cost,
    )

    reference_network_path = getattr(snakemake.input, "reference_network", None)
    # Handle case where reference_network might be empty list
    if isinstance(reference_network_path, list) and len(reference_network_path) == 0:
        reference_network_path = None
    build_biomass(  # add biomass production potential
        n=n,
        agri_potential=agri_potential,
        forestry_potential=forestry_potential,
        wastes_potential=wastes_potential,
        reference_network_path=reference_network_path,
    )

    build_dac(
        n,
        addi_costs=addi_costs,
        cost_multiplier=snakemake.params.sector["dac"]["cost_multiplier"],
        simplify_co2=simplify_co2,
        ts_cost=ts_cost,
    )

    # build_methanation(n, addi_costs=addi_costs, simplify_co2=simplify_co2, ts_cost=ts_cost)

    if not simplify_co2:
        build_co2_sequestration(n, addi_costs=addi_costs)
        build_co2_pipeline(n, addi_costs=addi_costs)

    ldes_options = snakemake.params.sector.get("ldes", {})
    if ldes_options["enable"]:
        build_ldes(n, addi_costs=addi_costs, ldes_options=ldes_options)

    only_power_options = snakemake.params.sector.get("only_power", {})
    if only_power_options["enable"]:
        # Set gas loads to zero
        gas_loads = n.loads[n.loads.carrier == "gas"].index
        n.loads_t.p_set[gas_loads] = 0

        # Set H2 loads to zero
        h2_loads = n.loads[n.loads.carrier == "h2"].index
        n.loads_t.p_set[h2_loads] = 0

        # # Set biomass loads to zero
        # biomass_loads = n.loads[n.loads.carrier == 'biomass'].index
        # n.loads_t.p_set[biomass_loads] = 0

        # Remove smr, bio-h2, methanation
        smr_links = n.links[n.links.carrier.str.contains("smr", case=False, na=False)].index
        h2_bio_links = n.links[n.links.carrier.str.contains("h2 bio", case=False, na=False)].index
        methanation_links = n.links[n.links.carrier.str.contains("methanation", case=False, na=False)].index
        links_to_remove = smr_links.union(h2_bio_links).union(methanation_links)
        n.mremove("Link", links_to_remove)

    logger.info("Applying seasonal capacity derates to new generators and links")
    # Load derates from CSV resource files
    state_derates, national_derates = load_derates_from_csv(
        snakemake.input.state_derates,
        snakemake.input.national_derates,
    )
    apply_derates_to_new_components(n, n.snapshots, state_derates, national_derates)

    # Needed as loads may be split off to urban/rural
    sanitize_carriers(n, snakemake.config)

    n.export_to_netcdf(snakemake.output.network)
