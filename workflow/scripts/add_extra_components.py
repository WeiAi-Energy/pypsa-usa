"""Adds extra extendable components to the clustered and simplified network."""

import functools
import logging
from collections.abc import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from _helpers import calculate_annuity, configure_logging, load_costs, log_network_schema
from add_electricity import add_missing_carriers, broadcast_investment_horizons_index
from ambient_derate import (
    aggregate_bus_temperature,
    ambient_derate_settings,
    apply_ambient_derate,
    load_bus_temperature,
    load_outage_curves,
)
from constants import (
    BECCS_CO2_REMOVAL,
    GENERATOR_CAPEX_STRATEGIES,
    HOURS_PER_YEAR,
    REGIONAL_MULTIPLIER_COL,
    RENEWABLE_INTERCONNECTION_COL,
    RENEWABLE_INTERCONNECTION_SHARE_COL,
)
from external_regions import (
    add_external_regions,
    convert_flowgates_to_state,
    format_flowgates_for_imports_exports,
    load_remote_unit_bundle,
    resolve_trade_costs,
)
from regional_cost import (
    build_bus_multipliers,
    regional_capital_cost,
    renewable_interconnection_capital_cost,
)
from shapely.geometry import Point
from tax_credits import check_tax_credit_years

idx = pd.IndexSlice

logger = logging.getLogger(__name__)

# carriers of the plants paying the fuel costs of the cost data: the AEO fuel costs
# of gas, coal and oil plants, the ATB fuel costs of nuclear and biomass plants
POWER_FUEL_CARRIERS = [
    "OCGT",
    "CCGT",
    "CCGT-95CCS",
    "CCGT-97CCS",
    "coal",
    "coal-95CCS",
    "coal-99CCS",
    "oil",
    "nuclear",
    "SMR",
    "biomass",
    "biomass-95CCS",
]


def add_co2_emissions(n, costs, carriers):
    """Add CO2 emissions to the network's carriers attribute."""
    suptechs = n.carriers.loc[carriers].index.str.split("-").str[0]
    missing_carriers = set(suptechs) - set(costs.index)
    if missing_carriers:
        logger.warning(
            f"CO2 emissions for carriers {missing_carriers} not defined in cost data.",
        )
        suptechs = suptechs.difference(missing_carriers)
    n.carriers.loc[suptechs, "co2_emissions"] = costs.co2_emissions[suptechs].values

    n.carriers = n.carriers.fillna(
        {"co2_emissions": 0},
    )  # TODO: FIX THIS ISSUE IN BUILD_COST_DATA- missing co2_emissions for some VRE carriers

    if any("CCS" in carrier for carrier in carriers):
        ccs_carriers = [carrier for carrier in carriers if "CCS" in carrier]
        for ccs_carrier in ccs_carriers:
            if ccs_carrier.startswith("biomass"):  # biomass has no base emissions: the removal of the cost data
                n.carriers.loc[ccs_carrier, "co2_emissions"] = costs.at[ccs_carrier, "co2_emissions"]
                continue
            base_carrier = ccs_carrier.split("-")[0]
            base_emissions = n.carriers.loc[base_carrier, "co2_emissions"]
            ccs_level = int(ccs_carrier.split("-")[1].replace("CCS", ""))
            ccs_emissions = (1 - ccs_level / 100) * base_emissions
            n.carriers.loc[ccs_carrier, "co2_emissions"] = ccs_emissions


def add_nice_carrier_names(n, config):
    carrier_i = n.carriers.index
    nice_names = (
        pd.Series(config["plotting"]["nice_names"]).reindex(carrier_i).fillna(carrier_i.to_series().str.title())
    )
    n.carriers["nice_name"] = nice_names
    colors = pd.Series(config["plotting"]["tech_colors"]).reindex(carrier_i)
    if colors.isna().any():
        missing_i = list(colors.index[colors.isna()])
        logger.warning(f"tech_colors for carriers {missing_i} not defined in config.")
    n.carriers["color"] = colors


def regional_capital_costs(
    costs: pd.DataFrame,
    carriers: pd.Series,
    buses: pd.Series,
    bus_multipliers,
    own_multipliers: pd.Series | None = None,
) -> pd.Series:
    """Per-component ``capital_cost`` with the ReEDS regional overnight multiplier.

    ``carriers`` and ``buses`` share one index (the components). Carriers the
    multiplier table does not map keep the uniform ``annualized_capex_fom``.

    ``own_multipliers`` is the :data:`REGIONAL_MULTIPLIER_COL` column of the
    components, where they have one. ``add_electricity`` sets it and clustering
    averages it over the merged generators, so it is the multiplier their
    ``capital_cost`` holds; where set it is used in place of the county of the bus.
    """
    capex = carriers.map(costs["annualized_capex_fom"]).astype(float)
    if bus_multipliers is None:
        return capex
    for carrier in set(carriers.unique()) & set(bus_multipliers.columns) & set(costs.index):
        members = carriers.index[carriers == carrier]
        capex.loc[members] = regional_capital_cost(
            costs,
            carrier,
            buses.loc[members],
            bus_multipliers,
            own_multiplier=None if own_multipliers is None else own_multipliers.loc[members],
        )
    return capex


def copy_capex_columns(n: pypsa.Network, source: pd.DataFrame, targets: pd.Index) -> None:
    """Carry the columns regional capital costs are rebuilt from over to new units.

    ``n.add`` does not take them, and the units added for the expansion of a
    generator price every later vintage from them.
    """
    for column in GENERATOR_CAPEX_STRATEGIES:
        if column in source.columns:
            n.generators.loc[targets, column] = source[column].to_numpy()


def apply_renewable_interconnection_costs(capex: pd.Series, costs: pd.DataFrame, gens: pd.DataFrame) -> pd.Series:
    """Reprice wind/solar that carry a ReEDS interconnection cost with this year's costs.

    ``add_electricity`` stores each generator's interconnection cost (overnight
    USD/MW) in :data:`RENEWABLE_INTERCONNECTION_COL`. Each vintage swaps it for the
    ATB grid connection of its own investment year, at that year's capital charge rate.
    Generators without the column keep ``capex``.

    A generator that clustering merged from members with and without a ReEDS site
    swaps only the share :data:`RENEWABLE_INTERCONNECTION_SHARE_COL` gives; the
    rest of its capacity keeps the ATB grid connection.
    """
    if RENEWABLE_INTERCONNECTION_COL not in gens.columns:
        return capex
    interconnection = pd.to_numeric(gens[RENEWABLE_INTERCONNECTION_COL], errors="coerce")
    share = (
        pd.to_numeric(gens[RENEWABLE_INTERCONNECTION_SHARE_COL], errors="coerce")
        if RENEWABLE_INTERCONNECTION_SHARE_COL in gens.columns
        else pd.Series(np.nan, index=gens.index)
    )
    capex = capex.copy()
    for carrier in set(gens.carrier[interconnection > 0]) & set(costs.index):
        members = gens.index[(gens.carrier == carrier) & (interconnection > 0)]
        capex.loc[members] = renewable_interconnection_capital_cost(
            costs,
            carrier,
            interconnection.loc[members],
            share=share.loc[members],
        )
    return capex


def attach_storageunits(n, costs, elec_opts, investment_year, bus_multipliers=None):
    carriers = elec_opts["extendable_carriers"]["StorageUnit"]
    carriers = [k for k in carriers if "battery_storage" in k]

    buses_i = n.buses.index

    add_missing_carriers(n, carriers)
    add_co2_emissions(n, costs, carriers)
    for carrier in carriers:
        max_hours = int(carrier.split("hr_")[0])
        roundtrip_correction = 0.5 if "battery" in carrier else 1

        n.add(
            "StorageUnit",
            buses_i,
            suffix=f" {carrier}_{investment_year}",
            bus=buses_i,
            carrier=carrier,
            p_nom_extendable=True,
            capital_cost=regional_capital_cost(costs, carrier, buses_i, bus_multipliers),
            marginal_cost=0,  # costs.at[carrier, "marginal_cost"], # TODO: FIX THIS ISSUE IN BUILD_COST_DATA
            efficiency_store=costs.at[carrier, "efficiency"] ** roundtrip_correction,
            efficiency_dispatch=costs.at[carrier, "efficiency"] ** roundtrip_correction,
            max_hours=max_hours / (costs.at[carrier, "efficiency"] ** roundtrip_correction),
            cyclic_state_of_charge=True,
            cyclic_state_of_charge_per_period=True,  # pypsa v1 flipped this default to False
            build_year=investment_year,
            lifetime=costs.at[carrier, "lifetime"],
        )


def attach_phs_storageunits(n: pypsa.Network, elec_opts, costs: pd.DataFrame):
    carriers = elec_opts["extendable_carriers"]["StorageUnit"]
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

        # Round CAPEX to $500 interval
        psh_resources["cost_kw_round"] = (psh_resources["cost_kw"] / 500).round() * 500

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
            / HOURS_PER_YEAR
        )

        region_onshore_psh_grp["marginal_cost"] = psh_vom

        # Set RT efficiency = 0.8
        efficiency_store = 0.894427191  # 0.894427191^2 = 0.8
        efficiency_dispatch = 0.894427191  # 0.894427191^2 = 0.8

        costs.at["PHS", "efficiency"] = efficiency_store
        costs.at["PHS", "co2_emissions"] = 0
        add_missing_carriers(n, ["PHS"])
        add_co2_emissions(n, costs, ["PHS"])
        n.add(
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
            max_hours=max_hours / efficiency_dispatch,
            cyclic_state_of_charge=True,
            cyclic_state_of_charge_per_period=True,  # pypsa v1 flipped this default to False
        )


def copy_timeseries_for_suffix(
    n: pypsa.Network,
    source_index: pd.Index,
    suffix: str | int,
    attrs: tuple[str, ...] = ("marginal_cost", "p_max_pu"),
) -> None:
    """
    Copy time dependent generator attributes onto suffixed duplicates.

    Only generators that actually carry a time series for ``attr`` are copied,
    since not all generators are time dependent.
    """
    for attr in attrs:
        source = n.generators_t[attr]
        columns = [x for x in source_index if x in source.columns]
        renamed = source[columns].rename(
            columns={x: f"{x} {suffix}" for x in columns},
        )
        n.generators_t[attr] = source.join(renamed)


def split_retirement_gens(
    n: pypsa.Network,
    costs: pd.DataFrame,
    carriers: Iterable[str] | None = None,
    economic: bool = True,
    keep_sites: bool = False,
) -> pd.Index:
    """
    Separate the existing capacity of extendable generators so it can retire.

    Every extendable generator of ``carriers`` (all carriers when ``None``) that
    holds existing capacity (``p_nom > 0``) is renamed ``"<name> existing"``,
    together with its time series:

    1. its capital cost is replaced by the fixed O&M cost,
    2. ``p_nom_max`` is set to ``p_nom`` and ``p_nom_min`` to zero,
    3. it stays extendable when ``economic`` (economic retirement), else it is
       fixed and retires on its lifetime.

    New capacity is never priced from these units: the new builds of each
    investment period take that period's cost table (:func:`attach_new_thermal`,
    :func:`attach_renewable_vintages`). With ``keep_sites`` (renewables) the
    original name is re-added as an empty resource site that keeps only what
    describes the resource: bus, developable potential, land region,
    capacity-factor profile and the columns regional capital costs are rebuilt
    from. :func:`attach_renewable_vintages` adds the vintages from it. Without it
    (thermal carriers) nothing is re-added.

    Returns the original names of the split generators.
    """
    retirement_mask = n.generators["p_nom_extendable"] & (n.generators["p_nom"] > 0)
    if carriers is not None:
        retirement_mask &= n.generators["carrier"].isin(list(carriers))
    retirement_gens = n.generators[retirement_mask].copy()
    if retirement_gens.empty:
        return retirement_gens.index

    # existing capacity pays only its fixed O&M, and cannot grow
    n.generators.loc[retirement_mask, "capital_cost"] = (
        costs.loc[retirement_gens.carrier, "opex_fixed_per_kw"].to_numpy() * 1e3
    )
    n.generators.loc[retirement_mask, "p_nom_max"] = retirement_gens.p_nom.to_numpy()
    n.generators.loc[retirement_mask, "p_nom_min"] = 0
    n.generators.loc[retirement_mask, "p_nom_extendable"] = economic

    existing = retirement_gens.index + " existing"
    rename_components(n, "Generator", dict(zip(retirement_gens.index, existing)))

    if not keep_sites:
        return retirement_gens.index

    site_attrs = {"land_region": retirement_gens.land_region} if "land_region" in retirement_gens else {}
    n.add(
        "Generator",
        retirement_gens.index,
        carrier=retirement_gens.carrier,
        bus=retirement_gens.bus,
        p_nom=0,
        p_nom_min=0,
        p_nom_max=retirement_gens.p_nom_max,
        p_nom_extendable=True,
        p_max_pu=retirement_gens.p_max_pu,
        build_year=n.investment_periods[0],
        **site_attrs,
    )
    copy_capex_columns(n, retirement_gens, retirement_gens.index)
    profiles = n.generators_t.p_max_pu
    with_profile = existing[existing.isin(profiles.columns)]
    n.generators_t.p_max_pu = profiles.join(
        profiles[with_profile].rename(columns=lambda name: name.removesuffix(" existing")),
    )
    return retirement_gens.index


def attach_renewable_vintages(
    n: pypsa.Network,
    costs: pd.DataFrame,
    sites: pd.DataFrame,
    investment_year: int,
    bus_multipliers=None,
) -> pd.Index:
    """
    Add the renewable new builds of one investment period at every resource site.

    A site supplies only what describes the resource: its bus, developable
    potential (``p_nom_max``, which the land-use constraint shares among all
    vintages of a land region), land region, capacity-factor profile, and the
    columns the regional capital cost is rebuilt from (the ReEDS multiplier of
    the counties of the site and its interconnection cost). Every technology
    parameter comes from ``costs`` of ``investment_year``: capital cost, marginal
    cost, VOM and lifetime. Vintages are named ``"<site> <year>"``.
    """
    if sites.empty:
        return pd.Index([])

    capital_cost = apply_renewable_interconnection_costs(
        regional_capital_costs(costs, sites.carrier, sites.bus, bus_multipliers, sites.get(REGIONAL_MULTIPLIER_COL)),
        costs,
        sites,
    )
    site_attrs = {"land_region": sites.land_region} if "land_region" in sites else {}
    n.add(
        "Generator",
        sites.index,
        suffix=f" {investment_year}",
        carrier=sites.carrier,
        bus=sites.bus,
        p_nom=0,
        p_nom_min=0,
        p_nom_max=sites.p_nom_max,
        p_nom_extendable=True,
        p_max_pu=sites.p_max_pu,
        efficiency=1,
        marginal_cost=sites.carrier.map(costs.marginal_cost).to_numpy(),
        capital_cost=capital_cost.to_numpy(),
        build_year=investment_year,
        lifetime=sites.carrier.map(costs.lifetime).to_numpy(),
        **site_attrs,
    )
    names = sites.index + f" {investment_year}"
    n.generators.loc[names, "vom_cost"] = sites.carrier.map(costs.opex_variable_per_mwh).to_numpy()
    copy_capex_columns(n, sites, names)
    copy_timeseries_for_suffix(n, sites.index, investment_year, attrs=("p_max_pu",))
    return names


def split_existing_fleet(
    n: pypsa.Network,
    elec_config: dict,
    costs: pd.DataFrame,
    economic: bool,
    profile_carriers: Iterable[str] = (),
) -> tuple[list[str], pd.DataFrame]:
    """
    Separate the existing fleet from the new builds of the extendable generator carriers.

    New builds take every technology parameter from the cost table of their
    investment period; the existing fleet only supplies the capacity that retires.
    Renewables (``electricity: renewable_carriers``, and every carrier of
    ``profile_carriers``, the technologies with a capacity-factor profile under
    ``renewable:``) are built at their resource sites only, every other extendable
    carrier but EGS at every bus. A renewable carrier without sites is not built,
    rather than added at every bus without a profile.

    1. The existing capacity of extendable generators is split off into
       ``"<name> existing"`` (:func:`split_retirement_gens`), retiring
       economically or on its lifetime for thermal carriers, on its lifetime for
       renewables.
    2. Extendable thermal generators without existing capacity are removed, so
       :func:`attach_new_thermal` adds the only new-build option per bus.
    3. The extendable renewable generators left are the resource sites
       :func:`attach_renewable_vintages` builds from.

    Returns the thermal carriers to build and the renewable sites.
    """
    extendable = list(elec_config["extendable_carriers"].get("Generator", []))
    renewable_carriers = set(elec_config.get("renewable_carriers") or []) | set(profile_carriers)
    renewable_new = [car for car in extendable if car in renewable_carriers]
    thermal_new = [car for car in extendable if car not in renewable_carriers and "EGS" not in car]

    split_retirement_gens(
        n,
        costs,
        set(elec_config.get("conventional_carriers") or []) | set(thermal_new),
        economic=economic,
    )
    split_retirement_gens(n, costs, renewable_carriers, economic=False, keep_sites=True)

    not_existing = ~n.generators.index.str.contains("existing")
    fleet_candidates = n.generators.index[
        n.generators.p_nom_extendable & n.generators.carrier.isin(thermal_new) & not_existing
    ]
    if not fleet_candidates.empty:
        logger.info(f"Removing {len(fleet_candidates)} extendable fleet generators without existing capacity.")
        n.remove("Generator", fleet_candidates)
        not_existing = ~n.generators.index.str.contains("existing")

    renewable_sites = n.generators[
        n.generators.p_nom_extendable & n.generators.carrier.isin(renewable_new) & not_existing
    ].copy()
    no_sites = sorted(set(renewable_new) - set(renewable_sites.carrier))
    if no_sites:
        logger.warning(f"No resource sites for the extendable renewable carriers {no_sites}; none are built.")
    return thermal_new, renewable_sites


def attach_multihorizon_egs(
    n: pypsa.Network,
    costs: pd.DataFrame,
    costs_dict: dict,
    gens: pd.DataFrame,
    investment_year: int,
):
    """
    Adds multiple investment options for EGS.
    Arguments:
    n: pypsa.Network,
    costs: pd.DataFrame,
        dataframe with costs of investment year
    costs_dict: dict,
        Dict of costs for each investment period
    carriers: List[str]
        List of carriers to add multiple investment options for.
    """
    if gens.empty or len(n.investment_periods) == 1:
        return

    lifetime = 25  # Following EGS supply curves by Aljubran et al. (2024)
    base_year = n.investment_periods[0]
    learning_ratio = costs.loc["EGS", "capex_per_kw"] / costs_dict[base_year].loc["EGS", "capex_per_kw"]
    capital_cost = learning_ratio * gens["capital_cost"]
    n.add(
        "Generator",
        gens.index,
        suffix=f" {investment_year}",
        carrier=gens.carrier,
        bus=gens.bus,
        p_nom_min=0,
        p_nom=0,
        p_nom_max=gens.p_nom_max,
        p_nom_extendable=True,
        ramp_limit_up=gens.ramp_limit_up,
        ramp_limit_down=gens.ramp_limit_down,
        efficiency=gens.efficiency,
        marginal_cost=gens.marginal_cost,
        p_min_pu=gens.p_min_pu,
        p_max_pu=gens.p_max_pu,
        capital_cost=capital_cost,
        build_year=investment_year,
        lifetime=lifetime,
    )

    # time dependent factors added after
    copy_timeseries_for_suffix(n, gens.index, investment_year)

    # shift over time to capture decline
    investment_year_idx = np.where(n.investment_periods == investment_year)[0][0]
    cars = list(
        n.generators_t["p_max_pu"].filter(like="EGS").filter(like=str(investment_year)).columns,
    )
    n.generators_t["p_max_pu"].loc[n.investment_periods[investment_year_idx:], cars] = (
        n.generators_t["p_max_pu"]
        .loc[
            n.investment_periods[: len(n.investment_periods) - investment_year_idx],
            cars,
        ]
        .values
    )


def attach_new_thermal(
    n: pypsa.Network,
    costs: pd.DataFrame,
    carriers: Iterable[str],
    investment_year: int,
    bus_multipliers=None,
    min_years: dict | None = None,
) -> pd.Index:
    """
    Add the dispatchable new builds of one investment period at every bus.

    One extendable candidate per bus and carrier, named ``"<bus> <carrier>_<year>"``,
    whether or not the carrier has plants in the network. Every technology parameter
    comes from ``costs`` of ``investment_year``: capital cost, efficiency, VOM,
    marginal cost and lifetime. The location dependent quantities are the ReEDS
    regional multiplier of the bus and the ambient-temperature derate, which
    :func:`apply_ambient_derate_new_generators` sets from the temperature of the bus.
    Ramp limits, minimum stable level and unit commitment keep the pypsa defaults,
    as the cost table has none. Carriers whose ``min_years`` entry is later than
    ``investment_year`` are skipped.
    """
    carriers = list(carriers)
    if not carriers:
        return pd.Index([])

    add_missing_carriers(n, carriers)
    add_co2_emissions(n, costs, carriers)
    min_years = min_years or {}
    buses = n.buses.index
    added = []
    for carrier in carriers:
        if min_years.get(carrier, 0) > investment_year:
            continue
        names = buses + f" {carrier}_{investment_year}"
        n.add(
            "Generator",
            names,
            bus=buses,
            carrier=carrier,
            p_nom_extendable=True,
            capital_cost=regional_capital_cost(costs, carrier, buses, bus_multipliers),
            marginal_cost=costs.at[carrier, "marginal_cost"],
            efficiency=costs.at[carrier, "efficiency"],
            build_year=investment_year,
            lifetime=costs.at[carrier, "lifetime"],
        )
        n.generators.loc[names, "vom_cost"] = costs.at[carrier, "opex_variable_per_mwh"]
        added.append(names)
    return pd.Index([]).append(added) if added else pd.Index([])


def apply_power_fuel_prices(n: pypsa.Network, costs_dict: dict[int, pd.DataFrame]) -> None:
    """
    Prices the fuel of plants, existing and new, at the fuel costs of each investment period.

    A plant's marginal cost in a period becomes its VOM plus the fuel cost of its
    carrier in that period's cost data (``fuel_cost_real_per_mwhth``, see
    build_cost_data) over its efficiency: the AEO fuel cost of gas, coal and oil
    plants, the ATB fuel cost of nuclear and biomass plants. This replaces the
    plant level fuel costs of the existing fleet.

    The cost only changes between investment periods, and only the periods a plant
    is active in (``get_active_assets``) enter the optimization. A plant whose cost
    differs between its active periods gets a ``marginal_cost`` time series; every
    other plant keeps none, its static ``marginal_cost`` being the cost of its first
    active period (of the first period if it is never active). With a single
    investment period no plant needs a series. The full snapshot x plant table,
    several GB at county resolution, is never built.
    """
    plants = n.generators[n.generators.carrier.isin(POWER_FUEL_CARRIERS)]
    if plants.empty:
        return

    vom = plants.get("vom_cost", pd.Series(np.nan, index=plants.index))
    vom = vom.fillna(plants.carrier.map(costs_dict[n.investment_periods[0]].opex_variable_per_mwh)).fillna(0)

    # USD/MWh_th of each carrier's fuel in each period -> USD/MWh_el of each plant in each period
    periods = n.investment_periods
    fuel_cost = pd.DataFrame({period: costs.fuel_cost_real_per_mwhth for period, costs in costs_dict.items()}).T
    cost = fuel_cost.loc[periods, plants.carrier].to_numpy() / plants.efficiency.to_numpy() + vom.to_numpy()

    active = np.vstack([n.get_active_assets("Generator", p).reindex(plants.index).to_numpy(bool) for p in periods])
    columns = np.arange(len(plants))
    static = cost[np.where(active.any(axis=0), active.argmax(axis=0), 0), columns]
    lowest = np.where(active, cost, np.inf).min(axis=0)
    highest = np.where(active, cost, -np.inf).max(axis=0)
    varies = active.any(axis=0) & ~(lowest == highest)  # a nan cost keeps its series too

    period_of_snapshot = periods.get_indexer(n.snapshots.get_level_values("period"))
    series = pd.DataFrame(
        cost[np.ix_(period_of_snapshot, np.flatnonzero(varies))],
        index=n.snapshots,
        columns=plants.index[varies],
    )
    others = n.generators_t.marginal_cost.drop(columns=plants.index, errors="ignore")
    n.generators_t.marginal_cost = series if others.columns.empty else pd.concat([others, series], axis=1)
    n.generators.loc[plants.index, "marginal_cost"] = static
    logger.info(
        f"Priced the fuel of {len(plants)} plants at the fuel costs of the cost data; "
        f"{varies.sum()} whose cost changes between their active periods get a time series",
    )


def apply_ptc(n: pypsa.Network, costs_dict: dict[int, pd.DataFrame]) -> None:
    """
    Take the levelized production tax credit off the marginal cost of new builds.

    Each vintage claims the credit of its build year, ``ptc_per_mwh`` of that year's
    costs: the ATB value of the selected case or the ``costs: ptc_modifier`` override
    (``tax_credits.resolve_tax_credits``), zero where none is granted. The
    credit is levelized over the cost recovery period like the capex it offsets, so
    it applies in every period the vintage operates (a vintage does not operate
    before its build year). Generators that are not extendable (the existing fleet)
    get none.
    """
    for build_year, costs in costs_dict.items():
        if "ptc_per_mwh" not in costs.columns:
            continue
        ptc = costs["ptc_per_mwh"][costs["ptc_per_mwh"] > 0]
        mask = (
            n.generators.p_nom_extendable
            & (n.generators.build_year == build_year)
            & n.generators.carrier.isin(ptc.index)
        )
        if not mask.any():
            continue
        credit = n.generators.loc[mask, "carrier"].map(ptc)
        n.generators.loc[credit.index, "marginal_cost"] -= credit

        marginal_cost = n.generators_t.marginal_cost
        if not credit.index.intersection(marginal_cost.columns).empty:
            # one new frame (x - 0 = x elsewhere) instead of a copy plus a column assignment
            n.generators_t.marginal_cost = marginal_cost.sub(credit.reindex(marginal_cost.columns, fill_value=0.0))

        claimed = ptc[ptc.index.isin(n.generators.loc[mask, "carrier"])]
        logger.info(
            f"PTC of {build_year} builds: " + ", ".join(f"{c} {v:.2f} USD/MWh" for c, v in claimed.items()),
        )


def apply_max_annual_growth_rate(n, max_growth):
    """
    Applies maximum annual growth rate to components specified in the
    configuration file.

    Arguments:
    n: pypsa.Network,
    max_growth: dict,
        Dict of maximum annual growth rate and base for each carrier.
        Format: #{carrier_name: {base: , rate: }}
    """
    if max_growth is None or len(n.investment_periods) <= 1:
        return

    years = n.investment_period_weightings.index.to_series().diff().dropna().mean()

    for carrier, growth_params in max_growth.items():
        base = growth_params.get("base", None)
        rate = growth_params.get("rate", None)

        if base is None and rate is None:
            continue

        p_nom = n.generators.p_nom.loc[n.generators.carrier == carrier].sum()
        n.carriers.loc[carrier, "max_growth"] = base or p_nom
        n.carriers.loc[carrier, "max_relative_growth"] = rate**years


def add_demand_response(
    n: pypsa.Network,
    dr_config: dict[str, str | float],
) -> None:
    """Add price based demand response to network."""
    n.add("Carrier", "demand_response", color="#dd2e23", nice_name="Demand Response")

    shift = dr_config.get("shift", 0)
    if shift == 0:
        logger.info(f"DR not applied as allowable shift is {shift}")
        return

    marginal_cost_storage = dr_config.get("marginal_cost", 0)
    if marginal_cost_storage == 0:
        logger.warning("No cost applied to demand response")

    # attach dr at all load locations

    buses = n.loads.bus
    df = n.buses[n.buses.index.isin(buses)].copy()

    # two storageunits for forward and backwards load shifting

    n.add(
        "Bus",
        name=df.index,
        suffix="-fwd-dr",
        x=df.x,
        y=df.y,
        carrier="demand_response",
        unit="MWh",
        country=df.country,
        reeds_zone=df.reeds_zone,
        reeds_ba=df.reeds_ba,
        interconnect=df.interconnect,
        trans_reg=df.trans_reg,
        trans_grp=df.trans_grp,
        reeds_state=df.reeds_state,
        substation_lv=df.substation_lv,
    )

    n.add(
        "Bus",
        name=df.index,
        suffix="-bck-dr",
        x=df.x,
        y=df.y,
        carrier="demand_response",
        unit="MWh",
        country=df.country,
        reeds_zone=df.reeds_zone,
        reeds_ba=df.reeds_ba,
        interconnect=df.interconnect,
        trans_reg=df.trans_reg,
        trans_grp=df.trans_grp,
        reeds_state=df.reeds_state,
        substation_lv=df.substation_lv,
    )

    # seperate charging/discharging links for easier constraint generation

    n.add(
        "Link",
        name=df.index,
        suffix="-fwd-dr-charger",
        bus0=df.index,
        bus1=df.index + "-fwd-dr",
        carrier="demand_response",
        p_nom_extendable=False,
        p_nom=np.inf,
    )

    n.add(
        "Link",
        name=df.index,
        suffix="-fwd-dr-discharger",
        bus0=df.index + "-fwd-dr",
        bus1=df.index,
        carrier="demand_response",
        p_nom_extendable=False,
        p_nom=np.inf,
    )

    n.add(
        "Link",
        name=df.index,
        suffix="-bck-dr-charger",
        bus0=df.index,
        bus1=df.index + "-bck-dr",
        carrier="demand_response",
        p_nom_extendable=False,
        p_nom=np.inf,
    )

    n.add(
        "Link",
        name=df.index,
        suffix="-bck-dr-discharger",
        bus0=df.index + "-bck-dr",
        bus1=df.index,
        carrier="demand_response",
        p_nom_extendable=False,
        p_nom=np.inf,
    )

    # backward stores have positive marginal cost storage and postive e
    # forward stores have negative marginal cost storage and negative e

    n.add(
        "Store",
        name=df.index,
        suffix="-bck-dr",
        bus=df.index + "-bck-dr",
        e_cyclic=True,
        e_cyclic_per_period=True,  # pypsa v1 flipped this default to False
        e_nom_extendable=False,
        e_nom=1e9,
        e_min_pu=0,
        e_max_pu=1,
        carrier="demand_response",
        marginal_cost_storage=marginal_cost_storage,
    )

    n.add(
        "Store",
        name=df.index,
        suffix="-fwd-dr",
        bus=df.index + "-fwd-dr",
        e_cyclic=True,
        e_cyclic_per_period=True,  # pypsa v1 flipped this default to False
        e_nom_extendable=False,
        e_nom=1e9,
        e_min_pu=-1,
        e_max_pu=0,
        carrier="demand_response",
        marginal_cost_storage=marginal_cost_storage * (-1),
    )


@functools.cache
def pypsa_renames_non_bus_components() -> bool:
    """
    Whether pypsa's rename_component_names can rename components other than buses.

    Up to and including pypsa 1.4.0, renaming e.g. a link renames it and its time series, then raises a KeyError
    while updating the references to it, because it looks for "link0", "link1", ... columns in every other component
    (PyPSA/PyPSA#2057). PyPSA/PyPSA#2058 fixed this after 1.4.0, but no release includes the fix yet, so rather than
    guess the version it will ship in, this tries a rename on a tiny network.
    """
    n = pypsa.Network()
    n.add("Bus", ["bus0", "bus1"])
    n.add("Link", "link", bus0="bus0", bus1="bus1")
    try:
        n.rename_component_names("Link", link="renamed")
    except KeyError:
        return False
    return n.links.index.tolist() == ["renamed"]


def rename_components(n: pypsa.Network, component: str, names: dict[str, str]) -> None:
    """
    Renames components, along with their time series and the references to them in other components.

    Uses pypsa's rename_component_names wherever it works: always for buses, and for every other component once pypsa
    includes PyPSA/PyPSA#2058 (see pypsa_renames_non_bus_components). With older pypsa versions, this renames the
    component and its time series itself, which leaves no stale references for the generators and links renamed in
    this module, since nothing refers to them by name.

    TODO: once the pinned pypsa includes PyPSA/PyPSA#2058, call n.rename_component_names(component, **names) at the
    call sites and drop this helper and pypsa_renames_non_bus_components.
    """
    if component == "Bus" or pypsa_renames_non_bus_components():
        n.rename_component_names(component, **names)
        return

    c = n.components[component]
    c.static = c.static.rename(index=names)
    for attr, df in c.dynamic.items():
        c.dynamic[attr] = df.rename(columns=names)


def add_co2_storage(n: pypsa.Network, config: dict, co2_storage_csv: str, costs: pd.DataFrame, sector: bool):
    """Adds node level CO2 (underground) storage."""
    # get node level CO2 (underground) storage potential and cost from CSV file
    co2_storage = pd.read_csv(co2_storage_csv).set_index("node")

    # add carrier to represent CO2
    n.add(
        "Carrier",
        ["co2"],
        color=config["plotting"]["tech_colors"]["co2"],
        nice_name=config["plotting"]["nice_names"]["co2"],
    )

    # add buses to represent node level CO2 captured by different processes
    n.add(
        "Bus",
        co2_storage.index,
        suffix=" co2 capture",
        carrier="co2",
    )

    # add stores to represent node level CO2 (underground) storage
    n.add(
        "Store",
        co2_storage.index,
        suffix=" co2 storage",
        bus=co2_storage.index + " co2 capture",
        e_nom_extendable=True,
        e_nom_max=co2_storage["potential [MtCO2]"] * 1e6,  # in tCO2
        marginal_cost=co2_storage["cost [USD/tCO2]"],
        carrier="co2",
    )

    # get CO2 intensity for gas and coal
    gas_co2_intensity = costs.loc["gas"]["co2_emissions"]
    coal_co2_intensity = costs.loc["coal"]["co2_emissions"]

    if sector:
        links = n.links.index[n.links.index.str.contains("CCS")]
        if not links.empty:  # found links equipped with CCS
            # specify links' bus4 to point to the CO2 capture bus of their node (e.g. "p9 co2 capture")
            n.links.loc[links, "bus4"] = n.links.loc[links, "bus1"] + " co2 capture"

            # calculate efficiencies, per MWh of fuel the links draw (CO2 of the fuel burnt, emitted or captured)
            efficiency2 = []  # to node or state atmosphere bus (e.g. "p9 pwr atmosphere", "CA pwr atmosphere")
            efficiency4 = []  # to node co2 capture bus (e.g. "p9 co2 capture")
            for index in links:
                if "biomass" in index:  # BECCS: the CO2 captured is removed from the atmosphere
                    efficiency2.append(-BECCS_CO2_REMOVAL)
                    efficiency4.append(BECCS_CO2_REMOVAL)
                    continue
                if "CCGT" in index:
                    efficiency = gas_co2_intensity
                elif "coal" in index:
                    efficiency = coal_co2_intensity
                else:
                    logger.warning(
                        f"Assuming a CO2 intensity equal to 1 given that link '{index}' is not powered by gas or coal",
                    )
                    efficiency = 1
                cc_level = (
                    int(index.split("-")[1].split("CC")[0]) / 100
                )  # extract CC level from index (e.g. index "p1 CCGT-95CCS_2030" returns 0.95)
                efficiency2.append(efficiency * (1 - cc_level))
                efficiency4.append(efficiency * cc_level)

            # set links' bus2 and bus4 efficiencies
            n.links.loc[links, "efficiency2"] = efficiency2
            n.links.loc[links, "efficiency4"] = efficiency4
            # other links have no fourth output
            n.links["bus4"] = n.links.bus4.fillna("")
            n.links["efficiency4"] = n.links.efficiency4.fillna(0)

    else:  # sector-less
        generators = n.generators.index[n.generators.index.str.contains("CCS")]
        if not generators.empty:  # found generators equipped with CCS
            # the plants keep their CCS name and carrier, so that carrier-keyed policies still apply, and
            # their capital cost, which excludes CO2 transport and storage (charged by the storage store)
            indexes = generators
            nodes = n.generators.loc[indexes, "bus"]  # node level electricity buses (e.g. "p9")

            # add buses to represent node level electricity CC generator
            n.add(
                "Bus",
                indexes,
                carrier=n.generators.loc[indexes].carrier,
            )

            # add buses to represent node level emitted CO2 by different processes
            granularity = config["dac"]["granularity"]
            if granularity == "nation":
                buses_atmosphere = pd.Series("atmosphere", index=indexes)
            elif granularity == "state":
                buses = n.buses[["x", "y"]].query("x != 0 and y != 0").copy()
                buses["geometry"] = buses.apply(lambda x: Point(x.x, x.y), axis=1)
                buses_gdf = gpd.GeoDataFrame(buses, crs="EPSG:4269")
                states_gdf = gpd.GeoDataFrame(
                    gpd.read_file(snakemake.input.county_shapes).dissolve("STUSPS")["geometry"],
                )
                buses_projected = buses_gdf.to_crs("EPSG:3857")
                states_projected = states_gdf.to_crs("EPSG:3857")
                states = gpd.sjoin_nearest(buses_projected, states_projected, how="left")["STUSPS"]
                buses_atmosphere = nodes.map(states) + " atmosphere"
            else:  # node
                buses_atmosphere = nodes + " atmosphere"
            # one atmosphere bus per node, state or nation, however many CC generators vent to it
            buses_atmosphere_unique = buses_atmosphere.unique()

            # add buses to represent (air) atmosphere where CO2 emissions are sent to
            n.add(
                "Bus",
                buses_atmosphere_unique,
                carrier="co2",
            )

            # add stores to represent (air) atmosphere where CO2 emissions are stored,
            # counted per investment period (as the regional co2 limits read them)
            n.add(
                "Store",
                buses_atmosphere_unique,
                bus=buses_atmosphere_unique,
                e_nom_extendable=True,
                e_min_pu=-1,
                e_initial_per_period=True,
                carrier="co2",
            )

            # calculate efficiencies, per MWh of electricity (CO2 of the fuel burnt, emitted or captured)
            efficiency2 = []  # to node or state atmosphere bus (e.g. "p9 atmosphere", "CA atmosphere")
            efficiency3 = []  # to node co2 capture bus (e.g. "p9 co2 capture")
            for index in indexes:
                generator_efficiency = n.generators.loc[index]["efficiency"]
                if "biomass" in index:  # BECCS: the CO2 captured is removed from the atmosphere
                    efficiency2.append(-BECCS_CO2_REMOVAL / generator_efficiency)
                    efficiency3.append(BECCS_CO2_REMOVAL / generator_efficiency)
                    continue
                if "CCGT" in index:
                    efficiency = 1 / generator_efficiency * gas_co2_intensity
                elif "coal" in index:
                    efficiency = 1 / generator_efficiency * coal_co2_intensity
                else:
                    logger.warning(
                        f"Assuming a CO2 intensity equal to 1 given that generator '{index}' is not powered by gas or coal",
                    )
                    efficiency = 1 / generator_efficiency * 1
                cc_level = (
                    int(index.split("-")[1].split("CC")[0]) / 100
                )  # extract CC level from index (e.g. index "p1 CCGT-95CCS_2030" returns 0.95)
                efficiency2.append(efficiency * (1 - cc_level))
                efficiency3.append(efficiency * cc_level)

            # add links to represent sending electricity (in MW) to the electricity bus (e.g. "p9" if ReEDS or "p100 0" if TAMU) as well as sending emitted CO2 (by the generator) to both the atmosphere bus and the co2 capture bus
            n.add(
                "Link",
                indexes,
                bus0=indexes,
                bus1=nodes,
                bus2=buses_atmosphere,
                bus3=nodes + " co2 capture",
                efficiency=1,
                efficiency2=efficiency2,
                efficiency3=efficiency3,
                p_nom_extendable=True,
                capital_cost=0,
                marginal_cost=0,
                carrier=n.generators.loc[indexes].carrier,
            )

            # (re-)attach generators to new buses (that represent node level CC generator)
            n.generators.loc[indexes, "bus"] = indexes


def add_co2_network(n: pypsa.Network, config: dict):
    """Adds CO2 (transportation) network."""
    # get electricity connections
    if config["model_topology"]["transmission_network"] == "reeds":
        connections = n.links[(n.links.carrier == "AC") & ~n.links.index.str.endswith("exp")]
    else:  # TAMU
        connections = n.lines

    # calculate annualized capital cost
    number_years = n.snapshot_weightings.generators.sum() / HOURS_PER_YEAR
    cost = (
        config["co2"]["network"]["capital_cost"]
        * calculate_annuity(config["co2"]["network"]["lifetime"], config["co2"]["network"]["discount_rate"])
        * number_years
    )

    # add links to represent CO2 (transportation) network based on electricity connections layout
    n.add(
        "Link",
        connections.index,
        suffix=" co2 transport",
        bus0=connections["bus0"] + " co2 capture",
        bus1=connections["bus1"] + " co2 capture",
        efficiency=1,
        p_min_pu=-1,
        p_nom_extendable=True,
        length=connections["length"].values,
        capital_cost=cost * connections["length"].values,
        marginal_cost=config["co2"]["network"]["marginal_cost"],
        carrier="co2",
        lifetime=config["co2"]["network"]["lifetime"],
    )


def check_ucap_vs_ambient_derate(ucap_config: dict, prime_movers: dict) -> None:
    """Refuse a static UCAP forced-outage derate on carriers that already carry the temperature-dependent one."""
    if not ucap_config.get("enable", False):
        return
    both = sorted(set(ucap_config.get("forced_outage_rates") or {}) & set(prime_movers))
    if both:
        raise ValueError(
            f"ucap.forced_outage_rates and conventional.ambient_derate.prime_movers both derate {both} for forced "
            "outages; applying both double-counts them. Drop these carriers from one of the two.",
        )


def apply_ambient_derate_new_generators(
    n: pypsa.Network,
    settings: dict,
    temperature_fn: str,
    outage_curves_fn: str,
    busmap_fn: str,
) -> None:
    """
    Temperature-dependent FOR derate for thermal generators added in this rule.

    The existing fleet carries the derate it got at its ``{simpl}`` bus in
    ``add_electricity``. New builds (:func:`attach_new_thermal`) are added with a
    static ``p_max_pu``; they get ``1 - FOR(T)`` from the area-weighted mean
    temperature of their clustered bus region.
    """
    temperature, area = load_bus_temperature(temperature_fn)
    busmap = pd.read_csv(busmap_fn, index_col=0).iloc[:, 0]
    temperature = aggregate_bus_temperature(temperature, area, busmap)
    apply_ambient_derate(
        n,
        temperature,
        settings,
        load_outage_curves(outage_curves_fn),
        broadcast_investment_horizons_index,
    )


def apply_ucap(n: pypsa.Network, ucap_config: dict) -> None:
    """
    Apply UCAP (Unforced Capacity) derating to conventional generators.

    Sets p_max_pu = 1 - FOR (Forced Outage Rate) for each carrier specified
    in the configuration. This converts ICAP to UCAP using the formula:
    UCAP = ICAP * (1 - FOR)

    For generators with time-varying p_max_pu values in n.generators_t.p_max_pu,
    the time series is multiplied by (1 - FOR).

    Arguments:
        n: pypsa.Network
        ucap_config: dict with 'enable' boolean and 'forced_outage_rates' dict
            mapping carrier names to FOR values in percent
    """
    if not ucap_config.get("enable", False):
        return

    forced_outage_rates = ucap_config.get("forced_outage_rates", {})
    if not forced_outage_rates:
        logger.warning("UCAP enabled but no forced_outage_rates defined")
        return

    for carrier, for_pct in forced_outage_rates.items():
        ucap_factor = 1 - (for_pct / 100)
        mask = n.generators["carrier"] == carrier
        if not mask.any():
            logger.debug(f"No generators found for carrier {carrier} when applying UCAP")
            continue

        gen_names = n.generators.index[mask]

        # Apply to static p_max_pu
        n.generators.loc[mask, "p_max_pu"] *= ucap_factor

        # Committable units carry a minimum stable level (p_min_pu) that must stay
        # below the derated p_max_pu, otherwise the unit's dispatch box is empty and
        # the MILP is infeasible. UCAP shrinks the whole usable range of the unit, so
        # scale p_min_pu by the same factor rather than leaving it at the ICAP level.
        if "committable" in n.generators:
            com_mask = mask & n.generators.committable.fillna(False).astype(bool)
            if com_mask.any():
                n.generators.loc[com_mask, "p_min_pu"] *= ucap_factor
                com_varying = [g for g in n.generators.index[com_mask] if g in n.generators_t.p_min_pu.columns]
                if com_varying:
                    n.generators_t.p_min_pu[com_varying] *= ucap_factor

        # Apply to time-varying p_max_pu if present
        gens_with_time_varying = [g for g in gen_names if g in n.generators_t.p_max_pu.columns]
        if gens_with_time_varying:
            n.generators_t.p_max_pu[gens_with_time_varying] *= ucap_factor
            logger.info(
                f"Applied UCAP derating to {carrier}: factor = {ucap_factor:.4f} (FOR = {for_pct}%) "
                f"[{len(gen_names)} static, {len(gens_with_time_varying)} time-varying]",
            )
        else:
            logger.info(f"Applied UCAP derating to {carrier}: factor = {ucap_factor:.4f} (FOR = {for_pct}%)")


def add_dac(n: pypsa.Network, config: dict, sector: bool):
    """Adds node level DAC capabilities."""
    # generate node level buses to represent emitted, captured and accounted CO2 and links to represent DAC in function of whether network is based on sectors or not
    if sector:
        # get DAC granularity/scope
        granularity = config["dac"]["granularity"]
        if granularity == "nation":
            granularity = "node"
            logger.warning(
                "Nation level DAC capabilities is not applicable for a network based on sectors - defaulting to node level instead",
            )

        # set number of elements based on electricity transmission network type
        if config["model_topology"]["transmission_network"] == "reeds":
            elements = 1
        else:  # TAMU
            elements = 2

        # get links that emit CO2 for all sectors
        links = n.links.query("bus2.str.endswith('-co2')")

        # set buses needed to create DAC links properly afterwards
        exists_atmosphere = set()
        exists_dac = set()
        buses_atmosphere = []
        buses_atmosphere_all = []
        buses_atmosphere_unique = []
        buses_co2_capture = []
        buses_co2_account = []
        buses_ac = []
        links_dac = []
        for index in links.index:
            bus2 = links.loc[index]["bus2"]  # e.g. "CA pwr-co2"
            node = " ".join(index.split(" ")[:elements])  # e.g. "p9" if ReEDS or "p100 0" if TAMU
            state = bus2.split(" ")[0]  # e.g. "CA"
            state_sector = bus2.split(" ")[1].split("-")[0]  # e.g. "pwr"
            if granularity == "node":
                atmosphere = f"{node} {state_sector} atmosphere"
            else:  # state
                atmosphere = f"{state} {state_sector} atmosphere"
            buses_atmosphere_all.append(atmosphere)
            if atmosphere not in exists_atmosphere:
                buses_atmosphere_unique.append(atmosphere)
                buses_co2_account.append(bus2)
                exists_atmosphere.add(atmosphere)
            dac = f"{node} {state_sector} dac"
            if dac not in exists_dac:
                buses_atmosphere.append(atmosphere)
                buses_co2_capture.append(f"{node} co2 capture")
                buses_ac.append(node)
                links_dac.append(dac)
                exists_dac.add(dac)

        # add node or state level buses to represent (air) atmosphere where CO2 emissions are sent to (on a per sector basis)
        n.add(
            "Bus",
            buses_atmosphere_unique,
            carrier="co2",
        )

        # add links from node or state level buses that represent (air) atmosphere to state level buses tracking CO2 emissions (on a per sector basis)
        n.add(
            "Link",
            buses_atmosphere_unique,
            bus0=buses_atmosphere_unique,
            bus1=buses_co2_account,
            efficiency=1,
            p_nom_extendable=True,
            capital_cost=0,
            marginal_cost=0,
            carrier="co2",
        )

        # redirect links that emit CO2 to node or state level buses that represent (air) atmosphere   # e.g. "p1 trn atmosphere"
        n.links.loc[links.index, "bus2"] = buses_atmosphere_all

    else:  # sector-less
        # set buses needed to create DAC links properly afterwards: one DAC per node with a CC generator, taking
        # CO2 from the atmosphere bus the generator vents to and sending it to the node's co2 capture bus
        cc_links = n.links[n.links.bus2.str.endswith("atmosphere") & n.links.bus3.str.endswith(" co2 capture")]
        cc_links = cc_links.drop_duplicates(subset="bus1")
        if cc_links.empty:
            logger.warning("Not adding DAC capabilities given that no CC generator vents to an atmosphere bus")
            return
        buses_atmosphere = cc_links.bus2.values
        buses_co2_capture = cc_links.bus3.values
        buses_ac = cc_links.bus1.values
        links_dac = cc_links.bus1.values + " dac"

    # add carrier to represent DAC
    n.add(
        "Carrier",
        ["dac"],
        color=config["plotting"]["tech_colors"]["dac"],
        nice_name=config["plotting"]["nice_names"]["dac"],
    )

    # calculate annualized capital cost
    number_years = n.snapshot_weightings.generators.sum() / HOURS_PER_YEAR
    cost = (
        config["dac"]["capital_cost"]
        * calculate_annuity(config["dac"]["lifetime"], config["dac"]["discount_rate"])
        * number_years
    )

    # add links to represent node level DAC capabilities
    n.add(
        "Link",
        links_dac,
        bus0=buses_atmosphere,
        bus1=buses_co2_capture,
        bus2=buses_ac,
        efficiency=1,  # in tCO2
        efficiency2=-config["dac"]["electricity_input"],  # in MWh (for each tCO2)
        p_nom_extendable=True,
        capital_cost=cost,
        marginal_cost=0,
        carrier="dac",
        lifetime=config["dac"]["lifetime"],
    )


def main(snakemake) -> None:
    """Add the extra extendable components to the clustered network."""
    n = pypsa.Network(snakemake.input.network)
    schema_entry = log_network_schema(n, stage="entry")
    elec_config = snakemake.config["electricity"]

    costs_dict = {
        n.investment_periods[i]: load_costs(
            snakemake.input.tech_costs[i],
            snakemake.params.costs,
            year=n.investment_periods[i],
        )
        for i in range(len(n.investment_periods))
    }
    check_tax_credit_years(snakemake.params.costs, n.investment_periods)

    if any("PHS" in s for s in elec_config["extendable_carriers"]["StorageUnit"]):
        attach_phs_storageunits(n, elec_config, costs_dict[n.investment_periods[0]])

    thermal_new, renewable_sites = split_existing_fleet(
        n,
        elec_config,
        costs_dict[n.investment_periods[0]],
        economic=snakemake.params.retirement == "economic",
        profile_carriers=(snakemake.config.get("renewable") or {}).keys(),
    )

    egs_gens = n.generators[n.generators["p_nom_extendable"]]
    egs_gens = egs_gens.loc[egs_gens["carrier"].str.contains("EGS")]

    # ReEDS county overnight-capex multipliers for new builds (None when disabled,
    # which leaves every capital cost at its uniform national value)
    bus_multipliers = build_bus_multipliers(
        n,
        snakemake.params.costs.get("regional_multipliers"),
        snakemake.input.get("reg_cap_cost_diff"),
        snakemake.input.get("county_shapes"),
    )
    min_years = snakemake.config["costs"].get("min_year")

    for investment_year in n.investment_periods:
        costs = costs_dict[investment_year]
        attach_storageunits(n, costs, elec_config, investment_year, bus_multipliers=bus_multipliers)
        attach_renewable_vintages(n, costs, renewable_sites, investment_year, bus_multipliers=bus_multipliers)
        attach_multihorizon_egs(n, costs, costs_dict, egs_gens, investment_year)
        attach_new_thermal(
            n,
            costs,
            thermal_new,
            investment_year,
            bus_multipliers=bus_multipliers,
            min_years=min_years,
        )

    # the sites only carried the resource to the vintages
    if not renewable_sites.empty:
        n.remove("Generator", renewable_sites.index)

    if not egs_gens.empty and not len(n.investment_periods) == 1:
        # Remove original EGS generators now that vintaged copies exist for
        # each investment period. Without this, the originals (build_year=0)
        # are always active, accumulate capacity through prepare_brownfield,
        # and compound across periods.
        n.remove("Generator", egs_gens.index)

    # gas, coal and oil plants pay the AEO fuel costs, nuclear and biomass plants the ATB
    # fuel costs, unless dynamic fuel prices are enabled
    if not snakemake.config["conventional"]["dynamic_fuel_price"].get("enable", False):
        apply_power_fuel_prices(n, costs_dict)

    # The ITC is in the capital costs already (capital charge rate, load_costs)
    apply_ptc(n, costs_dict)
    apply_max_annual_growth_rate(n, snakemake.config["costs"]["max_growth"])
    add_nice_carrier_names(n, snakemake.config)
    add_co2_emissions(n, costs_dict[n.investment_periods[0]], n.carriers.index)

    dr_config = snakemake.params.demand_response
    if dr_config:
        add_demand_response(n, dr_config)

    imports_config = snakemake.params.imports
    exports_config = snakemake.params.exports
    representation = imports_config.get("representation", "store")

    if snakemake.params.transmission_network == "reeds":
        # flowgates to limit the capacity (removed later if configured capacity limit is inf)
        flowgates = pd.read_csv(snakemake.input.flowgates)
        membership = pd.read_csv(snakemake.input.reeds_memberships)
        if snakemake.params.topological_boundaries == "state":
            zone_col = "reeds_state"
            flowgates = convert_flowgates_to_state(flowgates, membership)
            flowgates = format_flowgates_for_imports_exports(n, flowgates, zone_col)
            flowgates = flowgates.groupby(["r", "rr"], as_index=False).sum()
        elif snakemake.params.topological_boundaries == "county":
            zone_col = "county"
            flowgates = format_flowgates_for_imports_exports(n, flowgates, zone_col)
        elif snakemake.params.topological_boundaries == "reeds_zone":
            zone_col = "reeds_zone"
            flowgates = format_flowgates_for_imports_exports(n, flowgates, zone_col)
        else:
            raise ValueError(f"Invalid topological boundaries: {snakemake.params.topological_boundaries}")

    # Electricity imports configuration
    if imports_config.get("enable", False) and snakemake.params.transmission_network == "reeds":
        co2_emissions = imports_config.get("co2_emissions", 0)

        weather_year = snakemake.params.weather_year
        if isinstance(weather_year, list):
            year = weather_year[0]

        import_flowgates = flowgates.copy()
        if not imports_config.get("capacity_limit", True):
            import_flowgates["value"] = np.inf

        fuel_costs = resolve_trade_costs(n, imports_config, "imports", snakemake.params.eia_api, year)

        # Only `generator` mode places the CPUC contracted units behind the
        # boundary; in `store` mode add_electricity has already attached them
        # and the bundle is a sentinel.
        remote_bundle = load_remote_unit_bundle(snakemake.input.remote_units) if representation == "generator" else None

        add_external_regions(
            n,
            "imports",
            representation,
            import_flowgates,
            fuel_costs,
            co2_emissions,
            zone_col,
            remote_bundle=remote_bundle,
            membership=membership,
        )

    # Electricity exports configuration
    if exports_config.get("enable", False) and snakemake.params.transmission_network == "reeds":
        co2_emissions = 0

        weather_year = snakemake.params.weather_year
        if isinstance(weather_year, list):
            year = weather_year[0]

        # flowgates to limit the capacity
        export_flowgates = flowgates.copy()
        if not exports_config.get("capacity_limit", True):
            export_flowgates["value"] = np.inf

        fuel_costs = resolve_trade_costs(n, exports_config, "exports", snakemake.params.eia_api, year)

        add_external_regions(
            n,
            "exports",
            representation,
            export_flowgates,
            fuel_costs,
            co2_emissions,
            zone_col,
        )

    if snakemake.config["scenario"]["sector"] == "E":
        co2_storage = snakemake.config.get("co2", {}).get("storage", False)
        co2_network_enable = snakemake.config.get("co2", {}).get("network", {}).get("enable", False)
        dac_enable = snakemake.config.get("dac", {}).get("enable", False)

        # add node level CO2 (underground) storage
        if co2_storage:
            logger.info("Adding node level CO2 (underground) storage")
            add_co2_storage(n, snakemake.config, snakemake.input.co2_storage, costs, False)

        # add CO2 (transportation) network
        if co2_network_enable:
            if co2_storage:
                logger.info("Adding CO2 (transportation) network")
                add_co2_network(n, snakemake.config)
            else:
                logger.warning(
                    "Not adding CO2 (transportation) network given that CO2 (underground) storage is not enabled",
                )

        # add node level DAC capabilities
        if dac_enable:
            if co2_storage:
                logger.info("Adding DAC capabilities")
                add_dac(n, snakemake.config, False)
            else:
                logger.warning(
                    "Not adding DAC capabilities given that CO2 (underground) storage is not enabled",
                )

    ambient_settings = ambient_derate_settings(snakemake.config.get("conventional"))
    apply_ambient_derate_new_generators(
        n,
        ambient_settings,
        snakemake.input.ambient_temperature,
        snakemake.input.outage_curves,
        snakemake.input.busmap_c,
    )

    # Apply UCAP derating to conventional generators
    ucap_config = snakemake.config.get("ucap", {})
    check_ucap_vs_ambient_derate(ucap_config, ambient_settings["prime_movers"])
    if ucap_config.get("enable", False):
        logger.info("Applying UCAP derating to conventional generators")
        apply_ucap(n, ucap_config)

    n.consistency_check()
    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    log_network_schema(n, stage="exit", baseline=schema_entry)
    n.export_to_netcdf(snakemake.output[0])


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_extra_components",
            interconnect="western",
            simpl="20",
            clusters="4m",
        )
    configure_logging(snakemake)
    main(snakemake)
