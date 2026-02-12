"""Adds extra extendable components to the clustered and simplified network."""

import logging

import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging
from add_electricity import add_missing_carriers

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def add_co2_emissions(n, costs, carriers):
    """Add CO2 emissions to the network's carriers attribute."""
    suptechs = n.carriers.loc[carriers].index.str.split("-").str[0]
    missing_carriers = set(suptechs) - set(costs.index)
    if missing_carriers:
        logger.warning(
            f"CO2 emissions for carriers {missing_carriers} not defined in cost data.",
        )
        suptechs = suptechs.difference(missing_carriers)
    try:
        n.carriers.loc[suptechs, "co2_emissions"] = costs.co2_emissions[suptechs].values
    except KeyError:
        pass

    n.carriers = n.carriers.fillna(
        {"co2_emissions": 0},
    )  # TODO: FIX THIS ISSUE IN BUILD_COST_DATA- missing co2_emissions for some VRE carriers

    if any("CCS" in carrier for carrier in carriers):
        ccs_carriers = [carrier for carrier in carriers if "CCS" in carrier]
        for ccs_carrier in ccs_carriers:
            if ccs_carrier == "biomass-CCS":
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


def attach_storageunits(n, costs, elec_opts, investment_year):
    carriers = elec_opts["extendable_carriers"]["StorageUnit"]
    carriers = [k for k in carriers if "battery_storage" in k]

    buses_i = n.buses.index

    add_missing_carriers(n, carriers)
    add_co2_emissions(n, costs, carriers)
    for carrier in carriers:
        max_hours = int(carrier.split("hr_")[0])
        roundtrip_correction = 0.5 if "battery" in carrier else 1

        n.madd(
            "StorageUnit",
            buses_i,
            suffix=f" {carrier}_{investment_year}",
            bus=buses_i,
            carrier=carrier,
            p_nom_extendable=True,
            capital_cost=costs.at[carrier, "annualized_capex_fom"],
            marginal_cost=0,  # costs.at[carrier, "marginal_cost"], # TODO: FIX THIS ISSUE IN BUILD_COST_DATA
            efficiency_store=costs.at[carrier, "efficiency"] ** roundtrip_correction,
            efficiency_dispatch=costs.at[carrier, "efficiency"] ** roundtrip_correction,
            max_hours=max_hours,
            cyclic_state_of_charge=True,
            build_year=investment_year,
            lifetime=costs.at[carrier, "cost_recovery_period_years"],
        )


def add_existing_phs(n, costs, data_file):
    """
    Add existing pumped hydro storage units to the network based on ReEDS data.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network
    costs : pd.DataFrame
        Cost dataframe with technology parameters
    """
    # 1. Read data file
    data = pd.read_csv(data_file)

    # 2. Filter rows where tech column equals 'pumped-hydro', keep only cap, TSTATE, T_FOM columns
    phs_data = data[data["tech"] == "pumped-hydro"][["cap", "TSTATE", "T_FOM"]].copy()

    # 3. Aggregate by TSTATE column: sum cap, weighted average T_FOM by capacity
    phs_aggregated = phs_data.groupby("TSTATE").agg(
        {
            "cap": "sum",
            "T_FOM": lambda x: (x * phs_data.loc[x.index, "cap"]).sum() / phs_data.loc[x.index, "cap"].sum(),
        }
    )

    # 4. Techno-economic parameters
    efficiency_store = 0.894427191  # 0.894427191^2 = 0.8
    efficiency_dispatch = 0.894427191  # 0.894427191^2 = 0.8

    # Calculate capital cost (based on T_FOM)
    capital_cost = phs_aggregated["T_FOM"].values * 1000

    # Add carrier and emissions
    costs.at["PHS", "efficiency"] = efficiency_store
    costs.at["PHS", "co2_emissions"] = 0
    add_missing_carriers(n, ["PHS"])
    add_co2_emissions(n, costs, ["PHS"])

    # Add storage units using madd
    n.madd(
        "StorageUnit",
        names=phs_aggregated.index,
        suffix=" PHS",
        bus=phs_aggregated.index,
        carrier="PHS",
        p_nom_extendable=False,
        p_nom=phs_aggregated["cap"].values,
        p_nom_max=phs_aggregated["cap"].values,
        capital_cost=capital_cost,
        marginal_cost=0,
        efficiency_store=efficiency_store,
        efficiency_dispatch=efficiency_dispatch,
        max_hours=553.0 / 23.0 * efficiency_dispatch,
        cyclic_state_of_charge=True,
        lifetime=np.inf,
    )


def attach_stores(n, costs, elec_opts, investment_year):
    carriers = elec_opts["extendable_carriers"]["Store"]

    add_missing_carriers(n, carriers)
    add_co2_emissions(n, costs, carriers)

    buses_i = n.buses.index
    bus_sub_dict = {k: n.buses[k].values for k in ["x", "y", "country"]}

    if "H2" in carriers:
        h2_buses_i = n.madd("Bus", buses_i + " H2", carrier="H2", **bus_sub_dict)

        n.madd(
            "Store",
            h2_buses_i,
            bus=h2_buses_i,
            carrier="H2",
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=costs.at["hydrogen storage underground", "capital_cost"],
            build_year=investment_year,
            lifetime=costs.at["hydrogen storage underground", "lifetime"],
            suffix=f" {investment_year}",
        )

        n.madd(
            "Link",
            h2_buses_i + " Electrolysis",
            bus0=buses_i,
            bus1=h2_buses_i,
            carrier="H2 electrolysis",
            p_nom_extendable=True,
            efficiency=costs.at["electrolysis", "efficiency"],
            capital_cost=costs.at["electrolysis", "capital_cost"],
            marginal_cost=costs.at["electrolysis", "marginal_cost"],
            build_year=investment_year,
            lifetime=costs.at["electrolysis", "lifetime"],
            suffix=str(investment_year),
        )

        n.madd(
            "Link",
            h2_buses_i + " Fuel Cell",
            bus0=h2_buses_i,
            bus1=buses_i,
            carrier="H2 fuel cell",
            p_nom_extendable=True,
            efficiency=costs.at["fuel cell", "efficiency"],
            # NB: fixed cost is per MWel
            capital_cost=costs.at["fuel cell", "capital_cost"] * costs.at["fuel cell", "efficiency"],
            marginal_cost=costs.at["fuel cell", "marginal_cost"],
            build_year=investment_year,
            lifetime=costs.at["fuel cell", "lifetime"],
            suffix=str(investment_year),
        )


def split_retirement_gens(
    n: pypsa.Network,
    costs: pd.DataFrame,
    carriers: list[str] | None = None,
    economic: bool = True,
    add_new_gens: bool = True,
):
    """
    Seperates extendable conventional generators into existing and new
    generators to support economic or technical retirement.


    Specifically this function does the following:
    1. Creates duplicate generators for any that are tagged as extendable. For
    example, an extendable "CCGT" generator will be split into "CCGT existing" and "CCGT"
    2. Capital costs of existing extendable generators are replaced with fixed costs
    3. p_nom_max of existing extendable generators are set to p_nom
    4. p_nom_min of existing and new generators is set to zero

    Arguments:
    n: pypsa.Network,
    costs: pd.DataFrame,
    carriers: List[str]
        List of generator carriers to apply economic retirment to.
    economic: bool
        If True, enable economic retirement, else only allow lifetime
        retirement for the new generators
    """
    retirement_mask = (
        n.generators["p_nom_extendable"]
        & (n.generators["carrier"].isin(carriers) if carriers else True)
        & n.generators.p_nom
        > 0
    )
    retirement_gens = n.generators[retirement_mask]
    if retirement_gens.empty:
        return

    # Change capex to fixed OM cost for retiring generators
    n.generators["capital_cost"] = n.generators.apply(
        lambda row: (
            row["capital_cost"]
            if row.name not in (retirement_gens.index)
            else costs.at[row["carrier"], "opex_fixed_per_kw"]
            * 1e3
            * (1.29 if row["carrier"] == "CCGT" else (1.19 if row["carrier"] == "OCGT" else 1))
        ),
        axis=1,
    )

    # Apply VOM cost multiplier for CCGT carriers
    ccgt_mask = (n.generators["carrier"] == "CCGT") & (n.generators.index.isin(retirement_gens.index))
    n.generators.loc[ccgt_mask, "vom_cost"] = n.generators.loc[ccgt_mask, "vom_cost"] * 1.21

    # Rename retiring generators to include "existing" suffix
    n.generators.index = n.generators.apply(
        lambda row: (row.name if row.name not in (retirement_gens.index) else row.name + " existing"),
        axis=1,
    )

    n.generators["p_nom_max"] = np.where(
        retirement_mask,
        n.generators["p_nom"],
        n.generators["p_nom_max"],
    )

    n.generators["p_nom_min"] = np.where(
        retirement_mask,
        0,
        n.generators["p_nom_min"],
    )

    n.generators.loc[
        retirement_mask.values,
        "p_nom_extendable",
    ] = economic  # if economic retirement is true enable extendable

    # Rename time series data for retired generators (must happen regardless of add_new_gens)
    for attr in ["marginal_cost", "p_max_pu", "p_min_pu", "efficiency"]:
        if attr in n.generators_t and not n.generators_t[attr].empty:
            existing_cols = [x for x in retirement_gens.index if x in n.generators_t[attr].columns]
            if existing_cols:
                # Rename existing time series to " existing"
                rename_dict = {x: f"{x} existing" for x in existing_cols}
                n.generators_t[attr] = n.generators_t[attr].rename(columns=rename_dict)

    # Only add new generators if requested
    # Otherwise, they will be added to ALL buses via attach_multihorizon_new_generators
    if add_new_gens:
        # Adding Expanding generators for the first investment period
        # There are generators that exist today and could expand
        # in the first time horizon
        n.madd(
            "Generator",
            retirement_gens.index,
            carrier=retirement_gens.carrier,
            bus=retirement_gens.bus,
            p_nom_min=0,
            p_nom=0,
            p_nom_max=retirement_gens.p_nom_max,
            p_nom_extendable=True,
            ramp_limit_up=retirement_gens.ramp_limit_up,
            ramp_limit_down=retirement_gens.ramp_limit_down,
            efficiency=retirement_gens.carrier.map(costs.efficiency).fillna(retirement_gens.efficiency),
            marginal_cost=retirement_gens.carrier.map(costs.marginal_cost).fillna(retirement_gens.marginal_cost),
            capital_cost=retirement_gens.carrier.map(costs.annualized_capex_fom),
            vom_cost=retirement_gens.carrier.map(costs.opex_variable_per_mwh).fillna(0),
            build_year=n.investment_periods[0],
            lifetime=retirement_gens.carrier.map(costs.lifetime).fillna(np.inf),
            p_min_pu=retirement_gens.p_min_pu,
            p_max_pu=retirement_gens.p_max_pu,
            land_region=retirement_gens.land_region,
        )

        # Copy time series data for new generators from the renamed " existing" columns
        for attr in ["marginal_cost", "p_max_pu"]:
            if attr in n.generators_t and not n.generators_t[attr].empty:
                existing_cols = [
                    f"{x} existing" for x in retirement_gens.index if f"{x} existing" in n.generators_t[attr].columns
                ]
                if existing_cols:
                    # Copy from " existing" columns to new generator columns
                    new_cols_data = n.generators_t[attr][existing_cols].copy()
                    new_cols_data.columns = [x.replace(" existing", "") for x in new_cols_data.columns]
                    n.generators_t[attr] = n.generators_t[attr].join(new_cols_data)


def attach_multihorizon_existing_generators(
    n: pypsa.Network,
    costs: dict,
    gens: pd.DataFrame,
    investment_year: int,
):
    """
    Adds multiple investment options for generators types that were already
    existing in the network. Function used for all carriers, renewable and
    conventional. Generators are added only to the nodes where they already exist
    because their cost information is spatially resolved.

    Specifically this function does the following:
    1. Adds new generators for the given investment year, according that year's costs.
        if this is the first investment period we use the existing generator's p_nom and p_nom_min
    2. Adds time dependent factors for the new generators


    Arguments:
    n: pypsa.Network,
    costs_dict: dict,
        Dict of costs for each investment period
    carriers: List[str]
        List of carriers to add multiple investment options for
    """
    if gens.empty or len(n.investment_periods) == 1:
        return

    n.madd(
        "Generator",
        gens.index,
        suffix=f" {investment_year}",
        carrier=gens.carrier,
        bus=gens.bus,
        p_nom_min=0 if investment_year != n.investment_periods[0] else gens.p_nom_min,
        p_nom=0 if investment_year != n.investment_periods[0] else gens.p_nom,
        p_nom_max=gens.p_nom_max,
        p_nom_extendable=True,
        ramp_limit_up=gens.ramp_limit_up,
        ramp_limit_down=gens.ramp_limit_down,
        efficiency=gens.efficiency,
        marginal_cost=gens.marginal_cost,
        p_min_pu=gens.p_min_pu,
        p_max_pu=gens.p_max_pu,
        capital_cost=gens.carrier.map(costs.annualized_capex_fom),
        build_year=investment_year,
        lifetime=gens.carrier.map(costs.cost_recovery_period_years),
        land_region=gens.land_region,
    )

    # time dependent factors added after as not all generators are time dependent
    marginal_cost_t = n.generators_t["marginal_cost"][
        [x for x in gens.index if x in n.generators_t.marginal_cost.columns]
    ]
    marginal_cost_t = marginal_cost_t.rename(
        columns={x: f"{x} {investment_year}" for x in marginal_cost_t.columns},
    )
    n.generators_t["marginal_cost"] = n.generators_t["marginal_cost"].join(
        marginal_cost_t,
    )

    p_max_pu_t = n.generators_t["p_max_pu"][[x for x in gens.index if x in n.generators_t["p_max_pu"].columns]]
    p_max_pu_t = p_max_pu_t.rename(
        columns={x: f"{x} {investment_year}" for x in p_max_pu_t.columns},
    )
    n.generators_t["p_max_pu"] = n.generators_t["p_max_pu"].join(p_max_pu_t)


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
    n.madd(
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
    marginal_cost_t = n.generators_t["marginal_cost"][
        [x for x in gens.index if x in n.generators_t.marginal_cost.columns]
    ]
    marginal_cost_t = marginal_cost_t.rename(
        columns={x: f"{x} {investment_year}" for x in marginal_cost_t.columns},
    )
    n.generators_t["marginal_cost"] = n.generators_t["marginal_cost"].join(
        marginal_cost_t,
    )

    p_max_pu_t = n.generators_t["p_max_pu"][[x for x in gens.index if x in n.generators_t["p_max_pu"].columns]]

    p_max_pu_t = p_max_pu_t.rename(
        columns={x: f"{x} {investment_year}" for x in p_max_pu_t.columns},
    )

    n.generators_t["p_max_pu"] = n.generators_t["p_max_pu"].join(p_max_pu_t)

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


def attach_multihorizon_new_generators(n, costs, carriers, investment_year):
    """
    Attaches generators for carriers which did not previously exist in the
    network (CCS, H2, SMR, etc). These generators do not have spatially resolved
    costs, so they are added to all buses in the network.

    Unlike CT's and CCGT's we include nuclear in this function, since we assume
    they can be built anywhere in the network.

    Specifically this function does the following:
    1. Adds new carriers to the network
    2. Adds generators for the new carriers

    Arguments:
    n: pypsa.Network,
    costs: pd.DataFrame,
    carriers: List[str]
        List of carriers to add to the network
    investment_year: int
        Year of investment
    """
    if not carriers:
        return

    add_missing_carriers(n, carriers)
    add_co2_emissions(n, costs, carriers)
    min_years = snakemake.config["costs"].get("min_year")
    buses_i = n.buses.index
    for carrier in carriers:
        p_max_pu_t = None
        if min_years and min_years.get(carrier, 0) > investment_year:
            continue
        existing_gens = n.generators[
            (
                (n.generators.carrier == carrier)
                & ~n.generators.index.str.contains("existing")
                & (n.generators.build_year <= n.investment_periods[0])
            )
        ].copy()

        if not existing_gens.empty:
            p_max_pu_t = n.get_switchable_as_dense("Generator", "p_max_pu")
            p_max_pu_t = (p_max_pu_t[[x for x in existing_gens.index if x in p_max_pu_t.columns]]).mean().mean()

        n.madd(
            "Generator",
            buses_i,
            suffix=f" {carrier}_{investment_year}",
            bus=buses_i,
            carrier=carrier,
            p_nom_extendable=True,
            capital_cost=costs.at[carrier, "annualized_capex_fom"],
            marginal_cost=costs.at[carrier, "marginal_cost"],
            vom_cost=costs.at[carrier, "opex_variable_per_mwh"],
            efficiency=costs.at[carrier, "efficiency"],
            build_year=investment_year,
            lifetime=costs.at[carrier, "lifetime"],
            p_max_pu=p_max_pu_t if p_max_pu_t is not None else 1,
            ramp_limit_up=existing_gens.ramp_limit_up.mean() or 1,
            ramp_limit_down=existing_gens.ramp_limit_down.mean() or 1,
        )


def apply_itc(n, itc_modifier):
    """
    Applies investment tax credit to all extendable components in the network.

    Arguments:
    n: pypsa.Network,
    itc_modifier: dict,
        Dict of ITC modifiers for each carrier
    """
    for carrier in itc_modifier.keys():
        carrier_mask = n.generators["carrier"] == carrier
        n.generators.loc[carrier_mask, "capital_cost"] *= 1 - itc_modifier[carrier]

        carrier_mask = n.storage_units["carrier"] == carrier
        n.storage_units.loc[carrier_mask, "capital_cost"] *= 1 - itc_modifier[carrier]


def apply_ptc(n, ptc_modifier):
    """
    Applies production tax credit to all extendable components in the network.

    Arguments:
    n: pypsa.Network,
    ptc_modifier: dict,
        Dict of PTC modifiers for each carrier
    """
    for carrier in ptc_modifier.keys():
        carrier_mask = n.generators["carrier"] == carrier
        mc = n.get_switchable_as_dense("Generator", "marginal_cost").loc[
            :,
            carrier_mask,
        ]
        n.generators_t.marginal_cost.loc[:, carrier_mask] = mc - ptc_modifier[carrier]
        n.generators.loc[carrier_mask, "marginal_cost"] -= ptc_modifier[carrier]


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
        logger.info(f"DR not applied as allowable sift is {shift}")
        return

    marginal_cost_storage = dr_config.get("marginal_cost", 0)
    if marginal_cost_storage == 0:
        logger.warning("No cost applied to demand response")

    # attach dr at all load locations

    buses = n.loads.bus
    df = n.buses[n.buses.index.isin(buses)].copy()

    # two storageunits for forward and backwards load shifting

    n.madd(
        "Bus",
        names=df.index,
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

    n.madd(
        "Bus",
        names=df.index,
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

    n.madd(
        "Link",
        names=df.index,
        suffix="-fwd-dr-charger",
        bus0=df.index,
        bus1=df.index + "-fwd-dr",
        carrier="demand_response",
        p_nom_extendable=False,
        p_nom=np.inf,
    )

    n.madd(
        "Link",
        names=df.index,
        suffix="-fwd-dr-discharger",
        bus0=df.index + "-fwd-dr",
        bus1=df.index,
        carrier="demand_response",
        p_nom_extendable=False,
        p_nom=np.inf,
    )

    n.madd(
        "Link",
        names=df.index,
        suffix="-bck-dr-charger",
        bus0=df.index,
        bus1=df.index + "-bck-dr",
        carrier="demand_response",
        p_nom_extendable=False,
        p_nom=np.inf,
    )

    n.madd(
        "Link",
        names=df.index,
        suffix="-bck-dr-discharger",
        bus0=df.index + "-bck-dr",
        bus1=df.index,
        carrier="demand_response",
        p_nom_extendable=False,
        p_nom=np.inf,
    )

    # backward stores have positive marginal cost storage and postive e
    # forward stores have negative marginal cost storage and negative e

    n.madd(
        "Store",
        names=df.index,
        suffix="-bck-dr",
        bus=df.index + "-bck-dr",
        e_cyclic=True,
        e_nom_extendable=False,
        e_nom=np.inf,
        e_min_pu=0,
        e_max_pu=1,
        carrier="demand_response",
        marginal_cost_storage=marginal_cost_storage,
    )

    n.madd(
        "Store",
        names=df.index,
        suffix="-fwd-dr",
        bus=df.index + "-fwd-dr",
        e_cyclic=True,
        e_nom_extendable=False,
        e_nom=np.inf,
        e_min_pu=-1,
        e_max_pu=0,
        carrier="demand_response",
        marginal_cost_storage=marginal_cost_storage * (-1),
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_extra_components",
            case="HighE_reeds_new_h2storage_tes",
            transmission_network="reeds",
        )
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    elec_config = snakemake.config["electricity"]

    costs_dict = {
        n.investment_periods[i]: pd.read_csv(snakemake.input.tech_costs[i]).pivot(
            index="pypsa-name",
            columns="parameter",
            values="value",
        )
        for i in range(len(n.investment_periods))
    }

    add_existing_phs(n, costs_dict[n.investment_periods[0]], snakemake.input.existing_PHS)

    # Handle existing conventional generators for economic retirement
    # Only mark existing ones, don't create new ones here
    if snakemake.params.retirement == "economic":
        economic_retirement_gens = set(elec_config.get("conventional_carriers", None))
        split_retirement_gens(
            n,
            costs_dict[n.investment_periods[0]],
            economic_retirement_gens,
            economic=True,
            add_new_gens=False,  # Don't add new generators in split_retirement_gens
        )

    # Split renewable generators from the first investment period to support lifetime retirement
    split_retirement_gens(
        n,
        costs_dict[n.investment_periods[0]],
        set(elec_config.get("renewable_carriers", None)),
        economic=False,
        add_new_gens=True,  # Don't add new generators here either
    )

    multi_horizon_gens = n.generators[
        n.generators["p_nom_extendable"]
        & n.generators["carrier"].isin(elec_config["extendable_carriers"]["Generator"])
        & ~n.generators.index.str.contains("existing")
    ]

    multi_horizon_gens = multi_horizon_gens[
        multi_horizon_gens["carrier"].isin(
            [car for car in elec_config["extendable_carriers"]["Generator"] if "EGS" not in car],
        )
    ]
    egs_gens = n.generators[n.generators["p_nom_extendable"]]
    egs_gens = egs_gens.loc[egs_gens["carrier"].str.contains("EGS")]

    # Include all conventional carriers and new carriers to be added to ALL buses
    conventional_carriers = set(elec_config.get("conventional_carriers", []))
    extendable_carriers = set(elec_config["extendable_carriers"].get("Generator", []))

    # New carriers that should be added to all buses:
    # 1. Carriers not yet in network
    # 2. All conventional carriers (so they're available everywhere, not just where existing plants are)
    # 3. Nuclear (as it was originally included)
    new_carriers = list(
        (extendable_carriers - set(n.generators.carrier.unique()))  # Carriers not in network
        | (extendable_carriers & conventional_carriers)  # Extendable conventional carriers
        | ({"nuclear"} if "nuclear" in extendable_carriers else set()),  # Nuclear
    )

    for investment_year in n.investment_periods:
        costs = costs_dict[investment_year]
        attach_storageunits(n, costs, elec_config, investment_year)
        attach_multihorizon_existing_generators(
            n,
            costs,
            multi_horizon_gens,
            investment_year,
        )
        attach_multihorizon_egs(n, costs, costs_dict, egs_gens, investment_year)
        attach_multihorizon_new_generators(n, costs, new_carriers, investment_year)
        # attach_stores(n, costs, elec_config, investment_year)

    if not multi_horizon_gens.empty and not len(n.investment_periods) == 1:
        # Remove duplicate generators from first investment period,
        # created by attach_multihorizon_generators
        n.mremove("Generator", multi_horizon_gens.index)

    apply_itc(n, snakemake.config["costs"]["itc_modifier"])
    apply_ptc(n, snakemake.config["costs"]["ptc_modifier"])
    apply_max_annual_growth_rate(n, snakemake.config["costs"]["max_growth"])
    add_nice_carrier_names(n, snakemake.config)
    add_co2_emissions(n, costs_dict[n.investment_periods[0]], n.carriers.index)

    dr_config = snakemake.params.demand_response
    if dr_config:
        add_demand_response(n, dr_config)

    # generators_to_remove = [
    #     gen for gen in n.generators.index
    #     if (('coal' in gen) and ('existing' not in gen)) or
    #        (('nuclear' in gen) and ('_' not in gen)) and ('existing' not in gen) or
    #        (('biomass' in gen) and ('CCS' not in gen) and ('existing' not in gen))
    # ]
    generators_to_remove = [
        gen
        for gen in n.generators.index
        if (("coal" in gen) and ("existing" not in gen))
        or (("biomass" in gen) and ("existing" not in gen) and ("CCS" not in gen))
        or (("OCGT" in gen) and ("existing" not in gen))
        or
        # (('CCGT' in gen) and ('CCS' not in gen) and ('existing' not in gen)) or
        ((("nuclear" in gen) and ("_" not in gen)) and ("existing" not in gen))
    ]
    if generators_to_remove:
        n.mremove("Generator", generators_to_remove)

    n.consistency_check()
    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
