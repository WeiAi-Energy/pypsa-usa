"""Adds extra extendable components to the clustered and simplified network."""

import logging
from collections.abc import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from _helpers import calculate_annuity, configure_logging, set_case_config, update_config_from_wildcards
from add_electricity import add_missing_carriers
from opts._helpers import get_region_buses
from regional_cost import (
    SectorCosts,
    bus_multiplier_table,
    carrier_multiplier,
    county_multiplier_table,
    load_reg_cap_cost_diff,
    overnight_delta_capital_cost,
)
from pypsa.descriptors import get_switchable_as_dense as get_as_dense

idx = pd.IndexSlice

logger = logging.getLogger(__name__)

FLEXIBLE_ELECTROLYSIS_BUS_SUFFIX = " flexible electrolysis H2"
FLEXIBLE_ELECTROLYSIS_LINK_SUFFIX = " flexible electrolysis"

NEW_GAS_COST_CARRIERS = {"OCGT", "CCGT"}

# Base carriers whose existing units make a bus eligible for new nuclear. New
# nuclear is sited on the footprint of the existing thermal fleet -- coal,
# gas (OCGT/CCGT and their -CCS variants) and nuclear -- because those buses
# already have the steam/turbine interconnection, cooling and transmission
# headroom a new reactor needs. Matching is on the base carrier, so
# "CCGT-95CCS" counts as "CCGT" and "coal-95CCS" as "coal".
NUCLEAR_SITING_BASE_CARRIERS = {"coal", "CCGT", "OCGT", "nuclear"}

# `coal` is kept in electricity.conventional_carriers (and in
# extendable_carriers) purely so the existing coal fleet is loaded into the
# network and can define the new-nuclear siting set above. No coal capacity --
# existing or new-build -- is meant to survive into the solved network; see
# drop_coal_generators and its call site in __main__.
COAL_BASE_CARRIER = "coal"

# Base carriers that never get a new-build option, whatever
# electricity.extendable_carriers lists. The two get there differently:
#   - `coal` has no new-build option because the whole carrier leaves the network
#     (drop_coal_generators); it is configured only to seed the siting sets above.
#   - `OCGT` keeps its existing fleet -- those units still dispatch, still count
#     towards the new-nuclear footprint and still define the gas pool new
#     CCGT/CCGT-CCS is sited on -- but no new open-cycle capacity may be built.
# Matching is on the base carrier, so "OCGT-95CCS" is excluded along with "OCGT".
NO_NEW_BUILD_BASE_CARRIERS = {"coal", "OCGT"}

# Carriers with one generator per bus already attached by attach_wind_and_solar
# in add_electricity.py (real atlite resource p_nom_max + capacity-factor
# profile, using the same unified cost table). Their new-build capacity is
# already fully represented before this script ever runs, so they must be
# excluded from the generic attach_new_generators path -- adding a second,
# uncapped generator there would ignore the real resource limit entirely.
RESOURCE_PROFILE_CARRIERS = {"onwind", "offwind", "offwind_floating", "solar"}


def carrier_new_build_buses(
    n: pypsa.Network,
    carrier: str,
    all_buses_i: pd.Index,
    gas_union_buses_i: pd.Index,
) -> pd.Index:
    """
    Determine which buses are eligible for new-build capacity of `carrier`.

    Only called for carriers not in RESOURCE_PROFILE_CARRIERS -- wind/solar
    already have their own resource-capped, per-bus new-build option from
    add_electricity.py and never reach this function.

    - nuclear is restricted to the union of buses that already host a
      coal, OCGT, CCGT, CCGT-CCS or nuclear unit (see
      NUCLEAR_SITING_BASE_CARRIERS) -- the existing thermal footprint --
      rather than being buildable at every AC bus. If no such bus exists,
      nuclear is simply unbuildable (no fallback to all AC buses, which
      would void the restriction).
    - CCGT/CCGT-CCS share one pool: the union of buses with existing
      OCGT or CCGT capacity, since gas new-build is about existing gas
      infrastructure rather than a siting resource. (OCGT itself is never a
      new-build carrier -- see NO_NEW_BUILD_BASE_CARRIERS -- but its existing
      fleet stays in the network and its buses still define that pool.)
    - Every other carrier (coal, hydrogen_ct, ...) is
      restricted to buses that already have that carrier -- or, for a
      "-CCS" variant, its base carrier -- installed, since that's where the
      underlying site/interconnection exists.
    - For those other carriers, if no such bus exists, fall back to all AC
      buses rather than making it unbuildable.
      This inverts the siting restriction into no restriction at all, so it
      is logged -- it adds one extendable generator per AC bus.
    """
    if carrier == "nuclear":
        thermal_buses_i = pd.Index(
            n.generators.loc[
                n.generators.carrier.str.split("-").str[0].isin(NUCLEAR_SITING_BASE_CARRIERS),
                "bus",
            ].unique(),
        )
        if thermal_buses_i.empty:
            # Deliberately no fallback to all AC buses here (unlike the generic
            # branch below): falling back would turn the siting restriction into
            # no restriction at all and make nuclear buildable everywhere.
            logger.warning(
                "No existing %s capacity in the network -- new nuclear is unbuildable.",
                sorted(NUCLEAR_SITING_BASE_CARRIERS),
            )
            return thermal_buses_i
        logger.info(
            "New nuclear restricted to %s of %s AC buses with existing %s capacity.",
            len(thermal_buses_i),
            len(all_buses_i),
            sorted(NUCLEAR_SITING_BASE_CARRIERS),
        )
        return thermal_buses_i
    base_carrier = carrier.split("-")[0]
    if base_carrier in NEW_GAS_COST_CARRIERS:
        return gas_union_buses_i
    buses_i = pd.Index(n.generators.loc[n.generators.carrier == base_carrier, "bus"].unique())
    if buses_i.empty:
        logger.warning(
            "No existing %s capacity to site new %s against -- falling back to all "
            "%s AC buses. Add %s to electricity.conventional_carriers to restrict "
            "new-build to existing sites, or remove it from the new-build carriers.",
            base_carrier,
            carrier,
            len(all_buses_i),
            base_carrier,
        )
        return all_buses_i
    return buses_i


def add_co2_emissions(n, costs, carriers):
    """Add CO2 emissions to the network's carriers attribute."""
    suptechs = pd.Index(n.carriers.loc[carriers].index.str.split("-").str[0]).unique()
    missing_cost_carriers = set(suptechs) - set(costs.index)
    if missing_cost_carriers:
        logger.warning(
            f"CO2 emissions for carriers {missing_cost_carriers} not defined in cost data.",
        )
    missing_network_carriers = set(suptechs) - set(n.carriers.index)
    if missing_network_carriers:
        logger.warning(
            f"CO2 emissions target carriers {missing_network_carriers} not present in network carriers.",
        )

    valid_suptechs = suptechs.intersection(costs.index).intersection(n.carriers.index)
    n.carriers.loc[valid_suptechs, "co2_emissions"] = costs.co2_emissions[valid_suptechs].values

    n.carriers = n.carriers.fillna(
        {"co2_emissions": 0},
    )  # TODO: FIX THIS ISSUE IN BUILD_COST_DATA- missing co2_emissions for some VRE carriers

    if any("CCS" in carrier for carrier in carriers):
        ccs_carriers = [carrier for carrier in carriers if "CCS" in carrier]
        for ccs_carrier in ccs_carriers:
            base_carrier = ccs_carrier.split("-")[0]
            if base_carrier in n.carriers.index:
                base_emissions = n.carriers.loc[base_carrier, "co2_emissions"]
            elif base_carrier in costs.index:
                base_emissions = costs.at[base_carrier, "co2_emissions"]
                logger.warning(
                    "Base carrier %s for %s not present in network carriers; using costs table co2_emissions.",
                    base_carrier,
                    ccs_carrier,
                )
            else:
                logger.warning(
                    "Skipping CO2 emission derivation for %s because base carrier %s is missing from both network carriers and costs table.",
                    ccs_carrier,
                    base_carrier,
                )
                continue
            ccs_level = int(ccs_carrier.split("-")[1].replace("CCS", ""))
            # Only direct (combustion) emissions are capturable; the indirect
            # (upstream) share passes through at any capture rate.
            indirect_emissions = 0.0
            if "co2_emissions_indirect" in costs.columns and base_carrier in costs.index:
                carrier_indirect = costs.at[base_carrier, "co2_emissions_indirect"]
                if pd.notna(carrier_indirect):
                    indirect_emissions = carrier_indirect
            direct_emissions = max(base_emissions - indirect_emissions, 0.0)
            ccs_emissions = (1 - ccs_level / 100) * direct_emissions + indirect_emissions
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


def _build_bus_multipliers(n: pypsa.Network, snakemake):
    """Resolve the ReEDS county overnight-capex multipliers onto this network's buses.

    Returns ``None`` when the feature is disabled, which makes every downstream cost
    expression reduce to its unmodified form.
    """
    cfg = getattr(snakemake.params, "cost_multipliers", None) or {}
    if not cfg.get("enable", False):
        return None

    reg_cap_cost_diff = snakemake.input.get("reg_cap_cost_diff", None)
    if not reg_cap_cost_diff:
        logger.warning(
            "cost_multipliers.enable is true but no reg_cap_cost_diff input was provided; "
            "skipping regional overnight-capex multipliers.",
        )
        return None

    county_table = county_multiplier_table(
        load_reg_cap_cost_diff(reg_cap_cost_diff),
        cfg.get("column_to_tech", None),
    )
    ac_buses = n.buses[n.buses.carrier == "AC"]
    return bus_multiplier_table(ac_buses, county_table, snakemake.input.county_shapes)


def _regional_capital_cost(costs, carrier, buses_i, bus_multipliers):
    """Per-bus capital cost for an ATB carrier, scaling overnight capex only.

    Returns the uniform ``annualized_capex_fom`` scalar unchanged whenever the
    feature is off, the carrier carries no ReEDS multiplier column, or the cost
    table has no usable overnight breakdown to scale.
    """
    base = costs.at[carrier, "annualized_capex_fom"]
    multiplier = carrier_multiplier(bus_multipliers, carrier, buses_i)
    if np.isscalar(multiplier):
        return base
    overnight = costs.at[carrier, "capex_overnight_per_kw"] if "capex_overnight_per_kw" in costs.columns else np.nan
    if not np.isfinite(overnight):
        logger.warning(
            "No finite capex_overnight_per_kw for %r; skipping its regional multiplier.",
            carrier,
        )
        return base
    annuity_factor = calculate_annuity(
        costs.at[carrier, "cost_recovery_period_years"],
        costs.at[carrier, "wacc_real"],
    )
    return overnight_delta_capital_cost(base, overnight, annuity_factor, multiplier)


def attach_storageunits(n, costs, elec_opts, investment_year, bus_multipliers=None):
    carriers = elec_opts["extendable_carriers"]["StorageUnit"]
    carriers = [k for k in carriers if "battery_storage" in k]

    buses_i = n.buses.index[n.buses.carrier == "AC"]

    add_missing_carriers(n, carriers)
    add_co2_emissions(n, costs, carriers)
    for carrier in carriers:
        max_hours = int(carrier.split("hr_")[0])
        roundtrip_correction = 0.5 if "battery" in carrier else 1
        capital_cost = _regional_capital_cost(costs, carrier, buses_i, bus_multipliers)

        n.madd(
            "StorageUnit",
            buses_i,
            suffix=f" {carrier}_{investment_year}",
            bus=buses_i,
            carrier=carrier,
            p_nom_extendable=True,
            capital_cost=capital_cost,
            marginal_cost=0,  # costs.at[carrier, "marginal_cost"], # TODO: FIX THIS ISSUE IN BUILD_COST_DATA
            efficiency_store=costs.at[carrier, "efficiency"] ** roundtrip_correction,
            efficiency_dispatch=costs.at[carrier, "efficiency"] ** roundtrip_correction,
            max_hours=max_hours / (costs.at[carrier, "efficiency"] ** roundtrip_correction),
            cyclic_state_of_charge=True,
            build_year=investment_year,
            lifetime=costs.at[carrier, "cost_recovery_period_years"],
        )


def attach_tes_storageunits(n: pypsa.Network, sector_costs_path: str, bus_multipliers=None):
    """Attach reference TES buses, stores, chargers, and dischargers."""
    buses_i = n.buses.index[n.buses.carrier == "AC"]
    if buses_i.empty:
        return

    costs = SectorCosts(sector_costs_path)
    crp = costs.value("TES", "crp")
    # The regional multiplier scales the annuity term only, leaving the FOM share
    # untouched. All three cost items share it, and every madd below is keyed on
    # buses_i, so a per-bus array aligns positionally.
    multiplier = carrier_multiplier(bus_multipliers, "tes", buses_i)
    annuity = calculate_annuity(crp, costs.wacc_real("TES")) * multiplier + costs.value("TES", "FOM") / 100
    energy_cost = annuity * costs.value("TES", "energy_cost")
    charge_cost = annuity * costs.value("TES", "charge_cost")
    discharge_cost = annuity * costs.value("TES", "discharge_cost")
    lifetime = costs.lifetime("TES")
    build_year = int(n.investment_periods[0])

    add_missing_carriers(n, ["tes"])
    n.carriers.loc["tes", "co2_emissions"] = 0.0
    buses = n.buses.loc[buses_i]
    n.madd(
        "Bus",
        names=buses_i,
        suffix=" tes",
        x=buses.x.to_numpy(),
        y=buses.y.to_numpy(),
        carrier="tes",
        unit="MWh_th",
    )
    n.madd(
        "Store",
        names=buses_i + " tes",
        bus=buses_i + " tes",
        carrier="tes",
        e_nom_extendable=True,
        e_cyclic=True,
        standing_loss=costs.value("TES", "loss"),
        capital_cost=energy_cost,
        build_year=build_year,
        lifetime=lifetime,
    )
    n.madd(
        "Link",
        names=buses_i + " tes charger",
        bus0=buses_i,
        bus1=buses_i + " tes",
        carrier="tes",
        p_nom_extendable=True,
        capital_cost=charge_cost,
        efficiency=costs.value("TES", "charge_efficiency"),
        build_year=build_year,
        lifetime=lifetime,
    )
    n.madd(
        "Link",
        names=buses_i + " tes discharger",
        bus0=buses_i + " tes",
        bus1=buses_i,
        carrier="tes",
        p_nom_extendable=True,
        capital_cost=discharge_cost,
        efficiency=costs.value("TES", "discharge_efficiency"),
        marginal_cost=costs.value("TES", "VOM"),
        build_year=build_year,
        lifetime=lifetime,
    )


def flexible_electrolysis_accounting_region(config: dict) -> str:
    """Return the validated hydrogen balance accounting level.

    ``h2ptcreg`` (default) balances hydrogen production per 45V hydrogen PTC
    region; ``nation`` balances it once over the whole modelled system.
    """
    accounting_region = config.get("accounting_region", "h2ptcreg")
    if accounting_region not in ("h2ptcreg", "nation"):
        raise ValueError(
            "flexible_electrolysis 'accounting_region' must be 'h2ptcreg' or 'nation'; "
            f"got {accounting_region!r}.",
        )
    return accounting_region


def attach_flexible_electrolysis(
    n: pypsa.Network,
    config: dict,
    sector_costs_path: str,
    bus_multipliers=None,
):
    """Attach electrolysis links at every AC bus feeding a pure accounting H2 bus.

    With ``accounting_region: h2ptcreg`` there is one accounting H2 bus per 45V
    hydrogen PTC region (``h2ptcreg``, a bus attribute assigned in
    ``build_base_network``), and each AC bus feeds the bus of the region it sits
    in. With ``accounting_region: nation`` every electrolysis link instead feeds a
    single national accounting bus, so hydrogen production is only balanced in
    total. Either way, buses without an ``h2ptcreg`` (non-US) get no link. The
    links have ``efficiency = 0``, so nothing is
    injected into the H2 buses and their nodal
    balance holds trivially without any sink component. The annual hydrogen
    production is instead imposed in ``solve_network`` as a per-accounting-region
    constraint on the link electricity withdrawal (``p0``) times the conversion
    efficiency implied by ``h2 electrolysis`` / ``electricity-input`` in
    ``simple_sector_costs.csv``. The value is validated here so a broken cost file
    fails at build time.
    """
    if not config.get("enable", False):
        return

    accounting_region = flexible_electrolysis_accounting_region(config)

    if n.buses.index.str.endswith(FLEXIBLE_ELECTROLYSIS_BUS_SUFFIX).any():
        logger.info("Flexible electrolysis accounting buses already attached. Skipping duplicate attachment.")
        return

    costs = SectorCosts(sector_costs_path)
    electricity_input = costs.value("h2 electrolysis", "electricity-input")
    if electricity_input <= 0.0:
        raise ValueError(
            "'h2 electrolysis' 'electricity-input' in simple_sector_costs.csv must be positive; "
            f"got {electricity_input}.",
        )

    ac_buses = n.buses.index[n.buses.carrier == "AC"]
    if ac_buses.empty:
        logger.warning(
            "Flexible electrolysis is enabled but no AC bus exists; skipping electrolysis attachment.",
        )
        return

    if "h2ptcreg" not in n.buses.columns:
        raise ValueError(
            "Flexible electrolysis is enabled but the network has no 'h2ptcreg' bus attribute. "
            "It is assigned in build_base_network from repo_data/ReEDS_Constraints/membership.csv.",
        )

    # Non-US buses (Canadian/Mexican ReEDS zones) have no h2ptcreg and get no
    # electrolysis link. A netCDF round trip turns their missing value into "".
    bus_regions = n.buses.loc[ac_buses, "h2ptcreg"].replace("", np.nan).dropna()
    if bus_regions.empty:
        logger.warning(
            "Flexible electrolysis is enabled but no AC bus has an h2ptcreg region; "
            "skipping electrolysis attachment.",
        )
        return
    if len(bus_regions) < len(ac_buses):
        logger.info(
            "No h2ptcreg for %d AC bus(es) (non-US or unmapped location); they get no electrolysis link.",
            len(ac_buses) - len(bus_regions),
        )

    if accounting_region == "nation":
        # One accounting bus for the whole system: the h2ptcreg attribute is still
        # what selects the eligible (US) buses, but it no longer splits them.
        bus_regions = pd.Series("nation", index=bus_regions.index)

    add_missing_carriers(n, ["H2", "electrolysis"])
    if "co2_emissions" not in n.carriers.columns:
        n.carriers["co2_emissions"] = 0.0
    n.carriers.loc[["H2", "electrolysis"], "co2_emissions"] = 0.0

    regions = pd.Index(sorted(bus_regions.unique()))
    region_buses = pd.Index(regions + FLEXIBLE_ELECTROLYSIS_BUS_SUFFIX)

    # Place each regional accounting bus at the centroid of the AC buses feeding it.
    bus_kwargs = {}
    for coord in ("x", "y"):
        if coord in n.buses.columns:
            centroid = n.buses.loc[bus_regions.index, coord].groupby(bus_regions).mean()
            bus_kwargs[coord] = centroid.reindex(regions).astype(float).values

    n.madd(
        "Bus",
        region_buses,
        carrier="H2",
        **bus_kwargs,
    )

    if "country" in n.buses.columns:
        countries = n.buses.loc[bus_regions.index, "country"].groupby(bus_regions).first()
        n.buses.loc[region_buses, "country"] = countries.reindex(regions).values

    n.madd(
        "Link",
        bus_regions.index,
        suffix=FLEXIBLE_ELECTROLYSIS_LINK_SUFFIX,
        bus0=bus_regions.index,
        bus1=(bus_regions + FLEXIBLE_ELECTROLYSIS_BUS_SUFFIX).values,
        carrier="electrolysis",
        p_nom=0,
        p_nom_extendable=True,
        p_nom_max=1e5,
        # Zero efficiency: no hydrogen flows into the accounting bus, so its energy
        # balance is satisfied automatically. Hydrogen output is accounted for in
        # solve_network from p0 and the configured electricity input per hydrogen.
        efficiency=0.0,
        capital_cost=costs.annualized(
            "h2 electrolysis",
            overnight_multiplier=carrier_multiplier(bus_multipliers, "electrolysis", bus_regions.index),
        ),
        lifetime=costs.lifetime("h2 electrolysis"),
        build_year=int(n.investment_periods[0]),
    )
    logger.info(
        "Attached %d flexible electrolysis links across %d %s accounting region(s): %s.",
        len(bus_regions),
        len(regions),
        accounting_region,
        ", ".join(regions),
    )


def apply_gas_fuel_price(n: pypsa.Network, gas_price: float) -> None:
    """Apply a single configured natural gas price ($/MWh_th) to all gas-fueled generators (OCGT, CCGT, and their -CCS variants)."""
    gas_carriers = [carrier for carrier in n.generators.carrier.unique() if carrier.split("-")[0] in ("OCGT", "CCGT")]
    gens = n.generators[n.generators.carrier.isin(gas_carriers)]
    if gens.empty:
        return

    efficiency = n.generators.loc[gens.index, "efficiency"].replace(0, np.nan)
    if efficiency.isna().any():
        raise ValueError("Gas generator efficiency must be positive.")
    vom = n.generators.loc[gens.index].get(
        "vom_cost",
        pd.Series(0.0, index=gens.index),
    ).fillna(0.0)
    n.generators.loc[gens.index, "marginal_cost"] = vom + gas_price / efficiency


def remove_gas_generators(n: pypsa.Network) -> None:
    """Drop all gas-fueled generators (OCGT, CCGT, and their -CCS variants) for the 100VRE decarbonization scenario."""
    gas_carriers = [carrier for carrier in n.generators.carrier.unique() if carrier.split("-")[0] in ("OCGT", "CCGT")]
    gens = n.generators[n.generators.carrier.isin(gas_carriers)]
    if not gens.empty:
        logger.info("100VRE decarbonization: removing %s gas generators (%s).", len(gens), sorted(gas_carriers))
        n.mremove("Generator", gens.index)


def drop_coal_generators(n: pypsa.Network) -> None:
    """
    Drop every coal generator (``coal`` and its ``-CCS`` variants) from the network.

    Coal is listed in ``electricity.conventional_carriers`` and
    ``electricity.extendable_carriers`` only so that the existing coal fleet is
    attached in add_electricity.py and can define which buses are eligible for
    new nuclear (see ``NUCLEAR_SITING_BASE_CARRIERS``). Once that bus set has
    been computed, no coal capacity should remain: this must therefore be called
    after ``carrier_new_build_buses`` and before any coal generator could reach
    the optimisation.

    Note the contrast with OCGT, the other carrier in
    ``NO_NEW_BUILD_BASE_CARRIERS``: OCGT loses only its new-build option, its
    existing fleet stays in the network.
    """
    coal_i = n.generators.index[n.generators.carrier.str.split("-").str[0] == COAL_BASE_CARRIER]
    if coal_i.empty:
        return
    gens = n.generators.loc[coal_i]
    logger.info(
        "Removing %s coal generators (%s), total existing p_nom %.1f MW -- coal is only "
        "used to define the new-nuclear siting set.",
        len(coal_i),
        sorted(gens.carrier.unique()),
        gens.p_nom.fillna(0).sum(),
    )
    n.mremove("Generator", coal_i)


def new_build_carriers(extendable_carriers: Iterable[str]) -> list[str]:
    """
    Filter ``electricity.extendable_carriers["Generator"]`` down to the carriers this script attaches new-build generators for.

    Dropped here:

    - ``RESOURCE_PROFILE_CARRIERS`` (wind/solar), which already have a single
      resource-capped extendable generator per bus from add_electricity.py --
      adding a second, uncapped one would ignore the real resource limit.
    - ``NO_NEW_BUILD_BASE_CARRIERS`` (coal, OCGT), matched on the base carrier so
      the ``-CCS`` variants go with them.
    """
    return sorted(
        c
        for c in set(extendable_carriers)
        if c not in RESOURCE_PROFILE_CARRIERS and c.split("-")[0] not in NO_NEW_BUILD_BASE_CARRIERS
    )


def remove_negligible_potential_generators(n: pypsa.Network, threshold: float = 1.0) -> None:
    """Drop generators whose maximum capacity is below `threshold` MW (negligible buildable potential)."""
    gens = n.generators[n.generators.p_nom_max < threshold]
    if gens.empty:
        return
    with_existing = gens[gens.p_nom > 0]
    if not with_existing.empty:
        logger.warning(
            "Removing %s generators with p_nom_max < %s MW that still carry existing capacity (total p_nom %.1f MW).",
            len(with_existing),
            threshold,
            with_existing.p_nom.sum(),
        )
    logger.info(
        "Removing %s generators with p_nom_max < %s MW (%s).",
        len(gens),
        threshold,
        sorted(gens.carrier.unique()),
    )
    n.mremove("Generator", gens.index)


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


def attach_stores(n, costs, elec_opts, investment_year):
    carriers = elec_opts["extendable_carriers"]["Store"]

    add_missing_carriers(n, carriers)
    add_co2_emissions(n, costs, carriers)

    buses_i = n.buses.index[n.buses.carrier == "AC"]
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


def freeze_existing_generators(
    n: pypsa.Network,
    costs: pd.DataFrame,
    carriers: list[str] | None = None,
    economic: bool = True,
):
    """
    Converts today's fleet of extendable generators into fixed-capacity
    "existing" generators to support economic or technical retirement. Does
    NOT create a new-build capacity option: new-build generators for these
    carriers are attached separately by attach_new_generators, using
    unified cost data rather than a copy of the existing generator.

    Wind/solar are never passed through this function -- their existing
    capacity and buildable resource potential are already represented as a
    single extendable generator per bus by attach_wind_and_solar /
    attach_existing_renewable_capacities in add_electricity.py, with nothing
    left here to freeze into a separate row.

    Specifically this function does the following:
    1. Renames matching generators with an " existing" suffix. For example,
    an extendable "CCGT" generator becomes "CCGT existing".
    2. Capital costs of existing generators are replaced with fixed costs
    3. p_nom_max of existing generators is set to p_nom
    4. p_nom_min of existing generators is set to zero

    Arguments:
    n: pypsa.Network,
    costs: pd.DataFrame,
    carriers: List[str]
        List of generator carriers to apply retirement to.
    economic: bool
        If True, enable economic retirement, else only allow lifetime
        retirement.
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
            else costs.at[row["carrier"], "opex_fixed_per_kw"] * 1e3
        ),
        axis=1,
    )

    # Rename retiring generators to include "existing" suffix
    n.generators.index = n.generators.apply(
        lambda row: row.name if row.name not in (retirement_gens.index) else row.name + " existing",
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

    # time dependent factors renamed to match the " existing" suffix above
    marginal_cost_t = n.generators_t["marginal_cost"][
        [x for x in retirement_gens.index if x in n.generators_t.marginal_cost.columns]
    ]
    marginal_cost_t = marginal_cost_t.rename(
        columns={x: f"{x} existing" for x in marginal_cost_t.columns},
    )
    n.generators_t["marginal_cost"] = n.generators_t["marginal_cost"].join(
        marginal_cost_t,
    )

    p_max_pu_t = n.generators_t["p_max_pu"][
        [x for x in retirement_gens.index if x in n.generators_t["p_max_pu"].columns]
    ]
    p_max_pu_t = p_max_pu_t.rename(
        columns={x: f"{x} existing" for x in p_max_pu_t.columns},
    )
    n.generators_t["p_max_pu"] = n.generators_t["p_max_pu"].join(p_max_pu_t)


def attach_new_generators(n, costs, carriers, investment_year, carrier_buses=None, bus_multipliers=None):
    """
    Attaches new-build generators using unified, per-investment-year cost
    data (capital_cost, marginal_cost, efficiency, lifetime all come straight
    from `costs`) -- never copied or inherited from an existing generator.

    Bus eligibility per carrier is given by `carrier_buses` (see
    `carrier_new_build_buses`); a carrier missing from that dict defaults to
    all AC buses. p_max_pu profile are still informed by
    each bus's existing fleet of the same (or, for a "-CCS" carrier, its
    base) carrier where available, purely to seed reasonable operating
    characteristics -- this does not affect cost.

    Arguments:
    n: pypsa.Network,
    costs: pd.DataFrame,
    carriers: List[str]
        List of carriers to add new-build generators for
    investment_year: int
        Year of investment
    carrier_buses: dict[str, pd.Index] | None
        Per-carrier override for which buses to add the new generators to.
        Carriers not present in this dict default to all AC buses.
    """
    if not carriers:
        return

    add_missing_carriers(n, carriers)
    add_co2_emissions(n, costs, carriers)
    min_years = snakemake.config["costs"].get("min_year")
    all_buses_i = n.buses.index[n.buses.carrier == "AC"]
    if all_buses_i.empty:
        return
    carrier_buses = carrier_buses or {}
    for carrier in carriers:
        buses_i = carrier_buses.get(carrier, all_buses_i)
        if buses_i.empty:
            continue
        p_max_pu_t = None
        if min_years and min_years.get(carrier, 0) > investment_year:
            continue
        reference_carriers = [carrier]
        if "CCS" in carrier:
            base_carrier = carrier.split("-")[0]
            if base_carrier != carrier:
                reference_carriers.append(base_carrier)

        # build_year < investment_year (not <=) so a carrier processed earlier
        # in this same loop (e.g. CCGT before CCGT-95CCS) isn't picked up as
        # its own reference -- only generators that already existed before
        # this call count as "existing".
        existing_gens = n.generators.iloc[0:0].copy()
        for reference_carrier in reference_carriers:
            existing_gens = n.generators[
                (
                    (n.generators.carrier == reference_carrier)
                    & (n.generators.build_year < investment_year)
                )
            ].copy()
            if not existing_gens.empty:
                break

        profile_gens = existing_gens

        if not profile_gens.empty:
            p_max_pu_dense = n.get_switchable_as_dense("Generator", "p_max_pu")
            existing_cols = [x for x in profile_gens.index if x in p_max_pu_dense.columns]
            if existing_cols:
                p_max_pu_dense = p_max_pu_dense[existing_cols]
                existing_meta = n.generators.loc[existing_cols]
                weights_all = existing_meta.p_nom.fillna(0).clip(lower=0)
                if weights_all.sum() > 0:
                    p_max_pu_all = p_max_pu_dense.mul(weights_all, axis=1).sum(axis=1) / weights_all.sum()
                else:
                    p_max_pu_all = p_max_pu_dense.mean(axis=1)

                bus_to_existing_cols = existing_meta.groupby("bus").groups
                profiles = {}
                for bus in buses_i:
                    bus_existing_cols = bus_to_existing_cols.get(bus, [])
                    if len(bus_existing_cols) > 0:
                        bus_weights = weights_all.reindex(bus_existing_cols).fillna(0)
                        if bus_weights.sum() > 0:
                            bus_profile = (
                                p_max_pu_dense[bus_existing_cols].mul(bus_weights, axis=1).sum(axis=1)
                                / bus_weights.sum()
                            )
                        else:
                            bus_profile = p_max_pu_dense[bus_existing_cols].mean(axis=1)
                    else:
                        bus_profile = p_max_pu_all
                    profiles[f"{bus} {carrier}_{investment_year}"] = bus_profile
                p_max_pu_t = pd.DataFrame(profiles, index=p_max_pu_dense.index)

        n.madd(
            "Generator",
            buses_i,
            suffix=f" {carrier}_{investment_year}",
            bus=buses_i,
            carrier=carrier,
            p_nom_extendable=True,
            capital_cost=_regional_capital_cost(costs, carrier, buses_i, bus_multipliers),
            marginal_cost=costs.at[carrier, "marginal_cost"],
            vom_cost=costs.at[carrier, "opex_variable_per_mwh"],
            efficiency=costs.at[carrier, "efficiency"],
            build_year=investment_year,
            lifetime=costs.at[carrier, "lifetime"],
            p_max_pu=1,
        )

        if p_max_pu_t is not None and not p_max_pu_t.empty:
            n.generators_t.p_max_pu = pd.concat([n.generators_t.p_max_pu, p_max_pu_t], axis=1)


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


def trim_network(n, trim_topology):
    """
    Trim_network splits the network into two parts:
        - The internal network, which is the network within the specified zones.
        - The external network, which is the network outside the specified zones.

    The internal network is retained and unchanged. While the external network components are removed. The external buses which are directly connected to the internal network are aggregated to the `nerc_reg` value of their buses.
    The only generators kept are the OCGTs at the external buses, which are set to non-extendable.

    The external OCGT generators are set to the carrier name `imports` and retain the same emissions intensity.

    """
    retain_zones = trim_topology["zone"]
    internal_buses = get_region_buses(n, retain_zones)
    if internal_buses.empty:
        logger.warning("No internal buses found, skipping trim_network")
        return None

    # Get all lines and links connected to internal buses
    retain_lines = n.lines[n.lines.bus0.isin(internal_buses.index) | n.lines.bus1.isin(internal_buses.index)]
    retain_links = n.links[n.links.bus0.isin(internal_buses.index) | n.links.bus1.isin(internal_buses.index)]

    # Find buses to remove (those not connected to internal network)
    buses_to_remove = n.buses[
        ~n.buses.index.isin(retain_lines.bus0)
        & ~n.buses.index.isin(retain_lines.bus1)
        & ~n.buses.index.isin(retain_links.bus0)
        & ~n.buses.index.isin(retain_links.bus1)
    ]

    # Find external buses to keep (connected to internal network but not internal)
    external_buses_to_keep = n.buses.loc[
        ~n.buses.index.isin(buses_to_remove.index) & ~n.buses.index.isin(internal_buses.index)
    ]

    # Remove components at buses that are being removed
    for c in n.one_port_components:
        component = n.df(c)
        rm = component[component.bus.isin(buses_to_remove.index)]
        if not rm.empty:
            n.mremove(c, rm.index)

    # Remove lines and links at buses being removed
    for c in ["Line", "Link"]:
        component = n.df(c)
        rm = component[~component.bus0.isin(internal_buses.index) & ~component.bus1.isin(internal_buses.index)]
        if not rm.empty:
            n.mremove(c, rm.index)

    # Remove the buses
    n.mremove("Bus", buses_to_remove.index)

    # Get OCGT generators and calculate average marginal cost
    ocgt_gens = n.generators[n.generators.carrier == "OCGT"]
    avg_marginal_cost = get_as_dense(n, "Generator", "marginal_cost").loc[:, ocgt_gens.index].mean().mean()
    n.add("Carrier", "imports", co2_emissions=0.428, nice_name="imports")

    # remove existing oneport components at bus
    for c in n.one_port_components:
        component = n.df(c)
        rm = component[component.bus.isin(external_buses_to_keep.index)]
        if not rm.empty:
            logger.info(f"Removing {c} at external buses {external_buses_to_keep.index} with components {rm.index}")
            n.mremove(c, rm.index)

    # Handle external buses and their generators
    for bus in external_buses_to_keep.index:
        # Create new import generator
        bus_name = n.buses.loc[bus].name
        n.add(
            "Generator",
            f"import_{bus_name}",
            bus=bus,
            carrier="imports",
            p_nom=1e4,
            p_nom_extendable=False,
            marginal_cost=avg_marginal_cost,
            efficiency=1,
            build_year=n.investment_periods[0],
            lifetime=100,
        )

        # Change location names of external buses, append imports to the ['reeds_state', 'reeds_zone', 'reeds_ba', 'interconnect', 'trans_reg', 'trans_grp']
        n.buses.loc[bus, "reeds_state"] = f"imports_{n.buses.loc[bus, 'reeds_state']}"
        n.buses.loc[bus, "reeds_zone"] = f"imports_{n.buses.loc[bus, 'reeds_zone']}"
        n.buses.loc[bus, "reeds_ba"] = f"imports_{n.buses.loc[bus, 'reeds_ba']}"
        n.buses.loc[bus, "interconnect"] = f"imports_{n.buses.loc[bus, 'interconnect']}"
        n.buses.loc[bus, "trans_reg"] = f"imports_{n.buses.loc[bus, 'trans_reg']}"
        n.buses.loc[bus, "trans_grp"] = f"imports_{n.buses.loc[bus, 'trans_grp']}"

        # Set all links and lines connected to the bus as non-extendable
        for c in ["Line", "Link"]:
            attr_name = "p_nom_extendable" if c == "Link" else "s_nom_extendable"
            component = n.df(c)
            mask = (component.bus0 == bus) | (component.bus1 == bus)
            if mask.any():
                component.loc[mask, attr_name] = False
                n.df(c).update(component)

        # Remove the links which have "exp" in the name and are connected to the external buses
        links_to_remove = n.links[
            n.links.index.str.contains("exp")
            & (n.links.bus0.isin(external_buses_to_keep.index) | n.links.bus1.isin(external_buses_to_keep.index))
        ]
        n.mremove("Link", links_to_remove.index)

    # Update network topology
    n.determine_network_topology()


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_extra_components",
            case="test",
        )
    configure_logging(snakemake)
    set_case_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

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

    # County-level ReEDS overnight-capex multipliers, resolved onto this network's
    # buses. Buses here are finer than county resolution and no longer carry a
    # `county` column (simplify_network drops it), so they are matched to the
    # geographically nearest county the ReEDS table covers. Applied where each
    # component is created, so only the overnight share is ever scaled.
    bus_multipliers = _build_bus_multipliers(n, snakemake)

    if any("PHS" in s for s in elec_config["extendable_carriers"]["StorageUnit"]):
        attach_phs_storageunits(n, elec_config, costs_dict[n.investment_periods[0]])

    if snakemake.params.retirement == "economic":
        economic_retirement_gens = set(elec_config.get("conventional_carriers", None))
        freeze_existing_generators(
            n,
            costs_dict[n.investment_periods[0]],
            economic_retirement_gens,
            economic=True,
        )
    # Wind/solar are intentionally not passed through freeze_existing_generators:
    # their existing capacity + buildable resource potential are already a single
    # extendable generator per bus from add_electricity.py (see
    # attach_existing_renewable_capacities), so there's nothing to freeze here.

    # Bus eligibility for new-build capacity: nuclear at every AC bus;
    # CCGT/CCGT-CCS on the union of existing gas buses; everything
    # else restricted to buses that already have that carrier installed (see
    # carrier_new_build_buses).
    all_ac_buses_i = n.buses.index[n.buses.carrier == "AC"]
    gas_union_buses_i = pd.Index(
        n.generators.loc[n.generators.carrier.isin(NEW_GAS_COST_CARRIERS), "bus"].unique(),
    )

    # onwind/offwind_floating/solar are excluded: attach_wind_and_solar and
    # attach_existing_renewable_capacities (add_electricity.py) already give them a
    # single extendable generator per bus -- existing capacity as p_nom/p_nom_min,
    # resource potential as p_nom_max, new-build cost -- so there's nothing to add here.
    # Coal and OCGT are excluded as well (NO_NEW_BUILD_BASE_CARRIERS): no new capacity
    # is built for either. Coal additionally leaves the network entirely below; the
    # existing OCGT fleet stays and keeps dispatching.
    new_carriers = new_build_carriers(elec_config["extendable_carriers"].get("Generator", []))
    new_generator_carrier_buses = {
        carrier: carrier_new_build_buses(n, carrier, all_ac_buses_i, gas_union_buses_i) for carrier in new_carriers
    }

    # The new-build bus sets above are the last consumer of the existing coal fleet,
    # so coal leaves the network here -- before any generator is attached for the
    # investment periods and well before prepare_network/solve_network see it. The
    # existing OCGT fleet is deliberately kept; it only loses its new-build option.
    drop_coal_generators(n)

    for investment_year in n.investment_periods:
        costs = costs_dict[investment_year]
        attach_storageunits(n, costs, elec_config, investment_year, bus_multipliers=bus_multipliers)
        attach_new_generators(
            n,
            costs,
            new_carriers,
            investment_year,
            carrier_buses=new_generator_carrier_buses,
            bus_multipliers=bus_multipliers,
        )
        # attach_stores(n, costs, elec_config, investment_year)

    attach_flexible_electrolysis(
        n,
        snakemake.config.get("flexible_electrolysis", {}),
        snakemake.input.sector_costs,
        bus_multipliers=bus_multipliers,
    )

    if snakemake.params.add_extendable_tes:
        attach_tes_storageunits(n, snakemake.input.sector_costs, bus_multipliers=bus_multipliers)

    if snakemake.config.get("scenario", {}).get("decarbonization") == "100VRE":
        remove_gas_generators(n)

    apply_gas_fuel_price(n, snakemake.params.gas_fuel_price)

    apply_max_annual_growth_rate(n, snakemake.config["costs"]["max_growth"])
    add_nice_carrier_names(n, snakemake.config)
    add_co2_emissions(n, costs_dict[n.investment_periods[0]], n.carriers.index)

    trim_network_config = snakemake.params.trim_network
    if snakemake.params.trim_network:
        trim_network(n, trim_network_config)

    remove_negligible_potential_generators(n)

    n.consistency_check()
    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
