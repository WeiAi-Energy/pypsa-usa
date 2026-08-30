import logging  # noqa: D100
import numpy as np
import pandas as pd
import pypsa
from opts._helpers import (
    ceil_precision,
    filter_components,
    floor_precision,
    get_model_horizon,
    get_region_buses,
)
from pypsa.descriptors import get_switchable_as_dense as get_as_dense

logger = logging.getLogger(__name__)

RPS_CARRIERS = [
    "onwind",
    "offwind",
    "offwind_floating",
    "solar",
    "hydro",
    "geothermal",
    "biomass",
    "EGS",
]
CES_CARRIERS = [*RPS_CARRIERS, "nuclear", "SMR", "hydrogen_ct", "CCGT-95CCS", "CCGT-99CCS", "Coal-95CCS"]
RPS_DENOMINATOR_EXCLUDED_CARRIERS = {"load"}


def read_technology_capacity_targets(config):
    """Read the configured TCT table, returning an empty frame when disabled."""
    path = config.get("electricity", {}).get("technology_capacity_targets")
    columns = ["name", "planning_horizon", "region", "carrier", "min", "max"]
    return pd.read_csv(path) if path else pd.DataFrame(columns=columns)


def option_enabled(opts, option):
    """Return whether a dash-delimited option appears in a string or list."""
    opts = [opts] if isinstance(opts, str) else (opts or [])
    return option in [token for item in opts for token in str(item).split("-")]


def _as_clean_list(value):
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _planning_horizon_active_at_first_period(planning_horizon, periods):
    return planning_horizon == "all" or (bool(periods) and int(planning_horizon) <= int(min(periods)))


def _planning_horizon_relevant(planning_horizon, periods):
    return planning_horizon == "all" or (bool(periods) and int(planning_horizon) <= int(max(periods)))


def _target_components(n, component_type, target):
    component = n.df(component_type)
    if component.empty:
        return component.iloc[0:0]
    region_buses = get_region_buses(n, _as_clean_list(target.region))
    bus_col = "bus1" if component_type == "Link" else "bus"
    return component.loc[
        component.carrier.isin(_as_clean_list(target.carrier))
        & component[bus_col].isin(region_buses.index)
    ]


def _component_existing_mask(n, components):
    p_nom = pd.to_numeric(components.p_nom, errors="coerce").fillna(0)
    build_year = pd.to_numeric(components.get("build_year", np.nan), errors="coerce")
    first_period = min(n.investment_periods) if len(n.investment_periods) else np.inf
    name_existing = components.index.astype(str).str.contains("existing", case=False, regex=False)
    commissioned = build_year.isna() | (build_year <= first_period)
    return pd.Series(name_existing | ((p_nom > 0) & commissioned), index=components.index)


def remove_tct_blocked_components(
    n,
    config,
    components=("Generator", "StorageUnit", "Link"),
    target_names=None,
    max_values=("existing", "0", 0, 0.0),
    active_at_first_period_only=True,
    remove_existing_for_zero=True,
    remove_non_existing_for_zero=True,
):
    """Remove assets excluded by TCT ``max=existing`` or ``max=0`` rows."""
    targets = read_technology_capacity_targets(config)
    periods = list(n.investment_periods)
    names = set(target_names) if target_names is not None else None
    removed = {component: [] for component in components}
    for _, target in targets.iterrows():
        if names is not None and target["name"] not in names:
            continue
        if target["max"] not in max_values:
            continue
        # A max=existing row permanently forbids new capacity for its carrier/region, so
        # it's safe to remove candidates as soon as the row is relevant anywhere in the
        # model horizon. max=0 rows can alternate with non-zero caps across planning
        # horizons for the same target, so those stay gated to the first period only.
        horizon_check = _planning_horizon_relevant if target["max"] == "existing" else _planning_horizon_active_at_first_period
        if active_at_first_period_only and not horizon_check(target.planning_horizon, periods):
            continue
        for component_type in components:
            matches = _target_components(n, component_type, target)
            if matches.empty:
                continue
            existing = _component_existing_mask(n, matches)
            if target["max"] == "existing":
                matches = matches.loc[~existing]
            else:
                select = pd.Series(False, index=matches.index)
                if remove_existing_for_zero:
                    select |= existing
                if remove_non_existing_for_zero:
                    select |= ~existing
                matches = matches.loc[select]
            if not matches.empty:
                n.mremove(component_type, matches.index)
                removed[component_type].extend(matches.index.astype(str))
    return {component: names for component, names in removed.items() if names}


def check_tct_min_capacity_feasibility(n, config):
    """Reject minimum TCT rows that exceed finite installed plus buildable capacity."""
    periods = list(n.investment_periods)
    infeasible = []
    for _, target in read_technology_capacity_targets(config).iterrows():
        if pd.isna(target["min"]) or target["min"] == "existing":
            continue
        if not _planning_horizon_relevant(target.planning_horizon, periods):
            continue
        existing = potential = 0.0
        infinite = False
        for component_type in ("Generator", "StorageUnit", "Link"):
            matches = _target_components(n, component_type, target)
            if matches.empty:
                continue
            extendable = matches.p_nom_extendable.fillna(False).astype(bool)
            existing += pd.to_numeric(matches.p_nom[~extendable], errors="coerce").fillna(0).sum()
            p_nom_max = pd.to_numeric(matches.p_nom_max[extendable], errors="coerce")
            infinite |= p_nom_max.isna().any() or np.isinf(p_nom_max).any()
            potential += p_nom_max.replace([np.inf, -np.inf], np.nan).fillna(0).sum()
        available = np.inf if infinite else existing + potential
        if available < float(target["min"]):
            infeasible.append(f"{target['name']}: min {target['min']} MW > available {available} MW")
    if infeasible:
        raise ValueError("Infeasible TCT minimum capacity target(s): " + "; ".join(infeasible))


def add_technology_capacity_target_constraints(n, config):
    """Add reference-style TCT constraints for generators, storage units and links."""
    targets = read_technology_capacity_targets(config)
    if targets.empty:
        return
    check_tct_min_capacity_feasibility(n, config)
    model_horizon = get_model_horizon(n.model)
    added = 0
    for _, target in targets.iterrows():
        planning_horizon = target.planning_horizon
        if planning_horizon != "all" and int(planning_horizon) > max(model_horizon):
            continue
        region_buses = get_region_buses(n, _as_clean_list(target.region))
        if region_buses.empty:
            continue
        carriers = _as_clean_list(target.carrier)
        expressions = []
        existing_capacity = 0.0
        blocked_new_build = []
        for component_type, variable in (
            ("Generator", "Generator-p_nom"),
            ("StorageUnit", "StorageUnit-p_nom"),
            ("Link", "Link-p_nom"),
        ):
            extendable = filter_components(
                n,
                component_type,
                planning_horizon,
                carriers,
                region_buses.index,
                True,
            )
            existing = filter_components(
                n,
                component_type,
                planning_horizon,
                carriers,
                region_buses.index,
                False,
            )
            existing_capacity += existing.p_nom.sum()
            if not extendable.empty and variable in n.model.variables:
                expressions.append(n.model[variable].loc[extendable.index].sum())
                # Extendable assets already at (or past) their in-service capacity — e.g.
                # existing plants kept extendable only to allow retirement — aren't the
                # unconstrained new-build candidates this warning is meant to catch.
                if target["max"] == "existing":
                    new_build = extendable.loc[~_component_existing_mask(n, extendable)]
                    if not new_build.empty:
                        blocked_new_build.append(new_build)
        if not expressions:
            continue
        if target["max"] == "existing" and blocked_new_build:
            logger.warning(
                "TCT %s has max=existing but extendable %s capacity is still present in the "
                "model; run remove_tct_blocked_components before optimize() so the new "
                "candidates are deleted instead of left unconstrained.",
                target["name"],
                "/".join(carriers),
            )
        lhs = sum(expressions[1:], expressions[0])
        minimum = existing_capacity if target["min"] == "existing" else float(target["min"])
        region_key = "-".join(_as_clean_list(target.region)) or "all"
        constraint_key = f"{target['name']}_{region_key}_{planning_horizon}"
        if not np.isnan(minimum):
            rhs = floor_precision(minimum - existing_capacity, 2)
            n.model.add_constraints(
                lhs >= rhs,
                name=f"TCT-{constraint_key}_min",
            )
            added += 1
        # max=existing is enforced by deleting the new/extendable candidates for this
        # carrier/region before the model is built (see remove_tct_blocked_components),
        # not by an LP inequality here.
        if target["max"] != "existing" and not pd.isna(target["max"]):
            maximum = float(target["max"])
            if maximum < existing_capacity:
                raise ValueError(
                    f"TCT {target['name']} maximum {maximum} MW is below existing {existing_capacity} MW.",
                )
            rhs = ceil_precision(maximum - existing_capacity, 2)
            n.model.add_constraints(
                lhs <= rhs,
                name=f"TCT-{constraint_key}_max",
            )
            added += 1

    if added:
        logger.info("Added %d TCT constraints.", added)


def _process_reeds_data(filepath, carriers, value_col):
    """Helper function to process RPS or CES REEDS data."""
    reeds = pd.read_csv(filepath)

    # Handle both wide and long formats
    if "rps_all" not in reeds.columns:
        reeds = reeds.melt(
            id_vars="st",
            var_name="planning_horizon",
            value_name=value_col,
        )

    # Standardize column names
    reeds = reeds.rename(
        columns={"st": "region", "t": "planning_horizon", "rps_all": "pct"},
    )
    reeds["carrier"] = [", ".join(carriers)] * len(reeds)

    # Ensure the final dataframe has consistent columns
    reeds = reeds[["region", "planning_horizon", "carrier", "pct"]]
    reeds = reeds[reeds["pct"] > 0.0]  # Remove any rows with zero or negative percentages

    return reeds


def _collapse_portfolio_standards(n: pypsa.Network, planning_horizons: list[int], *args):
    """Collapse portfolio standards into a single row per region, planning horizon, and carrier."""
    expected_columns = ["region", "planning_horizon", "carrier", "pct"]
    dfs = [df[expected_columns] for df in args if not df.empty]
    if not dfs:
        return pd.DataFrame(columns=expected_columns)
    portfolio_standards = pd.concat(dfs, ignore_index=True)

    return portfolio_standards[
        (portfolio_standards.pct > 0.0)
        & (
            portfolio_standards.planning_horizon.isin(
                planning_horizons,
            )
        )
        & (portfolio_standards.region.isin(n.buses.reeds_state.unique()))
    ]


def add_RPS_constraints(n, config, snakemake=None):
    """
    Add Renewable Portfolio Standards (RPS) constraints to the network.

    This function enforces constraints on the percentage of electricity generation
    from renewable energy sources for specific regions and planning horizons.
    It reads the necessary data from configuration files and the network.

    The RPS/CES share is enforced against each region's physical generation, not
    against demand. Load-shedding generators are excluded from both sides because
    their dispatch represents unserved energy rather than electricity production.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object.
    config : dict
        A dictionary containing configuration settings and file paths.
    snakemake: object, optional
        Snakemake object containing inputs and parameters

    Returns
    -------
    None
    """
    # Get model horizon
    model_horizon = get_model_horizon(n.model)
    snapshot_weightings = n.snapshot_weightings.loc[n.snapshots].generators
    rps_scaling_days = 365.0

    # Initialize empty portfolio_standards instead of reading CSV
    portfolio_standards = pd.DataFrame(columns=["region", "planning_horizon", "carrier", "pct"])

    # Process RPS and CES REEDS data
    rps_reeds = _process_reeds_data(
        snakemake.input.rps_reeds,
        RPS_CARRIERS,
        value_col="pct",
    )
    ces_reeds = _process_reeds_data(
        snakemake.input.ces_reeds,
        CES_CARRIERS,
        value_col="pct",
    )

    # Concatenate all portfolio standards
    portfolio_standards = _collapse_portfolio_standards(
        n,
        snakemake.params.planning_horizons,
        portfolio_standards,
        rps_reeds,
        ces_reeds,
    )

    # Iterate through constraints and add RPS constraints to the model, one per
    # reeds_state -- no pooling across states via REC trading zones.
    added = 0
    for (state, planning_horizon, policy_carriers), zone_constraints in portfolio_standards.groupby(
        ["region", "planning_horizon", "carrier"],
    ):
        if planning_horizon not in model_horizon:
            continue
        region_buses = get_region_buses(n, [state])
        carriers = [carrier.strip() for carrier in policy_carriers.split(",")]

        # Filter region generators
        region_gens = n.generators[n.generators.bus.isin(region_buses.index)]
        region_gens_policy = region_gens[
            ~region_gens.carrier.isin(RPS_DENOMINATOR_EXCLUDED_CARRIERS)
        ]
        region_gens_eligible = region_gens_policy[
            region_gens_policy.carrier.isin(carriers)
        ]

        if region_gens_eligible.empty:
            logger.warning(
                "Skipping RPS row for %s in %s because no eligible generators exist.",
                state,
                planning_horizon,
            )
            continue

        period_weights = snapshot_weightings.loc[planning_horizon]

        # Eligible generation
        p_eligible = n.model["Generator-p"].sel(
            period=planning_horizon,
            Generator=region_gens_eligible.index,
        )
        p_eligible = p_eligible.mul(period_weights)

        # Required share of physical generation. In particular, load shedding is
        # unserved demand and must not create an additional renewable obligation.
        p_total = n.model["Generator-p"].sel(
            period=planning_horizon,
            Generator=region_gens_policy.index,
        )
        p_total = p_total.mul(period_weights)
        pct = zone_constraints["pct"].iloc[0]
        required_from_generation = pct * p_total.sum()

        lhs = p_eligible.sum() / rps_scaling_days - required_from_generation / rps_scaling_days
        rhs = 0 / rps_scaling_days
        policy_kind = "rps" if set(carriers) == set(RPS_CARRIERS) else "ces"

        n.model.add_constraints(
            lhs >= rhs,
            name=f"PortfolioStandard-{state}_{planning_horizon}_{policy_kind}_limit",
        )
        added += 1

    if added:
        logger.info("Added %d RPS/CES constraints.", added)


def add_regional_co2limit(n, config):
    """Adding regional regional CO2 Limits Specified in the config.yaml."""
    model_horizon = get_model_horizon(n.model)
    regional_co2_lims = pd.read_csv(
        config["electricity"]["regional_Co2_limits"],
        index_col=[0],
    )

    regional_co2_lims = regional_co2_lims[regional_co2_lims.planning_horizon.isin(n.investment_periods)]
    weightings = n.snapshot_weightings.loc[n.snapshots]

    for idx, emmission_lim in regional_co2_lims.iterrows():
        region_list = [region.strip() for region in emmission_lim.regions.split(",")]
        region_buses = get_region_buses(n, region_list)

        emissions = n.carriers.co2_emissions.fillna(0)[lambda ds: ds != 0]
        region_gens = n.generators[n.generators.bus.isin(region_buses.index)]
        region_gens_em = region_gens.query("carrier in @emissions.index")

        if region_buses.empty or region_gens_em.empty:
            continue

        region_co2lim = emmission_lim.limit
        planning_horizon = emmission_lim.planning_horizon
        if planning_horizon not in model_horizon:
            continue

        efficiency = get_as_dense(
            n,
            "Generator",
            "efficiency",
            inds=region_gens_em.index,
        )  # mw_elect/mw_th
        em_pu = region_gens_em.carrier.map(emissions) / efficiency  # tonnes_co2/mw_electrical
        em_pu = em_pu.multiply(weightings.generators, axis=0).loc[planning_horizon].fillna(0)

        # Emitting Gens
        p_em = n.model["Generator-p"].loc[:, region_gens_em.index].sel(period=planning_horizon)

        # CO2 Atmospheric Emissions
        if any(n.carriers.index.isin(["co2"])):
            co2_atm = n.stores.loc[["atmosphere" in name for name in n.stores.index]]
            last_timestep = n.snapshots.get_level_values(1)[-1]
            end_co2_atm_storage = (
                n.model["Store-e"].loc[:, co2_atm.index].sel(period=planning_horizon).sel(timestep=last_timestep)
            ).sum()
        else:
            end_co2_atm_storage = 0

        lhs = (p_em * em_pu).sum() + end_co2_atm_storage
        rhs = region_co2lim
        n.model.add_constraints(
            lhs <= rhs,
            name=f"RegionalCO2-{emmission_lim.name}_{planning_horizon}co2_limit",
        )

        logger.info(
            f"Adding regional Co2 Limit for {emmission_lim.name} in {planning_horizon} with limit {rhs}",
        )


def add_post_2032_gas_average_power_limit(n, build_year_threshold=2032, max_avg_ratio=0.4):
    """
    Limit weighted annual average dispatch of post-2032 CCGT/OCGT to a share of nameplate capacity.

    For each planning horizon and each eligible generator:
        sum_t(p_t * w_t) / sum_t(w_t) <= max_avg_ratio * p_nom
    """
    carriers = ["CCGT", "OCGT"]
    eligible = n.generators[n.generators.carrier.isin(carriers)].copy()
    if eligible.empty:
        return

    build_year = pd.to_numeric(eligible.build_year, errors="coerce")
    eligible = eligible[build_year > build_year_threshold]
    if eligible.empty:
        return

    model_horizon = get_model_horizon(n.model)
    planning_horizons = [year for year in n.investment_periods if year in model_horizon]
    if not planning_horizons:
        return

    ext_i_all = n.generators.query("p_nom_extendable").index

    for planning_horizon in planning_horizons:
        active = n.get_active_assets("Generator", planning_horizon)
        active_eligible = eligible.index.intersection(n.generators.index[active])
        if active_eligible.empty:
            continue

        period_weights = n.snapshot_weightings.generators.loc[planning_horizon]
        total_weight = float(period_weights.sum())
        if total_weight <= 0.0:
            logger.warning(
                "Post-2032 CCGT/OCGT average power limit skipped for %s: zero snapshot weight.",
                planning_horizon,
            )
            continue

        dispatch = n.model["Generator-p"].sel(period=planning_horizon, Generator=active_eligible)
        avg_dispatch = dispatch.mul(period_weights).sum("timestep") / total_weight

        fixed_i = active_eligible.difference(ext_i_all)
        if not fixed_i.empty:
            fixed_rhs = max_avg_ratio * n.generators.p_nom.loc[fixed_i]
            n.model.add_constraints(
                avg_dispatch.sel(Generator=fixed_i) <= fixed_rhs,
                name=f"GasAvgPower-post2032_gas_avg_power_fixed_{planning_horizon}",
            )

        extendable_i = active_eligible.intersection(ext_i_all)
        if not extendable_i.empty:
            extendable_p_nom = n.model["Generator-p_nom"].loc[extendable_i].rename({"Generator-ext": "Generator"})
            n.model.add_constraints(
                avg_dispatch.sel(Generator=extendable_i) <= max_avg_ratio * extendable_p_nom,
                name=f"GasAvgPower-post2032_gas_avg_power_ext_{planning_horizon}",
            )

        logger.info(
            "Added post-2032 CCGT/OCGT weighted annual average power limit for %s active generators in %s (<= %.0f%% of nameplate).",
            len(active_eligible),
            planning_horizon,
            max_avg_ratio * 100,
        )
