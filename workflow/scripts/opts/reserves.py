"""
Energy Reserve Margin (ERM) constraints for PyPSA-USA.

This module contains functions for implementing capacity adequacy constraints,
including energy reserve margins (ERM).
"""

import logging

import numpy as np
import pandas as pd
import pypsa
from linopy import merge
from opts._helpers import get_region_buses
from pypsa.descriptors import nominal_attrs
from xarray import DataArray

logger = logging.getLogger(__name__)


def _get_zero_emission_periods(n, config):
    """Return the set of planning horizons where the national (all-region) CO2 limit is 0.

    Used to exclude emitting generators from ERM capacity credit in periods
    where fossil dispatch is completely prohibited.
    """
    try:
        co2_lims = pd.read_csv(config["electricity"]["regional_Co2_limits"])
    except (KeyError, FileNotFoundError, TypeError):
        return set()
    mask = (co2_lims["regions"].str.strip() == "all") & (co2_lims["limit"] == 0)
    return set(co2_lims.loc[mask, "planning_horizon"])


def define_SU_reserve_constraints(n, sns):
    """Energy-balance constraints for the StorageUnit RESERVES shadow variables.

    Mirrors pypsa v1.3's internal ``define_storage_unit_constraints`` (xarray
    model-space via ``n.optimize._window`` and ``c.da``) so that MultiIndex
    snapshots align under xarray >= 2026; only the variable names differ
    (``*_RESERVES``) and there is no spill term in the shadow system.
    """
    m = n.model
    component = "StorageUnit"
    dim = "snapshot"
    window = n.optimize._window.subset(sns)
    c_obj = n.components[component]

    if c_obj.static.empty:
        return

    active = c_obj.da.active.sel(snapshot=sns, name=c_obj.active_assets)

    eh = window.snapshot_weightings("stores")

    eff_stand = (1 - c_obj.da.standing_loss.sel(snapshot=sns, name=c_obj.active_assets)) ** eh
    eff_dispatch = c_obj.da.efficiency_dispatch.sel(snapshot=sns, name=c_obj.active_assets)
    eff_store = c_obj.da.efficiency_store.sel(snapshot=sns, name=c_obj.active_assets)

    soc = m[f"{component}-state_of_charge_RESERVES"]

    lhs = [
        (-1, soc),
        (-1 / eff_dispatch * eh, m[f"{component}-p_dispatch_RESERVES"]),
        (eff_store * eh, m[f"{component}-p_store_RESERVES"]),
    ]

    # mask `include_previous_soc` excludes the first snapshot for non-cyclic assets
    noncyclic_b = ~c_obj.da.cyclic_state_of_charge.sel(name=c_obj.active_assets)
    include_previous_soc = (active.cumsum(dim) != 1).where(noncyclic_b, True)

    previous_soc = soc.where(active).ffill(dim).roll(snapshot=1).ffill(dim)

    # inflow and initial soc for noncyclic assets go to rhs
    soc_init = c_obj.da.state_of_charge_initial.sel(name=c_obj.active_assets)
    rhs = -c_obj.da.inflow.sel(snapshot=sns, name=c_obj.active_assets) * eh

    if n._multi_invest:
        # per-period cycling / initial-value reset, exactly as pypsa v1 does it
        per_period = c_obj.da.cyclic_state_of_charge_per_period.sel(
            name=c_obj.active_assets,
        ) | c_obj.da.state_of_charge_initial_per_period.sel(name=c_obj.active_assets)

        previous_soc_pp = window.roll_within_periods(soc)

        within_period = ~window.period_start_mask()
        include_previous_soc_pp = active & (
            within_period | c_obj.da.cyclic_state_of_charge_per_period.sel(name=c_obj.active_assets)
        )

        # the per-period inclusion is carried by the `include_previous_soc`
        # coefficient (not by masking `previous_soc_pp` to NaN, which v1 reads
        # as an absent term and would drop the period-start energy-balance row)
        previous_soc = previous_soc.where(~per_period, previous_soc_pp)
        include_previous_soc = include_previous_soc_pp.where(
            per_period,
            include_previous_soc,
        )

    lhs += [(eff_stand * include_previous_soc, previous_soc)]

    lhs = m.linexpr(*lhs)
    rhs = rhs.where(include_previous_soc, rhs - soc_init)

    m.add_constraints(lhs, "=", rhs, name=f"{component}-energy_balance_RESERVES", mask=active)


def define_operational_constraints_for_extendables(
    n: pypsa.Network,
    sns: pd.Index,
    c: str,
    attr: str,
) -> None:
    """
    Sets power dispatch constraints for extendable devices for a given
    component and a given attribute.

    Parameters
    ----------
    n : pypsa.Network
    sns : pd.Index
        Snapshots of the constraint.
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'
    """
    lhs_lower: DataArray | tuple
    lhs_upper: DataArray | tuple

    c_obj = n.components[c]
    ext_i = c_obj.extendables

    if ext_i.empty:
        return
    if isinstance(ext_i, pd.MultiIndex):
        ext_i = ext_i.unique(level="name")

    min_pu, max_pu = c_obj.get_bounds_pu(attr=attr)
    min_pu = min_pu.sel(name=ext_i)
    max_pu = max_pu.sel(name=ext_i)
    if "snapshot" in min_pu.dims:
        min_pu = min_pu.sel(snapshot=sns)
        max_pu = max_pu.sel(snapshot=sns)

    dispatch = n.model[f"{c}-{attr}_RESERVES"].sel(name=ext_i)
    capacity = n.model[f"{c}-{nominal_attrs[c]}"].sel(name=ext_i)

    active = c_obj.da.active.sel(name=ext_i, snapshot=sns)

    lhs_lower = dispatch - min_pu * capacity
    lhs_upper = dispatch - max_pu * capacity

    n.model.add_constraints(
        lhs_lower,
        ">=",
        0,
        name=f"{c}-ext-{attr}-lower_RESERVES",
        mask=active,
    )
    n.model.add_constraints(
        lhs_upper,
        "<=",
        0,
        name=f"{c}-ext-{attr}-upper_RESERVES",
        mask=active,
    )


def define_operational_constraints_for_non_extendables(
    n: pypsa.Network,
    sns: pd.Index,
    c: str,
    attr: str,
) -> None:
    """
    Sets power dispatch constraints for non-extendable and non-commitable
    assets for a given component and a given attribute.

    Parameters
    ----------
    n : pypsa.Network
    sns : pd.Index
        Snapshots of the constraint.
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'
    """
    dispatch_lower: DataArray | tuple
    dispatch_upper: DataArray | tuple

    c_obj = n.components[c]
    fix_i = c_obj.fixed.difference(c_obj.committables)

    if fix_i.empty:
        return

    nominal_fix = c_obj.da[nominal_attrs[c]].sel(name=fix_i)
    min_pu, max_pu = c_obj.get_bounds_pu(attr=attr)
    min_pu = min_pu.sel(name=fix_i)
    max_pu = max_pu.sel(name=fix_i)
    if "snapshot" in min_pu.dims:
        min_pu = min_pu.sel(snapshot=sns)
        max_pu = max_pu.sel(snapshot=sns)

    lower = min_pu * nominal_fix
    upper = max_pu * nominal_fix

    active = c_obj.da.active.sel(name=fix_i, snapshot=sns)

    dispatch_lower = n.model[f"{c}-{attr}_RESERVES"].sel(name=fix_i)
    dispatch_upper = n.model[f"{c}-{attr}_RESERVES"].sel(name=fix_i)

    n.model.add_constraints(
        dispatch_lower,
        ">=",
        lower,
        name=f"{c}-fix-{attr}-lower_RESERVES",
        mask=active,
    )
    n.model.add_constraints(
        dispatch_upper,
        "<=",
        upper,
        name=f"{c}-fix-{attr}-upper_RESERVES",
        mask=active,
    )


def _get_regional_demand(n, region_buses):
    """
    Calculate hourly demand for a specific region.

    Parameters
    ----------
    n : pypsa.Network
    region_buses : pd.DataFrame
        DataFrame containing buses in the region

    Returns
    -------
    pd.Series
        Hourly demand series for the region
    """
    rhs = (
        (-n.get_switchable_as_dense("Load", "p_set", n.snapshots) * n.loads.sign)
        .T.groupby(n.loads.bus)
        .sum()
        .T.reindex(columns=region_buses.index, fill_value=0)
    )

    return rhs


def _boundary_flow_terms(n, m, c, attr, buses, activity, efficiency=None):
    """Actual flows of component ``c`` crossing the boundary of ``buses``, summed per snapshot.

    Inflow (bus0 outside, bus1 inside) counts positive, scaled by ``efficiency``
    when given; outflow (bus0 inside, bus1 outside) counts negative.
    """
    df = n.components[c].static
    if df.empty:
        return []
    in_bus0 = df.bus0.isin(buses)
    in_bus1 = df.bus1.isin(buses)
    terms = []
    for idx, sign in [(df.index[~in_bus0 & in_bus1], 1), (df.index[in_bus0 & ~in_bus1], -1)]:
        if idx.empty:
            continue
        if sign > 0 and efficiency is not None:
            # scale the full variable before selecting, so the MultiIndex snapshot coords align
            expr = (DataArray(efficiency) * m[f"{c}-{attr}"]).sel(name=idx)
        else:
            expr = sign * m[f"{c}-{attr}"].sel(name=idx)
        expr = expr.where(activity.sel(name=idx))
        terms.append(expr.sum("name"))
    return terms


def define_erm_regional_constraints(
    n,
    snapshots,
    erm,
    region_name,
    region_buses,
    zero_emission_periods=None,
    emitting_carriers=None,
):
    """
    Define the ERM constraint of one region: one row per snapshot, summed over the region.

    Following GenX's capacity reserve margin, the region's reserve capacity plus its
    actual net imports (flows on lines and links crossing the region boundary) must
    cover ``(1 + erm)`` times the region's demand in every snapshot. Flows inside the
    region do not enter. Because the imports are the same flows as in the energy
    balance, this is equivalent to holding ``erm * demand`` of headroom in the region.

    Generators count with ``p_nom * p_max_pu``; storage counts with its shadow
    (RESERVES) dispatch. Activity masks drop assets that are not built or retired.
    Emitting generators get no credit in planning horizons whose national CO2 limit
    is zero, since they cannot dispatch there and are not firm capacity.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : pd.Index
        Snapshots of the constraint.
    erm : float
        Energy reserve margin as a fraction (e.g., 0.15 for 15%)
    region_name : str
        Name of the region for constraint naming
    region_buses : pd.DataFrame
        DataFrame containing buses in the region
    zero_emission_periods : set, optional
        Planning horizons (int years) where the national CO2 limit is 0.
        Emitting generators receive no ERM capacity credit in these periods.
    emitting_carriers : set, optional
        Carrier names with co2_emissions > 0. Required if zero_emission_periods is set.
    """
    sns = snapshots
    m = n.model
    buses = region_buses.index

    # Per-snapshot flag: True = snapshot lies in a zero-emission planning horizon
    if zero_emission_periods and emitting_carriers:
        period_per_snap = sns.get_level_values(0) if isinstance(sns, pd.MultiIndex) else sns
        snap_is_zero = pd.Series([p in zero_emission_periods for p in period_per_snap], index=sns, dtype=bool)
    else:
        snap_is_zero = pd.Series(False, index=sns, dtype=bool)

    def _activity(component, names=None):
        mask = n.components[component].get_activity_mask(sns)
        if names is not None:
            mask = mask[names]
        mask.index.name = "snapshot"
        mask.columns.name = "name"
        return mask

    def _gen_credit_activity(gens):
        """Activity mask of generators, with emitting carriers off in zero-emission periods."""
        activity = _activity("Generator", gens.index)
        if snap_is_zero.any() and emitting_carriers:
            fossil = gens.index[gens.carrier.isin(emitting_carriers)]
            if not fossil.empty:
                activity.loc[snap_is_zero, fossil] = False
                # pandas .loc boolean assignment can silently reset index/column names
                activity.index.name = "snapshot"
                activity.columns.name = "name"
                logger.debug(
                    f"Excluded {len(fossil)} emitting generators from ERM credit "
                    f"in zero-emission snapshots for region {region_name}.",
                )
        return activity

    exprs = []

    # Extendable generators: p_nom * p_max_pu
    region_gens = n.generators[n.generators.bus.isin(buses)]
    ext_gens = region_gens[region_gens.p_nom_extendable]
    if not ext_gens.empty:
        p_max_pu = n.get_switchable_as_dense("Generator", "p_max_pu", sns, inds=ext_gens.index)
        p_max_pu.index.name = "snapshot"
        p_max_pu.columns.name = "name"
        # wrap in DataArray so linopy keeps the flat 'snapshot' dim of the
        # MultiIndex frames instead of unstacking to period x timestep
        expr = m["Generator-p_nom"].loc[ext_gens.index] * DataArray(p_max_pu)
        # .where() drops terms (var label -1) instead of leaving zero coefficients
        expr = expr.where(DataArray(_gen_credit_activity(ext_gens)))
        exprs.append(expr.sum("name"))

    # Storage: shadow dispatch minus shadow charge
    region_su = n.storage_units.index[n.storage_units.bus.isin(buses)]
    if not region_su.empty:
        sign = DataArray(n.storage_units.sign.loc[region_su]) if "sign" in n.storage_units else 1
        su_activity = DataArray(_activity("StorageUnit", region_su))
        for attr, s in [("p_dispatch_RESERVES", 1), ("p_store_RESERVES", -1)]:
            expr = (s * sign * m[f"StorageUnit-{attr}"].sel(name=region_su)).where(su_activity)
            exprs.append(expr.sum("name"))

    # Actual flows across the region boundary
    if not n.lines.empty:
        exprs += _boundary_flow_terms(n, m, "Line", "s", buses, DataArray(_activity("Line")))
    if not n.links.empty:
        link_efficiency = n.get_switchable_as_dense("Link", "efficiency", sns)
        link_efficiency.index.name = "snapshot"
        link_efficiency.columns.name = "name"
        exprs += _boundary_flow_terms(
            n,
            m,
            "Link",
            "p",
            buses,
            DataArray(_activity("Link")),
            efficiency=link_efficiency,
        )

    # RHS: (1 + erm) * regional demand - credit of non-extendable generators
    regional_demand = _get_regional_demand(n, region_buses).loc[sns].sum(axis=1)
    rhs = regional_demand * (1.0 + erm)
    fix_gens = region_gens[~region_gens.p_nom_extendable]
    if not fix_gens.empty:
        p_max_pu = n.get_switchable_as_dense("Generator", "p_max_pu", sns, inds=fix_gens.index)
        p_max_pu.index.name = "snapshot"
        credit = p_max_pu * fix_gens.p_nom * _gen_credit_activity(fix_gens)
        rhs = rhs - credit.sum(axis=1)
    rhs.index = sns
    rhs.index.name = "snapshot"
    rhs = DataArray(rhs)

    if not exprs:
        if (rhs > 0).any().item():
            raise ValueError(f"ERM region {region_name} has no reserve resources but a positive requirement.")
        return

    lhs = merge(exprs, join="outer")

    # Snapshots where every term is masked out: drop the row, or fail if it is infeasible
    empty = (lhs.vars == -1).all("_term")
    mask = None
    if empty.any():
        if (empty & (rhs > 0)).any().item():
            raise ValueError(f"Empty LHS with positive RHS in ERM constraint of region {region_name}.")
        mask = ~empty

    m.add_constraints(lhs, ">=", rhs, name=f"GlobalConstraint-{region_name}_ERM", mask=mask)


def add_ERM_constraints(n, snapshots, config=None, snakemake=None, regional_erm_data=None):
    """
    Add Energy Reserve Margin (ERM) constraints for regional capacity adequacy.

    This function enforces that each region has sufficient firm capacity to meet
    peak demand plus a reserve margin. These resources must be "energy-backed" meaning
    resources like storage devices must have the state of charge to meet the reserve
    to contribute to the ERM.

    Creates one constraint per region spanning all investment periods, using activity
    masking to handle generator retirements and build years.

    As in GenX's capacity reserve margin, each region gets one constraint per
    snapshot summed over its buses, and lines and links enter only through their
    actual flows across the region boundary. This is a preventive requirement:
    each region must hold ``erm * demand`` of headroom (available capacity minus
    actual dispatch) on top of its actual net imports.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object
    config : dict, optional
        Configuration dictionary containing electricity.erm dict.
        Required if regional_erm_data not provided.
    snakemake : snakemake object, optional
        Not used in the new implementation, kept for API compatibility.
    regional_erm_data : dict, optional
        Direct input of ERM requirements as dict {region_name: erm_value}.
        If provided, this takes precedence over config data.
    """
    model = n.model

    # Get ERM data: dict {region_name: erm_value}
    # Default to 15% reserve margin for all regions if not specified
    default_erm = {"all": 0.15}

    if regional_erm_data is not None:
        erm_dict = regional_erm_data
    elif config is not None and config.get("electricity", {}).get("erm"):
        erm_dict = config["electricity"]["erm"]
    else:
        logger.info("No ERM configuration provided. Using default: {'all': 0.15}")
        erm_dict = default_erm

    # Identify planning horizons where fossil cannot count toward ERM (CO2 limit = 0)
    zero_emission_periods = _get_zero_emission_periods(n, config)
    emitting_carriers = set(n.carriers.index[n.carriers.co2_emissions.fillna(0) > 0])
    if zero_emission_periods:
        logger.info(
            f"Zero-emission periods detected {zero_emission_periods}. "
            f"Emitting carriers {emitting_carriers} will receive no ERM capacity credit "
            f"in those periods.",
        )

    for region_name, erm_value in erm_dict.items():
        region_name = region_name.strip()
        if erm_value is None:  # a null margin switches off a region inherited from a lower config layer
            logger.info(f"ERM for {region_name} is null; no constraint added.")
            continue
        region_buses = get_region_buses(n, [region_name])

        if region_buses.empty:
            continue

        logger.info(f"Adding ERM constraint for {region_name} with reserve level {erm_value}")

        # Create model variables to track storage contributions (only once)
        c = "StorageUnit"
        if not n.storage_units.empty and f"{c}-p_dispatch_RESERVES" not in model.variables:
            model.add_variables(
                -np.inf,
                model.variables["StorageUnit-p_dispatch"].upper,
                name=f"{c}-p_dispatch_RESERVES",
            )
            model.add_variables(
                -np.inf,
                model.variables["StorageUnit-p_store"].upper,
                name=f"{c}-p_store_RESERVES",
            )
            model.add_variables(
                -np.inf,
                model.variables["StorageUnit-state_of_charge"].upper,
                name=f"{c}-state_of_charge_RESERVES",
            )
            define_SU_reserve_constraints(n, snapshots)
            define_operational_constraints_for_extendables(n, snapshots, c, "state_of_charge")
            define_operational_constraints_for_extendables(n, snapshots, c, "p_dispatch")
            define_operational_constraints_for_extendables(n, snapshots, c, "p_store")
            define_operational_constraints_for_non_extendables(n, snapshots, c, "state_of_charge")
            define_operational_constraints_for_non_extendables(n, snapshots, c, "p_dispatch")
            define_operational_constraints_for_non_extendables(n, snapshots, c, "p_store")

        # Lines and links contribute their actual flows (Line-s, Link-p), which
        # PyPSA already bounds, so no shadow transmission variables are needed.

        define_erm_regional_constraints(
            n,
            snapshots,
            erm_value,
            region_name,
            region_buses,
            zero_emission_periods=zero_emission_periods,
            emitting_carriers=emitting_carriers,
        )
        logger.info(f"Added ERM constraint for {region_name}")


def store_ERM_duals(n):
    """
    Store the ERM shadow prices as ``n.buses_t["erm_price"]``.

    Each region's ERM constraint has one dual per snapshot; it is assigned to every
    bus of the region. Buses in several ERM regions get the sum of their duals, and
    buses in none get zero.
    """
    model = n.model
    prefix, suffix = "GlobalConstraint-", "_ERM"
    erm_constraints = [c for c in model.constraints if c.startswith(prefix) and c.endswith(suffix)]
    if not erm_constraints:
        return

    logger.info("Storing ERM shadow prices from optimization results")
    erm_price = pd.DataFrame(0.0, index=n.snapshots, columns=n.buses.index)
    for constraint in erm_constraints:
        region_name = constraint[len(prefix) : -len(suffix)]
        region_buses = get_region_buses(n, [region_name]).index
        dual = model.dual[constraint].to_pandas().reindex(n.snapshots).fillna(0.0)
        erm_price[region_buses] = erm_price[region_buses].add(dual, axis=0)
    n.buses_t["erm_price"] = erm_price
