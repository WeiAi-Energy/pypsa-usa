"""
Solves optimal operation and capacity for a network with the option to
iteratively optimize while updating line reactances.

This script is used for optimizing the electrical network as well as the
sector coupled network.

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.

The optimization is based on the :func:`network.optimize` function.
Additionally, some extra constraints specified in :mod:`solve_network` are added.

.. note::

    The rules ``solve_elec_networks`` and ``solve_sector_networks`` run
    the workflow for all scenarios in the configuration file (``scenario:``)
    based on the rule :mod:`solve_network`.
"""

import copy
import logging
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
import yaml
from pypsa.optimization.common import reindex
from _helpers import (
    configure_logging,
    set_case_config,
    update_config_from_wildcards,
)
from opts._helpers import patch_linopy_multiindex_assign
from opts.bidirectional_link import add_bidirectional_link_constraints
from opts.policy import (
    add_post_2032_gas_average_power_limit,
    add_regional_co2limit,
    add_RPS_constraints,
    add_technology_capacity_target_constraints,
)
from opts.representative_periods import (
    add_representative_period_storage_constraints,
)
from opts.reserves import (
    add_ERM_constraints,
    store_ERM_duals,
)
from regional_cost import SectorCosts

patch_linopy_multiindex_assign()

logger_gurobi = logging.getLogger("gurobipy")
logger_gurobi.propagate = False

logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)

FLEXIBLE_ELECTROLYSIS_LINK_SUFFIX = " flexible electrolysis"
FLEXIBLE_ELECTROLYSIS_BUS_SUFFIX = " flexible electrolysis H2"


def prepare_network(n, solve_opts=None):
    if "clip_p_max_pu" in solve_opts:
        df = n.generators_t.p_max_pu
        n.generators_t.p_max_pu = df.where(df > solve_opts["clip_p_max_pu"], other=0.0)
        df = n.generators_t.p_min_pu
        n.generators_t.p_min_pu = df.where(df > solve_opts["clip_p_max_pu"], other=0.0)

        df = n.links_t.p_max_pu
        n.links_t.p_max_pu = df.where(df > solve_opts["clip_p_max_pu"], other=0.0)
        df = n.links_t.p_min_pu
        n.links_t.p_min_pu = df.where(df > solve_opts["clip_p_max_pu"], other=0.0)

        df = n.storage_units_t.inflow
        n.storage_units_t.inflow = df.where(df > solve_opts["clip_p_max_pu"], other=0.0)
    load_shedding = solve_opts.get("load_shedding")
    if load_shedding:
        # intersect between macroeconomic and surveybased willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
        # TODO: retrieve color and nice name from config
        logger.warning("Adding load shedding generators.")
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
        buses_i = n.buses.query("carrier == 'AC'").index
        if not np.isscalar(load_shedding):
            load_shedding = 1e5  # USD/MWh

        n.madd(
            "Generator",
            buses_i,
            " load",
            bus=buses_i,
            carrier="load",
            marginal_cost=load_shedding,  # USD/MWh
            p_nom=1e4,  # MW
            p_nom_extendable=False,
        )

    if solve_opts.get("noisy_costs"):  ##random noise to costs of generators
        for t in n.iterate_components():
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (np.random.random(len(t.df)) - 0.5)

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        # Get first nhours for each level of the multi-index
        first_nhours = pd.MultiIndex.from_tuples(
            [
                snap
                for year in n.snapshots.get_level_values(0).unique()
                for snap in n.snapshots[n.snapshots.get_level_values(0) == year][:nhours]
            ],
            names=n.snapshots.names,
        )
        n.set_snapshots(first_nhours)
        n.snapshot_weightings[:] = 8760.0 / nhours

    return n


def flexible_electrolysis_accounting_region(flex_config):
    """Return the validated hydrogen balance accounting level.

    Mirrors ``add_extra_components.flexible_electrolysis_accounting_region``:
    ``h2ptcreg`` balances hydrogen production per 45V hydrogen PTC region,
    ``nation`` balances it once over the whole modelled system.
    """
    accounting_region = flex_config.get("accounting_region", "h2ptcreg")
    if accounting_region not in ("h2ptcreg", "nation"):
        raise ValueError(
            "flexible_electrolysis 'accounting_region' must be 'h2ptcreg' or 'nation'; "
            f"got {accounting_region!r}.",
        )
    return accounting_region


def h2ptcreg_hydrogen_shares(hydrogen_share_path):
    """Return each h2ptcreg region's share of national annual hydrogen demand.

    The file stores state-level shares (derived from the hourly state hydrogen
    demand profiles); they are summed up to the ``h2ptcreg`` level here.
    """
    if not hydrogen_share_path:
        raise ValueError(
            "Flexible electrolysis is enabled but no hydrogen_demand_share.csv path was provided "
            "to the hydrogen target constraint.",
        )
    shares = pd.read_csv(hydrogen_share_path)
    missing = {"h2ptcreg", "share"} - set(shares.columns)
    if missing:
        raise ValueError(f"{hydrogen_share_path} is missing required column(s): {sorted(missing)}.")
    return shares.groupby("h2ptcreg")["share"].sum()


def _electrolysis_capacity_terms(n, links):
    """Return fixed MW and the extendable-capacity expression for active links."""
    attributes = n.links.loc[links]
    extendable = attributes.p_nom_extendable.fillna(False).astype(bool)
    fixed_capacity = float(
        pd.to_numeric(attributes.loc[~extendable, "p_nom"], errors="coerce")
        .fillna(0.0)
        .sum(),
    )
    extendable_links = attributes.index[extendable]
    if extendable_links.empty:
        return fixed_capacity, None
    if "Link-p_nom" not in n.model.variables:
        raise ValueError(
            "Flexible electrolysis has extendable links, but Link-p_nom is unavailable.",
        )
    return fixed_capacity, n.model["Link-p_nom"].loc[extendable_links].sum()


def add_electrolysis_hydrogen_target_constraint(
    n,
    snapshots,
    config,
    sector_costs_path,
    hydrogen_share_path,
):
    """Fix annual electrolyzer H2 output per accounting region in every period.

    The accounting region follows ``flexible_electrolysis: accounting_region``.
    With ``h2ptcreg`` the configured national total is split across the 45V
    hydrogen PTC regions by their share of national hydrogen demand. With
    ``nation`` a single constraint requires the whole electrolyzer fleet to
    deliver the sum of those regional productions, i.e. the configured total.

    The electrolysis links carry ``efficiency = 0`` in the network so that the H2
    accounting buses balance trivially. Hydrogen output is derived from link
    electricity withdrawal ``p0`` and the conversion efficiency implied by
    ``h2 electrolysis`` / ``electricity-input`` in
    ``simple_sector_costs.csv``.

    The target is written as two rows per region and period rather than one. A
    per-snapshot variable ``h2_rate`` carries the region's instantaneous H2 output
    in MW_H2 and is pinned to the fleet's electricity withdrawal; the annual row
    then sums that one variable over the snapshots. This keeps the annual row
    sparse and both rows well scaled -- the aggregation row sits on the efficiency,
    the annual row on the snapshot weightings divided by ``1e3``, since the target
    is configured in TWh but accounted in GWh.
    """
    flex_config = config.get("flexible_electrolysis", {})
    if not flex_config.get("enable", False):
        return

    accounting_region = flexible_electrolysis_accounting_region(flex_config)

    flexible_links = n.links.index[
        (n.links.carrier == "electrolysis")
        & n.links.index.str.endswith(FLEXIBLE_ELECTROLYSIS_LINK_SUFFIX)
    ]
    if flexible_links.empty:
        logger.warning(
            "Flexible electrolysis is enabled, but no flexible electrolysis links were found.",
        )
        return

    if "Link-p" not in n.model.variables:
        logger.warning(
            "Flexible electrolysis constraint skipped: Link-p variable is unavailable.",
        )
        return

    if not sector_costs_path:
        raise ValueError(
            "Flexible electrolysis is enabled but no simple_sector_costs.csv path was provided "
            "to the hydrogen target constraint.",
        )
    total_target_twh = float(flex_config.get("annual_hydrogen_twh", 1612.0))
    electricity_input = SectorCosts(sector_costs_path).value(
        "h2 electrolysis",
        "electricity-input",
    )
    if electricity_input <= 0.0:
        raise ValueError(
            "'h2 electrolysis' 'electricity-input' in simple_sector_costs.csv must be positive; "
            f"got {electricity_input}.",
        )
    efficiency = 1.0 / electricity_input

    # Each link feeds the accounting H2 bus of the region it sits in, so the
    # region name is recoverable from bus1 (see add_extra_components.py).
    link_regions = n.links.loc[flexible_links, "bus1"].str.removesuffix(
        FLEXIBLE_ELECTROLYSIS_BUS_SUFFIX,
    )
    modelled_regions = pd.Index(sorted(link_regions.unique()))

    # The accounting buses are built in add_extra_components; a network built at a
    # different accounting level than the one configured here would silently get
    # the wrong targets.
    is_national_network = list(modelled_regions) == ["nation"]
    if (accounting_region == "nation") != is_national_network:
        raise ValueError(
            f"flexible_electrolysis 'accounting_region' is {accounting_region!r}, but the network's "
            f"electrolysis accounting bus(es) are {list(modelled_regions)}. Rebuild the network from "
            "add_extra_components after changing accounting_region.",
        )

    if accounting_region == "nation":
        # All links share one accounting bus, so the fleet total equals the sum of
        # the regional hydrogen productions, i.e. the configured national total.
        region_targets = pd.Series({"nation": total_target_twh})
    else:
        region_shares = h2ptcreg_hydrogen_shares(hydrogen_share_path)
        unknown = modelled_regions.difference(region_shares.index)
        if len(unknown):
            raise ValueError(
                f"No hydrogen demand share for h2ptcreg region(s) {list(unknown)} in {hydrogen_share_path}.",
            )

        # Renormalise over the modelled regions so the configured total is still met
        # when the network covers only part of the country (e.g. a single interconnect).
        region_shares = region_shares.reindex(modelled_regions)
        share_sum = float(region_shares.sum())
        if share_sum <= 0.0:
            logger.warning(
                "Flexible electrolysis constraint skipped: modelled regions have zero hydrogen demand share.",
            )
            return
        region_targets = region_shares / share_sum * total_target_twh

    link_p = n.model["Link-p"]
    weights = n.snapshot_weightings.generators.reindex(snapshots).fillna(0.0).astype(float)
    if isinstance(snapshots, pd.MultiIndex):
        periods = list(snapshots.get_level_values(0).unique())
    else:
        periods = [None]

    for period in periods:
        if period is None:
            period_snapshots = snapshots
            label = ""
        else:
            period_snapshots = snapshots[snapshots.get_level_values(0) == period]
            label = f"-{period}"

        if len(period_snapshots) == 0:
            continue

        period_weights = weights.loc[period_snapshots]
        period_hours = float(period_weights.sum())
        if period_hours <= 0.0:
            logger.warning(
                "Flexible electrolysis annual hydrogen constraint skipped for period %s: zero generator weight.",
                period,
            )
            continue

        for region, target_twh in region_targets.items():
            region_links = link_regions.index[link_regions == region]
            if period is not None:
                active = n.get_active_assets("Link", period).reindex(
                    region_links,
                    fill_value=False,
                )
                region_links = region_links[active.to_numpy(dtype=bool)]
            if region_links.empty:
                raise ValueError(
                    f"No active flexible electrolysis link for {region} in period {period}.",
                )

            # Sufficient capacity is already implied by the annual equality
            # together with the per-snapshot Link-p <= p_nom limits, so no
            # capacity-energy constraint is added. Non-extendable regions are
            # checked up front to report a clear error rather than an
            # infeasible LP.
            fixed_capacity, extendable_capacity = _electrolysis_capacity_terms(
                n,
                region_links,
            )
            if extendable_capacity is None:
                fixed_annual_twh = fixed_capacity * period_hours * efficiency / 1e6
                tolerance = 1e-9 * max(1.0, float(target_twh))
                if fixed_annual_twh + tolerance < float(target_twh):
                    raise ValueError(
                        f"Fixed flexible electrolysis capacity in {region} ({period}) "
                        f"can produce at most {fixed_annual_twh:.6g} TWh, below "
                        f"the {float(target_twh):.6g} TWh target.",
                    )

            # Aggregate the fleet per snapshot before summing over the year. Written
            # directly, the annual row carries one term per link and snapshot -- 144k
            # non-zeros for a nationwide fleet -- and a row that dense ties every
            # Link-p column into one row of the barrier's normal equations. The
            # aggregation is exact (associativity of the sum), so ``rate`` adds no
            # degree of freedom: it is the region's instantaneous H2 output in MW_H2,
            # pinned by its own equality.
            #
            # Slicing a snapshot MultiIndex drops its name, which linopy needs to
            # key the dimension off and match the Link-p snapshot axis.
            rate_coords = period_snapshots.copy()
            rate_coords.name = "snapshot"
            rate = n.model.add_variables(
                lower=0.0,
                coords=[rate_coords],
                name=f"FlexibleElectrolysis-h2_rate-{region}{label}",
            )
            n.model.add_constraints(
                rate - link_p.loc[period_snapshots, region_links].mul(efficiency).sum("Link"),
                "=",
                0.0,
                name=f"FlexibleElectrolysis-h2_rate-{region}{label}-definition",
            )
            n.model.add_constraints(
                rate.mul(period_weights / 1e3).sum() == float(target_twh) * 1e3,
                name=f"FlexibleElectrolysis-annual_hydrogen-{region}{label}",
            )

    logger.info(
        "Applied per-%s annual hydrogen targets (%.1f TWh total) across %d "
        "region(s): %s.",
        accounting_region,
        total_target_twh,
        len(region_targets),
        ", ".join(f"{r} {t:.1f} TWh" for r, t in region_targets.items()),
    )


def _get_line_x_sssc_total_max(config):
    """Return the optional LineX SSSC total capacity limit."""
    line_x_config = config.get("lines", {}).get("convert_lines_to_line_x", {})
    raw_limit = line_x_config.get("sssc_tot_max", np.inf)
    if raw_limit is None:
        return np.inf
    return float(raw_limit)


def _get_line_x_sssc_variable(model):
    """Return the Linopy variable representing LineX SSSC investment."""
    if not hasattr(model, "variables"):
        return None

    variable_names = list(model.variables)
    for name in ("LineX-sssc_nom", "LineX-sssc_nom_opt"):
        if name in variable_names:
            return model.variables[name]

    matches = [name for name in variable_names if name.startswith("LineX-") and "sssc_nom" in name]
    if len(matches) == 1:
        return model.variables[matches[0]]
    if len(matches) > 1:
        logger.warning(
            "LineX SSSC total capacity constraint skipped: ambiguous variables found: %s",
            ", ".join(matches),
        )
    return None


def add_line_x_sssc_total_max_constraint(n, snapshots, config):
    """Cap the sum of optimized LineX SSSC capacities when configured."""
    sssc_tot_max = _get_line_x_sssc_total_max(config)
    if not np.isfinite(sssc_tot_max):
        return

    line_xs = getattr(n, "line_xs", pd.DataFrame())
    if line_xs.empty:
        return

    line_x_sssc_var = _get_line_x_sssc_variable(n.model)
    if line_x_sssc_var is None:
        logger.warning("LineX SSSC total capacity constraint skipped: SSSC investment variable is unavailable.")
        return

    n.model.add_constraints(
        line_x_sssc_var.sum() <= sssc_tot_max,
        name="LineX-sssc_tot_max",
    )
    logger.info("Added LineX SSSC total capacity constraint at %.3f MW.", sssc_tot_max)


def _get_line_x_sssc_nom_max_pu(config):
    """Return the per-branch cap on ``sssc_nom`` as a share of the line capacity."""
    line_x_config = config.get("lines", {}).get("convert_lines_to_line_x", {})
    raw_ratio = line_x_config.get("sssc_nom_max_pu", 1.0)
    if raw_ratio is None:
        return np.inf
    return float(raw_ratio)


def _line_x_sssc_index_split(n, context):
    """
    Split the SSSC-extendable branches by whether their line capacity is a variable.

    Returns ``(sssc, sssc_ext_i, ext_i, fix_i)``, or ``None`` when there is no SSSC
    investment variable to work with. ``ext_i`` are the branches whose cap follows
    the optimized ``s_nom`` variable, ``fix_i`` those that keep a constant rating.
    """
    line_xs = getattr(n, "line_xs", pd.DataFrame())
    if line_xs.empty:
        return None

    sssc = _get_line_x_sssc_variable(n.model)
    if sssc is None:
        logger.warning("%s skipped: SSSC investment variable is unavailable.", context)
        return None

    sssc_dim = sssc.dims[0]
    sssc_ext_i = pd.Index(sssc.indexes[sssc_dim], name="LineX")
    if sssc_ext_i.empty:
        return None

    s_nom_var = n.model.variables["LineX-s_nom"] if "LineX-s_nom" in list(n.model.variables) else None
    if s_nom_var is None:
        ext_i = sssc_ext_i[:0]
    else:
        ext_i = sssc_ext_i.intersection(pd.Index(s_nom_var.indexes[s_nom_var.dims[0]])).rename("LineX")
    fix_i = sssc_ext_i.difference(ext_i).rename("LineX")
    return sssc, sssc_ext_i, ext_i, fix_i


def add_line_x_sssc_line_capacity_constraint(n, snapshots, config):
    """
    Cap each LineX SSSC rating at the capacity of its own line.

    ``sssc_nom <= sssc_nom_max_pu * s_nom``. Where the line itself is extendable the
    cap follows the optimized capacity variable, so more series compensation can only
    be bought together with a bigger line; otherwise the line's fixed rating is used.
    """
    ratio = _get_line_x_sssc_nom_max_pu(config)
    if not np.isfinite(ratio):
        return

    split = _line_x_sssc_index_split(n, "LineX SSSC line capacity constraint")
    if split is None:
        return
    sssc, sssc_ext_i, ext_i, fix_i = split
    line_xs = n.line_xs
    s_nom_var = n.model.variables["LineX-s_nom"] if not ext_i.empty else None
    sssc_dim = sssc.dims[0]

    if not ext_i.empty:
        lhs = reindex(sssc, sssc_dim, ext_i) - ratio * reindex(s_nom_var, s_nom_var.dims[0], ext_i)
        n.model.add_constraints(lhs, "<=", 0, name="LineX-sssc_nom-line_capacity")

    if not fix_i.empty:
        rhs = ratio * line_xs.s_nom.reindex(fix_i)
        n.model.add_constraints(reindex(sssc, sssc_dim, fix_i), "<=", rhs, name="LineX-fix-sssc_nom-line_capacity")

    logger.info(
        "Capped LineX SSSC capacity at %.3g x line capacity on %d branches (%d against the extendable s_nom).",
        ratio,
        len(sssc_ext_i),
        len(ext_i),
    )


def tighten_line_x_sssc_bound(n, snapshots, config):
    """
    Pull the ``sssc_nom`` upper bound down to what the constraints already imply.

    The bound comes from ``sssc_nom_max``, which ``prepare_network`` only clips at
    the line capacity cap, so every candidate is boxed at 10 GW while no candidate
    can exceed ``sssc_nom_max_pu * s_nom_max`` per branch nor ``sssc_tot_max`` over
    all of them together. On test_tr_4_3GVAsssc that leaves 8457 columns with a box
    3.3x wider than any feasible value, which is what the barrier scales and picks
    its starting point from. Every bound set here is implied by a constraint that is
    in the model anyway, so no feasible point is removed.

    The two caps are independent: dropping ``sssc_nom_max_pu`` must not take the
    ``sssc_tot_max`` bound with it, so this is its own step rather than a tail of
    ``add_line_x_sssc_line_capacity_constraint``.
    """
    ratio = _get_line_x_sssc_nom_max_pu(config)
    total_max = _get_line_x_sssc_total_max(config)
    if not np.isfinite(ratio) and not np.isfinite(total_max):
        return

    split = _line_x_sssc_index_split(n, "LineX SSSC bound tightening")
    if split is None:
        return
    sssc, sssc_ext_i, ext_i, fix_i = split

    line_xs = n.line_xs
    s_nom_max = line_xs.s_nom_max.reindex(sssc_ext_i).astype(float)
    s_nom = line_xs.s_nom.reindex(sssc_ext_i).astype(float)
    # against the extendable capacity the cap follows s_nom_max, otherwise the
    # branch keeps its fixed rating
    reference = pd.Series(np.inf, index=sssc_ext_i, dtype=float)
    if not ext_i.empty:
        reference.loc[ext_i] = s_nom_max.loc[ext_i]
    if not fix_i.empty:
        reference.loc[fix_i] = s_nom.loc[fix_i]

    implied = pd.Series(np.inf, index=sssc_ext_i, dtype=float)
    if np.isfinite(ratio):
        implied = ratio * reference
    if np.isfinite(total_max):
        # sssc_nom >= 0, so no single branch can exceed the total either
        implied = implied.clip(upper=total_max)

    current = sssc.upper.to_series().reindex(sssc_ext_i).astype(float)
    tightened = np.minimum(current, implied)
    changed = int((tightened < current).sum())
    if not changed:
        return

    sssc.upper = xr.DataArray(
        tightened.to_numpy(),
        coords=sssc.upper.coords,
        dims=sssc.upper.dims,
    )
    logger.info(
        "Tightened the sssc_nom upper bound on %d of %d branches to at most %.4g MW "
        "(was up to %.4g MW).",
        changed,
        len(sssc_ext_i),
        float(tightened.max()),
        float(current.max()),
    )


def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.optimization.optimize``.

    If you want to enforce additional custom constraints, this is a good
    location to add them. The arguments ``opts`` and
    ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    # Make snakemake available in function scope if it exists in global scope
    global_snakemake = globals().get("snakemake")

    # Define constraint application functions in a registry
    # Each function should take network and necessary parameters
    constraint_registry = {
        "RPS": lambda: (
            add_RPS_constraints(n, config, global_snakemake)
            if n.generators.p_nom_extendable.any()
            else None
        ),
        "REM": lambda: (
            add_regional_co2limit(n, config)
            if n.generators.p_nom_extendable.any()
            else None
        ),
        "ERM": lambda: (
            add_ERM_constraints(n, snapshots, config, global_snakemake)
            if n.generators.p_nom_extendable.any()
            else None
        ),
        "TCT": lambda: add_technology_capacity_target_constraints(n, config),
    }

    # Apply constraints based on options
    for opt in opts:
        if opt in constraint_registry:
            constraint_registry[opt]()

    # Always apply bidirectional link constraints
    add_bidirectional_link_constraints(n)

    # When representative periods are enabled, enforce cyclic closure
    # inside each representative period block for storage technologies.
    add_representative_period_storage_constraints(n, config, snapshots)
    add_electrolysis_hydrogen_target_constraint(
        n,
        snapshots,
        config,
        getattr(global_snakemake.input, "sector_costs", None) if global_snakemake else None,
        getattr(global_snakemake.input, "hydrogen_demand_share", None) if global_snakemake else None,
    )
    if config.get("scenario", {}).get("decarbonization") == "BAU":
        add_post_2032_gas_average_power_limit(n)
    add_line_x_sssc_total_max_constraint(n, snapshots, config)
    add_line_x_sssc_line_capacity_constraint(n, snapshots, config)
    tighten_line_x_sssc_bound(n, snapshots, config)


def _iterative_optimize_kwargs(cf_solving):
    """Forward explicitly configured outer-loop choices to PyPSA.

    Leaving an option unset is important: it lets the installed PyPSA version
    retain its own calibrated default instead of this workflow replacing it by
    ``None``. The step control and the convergence test are calibrated inside
    PyPSA against this workflow's own cases, so they are deliberately not
    repeated here - only whether to damp at all is a choice this repo makes.
    """
    return {
        name: cf_solving[name]
        for name in ("proximal",)
        if cf_solving.get(name) is not None
    }


def _run_standard_optimize(n, rolling_horizon, skip_iterations, cf_solving, **kwargs):
    """Run the standard PyPSA optimization path."""
    if rolling_horizon:
        kwargs["horizon"] = cf_solving.get("horizon", 365)
        kwargs["overlap"] = cf_solving.get("overlap", 0)
        n.optimize.optimize_with_rolling_horizon(**kwargs)
        status, condition = "", ""
    elif skip_iterations:
        status, condition = n.optimize(**kwargs)
    else:
        kwargs["track_iterations"] = cf_solving.get("track_iterations", False)
        kwargs["min_iterations"] = int(cf_solving.get("min_iterations", 4))
        kwargs["max_iterations"] = int(cf_solving.get("max_iterations", 6))
        kwargs["scheme"] = cf_solving.get("scheme", "slp")
        kwargs.update(_iterative_optimize_kwargs(cf_solving))
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            **kwargs,
        )

    return status, condition


def _check_optimize_status(status, condition, rolling_horizon):
    if status != "ok" and not rolling_horizon:
        logger.warning(
            f"Solving status '{status}' with termination condition '{condition}'",
        )
    condition_text = "" if condition is None else str(condition)
    if "infeasible" in condition_text:
        # n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'")


def run_optimize(n, rolling_horizon, skip_iterations, cf_solving, **kwargs):
    """Initiate the correct type of pypsa.optimize function."""

    status, condition = _run_standard_optimize(
        n,
        rolling_horizon,
        skip_iterations,
        cf_solving,
        **kwargs,
    )

    _check_optimize_status(status, condition, rolling_horizon)


def prepare_brownfield(n, planning_horizon):
    """Prepare the network for the next planning horizon by setting up brownfield constraints.
    Used for myopic foresight.

    This function:
    1. Sets minimum capacities for transmission lines and DC links
    2. Updates generator, link, and storage unit capacities
    3. Handles time-dependent data transfer between planning periods
    """
    # electric transmission grid set optimised capacities of previous as minimum
    n.lines.s_nom_min = n.lines.s_nom_opt  # for lines
    dc_i = n.links[n.links.carrier == "DC"].index
    n.links.loc[dc_i, "p_nom_min"] = n.links.loc[dc_i, "p_nom_opt"]  # for links

    for c in n.iterate_components(["Generator", "Link", "StorageUnit"]):
        nm = c.name
        # limit our components that we remove/modify to those prior to this time horizon
        c_lim = c.df.loc[n.get_active_assets(nm, planning_horizon)]

        logger.info(f"Preparing brownfield for the component {nm}")
        # attribute selection for naming convention
        attr = "p"
        # copy over asset sizing from previous period
        c_lim[f"{attr}_nom"] = c_lim[f"{attr}_nom_opt"]
        c_lim[f"{attr}_nom_extendable"] = False
        df = copy.deepcopy(c_lim)
        time_df = copy.deepcopy(c.pnl)

        for c_idx in c_lim.index:
            n.remove(nm, c_idx)

        for df_idx in df.index:
            if nm == "Generator":
                n.madd(
                    nm,
                    [df_idx],
                    carrier=df.loc[df_idx].carrier,
                    bus=df.loc[df_idx].bus,
                    p_nom_min=df.loc[df_idx].p_nom_min,
                    p_nom=df.loc[df_idx].p_nom,
                    p_nom_max=df.loc[df_idx].p_nom_max,
                    p_nom_extendable=df.loc[df_idx].p_nom_extendable,
                    ramp_limit_up=df.loc[df_idx].ramp_limit_up,
                    ramp_limit_down=df.loc[df_idx].ramp_limit_down,
                    efficiency=df.loc[df_idx].efficiency,
                    marginal_cost=df.loc[df_idx].marginal_cost,
                    capital_cost=df.loc[df_idx].capital_cost,
                    build_year=df.loc[df_idx].build_year,
                    lifetime=df.loc[df_idx].lifetime,
                    heat_rate=df.loc[df_idx].heat_rate,
                    fuel_cost=df.loc[df_idx].fuel_cost,
                    vom_cost=df.loc[df_idx].vom_cost,
                    carrier_base=df.loc[df_idx].carrier_base,
                    p_min_pu=df.loc[df_idx].p_min_pu,
                    p_max_pu=df.loc[df_idx].p_max_pu,
                    land_region=df.loc[df_idx].land_region,
                )
            else:
                n.add(nm, df_idx, **df.loc[df_idx])
        logger.info(n.consistency_check())

        # copy time-dependent
        selection = n.component_attrs[nm].type.str.contains("series")
        for tattr in n.component_attrs[nm].index[selection]:
            n.import_series_from_dataframe(time_df[tattr], nm, tattr)

    # roll over the last snapshot of time varying storage state of charge to be the state_of_charge_initial for the next time period
    n.storage_units.loc[:, "state_of_charge_initial"] = n.storage_units_t.state_of_charge.loc[planning_horizon].iloc[-1]


def solve_network(n, config, solving, opts="", **kwargs):
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    foresight = snakemake.params.foresight
    kwargs["multi_investment_periods"] = config["foresight"] == "perfect"

    kwargs["solver_options"] = solving["solver_options"][set_of_options] if set_of_options else {}
    kwargs["solver_name"] = solving["solver"]["name"]
    kwargs["extra_functionality"] = extra_functionality
    kwargs["transmission_losses"] = cf_solving.get("transmission_losses", False)
    kwargs["linearized_unit_commitment"] = cf_solving.get(
        "linearized_unit_commitment",
        False,
    )
    kwargs["assign_all_duals"] = cf_solving.get("assign_all_duals", False)

    sns_portion = cf_solving.get("snapshot_portion", None)
    if sns_portion:
        logger.info(f"Optimizing over snapshots from {sns_portion['start']} to {sns_portion['end']}")
        sns_portion = pd.date_range(start=sns_portion["start"], end=sns_portion["end"], freq="h")
        sns = n.snapshots
        sns_portion = sns[sns.get_level_values(1).isin(sns_portion)]
        sns_portion.name = "snapshot"
        kwargs["snapshots"] = sns_portion

    rolling_horizon = cf_solving.pop("rolling_horizon", False)
    skip_iterations = cf_solving.pop("skip_iterations", False)
    line_xs = getattr(n, "line_xs", pd.DataFrame())
    lines_extendable = n.lines.s_nom_extendable.any()
    line_xs_extendable = not line_xs.empty and line_xs.s_nom_extendable.any()
    if not lines_extendable and not line_xs_extendable:
        skip_iterations = True
        logger.info("No expandable lines or line_xs found. Skipping iterative solving.")

    # add to network for additional_constraints
    n.config = config
    n.opts = opts

    match foresight:
        case "perfect":
            run_optimize(n, rolling_horizon, skip_iterations, cf_solving, **kwargs)
        case "myopic":
            for i, planning_horizon in enumerate(n.investment_periods):
                sns_horizon = n.snapshots[n.snapshots.get_level_values(0) == planning_horizon]
                kwargs["snapshots"] = sns_horizon

                run_optimize(n, rolling_horizon, skip_iterations, cf_solving, **kwargs)

                if i == len(n.investment_periods) - 1:
                    logger.info(f"Final time horizon {planning_horizon}")
                    continue
                logger.info(f"Preparing brownfield from {planning_horizon}")
                prepare_brownfield(n, planning_horizon)

        case _:
            raise ValueError(f"Invalid foresight option: '{foresight}'. Must be 'perfect' or 'myopic'.")

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_network",
            case="test_tr",
            ll="v1.3",
            opts="RPS-TCT-ERM-6h",
            planning_horizons="2050",
        )
    configure_logging(snakemake)
    set_case_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    configured_opts = snakemake.params.opts
    if isinstance(configured_opts, str):
        configured_opts = [configured_opts]
    opts = [token for item in configured_opts for token in str(item).split("-") if token]
    solve_opts = snakemake.params.solving["options"]

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)

    n = prepare_network(
        n,
        solve_opts,
    )
    n = solve_network(
        n,
        config=snakemake.config,
        solving=snakemake.params.solving,
        opts=opts,
        log_fn=snakemake.log.solver,
    )

    if "ERM" in opts:
        store_ERM_duals(n)

    existing_meta = getattr(n, "meta", {})
    if not isinstance(existing_meta, dict):
        existing_meta = {}
    n.meta = {
        **snakemake.config,
        **existing_meta,
        "wildcards": dict(snakemake.wildcards),
    }
    n.export_to_netcdf(snakemake.output[0])
    with open(snakemake.output.config, "w") as file:
        yaml.dump(
            n.meta,
            file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
