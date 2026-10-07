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
    ELECTROLYSIS_ROW_SCALE,
    add_post_2032_gas_average_power_limit,
    add_regional_co2limit,
    add_RPS_constraints,
    add_technology_capacity_target_constraints,
    record_row_scale,
    row_scale,
    store_regional_co2_duals,
)
from opts.representative_periods import (
    add_representative_period_storage_constraints,
    split_capacity_by_representative_period,
)
from opts.reserves import (
    add_ERM_constraints,
    store_ERM_duals,
)

patch_linopy_multiindex_assign()

logger_gurobi = logging.getLogger("gurobipy")
logger_gurobi.propagate = False

logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)

FLEXIBLE_ELECTROLYSIS_LINK_SUFFIX = " flexible electrolysis"
FLEXIBLE_ELECTROLYSIS_BUS_SUFFIX = " flexible electrolysis H2"

DEFAULT_ANNUAL_ELECTRICITY_TWH = 1612.0

# solving.options switch for split_capacity_by_representative_period; on by default.
SPLIT_CAPACITY_OPTION = "split_capacity_by_representative_period"


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
    """Return the validated electrolysis accounting level.

    Mirrors ``add_extra_components.flexible_electrolysis_accounting_region``:
    ``h2ptcreg`` balances electrolysis per 45V hydrogen PTC region, ``trans_grp``
    per ReEDS transmission group, ``nation`` once over the whole modelled system.
    """
    accounting_regions = ("h2ptcreg", "trans_grp", "nation")
    accounting_region = flex_config.get("accounting_region", "h2ptcreg")
    if accounting_region not in accounting_regions:
        raise ValueError(
            "flexible_electrolysis 'accounting_region' must be one of "
            f"{list(accounting_regions)}; got {accounting_region!r}.",
        )
    return accounting_region


def _read_hydrogen_shares(hydrogen_share_path, level):
    """Return the state-level hydrogen demand shares summed up to ``level``."""
    if not hydrogen_share_path:
        raise ValueError(
            "Flexible electrolysis is enabled but no hydrogen_demand_share.csv path was provided "
            "to the hydrogen target constraint.",
        )
    shares = pd.read_csv(hydrogen_share_path)
    missing = {level, "share"} - set(shares.columns)
    if missing:
        raise ValueError(f"{hydrogen_share_path} is missing required column(s): {sorted(missing)}.")
    return shares.groupby(level)["share"].sum()


def h2ptcreg_hydrogen_shares(hydrogen_share_path):
    """Return each h2ptcreg region's share of national annual hydrogen demand.

    The file stores state-level shares (derived from the hourly state hydrogen
    demand profiles); they are summed up to the ``h2ptcreg`` level here.
    """
    return _read_hydrogen_shares(hydrogen_share_path, "h2ptcreg")


def trans_grp_hydrogen_shares(n, links, link_regions, hydrogen_share_path):
    """Return each trans_grp's share of national annual hydrogen demand.

    The hydrogen shares are only known per state, and a state can straddle several
    transmission groups (TX spans ERCOT, SPP_South, MISO_South and
    WestConnect_South). Each state's share is split across the groups of its
    electrolysis buses by their ``Pd``, the same within-state weight
    ``build_eer_demand`` splits state electricity demand by, keyed on the same
    ``reeds_state``. Hydrogen thus follows the model's own electricity load. Like
    that load, it is normalised over the buses present, so in a run that covers
    only part of a state the in-network part takes the whole state's share.
    """
    state_shares = _read_hydrogen_shares(hydrogen_share_path, "st")

    missing = {"Pd", "reeds_state"} - set(n.buses.columns)
    if missing:
        raise ValueError(
            "flexible_electrolysis 'accounting_region' is 'trans_grp' but the network's buses "
            f"lack {sorted(missing)}, needed to split state hydrogen shares.",
        )
    bus0 = n.links.loc[links, "bus0"].to_numpy()
    # trim_network relabels kept external buses as "imports_<state>".
    states = pd.Series(
        n.buses.reeds_state.reindex(bus0).astype(str).str.removeprefix("imports_").to_numpy(),
        index=links,
    )
    unknown = pd.Index(states.unique()).difference(state_shares.index)
    if len(unknown):
        raise ValueError(
            f"No hydrogen demand share for state(s) {list(unknown)} in {hydrogen_share_path}.",
        )

    weight = pd.Series(
        pd.to_numeric(n.buses.Pd.reindex(bus0), errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(),
        index=links,
    )
    state_weight = weight.groupby(states).transform("sum")
    empty = sorted(states[state_weight <= 0.0].unique())
    if empty:
        # Mirrors build_eer_demand, which drops a state's electricity demand alike.
        logger.warning("No electrolysis bus carries Pd in %s; their hydrogen share is dropped.", empty)
    within_state = (weight / state_weight).where(state_weight > 0.0, 0.0)

    link_shares = states.map(state_shares) * within_state
    return link_shares.groupby(link_regions.reindex(links)).sum()


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


def add_electrolysis_electricity_target_constraint(
    n,
    snapshots,
    config,
    hydrogen_share_path,
):
    """Fix annual electrolyzer electricity demand per accounting region in every period.

    The target is an electricity quantity (TWh_e): it fixes what the electrolyzer
    fleet withdraws from the grid, not what it delivers as hydrogen. Divide by the
    ``h2 electrolysis`` / ``electricity-input`` ratio in ``simple_sector_costs.csv``
    to recover the implied hydrogen output.

    The accounting region follows ``flexible_electrolysis: accounting_region``.
    With ``h2ptcreg`` the configured national total is split across the 45V
    hydrogen PTC regions by their share of national hydrogen demand -- the fleet
    has the same electricity input per unit of hydrogen everywhere, so splitting
    the electricity by hydrogen demand is the same split. ``trans_grp`` splits it the
    same way across ReEDS transmission groups, with each state's share divided among
    its groups by bus ``Pd`` (see ``trans_grp_hydrogen_shares``). With ``nation`` a single
    constraint requires the whole electrolyzer fleet to draw the configured total.

    The electrolysis links carry ``efficiency = 0`` in the network so that the H2
    accounting buses balance trivially; the constraint therefore acts on the link
    electricity withdrawal ``p0`` directly.

    One row per region and period, accounted in GWh_e: the snapshot weightings
    divided by ``ELECTROLYSIS_ROW_SCALE``, which keeps the coefficients at O(1) and
    the right hand side near 1e5.

    Not split into per-snapshot rate rows, even though the row carries 200k
    non-zeros on test_10k. Measured, splitting cost 3.5% Factor NZ and 6.9% Factor
    Ops -- same finding as ``add_regional_co2limit``.

    ``store_electrolysis_duals`` divides the scale back out of the shadow price.
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

    total_target_twh = float(
        flex_config.get("annual_electricity_twh", DEFAULT_ANNUAL_ELECTRICITY_TWH),
    )

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

    if accounting_region == "trans_grp" and not is_national_network:
        known_groups = set(n.buses.get("trans_grp", pd.Series(dtype=object)).dropna().astype(str))
        unknown = modelled_regions.difference(known_groups)
        if len(unknown):
            raise ValueError(
                "flexible_electrolysis 'accounting_region' is 'trans_grp', but the network's "
                f"electrolysis accounting bus(es) {list(unknown)} are not transmission groups. "
                "Rebuild the network from add_extra_components after changing accounting_region.",
            )

    if accounting_region == "nation":
        # All links share one accounting bus, so the fleet total equals the sum of
        # the regional hydrogen productions, i.e. the configured national total.
        region_targets = pd.Series({"nation": total_target_twh})
    else:
        if accounting_region == "trans_grp":
            region_shares = trans_grp_hydrogen_shares(
                n,
                flexible_links,
                link_regions,
                hydrogen_share_path,
            )
        else:
            region_shares = h2ptcreg_hydrogen_shares(hydrogen_share_path)
        unknown = modelled_regions.difference(region_shares.index)
        if len(unknown):
            raise ValueError(
                f"No hydrogen demand share for {accounting_region} region(s) {list(unknown)} in "
                f"{hydrogen_share_path}. If the network was built at a different accounting level, "
                "rebuild it from add_extra_components after changing accounting_region.",
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
                fixed_annual_twh = fixed_capacity * period_hours / 1e6
                tolerance = 1e-9 * max(1.0, float(target_twh))
                if fixed_annual_twh + tolerance < float(target_twh):
                    raise ValueError(
                        f"Fixed flexible electrolysis capacity in {region} ({period}) "
                        f"can consume at most {fixed_annual_twh:.6g} TWh_e, below "
                        f"the {float(target_twh):.6g} TWh_e target.",
                    )

            # Weightings carried on the variable's own coords: slicing a snapshot
            # MultiIndex drops its name, which linopy needs to key the dimension off.
            region_p = link_p.loc[period_snapshots, region_links]
            weights_da = xr.DataArray(
                period_weights.to_numpy() / ELECTROLYSIS_ROW_SCALE,
                dims=("snapshot",),
                coords={"snapshot": region_p.coords["snapshot"]},
            )
            constraint_name = f"FlexibleElectrolysis-annual_electricity-{region}{label}"
            n.model.add_constraints(
                (region_p * weights_da).sum(),
                "=",
                float(target_twh) * 1e6 / ELECTROLYSIS_ROW_SCALE,
                name=constraint_name,
            )
            record_row_scale(n, constraint_name, ELECTROLYSIS_ROW_SCALE)

    logger.info(
        "Applied per-%s annual electrolysis electricity targets (%.1f TWh_e total) across "
        "%d region(s): %s.",
        accounting_region,
        total_target_twh,
        len(region_targets),
        ", ".join(f"{r} {t:.1f} TWh_e" for r, t in region_targets.items()),
    )


def store_electrolysis_duals(n):
    """
    Store the target's shadow price in ``n.electrolysis_electricity_price``.

    A ``pd.Series`` in currency per MWh_e, indexed by constraint name: what one more
    MWh of required electrolyser withdrawal costs. An equality, so the sign follows
    the solver's convention. The row scale is divided back out.
    """
    prefix = "FlexibleElectrolysis-annual_electricity-"
    names = [name for name in n.model.constraints if name.startswith(prefix)]
    if not names:
        return

    prices = {}
    for name in names:
        constraint = n.model.constraints[name]
        dual = getattr(constraint, "dual", None)
        if dual is None:
            logger.warning("No dual available for %s; skipping its electricity price.", name)
            continue
        prices[name] = float(np.asarray(dual).reshape(-1)[0]) / row_scale(n, name)

    if not prices:
        return
    n.electrolysis_electricity_price = pd.Series(prices, name="electrolysis_price")
    logger.info(
        "Stored %d electrolysis electricity price(s): %s.",
        len(prices),
        ", ".join(f"{k} {v:.4g}/MWh_e" for k, v in prices.items()),
    )


# ``n.meta`` key the stored duals travel under. ``export_to_netcdf`` keeps only scalar
# network attributes, so the Series and DataFrame the ``store_*`` helpers attach to the
# network are dropped on export without a warning; ``n.meta`` is written as JSON and
# survives.
POLICY_DUALS_META_KEY = "policy_duals"

# Scalar prices, one value per constraint name: attribute set by the store_* helper.
SCALAR_DUAL_ATTRS = ("regional_co2_price", "electrolysis_electricity_price", "sssc_total_max_price")


def _json_number(value):
    """A finite float, or None where JSON has no number to hold it."""
    value = float(value)
    return value if np.isfinite(value) else None


def _json_label(label):
    """A snapshot label as JSON: a timestamp as ISO text, a (period, timestep) pair as a list."""
    if isinstance(label, tuple):
        return [_json_label(part) for part in label]
    if isinstance(label, pd.Timestamp):
        return label.isoformat()
    if isinstance(label, np.generic):
        return label.item()
    return label


def policy_duals_metadata(n):
    """
    JSON-safe copy of the policy duals the ``store_*`` helpers attached to ``n``.

    Values keep the solver's sign convention, as the helpers stored them (a binding
    ``<=`` budget such as the CO2 cap comes out negative):

    ``regional_co2_price``
        Currency per tonne CO2, by constraint name.
    ``electrolysis_electricity_price``
        Currency per MWh_e, by constraint name.
    ``sssc_total_max_price``
        Currency per MVAr and year, by constraint name.
    ``erm_region_price``
        ``{"snapshots": [...], "regions": [...], "values": [[...], ...]}``, one row
        per snapshot; a ``(period, timestep)`` snapshot is a two-item list.

    A dual that was not stored is left out, so an empty dict means nothing to keep.
    """
    duals = {}
    for attr in SCALAR_DUAL_ATTRS:
        prices = getattr(n, attr, None)
        if prices is not None and not prices.empty:
            duals[attr] = {str(name): _json_number(value) for name, value in prices.items()}

    erm = getattr(n, "erm_region_price", None)
    if erm is not None and not erm.empty:
        duals["erm_region_price"] = {
            "snapshots": [_json_label(label) for label in erm.index],
            "regions": [str(region) for region in erm.columns],
            "values": [[_json_number(value) for value in row] for row in erm.to_numpy()],
        }
    return duals


def read_policy_duals(n):
    """
    Inverse of :func:`policy_duals_metadata` on a solved network read back from disk.

    Returns a dict holding whichever of ``regional_co2_price``,
    ``electrolysis_electricity_price`` (both ``pd.Series``) and ``erm_region_price``
    (``pd.DataFrame``, snapshot by region) the solve stored. ERM rows are labelled
    with ``n.snapshots`` when they cover it, otherwise rebuilt from the stored labels.
    """
    meta = getattr(n, "meta", None) or {}
    stored = meta.get(POLICY_DUALS_META_KEY, {})
    duals = {}
    for attr in SCALAR_DUAL_ATTRS:
        if attr in stored:
            duals[attr] = pd.Series(stored[attr], name=attr, dtype=float)

    if "erm_region_price" in stored:
        erm = stored["erm_region_price"]
        labels = erm["snapshots"]
        if labels and isinstance(labels[0], list):
            index = pd.MultiIndex.from_tuples(
                [(period, pd.Timestamp(timestep)) for period, timestep in labels],
            )
        else:
            index = pd.DatetimeIndex(labels)
        if len(index) == len(n.snapshots) and index.equals(n.snapshots.set_names(index.names)):
            index = n.snapshots
        else:
            index = index.set_names(n.snapshots.names)
        frame = pd.DataFrame(erm["values"], index=index, columns=erm["regions"], dtype=float)
        frame.columns.name = "erm_region"
        duals["erm_region_price"] = frame
    return duals


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


# Name of the system-wide SSSC capacity cap; store_sssc_total_max_dual reads its dual.
SSSC_TOTAL_MAX_CONSTRAINT = "LineX-sssc_tot_max"


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
        name=SSSC_TOTAL_MAX_CONSTRAINT,
    )
    logger.info("Added LineX SSSC total capacity constraint at %.3f MW.", sssc_tot_max)


def store_sssc_total_max_dual(n):
    """
    Store the shadow price of the system SSSC cap in ``n.sssc_total_max_price``.

    A ``pd.Series`` in currency per MVAr and year, indexed by constraint name: what
    one more MVAr of allowed SSSC capacity saves, net of that MVAr's own annualised
    cost, since the SSSC capex is part of the objective. A binding cap comes out
    negative in the solver's convention. The row carries no snapshot dimension, so
    the capacity split leaves it on the original variable and its dual is unchanged.
    An uncapped run has no such row and stores nothing.
    """
    name = SSSC_TOTAL_MAX_CONSTRAINT
    if name not in n.model.constraints:
        return
    dual = getattr(n.model.constraints[name], "dual", None)
    if dual is None:
        logger.warning("No dual available for %s; skipping its SSSC capacity price.", name)
        return
    price = float(np.asarray(dual).reshape(-1)[0]) / row_scale(n, name)
    n.sssc_total_max_price = pd.Series({name: price}, name="sssc_total_max_price")
    logger.info("Stored the SSSC capacity cap price: %s %.4g/MVAr/yr.", name, price)


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
    add_electrolysis_electricity_target_constraint(
        n,
        snapshots,
        config,
        getattr(global_snakemake.input, "hydrogen_demand_share", None) if global_snakemake else None,
    )
    if config.get("scenario", {}).get("decarbonization") == "BAU":
        add_post_2032_gas_average_power_limit(n)
    add_line_x_sssc_total_max_constraint(n, snapshots, config)
    add_line_x_sssc_line_capacity_constraint(n, snapshots, config)
    tighten_line_x_sssc_bound(n, snapshots, config)
    # Last, so that it sees every per-snapshot row the steps above added.
    if _capacity_split_enabled(config.get("solving", {}).get("options", {})):
        split_capacity_by_representative_period(n, config, snapshots)


def _capacity_split_enabled(cf_solving):
    return bool(cf_solving.get(SPLIT_CAPACITY_OPTION, True))


def _warn_if_aggregator_undoes_capacity_split(cf_solving, solver_name, solver_options):
    """Gurobi's aggregator substitutes the per-period capacity copies straight back."""
    if solver_name != "gurobi" or not _capacity_split_enabled(cf_solving):
        return
    aggregate = next((value for key, value in solver_options.items() if key.lower() == "aggregate"), 1)
    if aggregate != 0:
        logger.warning(
            "%s is on but Gurobi's Aggregate is %s; presolve will substitute the "
            "per-period capacity copies back and the barrier fill will not shrink. "
            "Set Aggregate: 0.",
            SPLIT_CAPACITY_OPTION,
            aggregate,
        )


def _iterative_optimize_kwargs(cf_solving):
    """Forward explicitly configured outer-loop choices to PyPSA.

    Leaving an option unset is important: it lets the installed PyPSA version
    retain its own calibrated default instead of this workflow replacing it by
    ``None``. The convergence test is calibrated inside PyPSA against this
    workflow's own cases and is deliberately not repeated here; the strength of
    the damping is a choice this repo makes, so the proximal weight and the
    bounds of its adaptive rule are forwarded when they are configured.
    """
    return {
        name: cf_solving[name]
        for name in (
            "proximal",
            "proximal_weight",
            "proximal_adaptive",
            "proximal_ceiling",
        )
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
    _warn_if_aggregator_undoes_capacity_split(cf_solving, kwargs["solver_name"], kwargs["solver_options"])
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
    # assign_duals places neither row, and both are written row-scaled.
    if "REM" in opts:
        store_regional_co2_duals(n)
    store_electrolysis_duals(n)
    store_sssc_total_max_dual(n)

    existing_meta = getattr(n, "meta", {})
    if not isinstance(existing_meta, dict):
        existing_meta = {}
    n.meta = {
        **snakemake.config,
        **existing_meta,
        "wildcards": dict(snakemake.wildcards),
    }
    # Carried in n.meta because the attributes the store_* helpers set are not exported.
    policy_duals = policy_duals_metadata(n)
    if policy_duals:
        n.meta[POLICY_DUALS_META_KEY] = policy_duals
    else:
        n.meta.pop(POLICY_DUALS_META_KEY, None)
    n.export_to_netcdf(snakemake.output[0])
    # The config output stays a config: results live in the network, not here.
    config_meta = {key: value for key, value in n.meta.items() if key != POLICY_DUALS_META_KEY}
    with open(snakemake.output.config, "w") as file:
        yaml.dump(
            config_meta,
            file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
