"""
Energy Reserve Margin (ERM) constraints for PyPSA-USA.

The ERM is a capacity adequacy constraint. For every region and snapshot it requires
that the capacity the region can call on covers its demand plus a reserve margin::

    sum over the region of  p_nom * availability_factor           (physical generators)
  + sum over the region of  energy-backed discharge potential     (storage)
  + net actual power flow into the region across its boundary     (base state)
  - actual loss on the branches inside the region                 (base state)
  + (1 + erm) * actual load-shedding dispatch
  >= (1 + erm) * gross regional demand

Only the storage dischargers need a variable of their own. Every other term is either
a linear expression in the base-state variables the model already carries or a
constant, so the ERM adds no second copy of the physical dispatch, no second nodal
balance and no second voltage law.

Load shedding
-------------
Generators with carrier ``load`` represent unserved energy. Their nominal capacity
is excluded from callable generator capacity. Their actual dispatch relaxes the ERM
requirement, making the formula equivalent to applying the reserve margin to served
demand, ``gross demand - load shedding``. With zero shedding the original capacity
adequacy policy is unchanged; the high shedding marginal cost prices any relaxation.

Availability factor
-------------------
``p_max_pu`` is used directly. It already carries the VRE resource profile *and* the
temperature-dependent thermal derates applied in ``capacity_derates.py``, so no
separate capacity-credit table is needed.

Contributions are gross: nothing is netted out for what the base state is already
doing with the same asset. A generator counts its full available capacity even while
it is dispatched, and storage counts its discharge potential even while it charges.

Storage
-------
A discharger -- a ``StorageUnit``, or a ``Link`` that discharges a ``Store`` such as
TES -- gets one reserve-state variable capped two ways: by its rating against the
shared nominal capacity variable, and by the energy the *base-state* dispatch leaves
in the device at that snapshot. The second cap is what makes the contribution
energy-backed rather than a pure capacity credit. It converts power to energy over
one *physical* timestep (``storage_elapsed_hours``), which is the snapshot weighting
only while the snapshots run chronologically.

Cross-boundary flow
-------------------
The boundary term is the base state's *actual* flow on the branches crossing the
region boundary, so an import is credited only to the extent that the base-state
power flow -- which obeys the voltage law and the branch ratings -- actually delivers
it. Only electric branches count: passive branches, and links whose two ends are both
electricity buses. A ``tes`` or ``H2`` bus carries no region label, so a TES or
electrolysis link would otherwise be mistaken for a boundary crossing.

Where losses are modelled, half of a branch's loss is charged at each end, matching
PyPSA's own nodal balance.

Internal losses
---------------
A branch with both ends in the region books both of its half-losses inside the
region, so its whole loss is subtracted from the left-hand side. This is what keeps
the requirement flush with the base-state nodal balance summed over the region:
without it, at ``erm = 0`` a fully dispatched region would pass while its own
transmission losses went unserved. Like the boundary flow the loss is read off the
base state and is not scaled by ``1 + erm``, since the reserve state carries no power
flow of its own to derive a stressed-state loss from. Link losses enter through the
efficiency; a reversible link is left out rather than counted with the wrong sign.

Requirement granularity
-----------------------
One row per region and snapshot. Overlapping regions each get their own row and all
of them bind. ``erm: {all: X}`` is shorthand for "apply X to every ReEDS zone": a
single nationwide row would have no boundary, so its flow term would vanish and the
requirement would collapse into a nationwide capacity sum. A NERC-region key such as
``erm: {PJM: X}`` is expanded the same way, into one row per ReEDS zone PJM contains,
each requiring its own demand plus margin off its own boundary flow rather than
pooling capacity across zones that intra-region transmission may not actually reach.

Note the flip side of aggregating: within a region the requirement is one sum, so
intra-regional transmission earns no adequacy value -- a region containing both a
large generator and an unreachable load pocket will pass. Use a finer region
definition where that matters.
"""

import logging

import numpy as np
import pandas as pd
from opts._helpers import get_region_buses
from opts.representative_periods import storage_elapsed_hours
from pypsa.descriptors import (
    expand_series,
    get_activity_mask,
    get_bounds_pu,
    nominal_attrs,
)
from pypsa.descriptors import (
    get_switchable_as_dense as get_as_dense,
)
from pypsa.optimization.common import reindex
from xarray import DataArray

logger = logging.getLogger(__name__)

# This name deliberately contains no hyphen. PyPSA's ``assign_duals`` splits every
# name on the first "-" and reads the left half as a component, so a
# ``GlobalConstraint-`` or ``Bus-`` prefix makes PyPSA try to write a per-snapshot
# frame into a component that has no place for it. Without a hyphen the unpacking
# raises ``ValueError``, PyPSA skips the name, and we read the dual from the linopy
# model ourselves.
ERM_REQUIREMENT = "ERM_regional_requirement"


def erm_requirement_name(region):
    """
    Name of the per-region ERM requirement constraint.

    One constraint block per region, not one block merged across an "erm_region"
    dimension: regions vary hugely in how many generator/storage/boundary terms they
    carry, and linopy's ``merge`` pads every region's term list to the size of the
    largest one when concatenating along a new dimension, which is a dense array
    proportional to (regions x snapshots x the biggest region's term count) rather
    than to the actual number of terms. For a nationwide, regionally uneven network
    that padding is what runs the writer out of memory, not the problem itself.

    Uses "_" rather than "-" to keep the no-hyphen invariant above.
    """
    return f"{ERM_REQUIREMENT}_{region}"


def _named_snapshots(n, sns):
    """
    Snapshots carrying the index name linopy turns into the ``snapshot`` dimension.

    ``n.snapshots`` is named, but a caller passing a freshly built ``MultiIndex``
    is not, which would make linopy invent a ``dim_0`` and reject the variable.
    """
    expected = getattr(n.snapshots, "name", "snapshot")
    if getattr(sns, "name", None) == expected:
        return sns

    sns = sns.copy()
    sns.name = expected
    return sns


def _erm_buses(n):
    """Electricity buses the requirement is written over."""
    return n.buses.index[n.buses.carrier == "AC"]


def _passive_branch_components(n):
    return [c for c in n.passive_branch_components if not n.df(c).empty]


def _storage_discharge_links(n):
    """
    Links discharging a ``Store`` into the electricity network (TES, hydrogen).

    These are the only links that carry a reserve state, and their reserve flow is
    additionally capped by the energy in the connected store. Charge links need no
    treatment at all: the requirement credits gross discharge potential and never nets
    out what the base state withdraws.
    """
    if n.links.empty or n.stores.empty:
        return n.links.index[:0]

    store_buses = n.stores.bus.unique()
    return n.links.index[n.links.bus0.isin(store_buses) & ~n.links.bus1.isin(store_buses)]


def _get_bus_demand(n, buses):
    """Hourly demand at each bus."""
    return (
        (-get_as_dense(n, "Load", "p_set", n.snapshots) * n.loads.sign)
        .T.groupby(n.loads.bus)
        .sum()
        .T.reindex(columns=buses, fill_value=0)
    )


def _load_shedding_generators(n):
    """Generators whose dispatch represents unserved rather than supplied energy."""
    if n.generators.empty or "carrier" not in n.generators:
        return n.generators.index[:0]
    return n.generators.index[n.generators.carrier == "load"]


def _add_erm_variable(n, sns, c, attr, index=None, lower=-np.inf):
    """
    Create a reserve-state variable on ``index``, defaulting to the whole component.

    Restricting the index is what keeps the reserve state small: of the links only
    the ones that discharge a store carry one, not every link in the network.
    """
    assets_i = n.df(c).index if index is None else pd.Index(index).rename(c)
    active = get_activity_mask(n, c, sns, assets_i) if n._multi_invest else None
    n.model.add_variables(
        lower,
        coords=[sns, assets_i],
        name=f"{c}-{attr}_ERM",
        mask=active,
    )


def _define_erm_discharge_rating(n, sns, c, attr, assets_i):
    """
    Cap a reserve-state discharge variable by the asset's own rating.

    Extendable assets are capped against the *shared* nominal capacity variable, which
    is what ties adequacy to the investment decision; fixed assets get a constant
    bound. Only the upper cap is written -- the variable is created non-negative, so
    the lower bound needs no row of its own.
    """
    m = n.model
    var = m[f"{c}-{attr}_ERM"]
    ext_dim = f"{c}-ext"

    ext_i = assets_i.intersection(n.get_extendable_i(c)).rename(ext_dim)
    fix_i = assets_i.difference(ext_i).rename(c)

    if not ext_i.empty:
        max_pu = DataArray(get_bounds_pu(n, c, sns, ext_i, attr)[1])
        capacity = m[f"{c}-{nominal_attrs[c]}"].sel({ext_dim: ext_i})
        m.add_constraints(
            [(1, reindex(var, c, ext_i)), (-max_pu, capacity)],
            "<=",
            0,
            name=f"{c}-ext-{attr}-upper_ERM",
            mask=get_activity_mask(n, c, sns, ext_i),
        )

    if fix_i.empty:
        return

    max_pu = get_bounds_pu(n, c, sns, fix_i, attr)[1]
    nominal = n.df(c)[nominal_attrs[c]].reindex(fix_i)
    m.add_constraints(
        reindex(var, c, fix_i),
        "<=",
        max_pu.mul(nominal),
        name=f"{c}-fix-{attr}-upper_ERM",
        mask=get_activity_mask(n, c, sns, fix_i),
    )


def define_erm_storage_unit_capacity(n, sns, config=None):
    """
    Bound reserve-state storage discharge by the rating and the base-state SOC.

    The variable is denominated in delivered electricity, matching ``p_dispatch``.
    Discharging at the reserve level for one snapshot must not draw more energy than
    the state of charge holds at that snapshot, which is what makes the contribution
    energy-backed rather than a pure capacity credit.

    "For one snapshot" is a physical duration, so the hours come from
    ``storage_elapsed_hours`` rather than from ``snapshot_weightings.stores``: under
    representative periods the weighting is a cluster count, and using it here would
    demand the state of charge back a whole cluster's worth of discharge and strip
    storage of most of its adequacy contribution.
    """
    c = "StorageUnit"
    assets = n.df(c)
    if assets.empty:
        return

    m = n.model
    _add_erm_variable(n, sns, c, "p_dispatch", lower=0)
    _define_erm_discharge_rating(n, sns, c, "p_dispatch", assets.index)

    eh = expand_series(storage_elapsed_hours(n, sns, config), assets.index)
    eff_dispatch = get_as_dense(n, c, "efficiency_dispatch", sns)
    m.add_constraints(
        DataArray(eh / eff_dispatch) * m[f"{c}-p_dispatch_ERM"] - m[f"{c}-state_of_charge"],
        "<=",
        0,
        name=f"{c}-p_dispatch-soc-upper_ERM",
        mask=DataArray(get_activity_mask(n, c, sns)),
    )


def define_erm_store_link_capacity(n, sns, links_i, config=None):
    """
    Cap the reserve flow of storage discharge links by the connected store energy.

    The variable is denominated on ``bus0`` like ``Link-p``; the nodal balance
    applies the link efficiency when injecting at ``bus1``. The hours backing one
    snapshot of discharge are physical, for the reason given in
    ``define_erm_storage_unit_capacity``.
    """
    m = n.model
    c = "Link"
    links = n.links.loc[links_i]

    store_by_bus = pd.Series(n.stores.index.to_numpy(), index=n.stores.bus.to_numpy())
    store_by_bus = store_by_bus[~store_by_bus.index.duplicated(keep="first")]
    link_to_store = links.bus0.map(store_by_bus)

    reserve = m[f"{c}-p_ERM"].sel({c: links_i})
    eh = DataArray(
        expand_series(storage_elapsed_hours(n, sns, config), links_i).to_numpy(),
        coords=reserve.coords,
        dims=reserve.dims,
    )

    store_indexer = DataArray(link_to_store.to_numpy(), dims=[c], coords={c: links_i})
    store_e = m["Store-e"].sel(Store=store_indexer)

    store_activity = get_activity_mask(n, "Store", sns, pd.Index(link_to_store.unique()))
    store_activity.columns.name = "Store"
    store_active = DataArray(store_activity).sel(Store=store_indexer)
    link_active = DataArray(get_activity_mask(n, c, sns, links_i))

    m.add_constraints(
        eh * reserve - store_e,
        "<=",
        0,
        name=f"{c}-p-store-upper_ERM",
        mask=link_active & store_active,
    )


def define_erm_discharge_variables(n, sns, config=None):
    """
    Create the reserve state: the discharge potential of every storage device.

    Nothing else needs a variable. Generator contributions ride on the shared
    ``p_nom``, and the cross-boundary term reads the base-state flows.
    """
    if not n.storage_units.empty:
        define_erm_storage_unit_capacity(n, sns, config)

    if n.links.empty:
        return

    discharge_i = _storage_discharge_links(n)
    if discharge_i.empty:
        return

    _add_erm_variable(n, sns, "Link", "p", index=discharge_i, lower=0)
    _define_erm_discharge_rating(n, sns, "Link", "p", discharge_i)
    define_erm_store_link_capacity(n, sns, discharge_i, config)


def _erm_electric_link_i(n, buses):
    """
    Links that move electricity, i.e. both ends are electricity buses.

    A TES or electrolysis link has one end on a ``tes`` or ``H2`` bus, which carries no
    region label, so without this filter every one of them would look like a boundary
    crossing.
    """
    if n.links.empty:
        return n.links.index[:0]
    return n.links.index[n.links.bus0.isin(buses) & n.links.bus1.isin(buses)]


def _erm_boundary_flow(n, sns, region_buses, electric_links_i):
    """
    Net actual power flow into a region across its boundary, from base-state variables.

    A branch with exactly one end in the region crosses the boundary. Its contribution
    is the injection the base-state nodal balance books at the in-region end: ``+s`` at
    ``bus1`` and ``-s`` at ``bus0`` for a passive branch, ``+efficiency * p`` at
    ``bus1`` and ``-p`` at ``bus0`` for a link. Where losses are modelled, half the
    branch loss is charged at each end, exactly as PyPSA's own nodal balance does.

    Returns
    -------
    list of linopy.LinearExpression
        One term per branch component, each already summed over the crossing branches.
    """
    m = n.model
    terms = []

    for c in _passive_branch_components(n):
        df = n.df(c)
        into = df.bus1.isin(region_buses)
        crossing = df.index[into ^ df.bus0.isin(region_buses)]
        if crossing.empty:
            continue

        active = DataArray(get_activity_mask(n, c, sns, crossing))
        sign = DataArray(
            pd.Series(np.where(into[crossing], 1.0, -1.0), index=crossing.rename(c)),
        )
        terms.append((sign * m[f"{c}-s"].sel({c: crossing})).where(active).sum(c))

        if f"{c}-loss" in m.variables:
            loss = m[f"{c}-loss"].sel({c: crossing})
            terms.append((-0.5 * loss).where(active).sum(c))

    if len(electric_links_i):
        c = "Link"
        df = n.links.loc[electric_links_i]
        into = df.bus1.isin(region_buses)
        crossing = df.index[into ^ df.bus0.isin(region_buses)]
        if not crossing.empty:
            efficiency = get_as_dense(n, c, "efficiency", sns)[crossing]
            coef = pd.DataFrame(
                np.where(into[crossing].to_numpy(), efficiency.to_numpy(), -1.0),
                index=efficiency.index,
                columns=crossing.rename(c),
            )
            active = DataArray(get_activity_mask(n, c, sns, crossing))
            terms.append((DataArray(coef) * m[f"{c}-p"].sel({c: crossing})).where(active).sum(c))

    return terms


def _erm_internal_loss(n, sns, region_buses, electric_links_i):
    """
    Base-state loss on the branches with *both* ends inside the region, as a negative
    term on the left-hand side.

    Summing PyPSA's nodal balance over the region's buses leaves the identity::

        in-region dispatch + boundary net inflow - in-region branch loss = demand

    The boundary term already carries the half-loss a crossing branch books at its
    in-region end, but an internal branch books both halves inside the region and so
    contributes its *whole* loss. Without this term the requirement is slack by
    exactly that amount: at ``erm = 0`` a fully dispatched region would still pass
    while its own transmission losses go unserved.

    The loss is read off the base state and is *not* scaled by ``1 + erm``, for the
    same reason the boundary flow is not: the reserve state has no power flow of its
    own to derive a stressed-state loss from, and the base state is the only
    consistent source for both terms.

    Returns
    -------
    list of linopy.LinearExpression
        One term per branch component, each already summed over the internal branches.
    """
    m = n.model
    terms = []

    for c in _passive_branch_components(n):
        if f"{c}-loss" not in m.variables:
            continue

        df = n.df(c)
        internal = df.index[df.bus0.isin(region_buses) & df.bus1.isin(region_buses)]
        if internal.empty:
            continue

        active = DataArray(get_activity_mask(n, c, sns, internal))
        loss = m[f"{c}-loss"].sel({c: internal})
        terms.append((-1.0 * loss).where(active).sum(c))

    if len(electric_links_i):
        c = "Link"
        df = n.links.loc[electric_links_i]
        internal = df.index[df.bus0.isin(region_buses) & df.bus1.isin(region_buses)]

        # A link's loss is ``(1 - efficiency) * p``, which is only a loss while
        # ``p >= 0``. A reversible link flowing backwards would turn the term into a
        # credit, so it is dropped rather than counted with the wrong sign.
        if not internal.empty:
            reversible = internal[get_as_dense(n, c, "p_min_pu", sns)[internal].min() < 0]
            if not reversible.empty:
                logger.warning(
                    "%d reversible link(s) inside an ERM region are excluded from the "
                    "internal loss term: %s",
                    len(reversible),
                    ", ".join(reversible[:5]),
                )
                internal = internal.difference(reversible)

        if not internal.empty:
            efficiency = get_as_dense(n, c, "efficiency", sns)[internal]
            coef = -(1.0 - efficiency)
            coef.columns = coef.columns.rename(c)
            active = DataArray(get_activity_mask(n, c, sns, internal))
            terms.append((DataArray(coef) * m[f"{c}-p"].sel({c: internal})).where(active).sum(c))

    return terms


def _masked_factor(n, c, sns, assets_i, factor):
    """
    A per-asset, per-snapshot ``factor`` restricted to ``assets_i`` and zeroed where
    the asset is inactive, ready to become the coefficient of a requirement term.

    Both axis names are set explicitly rather than inherited. Selecting columns out of
    a frame keeps the *original* index name, so a subset taken with a ``-ext`` index
    would still be labelled with the plain component name and would broadcast against
    the capacity variable instead of lining up with it. And aligning two frames
    rebuilds the index, dropping the ``snapshot`` name that a snapshot MultiIndex
    carries alongside its level names -- which is what xarray keys the dimension off.
    """
    active = get_activity_mask(n, c, sns, assets_i)
    masked = factor[assets_i].where(active, 0.0)
    masked.index.name = active.index.name
    masked.columns.name = assets_i.name
    return masked


def define_erm_regional_requirements(n, sns, regions, buses):
    """
    Require each region's callable capacity to cover its demand plus its margin.

    One row per region and snapshot::

        physical generator capacity * availability
      + storage discharge potential
      + net actual boundary flow
      - actual loss on the branches inside the region
      + (1 + erm) * actual load-shedding dispatch
      >= (1 + erm) * gross regional demand

    Fixed capacity contributes a constant and is moved to the right-hand side, so the
    row only carries the extendable capacities, the reserve-state discharge variables
    and the base-state boundary flows and losses.

    Parameters
    ----------
    n : pypsa.Network
    sns : pd.Index
    regions : dict
        ``{region_name: (bus_index, erm_value)}``, only regions with a positive margin.
        Overlapping regions each get their own row and all of them bind.
    buses : pd.Index
        Electricity buses the requirement is written over.
    """
    if not regions:
        return

    m = n.model
    demand = _get_bus_demand(n, buses).loc[sns]
    electric_links_i = _erm_electric_link_i(n, buses)

    load_shedding_i = _load_shedding_generators(n)
    physical_gen_i = n.generators.index.difference(load_shedding_i)
    gen_ext_i = n.get_extendable_i("Generator").intersection(physical_gen_i)
    gen_availability = None if n.generators.empty else get_as_dense(n, "Generator", "p_max_pu", sns)

    discharge_i = _storage_discharge_links(n) if not n.links.empty else n.links.index[:0]
    link_efficiency = get_as_dense(n, "Link", "efficiency", sns) if len(discharge_i) else None

    added = 0
    for name, (region_buses, erm_value) in regions.items():
        terms = []
        offset = pd.Series(0.0, index=sns)

        if gen_availability is not None:
            gens_i = physical_gen_i[
                n.generators.bus.reindex(physical_gen_i).isin(region_buses)
            ]
            ext_i = gens_i.intersection(gen_ext_i).rename("Generator-ext")
            fix_i = gens_i.difference(ext_i).rename("Generator")

            if not ext_i.empty:
                available = _masked_factor(n, "Generator", sns, ext_i, gen_availability)
                capacity = m["Generator-p_nom"].sel({"Generator-ext": ext_i})
                terms.append((DataArray(available) * capacity).sum("Generator-ext"))

            if not fix_i.empty:
                available = _masked_factor(n, "Generator", sns, fix_i, gen_availability)
                offset = offset + available.mul(n.generators.p_nom.reindex(fix_i)).sum(axis=1)

        # A carrier="load" generator is unserved demand, not firm capacity. Its
        # dispatch reduces served load before the reserve margin is applied, which
        # appears as a positive (1 + erm) term on the left-hand side.
        region_shedding_i = load_shedding_i[
            n.generators.bus.reindex(load_shedding_i).isin(region_buses)
        ]
        if not region_shedding_i.empty:
            terms.append(
                (1.0 + erm_value)
                * m["Generator-p"]
                .sel(Generator=region_shedding_i)
                .sum("Generator"),
            )

        if not n.storage_units.empty:
            su_i = n.storage_units.index[n.storage_units.bus.isin(region_buses)]
            if not su_i.empty:
                terms.append(m["StorageUnit-p_dispatch_ERM"].sel(StorageUnit=su_i).sum("StorageUnit"))

        if len(discharge_i):
            # Link-p is denominated on bus0, so what reaches the electricity bus on
            # bus1 -- the side that has to be inside the region -- is p * efficiency.
            in_region = n.links.bus1.reindex(discharge_i).isin(region_buses).to_numpy()
            links_i = discharge_i[in_region]
            if len(links_i):
                efficiency = _masked_factor(n, "Link", sns, links_i, link_efficiency)
                terms.append((DataArray(efficiency) * m["Link-p_ERM"].sel(Link=links_i)).sum("Link"))

        terms.extend(_erm_boundary_flow(n, sns, region_buses, electric_links_i))
        terms.extend(_erm_internal_loss(n, sns, region_buses, electric_links_i))

        if not terms:
            logger.warning(f"No assets or boundary branches for ERM region {name}. Skipping.")
            continue

        lhs = terms[0]
        for term in terms[1:]:
            lhs = lhs + term

        region_rhs = demand[region_buses].sum(axis=1) * (1.0 + erm_value) - offset
        region_rhs.index.name = "snapshot"

        # Added one region at a time, rather than merged into a single constraint
        # block with an "erm_region" dimension: regions vary hugely in how many
        # generator/storage/boundary terms they carry, and merging along a new
        # dimension pads every region's term list to the size of the largest one,
        # which blows up memory for a nationwide network without adding any real
        # constraint content.
        m.add_constraints(lhs, ">=", DataArray(region_rhs), name=erm_requirement_name(name))
        added += 1

    if not added:
        return


def _expand_all_to_reeds_zones(n, erm_dict, buses):
    """
    Turn an ``all`` entry, and any NERC-region entry, into one entry per ReEDS zone.

    A region spanning several ReEDS zones has no boundary *between* those zones, so a
    single row over the whole region earns no adequacy value for the transmission
    between them (see the module docstring). ``all: X`` therefore means "apply X to
    every ReEDS zone", and a NERC-region key such as ``PJM: X`` means "apply X to
    every ReEDS zone PJM contains" -- both force each zone to individually cover its
    own demand plus reserve margin using its own boundary flow, rather than pooling
    capacity across zones that isn't reachable without binding transmission.

    Precedence where the same zone is reachable more than one way: an explicit
    ReEDS-zone key always wins, a NERC-region key wins over ``all``, and ``all`` fills
    in whatever is left. Any other key (state, country, interconnect, a bare bus name)
    is passed through unchanged.
    """
    if "reeds_zone" not in n.buses.columns:
        if "all" in erm_dict:
            logger.warning(
                "ERM 'all' cannot be expanded: buses carry no 'reeds_zone' column. "
                "Keeping 'all' as a single nationwide region.",
            )
        return erm_dict

    zone_labels = n.buses.reeds_zone.reindex(buses).dropna().astype(str)
    reeds_zones = set(zone_labels[zone_labels != ""].unique())
    if not reeds_zones:
        if "all" in erm_dict:
            logger.warning(
                "ERM 'all' cannot be expanded: no ReEDS zones on the electricity buses. "
                "Keeping 'all' as a single nationwide region.",
            )
        return erm_dict

    zones_by_nerc_region = {}
    if "nerc_reg" in n.buses.columns:
        pairs = n.buses.loc[buses, ["reeds_zone", "nerc_reg"]].dropna().astype(str)
        pairs = pairs[(pairs.reeds_zone != "") & (pairs.nerc_reg != "")]
        zones_by_nerc_region = pairs.groupby("nerc_reg")["reeds_zone"].apply(set).to_dict()

    expanded = {}

    # Lowest precedence: "all" fills every zone.
    if "all" in erm_dict:
        for zone in reeds_zones:
            expanded[zone] = erm_dict["all"]

    # Medium precedence: a NERC-region key fills the zones it contains.
    for key, value in erm_dict.items():
        if key == "all" or key in reeds_zones:
            continue
        for zone in zones_by_nerc_region.get(key, ()):
            expanded[zone] = value

    # Highest precedence: an explicit ReEDS-zone key always wins.
    for key, value in erm_dict.items():
        if key in reeds_zones:
            expanded[key] = value

    # Anything else (state, country, interconnect, a bare bus name) passes through.
    for key, value in erm_dict.items():
        if key == "all" or key in reeds_zones or key in zones_by_nerc_region:
            continue
        expanded.setdefault(key, value)

    return expanded


def add_ERM_constraints(
    n,
    snapshots,
    config=None,
    snakemake=None,
    regional_erm_data=None,
    transmission_losses=None,
):
    """
    Add Energy Reserve Margin (ERM) constraints for regional capacity adequacy.

    For each region and snapshot, the region's generator capacity times its
    availability factor, plus the energy-backed discharge potential of its storage,
    plus the base state's actual net flow across the region boundary, less the
    base-state loss on the branches inside the region, has to cover demand plus the
    region's reserve margin. Only the storage dischargers carry a
    reserve-state variable; see the module docstring for the full formulation.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object.
    snapshots : pd.Index
        Snapshots the requirement is written on.
    config : dict, optional
        Configuration dictionary containing ``electricity.erm``. Required if
        ``regional_erm_data`` is not provided. Also read for
        ``clustering.temporal.representative_periods``, which decides whether the
        hours backing a storage discharge come from the snapshot weightings or from
        the physical timestep; without it the network's own representative-period
        metadata decides.
    snakemake : snakemake object, optional
        Not used, kept for API compatibility.
    regional_erm_data : dict, optional
        Direct input of ERM requirements as ``{region_name: erm_value}``. If provided,
        this takes precedence over config data.
    transmission_losses : int, optional
        Not used. Whether the boundary and internal-loss terms carry a loss share is
        read off the base state, which is the only thing they can consistently follow.
    """
    # Get ERM data: dict {region_name: erm_value}
    # Default to 15% reserve margin for all regions if not specified
    default_erm = {"all": 0.15}

    if regional_erm_data is not None:
        erm_dict = regional_erm_data
    elif config is not None and config.get("electricity", {}).get("erm"):
        erm_dict = config["electricity"]["erm"]
    else:
        logger.info("No ERM configuration provided. Using default 0.15 for every ReEDS zone.")
        erm_dict = default_erm

    buses = _erm_buses(n)
    if buses.empty:
        logger.warning("No AC buses found. Skipping ERM constraints.")
        return

    snapshots = _named_snapshots(n, snapshots)
    erm_dict = _expand_all_to_reeds_zones(n, erm_dict, buses)

    regions = {}
    for region_name, erm_value in erm_dict.items():
        region_buses = get_region_buses(n, [region_name.strip()])
        region_buses = buses.intersection(region_buses.index)

        if region_buses.empty:
            logger.warning(f"No buses matched ERM region {region_name}. Skipping.")
            continue
        if erm_value <= 0:
            logger.info(f"ERM region {region_name} has a zero reserve level. Skipping.")
            continue

        regions[region_name] = (region_buses, erm_value)

    if not regions:
        logger.warning("No ERM region carries a positive reserve level. Skipping ERM constraints.")
        return

    logger.info("Added %d ERM constraints.", len(regions))

    define_erm_discharge_variables(n, snapshots, config)
    define_erm_regional_requirements(n, snapshots, regions, buses)


def store_ERM_duals(n):
    """
    Store Energy Reserve Margin (ERM) results if ERM constraints are activated.

    ``n.erm_region_price``
        Dual of the regional requirement, a ``snapshot`` by region frame. This is the
        marginal cost of tightening that region's reserve margin.

    The dischargers' reserve-state dispatch needs no handling here: those variables
    carry a real component prefix (``StorageUnit-p_dispatch_ERM`` lands in
    ``n.storage_units_t``, ``Link-p_ERM`` in ``n.links_t``), so PyPSA exports them by
    itself. Every other term of the requirement is a base-state variable that is
    already exported, or a constant.
    """
    logger.info("Storing ERM data from optimization results")
    constraints = n.model.constraints

    prefix = f"{ERM_REQUIREMENT}_"
    regions = [name[len(prefix) :] for name in constraints if name.startswith(prefix)]
    if not regions:
        return

    # Read each region's dual off its own constraint. ``model.dual[name]`` would
    # instead build a Dataset of *every* constraint's duals and pick one column out of
    # it -- once per region. The ERM blocks are one per region and carry no common
    # dimension beyond ``snapshot``, so that join is never exact: linopy falls back to
    # an outer join and warns "Coordinates across variables not equal" every time.
    region_dual = pd.DataFrame(
        {region: constraints[erm_requirement_name(region)].dual.to_pandas() for region in regions},
    )
    region_dual.index = n.snapshots
    region_dual.columns.name = "erm_region"
    n.erm_region_price = region_dual
