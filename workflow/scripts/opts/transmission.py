"""
Per-period transmission capacities and expansion limits for networks with several investment periods.

pypsa-bill gives every extendable line with ``s_nom_staged`` and link with ``p_nom_staged``
one capacity per investment period in a perfect-foresight optimisation
(:mod:`pypsa.optimization.staged`): ``s_nom_<period>`` for every period but the last, whose
capacity is ``s_nom``. These capacities do not decrease over the periods, bound the flow in
their period and replace ``s_nom`` in their period's investment cost, so a branch is paid for,
and carries flow, only from the period it is built in; PyPSA writes them to the columns
``s_nom_<period>_opt``. :func:`mark_staged_transmission` stages the AC lines and the AC/DC
links, :func:`add_period_transmission_limits` adds what is specific to PyPSA-USA.
"""

import logging

import linopy
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from opts.bidirectional_link import bidirectional_pairs
from pypsa.optimization.staged import (
    capacity_variables,
    period_capacity_column,
    period_capacity_variable,
)

logger = logging.getLogger(__name__)

# GlobalConstraint type written by prepare_network -> branch attribute that weights capacity.
# An infinite constant marks an `opt` period.
PERIOD_LIMIT_TYPES = {
    "transmission_volume_expansion_limit_per_period": "length",
    "transmission_expansion_cost_limit_per_period": "capital_cost",
}
BRANCHES = {"Line": ("s_nom", "s"), "Link": ("p_nom", "p")}
STAGED_SWITCH = {"Line": "s_nom_staged", "Link": "p_nom_staged"}

__all__ = [
    "add_period_transmission_limits",
    "has_period_transmission_limits",
    "installed_capacity",
    "mark_staged_transmission",
    "modelled_periods",
    "period_capacity_column",
    "period_capacity_variable",
    "store_period_transmission_capacity",
    "vintaged_branches",
]


def _period_limits(n: pypsa.Network) -> pd.DataFrame:
    glcs = n.global_constraints
    return glcs[glcs.type.isin(PERIOD_LIMIT_TYPES)]


def has_period_transmission_limits(n: pypsa.Network) -> bool:
    return not _period_limits(n).empty


def modelled_periods(n: pypsa.Network, snapshots: pd.Index) -> tuple[pd.Index, list]:
    """Snapshots and investment periods of the model being built (a myopic horizon models one period)."""
    m = n.model
    flows = [v for v in ("Line-s", "Link-p") if v in m.variables]
    sns = m.variables[flows[0]].indexes["snapshot"] if flows else snapshots
    return sns, list(pd.Index(sns.get_level_values(0)).unique())


def installed_capacity(c, branches: pd.Index, nom: str, period: int, last: int) -> pd.Series:
    """
    Capacity of the fixed ``branches`` in ``period``, as the per-period limits count it.

    That is ``nom``, or for an earlier period the value of an earlier solve of it
    (``s_nom_<period>_opt``) where there is one.
    """
    installed = c.static.loc[branches, nom]
    col = period_capacity_column(nom, period)
    if period != last and col in c.static:
        installed = c.static.loc[branches, col].fillna(installed)
    return installed


def vintaged_branches(names, periods) -> pd.Index:
    """
    Branches already split into vintages: ``<name>`` plus copies ``<name>_<period>``.

    cluster_network adds such copies for the transport model's interface links, each
    built in its own period, so their single capacity already grows period by period.
    """
    names = pd.Index(names)
    if names.empty:
        return names
    vintaged = []
    for period in periods:
        suffix = f"_{period}"
        copies = names[names.str.endswith(suffix)]
        bases = pd.Index(copies.str[: -len(suffix)])
        has_base = bases.isin(names)
        vintaged += [*copies[has_base], *bases[has_base]]
    return pd.Index(vintaged).unique()


def _da(series: pd.Series) -> xr.DataArray:
    return xr.DataArray(series.to_numpy(), coords={"name": series.index}, dims="name")


def _transmission(c) -> pd.Index:
    """AC lines and AC/DC links, the branches the ll limit covers."""
    return c.static.index[c.static.carrier.isin(["AC", "DC"])]


def mark_staged_transmission(n: pypsa.Network) -> None:
    """
    Give the AC lines and AC/DC links one capacity per investment period (pypsa-bill).

    Sets ``s_nom_staged`` / ``p_nom_staged``: on for the branches the ll limit covers, off
    for every other branch and for branches split into vintages (see
    :func:`vintaged_branches`), which keep their single capacity since it already applies
    from their build year. Takes effect in a perfect-foresight optimisation over several
    investment periods.
    """
    for c_name, switch in STAGED_SWITCH.items():
        c = n.components[c_name]
        if c.static.empty:
            continue
        transmission = c.static.index.isin(_transmission(c))
        vintaged = c.static.index.isin(vintaged_branches(c.static.index, n.investment_periods))
        c.static[switch] = transmission & ~vintaged


def _tie_bidirectional(n, cap, period):
    """Like add_bidirectional_link_constraints: a '_fwd'/'_rev' pair expands together in every period."""
    pairs = bidirectional_pairs(cap.indexes["name"])
    if not pairs:
        return
    fwd, rev = pd.Index(list(pairs)), pd.Index(list(pairs.values()))
    existing = n.links.loc[fwd, "p_nom"].to_numpy() - n.links.loc[rev, "p_nom"].to_numpy()
    lhs = cap.sel(name=fwd) - cap.sel(name=rev).assign_coords(name=fwd)
    n.model.add_constraints(lhs == _da(pd.Series(existing, index=fwd)), name=f"Link-p_nom_{period}-bidirectional")


def add_period_transmission_limits(n: pypsa.Network, snapshots: pd.Index) -> None:
    """
    Tie bidirectional links in each period and cap the transmission capacity of each period.

    pypsa-bill builds the per-period capacities (see the module docstring); this adds

    - for the '_fwd'/'_rev' pairs of bidirectional links, the tie of
      add_bidirectional_link_constraints for each period's capacity, and
    - one constraint per per-period GlobalConstraint, which caps the length- or
      cost-weighted capacity of the AC lines and AC/DC links active in its period: each
      staged branch with its capacity of the period, every other extendable branch with
      its single capacity, and the fixed ones as a constant.

    PyPSA's final iteration of ``optimize_transmission_expansion_iteratively`` fixes the
    lines at ``s_nom_opt``; pypsa-bill keeps their earlier-period flows within
    ``s_nom_<period>_opt``, and the limits count that capacity for them.
    """
    m = n.model
    _, periods = modelled_periods(n, snapshots)
    last = periods[-1]

    for period in periods[:-1]:
        var = period_capacity_variable("Link", "p_nom", period)
        if var in m.variables:
            _tie_bidirectional(n, m[var], period)

    limits = _period_limits(n)
    for name, glc in limits.iterrows():
        period = int(glc.investment_period)
        if period not in periods or not np.isfinite(glc.constant):
            continue
        attr = PERIOD_LIMIT_TYPES[glc.type]
        lhs, constant = [], 0.0
        for c_name, (nom, _) in BRANCHES.items():
            c = n.components[c_name]
            if c.static.empty:
                continue
            active = c.get_active_assets(investment_period=period)
            branches = _transmission(c).intersection(active.index[active])
            for capacity, names in capacity_variables(n, c_name, period, periods):
                ext = names.intersection(branches)
                if not ext.empty:
                    lhs.append((_da(c.static.loc[ext, attr]) * capacity.sel(name=ext)).sum())
                    branches = branches.difference(ext)
            installed = installed_capacity(c, branches, nom, period, last)
            constant += float(installed @ c.static.loc[branches, attr])
        if not lhs:
            continue
        m.add_constraints(linopy.merge(lhs), "<=", glc.constant - constant, name=f"GlobalConstraint-{name}")


def store_period_transmission_capacity(n: pypsa.Network, periods) -> None:
    """
    Complete the ``s_nom_<period>_opt`` / ``p_nom_<period>_opt`` columns for the solved ``periods``.

    The optimisation fills them for earlier periods of a perfect-foresight solve; the
    last solved period, and every branch it left empty, takes ``s_nom_opt``. A branch
    not active in a period (e.g. a later vintage) has no capacity in it.
    """
    last = max(periods)
    for c_name, (nom, _) in BRANCHES.items():
        c = n.components[c_name]
        if c.static.empty:
            continue
        for period in periods:
            col = period_capacity_column(nom, period)
            installed = c.static[f"{nom}_opt"].where(c.get_active_assets(investment_period=period), 0.0)
            if period == last or col not in c.static:
                c.static[col] = installed
            else:
                c.static[col] = c.static[col].fillna(installed)
