"""
Per-period SSSC capacities for perfect-foresight networks with several investment periods.

pypsa-bill gives an extendable static synchronous series compensator (SSSC) on a line one
capacity ``sssc_nom`` for all periods: its compensation ``q_sssc`` may use it in every
period, PyPSA charges it in every period the line is active, and ``sssc_nom_max_pu``
limits it to a share of the line's capacity ``s_nom`` (that of the last period, see
:mod:`opts.transmission`). As pypsa-bill does for line capacity, this adds a capacity
``sssc_nom_<period>`` for every modelled period but the last that

- lies between ``sssc_nom_min`` and the next period's capacity (the last period's is
  ``sssc_nom``),
- bounds ``|q_sssc|`` in that period's snapshots,
- replaces ``sssc_nom`` in that period's investment cost, and
- is limited to ``sssc_nom_max_pu`` times the line's capacity in that period,
  ``s_nom_<period>`` where pypsa-bill stages the line's capacity, or its fixed
  ``s_nom_<period>_opt`` in the final solve of the iterative expansion,

so an SSSC is paid for, and compensates, only from the period it is built in, and it can
only grow as fast as its line. Lines whose capacity is not staged have one capacity for
all periods; ``sssc_nom_<period> <= sssc_nom`` and pypsa-bill's limit on ``sssc_nom``
already cover them.

PyPSA writes each ``sssc_nom_<period>`` to the line column ``sssc_nom_<period>_opt``,
which :func:`store_period_sssc_capacity` completes for the last period. PyPSA's statistics
still charge ``sssc_nom_opt`` in every period, as they do ``s_nom_opt`` for lines.
"""

import logging

import linopy
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from opts.transmission import modelled_periods, period_capacity_column, period_capacity_variable

logger = logging.getLogger(__name__)

NOM = "sssc_nom"
NOMINAL = "Line-sssc_nom"
COMPENSATION = "Line-q_sssc"


def _da(series: pd.Series) -> xr.DataArray:
    return xr.DataArray(series.to_numpy(), coords={"name": series.index}, dims="name")


def add_period_sssc_constraints(n: pypsa.Network, snapshots: pd.Index) -> None:
    """
    Give extendable SSSCs a capacity per investment period (see the module docstring).

    Call after :func:`opts.transmission.add_period_transmission_limits`, whose per-period
    line capacities the SSSC capacities are limited by. Does nothing for a model with
    one investment period, e.g. a myopic horizon.
    """
    m = n.model
    if NOMINAL not in m.variables:
        return
    sns, periods = modelled_periods(n, snapshots)
    if len(periods) < 2:
        return
    earlier, last = periods[:-1], periods[-1]

    c = n.components["Line"]
    static = c.static
    nominal = m[NOMINAL]
    ext = pd.Index(nominal.indexes["name"])
    ratio = static.loc[ext, "sssc_nom_max_pu"].astype(float)
    limited = ext[np.isfinite(ratio.to_numpy())]
    cost = static.loc[ext, "capital_cost_sssc"].astype(float)
    weights = n.investment_period_weightings.objective
    q = m[COMPENSATION]

    capacity = {last: nominal}
    objective = []
    for i, period in enumerate(reversed(earlier)):
        following = capacity[earlier[-i]] if i else nominal
        col = period_capacity_column(NOM, period)
        if col not in static:
            static[col] = np.nan
        cap = m.add_variables(
            lower=_da(static.loc[ext, "sssc_nom_min"].astype(float)),
            name=period_capacity_variable(c.name, NOM, period),
        )
        capacity[period] = cap
        m.add_constraints(cap - following <= 0, name=f"Line-{NOM}_{period}-monotonic")

        # the compensation of the period's snapshots stays within the period's capacity
        sns_p = sns[sns.get_level_values(0) == period]
        q_p = q.sel(name=ext, snapshot=sns_p)
        active = c.da.active.sel(name=ext, snapshot=sns_p)
        m.add_constraints(q_p + cap >= 0, name=f"Line-{NOM}_{period}-q_sssc-lower", mask=active)
        m.add_constraints(q_p - cap <= 0, name=f"Line-{NOM}_{period}-q_sssc-upper", mask=active)

        # sssc_nom_<period> <= sssc_nom_max_pu * s_nom_<period>
        line_capacity = period_capacity_variable(c.name, "s_nom", period)
        staged = limited[:0]
        if line_capacity in m.variables:
            staged = limited.intersection(m[line_capacity].indexes["name"])
            if not staged.empty:
                m.add_constraints(
                    cap.sel(name=staged) - _da(ratio[staged]) * m[line_capacity].sel(name=staged) <= 0,
                    name=f"Line-{NOM}_{period}-line_capacity",
                )
        # a line the iterative expansion fixed for its final solve keeps its capacity of the period
        column = period_capacity_column("s_nom", period)
        if column in static:
            held = limited.difference(staged)
            held = held[~static.loc[held, "s_nom_extendable"].astype(bool).to_numpy()]
            built = static.loc[held, column].astype(float)
            held = built.index[(built.notna() & (built < static.loc[held, "s_nom"] - 1e-6)).to_numpy()]
            if not held.empty:
                m.add_constraints(
                    cap.sel(name=held) <= _da(ratio[held] * static.loc[held, column].astype(float)),
                    name=f"Line-{NOM}_{period}-line_capacity-fixed",
                )

        # PyPSA charges sssc_nom in every active period; charge sssc_nom_<period> instead
        active_p = c.get_active_assets(investment_period=period).reindex(ext, fill_value=False)
        weight = cost * active_p * weights[period]
        if (weight != 0).any():
            objective.append((_da(weight) * (cap - nominal)).sum())

    if objective:
        m.add_objective(m.objective.expression + linopy.merge(objective), overwrite=True)
    logger.info(f"Staged the capacity of {len(ext)} extendable SSSCs over investment periods {periods}.")


def store_period_sssc_capacity(n: pypsa.Network, periods) -> None:
    """
    Complete the ``sssc_nom_<period>_opt`` line columns for the solved ``periods``.

    The optimisation fills them for earlier periods of a perfect-foresight solve; the last
    period, and every line it left empty (e.g. a fixed SSSC), takes ``sssc_nom_opt``. A line
    not active in a period has no SSSC capacity in it.
    """
    c = n.components["Line"]
    lines = c.static
    periods = list(periods)
    if len(periods) < 2 or "sssc_nom_opt" not in lines or c.sssc_capable.empty:
        return
    last = max(periods)
    for period in periods:
        col = period_capacity_column(NOM, period)
        installed = lines["sssc_nom_opt"].where(c.get_active_assets(investment_period=period), 0.0)
        if period == last or col not in lines:
            lines[col] = installed
        else:
            lines[col] = lines[col].fillna(installed)
