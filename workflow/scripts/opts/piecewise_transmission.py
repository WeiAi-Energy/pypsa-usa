"""
Convex piecewise-linear expansion costs for AC lines.

A line with a cost curve pays for capacity added above its existing capacity
``s_nom_piecewise_base`` (its ``s_nom`` when the curve is attached) segment by segment:
``c_1`` per MW for the first ``b_1`` MW, ``c_2`` for the next ``b_2 - b_1`` MW, and so on.
The costs are in the units of ``capital_cost`` (annualised currency per MW) and must not
decrease from one segment to the next. The curve is then convex and the problem stays
linear: the solver fills the cheapest segments first. Capacity beyond the last
breakpoint cannot be built unless that breakpoint is ``inf``.

The curve replaces the line's linear ``capital_cost`` for added capacity in

- the objective, for every capacity the line has in the model: ``s_nom`` and, where
  pypsa-bill stages the line's capacity over several investment periods
  (:mod:`opts.transmission`), each ``s_nom_<period>``, each charged in the periods it
  applies to. As capacity is cumulative, a period pays
  the curve up to the capacity installed by then, so what an earlier period built
  takes the cheaper segments;
- the cost-weighted transmission limits (``lc``): PyPSA's
  ``transmission_expansion_cost_limit`` and the per-period
  ``transmission_expansion_cost_limit_per_period``.

Existing capacity keeps ``capital_cost``: the reference of an ``lc`` limit is unchanged
and the line counts ``capital_cost * s_nom_piecewise_base + curve(added)`` against it.
Volume limits (``lv``) weight capacity by length and are unaffected.

Formulation: for each capacity variable ``S`` of a line, segment variables
``0 <= delta_k <= b_k - b_{k-1}`` with ``sum_k delta_k = S - base``. PyPSA (and
pypsa-bill for each period's capacity) already charge ``capital_cost * S``; adding
``(c_k - capital_cost) * delta_k`` to the objective and to each ``lc`` limit turns that
into ``capital_cost * base + sum_k c_k * delta_k``, so ``capital_cost`` itself is not
touched.

PyPSA's statistics still price lines at ``capital_cost``; after the solve the column
``piecewise_expansion_cost`` (and ``piecewise_expansion_cost_<period>``) holds the cost
of each line's added capacity along its curve.

The iterative expansion (``optimize_transmission_expansion_iteratively``) fixes the lines
for its final solve and adds their investment above ``s_nom_min`` to the objective at
``capital_cost``; :func:`correct_iterated_objective` charges the curve lines along their
curves instead.
"""

import logging
import re

import linopy
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from opts.transmission import (
    installed_capacity,
    modelled_periods,
    period_capacity_column,
    period_capacity_variable,
)
from pypsa.optimization.staged import modelled_periods as modelled_periods_of_solve
from pypsa.optimization.staged import staged_branches

logger = logging.getLogger(__name__)

CURVE_COLUMNS = ["line", "breakpoint", "capital_cost"]
SEGMENT = "segment"
BASE_COLUMN = "s_nom_piecewise_base"
COST_COLUMN = "piecewise_expansion_cost"
NOMINAL = "Line-s_nom"
# Not "Line-...": PyPSA writes every Line-<attr> variable back to n.lines after the solve
PREFIX = "LineCostCurve"
COST_LIMIT = "transmission_expansion_cost_limit"
COST_LIMIT_PER_PERIOD = "transmission_expansion_cost_limit_per_period"
TRANSMISSION_CARRIERS = ["AC", "DC"]


def read_piecewise_line_costs(path) -> pd.DataFrame:
    """
    Read a cost-curve CSV with columns ``line``, ``breakpoint``, ``capital_cost``.

    One row per segment: ``breakpoint`` is the segment's upper end in MW above the
    line's existing capacity (``inf`` or empty for an unbounded last segment) and
    ``capital_cost`` its cost per MW, in the units of the network's ``capital_cost``.
    """
    curves = pd.read_csv(path, dtype={"line": str})
    missing = [col for col in CURVE_COLUMNS if col not in curves.columns]
    if missing:
        raise ValueError(f"Piecewise line cost file {path} lacks columns {missing}; expected {CURVE_COLUMNS}.")
    return curves[CURVE_COLUMNS]


def line_cost_segments(curves: pd.DataFrame, lines: pd.Index) -> pd.DataFrame:
    """
    Validate cost curves and turn them into segments.

    Returns a frame indexed by (``name``, ``segment``) with the segment's ``lower`` end
    and ``width`` in MW above existing capacity and its ``capital_cost``.
    """
    curves = curves[CURVE_COLUMNS].copy()
    curves["line"] = curves["line"].astype(str)
    curves["breakpoint"] = pd.to_numeric(curves["breakpoint"]).fillna(np.inf)
    curves["capital_cost"] = pd.to_numeric(curves["capital_cost"])

    unknown = pd.Index(curves.line.unique()).difference(lines)
    if not unknown.empty:
        raise ValueError(f"Piecewise line costs name lines that are not in the network: {unknown.tolist()[:20]}.")
    if not np.isfinite(curves.capital_cost).all():
        raise ValueError("Piecewise line costs need a finite capital_cost on every segment.")
    if (curves.breakpoint <= 0).any():
        raise ValueError("Piecewise line cost breakpoints are MW above existing capacity and must be positive.")
    duplicated = curves.duplicated(["line", "breakpoint"])
    if duplicated.any():
        raise ValueError(f"Lines with a repeated breakpoint: {curves.line[duplicated].unique().tolist()[:20]}.")

    curves = curves.sort_values(["line", "breakpoint"], kind="stable")
    by_line = curves.groupby("line", sort=False)
    falling = curves.line[by_line.capital_cost.diff() < 0].unique()
    if len(falling):
        raise ValueError(
            "Piecewise line costs must not decrease from one segment to the next (a convex curve); "
            f"they do for lines {falling.tolist()[:20]}.",
        )

    lower = by_line.breakpoint.shift(fill_value=0.0)
    index = pd.MultiIndex.from_arrays([curves.line, by_line.cumcount()], names=["name", SEGMENT])
    return pd.DataFrame(
        {
            "lower": lower.to_numpy(),
            "width": (curves.breakpoint - lower).to_numpy(),
            "capital_cost": curves.capital_cost.to_numpy(),
        },
        index=index,
    )


def attach_piecewise_line_costs(n: pypsa.Network, curves: pd.DataFrame) -> None:
    """Give the lines in ``curves`` their cost curves, measured from their current ``s_nom``."""
    segments = line_cost_segments(curves, n.lines.index)
    names = segments.index.unique("name")

    native = n.c.lines.piecewise.get("capital_cost")
    if native is not None and not native.empty:
        overlap = names.intersection(native.dropna(how="all", axis=1).columns.unique("name"))
        if not overlap.empty:
            raise ValueError(f"Lines {overlap.tolist()[:20]} already have a PyPSA piecewise capital_cost curve.")

    n.lines[BASE_COLUMN] = n.lines.s_nom.where(n.lines.index.isin(names))
    lines = n.lines.loc[names]
    below = names[lines.s_nom_extendable & (lines.s_nom_min < lines.s_nom - 1e-6)]
    if not below.empty:
        logger.warning(
            f"{len(below)} lines with a cost curve have s_nom_min below s_nom; the curve holds them at "
            f"their existing capacity or above (e.g. {below.tolist()[:5]}).",
        )
    not_extendable = names[~n.lines.loc[names, "s_nom_extendable"]]
    if not not_extendable.empty:
        logger.info(f"{len(not_extendable)} of the {len(names)} lines with a cost curve are not extendable.")

    n.line_cost_segments = segments
    logger.info(f"Attached piecewise expansion costs to {len(names)} lines ({len(segments)} segments).")


def piecewise_line_cost(segments: pd.DataFrame, increment: pd.Series) -> pd.Series:
    """Cost along each line's curve of adding ``increment`` MW (nothing for none or less)."""
    added = increment.reindex(segments.index.get_level_values("name")).to_numpy()
    filled = np.clip(added - segments["lower"].to_numpy(), 0.0, segments["width"].to_numpy())
    cost = pd.Series(filled * segments["capital_cost"].to_numpy(), index=segments.index)
    return cost.groupby(level="name").sum().reindex(increment.index)


def _segment_array(segments: pd.DataFrame, column: str, names: pd.Index) -> xr.DataArray:
    frame = segments[column].unstack(SEGMENT).reindex(names)
    frame.index.name, frame.columns.name = "name", SEGMENT
    return xr.DataArray(frame)


def _series(da: xr.DataArray, names: pd.Index) -> pd.Series:
    return pd.Series(da.sel(name=names).to_numpy(), index=names)


def _carriers(glc) -> list[str]:
    """The carriers of a GlobalConstraint, parsed like PyPSA does."""
    return [re.sub(r"[\[\]\(\)]", "", c.strip()) for c in str(glc.carrier_attribute).split(",")]


def _extend_constraint(m: linopy.Model, name: str, extra: list, rhs_shift: float) -> None:
    """Add ``extra`` terms to the left-hand side of constraint ``name`` and subtract ``rhs_shift`` from its right."""
    if not extra and rhs_shift == 0:
        return
    con = m.constraints[name]
    lhs, sign, rhs = con.lhs, con.sign, con.rhs
    if extra:
        lhs = linopy.merge([lhs, *extra])
    sign = sign.item() if isinstance(sign, xr.DataArray) else sign
    m.remove_constraints(name)
    m.add_constraints(lhs, sign, rhs - rhs_shift, name=name)


def add_piecewise_line_costs(n: pypsa.Network, snapshots: pd.Index) -> None:
    """
    Charge lines with a cost curve along it; call after :func:`opts.transmission.add_period_transmission_limits`.

    See the module docstring for the formulation.
    """
    segments = getattr(n, "line_cost_segments", None)
    if segments is None or segments.empty:
        return

    m = n.model
    c = n.c.lines
    names = segments.index.unique("name")
    ext = names.intersection(m.variables[NOMINAL].indexes["name"]) if NOMINAL in m.variables else names[:0]
    base = n.lines[BASE_COLUMN]
    capital_cost = n.lines.capital_cost

    multi = bool(getattr(n, "_multi_invest", False))
    _, model_periods = modelled_periods(n, snapshots)
    periods = model_periods if multi else [None]
    weightings = n.investment_period_weightings.objective

    def active_weight(lines, period):
        if period is None:
            return pd.Series(1.0, index=lines)
        active = c.get_active_assets(investment_period=period).reindex(lines, fill_value=False)
        return active.astype(float) * weightings[period]

    # The capacity variable of each line in each period, and the objective weight of its segments
    deltas = {}  # capacity variable -> segment variable
    by_period = {p: [] for p in periods}  # period -> [(capacity variable, lines)]
    weights = {}  # capacity variable -> objective weight per line
    if not ext.empty:
        weights[NOMINAL] = pd.Series(0.0, index=ext)
        for period in periods:
            staged = ext[:0]
            var = period_capacity_variable("Line", "s_nom", period) if period is not None else None
            if var is not None and var in m.variables:
                staged = ext.intersection(m.variables[var].indexes["name"])
            if not staged.empty:
                by_period[period].append((var, staged))
                weights[var] = active_weight(staged, period)
            rest = ext.difference(staged)
            if not rest.empty:
                by_period[period].append((NOMINAL, rest))
                weights[NOMINAL] = weights[NOMINAL].add(active_weight(rest, period), fill_value=0.0)

        cost = _segment_array(segments, "capital_cost", ext)
        width = _segment_array(segments, "width", ext)
        valid = width.notnull()
        periodized = _series(c.periodized_cost, ext)
        objective = []
        for var, weight in weights.items():
            lines = weight.index
            delta = m.add_variables(
                lower=0.0,
                upper=width.sel(name=lines).fillna(0.0),
                mask=valid.sel(name=lines),
                name=f"{PREFIX}-{var.removeprefix('Line-')}",
            )
            deltas[var] = delta
            capacity = m[var].sel(name=lines)
            m.add_constraints(
                delta.sum(SEGMENT) - capacity == -xr.DataArray(base[lines].to_numpy(), coords={"name": lines}),
                name=f"{PREFIX}-{var.removeprefix('Line-')}-sum",
            )
            # PyPSA charges periodized_cost * capacity; this makes it pay the curve for what is added
            coeff = (cost.sel(name=lines) - xr.DataArray(periodized[lines].to_numpy(), coords={"name": lines})) * (
                xr.DataArray(weight.to_numpy(), coords={"name": lines})
            )
            if (weight != 0).any():
                objective.append((coeff.fillna(0.0) * delta).sum())
        if objective:
            m.add_objective(m.objective.expression + linopy.merge(objective), overwrite=True)

    def curve_terms(var, lines, weight=None):
        """``(c_k - capital_cost) * delta_k`` of ``lines``, optionally weighted, as a cost limit counts them."""
        coeff = _segment_array(segments, "capital_cost", lines) - xr.DataArray(
            capital_cost[lines].to_numpy(),
            coords={"name": lines},
        )
        if weight is not None:
            coeff = coeff * xr.DataArray(weight[lines].to_numpy(), coords={"name": lines})
        return (coeff.fillna(0.0) * deltas[var].sel(name=lines)).sum()

    glcs = n.global_constraints

    # PyPSA's cost limit: capital_cost-weighted s_nom of extendable lines (see
    # pypsa.optimization.global_constraints.define_transmission_expansion_cost_limit)
    for name, glc in glcs[glcs.type == COST_LIMIT].iterrows():
        con = f"GlobalConstraint-{name}"
        lines = ext[n.lines.loc[ext, "carrier"].isin(_carriers(glc))]
        if con not in m.constraints or lines.empty:
            continue
        period = glc.get("investment_period", np.nan)
        if pd.notna(period):
            weight = c.get_active_assets(investment_period=int(period)).reindex(lines, fill_value=False).astype(float)
        elif multi:
            weight = sum(active_weight(lines, p) for p in periods)
        else:
            weight = pd.Series(1.0, index=lines)
        lines = weight.index[weight != 0]
        if not lines.empty:
            _extend_constraint(m, con, [curve_terms(NOMINAL, lines, weight)], 0.0)

    # Per-period cost limits (opts.transmission): capital_cost-weighted capacity in each period
    fixed = names.difference(ext)
    for name, glc in glcs[glcs.type == COST_LIMIT_PER_PERIOD].iterrows():
        con = f"GlobalConstraint-{name}"
        period = int(glc.investment_period)
        if con not in m.constraints or period not in model_periods:
            continue
        active = c.get_active_assets(investment_period=period)
        counted = names[
            n.lines.loc[names, "carrier"].isin(TRANSMISSION_CARRIERS) & active.reindex(names, fill_value=False)
        ]
        extra = []
        for var, lines in by_period.get(period, []):
            lines = lines.intersection(counted)
            if not lines.empty:
                extra.append(curve_terms(var, lines))
        # fixed lines enter as a constant capital_cost * capacity; count their added capacity on the curve
        held = fixed.intersection(counted)
        shift = 0.0
        if not held.empty:
            added = installed_capacity(c, held, "s_nom", period, model_periods[-1]) - base[held]
            shift = float((piecewise_line_cost(segments, added) - capital_cost[held] * added).sum())
        _extend_constraint(m, con, extra, shift)


def correct_iterated_objective(n: pypsa.Network, snapshots=None, multi_investment_periods: bool = False) -> float:
    """
    Charge the curve lines along their curves in the objective of the iterative expansion.

    ``optimize_transmission_expansion_iteratively`` fixes the lines for its final solve and
    then adds ``periodized_cost * (s_nom_<period> - s_nom_min)`` of each extendable line to
    the objective, for each solved investment period weighted as in the objective, with the
    capacity of that period (pypsa-bill). For a line with a cost curve this replaces that
    charge with the curve's, ``F(s_nom_<period>) - F(s_nom_min)`` with ``F(S) = periodized_cost
    * base + curve(S - base)`` as in the iterations (see :func:`add_piecewise_line_costs`).
    Like pypsa-bill it moves the correction from the objective constant to the objective.
    Call right after the iterative expansion, with its snapshots and
    ``multi_investment_periods``; returns the correction.
    """
    segments = getattr(n, "line_cost_segments", None)
    if segments is None or segments.empty:
        return 0.0
    c = n.c.lines
    static = c.static
    lines = segments.index.unique("name").intersection(static.index[static.s_nom_extendable.astype(bool)])
    if lines.empty:
        return 0.0
    base = static.loc[lines, BASE_COLUMN].astype(float)
    periodized = _series(c.periodized_cost, lines)

    def excess(capacity: pd.Series) -> pd.Series:
        """Cost of the capacity above ``base`` along the curve, less its charge at ``periodized_cost``."""
        added = (capacity - base).clip(lower=0.0)
        return piecewise_line_cost(segments, added) - periodized * added

    floor = excess(static.loc[lines, "s_nom_min"].astype(float))
    final = static.loc[lines, "s_nom_opt"].astype(float)

    # the periods and capacities pypsa-bill charged
    sns = n.snapshots if snapshots is None else snapshots
    multi = multi_investment_periods and isinstance(sns, pd.MultiIndex)
    if not multi:
        correction = float((excess(final) - floor).sum())
    else:
        periods = list(pd.Index(sns.get_level_values(0)).unique())
        staged_periods = modelled_periods_of_solve(n, sns, True)
        staged = staged_branches(n, "Line").intersection(lines) if staged_periods else lines[:0]
        weights = n.investment_period_weightings.objective
        correction = 0.0
        for period in periods:
            capacity = final.copy()
            column = period_capacity_column("s_nom", period)
            if period in staged_periods[:-1] and column in static:
                held = static.loc[staged, column].astype(float)
                capacity.loc[held.index[held.notna()]] = held.dropna()
            active = c.get_active_assets(investment_period=period).reindex(lines, fill_value=False).astype(float)
            correction += float((active * weights[period] * (excess(capacity) - floor)).sum())

    n._objective += correction
    n._objective_constant -= correction
    logger.info(f"Charged {len(lines)} lines along their cost curves: objective corrected by {correction:,.2f}.")
    return correction


def store_piecewise_line_costs(n: pypsa.Network) -> None:
    """
    Write the cost of each curve line's added capacity along its curve.

    ``piecewise_expansion_cost`` for ``s_nom_opt`` and ``piecewise_expansion_cost_<period>``
    for each ``s_nom_<period>_opt``, in the units of ``capital_cost`` times MW.
    """
    segments = getattr(n, "line_cost_segments", None)
    if segments is None or segments.empty:
        return
    names = segments.index.unique("name")
    base = n.lines.loc[names, BASE_COLUMN]
    columns = {COST_COLUMN: "s_nom_opt"}
    columns |= {f"{COST_COLUMN}_{p}": period_capacity_column("s_nom", p) for p in n.investment_periods}
    for target, source in columns.items():
        if source not in n.lines:
            continue
        n.lines[target] = np.nan
        n.lines.loc[names, target] = piecewise_line_cost(segments, n.lines.loc[names, source] - base)
