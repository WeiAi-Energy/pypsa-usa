"""Representative-period specific optimization constraints."""

import logging

import linopy
import numpy as np
import pandas as pd
from pypsa.descriptors import get_activity_mask
from pypsa.descriptors import (
    get_switchable_as_dense as get_as_dense,
)
from pypsa.optimization.constraints import expand_series
from xarray import DataArray, concat

logger = logging.getLogger(__name__)

def _get_period_hours(rep_cfg):
    """Return the representative/extreme period length in hours from config."""
    period_length = rep_cfg.get("period_length")
    if period_length is None:
        raise ValueError("representative_periods.period_length must be configured.")

    days = float(period_length)
    if days <= 0:
        raise ValueError("representative_periods.period_length must be positive.")
    return days * 24.0


def _get_representative_period_metadata(n):
    """Return serialized representative-period metadata stored on the prepared network."""
    meta = getattr(n, "meta", {}) or {}
    if not isinstance(meta, dict):
        return {}
    data = meta.get("representative_periods_plot_metadata", {})
    return data if isinstance(data, dict) else {}


def _to_multiindex_snapshots(snapshots):
    """Normalize optimization snapshots to a MultiIndex when possible."""
    if isinstance(snapshots, pd.MultiIndex):
        return snapshots

    if isinstance(snapshots, pd.Index) and len(snapshots) and isinstance(snapshots[0], tuple):
        return pd.MultiIndex.from_tuples(snapshots)

    return None


def _get_representative_blocks(snapshots, base_hours, metadata=None):
    """
    Build representative-period blocks from model snapshots.

    Returns a dict per investment period with one entry per representative block.
    """
    if snapshots is None or snapshots.nlevels < 2:
        return {}

    names = list(snapshots.names or [])
    period_level = "period" if "period" in names else 0
    timestep_level = "timestep" if "timestep" in names else 1

    blocks = {}
    for year in snapshots.get_level_values(period_level).unique():
        mask = snapshots.get_level_values(period_level) == year
        period_snapshots = snapshots[mask]
        timesteps = pd.DatetimeIndex(snapshots[mask].get_level_values(timestep_level))
        if len(timesteps) < 2:
            continue

        diffs = timesteps.to_series().diff().dropna().dt.total_seconds().to_numpy() / 3600.0
        if len(diffs) == 0:
            continue
        if np.any(diffs <= 0):
            raise ValueError(f"Non-positive timestep spacing detected in period {year}.")
        if not np.allclose(diffs, diffs[0]):
            raise ValueError(f"Irregular timestep spacing detected in period {year}.")

        timestep_hours = float(diffs[0])
        year_metadata = (metadata or {}).get(str(year), {})
        period_entries = year_metadata.get("periods", []) if isinstance(year_metadata, dict) else []
        year_blocks = []

        if period_entries:
            offset = 0
            for entry in period_entries:
                steps = int(entry.get("steps", 0))
                if steps <= 0:
                    continue
                if (offset + steps) > len(period_snapshots):
                    raise ValueError(
                        f"Representative-period metadata for {year} exceeds available snapshots.",
                    )
                block_snapshots = period_snapshots[offset : offset + steps]
                year_blocks.append(
                    {
                        "snapshots": block_snapshots,
                        "steps": steps,
                        "timestep_hours": timestep_hours,
                        "period_id": entry.get("period_id"),
                        "kind": entry.get("kind", "representative"),
                    },
                )
                offset += steps

            if offset < len(period_snapshots):
                logger.warning(
                    "Representative-period metadata for %s covers %s/%s snapshots. "
                    "Ignoring %s trailing snapshots for cyclic constraints.",
                    year,
                    offset,
                    len(period_snapshots),
                    len(period_snapshots) - offset,
                )
        else:
            steps = int(round(base_hours / timestep_hours))
            if steps <= 0 or not np.isclose(steps * timestep_hours, base_hours):
                raise ValueError(
                    f"Representative period length ({base_hours}h) is incompatible with "
                    f"timestep resolution ({timestep_hours:g}h) in period {year}.",
                )

            n_blocks = len(timesteps) // steps
            if n_blocks == 0:
                continue
            used = n_blocks * steps
            if used < len(timesteps):
                logger.warning(
                    "Period %s snapshots do not align with representative period length. "
                    "Ignoring %s trailing snapshots for cyclic constraints.",
                    year,
                    len(timesteps) - used,
                )
            for start in range(0, used, steps):
                block_snapshots = period_snapshots[start : start + steps]
                year_blocks.append(
                    {
                        "snapshots": block_snapshots,
                        "steps": steps,
                        "timestep_hours": timestep_hours,
                        "period_id": None,
                        "kind": "representative",
                    },
                )

        if year_blocks:
            blocks[year] = year_blocks

    return blocks


def _get_physical_elapsed_hours(snapshots):
    """Return physical timestep duration in hours for each optimization snapshot."""
    if snapshots is None or snapshots.nlevels < 2:
        return pd.Series(dtype="float64")

    names = list(snapshots.names or [])
    period_level = "period" if "period" in names else 0
    timestep_level = "timestep" if "timestep" in names else 1

    elapsed_parts = []
    for period in snapshots.get_level_values(period_level).unique():
        period_mask = snapshots.get_level_values(period_level) == period
        period_snapshots = snapshots[period_mask]
        timesteps = pd.DatetimeIndex(period_snapshots.get_level_values(timestep_level))

        if len(timesteps) < 2:
            raise ValueError(
                f"Unable to infer physical timestep duration for representative period {period}.",
            )

        diffs = timesteps.to_series().diff().dropna().dt.total_seconds().to_numpy() / 3600.0
        if np.any(diffs <= 0):
            raise ValueError(f"Non-positive timestep spacing detected in period {period}.")
        if not np.allclose(diffs, diffs[0]):
            raise ValueError(f"Irregular timestep spacing detected in period {period}.")

        elapsed_parts.append(pd.Series(float(diffs[0]), index=period_snapshots))

    return pd.concat(elapsed_parts).reindex(snapshots)


def _build_block_cyclic_previous_state(var, active, snapshots, blocks):
    """
    Build the previous-state term for representative-period cycling.

    This mirrors PyPSA's cyclic logic: for the first snapshot in each
    representative block, the previous state is the final state of that block,
    not a fixed initial value.
    """
    dim = "snapshot"
    period_level = "period" if "period" in (snapshots.names or []) else 0

    default_include_previous = active.cumsum(dim) != 1
    default_previous = (
        var.where(active)
        .ffill(dim)
        .roll(snapshot=1)
        .ffill(dim)
        .where(default_include_previous)
    )

    previous_parts = []
    include_parts = []

    for period in snapshots.get_level_values(period_level).unique():
        period_mask = snapshots.get_level_values(period_level) == period
        period_snapshots = snapshots[period_mask]

        if period not in blocks:
            previous_parts.append(default_previous.data.sel(snapshot=period_snapshots))
            include_parts.append(default_include_previous.sel(snapshot=period_snapshots))
            continue

        period_var = var.sel(snapshot=period_snapshots)
        period_active = active.sel(snapshot=period_snapshots)
        offset = 0

        for block in blocks[period]:
            steps = int(block["steps"])
            selection = slice(offset, offset + steps)
            block_var = period_var.isel(snapshot=selection)
            block_active = period_active.isel(snapshot=selection)
            previous_parts.append(
                block_var.where(block_active).ffill(dim).roll(snapshot=1).ffill(dim).where(block_active).data,
            )
            include_parts.append(block_active)
            offset += steps

        if offset < len(period_snapshots):
            tail_snapshots = period_snapshots[offset:]
            previous_parts.append(default_previous.data.sel(snapshot=tail_snapshots))
            include_parts.append(default_include_previous.sel(snapshot=tail_snapshots))

    previous = var.__class__(concat(previous_parts, dim=dim), var.model, var.name)
    include_previous = concat(include_parts, dim=dim)
    return previous, include_previous


def _select_component_axis(obj, component_names):
    """Select component coordinates regardless of whether the axis is called `name` or the component class."""
    component_dim = next((dim for dim in getattr(obj, "dims", ()) if dim != "snapshot"), None)
    if component_dim is None:
        return obj
    return obj.sel({component_dim: pd.Index(component_names)})


def _define_representative_period_storage_unit_constraints(n, sns, blocks, elapsed_hours):
    m = n.model
    c = "StorageUnit"
    assets = n.df(c)
    active = DataArray(get_activity_mask(n, c, sns))

    if assets.empty:
        return 0

    eh = expand_series(elapsed_hours.loc[sns], assets.index)
    eff_stand = (1 - get_as_dense(n, c, "standing_loss", sns)).pow(eh)
    eff_dispatch = get_as_dense(n, c, "efficiency_dispatch", sns)
    eff_store = get_as_dense(n, c, "efficiency_store", sns)

    soc = m[f"{c}-state_of_charge"]

    previous_soc, include_previous_soc = _build_block_cyclic_previous_state(
        soc,
        active,
        sns,
        blocks,
    )

    lhs = [
        (-1, soc),
        (-1 / eff_dispatch * eh, m[f"{c}-p_dispatch"]),
        (eff_store * eh, m[f"{c}-p_store"]),
        (eff_stand, previous_soc),
    ]

    if f"{c}-spill" in m.variables:
        lhs += [(-eh, m[f"{c}-spill"])]

    rhs = DataArray(-get_as_dense(n, c, "inflow", sns).mul(eh))
    m.add_constraints(lhs, "=", rhs, name=f"{c}-energy_balance", mask=include_previous_soc)
    return int(active.sum().item())


def _define_representative_period_store_constraints(n, sns, blocks, elapsed_hours, store_names=None):
    m = n.model
    c = "Store"
    assets = n.df(c)
    if store_names is not None:
        assets = assets.loc[pd.Index(store_names).intersection(assets.index)]

    if assets.empty:
        return 0

    active = _select_component_axis(DataArray(get_activity_mask(n, c, sns)), assets.index)
    eh = expand_series(elapsed_hours.loc[sns], assets.index)
    eff_stand = (1 - get_as_dense(n, c, "standing_loss", sns)[assets.index]).pow(eh)

    e = _select_component_axis(m[f"{c}-e"], assets.index)
    p = _select_component_axis(m[f"{c}-p"], assets.index)

    previous_e, include_previous_e = _build_block_cyclic_previous_state(
        e,
        active,
        sns,
        blocks,
    )

    lhs = [(-1, e), (-eh, p), (eff_stand, previous_e)]
    rhs = 0
    m.add_constraints(lhs, "=", rhs, name=f"{c}-energy_balance", mask=include_previous_e)
    return int(active.sum().item())


def _define_default_store_constraints(n, sns, store_names):
    """Re-add PyPSA default store balance constraints for selected stores."""
    m = n.model
    c = "Store"
    assets = n.df(c).loc[pd.Index(store_names).intersection(n.df(c).index)]

    if assets.empty:
        return 0

    active = _select_component_axis(DataArray(get_activity_mask(n, c, sns)), assets.index)
    eh = expand_series(n.snapshot_weightings.stores[sns], assets.index)
    eff_stand = (1 - get_as_dense(n, c, "standing_loss", sns)[assets.index]).pow(eh)

    e = _select_component_axis(m[f"{c}-e"], assets.index)
    p = _select_component_axis(m[f"{c}-p"], assets.index)

    lhs = [(-1, e), (-eh, p)]

    noncyclic_b = ~assets.e_cyclic.to_xarray()
    include_previous_e = (active.cumsum("snapshot") != 1).where(noncyclic_b, True)
    previous_e = e.where(active).ffill("snapshot").roll(snapshot=1).ffill("snapshot").where(include_previous_e)

    e_init = assets.e_initial.to_xarray()

    if isinstance(sns, pd.MultiIndex):
        periods = e.coords["period"]
        per_period = assets.e_cyclic_per_period.to_xarray() | assets.e_initial_per_period.to_xarray()

        ps = sns.unique("period")
        sl = slice(None)
        previous_e_pp_list = [e.data.sel(snapshot=(period, sl)).roll(snapshot=1) for period in ps]
        previous_e_pp = concat(previous_e_pp_list, dim="snapshot")

        include_previous_e_pp = active & (periods == periods.shift(snapshot=1))
        include_previous_e_pp = include_previous_e_pp.where(noncyclic_b, True)
        previous_e_pp = previous_e_pp.where(include_previous_e_pp.values, linopy.variables.FILL_VALUE)

        previous_e = previous_e.where(~per_period, previous_e_pp)
        include_previous_e = include_previous_e_pp.where(per_period, include_previous_e)

    lhs += [(eff_stand, previous_e)]
    rhs = -e_init.where(~include_previous_e, 0)
    m.add_constraints(lhs, "=", rhs, name=f"{c}-energy_balance-default", mask=active)
    return int(active.sum().item())


def add_representative_period_storage_constraints(n, config, snapshots):
    """Rebuild storage balance constraints with PyPSA-style cyclic logic per block."""
    rep_cfg = (
        config.get("clustering", {})
        .get("temporal", {})
        .get("representative_periods", {})
    )
    if not rep_cfg.get("enable", False):
        return

    snapshot_index = _to_multiindex_snapshots(snapshots)
    if snapshot_index is None or snapshot_index.nlevels < 2:
        logger.warning(
            "Representative periods are enabled, but optimization snapshots are not a MultiIndex. "
            "Skipping representative-period storage cyclic constraints.",
        )
        return

    base_hours = _get_period_hours(rep_cfg)
    logger.info(
        "Representative periods enabled for optimization: period_length=%s days (%sh).",
        base_hours / 24.0,
        base_hours,
    )

    metadata = _get_representative_period_metadata(n)
    blocks = _get_representative_blocks(snapshot_index, base_hours, metadata=metadata)
    if not blocks:
        logger.warning("No representative-period blocks detected. Skipping storage cyclic constraints.")
        return
    elapsed_hours = _get_physical_elapsed_hours(snapshot_index)

    for investment_period, block_entries in blocks.items():
        unique_steps = sorted({int(entry["steps"]) for entry in block_entries})
        logger.info(
            "Applying representative-period storage cyclic logic for %s: %s blocks, step counts=%s at %sh physical resolution.",
            investment_period,
            len(block_entries),
            unique_steps,
            float(block_entries[0]["timestep_hours"]),
        )

    constraint_names = [
        name
        for name in (
            "StorageUnit-energy_balance",
            "Store-energy_balance",
            "Store-energy_balance-default",
        )
        if name in n.model.constraints
    ]
    if constraint_names:
        n.model.remove_constraints(constraint_names)

    su_constraints = _define_representative_period_storage_unit_constraints(
        n,
        snapshot_index,
        blocks,
        elapsed_hours,
    )
    exempt_stores = pd.Index([])
    if not n.stores.empty and "exclude_representative_periods" in n.stores.columns:
        exempt_stores = n.stores.index[n.stores["exclude_representative_periods"].fillna(False).astype(bool)]
    representative_stores = n.stores.index.difference(exempt_stores)

    store_constraints = 0
    if len(representative_stores):
        store_constraints = _define_representative_period_store_constraints(
            n,
            snapshot_index,
            blocks,
            elapsed_hours,
            representative_stores,
        )

    default_store_constraints = 0
    if len(exempt_stores):
        default_store_constraints = _define_default_store_constraints(
            n,
            snapshot_index,
            exempt_stores,
        )
    logger.info(
        "Rebuilt representative-period energy balances: StorageUnit=%s, Store=%s, StoreDefault=%s.",
        su_constraints,
        store_constraints,
        default_store_constraints,
    )
