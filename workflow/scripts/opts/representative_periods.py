"""
Solve-side handling of representative periods.

``prepare_network`` (see ``select_representative_periods.py``) reduces a network to a
few representative blocks and records them in ``n.meta["representative_periods"]``.
The snapshots keep their real timestamps, so consecutive snapshots of two blocks are
months apart, and every constraint that links a snapshot to the one before it has
to stop at block boundaries:

* storage balances (``StorageUnit-energy_balance``, ``Store-energy_balance``, and
  the ERM's shadow ``StorageUnit-energy_balance_RESERVES``) cycle within each block:
  the first snapshot of a block follows its last. Every storage asset cycles this
  way, whatever its cyclic flags say;
* ramp limits are dropped at the first snapshot of each block.

Elapsed hours stay right without any change: ``snapshot_weightings.stores`` still
holds the length of one timestep, while ``objective`` and ``generators`` carry the
block weights.
"""

import logging

import numpy as np
import pandas as pd
from pypsa.optimization.window import snapshot_array

logger = logging.getLogger(__name__)

META_KEY = "representative_periods"


def representative_periods_metadata(n) -> dict:
    """The block metadata ``prepare_network`` stored on ``n``, or ``{}`` without representative periods."""
    meta = getattr(n, "meta", None) or {}
    data = meta.get(META_KEY) if isinstance(meta, dict) else None
    return data if isinstance(data, dict) else {}


def block_ids(n, sns) -> np.ndarray | None:
    """
    Block of each snapshot of ``sns`` (in model labels), numbered across investment periods.

    Returns ``None`` without representative periods. Raises if a snapshot falls outside
    every block or a block's snapshots are not contiguous.
    """
    metadata = representative_periods_metadata(n)
    if not metadata:
        return None

    index = n.optimize._window.subset(sns).network_index
    periods = index.get_level_values("period")
    timesteps = pd.DatetimeIndex(index.get_level_values("timestep"))

    ids = np.full(len(index), -1, dtype=np.int64)
    offset = 0
    for period in periods.unique():
        entries = sorted(metadata["periods"][str(period)], key=lambda e: pd.Timestamp(e["start"]))
        starts = pd.DatetimeIndex([e["start"] for e in entries])
        ends = pd.DatetimeIndex([e["end"] for e in entries])
        mask = np.asarray(periods == period)
        t = timesteps[mask]
        block = starts.searchsorted(t, side="right") - 1
        inside = (block >= 0) & (t <= ends[np.maximum(block, 0)])
        if not inside.all():
            raise ValueError(f"Snapshots {list(t[~inside][:3])} of {period} lie outside every representative period.")
        ids[mask] = block + offset
        offset += len(starts)

    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    if len(starts) != len(np.unique(ids)):
        raise ValueError("The snapshots of a representative period are not contiguous.")
    return ids


def _block_starts(ids: np.ndarray) -> np.ndarray:
    return np.r_[True, ids[1:] != ids[:-1]]


def previous_in_block(n, var, sns):
    """
    ``var`` one snapshot earlier, wrapping from each block's first snapshot to its last.

    Returns ``None`` without representative periods.
    """
    ids = block_ids(n, sns)
    if ids is None:
        return None
    starts = np.flatnonzero(_block_starts(ids))
    ends = np.append(starts[1:], len(ids)) - 1
    positions = np.arange(len(ids)) - 1
    positions[starts] = ends
    return n.optimize._window.subset(sns).take(var, positions)


def _rebuild_storage_unit_balance(n, sns):
    m = n.model
    c = n.components["StorageUnit"]
    name = "StorageUnit-energy_balance"
    if c.static.empty or name not in m.constraints:
        return 0

    window = n.optimize._window.subset(sns)
    assets = c.active_assets
    active = c.da.active.sel(snapshot=sns, name=assets)
    eh = window.snapshot_weightings("stores")
    eff_stand = (1 - c.da.standing_loss.sel(snapshot=sns, name=assets)) ** eh
    eff_dispatch = c.da.efficiency_dispatch.sel(snapshot=sns, name=assets)
    eff_store = c.da.efficiency_store.sel(snapshot=sns, name=assets)

    soc = m["StorageUnit-state_of_charge"]
    lhs = m.linexpr(
        (-1, soc),
        (-1 / eff_dispatch * eh, m["StorageUnit-p_dispatch"]),
        (eff_store * eh, m["StorageUnit-p_store"]),
        (eff_stand, previous_in_block(n, soc, sns)),
    )
    if "StorageUnit-spill" in m.variables:
        lhs = lhs - (eh * m["StorageUnit-spill"]).fillna(0)
    rhs = -c.da.inflow.sel(snapshot=sns, name=assets) * eh

    m.remove_constraints(name)
    m.add_constraints(lhs, "=", rhs, name=name, mask=active)
    return len(assets)


def _rebuild_store_balance(n, sns):
    m = n.model
    c = n.components["Store"]
    name = "Store-energy_balance"
    if c.static.empty or name not in m.constraints:
        return 0

    window = n.optimize._window.subset(sns)
    assets = c.active_assets
    active = c.da.active.sel(snapshot=sns, name=assets)
    eh = window.snapshot_weightings("stores")
    eff_stand = (1 - c.da.standing_loss.sel(snapshot=sns, name=assets)) ** eh

    e = m["Store-e"]
    lhs = m.linexpr((-1, e), (-eh, m["Store-p"]), (eff_stand, previous_in_block(n, e, sns)))

    m.remove_constraints(name)
    m.add_constraints(lhs, "=", 0, name=name, mask=active)
    return len(assets)


def _drop_ramp_limits_between_blocks(n, sns, ids):
    """Mask the ramp-limit rows at the first snapshot of every block but the first."""
    m = n.model
    window = n.optimize._window.subset(sns)
    cut = _block_starts(ids)
    cut[0] = False  # PyPSA's own first-snapshot handling (p_init) stays
    keep = snapshot_array(~cut, window.model_index)

    dropped = 0
    for name in [c for c in m.constraints if "-ramp_limit_" in c]:
        con = m.constraints[name]
        if "snapshot" not in con.labels.dims:
            continue
        mask = (con.labels != -1) & keep
        dropped += int(((con.labels != -1) & ~keep).sum())
        lhs, sign, rhs = con.lhs, con.sign, con.rhs
        m.remove_constraints(name)
        m.add_constraints(lhs, sign, rhs, name=name, mask=mask)
    return dropped


def add_representative_period_constraints(n, sns) -> None:
    """Cycle storage within each representative period and drop ramp limits between periods."""
    ids = block_ids(n, sns)
    if ids is None:
        return

    for c, flag in (("StorageUnit", "cyclic_state_of_charge"), ("Store", "e_cyclic")):
        static = n.components[c].static
        if not static.empty and not static[flag].all():
            logger.info(
                "%d %s(s) are not cyclic; under representative periods they cycle within each period anyway.",
                int((~static[flag]).sum()),
                c,
            )
    if "committable" in n.generators and n.generators.committable.any():
        logger.warning(
            "Committable generators under representative periods: minimum up/down times and start-ups "
            "still run across period boundaries.",
        )

    units = _rebuild_storage_unit_balance(n, sns)
    stores = _rebuild_store_balance(n, sns)
    ramps = _drop_ramp_limits_between_blocks(n, sns, ids)
    logger.info(
        "Representative periods (%d): storage cycles within each period (%d StorageUnits, %d Stores); "
        "%d ramp-limit rows between periods dropped.",
        len(np.unique(ids)),
        units,
        stores,
        ramps,
    )
