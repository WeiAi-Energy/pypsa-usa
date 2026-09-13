import os
import sys

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from opts.representative_periods import (
    _get_representative_blocks,
    add_representative_period_storage_constraints,
    extreme_period_snapshots,
    storage_elapsed_hours,
)

STORE = "ac tes"
REPRESENTATIVE_CONFIG = {
    "clustering": {"temporal": {"representative_periods": {"enable": True, "period_length": 1}}},
}


def test_get_representative_blocks_uses_metadata_steps_for_mixed_lengths():
    timesteps = pd.date_range("2030-01-01 00:00", periods=9, freq="h")
    snapshots = pd.MultiIndex.from_arrays(
        [np.repeat(2030, len(timesteps)), timesteps],
        names=["period", "timestep"],
    )
    metadata = {
        "2030": {
            "periods": [
                {"period_id": 0, "kind": "representative", "steps": 4},
                {"period_id": 1, "kind": "extreme", "steps": 2},
                {"period_id": 2, "kind": "representative", "steps": 3},
            ],
        },
    }

    blocks = _get_representative_blocks(
        snapshots,
        base_hours=24.0,
        metadata=metadata,
    )

    assert [entry["steps"] for entry in blocks[2030]] == [4, 2, 3]
    assert [entry["kind"] for entry in blocks[2030]] == [
        "representative",
        "extreme",
        "representative",
    ]


def _hourly_network(weighting):
    """Two-snapshot network whose weighting is deliberately not its timestep length."""
    timesteps = pd.date_range("2030-01-01 00:00", periods=2, freq="h")
    snapshots = pd.MultiIndex.from_arrays(
        [np.repeat(2030, len(timesteps)), timesteps],
        names=["period", "timestep"],
    )
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.set_investment_periods(periods=[2030])
    n.snapshot_weightings.loc[:, :] = float(weighting)
    return n


def test_storage_elapsed_hours_falls_back_to_the_weighting_without_representative_periods():
    n = _hourly_network(weighting=6.0)

    elapsed = storage_elapsed_hours(n, n.snapshots)

    assert list(elapsed) == [6.0, 6.0]
    assert elapsed.index.equals(n.snapshots)


def test_storage_elapsed_hours_uses_the_timestep_under_representative_periods():
    n = _hourly_network(weighting=6.0)
    n.meta = {
        "representative_periods_plot_metadata": {
            "2030": {"periods": [{"period_id": 0, "kind": "representative", "steps": 2}]},
        },
    }

    from_metadata = storage_elapsed_hours(n, n.snapshots)
    from_config = storage_elapsed_hours(
        _hourly_network(weighting=6.0),
        n.snapshots,
        {"clustering": {"temporal": {"representative_periods": {"enable": True}}}},
    )

    assert list(from_metadata) == [1.0, 1.0]
    assert list(from_config) == [1.0, 1.0]
    assert from_metadata.index.equals(n.snapshots)


def _metadata_network(periods, hours=6):
    """Network whose snapshots are split into the given representative-period blocks."""
    timesteps = pd.date_range("2030-01-01 00:00", periods=hours, freq="h")
    snapshots = pd.MultiIndex.from_arrays(
        [np.repeat(2030, len(timesteps)), timesteps],
        names=["period", "timestep"],
    )
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.set_investment_periods(periods=[2030])
    if periods is not None:
        n.meta = {"representative_periods_plot_metadata": {"2030": {"periods": periods}}}
    return n


def test_extreme_period_snapshots_returns_only_the_extreme_blocks():
    """Blocks marked extreme are picked out in model order, across several of them."""
    n = _metadata_network(
        [
            {"period_id": 0, "kind": "extreme", "steps": 2},
            {"period_id": 1, "kind": "representative", "steps": 2},
            {"period_id": 2, "kind": "extreme", "steps": 2},
        ],
    )

    extreme = extreme_period_snapshots(n, n.snapshots)

    assert list(extreme) == list(n.snapshots[[0, 1, 4, 5]])
    assert extreme.names == n.snapshots.names


def test_extreme_period_snapshots_is_empty_when_no_block_is_extreme():
    """Empty means "representative periods are on and none of them is a stress block"."""
    n = _metadata_network([{"period_id": 0, "kind": "representative", "steps": 6}])

    extreme = extreme_period_snapshots(n, n.snapshots)

    assert extreme is not None
    assert extreme.empty


def test_extreme_period_snapshots_is_none_without_representative_periods():
    """No metadata and no config: nothing to narrow, so callers use every snapshot."""
    n = _metadata_network(periods=None)

    assert extreme_period_snapshots(n, n.snapshots) is None


def test_extreme_period_snapshots_is_none_when_enabled_without_metadata():
    """Config alone cannot tell an extreme block from a representative one."""
    n = _metadata_network(periods=None)
    config = {"clustering": {"temporal": {"representative_periods": {"enable": True, "period_length": 1}}}}

    assert extreme_period_snapshots(n, n.snapshots, config) is None


def test_storage_elapsed_hours_on_a_subset_that_skips_a_block():
    """A gap where a skipped block was is not a timestep: measure on the full timeline."""
    n = _metadata_network(
        [
            {"period_id": 0, "kind": "extreme", "steps": 2},
            {"period_id": 1, "kind": "representative", "steps": 2},
            {"period_id": 2, "kind": "extreme", "steps": 2},
        ],
    )
    n.snapshot_weightings.loc[:, :] = 9.0

    elapsed = storage_elapsed_hours(n, extreme_period_snapshots(n, n.snapshots))

    assert list(elapsed) == [1.0, 1.0, 1.0, 1.0]
    assert list(elapsed.index) == list(n.snapshots[[0, 1, 4, 5]])


def _store_network(block_steps, freq="h", weighting=1.0, standing_loss=0.0):
    """
    Network with one store whose snapshots split into the given blocks, model built.

    The store is added with ``e_cyclic=False``: ``prepare_network`` clears the
    built-in cyclic flags whenever representative-period metadata is present, so
    the only closure a store gets is the per-block one rebuilt in solve_network.
    """
    timesteps = pd.date_range("2030-01-01 00:00", periods=sum(block_steps), freq=freq)
    snapshots = pd.MultiIndex.from_arrays(
        [np.repeat(2030, len(timesteps)), timesteps],
        names=["period", "timestep"],
    )
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.set_investment_periods(periods=[2030])
    n.snapshot_weightings.loc[:, :] = float(weighting)
    n.meta = {
        "representative_periods_plot_metadata": {
            "2030": {
                "periods": [
                    {"period_id": i, "kind": "representative", "steps": steps}
                    for i, steps in enumerate(block_steps)
                ],
            },
        },
    }

    n.madd("Carrier", ["AC", "tes", "CCGT"])
    n.add("Bus", "ac", carrier="AC")
    n.add("Bus", STORE, carrier="tes")
    n.add("Load", "load", bus="ac", p_set=10.0)
    n.add("Generator", "gas", bus="ac", carrier="CCGT", p_nom=100.0, marginal_cost=50.0)
    n.add(
        "Store",
        STORE,
        bus=STORE,
        carrier="tes",
        e_nom=100.0,
        e_cyclic=False,
        standing_loss=standing_loss,
    )
    n.add("Link", "ac tes charger", bus0="ac", bus1=STORE, carrier="tes", p_nom=50.0, efficiency=0.98)
    n.add("Link", "ac tes discharger", bus0=STORE, bus1="ac", carrier="tes", p_nom=50.0, efficiency=0.5)

    n.optimize.create_model(multi_investment_periods=True)
    return n


def _balance_terms(n, position):
    """The store's energy-balance row at snapshot ``position``, as {term: coefficient}.

    Variables are labelled ``e[t<i>]``/``p[t<i>]`` by snapshot position, so a term
    sourced from another block is immediately visible.
    """
    label_names = {}
    for variable, short in (("Store-e", "e"), ("Store-p", "p")):
        labels = n.model.variables[variable].labels.sel(Store=STORE)
        for i, snapshot in enumerate(n.snapshots):
            label_names[int(labels.sel(snapshot=snapshot).item())] = f"{short}[t{i}]"

    constraint = n.model.constraints["Store-energy_balance"]
    selection = {"snapshot": n.snapshots[position], "Store": STORE}
    variables = constraint.vars.sel(selection).values.flat
    coefficients = constraint.coeffs.sel(selection).values.flat
    return {
        label_names[int(label)]: float(coefficient)
        for label, coefficient in zip(variables, coefficients)
        if int(label) != -1
    }


def test_store_energy_balance_closes_each_block_and_never_couples_two_blocks():
    """Within a block the state of charge recurses hour to hour; the first hour of a
    block takes its previous state from that same block's last hour, never from the
    preceding block."""
    n = _store_network([3, 3])

    # What PyPSA itself built: the opening hour has no previous-state term at all,
    # since a non-cyclic store starts from the constant e_initial.
    assert "e[t2]" not in _balance_terms(n, 0)

    add_representative_period_storage_constraints(n, REPRESENTATIVE_CONFIG, n.snapshots)

    previous_of = {0: 2, 1: 0, 2: 1, 3: 5, 4: 3, 5: 4}
    for position, previous in previous_of.items():
        # An exact dict match is the point: any surviving cross-block term, or a
        # second previous-state term, shows up as an extra key.
        assert _balance_terms(n, position) == {
            f"e[t{position}]": -1.0,
            f"p[t{position}]": -1.0,
            f"e[t{previous}]": 1.0,
        }

    constraint = n.model.constraints["Store-energy_balance"]
    assert (constraint.sign == "=").all()
    assert float(constraint.rhs.sel(Store=STORE).sum()) == 0.0
    assert bool(constraint.mask.sel(Store=STORE).all())


def test_store_energy_balance_wraps_on_the_metadata_block_lengths():
    """Blocks of unequal length wrap on their own last hour, not on a fixed stride."""
    n = _store_network([2, 4])

    add_representative_period_storage_constraints(n, REPRESENTATIVE_CONFIG, n.snapshots)

    previous_of = {0: 1, 1: 0, 2: 5, 3: 2, 4: 3, 5: 4}
    for position, previous in previous_of.items():
        assert _balance_terms(n, position) == {
            f"e[t{position}]": -1.0,
            f"p[t{position}]": -1.0,
            f"e[t{previous}]": 1.0,
        }


def test_store_energy_balance_uses_physical_hours_not_the_snapshot_weighting():
    """Under representative periods the weighting is a cluster count, so the energy
    balance has to measure a timestep by the snapshot spacing instead."""
    n = _store_network([2, 2], freq="3h", weighting=9.0, standing_loss=0.01)

    add_representative_period_storage_constraints(n, REPRESENTATIVE_CONFIG, n.snapshots)

    terms = _balance_terms(n, 1)
    assert terms["p[t1]"] == pytest.approx(-3.0)
    assert terms["e[t0]"] == pytest.approx(0.99**3)
