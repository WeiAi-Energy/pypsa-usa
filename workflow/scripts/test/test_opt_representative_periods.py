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
    split_capacity_by_representative_period,
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


def _capacity_network(block_steps):
    """
    One bus with extendable gas, solar and a battery over the given blocks.

    Every snapshot carries the same weighting, so the blocks can only be told apart
    from the metadata, never from the weights.
    """
    timesteps = pd.date_range("2030-01-01 00:00", periods=sum(block_steps), freq="h")
    snapshots = pd.MultiIndex.from_arrays(
        [np.repeat(2030, len(timesteps)), timesteps],
        names=["period", "timestep"],
    )
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.set_investment_periods(periods=[2030])
    n.snapshot_weightings.loc[:, :] = 1.0
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

    steps = len(snapshots)
    n.madd("Carrier", ["AC", "CCGT", "solar", "battery"])
    n.add("Bus", "ac", carrier="AC")
    n.add("Load", "load", bus="ac", p_set=pd.Series(np.linspace(5.0, 15.0, steps), index=snapshots))
    n.add("Generator", "gas", bus="ac", carrier="CCGT", p_nom_extendable=True, capital_cost=40.0, marginal_cost=50.0)
    n.add(
        "Generator",
        "solar",
        bus="ac",
        carrier="solar",
        p_nom_extendable=True,
        capital_cost=15.0,
        p_max_pu=pd.Series(np.clip(np.sin(np.linspace(0, 3 * np.pi, steps)), 0, 1), index=snapshots),
    )
    n.add(
        "StorageUnit",
        "battery",
        bus="ac",
        carrier="battery",
        p_nom_extendable=True,
        capital_cost=5.0,
        max_hours=2.0,
        efficiency_store=0.9,
        efficiency_dispatch=0.9,
    )
    return n


def _representative_model(block_steps):
    n = _capacity_network(block_steps)
    n.optimize.create_model(multi_investment_periods=True)
    add_representative_period_storage_constraints(n, REPRESENTATIVE_CONFIG, n.snapshots)
    return n


def test_split_points_each_period_at_its_own_capacity_copy():
    """Every per-snapshot row of a period references that period's copy, never the
    original or another period's copy; rows without a snapshot keep the original."""
    block_steps = [2, 3, 2]
    n = _representative_model(block_steps)
    m = n.model
    original = m.variables["Generator-p_nom"].labels
    lower_row_vars = m.constraints["Generator-ext-p_nom-lower"].vars.values.copy()
    dispatch_lower_vars = m.constraints["Generator-ext-p-lower"].vars.values.copy()
    assert (m.constraints["Generator-ext-p-lower"].coeffs.sel(_term=1) == 0).all()

    split_capacity_by_representative_period(n, REPRESENTATIVE_CONFIG, n.snapshots)

    copies = m.variables["period_split_Generator-p_nom"].labels
    block_of = np.repeat(np.arange(len(block_steps)), block_steps)
    upper = m.constraints["Generator-ext-p-upper"]
    capacity_labels = set(original.values.flat) | set(copies.values.flat)
    for position, snapshot in enumerate(n.snapshots):
        for generator in ("gas", "solar"):
            row = {"snapshot": snapshot, "Generator-ext": generator}
            terms = {
                int(label)
                for label, coeff in zip(upper.vars.sel(row).values.flat, upper.coeffs.sel(row).values.flat)
                if coeff != 0
            }
            own = int(copies.sel(period_block=block_of[position], **{"Generator-ext": generator}))
            # solar at night has p_max_pu = 0 and so no capacity term at all
            expected = {own} if (generator == "gas" or n.generators_t.p_max_pu.loc[snapshot, "solar"] > 0) else set()
            assert terms & capacity_labels == expected

    np.testing.assert_array_equal(m.constraints["Generator-ext-p_nom-lower"].vars.values, lower_row_vars)
    # p_min_pu = 0: the dispatch lower bound holds 0 * p_nom, which never reaches the solver
    np.testing.assert_array_equal(m.constraints["Generator-ext-p-lower"].vars.values, dispatch_lower_vars)
    link = m.constraints["period_split_Generator-p_nom-link"]
    assert (link.labels >= 0).sum() == len(block_steps) * 2
    assert (link.sign == "=").all()
    assert "period_split_StorageUnit-p_nom" in m.variables


def test_split_leaves_the_optimum_unchanged():
    """The split is an exact reformulation: same cost, capacities and prices."""
    results = {}
    for split in (False, True):
        n = _capacity_network([2, 3, 2])

        def extra(network, snapshots, split=split):
            add_representative_period_storage_constraints(network, REPRESENTATIVE_CONFIG, snapshots)
            if split:
                assert split_capacity_by_representative_period(network, REPRESENTATIVE_CONFIG, snapshots) > 0

        status, _ = n.optimize(solver_name="highs", multi_investment_periods=True, extra_functionality=extra)
        assert status == "ok"
        results[split] = n

    base, split = results[False], results[True]
    assert split.objective == pytest.approx(base.objective, rel=1e-8)
    pd.testing.assert_series_equal(split.generators.p_nom_opt, base.generators.p_nom_opt, rtol=1e-6)
    pd.testing.assert_series_equal(split.storage_units.p_nom_opt, base.storage_units.p_nom_opt, rtol=1e-6)
    pd.testing.assert_frame_equal(split.buses_t.marginal_price, base.buses_t.marginal_price, rtol=1e-6, atol=1e-6)
    assert "period_split_Generator-p_nom_opt" not in split.generators


def test_split_is_a_no_op_without_representative_periods():
    n = _representative_model([2, 2])
    before = set(n.model.variables)

    added = split_capacity_by_representative_period(
        n,
        {"clustering": {"temporal": {"representative_periods": {"enable": False}}}},
        n.snapshots,
    )

    assert added == 0
    assert set(n.model.variables) == before


def test_split_leaves_a_variable_only_one_period_uses():
    """A snapshot-independent variable referenced from a single period is already
    local to it and gets no copy."""
    n = _representative_model([2, 2])
    m = n.model
    x = m.add_variables(lower=0, name="custom_x")
    first = n.snapshots[:1]
    m.add_constraints(m.variables["Generator-p"].sel(snapshot=first, Generator="gas") - x <= 0, name="custom_first")

    split_capacity_by_representative_period(n, REPRESENTATIVE_CONFIG, n.snapshots)

    assert "period_split_custom_x" not in m.variables
    assert int(x.labels) in set(m.constraints["custom_first"].vars.values.flat)
    assert "period_split_Generator-p_nom" in m.variables


def _meshed_capacity_network(block_steps):
    """``_capacity_network`` spread over a triangle of extendable lines, so the
    transmission iteration has a cycle whose voltage law it linearises."""
    n = _capacity_network(block_steps)
    n.add("Bus", "b1", carrier="AC")
    n.add("Bus", "b2", carrier="AC")
    n.generators.loc["solar", "bus"] = "b1"
    n.storage_units.loc["battery", "bus"] = "b1"
    n.loads.loc["load", "bus"] = "b2"
    for name, (bus0, bus1, x) in {"l0": ("ac", "b1", 0.1), "l1": ("b1", "b2", 0.2), "l2": ("b2", "ac", 0.3)}.items():
        n.add(
            "Line",
            name,
            bus0=bus0,
            bus1=bus1,
            x=x,
            r=0.01,
            s_nom=4.0,
            s_nom_min=4.0,
            s_nom_extendable=True,
            capital_cost=1.0 + x,
        )
    return n


def test_every_transmission_iteration_solves_the_split_model(monkeypatch):
    """Each iterate and the closing report solve build a fresh model; each of them
    reaches the solver with its capacities split, and the run lands on the same plan
    as without the split."""
    import linopy

    solved = []
    original_solve = linopy.Model.solve

    def recording_solve(self, *args, **kwargs):
        solved.append({name for name in self.variables if name.startswith("period_split_")})
        return original_solve(self, *args, **kwargs)

    monkeypatch.setattr(linopy.Model, "solve", recording_solve)

    results = {}
    for split in (False, True):
        solved.clear()
        n = _meshed_capacity_network([2, 3, 2])

        def extra(network, snapshots, split=split):
            add_representative_period_storage_constraints(network, REPRESENTATIVE_CONFIG, snapshots)
            if split:
                split_capacity_by_representative_period(network, REPRESENTATIVE_CONFIG, snapshots)

        status, _ = n.optimize.optimize_transmission_expansion_iteratively(
            solver_name="highs",
            multi_investment_periods=True,
            min_iterations=3,
            max_iterations=3,
            extra_functionality=extra,
        )
        assert status == "ok"
        results[split] = (n, list(solved))

    base, base_solves = results[False]
    split, split_solves = results[True]
    # three iterates and the report solve
    assert len(split_solves) == len(base_solves) == 4
    assert all(not names for names in base_solves)
    for names in split_solves[:-1]:
        assert {"period_split_Line-s_nom", "period_split_Generator-p_nom", "period_split_StorageUnit-p_nom"} <= names
    # the report solve pins the lines; the generation capacities stay split
    assert "period_split_Generator-p_nom" in split_solves[-1]

    assert split.objective == pytest.approx(base.objective, rel=1e-6)
    pd.testing.assert_series_equal(split.lines.s_nom_opt, base.lines.s_nom_opt, rtol=1e-5)
    pd.testing.assert_series_equal(split.generators.p_nom_opt, base.generators.p_nom_opt, rtol=1e-5)
