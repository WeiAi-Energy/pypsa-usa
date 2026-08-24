import pandas as pd
import pypsa
import pytest

from postprocess_io import (
    load_postprocess_network,
    representative_period_metadata,
    snapshot_weight_series,
)


def _write_network(path):
    n = pypsa.Network()
    snapshots = pd.MultiIndex.from_product(
        [[2050], pd.date_range("2012-01-01", periods=2, freq="h")],
        names=["investment_period", "source_time"],
    )
    n.set_snapshots(snapshots)
    n.snapshot_weightings.loc[:, "objective"] = [3.0, 4.0]
    n.snapshot_weightings.loc[:, "generators"] = [5.0, 6.0]
    n.meta = {"representative_periods_plot_metadata": {"2050": {"periods": []}}}
    n.export_to_netcdf(path)


def test_load_postprocess_network_normalizes_snapshot_names(tmp_path):
    path = tmp_path / "network.nc"
    _write_network(path)

    n = load_postprocess_network(path)

    assert n.snapshots.names == ["period", "timestep"]
    assert n.snapshot_weightings.index.equals(n.snapshots)
    assert snapshot_weight_series(n, "generators").tolist() == [5.0, 6.0]
    assert "2050" in representative_period_metadata(n)


def test_snapshot_weight_series_rejects_incomplete_weights():
    n = pypsa.Network()
    snapshots = pd.MultiIndex.from_product(
        [[2050], pd.date_range("2012-01-01", periods=2, freq="h")],
        names=["period", "timestep"],
    )
    n.set_snapshots(snapshots)
    n.snapshot_weightings.loc[:, :] = float("nan")

    with pytest.raises(ValueError, match="no complete"):
        snapshot_weight_series(n)

