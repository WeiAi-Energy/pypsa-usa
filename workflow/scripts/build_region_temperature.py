"""
Build per-bus air temperature for the representative hours.

Only the hours actually used by the model are needed, so this downloads ERA5
2 m air temperature for just those hours -- typically a handful of days across
a handful of months rather than the full 15 weather years. Cutouts are cached
on disk per calendar month; the CDS request for a month covers the day x hour
combinations the current representative-period selection needs there, not the
whole month.

Because a month's cutout is cached under one file, a different representative-
period selection that touches the same month can find a stale cutout on disk
that is missing hours it needs. ``build_monthly_cutout`` detects this, deletes
the stale ``.nc`` file, and rebuilds it for the current selection.

Timezone
--------
ERA5 timestamps are UTC, and so are the ``source_timestep`` values in
``snapshots.csv``: renewable profiles are natively UTC, and EER demand is rolled
CST -> UTC by ``build_eer_demand.ReadEer`` before selection. Temperature is
therefore sliced at the source hours **verbatim, with no shift**, and the script
asserts every source hour is present in the cutout. Applying a local-time offset
here would silently put the derate out of phase with load and wind/solar.

Bus matching
------------
Each clustered AC bus is matched directly to the ERA5 grid cell nearest its
(x, y) coordinate. No county shapes or population weighting are involved --
a bus simply reads off the raw grid cell it falls in.
"""

import logging
import os

import atlite
import atlite.datasets.era5 as era5
import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_cds_api, configure_logging
from scipy import sparse
from scipy.spatial import cKDTree
from select_representative_periods import read_representative_snapshots

logger = logging.getLogger(__name__)

GRID_DEG = 0.5  # Coarse grid is plenty for per-bus temperature; keeps CDS requests small.


def patch_era5_air_temperature_only() -> None:
    """Make atlite's ERA5 ``temperature`` feature retrieve only ``2m_temperature``."""
    if era5.features.get("temperature") == ["temperature"]:
        return

    era5.features["temperature"] = ["temperature"]

    def get_data_temperature(retrieval_params):
        ds = era5.retrieve_data(variable=["2m_temperature"], **retrieval_params)
        ds = era5._rename_and_clean_coords(ds)
        return ds.rename({"t2m": "temperature"})

    era5.get_data_temperature = get_data_temperature


def required_months(source_timesteps: pd.DatetimeIndex) -> list[pd.Period]:
    """Return the calendar months the representative source hours fall in."""
    months = pd.PeriodIndex(pd.DatetimeIndex(source_timesteps), freq="M").unique()
    return sorted(months)


def build_monthly_cutout(path, month: pd.Period, hours: pd.DatetimeIndex, bounds: dict) -> atlite.Cutout:
    """Create (or reuse) a 2 m air-temperature cutout covering only ``hours`` (all within ``month``).

    atlite derives the CDS request's day/time lists straight from the cutout's
    time coordinate (``atlite.datasets.era5.retrieval_times``), so passing the
    exact representative hours here -- instead of a full-month slice -- keeps
    the download to the day x hour combinations actually needed rather than
    the whole month.

    If ``path`` already exists on disk, atlite reuses it as-is and ignores
    ``hours``. A cutout built for a different representative-period selection
    can then be missing hours the current selection needs; that is detected
    here and the stale file is deleted so it gets rebuilt from scratch for
    ``hours``.
    """
    path = str(path)
    if not path.endswith(".nc"):
        raise ValueError(f"Cutout path must be a .nc file, got {path}.")

    # The cutout is a cached side-product, not a declared rule output, so Snakemake
    # does not create its directory. atlite writes via mkstemp(dir=...) and fails
    # with FileNotFoundError if it is missing.
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def open_cutout() -> atlite.Cutout:
        return atlite.Cutout(
            path,
            module="era5",
            x=slice(*bounds["x"]),
            y=slice(*bounds["y"]),
            dx=GRID_DEG,
            dy=GRID_DEG,
            time=hours,
        )

    if os.path.exists(path):
        existing = open_cutout()
        available = pd.DatetimeIndex(existing.data.indexes["time"])
        missing = hours.difference(available)
        if not missing.empty:
            logger.warning(
                "ERA5 cutout for %s is missing %s representative hours (first: %s); deleting "
                "%s and rebuilding it for the current representative-period selection.",
                month,
                len(missing),
                missing[0],
                path,
            )
            existing.data.close()
            os.remove(path)

    cutout = open_cutout()
    logger.info("Preparing 2 m air-temperature cutout for %s hours in %s at %s.", len(hours), month, path)
    cutout.prepare(features=["temperature"])
    return cutout


def nearest_cell_matrix(cutout: atlite.Cutout, buses: pd.DataFrame) -> sparse.csr_matrix:
    """Build a (bus x grid cell) matrix picking out each bus's nearest ERA5 grid cell."""
    grid = cutout.grid
    tree = cKDTree(grid[["x", "y"]].to_numpy())
    _, nearest = tree.query(buses[["x", "y"]].to_numpy())
    rows = np.arange(len(buses))
    data = np.ones(len(buses))
    return sparse.csr_matrix((data, (rows, nearest)), shape=(len(buses), len(grid)))


def bus_air_temperature(cutout: atlite.Cutout, matrix: sparse.csr_matrix, index: pd.Index) -> pd.DataFrame:
    """Hourly air temperature in degC as a (time, bus) frame."""
    temperature = cutout.temperature(matrix=matrix, index=index)
    time_dim = "time" if "time" in temperature.dims else temperature.dims[0]
    bus_dim = next(dim for dim in temperature.dims if dim != time_dim)
    return temperature.transpose(time_dim, bus_dim).to_pandas()


def main(snakemake) -> None:
    snapshots, _, source_timesteps = read_representative_snapshots(snakemake.input.representative_snapshots)

    network = pypsa.Network(snakemake.input.network)
    buses = network.buses.loc[network.buses.carrier == "AC", ["x", "y"]]
    if buses.empty:
        raise ValueError("Clustered network has no AC buses to build temperature for.")

    bounds = snakemake.config["atlite"]["interconnects"][snakemake.params.interconnect]
    cutout_dir = snakemake.params.cutout_dir

    patch_era5_air_temperature_only()

    frames = []
    for month in required_months(source_timesteps):
        wanted = pd.DatetimeIndex(
            source_timesteps[pd.PeriodIndex(source_timesteps, freq="M") == month],
        ).unique().sort_values()
        cutout = build_monthly_cutout(f"{cutout_dir}/temperature_era5_{month}.nc", month, wanted, bounds)

        available = pd.DatetimeIndex(cutout.data.indexes["time"])
        missing = wanted.difference(available)
        if not missing.empty:
            raise ValueError(
                f"ERA5 cutout for {month} is still missing {len(missing)} representative hours "
                f"after a rebuild (first: {missing[0]}). ERA5 and the representative source hours "
                "are both UTC, so no timezone shift should be applied.",
            )

        cutout.data = cutout.data.sel(time=wanted)
        matrix = nearest_cell_matrix(cutout, buses)
        frames.append(bus_air_temperature(cutout, matrix, buses.index))

    temperatures = pd.concat(frames).sort_index()
    # Reindex onto the source hours, then relabel with the network's snapshots.
    temperatures = temperatures.reindex(pd.DatetimeIndex(source_timesteps))
    if temperatures.isna().any().any():
        raise ValueError("Bus temperatures do not cover every representative source hour.")

    table = temperatures.reset_index(drop=True)
    table.insert(0, "timestep", pd.DatetimeIndex(snapshots.get_level_values("timestep")))
    table.insert(0, "period", snapshots.get_level_values("period").astype(int))
    table.to_csv(snakemake.output.region_temperature, index=False)
    logger.info(
        "Wrote bus temperature for %s buses over %s snapshots (range %.1f to %.1f degC) to %s.",
        temperatures.shape[1],
        len(table),
        float(temperatures.to_numpy().min()),
        float(temperatures.to_numpy().max()),
        snakemake.output.region_temperature,
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_region_temperature", case="test")
    configure_logging(snakemake)
    configure_cds_api(snakemake)
    main(snakemake)
