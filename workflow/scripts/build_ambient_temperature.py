"""
Hourly ambient air temperature of each ``{simpl}`` bus region from the ERA5 cutout.

The temperature of a bus is the area-weighted mean 2 m air temperature (degC)
over the bus's onshore region. It feeds the temperature-dependent forced-outage
derate of thermal generators (``conventional.ambient_derate``, see
``ambient_derate.py``).

The weather year is the first entry of ``renewable_weather_years``. It is laid
onto the model snapshots the same way every other weather-driven input is
(``_helpers.weather_year_hours``), so the temperature of a snapshot comes from
the same weather hour as its load and renewable profiles.
"""

import logging

import atlite
import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr
from _helpers import (
    configure_logging,
    get_snapshots,
    label_with_snapshots,
    stamp_weather_calendar,
    weather_year_hours,
)

logger = logging.getLogger(__name__)

AREA_CRS = 5070  # CONUS Albers equal area


def weather_hours_for_snapshots(sns: pd.DatetimeIndex, weather_year: int) -> pd.DatetimeIndex:
    """The cutout hours the snapshots read when the weather year differs from the snapshot year."""
    shifted = pd.DatetimeIndex([t.replace(year=weather_year) for t in sns])
    return weather_year_hours(shifted)


def area_weights(cutout: atlite.Cutout, regions: gpd.GeoSeries) -> sp.csr_matrix:
    """
    Row-normalised overlap matrix (regions x cells): row ``i`` averages a cell
    field over region ``i`` weighted by the area each cell shares with it.
    A region the grid does not overlap (smaller than a cell's centre spacing
    never happens with ERA5, but guard it) takes the cell nearest its centroid.
    """
    matrix = sp.csr_matrix(cutout.indicatormatrix(regions, shapes_crs=4326))
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    empty = np.flatnonzero(row_sum == 0)
    if len(empty):
        matrix = matrix.tolil()
        grid = cutout.grid
        for i in empty:
            c = regions.iloc[i].centroid
            j = int(((grid.x - c.x) ** 2 + (grid.y - c.y) ** 2).to_numpy().argmin())
            matrix[i, j] = 1.0
        matrix = matrix.tocsr()
        row_sum = np.asarray(matrix.sum(axis=1)).ravel()
        logger.warning(f"{len(empty)} regions do not overlap the cutout grid; using the nearest cell for them.")
    return sp.diags(1.0 / row_sum) @ matrix


def main(snakemake):
    sns = get_snapshots(snakemake.params.snapshots)
    weather_year = int(snakemake.params.weather_years[0])

    regions = gpd.read_file(snakemake.input.regions_onshore).set_index("name")
    regions = regions.to_crs(4326)
    regions.index = regions.index.astype(str)

    cutout = atlite.Cutout(snakemake.input.cutout)
    # crop to the footprint so the conversion only reads the cells it needs
    minx, miny, maxx, maxy = regions.total_bounds
    dx, dy = cutout.dx, cutout.dy
    cutout = cutout.sel(x=slice(minx - 2 * dx, maxx + 2 * dx), y=slice(miny - 2 * dy, maxy + 2 * dy))
    cutout.data = cutout.data.sel(time=weather_hours_for_snapshots(sns, weather_year))

    matrix = area_weights(cutout, regions.geometry)
    temperature = cutout.temperature(matrix=matrix, index=regions.index, show_progress=False)
    temperature = temperature.rename({temperature.dims[-1]: "bus"}) if "bus" not in temperature.dims else temperature
    temperature = label_with_snapshots(temperature.transpose("time", "bus"), sns).astype("float32")

    area = regions.to_crs(AREA_CRS).area.div(1e6)
    temperature = temperature.assign_coords(area=("bus", area.reindex(temperature.indexes["bus"]).to_numpy()))
    temperature.name = "temperature"
    temperature.attrs.update(
        units="degC",
        weather_year=weather_year,
        description="area-weighted mean 2 m air temperature of the onshore bus region",
    )
    temperature["area"].attrs["units"] = "km2"

    if temperature.isnull().any():
        raise ValueError("The cutout left NaN temperatures for some buses; check that it covers the footprint.")
    logger.info(
        f"Built ambient temperature for {temperature.sizes['bus']} buses, weather year {weather_year}: "
        f"{float(temperature.min()):.1f} to {float(temperature.max()):.1f} degC.",
    )
    stamp_weather_calendar(temperature).to_netcdf(snakemake.output.temperature)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_ambient_temperature", interconnect="western", simpl="33")
    configure_logging(snakemake)
    xr.set_options(keep_attrs=True)
    main(snakemake)
