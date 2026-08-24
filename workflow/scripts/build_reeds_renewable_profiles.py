"""Build 15-weather-year ReEDS renewable profiles on model substations."""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import tables
import xarray as xr
from _helpers import configure_logging, get_weather_year_snapshots

logger = logging.getLogger(__name__)

CF_SCALE = 1e-4  # cf_profile_<year> stored as integers (x 1e4)
FLOATING_DEPTH_M = 60.0  # <= this depth -> fixed offwind, else floating
# ReEDS UPV hourly profiles are in MW_AC,grid / MW_DC,array; multiply by the
# inverter loading ratio to reach MW_AC,grid / MW_AC,nameplate (the AC-nameplate
# basis the model's p_nom + capex and the supply-curve `cf` column use). See the
# ReEDS UPV technology assumptions (1-axis, ILR 1.34). Capped at 1.0 since AC
# output cannot exceed the inverter rating.
#
# The clip is non-linear, so this must be applied *per site* before any spatial
# aggregation -- scaling an already-averaged profile would overstate solar output.
# That is why `national_available_generation` takes a `transform` callback.
SOLAR_INVERTER_LOADING_RATIO = 1.34
SUPPORTED_WEATHER_YEARS = (2007, 2008, 2009, 2010, 2011, 2012, 2013, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023)

REEDS_TECH = {
    "onwind": {
        "sc": "onwind/sc_wind-ons_reference.csv",
        "cf": "onwind/cf_wind-ons_reference.h5",
        "sitemap": "onwind/sitemap.csv",
        "interconnection": "interconnection_land.h5",
        "cost_field": "cost_total_trans_usd_per_mw",
    },
    "solar": {
        "sc": "solar/sc_upv_reference.csv",
        "cf": "solar/cf_upv_reference.h5",
        "sitemap": "solar/sitemap.csv",
        "interconnection": "interconnection_land.h5",
        "cost_field": "cost_total_trans_usd_per_mw",
    },
    "offwind": {
        "sc": "offwind/sc_wind-ofs_reference.csv",
        "cf": "offwind/cf_wind-ofs_reference.h5",
        "sitemap": "offwind/sitemap_offshore.csv",
        "interconnection": "interconnection_offshore.h5",
        "cost_field": "cost_total_trans_usd_per_mw|radial",
    },
}
REEDS_TECH["offwind_floating"] = REEDS_TECH["offwind"]


def load_sites(vre_dir: str, interconnection_dir: str, tech: str) -> pd.DataFrame:
    """Load and join ReEDS supply-curve, site, and interconnection data."""
    cfg = REEDS_TECH[tech]
    supply_curve = pd.read_csv(f"{vre_dir}/{cfg['sc']}")
    sitemap = pd.read_csv(f"{vre_dir}/{cfg['sitemap']}")
    with tables.open_file(f"{interconnection_dir}/{cfg['interconnection']}", "r") as h5:
        interconnection = pd.DataFrame(
            {
                "sc_point_gid": h5.get_node("/data/sc_point_gid")[:],
                "FIPS": [value.decode() for value in h5.get_node("/data/FIPS")[:]],
                "cost_trans_usd_per_mw": h5.get_node(
                    "/data/" + cfg["cost_field"],
                )[:],
            },
        )
    sites = supply_curve.merge(sitemap, on="sc_point_gid", how="inner").merge(
        interconnection,
        on="sc_point_gid",
        how="left",
    )
    sites["cost_trans_usd_per_mw"] = sites["cost_trans_usd_per_mw"].fillna(0.0)
    return sites


def split_offshore_by_depth(
    sites: pd.DataFrame,
    gebco_path: str,
    want_floating: bool,
) -> pd.DataFrame:
    """Split the shared ReEDS offshore curve into fixed and floating sites."""
    with rasterio.open(gebco_path) as src:
        west_edge = src.bounds.left
        elevation = np.array(
            [value[0] for value in src.sample(zip(sites.longitude, sites.latitude))],
            dtype=float,
        )
        nodata = src.nodata
    depth = -elevation
    outside = (
        (sites.longitude.to_numpy() < west_edge)
        | (elevation == nodata)
        | (np.abs(elevation) > 1e4)
    )
    floating = np.where(outside, True, depth > FLOATING_DEPTH_M)
    return sites.loc[floating if want_floating else ~floating].copy()


def read_cf_for_sites(
    cf_path: str,
    gids: list[int],
    weather_year: int,
    rows=None,
) -> pd.DataFrame:
    """
    Read one ReEDS profile block without shifting time zones.

    ``rows`` optionally restricts the read to those hour-of-year positions, which
    is how representative-period runs avoid pulling all 8760 hours off disk. The
    returned frame is indexed by the row positions actually read.
    """
    with tables.open_file(cf_path, "r") as h5:
        all_gids = h5.get_node("/columns")[:]
        gid_to_column = {int(gid): index for index, gid in enumerate(all_gids)}
        present_gids = [gid for gid in gids if gid in gid_to_column]
        columns = [gid_to_column[gid] for gid in present_gids]
        node_name = f"/cf_profile_{weather_year}"
        if node_name not in h5:
            raise ValueError(f"ReEDS profile {cf_path} has no weather year {weather_year}.")
        node = h5.get_node(node_name)
        if rows is None:
            index = pd.RangeIndex(node.shape[0])
            values = node[:, :][:, columns].astype(np.float32)
        else:
            # int64 explicitly: numpy's default int is 32-bit on Windows, which
            # would give this index a different dtype than the full-read path.
            index = pd.Index(np.asarray(rows, dtype=np.int64))
            # Read each requested hour individually: a handful of representative
            # windows is far cheaper this way than materialising all 8760 rows.
            values = np.vstack([node[int(row), :][columns] for row in index]).astype(np.float32)
    return pd.DataFrame(values * CF_SCALE, columns=present_gids, index=index)


def national_available_generation(
    cf_path: str,
    gids: list[int],
    capacities,
    weather_year: int,
    transform=None,
    row_block: int = 2048,
) -> np.ndarray:
    """
    Return total available generation, ``sum_i(capacity_i * cf_i(t))``, for one weather year.

    This is the wind/solar clustering feature: a single MW series per technology
    summed over every ReEDS site, with no spatial aggregation. Reads in row
    blocks and collapses as it goes, so peak memory stays at one block rather
    than the whole (8760 x n_sites) table.

    ``transform`` is applied to each block of *site-level* CF values before
    summing, so non-linear per-site steps (solar's inverter loading ratio
    followed by a clip at 1.0) land on the same side of the aggregation as they
    do in ``main``.
    """
    capacities = pd.Series(capacities, dtype="float64")
    with tables.open_file(cf_path, "r") as h5:
        all_gids = h5.get_node("/columns")[:]
        gid_to_column = {int(gid): index for index, gid in enumerate(all_gids)}
        present_gids = [gid for gid in gids if gid in gid_to_column]
        if not present_gids:
            raise ValueError(f"None of the requested sites are present in {cf_path}.")
        columns = [gid_to_column[gid] for gid in present_gids]
        weights = capacities.reindex(present_gids).fillna(0.0).to_numpy(dtype=float)
        if float(weights.sum()) <= 0:
            raise ValueError(f"Requested sites in {cf_path} carry no positive capacity.")

        node_name = f"/cf_profile_{weather_year}"
        if node_name not in h5:
            raise ValueError(f"ReEDS profile {cf_path} has no weather year {weather_year}.")
        node = h5.get_node(node_name)
        out = np.empty(node.shape[0], dtype=np.float64)
        for start in range(0, node.shape[0], row_block):
            stop = min(start + row_block, node.shape[0])
            block = node[start:stop, :][:, columns].astype(np.float64) * CF_SCALE
            if transform is not None:
                block = transform(block)
            out[start:stop] = block.dot(weights)
    return out


def source_rows_by_weather_year(source_timesteps, years):
    """
    Map representative source hours onto per-weather-year row positions.

    Returns ``{weather_year: (row_positions, timestamps)}`` for the years the
    representative windows actually touch, so untouched weather years are never
    opened.
    """
    source_timesteps = pd.DatetimeIndex(source_timesteps)
    mapping = {}
    for year in years:
        year_snapshots = get_weather_year_snapshots([year], drop_leap_day=True)
        wanted = pd.DatetimeIndex(sorted(set(source_timesteps[source_timesteps.year == year])))
        if wanted.empty:
            continue
        positions = year_snapshots.get_indexer(wanted)
        if (positions < 0).any():
            missing = wanted[positions < 0]
            raise ValueError(
                f"Representative source hours are not on the {year} weather-year calendar "
                f"({len(missing)} missing, first {missing[0]}).",
            )
        mapping[year] = (positions, wanted)
    return mapping


def map_sites_to_regions(sites: pd.DataFrame, regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    points = gpd.GeoDataFrame(
        sites,
        geometry=gpd.points_from_xy(sites.longitude, sites.latitude),
        crs=4326,
    )
    return gpd.sjoin(
        points,
        regions.to_crs(4326)[["geometry"]],
        how="inner",
        predicate="within",
    ).rename(columns={"index_right": "bus"})


def aggregate_static_site_data(joined: gpd.GeoDataFrame):
    p_nom_max = joined.groupby("bus").capacity.sum()
    cost_trans = joined.groupby("bus")[["cost_trans_usd_per_mw", "capacity"]].apply(
        lambda group: np.average(
            group.cost_trans_usd_per_mw,
            weights=group.capacity,
        )
        if group.capacity.sum() > 0
        else 0.0,
    )
    lcoe_cf = joined.groupby("bus")[["cf", "capacity"]].apply(
        lambda group: np.average(
            group.loc[group.cf.notna(), "cf"],
            weights=group.loc[group.cf.notna(), "capacity"],
        )
        if group.loc[group.cf.notna(), "capacity"].sum() > 0
        else np.nan,
    )
    return p_nom_max, cost_trans, lcoe_cf


def aggregate_profile(
    joined: gpd.GeoDataFrame,
    cf: pd.DataFrame,
    buses: pd.Index,
) -> pd.DataFrame:
    profile = pd.DataFrame(0.0, index=cf.index, columns=buses, dtype=np.float32)
    for bus, group in joined.groupby("bus"):
        gids = [gid for gid in group.sc_point_gid if gid in cf.columns]
        if not gids:
            continue
        weights = group.set_index("sc_point_gid").loc[gids, "capacity"].to_numpy()
        if weights.sum() > 0:
            profile[bus] = np.average(cf[gids].to_numpy(), axis=1, weights=weights).astype(np.float32)
    return profile


def main(snakemake) -> None:
    configure_logging(snakemake)
    tech = snakemake.wildcards.technology
    years = tuple(int(year) for year in snakemake.params.renewable_weather_years)
    if years != SUPPORTED_WEATHER_YEARS:
        raise ValueError(
            "renewable_weather_years must be the ordered 15-year ReEDS set "
            f"{SUPPORTED_WEATHER_YEARS}; received {years}.",
        )

    regions = gpd.read_file(snakemake.input.regions).set_index("name").rename_axis("bus")
    sites = load_sites(
        snakemake.params.reeds_vre_dir,
        snakemake.params.reeds_interconnection_dir,
        tech,
    )
    if tech.startswith("offwind"):
        sites = split_offshore_by_depth(
            sites,
            snakemake.input.gebco,
            tech == "offwind_floating",
        )

    joined = map_sites_to_regions(sites, regions)
    p_nom_max, cost_trans, lcoe_cf = aggregate_static_site_data(joined)
    buses = p_nom_max.index
    cf_path = f"{snakemake.params.reeds_vre_dir}/{REEDS_TECH[tech]['cf']}"

    # With representative periods active, only the selected hours are built. The
    # selection ran upstream on national-aggregate features, so the per-bus
    # profiles below never have to touch all 15 weather years.
    representative_snapshots = getattr(snakemake.input, "representative_snapshots", None)
    if representative_snapshots is not None:
        from select_representative_periods import read_representative_snapshots

        _, _, source_timesteps = read_representative_snapshots(representative_snapshots)
        rows_by_year = source_rows_by_weather_year(source_timesteps, years)
        logger.info(
            "Building %s representative hours for %s across %s of %s weather years.",
            sum(len(rows) for rows, _ in rows_by_year.values()),
            tech,
            len(rows_by_year),
            len(years),
        )
    else:
        rows_by_year = {
            year: (None, get_weather_year_snapshots([year], drop_leap_day=True)) for year in years
        }

    yearly_profiles = []
    for weather_year, (rows, year_snapshots) in rows_by_year.items():
        cf = read_cf_for_sites(cf_path, sites.sc_point_gid.tolist(), weather_year, rows=rows)
        if tech == "solar":
            cf = (cf * SOLAR_INVERTER_LOADING_RATIO).clip(upper=1.0)
        profile = aggregate_profile(joined, cf, buses)
        if len(profile) != len(year_snapshots):
            raise ValueError(
                f"ReEDS {tech} {weather_year} has {len(profile)} rows; expected {len(year_snapshots)}.",
            )
        profile.index = year_snapshots
        yearly_profiles.append(profile)

    profile_df = pd.concat(yearly_profiles).sort_index()
    dataset = xr.Dataset(
        {
            "profile": (("time", "bus"), profile_df.to_numpy()),
            "weight": ("bus", p_nom_max.reindex(buses).to_numpy()),
            "p_nom_max": ("bus", p_nom_max.reindex(buses).to_numpy()),
            "lcoe_cf": ("bus", lcoe_cf.reindex(buses).to_numpy()),
            "average_distance": ("bus", np.zeros(len(buses))),
            "cost_trans_usd_per_mw": ("bus", cost_trans.reindex(buses).to_numpy()),
        },
        coords={"time": profile_df.index, "bus": buses},
    )
    if tech.startswith("offwind"):
        dataset["underwater_fraction"] = ("bus", np.zeros(len(buses)))
    dataset.to_netcdf(snakemake.output.profile)
    logger.info("Wrote %s with %s hourly snapshots.", snakemake.output.profile, len(profile_df))


if __name__ == "__main__":
    main(snakemake)
