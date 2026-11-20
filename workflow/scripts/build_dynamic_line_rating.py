"""
Aggregate hourly dynamic line ratings (DLR) of the base network onto the model branches.

Input is a CSV of hourly relative ratings with one column per line of
``elec_base_network.nc`` (``lines.dynamic_line_rating.file``, e.g. ``dlr_2012.csv``:
8760 rows from 1 January 00:00 of the weather year). Output is a table of hourly
relative ratings per model branch (``Line``, and the ``AC`` ``Link`` s of a transport
model) on the 365-day calendar of the weather year, which ``prepare_network``
multiplies onto ``s_max_pu`` / ``p_max_pu``.

Mapping
-------
* Each base line is followed through the busmaps (base -> substation -> simpl ->
  clusters) to the two model buses its ends land on. Lines whose ends land on the
  same bus are internal and dropped.
* A model branch joining buses (A, B) takes the s_nom-weighted mean of the base
  lines mapped onto (A, B), weights normalized by the summed base s_nom of those
  lines (not by the branch's own s_nom, which clustering/calibration may change).
  Every Line / AC Link joining (A, B) gets that rating (forward, reverse and
  vintaged ITL links alike); branches without base lines keep 1.0.
* Base lines whose ``length`` (great-circle distance x ``lines.length_factor``,
  as ``build_base_network`` stores it) exceeds
  ``lines.dynamic_line_rating.max_length_km`` keep the static rating (1.0).

Time axis
---------
The input rows are taken by position as the hours of the weather year on a
365-day calendar (1 Jan 00:00 ... 31 Dec 23:00 labels). For a leap weather year
whose input holds 1 Jan - 30 Dec (Feb 29 kept, Dec 31 dropped, as ``dlr_2012.csv``
and ReEDS do), the Feb 29 hours take the 1 March labels - the convention of the
ReEDS / GODEEEP capacity factors and EER demand in PyPSA-USA.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import pypsa
import scipy.sparse as sp
from _helpers import configure_logging
from constants import HOURS_PER_YEAR

logger = logging.getLogger(__name__)

ROWS_PER_CHUNK = 500  # CSV rows read at a time; the full table of a national grid is several GB


def normalize_ids(values) -> pd.Index:
    """Bus labels as strings without the float suffix of numeric labels ("35827.0" -> "35827")."""
    return pd.Index(pd.Index(values).astype(str).str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True))


def compose_busmaps(buses, busmap_files) -> pd.Series:
    """Model bus of every base bus, following the busmaps in order (NaN if unmapped)."""
    current = pd.Series(normalize_ids(buses), index=normalize_ids(buses))
    for fn in busmap_files:
        busmap = pd.read_csv(fn, index_col=0, dtype=str).iloc[:, 0]
        busmap = pd.Series(normalize_ids(busmap.to_numpy()), index=normalize_ids(busmap.index))
        current = current.map(busmap[~busmap.index.duplicated(keep="first")])
    return current


def pair_keys(bus0, bus1) -> np.ndarray:
    """Order-independent key of each bus pair."""
    a, b = np.asarray(bus0, dtype=str), np.asarray(bus1, dtype=str)
    return np.where(a < b, np.char.add(np.char.add(a, "|"), b), np.char.add(np.char.add(b, "|"), a))


def model_branches(n: pypsa.Network) -> pd.DataFrame:
    """Lines and AC links of the model network that get a rating."""
    lines = n.lines[["bus0", "bus1"]].assign(component="Line")
    links = n.links.loc[n.links.carrier == "AC", ["bus0", "bus1"]].assign(component="Link")
    branches = pd.concat([lines, links])
    branches.index = branches.index.astype(str)
    if branches.index.duplicated().any():
        raise ValueError("Line and AC Link names overlap; cannot store both in one rating table.")
    return branches


def rating_weights(base: pd.DataFrame, bus_to_model: pd.Series, branches: pd.DataFrame):
    """Sparse (base lines x branches) weights; base s_nom shares normalized per branch."""
    c0 = bus_to_model.reindex(normalize_ids(base.bus0)).to_numpy()
    c1 = bus_to_model.reindex(normalize_ids(base.bus1)).to_numpy()
    unmapped = pd.isna(c0) | pd.isna(c1)
    internal = ~unmapped & (c0 == c1)
    crossing = ~unmapped & ~internal

    rows = pd.DataFrame(
        {
            "row": np.flatnonzero(crossing),
            "key": pair_keys(c0[crossing], c1[crossing]),
            "s_nom": base.s_nom.to_numpy(float)[crossing],
        },
    )
    cols = pd.DataFrame({"col": np.arange(len(branches)), "key": pair_keys(branches.bus0, branches.bus1)})
    pairs = rows.merge(cols, on="key")
    total = pairs.groupby("col")["s_nom"].transform("sum")
    count = pairs.groupby("col")["row"].transform("count")
    w = np.where(total > 0, pairs.s_nom / total.where(total > 0, 1.0), 1.0 / count)
    weights = sp.csr_matrix((w, (pairs.row, pairs.col)), shape=(len(base), len(branches)))
    mapped = np.zeros(len(branches), dtype=bool)
    mapped[pairs.col.unique()] = True
    logger.info(
        f"{len(base)} base lines: {internal.sum()} internal to a model bus, {unmapped.sum()} with an unmapped bus, "
        f"{crossing.sum() - pairs.row.nunique()} between model buses no branch joins. "
        f"{mapped.sum()} of {len(branches)} model branches rated; {(~mapped).sum()} without base lines keep 1.0.",
    )
    return weights, mapped


def calendar_labels(year: int) -> pd.DatetimeIndex:
    """Hourly labels of ``year`` on a 365-day calendar (29 February dropped)."""
    hours = pd.date_range(f"{year}-01-01", periods=HOURS_PER_YEAR + 24, freq="h")
    hours = hours[hours.year == year]
    return hours[~((hours.month == 2) & (hours.day == 29))]


def aggregate_dynamic_line_rating(
    dlr_file: str,
    base_network: pypsa.Network,
    busmap_files: list[str],
    network: pypsa.Network,
    max_length_km: float | None,
) -> pd.DataFrame:
    """Hourly relative rating of every model branch, labelled on the weather year's 365-day calendar."""
    t0 = time.time()
    base = base_network.lines[["bus0", "bus1", "s_nom", "length"]].copy()
    base.index = base.index.astype(str)

    columns = pd.read_csv(dlr_file, nrows=1, index_col=0).columns.astype(str)
    if set(columns) != set(base.index):
        raise ValueError(
            f"{dlr_file} has {len(columns)} line columns, the base network has {len(base)} lines and the IDs differ "
            f"(e.g. {sorted(set(columns) ^ set(base.index))[:5]}): the DLR was computed for another base network "
            "(check model_topology: transmission_network).",
        )
    base = base.loc[columns]  # file column order

    capped = np.zeros(len(base), dtype=bool)
    if max_length_km is not None:
        capped = base.length.to_numpy() > max_length_km
        logger.info(
            f"{capped.sum()} base lines longer than {max_length_km} km "
            "keep the static rating.",
        )

    bus_to_model = compose_busmaps(pd.unique(pd.concat([base.bus0, base.bus1])), busmap_files)
    branches = model_branches(network)
    weights, mapped = rating_weights(base, bus_to_model, branches)
    weights_t = weights.T.tocsr()

    parts, stamps = [], []
    for chunk in pd.read_csv(dlr_file, index_col=0, chunksize=ROWS_PER_CHUNK):
        values = chunk.to_numpy(dtype=np.float32)
        values[:, capped] = 1.0
        parts.append(np.asarray(weights_t.dot(values.T).T, dtype=np.float32))
        stamps.extend(chunk.index)
        logger.info(f"  {len(stamps)} of {HOURS_PER_YEAR} hours aggregated ({time.time() - t0:.0f}s).")
    rating = np.vstack(parts)
    rating[:, ~mapped] = 1.0

    stamps = pd.DatetimeIndex(pd.to_datetime(stamps, utc=True)).tz_localize(None)
    year = stamps[0].year
    if len(stamps) != HOURS_PER_YEAR or stamps[0] != pd.Timestamp(year=year, month=1, day=1):
        raise ValueError(
            f"Expected {HOURS_PER_YEAR} hourly rows from 1 January 00:00 in {dlr_file}, "
            f"got {len(stamps)} from {stamps[0]}.",
        )
    labels = calendar_labels(year)
    if (stamps != labels).any():
        logger.info(
            f"Rows placed by position on the 365-day calendar of {year} (input runs {stamps[0]} - {stamps[-1]})."
        )

    out = pd.DataFrame(rating, index=pd.Index(labels, name="timestep"), columns=branches.index).round(4)
    m = out.loc[:, mapped].to_numpy()
    logger.info(
        f"{out.shape[1]} branches ({(branches.component == 'Line').sum()} Lines, "
        f"{(branches.component == 'Link').sum()} AC Links) x {len(out)} h; rated branches mean "
        f"{m.mean() if m.size else float('nan'):.3f}, {(m < 1).mean() if m.size else 0:.1%} of hours below static "
        f"({time.time() - t0:.0f}s).",
    )
    return out


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_dynamic_line_rating", interconnect="usa", simpl="county", clusters="all")
    configure_logging(snakemake)
    logging.getLogger("pypsa").setLevel(logging.WARNING)

    dlr = snakemake.params.dynamic_line_rating
    rating = aggregate_dynamic_line_rating(
        dlr_file=snakemake.input.dlr,
        base_network=pypsa.Network(snakemake.input.base_network),
        busmap_files=[snakemake.input.busmap_b, snakemake.input.busmap_s, snakemake.input.busmap_c],
        network=pypsa.Network(snakemake.input.network),
        max_length_km=dlr.get("max_length_km"),
    )
    rating.to_parquet(snakemake.output[0])
