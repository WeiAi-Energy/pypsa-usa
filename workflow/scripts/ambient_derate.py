"""
Temperature-dependent forced-outage derate for thermal generators (PyPSA-USA #850).

The available capacity of a thermal unit follows the ambient temperature of
its bus,

    p_max_pu(t) = 1 - FOR(T_bus(t)),

where FOR is the forced outage rate of the unit's prime mover as a function of
temperature from Murphy, Lavin & Apt (2019), as ReEDS does it
(``reeds/input_processing/outage_rates.py``): linear between the published
points (-15 to 35 degC in 5 degC steps), extended linearly through the two
outermost points beyond them, and held within ``[0, max_outage]`` (0.4, the
PJM gas CT capacity credit ReEDS caps the extrapolation at).

``T_bus`` is the area-weighted mean air temperature of the bus's onshore
region, built by ``build_ambient_temperature`` from the ERA5 cutout. It
replaces the EIA-860 summer/winter rating, which picked a value by calendar
month regardless of the weather. Carriers without a curve (geothermal, for
example) are not derated.

The derate is applied wherever thermal generators enter the network: the
existing fleet in ``add_electricity`` (before clustering, at the ``{simpl}``
bus, so a clustered generator takes the capacity-weighted mean of its
members' profiles), and in ``add_extra_components`` the generators of carriers
with no existing fleet, at the clustered bus (area-weighted mean temperature
of the merged region).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
import xarray as xr
from _helpers import check_weather_calendar

logger = logging.getLogger(__name__)

DEFAULT_OUTAGE_CURVES = "repo_data/plants/outage_forced_temperature_murphy2019.csv"
DEFAULT_MAX_OUTAGE = 0.4
#: carrier -> Murphy et al. (2019) prime mover, as ReEDS maps its techs
DEFAULT_PRIME_MOVERS = {
    "CCGT": "combined_cycle",
    "CCGT-95CCS": "combined_cycle",
    "CCGT-97CCS": "combined_cycle",
    "OCGT": "combustion_turbine",
    "hydrogen_ct": "combustion_turbine",
    "oil": "diesel",
    "coal": "steam",
    "coal-95CCS": "steam",
    "coal-99CCS": "steam",
    "biomass": "steam",
    "waste": "steam",
    "nuclear": "nuclear",
    "SMR": "nuclear",
}


def ambient_derate_settings(conventional: dict | None) -> dict:
    """``conventional.ambient_derate`` with defaults for keys a self-contained config leaves out."""
    cfg = dict((conventional or {}).get("ambient_derate") or {})
    return {
        "prime_movers": dict(cfg.get("prime_movers") or DEFAULT_PRIME_MOVERS),
        "max_outage": float(cfg.get("max_outage", DEFAULT_MAX_OUTAGE)),
    }


def load_outage_curves(path: str) -> pd.DataFrame:
    """Return the forced outage rate (fraction) by temperature (index, degC) and prime mover (columns)."""
    curves = pd.read_csv(path, comment="#").pivot(
        index="deg_celsius",
        columns="prime_mover",
        values="outage_frac",
    )
    return curves.sort_index().astype(float)


def forced_outage_rate(
    temperature: pd.DataFrame | np.ndarray,
    curve: pd.Series,
    max_outage: float = 0.4,
):
    """
    Evaluate one prime mover's FOR curve at ``temperature`` (degC).

    Inside the published range the curve is interpolated linearly. Outside it
    the line through the two outermost points is extended, and the result is
    held within ``[0, max_outage]``.
    """
    x = curve.index.to_numpy(dtype=float)
    y = curve.to_numpy(dtype=float)
    t = np.asarray(temperature, dtype=float)

    rate = np.interp(t, x, y)
    slope_low = (y[1] - y[0]) / (x[1] - x[0])
    slope_high = (y[-1] - y[-2]) / (x[-1] - x[-2])
    rate = np.where(t < x[0], y[0] + slope_low * (t - x[0]), rate)
    rate = np.where(t > x[-1], y[-1] + slope_high * (t - x[-1]), rate)
    rate = np.clip(rate, 0.0, max_outage)

    if isinstance(temperature, pd.DataFrame):
        return pd.DataFrame(rate, index=temperature.index, columns=temperature.columns)
    return rate


def load_bus_temperature(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Read the output of ``build_ambient_temperature``.

    Returns the temperature (time x bus, degC) and the onshore region area of
    each bus (km2), the weight for averaging buses that are clustered together.
    """
    da = xr.open_dataarray(path)
    check_weather_calendar(da, path)
    temperature = da.transpose("time", "bus").to_pandas()
    temperature.columns = temperature.columns.astype(str)
    area = da["area"].to_pandas()
    area.index = area.index.astype(str)
    da.close()
    return temperature, area


def aggregate_bus_temperature(
    temperature: pd.DataFrame,
    area: pd.Series,
    busmap: pd.Series,
) -> pd.DataFrame:
    """
    Area-weighted mean temperature of the regions mapped together by ``busmap``.

    The mean over the union of the member regions, so a clustered bus gets the
    average temperature of its whole region.
    """
    busmap = busmap.astype(str)
    busmap.index = busmap.index.astype(str)
    members = temperature.columns.intersection(busmap.index)
    weights = area.reindex(members).fillna(0.0)
    groups = busmap.loc[members]
    weighted = temperature[members].mul(weights, axis=1).T.groupby(groups).sum().T
    total = weights.groupby(groups).sum()
    return weighted.div(total.where(total > 0), axis=1)


def ambient_derate_profiles(
    temperature: pd.DataFrame,
    generators: pd.DataFrame,
    prime_movers: dict[str, str],
    curves: pd.DataFrame,
    max_outage: float = 0.4,
) -> pd.DataFrame:
    """
    Return ``1 - FOR(T_bus(t))`` (time x generator) for ``generators``.

    ``generators`` carries ``bus`` and ``carrier`` columns. Generators whose
    carrier has no entry in ``prime_movers``, or whose bus has no temperature,
    are left out of the result; the caller keeps their existing derate.
    """
    unknown = set(prime_movers.values()) - set(curves.columns)
    if unknown:
        raise ValueError(
            f"conventional.ambient_derate.prime_movers names prime movers {sorted(unknown)} that the outage "
            f"curve file does not have (it has {sorted(curves.columns)}).",
        )

    gens = generators[generators.carrier.isin(prime_movers.keys())]
    no_temp = ~gens.bus.astype(str).isin(temperature.columns)
    if no_temp.any():
        logger.warning(
            f"No ambient temperature for the buses of {no_temp.sum()} generators "
            f"({sorted(gens.loc[no_temp, 'bus'].astype(str).unique())[:10]}); they keep their existing derate.",
        )
        gens = gens[~no_temp]

    profiles = []
    for pm, pm_gens in gens.groupby(gens.carrier.map(prime_movers)):
        buses = pm_gens.bus.astype(str).unique()
        available = 1.0 - forced_outage_rate(temperature[buses], curves[pm], max_outage)
        cols = available[pm_gens.bus.astype(str)]
        cols.columns = pm_gens.index
        profiles.append(cols)

    if not profiles:
        return pd.DataFrame(index=temperature.index)
    return pd.concat(profiles, axis=1)


def clip_p_min_pu_to_p_max_pu(n, gens: pd.Index, headroom: float = 0.95) -> None:
    """
    Keep ``p_min_pu`` under the lowest time-varying ``p_max_pu`` of ``gens``.

    A unit whose minimum stable level exceeds its available capacity in some
    hour is infeasible there (must-run) or forced off (committable). The 5 %
    headroom matches the clip of the minimum load in ``add_electricity``.
    """
    gens = pd.Index(gens).intersection(n.generators_t.p_max_pu.columns)
    if gens.empty:
        return
    cap = n.generators_t.p_max_pu[gens].min().mul(headroom)
    clipped = n.generators.loc[gens, "p_min_pu"] > cap
    if clipped.any():
        logger.info(f"Lowered p_min_pu of {clipped.sum()} generators below their lowest derated p_max_pu.")
        n.generators.loc[gens, "p_min_pu"] = np.minimum(n.generators.loc[gens, "p_min_pu"], cap.round(3))


def apply_ambient_derate(
    n,
    temperature: pd.DataFrame,
    settings: dict,
    curves: pd.DataFrame,
    broadcast: Callable,
) -> pd.Index:
    """
    Set ``p_max_pu(t) = 1 - FOR(T_bus(t))`` on the thermal generators of ``n`` that have no profile yet.

    ``temperature`` (time x bus) must be keyed by the buses of ``n``.
    Generators that already carry a ``p_max_pu`` time series (copies of an
    existing fleet, which inherit its derate) are left alone. ``broadcast``
    lays the single-year profile onto the snapshots of ``n``
    (``add_electricity.broadcast_investment_horizons_index``).

    Returns the generators it derated.
    """
    prime_movers = settings["prime_movers"]
    gens = n.generators[n.generators.carrier.isin(prime_movers)]
    gens = gens[~gens.index.isin(n.generators_t.p_max_pu.columns)]
    if gens.empty:
        return pd.Index([])

    available = ambient_derate_profiles(
        temperature,
        gens[["bus", "carrier"]],
        prime_movers,
        curves,
        max_outage=settings["max_outage"],
    )
    if available.empty:
        return pd.Index([])

    available = broadcast(n, available).round(3)
    derated = available.columns
    n.generators.loc[derated, "p_max_pu"] = 1.0
    n.generators_t.p_max_pu = pd.concat([n.generators_t.p_max_pu, available], axis=1)
    clip_p_min_pu_to_p_max_pu(n, derated)

    info = n.generators.loc[derated, ["carrier", "p_nom"]].assign(
        energy=available.mean() * n.generators.loc[derated, "p_nom"],
        mean_pu=available.mean(),
        min_pu=available.min(),
    )
    summary = info.groupby("carrier").agg(
        n=("p_nom", "size"),
        GW=("p_nom", "sum"),
        energy=("energy", "sum"),
        mean_pu=("mean_pu", "mean"),
        min_pu=("min_pu", "min"),
    )
    summary["GW"] = summary.GW / 1e3
    # capacity-weighted where there is capacity, plain mean otherwise (new builds)
    summary["mean_p_max_pu"] = (summary.energy / (summary.GW * 1e3)).where(summary.GW > 0, summary.mean_pu)
    logger.info(
        "Applied the ambient-temperature FOR derate to %d generators:\n%s",
        len(derated),
        summary[["n", "GW", "mean_p_max_pu", "min_pu"]].round(3),
    )
    return derated
