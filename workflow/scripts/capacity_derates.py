"""
Temperature-dependent capacity derates for thermal generators and TES.

Forced-outage rates rise steeply at both temperature extremes (Murphy et al.
2019, https://doi.org/10.1016/j.apenergy.2019.113513). This module turns an
hourly regional temperature table into a per-asset availability multiplier on
``p_max_pu``.

This replaces the previous static summer/winter derate, which keyed off the
snapshot month. That is no longer possible: representative-period snapshots
carry *synthetic* contiguous timestamps, so a block sourced from December can be
labelled with January dates. The region temperature table is therefore built
against the real (UTC) source hours upstream and arrives here already indexed by
the network's own snapshots -- never re-derive a season or calendar date from a
representative snapshot label.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
from pypsa.descriptors import get_switchable_as_dense as get_as_dense

logger = logging.getLogger(__name__)

TEMP_MIN = -50
TEMP_MAX = 60
MAX_EXTRAPOLATED_OUTAGE_FORCED = 0.4

# Carrier -> Murphy et al. prime mover. Available curves: combined_cycle,
# combustion_turbine, steam, nuclear, hydro_and_psh, diesel.
# NOTE: `coal` is deliberately absent -- the ported mapping does not cover it, so
# coal units keep p_max_pu = 1.0. Add "coal": "steam" here to derate them too.
DYNAMIC_PRIME_MOVER_BY_CARRIER = {
    "CCGT": "combined_cycle",
    "CCGT-95CCS": "combined_cycle",
    "CCGT-97CCS": "combined_cycle",
    "CCHT": "combined_cycle",
    "hydrogen_ct": "combined_cycle",
    "OCGT": "combustion_turbine",
    "OCHT": "combustion_turbine",
    "nuclear": "nuclear",
}

STORAGE_PRIME_MOVER_BY_CARRIER = {"PHS": "hydro_and_psh"}

LINK_PRIME_MOVER_BY_CARRIER = {
    "CCHT": "combined_cycle",
    "OCHT": "combustion_turbine",
    "tes": "combined_cycle",
}

COMPONENT_LIST_NAME = {
    "Generator": "generators",
    "Link": "links",
    "StorageUnit": "storage_units",
}


def extrapolate_forward_backward(
    dfin: pd.DataFrame,
    xmin: int = TEMP_MIN,
    xmax: int = TEMP_MAX,
    numfitvals: int = 2,
    polyfit_deg: int = 1,
    ymin: float = 0.0,
    ymax: float = MAX_EXTRAPOLATED_OUTAGE_FORCED,
) -> pd.DataFrame:
    """Extrapolate temperature-outage curves to integer temperatures."""
    xs_low = dfin.index[:numfitvals].values
    xs_high = dfin.index[-numfitvals:].values

    slope_low, intercept_low = np.polyfit(xs_low, dfin.loc[xs_low].values, deg=polyfit_deg)
    slope_high, intercept_high = np.polyfit(xs_high, dfin.loc[xs_high].values, deg=polyfit_deg)

    return (
        pd.concat(
            [
                pd.DataFrame({xmin: intercept_low + slope_low * xmin}, index=dfin.columns).T,
                dfin,
                pd.DataFrame({xmax: intercept_high + slope_high * xmax}, index=dfin.columns).T,
            ],
        )
        .reindex(range(xmin, xmax + 1))
        .interpolate("linear")
        .clip(lower=ymin, upper=ymax)
    )


def load_forced_outage_temperature_curves(path: str) -> pd.DataFrame:
    """Read Murphy et al. temperature-dependent forced outage rates."""
    raw = pd.read_csv(path, comment="#")
    required = {"prime_mover", "deg_celsius", "outage_frac"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing forced-outage columns: {sorted(missing)}")
    curves = raw.pivot(index="deg_celsius", columns="prime_mover", values="outage_frac").sort_index()
    return extrapolate_forward_backward(curves)


def read_region_temperatures(path: str, snapshots: pd.Index) -> pd.DataFrame:
    """
    Read the region temperature table and check it matches the network snapshots.

    The table is written upstream against the real UTC source hours and labelled
    with the network's own (period, timestep) snapshots, so alignment here is an
    exact-match assertion rather than a calendar join.
    """
    table = pd.read_csv(path, parse_dates=["timestep"])
    required = {"period", "timestep"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Region temperature table is missing columns {sorted(missing)}.")

    index = pd.MultiIndex.from_arrays(
        [table.period.astype(int), pd.DatetimeIndex(table.timestep)],
        names=["period", "timestep"],
    )
    temperatures = table.drop(columns=["period", "timestep"]).astype(float)
    temperatures.index = index
    temperatures.columns = temperatures.columns.astype(str)

    if not temperatures.index.equals(snapshots):
        raise ValueError(
            "Region temperatures are not aligned with the network snapshots. Rebuild "
            "region_temperature.csv from the same representative-period selection.",
        )
    return temperatures


def outage_fraction(curves: pd.DataFrame, prime_mover: str, temperatures: pd.DataFrame) -> pd.DataFrame:
    """Interpolate outage fraction for each temperature value."""
    if prime_mover not in curves.columns:
        raise KeyError(f"Missing forced-outage curve for prime mover {prime_mover!r}")
    x = curves.index.to_numpy(dtype=float)
    y = curves[prime_mover].to_numpy(dtype=float)
    values = np.interp(temperatures.to_numpy(dtype=float), x, y)
    return pd.DataFrame(values, index=temperatures.index, columns=temperatures.columns).clip(lower=0.0, upper=1.0)


def _multiply_p_max_pu(
    n: pypsa.Network,
    component: str,
    names: pd.Index,
    availability: pd.DataFrame,
) -> None:
    """Multiply the given assets' ``p_max_pu`` by an availability time series."""
    names = pd.Index(names)
    if names.empty:
        return

    factors = availability.reindex(index=n.snapshots, columns=names).fillna(1.0)

    current = get_as_dense(n, component, "p_max_pu").reindex(index=n.snapshots, columns=names)
    table = getattr(n, COMPONENT_LIST_NAME[component])
    static = table.loc[names, "p_max_pu"].fillna(1.0) if "p_max_pu" in table else pd.Series(1.0, index=names)
    current = current.fillna(static)
    adjusted = current.mul(factors).clip(lower=0.0, upper=1.0)

    if "p_min_pu" in table:
        p_min_pu = pd.to_numeric(table.loc[names, "p_min_pu"], errors="coerce").fillna(0.0)
        violated = names[(adjusted.min(axis=0) < p_min_pu).to_numpy()]
        if len(violated):
            logger.warning(
                "Temperature derates push p_max_pu below p_min_pu for %s %s assets (e.g. %s); "
                "these may be infeasible unless p_min_pu is relaxed.",
                len(violated),
                component,
                list(violated[:5]),
            )

    pnl = getattr(n, COMPONENT_LIST_NAME[component] + "_t")
    existing = pnl["p_max_pu"].drop(columns=names.intersection(pnl["p_max_pu"].columns), errors="ignore")
    pnl["p_max_pu"] = existing.join(adjusted)


def _component_bus(n: pypsa.Network, component: str, names: pd.Index) -> pd.Series:
    """Return the bus whose temperature governs each asset."""
    if component == "Link":
        buses = n.links.loc[names, "bus1"].astype(str)
        fallback = n.links.loc[names, "bus0"].astype(str)
        return buses.where(buses.isin(n.buses.index), fallback)
    if component == "StorageUnit":
        return n.storage_units.loc[names, "bus"].astype(str)
    return n.generators.loc[names, "bus"].astype(str)


def _temperature_for_assets(temperatures: pd.DataFrame, buses: pd.Series) -> pd.DataFrame:
    """Broadcast regional temperatures onto assets, falling back to the national mean."""
    national = temperatures.mean(axis=1)
    data = {}
    missing = []
    for name, bus in buses.items():
        if bus in temperatures.columns:
            data[name] = temperatures[bus]
        else:
            data[name] = national
            missing.append(bus)
    if missing:
        logger.warning("Using mean regional temperature for missing buses: %s", sorted(set(missing)))
    return pd.DataFrame(data, index=temperatures.index)


def apply_dynamic_component_derates(
    n: pypsa.Network,
    component: str,
    carrier_to_prime_mover: dict[str, str],
    temperatures: pd.DataFrame,
    outage_curves: pd.DataFrame,
    name_filter=None,
) -> None:
    """Apply temperature-dependent outage rates to one component table."""
    table = getattr(n, COMPONENT_LIST_NAME[component])
    if table.empty:
        return

    for carrier, prime_mover in carrier_to_prime_mover.items():
        names = table.index[table.carrier == carrier]
        if name_filter is not None:
            names = pd.Index([name for name in names if name_filter(name, carrier)])
        if names.empty:
            continue
        asset_temp = _temperature_for_assets(temperatures, _component_bus(n, component, names))
        outage = outage_fraction(outage_curves, prime_mover, asset_temp)
        _multiply_p_max_pu(n, component, names, 1.0 - outage)
        logger.info(
            "Applied dynamic %s derate to %d %s %s assets (mean availability %.4f).",
            prime_mover,
            len(names),
            carrier,
            component,
            float((1.0 - outage).to_numpy().mean()),
        )


def apply_temperature_derates(
    n: pypsa.Network,
    temperature_path: str,
    outage_temperature_path: str,
) -> None:
    """Apply temperature derates to thermal generators, PHS, and TES dischargers."""
    temperatures = read_region_temperatures(temperature_path, n.snapshots)
    curves = load_forced_outage_temperature_curves(outage_temperature_path)

    apply_dynamic_component_derates(n, "Generator", DYNAMIC_PRIME_MOVER_BY_CARRIER, temperatures, curves)
    apply_dynamic_component_derates(n, "StorageUnit", STORAGE_PRIME_MOVER_BY_CARRIER, temperatures, curves)
    # Only the discharging half of a TES pair produces power, so only it derates.
    apply_dynamic_component_derates(
        n,
        "Link",
        LINK_PRIME_MOVER_BY_CARRIER,
        temperatures,
        curves,
        name_filter=lambda name, carrier: carrier != "tes" or str(name).endswith(" tes discharger"),
    )
