# By PyPSA-USA Authors
"""
Regional overnight-CAPEX multipliers from NREL ReEDS ``reg_cap_cost_diff``.

The ReEDS file ``reg_cap_cost_diff_default.csv`` is indexed by county
(``r`` = ``p`` + 5-digit FIPS) with ``|``-delimited technology-group columns whose
values are *relative* cost differences centred on 0 (ReEDS applies them as
``1 + x``). They scale the **overnight** capital cost only -- never FOM/VOM -- and
are dimensionless, so no currency-year conversion is needed.

The group -> pypsa-carrier mapping is supplied by the caller (config
``cost_multipliers.column_to_tech``); this module never hard-codes it.

Because this fork's buses are finer than county resolution (~10k clustered buses
vs ~3.1k counties) and ``county`` is dropped during ``simplify_network``, buses are
mapped back onto counties geographically: each bus takes the county whose polygon
contains it, falling back to the nearest county when it lies outside every
eligible polygon (offshore buses, or counties absent from the ReEDS table).

Wind and solar are intentionally *not* driven by this table -- ReEDS publishes no
wind/UPV column. Their regional cost comes from the per-site interconnection cost
``cost_trans_usd_per_mw`` carried in the renewable profile NetCDF, applied via
``renewable_interconnection_capital_cost``.

Multipliers are applied where each component is created (``add_electricity`` for
wind/solar, ``add_extra_components`` for everything else), folded directly into the
cost expression so FOM is never touched.

Transmission line costs are the other regional dimension handled here, from a
different pair of ReEDS tables: ``transmission_distance_cost_500kVac_county.csv``
gives the region- and route-resolved USD/MW-km for AC corridors, and
``rev_transmission_basecost.csv`` gives the voltage-class and DC/AC relationships.
See the section header above :func:`load_transmission_basecost` for how the two
combine, and ``add_electricity.update_transmission_costs`` for where they are applied.

``SectorCosts`` also lives here: it is the reader for the electrolyzer/TES rows in
``simple_sector_costs.csv``, and its ``annualized`` is the entry point through which
those two carriers receive the regional multiplier.
"""

import logging

import numpy as np
import pandas as pd
from _helpers import calculate_annuity, get_currency_conversion_factor

logger = logging.getLogger(__name__)

# Equal-area CONUS projection: nearest-neighbour distances must be metric.
EQUAL_AREA_CRS = 5070


def load_reg_cap_cost_diff(path: str) -> pd.DataFrame:
    """Load the ReEDS relative-cost-difference file, indexed by county code ``r``."""
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "r"}).set_index("r")
    return df.apply(pd.to_numeric, errors="coerce")


def county_multiplier_table(
    reg_cap_diff: pd.DataFrame,
    column_to_tech: dict[str, list[str]],
) -> pd.DataFrame:
    """Expand ``|``-delimited group columns to a (county x pypsa-carrier) multiplier.

    ``column_to_tech`` (from config ``cost_multipliers.column_to_tech``) maps each
    ReEDS group column to the pypsa carriers it applies to. Returns ``1 + x`` per
    (county, carrier); carriers absent from the mapping are simply not produced
    (callers default them to 1.0).
    """
    if not column_to_tech:
        raise ValueError(
            "column_to_tech is required (config cost_multipliers.column_to_tech); "
            "no built-in default is provided.",
        )
    out = {}
    for column, carriers in column_to_tech.items():
        if column not in reg_cap_diff.columns:
            logger.warning("reg_cap_cost_diff column %r not found; skipping.", column)
            continue
        mult = 1.0 + reg_cap_diff[column]
        for carrier in carriers:
            out[carrier] = mult
    table = pd.DataFrame(out)
    table.index.name = "r"
    return table


def assign_bus_counties(
    buses: pd.DataFrame,
    county_shapes_path: str,
    valid_counties=None,
) -> pd.Series:
    """Map each bus to a county code (``p`` + FIPS) by geographic proximity.

    A bus inside a county polygon gets that county (nearest-join distance 0), which
    is equivalent to a point-in-polygon test. A bus outside every eligible polygon
    -- offshore, or inside a county the ReEDS table does not cover -- falls back to
    the nearest eligible county.

    ``valid_counties`` restricts the candidate polygons (pass the ReEDS table index)
    so that every bus resolves to a county that actually carries a multiplier.
    """
    # Imported lazily: callers that only need SectorCosts should not pay for it.
    import geopandas as gpd

    counties = gpd.read_file(county_shapes_path)
    geoid = counties["GEOID"].astype(str)
    # build_shapes.py already p-prefixes its county_shapes.geojson; the raw Census
    # shapefile does not. Accept either.
    counties["r"] = geoid.where(geoid.str.startswith("p"), "p" + geoid)

    if valid_counties is not None:
        eligible = counties["r"].isin(set(valid_counties))
        dropped = int((~eligible).sum())
        if dropped:
            logger.info(
                "Excluding %d county polygon(s) absent from the ReEDS cost table; "
                "buses there snap to the nearest covered county.",
                dropped,
            )
        counties = counties[eligible]
    if counties.empty:
        raise ValueError("No county polygons remain after filtering to the ReEDS cost table.")

    points = gpd.GeoDataFrame(
        index=buses.index,
        geometry=gpd.points_from_xy(buses["x"].astype(float), buses["y"].astype(float)),
        crs="EPSG:4326",
    ).to_crs(EQUAL_AREA_CRS)

    joined = gpd.sjoin_nearest(
        points,
        counties[["r", "geometry"]].to_crs(EQUAL_AREA_CRS),
        how="left",
    )
    # sjoin_nearest emits one row per tied polygon; collapse deterministically.
    return joined.groupby(level=0)["r"].first().reindex(buses.index)


def bus_multiplier_table(
    buses: pd.DataFrame,
    county_table: pd.DataFrame,
    county_shapes_path: str,
) -> pd.DataFrame:
    """Build a (bus x pypsa-carrier) multiplier table from the county table.

    Buses that resolve to no county default to 1.0 so downstream cost expressions
    reduce to their unmodified form.
    """
    bus_county = assign_bus_counties(
        buses,
        county_shapes_path,
        valid_counties=county_table.index,
    )
    unresolved = int(bus_county.isna().sum())
    if unresolved:
        logger.warning("%d bus(es) could not be matched to a county; using multiplier 1.0.", unresolved)

    table = county_table.reindex(bus_county.to_numpy())
    table.index = buses.index
    table = table.astype(float).fillna(1.0)
    for carrier in table.columns:
        col = table[carrier]
        logger.info(
            "Regional overnight-capex multiplier for %r: min=%.3f mean=%.3f max=%.3f over %d bus(es).",
            carrier,
            col.min(),
            col.mean(),
            col.max(),
            len(col),
        )
    return table


def carrier_multiplier(bus_multipliers, carrier: str, buses_i):
    """Per-bus multiplier array for ``carrier``, aligned to ``buses_i``.

    Returns the scalar ``1.0`` when the feature is disabled or the carrier is not
    mapped, so every caller's cost expression degrades to its original form.
    """
    if bus_multipliers is None or carrier not in bus_multipliers.columns:
        return 1.0
    return bus_multipliers[carrier].reindex(buses_i).fillna(1.0).to_numpy()


def overnight_delta_capital_cost(
    annualized_capex_fom,
    capex_overnight_per_kw,
    annuity_factor,
    multiplier,
):
    """Capital cost with the overnight portion scaled by ``multiplier`` only.

    ``capital_cost = annualized_capex_fom + annuity * (m - 1) * capex_overnight_per_kw * 1e3``

    Leaves FOM and the non-overnight capex (grid connection, construction finance)
    bundled in ``annualized_capex_fom`` untouched. Units: ``capex_overnight_per_kw``
    in USD/kW -> *1e3 to USD/MW, matching ``annualized_capex_fom``.
    """
    return annualized_capex_fom + annuity_factor * (multiplier - 1.0) * capex_overnight_per_kw * 1e3


def renewable_interconnection_capital_cost(
    capex_overnight_per_kw,
    capex_construction_finance_factor,
    opex_fixed_per_kw,
    cost_trans_usd_per_mw,
    annuity_factor,
):
    """Wind/solar capital cost with ATB grid connection replaced by the ReEDS cost.

    ``capital_cost = annuity * ((capex_overnight + construction_finance) * 1e3
    + cost_trans_usd_per_mw) + FOM * 1e3``

    ``cost_trans_usd_per_mw`` (ReEDS ``cost_total_trans_usd_per_mw``, already USD/MW,
    2022 USD) is a capital cost, so it is annuitized. ATB's generic
    ``capex_grid_connection_per_kw`` is intentionally excluded to avoid double
    counting interconnection costs. FOM is already annual and is not annuitized.
    """
    base_capex_per_mw = (capex_overnight_per_kw + capex_construction_finance_factor) * 1e3
    return annuity_factor * (base_capex_per_mw + cost_trans_usd_per_mw) + opex_fixed_per_kw * 1e3


# ---------------------------------------------------------------------------
# Transmission line costs: voltage level and route region
# ---------------------------------------------------------------------------
#
# Two previously unused ReEDS tables supply the two dimensions that a single
# national USD/MW-km cannot express:
#
# * ``transmission_distance_cost_500kVac_county.csv`` -- 500 kV AC cost for
#   ~799k county pairs. Its ``length_miles`` is the great-circle distance between
#   county centroids (median ratio 0.998 against recomputed centroid distances),
#   so ReEDS's terrain and route-detour costs live entirely in the *price*, not in
#   the length. Dividing gives a USD/MW **per great-circle km** field that already
#   carries regional base cost, terrain and detour.
# * ``rev_transmission_basecost.csv`` -- USD/MW-mile by voltage class and by four
#   ReEDS cost regions. Taking each voltage's ratio to that region's own 500 kV AC
#   entry first, then the median across regions, isolates the voltage relationship
#   from the regional level -- the level is already carried by the county table
#   above, so re-applying a regional factor would double count it. Those ratios are
#   then reduced to a single power-law exponent in voltage (see
#   :func:`voltage_cost_exponent`), so every voltage class the network carries is
#   priced by the same law rather than by segment.
#
# Both are in 2004 USD; callers convert with ``get_currency_conversion_factor``.

MILES_TO_KM = 1.609344

#: Currency year of both ReEDS transmission tables.
REEDS_TRANSMISSION_COST_YEAR = 2004

#: The county cost field for one county is the median over its ``K`` shortest
#: incident county pairs, not over all of them. A county carries a median of ~514
#: pairs reaching out to ~700 km, and the per-km cost falls with pair length
#: (median 1955 USD2004/MW/km below 25 km against 1298 at 300-450 km), so a
#: whole-county average both blurs the local corridor and biases short lines low.
#: Leave-one-out error against the pair values on pairs under 60 km -- the scale
#: real lines live at -- drops from 5.5% MAPE / 0.874 R2 for all incident pairs to
#: 2.7% / 0.954 at K=8, which is also about the number of neighbours a US county
#: actually has.
NEAR_NEIGHBOUR_PAIRS = 8

#: Cost-region columns of ``rev_transmission_basecost.csv``.
REEDS_BASECOST_REGIONS = ("TEPPC", "SCE", "MISO", "Southeast")

#: Voltage the county cost table is quoted at, and the denominator of the
#: voltage ratios.
REFERENCE_VOLTAGE_KV = 500.0


def load_transmission_basecost(path: str) -> pd.DataFrame:
    """Load ``rev_transmission_basecost.csv``, indexed by voltage-class label.

    The file carries a units row under the header (``kV``, ``MW``,
    ``USD2004perMWmile`` ...) which is dropped, and is written with a BOM.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["Voltage"] != "kV"].set_index("Voltage")
    numeric = ["Capacity", *REEDS_BASECOST_REGIONS]
    missing = set(numeric).difference(df.columns)
    if missing:
        raise ValueError(f"rev_transmission_basecost is missing columns {sorted(missing)}.")
    return df[numeric].apply(pd.to_numeric)


def voltage_cost_anchors(basecost: pd.DataFrame) -> pd.Series:
    """Per-voltage cost ratios relative to 500 kV AC, indexed by kV, ascending.

    Ratios are formed *inside* each cost region first, then the median is taken
    across regions. The four regions disagree substantially on the absolute
    voltage relationship (69 kV/500 kV runs 2.97 in Southeast to 6.62 in MISO), so
    the median is the estimator that does not let one study dominate.
    """
    ac_rows = basecost.index[basecost.index.str.contains("AC")]
    if ac_rows.empty:
        raise ValueError("rev_transmission_basecost has no AC voltage rows.")
    ac = basecost.loc[ac_rows, list(REEDS_BASECOST_REGIONS)]
    ac.index = ac_rows.str.extract(r"(\d+)", expand=False).astype(float)

    if REFERENCE_VOLTAGE_KV not in ac.index:
        raise ValueError(f"rev_transmission_basecost has no {REFERENCE_VOLTAGE_KV:.0f} kV AC row.")
    ratios = ac.div(ac.loc[REFERENCE_VOLTAGE_KV])
    anchors = ratios.median(axis=1).sort_index()
    anchors.index.name = "v_nom"
    logger.info(
        "Voltage cost anchors relative to %.0f kV AC (median of %s): %s",
        REFERENCE_VOLTAGE_KV,
        ", ".join(REEDS_BASECOST_REGIONS),
        ", ".join(f"{kv:.0f}kV={ratio:.4f}" for kv, ratio in anchors.items()),
    )
    return anchors


def dc_ac_line_cost_ratio(basecost: pd.DataFrame) -> float:
    """500 kV DC bipole line cost as a fraction of 500 kV AC single-circuit.

    This is a single number rather than a per-region one: all four cost regions
    give exactly 0.413686, and the ReEDS balancing-area distance-cost files carry
    the same ratio on every one of their 17,822 pairs. Asserting the agreement
    keeps a future data refresh from silently collapsing a per-region spread into
    one number.
    """
    ac_label = f"{REFERENCE_VOLTAGE_KV:.0f}ACsingle"
    dc_label = f"{REFERENCE_VOLTAGE_KV:.0f}DCbipole"
    for label in (ac_label, dc_label):
        if label not in basecost.index:
            raise ValueError(f"rev_transmission_basecost has no {label!r} row.")

    per_region = basecost.loc[dc_label, list(REEDS_BASECOST_REGIONS)] / basecost.loc[
        ac_label,
        list(REEDS_BASECOST_REGIONS),
    ]
    if not np.allclose(per_region, per_region.iloc[0], rtol=1e-4):
        raise ValueError(
            "DC/AC line cost ratio differs across ReEDS cost regions "
            f"({per_region.round(6).to_dict()}); the county cost table only resolves AC, "
            "so a per-region DC ratio would need its own regional mapping.",
        )
    ratio = float(per_region.mean())
    logger.info("DC/AC line cost ratio at %.0f kV: %.6f", REFERENCE_VOLTAGE_KV, ratio)
    return ratio


def voltage_cost_exponent(anchors: pd.Series) -> float:
    """Fit ``ratio = (kV / 500)^b`` to :func:`voltage_cost_anchors`, returning ``b``.

    Cost per MW-km falls with voltage as a power law: one exponent for the whole
    range, monotone and smooth everywhere. The fit is a log-log least squares
    **constrained through the reference voltage**, so ``ratio(500 kV) = 1`` exactly.
    That constraint is not cosmetic -- the county cost field is quoted at 500 kV AC,
    so a free intercept (which fits the anchors slightly better, b = -0.780) would
    put ``ratio(500 kV) = 1.063`` and silently inflate every line in the network by
    6.3%.

    The four anchors are not exactly collinear in log-log: their consecutive slopes
    are -0.647, -0.696 and -0.962, i.e. the relationship steepens with voltage. A
    single exponent therefore trades per-anchor exactness for a globally consistent
    law -- worst residual -10.4% at 230 kV, and -0.9% on the length-weighted network
    total. In exchange the curve has no slope discontinuity at the ends of the anchor
    range, which the previous piecewise form did: it extrapolated past 500 kV on the
    global slope -0.780 while the last measured segment ran at -0.962.
    """
    log_kv = np.log(anchors.index.to_numpy(dtype=float) / REFERENCE_VOLTAGE_KV)
    log_ratio = np.log(anchors.to_numpy(dtype=float))
    exponent = float((log_kv * log_ratio).sum() / (log_kv * log_kv).sum())

    fitted = np.exp(exponent * log_kv)
    residuals = fitted / anchors.to_numpy(dtype=float) - 1.0
    logger.info(
        "Voltage cost exponent b = %.4f in ratio = (kV/%.0f)^b; anchor residuals %s.",
        exponent,
        REFERENCE_VOLTAGE_KV,
        ", ".join(f"{kv:.0f}kV={res:+.1%}" for kv, res in zip(anchors.index, residuals)),
    )
    return exponent


def voltage_ratio(v_nom, exponent: float):
    """Cost ratio relative to 500 kV AC at arbitrary nominal voltages.

    ``ratio = (v_nom / 500)^exponent`` with ``exponent`` from
    :func:`voltage_cost_exponent`. The synthetic TAMU network carries
    100/115/161/345/765 kV classes that ReEDS does not publish -- 2/5 of all lines --
    and the power law covers them on the same footing as the measured ones, with no
    interpolate-vs-extrapolate seam.

    Voltages that are missing fall back to the reference voltage (ratio 1.0).
    """
    kv = np.asarray(v_nom, dtype=float)

    unresolved = ~np.isfinite(kv) | (kv <= 0)
    if unresolved.any():
        logger.warning(
            "%d branch(es) have no usable v_nom; pricing them at %.0f kV.",
            int(unresolved.sum()),
            REFERENCE_VOLTAGE_KV,
        )
        kv = np.where(unresolved, REFERENCE_VOLTAGE_KV, kv)

    return (kv / REFERENCE_VOLTAGE_KV) ** exponent


def load_transmission_pair_costs(path: str) -> pd.DataFrame:
    """Load the ReEDS county-pair 500 kV AC costs as a symmetric long table.

    Returns columns ``r``, ``rr``, ``length_km``, ``unit_cost`` where
    ``unit_cost`` is 2004 USD/MW per great-circle km. Both orderings of every pair
    are present so callers can look up and group without re-symmetrising.
    """
    df = pd.read_csv(path)
    required = {"r", "rr", "length_miles", "USD2004perMW"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Transmission distance-cost file is missing columns {sorted(missing)}.")

    df["length_km"] = pd.to_numeric(df["length_miles"], errors="coerce") * MILES_TO_KM
    df["unit_cost"] = pd.to_numeric(df["USD2004perMW"], errors="coerce") / df["length_km"]
    df = df[["r", "rr", "length_km", "unit_cost"]].replace([np.inf, -np.inf], np.nan).dropna()

    flipped = df.rename(columns={"r": "rr", "rr": "r"})
    both = pd.concat([df, flipped[df.columns]], ignore_index=True)
    logger.info(
        "Loaded %d county-pair transmission costs over %d counties.",
        len(df),
        both["r"].nunique(),
    )
    return both


def pair_unit_costs(pair_table: pd.DataFrame) -> pd.Series:
    """``(r, rr) -> unit_cost`` lookup over both orderings of each pair."""
    series = pair_table.set_index(["r", "rr"])["unit_cost"]
    return series[~series.index.duplicated()]


def county_unit_cost_field(
    pair_table: pd.DataFrame,
    n_nearest: int = NEAR_NEIGHBOUR_PAIRS,
) -> pd.Series:
    """Per-county unit cost: median over the ``n_nearest`` shortest incident pairs.

    Counties with fewer incident pairs than that simply use all of them.
    """
    nearest = pair_table.sort_values(["r", "length_km"]).groupby("r", sort=False).head(n_nearest)
    field = nearest.groupby("r")["unit_cost"].median()
    logger.info(
        "County unit-cost field over %d counties from the %d nearest pairs each: "
        "min=%.0f median=%.0f max=%.0f USD%d/MW/km at %.0f kV AC.",
        len(field),
        n_nearest,
        field.min(),
        field.median(),
        field.max(),
        REEDS_TRANSMISSION_COST_YEAR,
        REFERENCE_VOLTAGE_KV,
    )
    return field


def _state_fips(counties: pd.Series) -> pd.Series:
    """State FIPS from a ReEDS county code (``p`` + 5-digit FIPS)."""
    return counties.astype("string").str.slice(1, 3)


def transmission_unit_costs(
    county0: pd.Series,
    county1: pd.Series,
    pair_costs: pd.Series,
    field: pd.Series,
) -> pd.Series:
    """Per-branch 500 kV AC unit cost in 2004 USD/MW per great-circle km.

    Resolved in four steps, each catching what the previous one cannot:

    1. the exact ``(county0, county1)`` pair, which carries the measured corridor;
    2. the mean of the two counties' :func:`county_unit_cost_field` values -- the
       main path, because 77.6% of TAMU lines have both ends in one county and
       ReEDS publishes no same-county pair;
    3. the median field value across the counties of the same state, which covers
       Connecticut (ReEDS uses its post-2022 planning regions ``p09110``-``p09190``
       while the census county shapes here use the legacy FIPS ``p09001``-``p09015``);
    4. the national median, for buses with no county at all (offshore).
    """
    index = county0.index
    c0 = county0.astype("string")
    c1 = county1.astype("string")

    unit = pd.Series(np.nan, index=index, dtype=float)

    # 1. exact county pair
    lookup = pd.MultiIndex.from_arrays([c0.fillna(""), c1.fillna("")])
    unit = pd.Series(pair_costs.reindex(lookup).to_numpy(), index=index, dtype=float)
    from_pair = int(unit.notna().sum())

    # 2. mean of the two endpoint county fields
    endpoint_mean = pd.concat(
        [
            pd.Series(field.reindex(c0.to_numpy()).to_numpy(), index=index, dtype=float),
            pd.Series(field.reindex(c1.to_numpy()).to_numpy(), index=index, dtype=float),
        ],
        axis=1,
    ).mean(axis=1)
    unit = unit.fillna(endpoint_mean)
    from_field = int(unit.notna().sum()) - from_pair

    # 3. state median of the field
    state_field = field.groupby(_state_fips(pd.Series(field.index, index=field.index))).median()
    branch_state = _state_fips(c0).fillna(_state_fips(c1))
    unit = unit.fillna(pd.Series(state_field.reindex(branch_state.to_numpy()).to_numpy(), index=index, dtype=float))
    from_state = int(unit.notna().sum()) - from_pair - from_field

    # 4. national median
    unit = unit.fillna(float(field.median()))
    from_national = int(unit.notna().sum()) - from_pair - from_field - from_state

    logger.info(
        "Transmission unit costs resolved for %d branch(es): %d from the county pair, "
        "%d from the endpoint county field, %d from the state median, %d from the national median.",
        len(unit),
        from_pair,
        from_field,
        from_state,
        from_national,
    )
    return unit


class SectorCosts:
    """Read the reference ``simple_sector_costs.csv`` and annualize capex."""

    def __init__(self, path: str) -> None:
        costs = pd.read_csv(path)
        required = {"pypsa-name", "parameter", "value", "unit", "currency_year"}
        missing = required.difference(costs.columns)
        if missing:
            raise ValueError(f"Missing sector cost columns: {sorted(missing)}")
        costs["value"] = pd.to_numeric(costs.value, errors="coerce")
        costs["currency_year"] = pd.to_numeric(costs.currency_year, errors="coerce")
        self.costs = costs.set_index(["pypsa-name", "parameter"]).sort_index()

    def row(self, technology: str, parameter: str) -> pd.Series:
        try:
            return self.costs.loc[(technology, parameter)]
        except KeyError as exc:
            raise KeyError(f"Missing sector cost for {technology!r}, {parameter!r}.") from exc

    def value(self, technology: str, parameter: str, default=None) -> float:
        try:
            value = self.row(technology, parameter).value
        except KeyError:
            if default is None:
                raise
            return float(default)
        if pd.isna(value) and default is not None:
            return float(default)
        return float(value)

    def wacc_real(self, technology: str) -> float:
        return self.value(technology, "wacc_real")

    def _currency_factor(self, technology: str) -> float:
        row = self.row(technology, "investment")
        unit = str(row.unit)
        currency = "EUR" if unit.startswith("EUR/") else "USD"
        if not unit.startswith(("EUR/", "USD/")) or pd.isna(row.currency_year):
            raise ValueError(f"Unsupported investment currency/unit for {technology}: {unit}")
        return get_currency_conversion_factor(int(row.currency_year), currency)

    def _denominator_factor(self, technology: str) -> float:
        denominator = str(self.row(technology, "investment").unit).split("/", 1)[1]
        if "tCO2" in denominator:
            return 1.0
        base = denominator.split("/", 1)[0]
        if base.startswith("kW"):
            return 1e3
        if base.startswith("MW"):
            return 1.0
        raise ValueError(f"Unsupported investment basis for {technology}: {denominator}")

    def annualized(self, technology: str, overnight_multiplier=1.0):
        """Annualized capex.

        ``overnight_multiplier`` scales the annuity term only, leaving the FOM share
        untouched -- this is how the regional ReEDS overnight-capex multiplier enters
        (see ``regional_cost.py``). It accepts a scalar or a per-bus array; the
        default 1.0 reproduces the unmodified cost exactly.
        """
        return (
            (
                calculate_annuity(
                    self.value(technology, "crp"),
                    self.wacc_real(technology),
                )
                * overnight_multiplier
                + self.value(technology, "FOM", 0.0) / 100
            )
            * self.value(technology, "investment")
            * self._currency_factor(technology)
            * self._denominator_factor(technology)
        )

    def lifetime(self, technology: str) -> float:
        lifetime = self.value(technology, "lifetime", np.inf)
        return np.inf if lifetime == 0 else lifetime
