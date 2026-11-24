"""
Prepare PyPSA network for solving according to :ref:`opts` and :ref:`ll`, such
as.

- adding an annual **limit** of carbon-dioxide emissions,
- adding an exogenous **price** per tonne emissions of carbon-dioxide (or other kinds),
- setting an **N-1 security margin** factor for transmission line capacities,
- specifying an expansion limit on the **cost** of transmission expansion,
- specifying an expansion limit on the **volume** of transmission expansion, and
- reducing the **temporal** resolution by averaging over multiple hours
  or segmenting time series into chunks of varying lengths using ``tsam``.
"""

import logging
import re

import numpy as np
import pandas as pd
import pypsa
from _helpers import (
    configure_logging,
    is_transport_model,
    load_costs,
    log_network_schema,
    set_scenario_config,
    update_config_from_wildcards,
)
from constants import HOURS_PER_YEAR
from opts.bidirectional_link import bidirectional_pairs

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def get_social_discount(t, r=0.01):
    """
    Calculate for a given time t and social discount rate r [per unit] the
    social discount.
    """
    return 1 / (1 + r) ** t


def get_investment_weighting(time_weighting, r=0.01):
    """
    Define cost weighting.

    Returns cost weightings depending on the the time_weighting
    (pd.Series) and the social discountrate r
    """
    end = time_weighting.cumsum()
    start = time_weighting.cumsum().shift().fillna(0)
    return pd.concat([start, end], axis=1).apply(
        lambda x: sum(get_social_discount(t, r) for t in range(int(x.iloc[0]), int(x.iloc[1]))),
        axis=1,
    )


def add_co2limit(n, co2limit, num_years=1.0):
    n.add(
        "GlobalConstraint",
        "CO2Limit",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=co2limit * num_years,
    )


def add_gaslimit(n, gaslimit, num_years=1.0):
    sel = n.carriers.index.intersection(["OCGT", "CCGT", "CHP"])
    n.carriers.loc[sel, "gas_usage"] = 1.0

    n.add(
        "GlobalConstraint",
        "GasLimit",
        carrier_attribute="gas_usage",
        sense="<=",
        constant=gaslimit * num_years,
    )


def add_emission_prices(n, emission_prices={"co2": 0.0}, exclude_co2=False):
    if exclude_co2:
        emission_prices.pop("co2")
    ep = (pd.Series(emission_prices).rename(lambda x: x + "_emissions") * n.carriers.filter(like="_emissions")).sum(
        axis=1,
    )
    gen_ep = n.generators.carrier.map(ep) / n.generators.efficiency
    n.generators["marginal_cost"] += gen_ep
    n.generators_t["marginal_cost"] += gen_ep[n.generators_t["marginal_cost"].columns]
    su_ep = n.storage_units.carrier.map(ep) / n.storage_units.efficiency_dispatch
    n.storage_units["marginal_cost"] += su_ep


def set_line_s_max_pu(n, transport_model, n1_derate=0.7):
    """Set the static line ``s_max_pu`` to the N-1 derate (``lines.N1_derate``)."""
    if not transport_model:
        logger.info(f"N-1 security margin of lines set to {n1_derate}")
        n.lines["s_max_pu"] = n1_derate


def apply_dynamic_line_rating(n, fn):
    """
    Scale branch ratings by hourly relative line ratings read from ``fn``.

    ``fn`` (.parquet or .csv) holds one column per model Line or AC Link and one row
    per hour of the weather year (365-day calendar), as written by rule
    ``build_dynamic_line_rating`` from the dynamic (``dlr``) or static (``slr``)
    ratings. Snapshots are matched by relabelling each timestep to the weather year.
    The rating multiplies the static rating independently of the N-1 derate: lines
    get ``s_max_pu_t = N1_derate * rating`` (``s_max_pu`` as ``set_line_s_max_pu``
    left it), AC links ``p_max_pu_t = p_max_pu * rating``.
    """
    rating = pd.read_parquet(fn) if str(fn).endswith(".parquet") else pd.read_csv(fn, index_col=0)
    rating.index = pd.DatetimeIndex(rating.index)
    rating.columns = rating.columns.astype(str)

    weather_year = rating.index[0].year
    timesteps = n.snapshots.get_level_values("timestep")
    rating = rating.reindex(pd.DatetimeIndex([t.replace(year=weather_year) for t in timesteps]))
    if rating.isna().any().any():
        raise ValueError(f"{fn} does not cover every snapshot of weather year {weather_year}.")
    rating.index = n.snapshots

    applied = 0
    for static, dynamic, attr, names in [
        (n.lines, n.lines_t, "s_max_pu", n.lines.index),
        (n.links, n.links_t, "p_max_pu", n.links.index[n.links.carrier == "AC"]),
    ]:
        cols = rating.columns.intersection(names.astype(str))
        if cols.empty:
            continue
        if len(cols) < len(names):
            logger.warning(f"{len(names) - len(cols)} of {len(names)} branches have no dynamic rating in {fn}.")
        scaled = rating[cols].mul(static.loc[cols, attr], axis=1)
        dynamic[attr] = pd.concat([dynamic[attr].drop(columns=cols, errors="ignore"), scaled], axis=1)
        applied += len(cols)
    if not applied:
        raise ValueError(f"No column of {fn} names a Line or AC Link of this network.")
    logger.info(
        f"Line rating from {fn} (weather year {weather_year}) applied to {applied} branches; "
        f"mean relative rating {rating.to_numpy().mean():.3f}."
    )


LL_PERIOD_PATTERN = re.compile(r"(\d{4})([vc])([0-9.]+|opt)")


def parse_ll_by_period(ll: str) -> dict[int, tuple[str, str]] | None:
    """
    Read a per-period ll wildcard such as ``2040v1.1+2050v1.2``.

    Returns ``{period: (ll_type, factor)}``, or None for a single limit such as ``v1.1``.
    """
    if not ll[:1].isdigit():
        return None
    limits = {}
    for part in ll.split("+"):
        match = LL_PERIOD_PATTERN.fullmatch(part)
        if match is None:
            raise ValueError(f"Cannot read per-period transmission limit {part!r} in ll wildcard {ll!r}.")
        limits[int(match[1])] = (match[2], match[3])
    return limits


def halve_paired_link_length(n):
    """
    Give each link of a ``_fwd``/``_rev`` pair of AC/DC links half the corridor's length.

    A pair is one corridor (an HVDC line, or an interface of the transport model) with a
    link per direction; both links carry the corridor's length and expand together (the
    bidirectional link constraint). Their capital cost is already split in half, so the
    cost (``c``) limit counts the corridor once; halving the length does the same for the
    volume (``v``) limit and its reference. Call after the link costs and efficiencies,
    which use the full length, are set, and before the transmission limits.
    """
    if n.links.empty:
        return
    pairs = bidirectional_pairs(n.links.index[n.links.carrier.isin(["AC", "DC"])])
    paired = pd.Index([*pairs, *pairs.values()])
    if paired.empty:
        return
    n.links.loc[paired, "length"] /= 2
    logger.info(f"Halved the length of {len(paired)} links in {len(pairs)} '_fwd'/'_rev' pairs.")


def _transmission_reference(n, ll_type):
    """Today's length- (``v``) or cost- (``c``) weighted capacity of AC lines and AC/DC links."""
    col = "capital_cost" if ll_type == "c" else "length"
    links = n.links[n.links.carrier.isin(["AC", "DC"])] if not n.links.empty else n.links
    return n.lines.s_nom @ n.lines[col] + links.p_nom @ links[col]


def _make_transmission_extendable(n):
    """Let AC lines and AC/DC links expand from today's capacity; all are non-extendable before this."""
    n.lines["s_nom_min"] = n.lines.s_nom
    n.lines["s_nom_extendable"] = True
    if n.links.empty:
        return
    links = n.links.carrier.isin(["AC", "DC"])
    n.links.loc[links, "p_nom_min"] = n.links.loc[links, "p_nom"]
    n.links.loc[links, "p_nom_extendable"] = True


def apply_transmission_limit(n, ll: str):
    """
    Set the transmission limits of the ll wildcard.

    With several investment periods every ll, per-period or single, gets a limit per
    period, so that lines expand period by period; a single ll applies to all of them.
    """
    limits = parse_ll_by_period(ll)
    if limits is None and len(n.investment_periods) > 1:
        limits = {int(n.investment_periods[0]): (ll[0], ll[1:])}
    if limits is None:
        return set_transmission_limit(n, ll[0], ll[1:])
    return set_transmission_limit_by_period(n, limits)


def set_transmission_limit_by_period(n, limits):
    """
    Set one transmission limit per investment period.

    Used for a per-period ll wildcard, and for a single ll (e.g. ``v1.1``) on a network
    with several investment periods, which then applies to every period. Each limit caps the length- or cost-weighted capacity installed by that period,
    relative to today's. A period without its own entry takes the latest earlier one;
    periods before the first entry get no expansion. The limits are stored as
    GlobalConstraints of type ``transmission_*_limit_per_period`` (PyPSA ignores
    these types) and built in solve_network, which gives every line a capacity per
    period so that a limit binds only on what is built by its period.
    """
    periods = list(n.investment_periods)
    unknown = sorted(set(limits) - set(periods))
    if unknown:
        raise ValueError(f"ll names years {unknown} that are not planning horizons {periods}.")

    first = min(limits)
    limit = (limits[first][0], "1.0")
    by_period = {}
    for period in periods:
        limit = limits.get(period, limit)
        by_period[period] = limit
    logger.info(f"Setting transmission limits per period: {by_period}")

    if any(factor == "opt" or float(factor) > 1.0 for _, factor in by_period.values()):
        _make_transmission_extendable(n)
    else:
        logger.info("No period allows transmission expansion; lines and links stay fixed.")
        return n

    # an `opt` period gets an infinite limit, so solve_network still builds its capacity
    for period, (ll_type, factor) in by_period.items():
        con_type = "expansion_cost" if ll_type == "c" else "volume_expansion"
        constant = np.inf if factor == "opt" else round(float(factor) * _transmission_reference(n, ll_type), 2)
        n.add(
            "GlobalConstraint",
            f"l{ll_type}_limit_{period}",
            type=f"transmission_{con_type}_limit_per_period",
            investment_period=period,
            sense="<=",
            constant=constant,
            carrier_attribute="AC, DC",
        )
    return n


def set_transmission_limit(n, ll_type, factor):
    """
    Set transmission limits according to ll wildcard.

    For transport models we track expandable AC links via their carrier
    initially, then re-name them to AC. We don't set expandability
    earlier in the model to avoid rebuilding the network multiple times
    from earlier stages when testing sensitivities to the transmission
    limits wildcards.
    """
    logger.info(f"Setting transmission limit for {ll_type} to {factor}")

    dc_links = n.links.carrier == "DC" if not n.links.empty else pd.Series()
    ac_links_existing = n.links.carrier == "AC" if not n.links.empty else pd.Series()

    lines_s_nom = n.lines.s_nom
    col = "capital_cost" if ll_type == "c" else "length"
    ref = (
        lines_s_nom @ n.lines[col]
        + n.links.loc[dc_links, "p_nom"] @ n.links.loc[dc_links, col]
        + n.links.loc[ac_links_existing, "p_nom"] @ n.links.loc[ac_links_existing, col]
    )

    if factor == "opt" or float(factor) > 1.0:
        # if opt allows expansion set respective lines/links to extendable
        # all links prior to this point have extendable set to false
        n.lines["s_nom_min"] = lines_s_nom
        n.lines["s_nom_extendable"] = True

        n.links.loc[dc_links, "p_nom_min"] = n.links.loc[dc_links, "p_nom"]
        n.links.loc[dc_links, "p_nom_extendable"] = True

        n.links.loc[ac_links_existing, "p_nom_min"] = n.links.loc[ac_links_existing, "p_nom"]
        n.links.loc[ac_links_existing, "p_nom_extendable"] = True
    if factor != "opt":
        con_type = "expansion_cost" if ll_type == "c" else "volume_expansion"
        rhs = round(float(factor) * ref, 2)
        n.add(
            "GlobalConstraint",
            f"l{ll_type}_limit",
            type=f"transmission_{con_type}_limit",
            sense="<=",
            constant=rhs,
            carrier_attribute="AC, DC",
        )

    return n


def average_every_nhours(n, offset):
    logger.info(f"Resampling the network to {offset}")
    m = n.copy(snapshots=[])

    def resample_multi_index(df, offset, func):
        sw = []
        for year in df.index.levels[0]:
            sw_ = df.loc[year].resample(offset).apply(func)
            sns = sw_.index
            sns = sns[~((sns.month == 2) & (sns.day == 29))]
            sw_ = sw_.loc[sns]
            sw.append(sw_)
        snapshot_weightings = pd.concat(sw)
        snapshot_weightings.index = pd.MultiIndex.from_arrays(
            [snapshot_weightings.index.year, snapshot_weightings.index],
            names=["period", "timestep"],
        )
        return snapshot_weightings

    snapshot_weightings = resample_multi_index(n.snapshot_weightings, offset, "sum")
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings
    m.investment_periods = n.investment_periods

    for c in n.components:
        pnl = getattr(m, c.list_name + "_t")
        for k, df in c.dynamic.items():
            if not df.empty:
                pnl[k] = resample_multi_index(df, offset, "mean")

    rescale_uc_attrs_to_resolution(m, offset)
    return m


def _hours_per_snapshot(offset: str) -> int:
    """Number of hours collapsed into one snapshot by an ``Nh`` resample offset."""
    match = re.fullmatch(r"(\d*)h", offset.strip().lower())
    if not match:
        raise ValueError(f"Cannot read an hourly step out of resample offset {offset!r}")
    return int(match.group(1) or 1)


def rescale_uc_attrs_to_resolution(n, offset: str) -> None:
    """
    Convert the unit-commitment attributes from hourly to snapshot units.

    ``build_powerplants`` writes ``min_up_time`` / ``min_down_time`` in HOURS and
    ``ramp_limit_up`` / ``ramp_limit_down`` per-unit per HOUR, because that is how
    WECC ADS reports them. PyPSA instead reads ``min_up_time`` / ``min_down_time``
    in SNAPSHOTS and the ramp limits per SNAPSHOT. At native hourly resolution the
    two coincide, but under an ``Nh`` option (California runs ``REM-3h``) they
    diverge by exactly N: a 6 h minimum up time would be enforced as 18 h, and a
    50 %/h ramp would be enforced as 50 % per 3 h.

    Rescaling is applied only when the network actually has committable
    generators, so runs with ``conventional.unit_commitment: false`` — where the
    UC attributes are PyPSA defaults and the ramp limits are the only affected
    column — keep their previous behavior exactly.
    """
    gens = n.generators
    if "committable" not in gens or not gens.committable.any():
        return

    nhours = _hours_per_snapshot(offset)
    if nhours == 1:
        return

    com = gens.index[gens.committable.fillna(False).astype(bool)]

    for col in ("min_up_time", "min_down_time"):
        if col not in gens:
            continue
        hours = pd.to_numeric(gens.loc[com, col], errors="coerce").fillna(0.0)
        # Round up: a 2 h minimum up time still blocks the whole 3 h snapshot.
        n.generators.loc[com, col] = np.ceil(hours / nhours).astype(int)

    for col in ("ramp_limit_up", "ramp_limit_down"):
        if col not in gens:
            continue
        per_hour = pd.to_numeric(gens.loc[com, col], errors="coerce")
        n.generators.loc[com, col] = (per_hour * nhours).clip(upper=1.0)

    logger.info(
        "Rescaled unit-commitment attributes of %d committable generators from hourly to %d-hour snapshots.",
        len(com),
        nhours,
    )


def is_leap_year(year: int) -> bool:
    """Check if a given year is a leap year."""
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return True
    else:
        return False


def apply_time_segmentation(n, segments, solver_name="cbc"):
    try:
        import tsam.timeseriesaggregation as tsam
    except ImportError:
        raise ModuleNotFoundError(
            "Optional dependency 'tsam' not found.Install via 'pip install tsam'",
        )

    # get all time-dependent data
    columns = pd.MultiIndex.from_tuples([], names=["component", "key", "asset"])
    raw = pd.DataFrame(index=n.snapshots, columns=columns)
    for c in n.components:
        for attr, pnl in c.dynamic.items():
            # exclude e_min_pu which is used for SOC of EVs in the morning
            if not pnl.empty and attr != "e_min_pu":
                df = pnl.copy()
                df.columns = pd.MultiIndex.from_product([[c.name], [attr], df.columns])
                raw = pd.concat([raw, df], axis=1)
    raw = raw.dropna(axis=1)
    sn_weightings = {}

    for year in raw.index.levels[0]:
        logger.info(f"Find representative snapshots for {year}.")
        raw_t = raw.loc[year]
        # normalise all time-dependent data
        annual_max = raw_t.max().replace(0, 1)
        raw_t = raw_t.div(annual_max, level=0)

        # hack to get around that TSAM will add leap days in
        if is_leap_year(year):
            raw_t.index = raw_t.index.map(lambda x: x.replace(year=year + 1))

        # get representative segments
        agg = tsam.TimeSeriesAggregation(
            raw_t,
            hoursPerPeriod=len(raw_t),
            noTypicalPeriods=1,
            noSegments=int(segments),
            segmentation=True,
            solver=solver_name,
        )
        segmented = agg.createTypicalPeriods()

        weightings = segmented.index.get_level_values("Segment Duration")
        offsets = np.insert(np.cumsum(weightings[:-1]), 0, 0)
        timesteps = [raw_t.index[0] + pd.Timedelta(f"{offset}h") for offset in offsets]

        snapshots = pd.DatetimeIndex(timesteps)

        if is_leap_year(year):
            snapshots = snapshots.map(lambda x: x.replace(year=year))

        sn_weightings[year] = pd.Series(
            weightings,
            index=snapshots,
            name="weightings",
            dtype="float64",
        )

    sn_weightings = pd.concat(sn_weightings)
    n.set_snapshots(sn_weightings.index)
    n.snapshot_weightings = n.snapshot_weightings.mul(sn_weightings, axis=0)

    return n


def set_line_nom_max(
    n,
    s_nom_max_set=np.inf,
    p_nom_max_set=np.inf,
    s_nom_max_ext=np.inf,
    p_nom_max_ext=np.inf,
):
    if np.isfinite(s_nom_max_ext) and s_nom_max_ext > 0:
        logger.info(f"Limiting line extensions to {s_nom_max_ext} MW")
        n.lines["s_nom_max"] = n.lines["s_nom"] + s_nom_max_ext

    if np.isfinite(p_nom_max_ext) and p_nom_max_ext > 0:
        logger.info(f"Limiting link extensions to {p_nom_max_ext} MW")
        hvdc = n.links.index[n.links.carrier == "DC"]
        n.links.loc[hvdc, "p_nom_max"] = n.links.loc[hvdc, "p_nom"] + p_nom_max_ext

    n.lines["s_nom_max"] = n.lines.s_nom_max.clip(upper=s_nom_max_set)
    n.links["p_nom_max"] = n.links.p_nom_max.clip(upper=p_nom_max_set)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_network",
            # simpl="",
            clusters="100",
            interconnect="western",
            ll="v1.0",
            opts="500SEG",
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)
    params = snakemake.params
    transport_model = is_transport_model(params.transmission_network)

    n = pypsa.Network(snakemake.input[0])
    schema_entry = log_network_schema(n, stage="entry")
    num_years = n.snapshot_weightings.loc[n.investment_periods[0]].objective.sum() / HOURS_PER_YEAR
    costs = load_costs(snakemake.input.tech_costs, params.costs)
    # Set Investment Period Year Weightings
    # 'fillna(1)' needed if only one period
    inv_per_time_weight = n.investment_periods.to_series().diff().shift(-1).ffill().fillna(1)
    n.investment_period_weightings["years"] = inv_per_time_weight
    # set Investment Period Objective weightings
    social_discountrate = params.costs["social_discount_rate"]
    objective_w = get_investment_weighting(
        n.investment_period_weightings["years"],
        social_discountrate,
    )
    n.investment_period_weightings["objective"] = objective_w

    set_line_s_max_pu(n, transport_model, params.lines["N1_derate"])

    # dlr (lines.dynamic_line_rating: true) or slr ratings, multiplied onto N1_derate;
    # None where no rating table exists (see line_rating_kind in build_electricity.smk)
    if params.line_rating:
        apply_dynamic_line_rating(n, snakemake.input.line_rating)

    # temporal averaging
    time_resolution = params.time_resolution
    is_string = isinstance(time_resolution, str)
    if is_string and time_resolution.lower().endswith("h"):
        # pandas 3 only accepts the lowercase 'h' offset alias, and this config
        # knob is documented as "int H", so normalise before resampling.
        n = average_every_nhours(n, time_resolution.lower())

    # segments with package tsam

    if is_string and time_resolution.lower().endswith("seg"):
        solver_name = snakemake.config["solving"]["solver"]["name"]
        segments = int(time_resolution.lower().replace("seg", ""))
        n = apply_time_segmentation(n, segments, solver_name)

    if params.co2limit_enable:
        add_co2limit(n, params.co2limit, num_years)

    if params.gaslimit_enable:
        add_gaslimit(n, params.gaslimit, num_years)

    emission_prices = params.costs["emission_prices"]
    if emission_prices["enable"]:
        add_emission_prices(
            n,
            dict(co2=params.costs["emission_prices"]["co2"]),
        )

    ll_type, factor = snakemake.wildcards.ll[0], snakemake.wildcards.ll[1:]
    set_transmission_limit(n, ll_type, factor)

    set_line_nom_max(
        n,
        s_nom_max_set=params.lines.get("s_nom_max", np.inf),
        p_nom_max_set=params.links.get("p_nom_max", np.inf),
        s_nom_max_ext=params.lines.get("max_extension", np.inf),
        p_nom_max_ext=params.links.get("max_extension", np.inf),
    )

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    log_network_schema(n, stage="exit", baseline=schema_entry)
    n.export_to_netcdf(snakemake.output[0])
