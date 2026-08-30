"""
Prepare PyPSA network for solving according to :ref:`opts` and :ref:`ll`, such
as.

- adding an annual **limit** of carbon-dioxide emissions,
- setting an **N-1 security margin** factor for transmission line capacities,
- specifying an expansion limit on the **cost** of transmission expansion,
- specifying an expansion limit on the **volume** of transmission expansion, and
- reducing the **temporal** resolution by averaging over multiple hours
  or segmenting time series into chunks of varying lengths using ``tsam``.
"""

from collections import deque
import json
import logging

import constants as const
import numpy as np
import pandas as pd
import pypsa
from capacity_derates import apply_temperature_derates
from opts.policy import option_enabled, remove_tct_blocked_components
from pypsa.components import convert_lines_to_line_x
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from _helpers import (
    calculate_annuity,
    configure_logging,
    get_complete_bidirectional_link_pairs,
    is_transport_model,
    set_case_config,
    update_config_from_wildcards,
)

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def get_tree_buses_from_lines(n: pypsa.Network) -> pd.Index:
    """
    Identify tree buses by iteratively pruning leaf buses from the line graph.

    The remaining buses form the 2-core of the undirected graph induced by
    ``n.lines``. All removed buses are treated as tree buses.
    """
    if n.lines.empty:
        return pd.Index([])

    line_endpoints = n.lines[["bus0", "bus1"]]
    line_names = n.lines.index
    buses = pd.Index(pd.unique(line_endpoints.to_numpy().ravel()))
    if buses.empty:
        return pd.Index([])

    neighbors = {bus: [] for bus in buses}
    degree = pd.Series(0, index=buses, dtype="int64")

    for line_name, bus0, bus1 in line_endpoints.itertuples(index=True, name=None):
        neighbors[bus0].append((line_name, bus1))
        neighbors[bus1].append((line_name, bus0))
        degree.at[bus0] += 1
        degree.at[bus1] += 1

    active_lines = pd.Series(True, index=line_names, dtype="bool")
    queue = deque(degree.index[degree <= 1].tolist())
    tree_buses = []
    visited = set()

    while queue:
        bus = queue.popleft()
        if bus in visited:
            continue
        visited.add(bus)
        tree_buses.append(bus)

        for line_name, neighbor in neighbors[bus]:
            if not active_lines.at[line_name]:
                continue
            active_lines.at[line_name] = False
            degree.at[bus] -= 1
            degree.at[neighbor] -= 1
            if neighbor not in visited and degree.at[neighbor] <= 1:
                queue.append(neighbor)

    return pd.Index(tree_buses)


def _get_line_x_conversion_config(config):
    """Return LineX conversion config."""
    return config.get("convert_lines_to_line_x", {})


def _get_sssc_annualized_capex_per_mw(config):
    """Calculate annualized SSSC capex per MW from config."""
    sssc_costs = _get_line_x_conversion_config(config)
    return (
        calculate_annuity(
            sssc_costs["cost_recovery_period_years"],
            sssc_costs["wacc_real"],
        )
        * sssc_costs["capex_per_kva"]
        * 1e3
    )


def _get_phase_shifting_transformer_annualized_capex_per_mw(config):
    """Calculate annualized phase-shifting-transformer capex per MW from config."""
    pst_costs = config["phase_shifting_transformer"]
    return (
        calculate_annuity(
            pst_costs["cost_recovery_period_years"],
            pst_costs["wacc_real"],
        )
        * pst_costs["capex_per_kw"]
        * 1e3
    )


def _get_line_x_conversion_candidates(
    n: pypsa.Network,
    tree_buses: pd.Index,
) -> tuple[pd.Index, int]:
    """Return eligible Line names for conversion to LineX assets."""
    line_names = n.lines.index[
        ~n.lines.bus0.isin(tree_buses) & ~n.lines.bus1.isin(tree_buses)
    ]
    excluded_tree_lines = len(n.lines) - len(line_names)
    return line_names, excluded_tree_lines


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


def set_line_s_max_pu(n, transport_model, s_max_pu=1.0):
    if not transport_model:
        logger.info(f"N-1 security margin of lines set to {s_max_pu}")
        n.lines["s_max_pu"] = s_max_pu


def _apply_bidirectional_transmission_link_volume_correction(n):
    """
    Halve `length` for AC/DC `_fwd`/`_rev` link pairs.

    Transmission interfaces are modeled as two directional links, but the
    transmission volume constraint in PyPSA sums `length * p_nom` over links.
    Halving each direction's length makes the pair count once per physical
    corridor while preserving the existing `_fwd`/`_rev` formulation.
    """
    if n.links.empty or "length" not in n.links.columns:
        return

    transmission_links = n.links[n.links.carrier.isin(["AC", "DC"])]
    complete_pairs = get_complete_bidirectional_link_pairs(transmission_links)
    if not complete_pairs:
        return

    paired_links = pd.Index(
        [link_name for pair in complete_pairs.values() for link_name in pair.values()],
    )
    logger.info(
        "Halving `length` for %s bidirectional AC/DC transmission link pairs "
        "so each physical corridor is counted once in transmission volume limits.",
        len(complete_pairs),
    )
    n.links.loc[paired_links, "length"] = n.links.loc[paired_links, "length"] / 2


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

    _apply_bidirectional_transmission_link_volume_correction(n)

    dc_links = n.links.carrier == "DC" if not n.links.empty else pd.Series(dtype=bool)
    ac_links_existing = n.links.carrier == "AC" if not n.links.empty else pd.Series(dtype=bool)

    line_xs = getattr(n, "line_xs", pd.DataFrame())
    line_xs_s_nom = line_xs.s_nom if not line_xs.empty else pd.Series(dtype="float64")
    lines_s_nom = n.lines.s_nom
    col = "capital_cost" if ll_type == "c" else "length"
    line_x_col = "capital_cost_line" if ll_type == "c" else "length"
    ref = (
        lines_s_nom @ n.lines[col]
        + line_xs_s_nom @ line_xs.get(line_x_col, pd.Series(0.0, index=line_xs.index, dtype="float64"))
        + n.links.loc[dc_links, "p_nom"] @ n.links.loc[dc_links, col]
        + n.links.loc[ac_links_existing, "p_nom"] @ n.links.loc[ac_links_existing, col]
    )

    is_unbounded = factor == "inf"

    if factor == "opt" or is_unbounded or float(factor) > 1.0:
        # if opt allows expansion set respective lines/links to extendable
        # all links prior to this point have extendable set to false
        n.lines["s_nom_min"] = lines_s_nom
        n.lines["s_nom_extendable"] = True
        if not line_xs.empty:
            line_xs["s_nom_min"] = line_xs_s_nom
            line_xs["s_nom_extendable"] = True

        n.links.loc[dc_links, "p_nom_min"] = n.links.loc[dc_links, "p_nom"]
        n.links.loc[dc_links, "p_nom_extendable"] = True

        n.links.loc[ac_links_existing, "p_nom_min"] = n.links.loc[ac_links_existing, "p_nom"]
        n.links.loc[ac_links_existing, "p_nom_extendable"] = True
    if factor not in {"opt", "inf"}:
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


def _infer_timestep_hours(n):
    """Infer the timestep spacing (hours) from an investment period's snapshots."""
    snapshots = n.snapshots
    if isinstance(snapshots, pd.MultiIndex):
        period = snapshots.get_level_values(0)[0]
        timesteps = pd.DatetimeIndex(snapshots[snapshots.get_level_values(0) == period].get_level_values(1))
    else:
        timesteps = pd.DatetimeIndex(snapshots)
    if len(timesteps) < 2:
        raise ValueError("Cannot infer timestep resolution from fewer than 2 snapshots.")
    return (timesteps[1] - timesteps[0]).total_seconds() / 3600.0


def _rescale_representative_metadata_steps(metadata, scale_factor):
    """
    Rescale representative-period step counts to match a temporal-resolution change.

    ``representative_metadata`` is written by ``select_representative_periods`` at
    the native (hourly) resolution used for tsam clustering. When ``prepare_network``
    subsequently averages snapshots to a coarser resolution (e.g. the ``3h`` in an
    ``ERM-3h`` opts wildcard), every block's snapshot count shrinks by the same
    factor, but the metadata is not otherwise touched -- so ``steps`` must be
    rescaled here or downstream block-slicing in ``opts/representative_periods.py``
    will consume the wrong number of snapshots per block and overrun the index.
    """
    if np.isclose(scale_factor, 1.0):
        return metadata

    for horizon_key, horizon_data in metadata.items():
        periods = horizon_data.get("periods", []) if isinstance(horizon_data, dict) else []
        for entry in periods:
            steps = entry.get("steps", 0)
            new_steps = steps / scale_factor
            if not np.isclose(new_steps, round(new_steps)):
                raise ValueError(
                    f"Representative period {entry.get('period_id')} for investment period "
                    f"{horizon_key} has {steps} native steps, which is incompatible with a "
                    f"{scale_factor:g}x temporal-resolution change.",
                )
            entry["steps"] = int(round(new_steps))
    return metadata


def average_every_nhours(n, offset):
    logger.info(f"Resampling the network to {offset}")
    m = n.copy(with_time=False)

    def resample_multi_index(df, offset, func):
        resampled = []
        for period in df.index.unique("period"):
            sw_ = df.xs(period, level="period").resample(offset).apply(func)
            sns = sw_.index
            sns = sns[~((sns.month == 2) & (sns.day == 29))]
            sw_ = sw_.loc[sns]
            sw_.index = pd.MultiIndex.from_arrays(
                [np.repeat(period, len(sw_)), sw_.index],
                names=["period", "timestep"],
            )
            resampled.append(sw_)
        return pd.concat(resampled)

    snapshot_weightings = resample_multi_index(n.snapshot_weightings, offset, "sum")
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings
    m.investment_periods = n.investment_periods

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name + "_t")
        for k, df in c.pnl.items():
            if not df.empty:
                pnl[k] = resample_multi_index(df, offset, "mean")
    return m


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
    for c in n.iterate_components():
        for attr, pnl in c.pnl.items():
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
    if np.isfinite(s_nom_max_ext) and s_nom_max_ext == 0:
        logger.info("Disabling line and LineX expansion")
        n.lines["s_nom_extendable"] = False
        n.lines["s_nom_max"] = n.lines["s_nom"]
        if hasattr(n, "line_xs") and not n.line_xs.empty:
            n.line_xs["s_nom_extendable"] = False
            n.line_xs["s_nom_max"] = n.line_xs["s_nom"]

    if np.isfinite(s_nom_max_ext) and s_nom_max_ext > 0:
        logger.info(f"Limiting line and LineX extensions to {s_nom_max_ext} MW")
        n.lines["s_nom_max"] = n.lines["s_nom"] + s_nom_max_ext
        if hasattr(n, "line_xs") and not n.line_xs.empty:
            n.line_xs["s_nom_max"] = n.line_xs["s_nom"] + s_nom_max_ext

    hvdc = n.links.index[n.links.carrier == "DC"]
    if np.isfinite(p_nom_max_ext) and p_nom_max_ext == 0:
        logger.info("Disabling DC link expansion")
        n.links.loc[hvdc, "p_nom_extendable"] = False
        n.links.loc[hvdc, "p_nom_max"] = n.links.loc[hvdc, "p_nom"]

    if np.isfinite(p_nom_max_ext) and p_nom_max_ext > 0:
        logger.info(f"Limiting link extensions to {p_nom_max_ext} MW")
        n.links.loc[hvdc, "p_nom_max"] = n.links.loc[hvdc, "p_nom"] + p_nom_max_ext

    # Clip to the configured cap, but never below s_nom_min: a branch whose existing
    # capacity already exceeds s_nom_max_set would otherwise end up with s_nom_max <
    # s_nom_min, an infeasible bound pair for its own extendable capacity variable.
    # Two sequential single-bound clips (rather than one clip(lower=..., upper=...)
    # call) because pandas resolves a lower > upper conflict by returning upper, not
    # lower -- the floor must be applied only after the cap.
    n.lines["s_nom_max"] = n.lines.s_nom_max.clip(upper=s_nom_max_set).clip(lower=n.lines.s_nom_min)
    if hasattr(n, "line_xs") and not n.line_xs.empty:
        n.line_xs["s_nom_max"] = n.line_xs.s_nom_max.clip(upper=s_nom_max_set).clip(lower=n.line_xs.s_nom_min)
        if "sssc_nom_max" in n.line_xs.columns:
            n.line_xs["sssc_nom_max"] = n.line_xs.sssc_nom_max.clip(upper=s_nom_max_set)
    n.links.loc[hvdc, "p_nom_max"] = n.links.loc[hvdc, "p_nom_max"].clip(upper=p_nom_max_set)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_network",
            case="test",
            ll="v1.30",
            opts="ERM-3h",
        )
    configure_logging(snakemake)
    set_case_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)
    params = snakemake.params
    transport_model = is_transport_model(params.transmission_network)

    n = pypsa.Network(snakemake.input.network)

    representative_metadata = {}
    metadata_path = getattr(snakemake.input, "representative_metadata", None)
    if metadata_path is not None:
        with open(metadata_path) as stream:
            representative_metadata = json.load(stream)
        # Built-in cyclic flags stay disabled: representative-period cyclic
        # closure is enforced explicitly per block in solve_network.
        if not n.storage_units.empty:
            n.storage_units["cyclic_state_of_charge"] = False
            n.storage_units["cyclic_state_of_charge_per_period"] = False
        if not n.stores.empty:
            n.stores["e_cyclic"] = False
            n.stores["e_cyclic_per_period"] = False

    num_years = n.snapshot_weightings.loc[n.investment_periods[0]].objective.sum() / 8760.0
    costs = pd.read_csv(snakemake.input.tech_costs)
    costs = costs.pivot(index="pypsa-name", columns="parameter", values="value")
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

    set_line_s_max_pu(n, transport_model, params.lines["s_max_pu"])
    n.links.loc[n.links.carrier == "DC", "p_max_pu"] = params.links["p_max_pu"]

    # Temperature-dependent capacity derates, applied before any temporal
    # averaging so the resampling sees derated profiles. This replaces the static
    # summer/winter derate that used to run in add_electricity, which is no longer
    # possible now that snapshots carry synthetic timestamps.
    region_temperature = getattr(snakemake.input, "region_temperature", None)
    if region_temperature is not None:
        apply_temperature_derates(
            n,
            region_temperature,
            snakemake.input.outage_forced_temperature,
        )
    else:
        logger.warning(
            "No region temperature input: skipping temperature-dependent capacity derates. "
            "Thermal generators keep p_max_pu = 1.0 (the static seasonal derate has been removed).",
        )

    # temporal averaging
    time_resolution = params.time_resolution
    is_string = isinstance(time_resolution, str)
    if is_string and time_resolution.lower().endswith("h"):
        native_timestep_hours = _infer_timestep_hours(n) if representative_metadata else None
        n = average_every_nhours(n, time_resolution)
        if representative_metadata:
            resampled_timestep_hours = _infer_timestep_hours(n)
            scale_factor = resampled_timestep_hours / native_timestep_hours
            representative_metadata = _rescale_representative_metadata_steps(
                representative_metadata,
                scale_factor,
            )
            logger.info(
                "Rescaled representative-period metadata steps for %sh -> %sh resolution (factor %.3g).",
                native_timestep_hours,
                resampled_timestep_hours,
                scale_factor,
            )

    # segments with package tsam
    if is_string and time_resolution.lower().endswith("seg"):
        if representative_metadata:
            raise ValueError(
                "clustering.temporal.representative_periods is not compatible with a 'seg' "
                "time_resolution: representative-period cyclic constraints require uniform, "
                "block-aligned timesteps.",
            )
        solver_name = snakemake.config["solving"]["solver"]["name"]
        segments = int(time_resolution.lower().replace("seg", ""))
        n = apply_time_segmentation(n, segments, solver_name)

    if params.co2limit_enable:
        add_co2limit(n, params.co2limit, num_years)

    if params.gaslimit_enable:
        add_gaslimit(n, params.gaslimit, num_years)

    line_x_config = _get_line_x_conversion_config(params.lines)
    line_x_conversion = line_x_config.get("enable", False)
    if line_x_conversion:
        sssc_annualized_capex_per_mw = _get_sssc_annualized_capex_per_mw(
            params.lines,
        )
        tree_buses = get_tree_buses_from_lines(n)
        line_names, excluded_tree_lines = _get_line_x_conversion_candidates(
            n,
            tree_buses,
        )
        convert_lines_to_line_x(
            n,
            names=line_names,
            capital_cost_sssc=sssc_annualized_capex_per_mw,
        )
        logger.info(
            "Converted %s Lines to LineXs after excluding %s tree-connected lines.",
            len(line_names),
            excluded_tree_lines,
        )

    ll_type, factor = snakemake.wildcards.ll[0], snakemake.wildcards.ll[1:]
    set_transmission_limit(
        n,
        ll_type,
        factor,
    )

    set_line_nom_max(
        n,
        s_nom_max_set=params.lines.get("s_nom_max", np.inf),
        p_nom_max_set=params.links.get("p_nom_max", np.inf),
        s_nom_max_ext=params.lines.get("max_extension", np.inf),
        p_nom_max_ext=params.links.get("max_extension", np.inf),
    )

    if option_enabled(snakemake.wildcards.opts, "TCT"):
        remove_tct_blocked_components(
            n,
            snakemake.config,
            components=("Generator", "StorageUnit", "Link"),
        )

    existing_meta = getattr(n, "meta", {})
    if not isinstance(existing_meta, dict):
        existing_meta = {}
    if representative_metadata:
        existing_meta["representative_periods_plot_metadata"] = representative_metadata
    n.meta = {
        **snakemake.config,
        **existing_meta,
        "wildcards": dict(snakemake.wildcards),
    }
    n.export_to_netcdf(snakemake.output[0])
