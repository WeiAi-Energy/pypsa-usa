"""Build case-scoped, clustered electricity demand from EER 2025 profiles."""

import logging
from typing import ClassVar

import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging, get_weather_year_snapshots, read_network
from select_representative_periods import read_representative_snapshots
from constants import CODE_2_STATE, STATE_2_CODE

logger = logging.getLogger(__name__)


def state_code(value: object) -> str | None:
    label = str(value).strip()
    if label in CODE_2_STATE:
        return label
    return STATE_2_CODE.get(label) or STATE_2_CODE.get(label.title())


class ReadEer:
    """Read every configured EER weather year, shifting each year CST -> UTC."""

    MODEL_YEARS: ClassVar[tuple[int, ...]] = (2021, 2025, 2030, 2035, 2040, 2045, 2050)
    WEATHER_YEARS: ClassVar[tuple[int, ...]] = (
        2007,
        2008,
        2009,
        2010,
        2011,
        2012,
        2013,
        2016,
        2017,
        2018,
        2019,
        2020,
        2021,
        2022,
        2023,
    )
    HOURS_PER_YEAR: ClassVar[int] = 8760
    CST_TO_UTC_SHIFT: ClassVar[int] = 6

    def __init__(self, filepath: str, planning_horizons, weather_years) -> None:
        self.filepath = filepath
        self.planning_horizons = tuple(int(year) for year in planning_horizons)
        self.weather_years = tuple(int(year) for year in weather_years)
        invalid_horizons = sorted(set(self.planning_horizons) - set(self.MODEL_YEARS))
        if invalid_horizons:
            raise ValueError(f"Unsupported EER planning horizons: {invalid_horizons}.")
        if self.weather_years != self.WEATHER_YEARS:
            raise ValueError(
                f"EER weather years must be {self.WEATHER_YEARS}; received {self.weather_years}.",
            )

    @staticmethod
    def _decode(value) -> str:
        return value.decode() if isinstance(value, bytes) else str(value)

    def _read_segment(self, group, weather_year: int) -> pd.DataFrame:
        start = self.WEATHER_YEARS.index(weather_year) * self.HOURS_PER_YEAR
        stop = start + self.HOURS_PER_YEAR
        states = [
            self._decode(column)
            for column in group.columns[:]
            if self._decode(column) != "datetime"
        ]
        # Keep the roll inside each 8760-hour block.  Concatenating first would
        # incorrectly exchange six hours between adjacent weather years.
        data = {
            state: np.roll(
                getattr(group, state)[start:stop],
                self.CST_TO_UTC_SHIFT,
            )
            for state in states
        }
        snapshots = get_weather_year_snapshots([weather_year], drop_leap_day=True)
        return pd.DataFrame(data, index=snapshots)

    def read(self) -> pd.DataFrame:
        import tables

        periods = []
        with tables.open_file(self.filepath, "r") as h5:
            for planning_horizon in self.planning_horizons:
                try:
                    group = h5.get_node(f"/{planning_horizon}")
                except tables.NoSuchNodeError as exc:
                    raise ValueError(
                        f"EER file has no model year {planning_horizon}.",
                    ) from exc
                weather = pd.concat(
                    [self._read_segment(group, year) for year in self.weather_years],
                )
                weather.index = pd.MultiIndex.from_arrays(
                    [
                        np.repeat(planning_horizon, len(weather)),
                        weather.index,
                    ],
                    names=["period", "timestep"],
                )
                periods.append(weather)
        return pd.concat(periods)


def build_allocation(buses: pd.DataFrame, states: pd.Index) -> pd.DataFrame:
    """
    Split each state's EER demand across the buses that will carry it.

    A bus takes the share of its state's demand given by its own ``Pd`` -- the
    peak-demand weight the network already carries, summed through
    mandatory topology-reduction stage. Reading the weight off the
    network the demand is about to be attached to, rather than re-deriving it
    from the base network via the busmap, keeps the allocation consistent with
    every other consumer of ``Pd`` -- notably the zero-injection test in
    ``reduce_low_degree_buses_and_merge_parallel_lines``, which decides which buses survive at all.

    The state key is ``reeds_state``, the same key the policy constraints use.
    It disagrees with the base network's ``state`` attribute on a small number
    of border substations, so this is not bit-identical to allocating off the
    base network; it is self-consistent with the rest of the model instead.
    """
    if not {"Pd", "reeds_state"}.issubset(buses.columns):
        raise ValueError("Clustered buses require Pd and reeds_state for EER allocation.")

    allocation = pd.DataFrame(
        {
            "state": buses.reeds_state.map(state_code),
            "Pd": pd.to_numeric(buses.Pd, errors="coerce").fillna(0.0),
        },
        index=buses.index,
    )
    allocation = allocation[allocation.state.isin(states)]
    if allocation.empty:
        raise ValueError("No clustered buses map onto an EER state.")

    totals = allocation.groupby("state", observed=True).Pd.transform("sum")
    if (totals <= 0).any():
        empty_states = sorted(allocation.state[totals <= 0].unique())
        logger.warning("No bus carries Pd in %s; their demand is dropped.", empty_states)
        allocation, totals = allocation[totals > 0], totals[totals > 0]

    allocation["weight"] = allocation.Pd / totals
    return allocation.rename_axis("bus").reset_index()[["state", "bus", "weight"]]


def select_representative_demand(
    demand: pd.DataFrame,
    snapshots: pd.MultiIndex,
    source_timesteps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Slice UTC demand down to the representative hours and relabel it.

    ``demand`` is indexed by (period, real weather hour) while the clustered
    network carries (period, synthetic timestep), so the two cannot be aligned
    directly -- the source hour recorded per snapshot is the bridge.
    """
    source_index = pd.MultiIndex.from_arrays(
        [snapshots.get_level_values("period"), pd.DatetimeIndex(source_timesteps)],
        names=demand.index.names,
    )
    sliced = demand.reindex(source_index)
    if sliced.isna().any().any():
        raise ValueError(
            "EER demand does not cover every representative source hour. Check that the "
            "representative-period selection and the demand file use the same weather years.",
        )
    sliced.index = snapshots
    return sliced


def main(snakemake) -> None:
    configure_logging(snakemake)
    network = read_network(snakemake.input.network)
    demand = ReadEer(
        snakemake.input.electricity_demand,
        snakemake.params.planning_horizons,
        snakemake.params.renewable_weather_years,
    ).read()

    representative_snapshots = getattr(snakemake.input, "representative_snapshots", None)
    if representative_snapshots is not None:
        snapshots, _, source_timesteps = read_representative_snapshots(representative_snapshots)
        if not snapshots.equals(network.snapshots):
            raise ValueError(
                "Clustered network snapshots do not match the representative-period selection.",
            )
        demand = select_representative_demand(demand, snapshots, source_timesteps)
    else:
        if not demand.index.equals(network.snapshots):
            demand = demand.reindex(network.snapshots)
        if demand.isna().any().any():
            raise ValueError("EER demand does not cover all network snapshots.")

    demand = demand.rename(columns={column: state_code(column) for column in demand.columns})
    demand = demand.loc[:, ~demand.columns.duplicated()]
    demand = demand[[column for column in demand.columns if column in CODE_2_STATE]]
    ac_buses = network.buses.index[network.buses.carrier == "AC"]
    allocation = build_allocation(network.buses.loc[ac_buses], demand.columns)
    output = pd.DataFrame(0.0, index=network.snapshots, columns=ac_buses)
    for state, group in allocation.groupby("state", observed=True):
        output.loc[:, group.bus] += demand[state].to_numpy()[:, None] * group.weight.to_numpy()[None, :]
    output.to_pickle(snakemake.output.elec_demand)
    logger.info("Wrote EER demand with %s snapshots to %s.", len(output), snakemake.output.elec_demand)


if __name__ == "__main__":
    main(snakemake)
