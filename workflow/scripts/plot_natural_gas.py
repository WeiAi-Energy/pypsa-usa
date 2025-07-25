"""Plotting for natural gas networks."""

import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path

# Optional used as 'arg: callable | None = None' gives TypeError with py3.11
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import pypsa
from _helpers import configure_logging
from constants import NG_MWH_2_MMCF, MMBTU_MWHthemal, Month
from summary_natural_gas import (
    get_gas_demand,
    get_gas_processing,
    get_imports_exports,
    get_linepack,
    get_ng_price,
    get_underground_storage,
)

logger = logging.getLogger(__name__)


MWH_2_MMCF = NG_MWH_2_MMCF

FIG_HEIGHT = 5
FIG_WIDTH = 14


@dataclass
class PlottingData:
    """Class for ploting sector network data."""

    name: str
    getter: callable
    plotter: callable
    nice_name: str | None = None
    unit: str | None = None
    converter: float | None = 1.0
    resample: str | None = None  # "D", "W", "12h" for example
    resample_func: callable | None = None  # pd.Series.sum for example
    plot_by_month: bool | None = False  # not resampled


def _get_month_name(month: Month) -> str:
    return month.name.capitalize()


def _resample_data(df: pd.DataFrame, freq: str, agg_func: callable) -> pd.DataFrame:
    """Helper for resampling data based on input function."""
    if df.empty:
        return df
    else:
        return df.groupby("period").resample(freq, level="timestep").apply(agg_func)


def _group_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.T.groupby(level=0).sum().T


def _sum_state_data(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Sums state data together."""
    if not data:
        return pd.DataFrame()

    dfs = [y for _, y in data.items()]
    return pd.concat(dfs, axis=1)


def _sum_state_trade_data(
    data: dict[dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """Sums state data together."""
    import_data = {}
    export_data = {}

    for state, trade_data in data.items():
        import_data[state] = trade_data["imports"]
        export_data[state] = trade_data["exports"]

    import_data = _sum_state_data(import_data)
    export_data = _sum_state_data(export_data)

    return {"imports": import_data, "exports": export_data}


def _is_trade_data(data: dict[str, Any]) -> bool:
    """
    Trade data has nested dictionaries.

    Other data does not
    """
    for value in data.values():
        if isinstance(value, dict):
            return True
    return False


def plot_gas(
    data: pd.DataFrame,
    title: str,
    units: str,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """General gas plotting function."""
    df = data.copy()

    if df.empty:
        return plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))

    periods = data.index.get_level_values("period").unique()

    n_rows = len(periods)

    fig, axs = plt.subplots(n_rows, 1, figsize=(FIG_WIDTH, FIG_HEIGHT * n_rows))

    for i, period in enumerate(periods):
        period_data = df[df.index.get_level_values("period") == period].droplevel(
            "period",
        )
        if n_rows > 1:
            ax = axs[i]
        else:
            ax = axs
        period_data.plot(
            kind="line",
            ax=ax,
            title=title,
            xlabel="",
            ylabel=f"({units})",
        )

    return fig, axs


def plot_gas_trade(
    data: dict[str, pd.DataFrame],  # str is 'imports' or 'exports'
    title: str,
    units: str,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """General gas trade plotting function."""
    # periods will be the same for imports or exports
    periods = data["imports"].index.get_level_values("period").unique()

    n_rows = len(periods)

    fig, axs = plt.subplots(
        n_rows,
        2,
        sharey=True,
        figsize=(FIG_WIDTH, FIG_HEIGHT * n_rows),
    )

    for i, period in enumerate(periods):
        # plot imports

        imports = data["imports"].copy()

        import_period_data = imports[imports.index.get_level_values("period") == period].droplevel(
            "period",
        )

        ax = axs[i, 0] if n_rows > 1 else axs[0]

        if not import_period_data.empty:
            import_period_data.plot(
                kind="line",
                ax=ax,
                xlabel="",
                ylabel=f"({units})",
                title="Imports",
            )

        # plot exports

        exports = data["exports"].copy()

        export_period_data = exports[exports.index.get_level_values("period") == period].droplevel(
            "period",
        )

        ax = axs[i, 1] if n_rows > 1 else axs[1]

        if not export_period_data.empty:
            export_period_data.plot(
                kind="line",
                ax=ax,
                xlabel="",
                ylabel=f"({units})",
                title="Exports",
            )

    fig.suptitle(title)

    return fig, axs


PLOTTING_META = [
    {
        "name": "fuel_price",
        "nice_name": "State Level Natural Gas Price",
        "unit": "$/MMBTU",
        "converter": (1 / MMBTU_MWHthemal),  # $/MWh -> $/MMBTU
        "getter": get_ng_price,
        "plotter": plot_gas,
        "resample": "D",
        "resample_func": pd.Series.mean,
        "plot_by_month": False,
    },
    {
        "name": "demand",
        "nice_name": "Natural Gas Demand",
        "unit": "MMCF",
        "converter": MWH_2_MMCF,
        "getter": get_gas_demand,
        "plotter": plot_gas,
        "resample": "D",
        "resample_func": pd.Series.mean,
        "plot_by_month": True,
    },
    {
        "name": "processing",
        "nice_name": "Natural Gas Processed",
        "unit": "MMCF",
        "converter": MWH_2_MMCF,
        "getter": get_gas_processing,
        "plotter": plot_gas,
        "resample": "D",
        "resample_func": pd.Series.sum,
        "plot_by_month": True,
    },
    {
        "name": "linepack",
        "nice_name": "Natural Gas in Linepack",
        "unit": "MMCF",
        "converter": MWH_2_MMCF,
        "getter": get_linepack,
        "plotter": plot_gas,
        "resample": "D",
        "resample_func": pd.Series.sum,
        "plot_by_month": True,
    },
    {
        "name": "storage",
        "nice_name": "Natural Gas in Underground Storage",
        "unit": "MMCF",
        "converter": MWH_2_MMCF,
        "getter": get_underground_storage,
        "plotter": plot_gas,
        "resample": "D",
        "resample_func": pd.Series.sum,
        "plot_by_month": True,
    },
    {
        "name": "domestic_trade",
        "nice_name": "Natural Gas Traded Domestically",
        "unit": "MMCF",
        "converter": MWH_2_MMCF,
        "getter": partial(get_imports_exports, international=False),
        "plotter": plot_gas_trade,
        "resample": "D",
        "resample_func": pd.Series.mean,
        "plot_by_month": True,
    },
    {
        "name": "international_trade",
        "nice_name": "Natural Gas Traded Internationally",
        "unit": "MMCF",
        "converter": MWH_2_MMCF,
        "getter": partial(get_imports_exports, international=True),
        "plotter": plot_gas_trade,
        "resample": "D",
        "resample_func": pd.Series.mean,
        "plot_by_month": True,
    },
]

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_natural_gas",
            simpl="10",
            opts="3h",
            clusters="4m",
            ll="v1.0",
            sector_opts="",
            sector="E-G",
            planning_horizons="2018",
            interconnect="western",
        )
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)

    output_files = snakemake.output

    states = n.buses[n.buses.reeds_state != ""].reeds_state.unique().tolist()
    states += ["system"]

    plotting_metadata = [PlottingData(**x) for x in PLOTTING_META]

    # hack to only read in the network once, but get images to all states independently
    # ie.
    # "interconnect}/figures/s{{simpl}}_c{{clusters}}/l{{ll}}_{{opts}}_{{sector}}/system/natural_gas/%s.png"

    # {result_name: {state: save_path.png}}
    expected_figures = {}
    for output_file in output_files:
        p = Path(output_file)
        root_path = list(p.parts[:-3])  # path up to the 'system/natural_gas/%s.png'
        figure_name = list(p.parts[-2:])  # path of 'natural_gas/%s.png'
        result = p.stem  # ie. 'demand'
        state_paths = {}
        for state in states:
            full_path = [*root_path, state, *figure_name]
            full_path = Path("/".join(full_path))
            state_paths[state] = full_path
        expected_figures[result] = state_paths

    for meta in plotting_metadata:
        if meta.name not in expected_figures:
            logger.warning(f"Not expecting {meta.name} natural gas chart")
            continue

        data = meta.getter(n)

        for state in states:
            if state == "system":
                if _is_trade_data(data):
                    state_data = _sum_state_trade_data(data)
                else:
                    state_data = _sum_state_data(data)
            else:
                try:
                    state_data = data[state]
                except KeyError:
                    logger.info(f"No {meta.nice_name} data for {state}")
                    continue

            if isinstance(state_data, pd.DataFrame):
                state_data = _group_data(state_data).mul(meta.converter)
            # trade data tracked a little different
            else:
                state_data["imports"] = _group_data(state_data["imports"]).mul(
                    meta.converter,
                )
                state_data["exports"] = _group_data(state_data["exports"]).mul(
                    meta.converter,
                )

            if meta.resample:
                title = f"{state} {meta.nice_name} resampled to {meta.resample}"
                if isinstance(state_data, pd.DataFrame):
                    state_data_resampled = _resample_data(
                        state_data,
                        meta.resample,
                        meta.resample_func,
                    )
                elif isinstance(state_data, dict):
                    state_data_resampled = {}
                    for k, v in state_data.items():
                        state_data_resampled[k] = _resample_data(
                            v,
                            meta.resample,
                            meta.resample_func,
                        )
            else:
                title = f"{state} {meta.nice_name}"
                state_data_resampled = state_data

            units = meta.unit

            fig, _ = meta.plotter(state_data_resampled, title=title, units=units)
            fig.tight_layout()

            save_path = expected_figures[meta.name][state]
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path))
            plt.close(fig)

            if not meta.plot_by_month:
                continue

            months = {month.value: _get_month_name(month) for month in Month}

            for month_i, month_name in months.items():
                if isinstance(state_data, pd.DataFrame):
                    if not state_data.empty:
                        state_data_month = state_data[state_data.index.get_level_values("timestep").month == month_i]
                    else:
                        state_data_month = state_data
                elif isinstance(state_data, dict):
                    state_data_month = {}
                    for k, v in state_data.items():
                        state_data_month[k] = v[v.index.get_level_values("timestep").month == month_i]

                title = f"{state} {meta.nice_name} {month_name}"

                fig, _ = meta.plotter(state_data_month, title=title, units=units)
                fig.tight_layout()

                # this is ugly, but just create subdir of name and index by month
                save_path = Path(
                    expected_figures[meta.name][state].parent,
                    meta.name,
                    f"{month_name}.png",
                )
                if not save_path.parent.exists():
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(save_path))
                plt.close(fig)
