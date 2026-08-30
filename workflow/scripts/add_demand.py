"""
Adds demand to the network.

Depending on study, the load will all be aggregated to a single load
type, or distributed to different sectors and end use fuels.
"""

import logging
import pandas as pd
import pypsa
from _helpers import configure_logging, mock_snakemake, read_network

logger = logging.getLogger(__name__)

DISTRIBUTION_LOSS_PCT = 0.05


def apply_ac_demand_losses(df: pd.DataFrame, carrier: str) -> pd.DataFrame:
    """Scale AC demand up by the fixed distribution-loss factor."""
    if carrier != "AC":
        return df

    logger.info(
        "Scaling AC demand by %.4f to account for distribution losses",
        1 + DISTRIBUTION_LOSS_PCT,
    )
    return df.mul(1 + DISTRIBUTION_LOSS_PCT)


def attach_demand(n: pypsa.Network, df: pd.DataFrame, carrier: str, suffix: str):
    """
    Add demand to network from specified configuration setting.

    Returns network with demand added.
    """
    # The demand pickle is written with the network's own (period, timestep)
    # MultiIndex, which pd.to_datetime cannot convert -- only coerce flat indices.
    if not isinstance(df.index, pd.MultiIndex):
        df.index = pd.to_datetime(df.index)
    assert len(df.index) == len(
        n.snapshots,
    ), "Demand time series length does not match network snapshots"
    df.index = n.snapshots
    n.madd(
        "Load",
        df.columns,
        suffix=suffix,
        bus=df.columns,
        p_set=df,
        carrier=carrier,
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "add_demand",
            case="test",
        )
    configure_logging(snakemake)

    demand_files = snakemake.input.demand
    n = read_network(snakemake.input.network)

    planning_horizons = snakemake.params.planning_horizons
    if list(n.investment_periods) != [int(year) for year in planning_horizons]:
        raise ValueError("Clustered network investment periods do not match the case configuration.")

    if isinstance(demand_files, str):
        demand_files = [demand_files]

    if len(demand_files) != 1:
        raise ValueError("Electricity-only demand requires exactly one EER profile.")

    df = pd.read_pickle(demand_files[0])
    df = apply_ac_demand_losses(df, "AC")
    attach_demand(n, df, "AC", "")
    logger.info("Electricity demand added to network")

    n.export_to_netcdf(snakemake.output.network)
