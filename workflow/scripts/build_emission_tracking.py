"""Module for building state and sector level co2 tracking."""

import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd
import pypsa
from _helpers import calculate_annuity, get_currency_conversion_factor
from constants import CO2_MAX, discount_rate
from duckdb.experimental.spark import DataFrame

logger = logging.getLogger(__name__)


def build_co2_tracking(
    n: pypsa.Network,
    config: dict[str, Any] | None = None,
    simpsec: bool = False,
    simplify_co2: bool = True,
    co2_storage_file: str = None,
    addi_costs: DataFrame = None,
) -> None:
    """Main funtion to interface with."""
    states = n.buses.STATE.unique()

    sectors = ["pwr", "trn", "res", "com", "ind"] if simpsec == False else [""]

    if not config:
        config = {}

    if "co2" not in n.carriers:
        _add_co2_carrier(n, config)

    _build_co2_bus(n, states, sectors, simplify_co2)
    _build_co2_store(n, states, sectors, simplify_co2, co2_storage_file, addi_costs)


def build_ch4_tracking(
    n: pypsa.Network,
    gwp: float,
    upstream_leakage_rate: float,
    downstream_leakage_rate: float,
    plotting_config: dict[str, Any] | None = None,
) -> None:
    """
    Builds CH4 tracking.

    Natural gas network must already be constructed
    """
    states = [x for x in n.buses.STATE.dropna().unique() if x != np.nan]

    if not plotting_config:
        plotting_config = {}

    if "ch4" not in n.carriers:
        _add_ch4_carrier(n, plotting_config)

    _build_ch4_bus(n, states)
    _build_ch4_store(n, states)
    _build_ch4_upstream(n, gwp, upstream_leakage_rate)
    _build_ch4_downstream(n, gwp, downstream_leakage_rate)

    # supress pypsa warnings
    n.links["bus3"] = n.links.bus3.fillna("")
    n.links["efficiency3"] = n.links.efficiency3.fillna(0)


def _add_co2_carrier(n, config: dict[Any]):
    try:
        nice_name = config["plotting"]["nice_names"]["co2"]
    except KeyError:
        nice_name = "CO2"
    try:
        color = config["plotting"]["tech_colors"]["co2"]
    except KeyError:
        color = "#000000"  # black

    n.add("Carrier", "co2", nice_name=nice_name, color=color)


def _build_co2_bus(n: pypsa.Network, states: list[str], sectors: list[str], simplify_co2: bool):
    """Builds state level co2 bus per sector."""
    df = pd.DataFrame(itertools.product(states, sectors), columns=["state", "sector"])
    if sectors != [""]:
        df.index = df.state + " " + df.sector
        n.madd("Bus", df.index, suffix="-co2", carrier="co2", STATE=df.state)
    else:
        df.index = df.state
        n.madd("Bus", df.index, suffix=" co2 atmosphere", carrier="co2", STATE=df.state)
        if simplify_co2:
            return
        else:
            n.madd("Bus", df.index, suffix=" co2 capture", carrier="co2", STATE=df.state)
            n.madd("Bus", df.index, suffix=" co2 sequestration", carrier="co2", STATE=df.state)


def _build_co2_store(
    n: pypsa.Network,
    states: list[str],
    sectors: list[str],
    simplify_co2: bool,
    co2_storage_file: str = None,
    addi_costs: pd.DataFrame = None,
):
    """Builds state level co2 stores per sector."""
    df = pd.DataFrame(itertools.product(states, sectors), columns=["state", "sector"])
    if sectors != [""]:
        df.index = df.state + " " + df.sector
        n.madd(
            "Store",
            df.index,
            suffix="-co2",
            bus=df.index + "-co2",
            e_nom_extendable=False,
            marginal_cost=0,
            e_nom=1e9,
            e_initial=0,
            e_cyclic=False,
            e_cyclic_per_period=False,
            standing_loss=0,
            e_min_pu=-1,
            e_max_pu=1,
            carrier="co2",
        )
    else:
        df.index = df.state
        if simplify_co2:
            n.madd(
                "Store",
                df.index,
                suffix=" co2 atmosphere",
                bus=df.index + " co2 atmosphere",
                e_nom_extendable=False,
                e_nom=CO2_MAX,
                e_initial=0,
                e_min_pu=-1,
                e_max_pu=1,
                e_cyclic=False,
                e_cyclic_per_period=False,
                standing_loss=0,
                carrier="co2 atmosphere",
            )
        else:
            co2_storage = pd.read_csv(co2_storage_file)
            # Rename columns for easier access
            co2_storage.columns = ["state", "potential_MtCO2", "cost_USD_per_tCO2"]
            co2_storage = co2_storage.set_index("state")

            # Convert potential from MtCO2 to tCO2 (1 MtCO2 = 1,000,000 tCO2)
            co2_storage["potential_tCO2"] = co2_storage["potential_MtCO2"] * 1e6

            e_nom_max_values = df.state.map(co2_storage["potential_tCO2"]) / 1e2
            capital_cost = df.state.map(co2_storage["cost_USD_per_tCO2"])

            n.madd(
                "Store",
                df.index,
                suffix=" co2 sequestration",
                bus=df.index + " co2 sequestration",
                e_nom_extendable=True,
                capital_cost=capital_cost,
                e_nom=0,
                e_nom_max=e_nom_max_values,
                e_initial=0,
                e_min_pu=0,
                e_max_pu=1,
                e_cyclic=False,
                e_cyclic_per_period=False,
                standing_loss=0,
                carrier="co2 sequestration",
            )

            # Calculate CO2 tank capital cost from cost data
            co2_tank_params = addi_costs.loc["CO2 tank"]
            lifetime = co2_tank_params["lifetime"]
            fom_rate = co2_tank_params["FOM"]
            investment = co2_tank_params["investment"]  # EUR/tCO2
            currency_year = co2_tank_params["currency_year"]

            # Calculate annualized capital cost
            conversion_factor = get_currency_conversion_factor(currency_year, "EUR")
            capital_cost = (
                (calculate_annuity(lifetime, discount_rate) + fom_rate * 0.01) * investment * conversion_factor
            )

            n.madd(
                "Store",
                df.index,
                suffix=" co2 capture",
                bus=df.index + " co2 capture",
                e_nom_extendable=True,
                capital_cost=capital_cost,  # Assign CO2 tank capital cost
                e_nom=0,
                e_nom_max=CO2_MAX,
                e_initial=0,
                e_min_pu=0,
                e_max_pu=1,
                e_cyclic=True,
                standing_loss=0,
                carrier="co2 capture",
                lifetime=lifetime,
            )

            n.madd(
                "Store",
                df.index,
                suffix=" co2 atmosphere",
                bus=df.index + " co2 atmosphere",
                e_nom_extendable=False,
                e_nom=CO2_MAX,
                e_initial=0,
                e_min_pu=-1,
                e_max_pu=1,
                e_cyclic=False,
                e_cyclic_per_period=False,
                standing_loss=0,
                carrier="co2 atmosphere",
            )


def _add_ch4_carrier(n, config: dict[Any]):
    try:
        nice_name = config["plotting"]["nice_names"]["ch4"]
    except KeyError:
        nice_name = "CH4"
    try:
        color = config["plotting"]["tech_colors"]["co2"]
    except KeyError:
        color = "#000000"  # black

    n.add("Carrier", "ch4", nice_name=nice_name, color=color)


def _build_ch4_bus(n: pypsa.Network, states: list[str]):
    """Builds state level co2 bus per sector."""
    df = pd.DataFrame(states, columns=["state"])
    df.index = df.state

    n.madd("Bus", df.index, suffix=" gas-ch4", carrier="ch4", STATE=df.state)


def _build_ch4_store(n: pypsa.Network, states: list[str]):
    """Builds state level co2 stores per sector."""
    df = pd.DataFrame(states, columns=["state"])
    df.index = df.state

    n.madd(
        "Store",
        df.index,
        suffix=" gas-ch4",
        bus=df.index + " gas-ch4",
        e_nom_extendable=False,
        marginal_cost=0,
        e_nom=np.inf,
        e_initial=0,
        e_cyclic=False,
        e_cyclic_per_period=False,
        standing_loss=0,
        e_min_pu=0,
        e_max_pu=1,
        carrier="ch4",
    )


def _build_ch4_upstream(n, gwp: float, leakage_rate: float):
    """Modifies existing gas production links."""
    # first extract out exising gas production links
    links = n.links[n.links.carrier == "gas production"].index

    # calculate co2e value per unit injected to the ng system
    emissions = gwp * leakage_rate

    # append the connection to methane stores

    if "bus3" in n.links.columns:
        assert all(n.links.loc[links].bus3.isna())
    if "efficiency3" in n.links.columns:
        assert all(n.links.loc[links].efficiency3.isna())

    n.links.loc[links, "bus3"] = n.links.loc[links,].bus1 + "-ch4"  # 'CA gas-ch4'
    n.links.loc[links, "efficiency3"] = emissions


def _build_ch4_downstream(n, gwp: float, leakage_rate: float):
    """Modifies existing gas consuming links."""
    # want all gas links that originate at the state and are not trade or storage related

    gas_buses = n.buses[n.buses.carrier == "gas"]
    gas_users = n.links[(n.links.bus0.isin(gas_buses.index)) & ~(n.links.carrier.isin(["gas storage", "gas trade"]))]

    links = gas_users.index

    # calculate co2e value per unit injected to the ng system
    emissions = gwp * leakage_rate

    # append the connection to methane stores

    assert all(n.links.loc[links].bus3.isna())
    assert all(n.links.loc[links].efficiency3.isna())

    n.links.loc[links, "bus3"] = n.links.loc[links,].bus0 + "-ch4"  # 'CA gas-ch4'
    n.links.loc[links, "efficiency3"] = emissions
