"""Module for adding the gas sector.

This module will add a state level copperplate natural gas network to the model.
Specifically, it will do the following

- Creates capacity constrained pipelines between state gas buses (links)
- Creates capacity constraind natural gas processing facilites (generators)
- Creates capacity and energy constrainted underground gas storage facilities
- Creates energy constrained linepack storage (storage units)
- Creates capacity constrained pipelines to states neighbouring the interconnect
- Creates capacity and energy constrained import/exports to international connections
- Adds import/export historical natural gas prices
"""

import logging
from abc import ABC, abstractmethod
from math import pi
from typing import Any

import eia
import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import yaml
from _helpers import calculate_annuity, get_currency_conversion_factor
from constants import (
    CODE_2_STATE,
    EMPTY_STATES,
    NG_MWH_2_MMCF,
    STATE_2_CODE,
    STATES_INTERCONNECT_MAPPER,
    HHV_to_LHV_CH4,
    discount_rate,
    leakage_rate,
)
from pypsa.components import Network
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


###
# Constants
###

# for converting everthing into MWh_th
MWH_2_MMCF = NG_MWH_2_MMCF
KJ_2_MWH = (1 / 1000) * (1 / 3600)

###
# Geolocation of Assets class
###


class StateGeometry:
    """Holds state boundry data."""

    def __init__(self, shapefile: str) -> None:
        """Counties shapefile."""
        self._counties = gpd.read_file(shapefile)
        self._state_center_points = None
        self._states = None

    @property
    def counties(self) -> gpd.GeoDataFrame:
        """Spatially resolved counties."""
        return self._counties

    @property
    def states(self) -> gpd.GeoDataFrame:
        """Spatially resolved states."""
        if self._states:
            return self._states
        else:
            self._states = self._get_state_boundaries()
            return self._states

    @property
    def state_center_points(self) -> gpd.GeoDataFrame:
        """Center points of Sates."""
        if self._state_center_points:
            return self._state_center_points
        else:
            if not self._states:
                self._states = self._get_state_boundaries()
            self._state_center_points = self._get_state_center_points()
            return self._state_center_points

    def _get_state_boundaries(self) -> gpd.GeoDataFrame:
        """Gets admin boundaries of state."""
        return (
            self._counties.dissolve("STATE_NAME")
            .rename(columns={"STUSPS": "STATE"})
            .reset_index()[["STATE_NAME", "STATE", "geometry"]]
        )

    def _get_state_center_points(self) -> gpd.GeoDataFrame:
        """Gets centerpoints of states using county shapefile."""
        gdf = self._states.copy().rename(columns={"geometry": "shape"})
        gdf["geometry"] = gdf["shape"].map(lambda x: x.centroid)
        gdf[["x", "y"]] = gdf["geometry"].apply(
            lambda x: pd.Series({"x": x.x, "y": x.y}),
        )
        return gdf[["STATE", "x", "y"]]


###
# MAIN DATA INTERFACE
###


class GasData(ABC):
    """Interface with any gas data."""

    state_2_interconnect = STATES_INTERCONNECT_MAPPER
    state_2_name = CODE_2_STATE
    name_2_state = STATE_2_CODE
    states_2_remove = EMPTY_STATES

    def __init__(self, year: int, interconnect: str) -> None:
        self.year = year
        if interconnect.lower() not in ("western", "eastern", "texas", "usa"):
            logger.debug(f"Invalid interconnect of {interconnect}. Setting to 'usa'")
            self.interconnect = "usa"  # no filtering of data
        else:
            self.interconnect = interconnect.lower()
        self._data = self._get_data()

    @property
    def data(self) -> pd.DataFrame:
        """Get formatted data."""
        return self._data

    @abstractmethod
    def read_data(self) -> pd.DataFrame | gpd.GeoDataFrame:
        """Read in data."""
        pass

    @abstractmethod
    def format_data(self, data: pd.DataFrame | gpd.GeoDataFrame) -> pd.DataFrame:
        """Format dataset."""
        pass

    def _get_data(self) -> pd.DataFrame:
        data = self.read_data()
        return self.format_data(data)

    @abstractmethod
    def build_infrastructure(self, n: pypsa.Network) -> None:
        """Add pypsa components to network."""
        pass

    def filter_on_interconnect(
        self,
        df: pd.DataFrame,
        additional_removals: list[str] | None = None,
    ) -> pd.DataFrame:
        """Name of states must be in column called 'STATE'."""
        states_2_remove = self.states_2_remove
        if additional_removals:
            states_2_remove += additional_removals

        if "STATE" not in df.columns:
            logger.debug(
                "Natual gas data not filtered due to incorrect data formatting",
            )
            return df

        df = df[~df.STATE.isin(states_2_remove)].copy()

        if self.interconnect == "usa":
            return df
        else:
            df["interconnect"] = df.STATE.map(self.state_2_interconnect)
            assert not df.interconnect.isna().any()
            df = df[df.interconnect == self.interconnect]
            if df.empty:
                logger.warning(
                    f"Empty natural gas data for interconnect {self.interconnect}",
                )
            return df.drop(columns="interconnect")

    @abstractmethod
    def filter_on_sate(
        self,
        n: pypsa.Network,
        df: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Called before adding infrastructure to check if only modelling a subset
        of interconnect.
        """
        pass


class GasBuses(GasData):
    """
    Creator for natural gas buses.

    Argumets:
        County shapefile of United States
    """

    def __init__(self, interconnect: str, counties: str) -> None:
        self.states = StateGeometry(counties)
        super().__init__(
            year=2020,
            interconnect=interconnect,
        )  # year locked for location mapping

    def read_data(self) -> gpd.GeoDataFrame:
        """Read in state centerpoints."""
        return pd.DataFrame(self.states.state_center_points)

    def format_data(self, data: gpd.GeoDataFrame) -> pd.DataFrame:
        """Format bus data."""
        data = pd.DataFrame(data)
        data["name"] = data.STATE.map(self.state_2_name)
        return self.filter_on_interconnect(data)

    def filter_on_sate(
        self,
        n: pypsa.Network,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Filter formatted data to only include states in geographic scope."""
        states_in_model = n.buses[
            ~n.buses.carrier.isin(
                ["gas storage", "gas trade", "gas pipeline"],
            )
        ].reeds_state.unique()

        if "STATE" not in df.columns:
            logger.debug(
                "Natual gas data not filtered due to incorrect data formatting",
            )
            return df

        df = df[df.STATE.isin(states_in_model)].copy()

        return df

    def build_infrastructure(self, n: Network) -> None:
        """Add pypsa components to network."""
        df = self.filter_on_sate(n, self.data)

        states = df.set_index("STATE")

        n.madd(
            "Bus",
            names=states.index,
            suffix=" gas",
            x=states.x,
            y=states.y,
            carrier="gas",
            unit="MWh_th",
            interconnect=self.interconnect,
            country=states.index,  # for consistency
            STATE=states.index,
            STATE_NAME=states.name,
        )


class GasStorage(GasData):
    """Creator for underground storage."""

    def __init__(self, year: int, interconnect: str, storage_plant_path: str = None, agg_storage: int = 1) -> None:
        self.storage_plant_path = storage_plant_path
        self.agg_storage = agg_storage

        super().__init__(year, interconnect)

    def read_data(self):
        """Read and process CSV storage plant data."""
        df = pd.read_csv(self.storage_plant_path)

        # Clean column names (remove line breaks and <BR> tags)
        df.columns = df.columns.str.replace("<BR>", " ", regex=False).str.replace("\n", " ", regex=False).str.strip()

        # Map cleaned column names to standard names
        column_mapping = {
            "ReportState": "Report State",
            "Report State": "Report State",  # Keep if already correct
            "Base Gas": "Base Gas Capacity",
            "Working Gas Capacity(Mcf)": "Working Gas Capacity",
            "Total Field Capacity(Mcf)": "Total Field Capacity",
            "Maximum Daily Delivery(Mcf)": "Maximum Daily Delivery",
            "Field Type": "Field Type",
            "Status": "Status",
        }

        # Find and rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})

        # Drop inactive facilities
        if "Status" in df.columns:
            df = df[df["Status"] != "Inactive"].copy()

        # Convert numeric columns
        numeric_cols = [
            "Base Gas Capacity",
            "Working Gas Capacity",
            "Total Field Capacity",
            "Maximum Daily Delivery",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Fill missing values using regression
        df = self._fill_missing_values(df)

        return df

    def format_data(
        self,
        df: pd.DataFrame,
        retro_storage_type: list = None,
        h2_storage_costs_by_type: dict = None,
        acaes_storage_costs_by_type: dict = None,
    ):
        """Format storage data and separate retrofitable and non-retrofitable storage."""
        df = df.copy()

        # Convert MCF to MWh_th conversion
        MWH_2_MCF = MWH_2_MMCF / 1000

        # Convert capacity columns from MCF to MWh_th
        df["Base Gas Capacity"] = df["Base Gas Capacity"] * MWH_2_MCF
        df["Working Gas Capacity"] = df["Working Gas Capacity"] * MWH_2_MCF
        df["Total Field Capacity"] = df["Total Field Capacity"] * MWH_2_MCF

        # Convert Maximum Daily Delivery from MCF/day to MW
        df["Maximum Daily Delivery"] = df["Maximum Daily Delivery"] * MWH_2_MCF / 24

        # Filter by interconnect
        df = self._filter_csv_by_interconnect(df)

        # Determine retrofitable field types
        if retro_storage_type is None:
            retro_storage_type = []

        # Add retrofitable flag
        df["retrofitable"] = df["Field Type"].isin(retro_storage_type)

        # Aggregate based on agg_storage level
        if self.agg_storage > 0:  # Level 1
            # Determine state column name
            state_col = "Report State"

            # Calculate weighted average H2 storage cost by state before aggregation
            if h2_storage_costs_by_type is not None:
                df["h2_investment_cost"] = (
                    df["Field Type"]
                    .map(h2_storage_costs_by_type)
                    .fillna(
                        h2_storage_costs_by_type.get("Depleted Field", 0),
                    )
                )

            # Calculate weighted average ACAES storage cost by state before aggregation
            if acaes_storage_costs_by_type is not None:
                df["acaes_investment_cost"] = (
                    df["Field Type"]
                    .map(acaes_storage_costs_by_type)
                    .fillna(
                        acaes_storage_costs_by_type.get("Depleted Field", 0),
                    )
                )

            # Calculate weighted average H2 costs for each state-type combination
            state_type_h2_costs = {}
            state_type_acaes_costs = {}
            retro_df = df[df["retrofitable"]].copy()

            if not retro_df.empty:
                for state in retro_df[state_col].unique():
                    for field_type in retro_df["Field Type"].unique():
                        state_type_data = retro_df[
                            (retro_df[state_col] == state) & (retro_df["Field Type"] == field_type)
                        ]
                        if not state_type_data.empty:
                            total_capacity = state_type_data["Total Field Capacity"].sum()

                            weighted_h2_cost = (
                                state_type_data["Total Field Capacity"] * state_type_data["h2_investment_cost"]
                            ).sum() / total_capacity
                            state_type_h2_costs[(state, field_type)] = weighted_h2_cost
                            weighted_acaes_cost = (
                                state_type_data["Total Field Capacity"] * state_type_data["acaes_investment_cost"]
                            ).sum() / total_capacity
                            state_type_acaes_costs[(state, field_type)] = weighted_acaes_cost

            # Aggregate retrofitable storage by state and field type
            agg_df_retro = (
                df[df["retrofitable"]]
                .groupby([state_col, "Field Type"])
                .agg(
                    {
                        "Base Gas Capacity": "sum",
                        "Working Gas Capacity": "sum",
                        "Total Field Capacity": "sum",
                        "Maximum Daily Delivery": "sum",
                    }
                )
                .reset_index()
            )
            agg_df_retro["retrofitable"] = True

            # Add H2 storage costs
            if h2_storage_costs_by_type is not None:
                agg_df_retro["h2_weighted_investment_cost"] = agg_df_retro.apply(
                    lambda row: state_type_h2_costs.get(
                        (row[state_col], row["Field Type"]),
                        h2_storage_costs_by_type.get("Depleted Field", 0),
                    ),
                    axis=1,
                )
            # Add ACAES costs
            if acaes_storage_costs_by_type is not None:
                agg_df_retro["acaes_weighted_investment_cost"] = agg_df_retro.apply(
                    lambda row: state_type_acaes_costs.get(
                        (row[state_col], row["Field Type"]),
                        acaes_storage_costs_by_type.get("Depleted Field", 0),
                    ),
                    axis=1,
                )

            # Aggregate non-retrofitable storage by state and field type
            agg_df_non_retro = (
                df[~df["retrofitable"]]
                .groupby([state_col, "Field Type"])
                .agg(
                    {
                        "Base Gas Capacity": "sum",
                        "Working Gas Capacity": "sum",
                        "Total Field Capacity": "sum",
                        "Maximum Daily Delivery": "sum",
                    }
                )
                .reset_index()
            )
            agg_df_non_retro["retrofitable"] = False

            if h2_storage_costs_by_type is not None:
                agg_df_non_retro["h2_weighted_investment_cost"] = h2_storage_costs_by_type.get("Depleted Field", 0)

            if acaes_storage_costs_by_type is not None:
                agg_df_non_retro["acaes_weighted_investment_cost"] = acaes_storage_costs_by_type.get(
                    "Depleted Field", 0
                )

            # Combine both dataframes
            agg_df = pd.concat([agg_df_retro, agg_df_non_retro], ignore_index=True)

            # Rename state column to State and filter out zero capacity entries
            agg_df = agg_df.rename(columns={state_col: "State"})
            agg_df = agg_df[agg_df["Total Field Capacity"] > 0]

            return agg_df

        # If agg_storage == 0, return individual facilities (original False behavior)
        return df

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using linear regression models."""
        df = df.copy()

        # Get unique field types
        field_types = df["Field Type"].unique()

        for field_type in field_types:
            if pd.isna(field_type):
                continue

            field_mask = df["Field Type"] == field_type
            field_df = df[field_mask].copy()

            # 1. Base gas ratio regression: base_gas_ratio = Base Gas / Total Field Capacity
            # Use rows where both Base Gas and Total Field Capacity are not 0
            valid_ratio_mask = (field_df["Base Gas Capacity"] > 0) & (field_df["Total Field Capacity"] > 0)

            if valid_ratio_mask.sum() > 1:  # Need at least 2 points for regression
                valid_ratio_df = field_df[valid_ratio_mask]

                # Linear regression of base_gas_ratio vs ln(Base Gas)
                X_ratio = np.log(valid_ratio_df["Base Gas Capacity"]).values.reshape(-1, 1)
                y_ratio = (valid_ratio_df["Base Gas Capacity"] / valid_ratio_df["Total Field Capacity"]).values

                ratio_model = LinearRegression()
                ratio_model.fit(X_ratio, y_ratio)

                # Use mean ratio as fallback if regression fails
                mean_ratio = y_ratio.mean()

                # Fill missing Total Field Capacity using regression-predicted base_gas_ratio
                missing_total_mask = field_mask & (df["Total Field Capacity"] == 0) & (df["Base Gas Capacity"] > 0)
                if missing_total_mask.sum() > 0:
                    missing_df = df[missing_total_mask]
                    ln_base_gas = np.log(missing_df["Base Gas Capacity"]).values.reshape(-1, 1)
                    try:
                        predicted_ratio = ratio_model.predict(ln_base_gas)
                        predicted_total = missing_df["Base Gas Capacity"] / predicted_ratio
                    except:
                        # Fallback to mean ratio
                        predicted_total = missing_df["Base Gas Capacity"] / mean_ratio

                    df.loc[missing_total_mask, "Total Field Capacity"] = predicted_total

            # 2. Fill Working Gas Capacity = Total Field Capacity - Base Gas (if Working Gas is 0)
            missing_working_mask = (
                (df["Working Gas Capacity"] == 0) & (df["Total Field Capacity"] > 0) & (df["Base Gas Capacity"] > 0)
            )

            if missing_working_mask.sum() > 0:
                df.loc[missing_working_mask, "Working Gas Capacity"] = (
                    df.loc[missing_working_mask, "Total Field Capacity"]
                    - df.loc[missing_working_mask, "Base Gas Capacity"]
                )

            # 3. Maximum Daily Delivery regression: ln(Max Daily) vs ln(Base Gas) and ln(Total Capacity)
            # Use rows where Base Gas, Total Field Capacity, and Maximum Daily Delivery are all > 0
            valid_delivery_mask = (
                (field_df["Base Gas Capacity"] > 0)
                & (field_df["Total Field Capacity"] > 0)
                & (field_df["Maximum Daily Delivery"] > 0)
            )

            if valid_delivery_mask.sum() > 2:  # Need at least 3 points for multiple regression
                valid_delivery_df = field_df[valid_delivery_mask]

                # Multiple linear regression: ln(Max Daily) vs ln(Base Gas) and ln(Total Capacity)
                X_delivery = np.column_stack(
                    [
                        np.log(valid_delivery_df["Base Gas Capacity"]),
                        np.log(valid_delivery_df["Total Field Capacity"]),
                    ]
                )
                y_delivery = np.log(valid_delivery_df["Maximum Daily Delivery"])

                delivery_model = LinearRegression()
                delivery_model.fit(X_delivery, y_delivery)

                # Fill missing Maximum Daily Delivery
                missing_delivery_mask = (
                    field_mask
                    & (df["Maximum Daily Delivery"] == 0)
                    & (df["Base Gas Capacity"] > 0)
                    & (df["Total Field Capacity"] > 0)
                )

                if missing_delivery_mask.sum() > 0:
                    missing_df = df[missing_delivery_mask]
                    X_pred = np.column_stack(
                        [
                            np.log(missing_df["Base Gas Capacity"]),
                            np.log(missing_df["Total Field Capacity"]),
                        ]
                    )
                    try:
                        predicted_ln_delivery = delivery_model.predict(X_pred)
                        predicted_delivery = np.exp(predicted_ln_delivery)
                        df.loc[missing_delivery_mask, "Maximum Daily Delivery"] = predicted_delivery
                    except:
                        logger.warning(f"Failed to predict Maximum Daily Delivery for field type {field_type}")

        return df

    def _filter_csv_by_interconnect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter CSV data by interconnect."""
        if self.interconnect == "usa":
            return df
        else:
            # Map state to interconnect
            state_col = "Report State"
            df["interconnect"] = df[state_col].map(self.state_2_interconnect)
            df = df[df["interconnect"] == self.interconnect]
            return df.drop(columns="interconnect")

    def filter_on_sate(self, n: pypsa.Network, df: pd.DataFrame) -> pd.DataFrame:
        """Filter formatted data to only include states in geographic scope."""
        states_in_model = n.buses[
            ~n.buses.carrier.isin(
                ["gas storage", "gas trade", "gas pipeline"],
            )
        ].reeds_state.unique()

        if self.agg_storage:
            # For aggregated data, filter by State column
            if "State" not in df.columns:
                logger.debug(
                    "Natural gas data not filtered due to incorrect data formatting",
                )
                return df
            df = df[df.State.isin(states_in_model)].copy()
        else:
            # For individual plant data, filter by Report State column
            state_col = "Report State"
            if state_col not in df.columns:
                possible_names = ["ReportState", "State"]
                for name in possible_names:
                    if name in df.columns:
                        state_col = name
                        break

            if state_col not in df.columns:
                logger.debug("Natural gas data not filtered due to incorrect data formatting")
                return df

            df = df[df[state_col].isin(states_in_model)].copy()

        return df

    def _get_data(
        self,
        retro_storage_type: list = None,
        h2_storage_costs_by_type: dict = None,
        acaes_storage_costs_by_type: dict = None,
    ) -> pd.DataFrame:
        """Override to pass retro_storage_type and h2_storage_costs_by_type to format_data."""
        data = self.read_data()
        return self.format_data(data, retro_storage_type, h2_storage_costs_by_type, acaes_storage_costs_by_type)

    def _get_h2_storage_costs_by_field_type(self, addi_costs: pd.DataFrame) -> dict:
        """
        Get H2 storage investment costs by field type.

        Returns
        -------
            dict: Mapping of field type to investment cost (USD/kWh_H2)
        """
        field_type_cost_mapping = {
            "Depleted Field": "hydrogen storage underground field",
            "Salt Dome": "hydrogen storage underground cavern",
            "Aquifer": "hydrogen storage underground aquifer",
        }

        costs_by_type = {}

        for field_type, cost_key in field_type_cost_mapping.items():
            costs_by_type[field_type] = addi_costs.loc[cost_key, "investment"]

        return costs_by_type

    def _get_acaes_storage_costs_by_field_type(self, addi_costs):
        """Get ACAES storage costs by field type."""
        acaes_storage_costs = {}

        for field_type in ["Depleted Field", "Salt Dome", "Aquifer"]:
            if field_type == "Depleted Field":
                cost_key = "ACAES_field"
            elif field_type == "Salt Dome":
                cost_key = "ACAES_cavern"
            elif field_type == "Aquifer":
                cost_key = "ACAES_aquifer"

            acaes_storage_costs[field_type] = addi_costs.loc[cost_key, "air_storage_cost"]

        return acaes_storage_costs

    def build_infrastructure(self, n: pypsa.Network, **kwargs):
        """Add pypsa components to network."""
        retro_storage_type = kwargs.get("retro_storage_type", [])
        retro_storage_h2 = kwargs.get("retro_storage_h2", False)
        retro_storage_acaes = kwargs.get("retro_storage_acaes", False)
        addi_costs = kwargs.get("addi_costs", None)

        # Get field-type-specific storage costs
        h2_storage_costs_by_type = self._get_h2_storage_costs_by_field_type(addi_costs)
        acaes_storage_costs_by_type = self._get_acaes_storage_costs_by_field_type(addi_costs)

        # Get formatted data with retrofit type separation and H2 costs
        df = self._get_data(retro_storage_type, h2_storage_costs_by_type, acaes_storage_costs_by_type)
        df = self.filter_on_sate(n, df)

        cyclic_storage = kwargs.get("cyclic_storage", True)
        storage_eng_retro_factor_h2 = kwargs.get("storage_eng_retro_factor_h2", 0.2551)
        storage_pow_retro_factor_h2 = kwargs.get("storage_pow_retro_factor_h2", 0.4)
        storage_eng_retro_factor_acaes = kwargs.get("storage_eng_retro_factor_acaes", 0.0093)

        # Get cost parameters from addi_costs
        gas_comp_elec_input = addi_costs.loc["natural gas storage compressor", "compression-electricity-input"]
        gas_comp_investment = addi_costs.loc["natural gas storage compressor", "investment"]
        gas_comp_fom = addi_costs.loc["natural gas storage compressor", "FOM"]
        gas_comp_lifetime = addi_costs.loc["natural gas storage compressor", "lifetime"]
        gas_com_currency_year = addi_costs.loc["natural gas storage compressor", "currency_year"]

        h2_comp_elec_input = addi_costs.loc["hydrogen storage compressor", "compression-electricity-input"]
        h2_comp_investment = addi_costs.loc["hydrogen storage compressor", "investment"]
        h2_comp_fom = addi_costs.loc["hydrogen storage compressor", "FOM"]
        h2_comp_lifetime = addi_costs.loc["hydrogen storage compressor", "lifetime"]
        h2_com_currency_year = addi_costs.loc["hydrogen storage compressor", "currency_year"]

        h2_storage_lifetime = addi_costs.loc["hydrogen storage underground", "lifetime"]
        h2_storage_currency_year = addi_costs.loc["hydrogen storage underground", "currency_year"]

        gas_link_capacity_cost = (
            gas_comp_fom
            * 0.01
            * gas_comp_investment
            * get_currency_conversion_factor(
                gas_com_currency_year,
                "EUR",
            )
            * 1e3
        )
        h2_link_capacity_cost = (
            (
                calculate_annuity(
                    h2_comp_lifetime,
                    discount_rate,
                )
                + h2_comp_fom * 0.01
            )
            * h2_comp_investment
            * get_currency_conversion_factor(
                h2_com_currency_year,
                "EUR",
            )
            * 1e3
        )

        acaes_charge = addi_costs.loc["ACAES", "charge_cost"]  # USD/MW_el
        acaes_charge_efficiency = addi_costs.loc["ACAES", "charge_efficiency"]
        acaes_discharge = addi_costs.loc["ACAES", "discharge_cost"]  # USD/MW
        acaes_discharge_efficiency = addi_costs.loc["ACAES", "discharge_efficiency"]
        acaes_tes_cost = addi_costs.loc["ACAES", "tes_cost"]  # USD/MWh, working energy
        acaes_FOM = addi_costs.loc["ACAES", "FOM"]
        acaes_VOM = addi_costs.loc["ACAES", "VOM"]
        acaes_loss = addi_costs.loc["ACAES", "loss"]
        acaes_currency_year = addi_costs.loc["ACAES", "currency_year"]
        acaes_lifetime = addi_costs.loc["ACAES", "lifetime"]

        if self.agg_storage > 0:  # Aggregated storage (level 1)
            # Separate retrofitable and non-retrofitable storage
            df_retro = df[df["retrofitable"]].copy()
            df_non_retro = df[~df["retrofitable"]].copy()

            # Build non-retrofitable gas storage
            if not df_non_retro.empty:
                self._build_aggregated_by_type_gas_storage(
                    n,
                    df_non_retro,
                    "gas storage nonretrofit",
                    cyclic_storage,
                    gas_comp_elec_input,
                    gas_link_capacity_cost,
                    gas_comp_lifetime,
                )

            # Build retrofitable gas storage
            if not df_retro.empty:
                self._build_aggregated_by_type_gas_storage(
                    n,
                    df_retro,
                    "gas storage retrofit",
                    cyclic_storage,
                    gas_comp_elec_input,
                    gas_link_capacity_cost,
                    gas_comp_lifetime,
                )

                # Build H2 storage for retrofitable facilities
                if retro_storage_h2:
                    self._build_aggregated_by_type_h2_storage(
                        n,
                        df_retro,
                        cyclic_storage,
                        storage_eng_retro_factor_h2,
                        storage_pow_retro_factor_h2,
                        h2_comp_elec_input,
                        h2_link_capacity_cost,
                        h2_comp_lifetime,
                        h2_storage_lifetime,
                        h2_storage_currency_year,
                    )
                # Build ACAES for retrofitable facilities
                if retro_storage_acaes:
                    self._build_aggregated_by_type_acaes_storage(
                        n,
                        df_retro,
                        cyclic_storage,
                        storage_eng_retro_factor_acaes,
                        acaes_charge,
                        acaes_charge_efficiency,
                        acaes_discharge,
                        acaes_discharge_efficiency,
                        acaes_tes_cost,
                        acaes_FOM,
                        acaes_VOM,
                        acaes_loss,
                        acaes_currency_year,
                        acaes_lifetime,
                    )
        else:  # Individual facility storage (agg_storage == 0)
            # [Original individual storage logic remains the same]
            df_non_retro = df[~df["retrofitable"]].copy()
            if not df_non_retro.empty:
                self._build_individual_gas_storage(
                    n,
                    df_non_retro,
                    cyclic_storage,
                    False,
                    gas_comp_elec_input,
                    gas_link_capacity_cost,
                    gas_comp_lifetime,
                )

            df_retro = df[df["retrofitable"]].copy()
            if not df_retro.empty:
                self._build_individual_gas_storage(
                    n,
                    df_retro,
                    cyclic_storage,
                    True,
                    gas_comp_elec_input,
                    gas_link_capacity_cost,
                    gas_comp_lifetime,
                )

                if retro_storage_h2:
                    self._build_individual_h2_storage(
                        n,
                        df_retro,
                        cyclic_storage,
                        storage_eng_retro_factor_h2,
                        storage_pow_retro_factor_h2,
                        h2_comp_elec_input,
                        h2_link_capacity_cost,
                        h2_comp_lifetime,
                        h2_storage_costs_by_type,
                        h2_storage_lifetime,
                        h2_storage_currency_year,
                    )

    def _build_aggregated_by_type_gas_storage(
        self,
        n: pypsa.Network,
        df: pd.DataFrame,
        carrier_base: str,
        cyclic_storage: bool,
        gas_comp_elec_input: float,
        gas_link_capacity_cost: float,
        gas_comp_lifetime: float,
    ):
        """Build gas storage aggregated by state and storage type (for agg_storage == 1)."""
        # Create index based on state and field type
        # Format: "STATE gas storage FIELD_TYPE retrofit/nonretrofit"
        suffix_type = "nonretrofit" if "non" in carrier_base else "retrofit"
        if "Field Type" in df.columns:
            df["index_name"] = df["State"] + " gas storage " + df["Field Type"] + " " + suffix_type
        else:
            df["index_name"] = df["State"] + " gas storage " + suffix_type

        df.index = df["index_name"]

        # Create carriers for each storage type
        if "Field Type" in df.columns:
            for field_type in df["Field Type"].unique():
                carrier_name = f"gas storage {field_type} {suffix_type}"
                if carrier_name not in n.carriers.index:
                    n.add("Carrier", carrier_name, color="#d35050", nice_name=f"Gas Storage {field_type} {suffix_type}")
                carriers = df["Field Type"].apply(lambda x: f"gas storage {x} {suffix_type}")
        else:
            carrier_name = f"gas storage {suffix_type}"
            if carrier_name not in n.carriers.index:
                n.add("Carrier", carrier_name, color="#d35050", nice_name=f"Gas Storage {suffix_type}")
            carriers = [carrier_name] * len(df)

        n.madd(
            "Bus",
            names=df.index,
            suffix="",  # Already included in index_name
            carrier=carriers,
            unit="MWh_th",
            interconnect=self.interconnect,
            country=df.State,
            reeds_state=df.State,
            STATE=df.State,
        )

        n.madd(
            "Store",
            names=df.index,
            suffix="",  # Already included in index_name
            bus=df.index,
            carrier=carriers,
            e_nom_extendable=True,
            e_nom=df["Total Field Capacity"],
            e_nom_min=0,
            e_nom_max=df["Total Field Capacity"],
            e_cyclic=cyclic_storage,
            e_min_pu=df["Base Gas Capacity"] / df["Total Field Capacity"],
            e_max_pu=1,
            capital_cost=0.01,
            standing_loss=8.0e-6,
            lifetime=np.inf,
            build_year=n.investment_periods[0],
        )

        # Charge and discharge links
        n.madd(
            "Link",
            names=df.index,
            suffix=" charge",
            carrier=carriers,
            bus0=df.State + " gas",
            bus1=df.index,
            bus2=df.State,
            efficiency=1,
            efficiency2=-gas_comp_elec_input,
            p_nom=df["Maximum Daily Delivery"] * 132 / 359,
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=True,
            p_nom_min=0,
            p_nom_max=df["Maximum Daily Delivery"] * 132 / 359,
            capital_cost=gas_link_capacity_cost,
            lifetime=np.inf,
            build_year=n.investment_periods[0],
        )

        n.madd(
            "Link",
            names=df.index,
            suffix=" discharge",
            carrier=carriers,
            bus0=df.index,
            bus1=df.State + " gas",
            p_nom=df["Maximum Daily Delivery"],
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=True,
            p_nom_min=0,
            p_nom_max=df["Maximum Daily Delivery"],
            capital_cost=0.1,
            lifetime=np.inf,
            build_year=n.investment_periods[0],
        )

    def _build_aggregated_by_type_h2_storage(
        self,
        n: pypsa.Network,
        df: pd.DataFrame,
        cyclic_storage: bool,
        storage_eng_retro_factor: float,
        storage_pow_retro_factor: float,
        h2_comp_elec_input: float,
        h2_link_capacity_cost: float,
        h2_comp_lifetime: float,
        h2_storage_lifetime: float,
        h2_storage_currency_year: int,
    ):
        """Build H2 storage aggregated by state and storage type (for agg_storage == 1)."""
        # Create index based on state and field type
        # Format: "STATE h2 storage FIELD_TYPE retrofit"
        if "Field Type" in df.columns:
            df["h2_index_name"] = df["State"] + " h2 storage " + df["Field Type"] + " retrofit"
        else:
            df["h2_index_name"] = df["State"] + " h2 storage retrofit"

        df.index = df["h2_index_name"]

        # Create carriers for each storage type
        if "h2 storage retrofit" not in n.carriers.index:
            n.add("Carrier", "h2 storage retrofit", color="#ea048a", nice_name="H2 Storage Retrofit")

        # Calculate capital cost using pre-calculated weighted investment costs
        h2_storage_capital_cost = (
            calculate_annuity(h2_storage_lifetime, discount_rate)
            * df["h2_weighted_investment_cost"]
            * 1e3
            * get_currency_conversion_factor(h2_storage_currency_year, "USD")
        )

        n.madd(
            "Bus",
            names=df.index,
            suffix="",  # Already included in h2_index_name
            carrier="h2 storage retrofit",
            unit="MWh_th",
            interconnect=self.interconnect,
            country=df.State,
            reeds_state=df.State,
            STATE=df.State,
        )

        n.madd(
            "Store",
            names=df.index,
            suffix="",  # Already included in h2_index_name
            bus=df.index,
            carrier="h2 storage retrofit",
            e_nom_extendable=True,
            e_nom=0,
            e_nom_min=0,
            e_nom_max=df["Total Field Capacity"] * storage_eng_retro_factor,
            e_cyclic=cyclic_storage,
            e_min_pu=df["Base Gas Capacity"] / df["Total Field Capacity"],
            e_max_pu=1,
            capital_cost=h2_storage_capital_cost,
            standing_loss=8.0e-6,
            lifetime=h2_storage_lifetime,
            build_year=n.investment_periods[0],
        )

        n.madd(
            "Link",
            names=df.index,
            suffix=" charge",
            carrier="h2 storage retrofit",
            bus0=df.State + " h2",
            bus1=df.index,
            bus2=df.State,
            efficiency=1,
            efficiency2=-h2_comp_elec_input,
            p_nom=0,
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=True,
            p_nom_min=0,
            # p_nom_max=df['Maximum Daily Delivery'] * 132 / 359 * storage_pow_retro_factor,
            capital_cost=h2_link_capacity_cost,
            lifetime=h2_comp_lifetime,
            build_year=n.investment_periods[0],
        )

        n.madd(
            "Link",
            names=df.index,
            suffix=" discharge",
            carrier="h2 storage retrofit",
            bus0=df.index,
            bus1=df.State + " h2",
            p_nom=0,
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=True,
            p_nom_min=0,
            p_nom_max=df["Maximum Daily Delivery"] * storage_pow_retro_factor,
            capital_cost=0,
            lifetime=h2_storage_lifetime,
            build_year=n.investment_periods[0],
        )

    def _build_aggregated_by_type_acaes_storage(
        self,
        n,
        df,
        cyclic_storage,
        storage_eng_retro_factor_acaes,
        acaes_charge,
        acaes_charge_efficiency,
        acaes_discharge,
        acaes_discharge_efficiency,
        acaes_tes_cost,
        acaes_FOM,
        acaes_VOM,
        acaes_loss,
        acaes_currency_year,
        acaes_lifetime,
    ):
        """Build aggregated ACAES storage by field type."""
        # Create index based on state and field type
        # Format: "STATE acaes FIELD_TYPE retrofit"
        if "Field Type" in df.columns:
            df["acaes_index_name"] = df["State"] + " acaes " + df["Field Type"] + " retrofit"
        else:
            df["acaes_index_name"] = df["State"] + " acaes retrofit"

        df.index = df["acaes_index_name"]

        # Create carriers for each storage type
        if "acaes retrofit" not in n.carriers.index:
            n.add("Carrier", "acaes retrofit", color="#ff69b4", nice_name="ACAES Retrofit")

        # Calculate capital cost using pre-calculated weighted investment costs
        annuity_acaes = calculate_annuity(acaes_lifetime, discount_rate)

        acaes_charge_capital_cost = (
            (annuity_acaes + acaes_FOM * 0.01)
            * acaes_charge
            * get_currency_conversion_factor(acaes_currency_year, "USD")
        )
        acaes_discharge_capital_cost = (
            (annuity_acaes + acaes_FOM * 0.01)
            * acaes_discharge
            * get_currency_conversion_factor(acaes_currency_year, "USD")
        )
        acaes_discharge_marginal_cost = acaes_VOM * 0.01 * acaes_discharge

        acaes_air_storage_capital_cost = (
            (annuity_acaes + acaes_FOM * 0.01)
            * df["acaes_weighted_investment_cost"]
            * get_currency_conversion_factor(acaes_currency_year, "USD")
        )
        acaes_tes_capital_cost = (
            (annuity_acaes + acaes_FOM * 0.01)
            * acaes_tes_cost
            * (df["Working Gas Capacity"] / df["Total Field Capacity"]).clip(upper=0.3)
            * get_currency_conversion_factor(acaes_currency_year, "USD")
        )
        acaes_storage_capital_cost = acaes_air_storage_capital_cost + acaes_tes_capital_cost

        n.madd(
            "Bus",
            names=df.index,
            carrier="acaes retrofit",
            unit="MWh_th",
            interconnect=self.interconnect,
            country=df.State,
            reeds_state=df.State,
            STATE=df.State,
        )

        n.madd(
            "Store",
            names=df.index,
            bus=df.index,
            carrier="acaes retrofit",
            e_nom_extendable=True,
            e_nom=0,
            e_nom_min=0,
            e_nom_max=df["Total Field Capacity"] * storage_eng_retro_factor_acaes,
            e_cyclic=cyclic_storage,
            e_min_pu=(df["Base Gas Capacity"] / df["Total Field Capacity"]).clip(lower=0.7),
            e_max_pu=1,
            capital_cost=acaes_storage_capital_cost,
            standing_loss=acaes_loss,
            lifetime=acaes_lifetime,
            build_year=n.investment_periods[0],
        )

        n.madd(
            "Link",
            names=df.index,
            suffix=" charge",
            carrier="acaes retrofit",
            bus0=df.State,
            bus1=df.index,
            efficiency=acaes_charge_efficiency,
            p_nom=0,
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=True,
            capital_cost=acaes_charge_capital_cost,
            lifetime=acaes_lifetime,
            build_year=n.investment_periods[0],
        )

        n.madd(
            "Link",
            names=df.index,
            suffix=" discharge",
            carrier="acaes retrofit",
            bus0=df.index,
            bus1=df.State,
            efficiency=acaes_discharge_efficiency,
            p_nom=0,
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=True,
            capital_cost=acaes_discharge_capital_cost,
            marginal_cost=acaes_discharge_marginal_cost,
            lifetime=acaes_lifetime,
            build_year=n.investment_periods[0],
        )

    def _build_individual_gas_storage(
        self,
        n: pypsa.Network,
        df: pd.DataFrame,
        cyclic_storage: bool,
        is_retrofitable: bool,
        gas_comp_elec_input: float,
        gas_link_capacity_cost: float,
        gas_comp_lifetime: float,
    ):
        """Build individual gas storage facilities."""
        # Create carrier for each storage type
        field_types = df["Field Type"].dropna().unique()

        for field_type in field_types:
            suffix = " retrofit" if is_retrofitable else " nonretrofit"
            carrier_name = f"gas storage {field_type}{suffix}"
            if carrier_name not in n.carriers.index:
                n.add("Carrier", carrier_name, color="#d35050", nice_name=f"Gas Storage-{field_type}")

        # Sort by state and storage type, then by Working Gas Capacity in ascending order
        # and assign index for each facility within each state-storage type group
        df_sorted = df.sort_values(["Report State", "Field Type", "Working Gas Capacity"])

        # Create index for each state-storage type combination
        df_sorted["facility_index"] = df_sorted.groupby(["Report State", "Field Type"]).cumcount() + 1

        def create_facility_name(row):
            state = row.get("Report State") or row.get("State", "Unknown")
            storage_type = str(row.get("Field Type", "Unknown"))
            index = row.get("facility_index", 1)
            suffix = " retrofit" if is_retrofitable else " nonretrofit"
            return f"{state} gas storage {storage_type}{suffix} {index}"

        df_sorted["facility_name"] = df_sorted.apply(create_facility_name, axis=1)
        df = df_sorted.set_index("facility_name")

        suffix = " retrofit" if is_retrofitable else " nonretrofit"
        n.madd(
            "Bus",
            names=df.index,
            carrier=[f"gas storage {field_type}{suffix}" for field_type in df["Field Type"]],
            unit="MWh_th",
            interconnect=self.interconnect,
            country=[row.get("Report State", row.get("State", "Unknown")) for _, row in df.iterrows()],
            reeds_state=df["Report State"],
            STATE=df["Report State"],
        )

        n.madd(
            "Store",
            names=df.index,
            bus=df.index,
            carrier=[f"gas storage {field_type}{suffix}" for field_type in df["Field Type"]],
            e_nom_extendable=True,
            e_nom=df["Total Field Capacity"],
            e_nom_min=0,
            e_nom_max=df["Total Field Capacity"],
            e_cyclic=cyclic_storage,
            e_min_pu=df["Base Gas Capacity"] / df["Total Field Capacity"],
            e_max_pu=1,
            capital_cost=0.1,  # 0.1 USD/MWh/a to prefer decommissioning to address degeneracy
            standing_loss=8.0e-6,  # 1%/month
            lifetime=np.inf,
            build_year=n.investment_periods[0],
        )

        # Charge links
        n.madd(
            "Link",
            names=df.index,
            suffix=" charge",
            carrier=[f"gas storage {field_type}{suffix}" for field_type in df["Field Type"]],
            bus0=[f"{state} gas" for state in df["Report State"]],  # state gas
            bus1=df.index,  # gas storage
            bus2=[f"{state}" for state in df["Report State"]],  # state AC - compression electricity
            efficiency=1,
            efficiency2=-gas_comp_elec_input,
            p_nom=df["Maximum Daily Delivery"] * 132 / 359,
            # ratio of max injection rate to withdraw rate, https://ir.eia.gov/ngs/ngs.html
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=True,
            p_nom_min=0,
            p_nom_max=df["Maximum Daily Delivery"] * 132 / 359,
            capital_cost=gas_link_capacity_cost,
            lifetime=np.inf,
            build_year=n.investment_periods[0],
        )

        # Discharge links
        n.madd(
            "Link",
            names=df.index,
            suffix=" discharge",
            carrier=[f"gas storage {field_type}{suffix}" for field_type in df["Field Type"]],
            bus0=df.index,
            bus1=[f"{state} gas" for state in df["Report State"]],
            p_nom=df["Maximum Daily Delivery"],
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=True,
            p_nom_min=0,
            p_nom_max=df["Maximum Daily Delivery"],
            capital_cost=1,  # 1 USD/MW/a to prefer decommissioning to address degeneracy
            lifetime=np.inf,
            build_year=n.investment_periods[0],
        )

    def _build_individual_h2_storage(
        self,
        n: pypsa.Network,
        df: pd.DataFrame,
        cyclic_storage: bool,
        storage_eng_retro_factor: float,
        storage_pow_retro_factor: float,
        h2_comp_elec_input: float,
        h2_link_capacity_cost: float,
        h2_comp_lifetime: float,
        h2_storage_costs_by_type: dict,
        h2_storage_lifetime: float,
        h2_storage_currency_year: int,
    ):
        """Build individual H2 storage facilities with field-type-specific costs."""
        field_types = df["Field Type"].dropna().unique()

        # Create hydrogen storage carriers for each field type
        for field_type in field_types:
            suffix = " retrofit"
            h2_carrier_name = f"h2 storage {field_type}{suffix}"
            if h2_carrier_name not in n.carriers.index:
                n.add("Carrier", h2_carrier_name, color="#ea048a", nice_name=f"H2 Storage {field_type}")

        # Sort and create facility names (existing logic)
        df_sorted = df.sort_values(["Report State", "Field Type", "Working Gas Capacity"])
        df_sorted["facility_index"] = df_sorted.groupby(["Report State", "Field Type"]).cumcount() + 1

        def create_h2_facility_name(row):
            state = row.get("Report State") or row.get("State", "Unknown")
            storage_type = str(row.get("Field Type", "Unknown"))
            index = row.get("facility_index", 1)
            suffix = " retrofit"
            return f"{state} h2 storage {storage_type}{suffix} {index}"

        df_sorted["h2_facility_name"] = df_sorted.apply(create_h2_facility_name, axis=1)
        df_h2 = df_sorted.copy()
        df_h2.index = df_h2["h2_facility_name"]

        # Calculate field-type-specific capital costs
        def get_field_type_capital_cost(field_type):
            investment_cost = h2_storage_costs_by_type.get(
                field_type,
                h2_storage_costs_by_type.get("Depleted Field"),
            )
            return (
                calculate_annuity(h2_storage_lifetime, discount_rate)
                * investment_cost
                * 1e3
                * get_currency_conversion_factor(h2_storage_currency_year, "USD")
            )

        df_h2["h2_storage_capital_cost"] = df_h2["Field Type"].apply(get_field_type_capital_cost)

        # Add hydrogen storage buses (existing logic)
        suffix = " retrofit"
        n.madd(
            "Bus",
            names=df_h2.index,
            carrier=[f"h2 storage {field_type}{suffix}" for field_type in df_h2["Field Type"]],
            unit="MWh_th",
            interconnect=self.interconnect,
            country=[row.get("Report State", row.get("State", "Unknown")) for _, row in df_h2.iterrows()],
            reeds_state=df_h2["Report State"],
            STATE=df["Report State"],
        )

        # Add hydrogen storage stores with field-type-specific costs
        n.madd(
            "Store",
            names=df_h2.index,
            bus=df_h2.index,
            carrier=[f"h2 storage {field_type}{suffix}" for field_type in df_h2["Field Type"]],
            e_nom_extendable=True,
            e_nom=0,
            e_nom_min=0,
            e_nom_max=df_h2["Total Field Capacity"] * storage_eng_retro_factor,
            e_cyclic=cyclic_storage,
            e_min_pu=df_h2["Base Gas Capacity"] / df_h2["Total Field Capacity"],
            e_max_pu=1,
            capital_cost=df_h2["h2_storage_capital_cost"],  # Use field-type-specific costs
            standing_loss=8.0e-6,  # 1%/month
            lifetime=h2_storage_lifetime,
            build_year=n.investment_periods[0],
        )

        # Add charge and discharge links (existing logic with minor adjustments)
        n.madd(
            "Link",
            names=df_h2.index,
            suffix=" charge",
            carrier=[f"h2 storage {field_type}{suffix}" for field_type in df_h2["Field Type"]],
            bus0=[f"{state} h2" for state in df_h2["Report State"]],
            bus1=df_h2.index,
            bus2=[f"{state}" for state in df_h2["Report State"]],
            efficiency=1,
            efficiency2=-h2_comp_elec_input,
            p_nom=0,
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=True,
            p_nom_min=0,
            # p_nom_max=df_h2['Maximum Daily Delivery'] * 132 / 359 * storage_pow_retro_factor,
            capital_cost=h2_link_capacity_cost,
            lifetime=h2_comp_lifetime,
            build_year=n.investment_periods[0],
        )

        n.madd(
            "Link",
            names=df_h2.index,
            suffix=" discharge",
            carrier=[f"h2 storage {field_type}{suffix}" for field_type in df_h2["Field Type"]],
            bus0=df_h2.index,
            bus1=[f"{state} h2" for state in df_h2["Report State"]],
            p_nom=0,
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=True,
            p_nom_min=0,
            p_nom_max=df_h2["Maximum Daily Delivery"] * storage_pow_retro_factor,
            capital_cost=0,
            lifetime=np.inf,
            build_year=n.investment_periods[0],
        )


class GasProcessing(GasData):
    """Creator for processing capacity."""

    def __init__(self, year: int, interconnect: str, api: str) -> None:
        self.api = api
        super().__init__(year=year, interconnect=interconnect)

    def read_data(self) -> pd.DataFrame:
        """Read in data from EIA API."""
        return eia.Production("gas", "market", self.year, self.api).get_data()

    def format_data(self, data: pd.DataFrame):
        """Format processing capacity data."""
        df = data.copy()

        df["value"] = (
            df.value.astype(float) * MWH_2_MMCF / 30 / 24
        )  # get monthly average hourly capacity (based on 30 days / month)
        df = (
            df.reset_index()
            .drop(columns=["period", "series-description", "units"])  # units in MW_th
            .groupby(["state"])
            .max()  # get average yearly capacity
            .reset_index()
            .rename(columns={"state": "STATE", "value": "p_nom"})
        )
        return self.filter_on_interconnect(df, ["U.S."])

    def filter_on_sate(
        self,
        n: pypsa.Network,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Filter formatted data to only include states in geographic scope."""
        states_in_model = n.buses[
            ~n.buses.carrier.isin(
                ["gas storage", "gas trade", "gas pipeline"],
            )
        ].reeds_state.unique()

        if "STATE" not in df.columns:
            logger.debug(
                "Natual gas data not filtered due to incorrect data formatting",
            )
            return df

        df = df[df.STATE.isin(states_in_model)].copy()

        return df

    def build_infrastructure(self, n: pypsa.Network, **kwargs):
        """Add pypsa components to network."""
        cost_factor = kwargs.get("cost_factor", 1.0)
        eia_data = pd.read_csv("repo_data/EIA_AEO_2025_Table_59.csv", index_col=0)
        eia_data = eia_data.rename_axis("region")

        cost_year = "2050"
        gas_costs_mcf = eia_data[[cost_year]].copy()

        gas_costs_mwh = gas_costs_mcf[cost_year] / (NG_MWH_2_MMCF / 1000) / 1.07  # 2024 -> 2022 USD
        gas_costs = gas_costs_mwh.to_frame(name="marginal_cost")

        state_to_region = {
            "ME": "East",
            "NH": "East",
            "VT": "East",
            "MA": "East",
            "RI": "East",
            "CT": "East",
            "NY": "East",
            "NJ": "East",
            "PA": "East",
            "DE": "East",
            "MD": "East",
            "WV": "East",
            "VA": "East",
            "NC": "East",
            "SC": "East",
            "GA": "East",
            "FL": "East",
            "AL": "Gulf Coast",
            "MS": "Gulf Coast",
            "LA": "Gulf Coast",
            "TX": "Gulf Coast",
            "AR": "Midcontinent",
            "OK": "Midcontinent",
            "KS": "Midcontinent",
            "NE": "Midcontinent",
            "SD": "Northern Great Plains",
            "ND": "Northern Great Plains",
            "MT": "Rocky Mountain",
            "WY": "Rocky Mountain",
            "CO": "Rocky Mountain",
            "NM": "Southwest",
            "AZ": "Southwest",
            "UT": "Rocky Mountain",
            "ID": "Rocky Mountain",
            "OH": "East",
            "KY": "East",
            "TN": "East",
            "IN": "Midcontinent",
            "IL": "Midcontinent",
            "MO": "Midcontinent",
            "IA": "Midcontinent",
            "MN": "Midcontinent",
            "WI": "Midcontinent",
            "MI": "Midcontinent",
            "WA": "West Coast",
            "OR": "West Coast",
            "CA": "West Coast",
            "NV": "Southwest",
        }

        df = self.filter_on_sate(n, self.data)
        df = df[df["p_nom"] > 1]
        df = df.set_index("STATE")
        df["bus"] = df.index + " gas"

        df["region"] = df.index.map(state_to_region)

        df = df.join(gas_costs, on="region")

        capacity_mult = kwargs.get("capacity_multiplier", 1)
        p_nom_extendable = True
        p_nom_mult = 1 if capacity_mult >= 1 else capacity_mult
        p_nom_max_mult = capacity_mult

        if "gas production" not in n.carriers.index:
            n.add("Carrier", "gas production", color="#d35050", nice_name="Gas Production")

        n.madd(
            "Bus",
            names=df.index,
            suffix=" gas production",
            carrier="gas production",
            unit="MWh_th",
            country=df.index,
            interconnect=self.interconnect,
            STATE=df.index,
        )

        n.madd(
            "Link",
            names=df.index,
            suffix=" gas production",
            carrier="gas production",
            unit="MW",
            bus0=df.index + " gas production",
            bus1=df.index + " gas",
            bus2=df.index + " co2 atmosphere",
            efficiency=1,
            efficiency2=leakage_rate * 0.072 * 28,  # 1 MWh ng = 0.072 ton, GWP = 28
            p_nom_extendable=p_nom_extendable,
            capital_cost=(525000 * 1.23 / (NG_MWH_2_MMCF / 24) * 0.03 + 2357)
            * HHV_to_LHV_CH4,  # https://ingaa.org/wp-content/uploads/2016/04/27962.pdf, https://www.aer.ca/data-and-performance-reports/statistical-reports/alberta-energy-outlook-st98/natural-gas/natural-gas-supply-costs
            marginal_cost=df.marginal_cost * cost_factor * HHV_to_LHV_CH4,
            p_nom=df.p_nom * p_nom_mult,
            p_nom_min=0,
            p_nom_max=df.p_nom * p_nom_max_mult,
            lifetime=np.inf,
            build_year=n.investment_periods[0],
        )

        gas_prod_links = n.links[n.links.carrier == "gas production"].copy()
        bus_to_state = n.buses.STATE
        gas_prod_links.index = gas_prod_links.bus1.map(bus_to_state)
        e_nom_max_values = gas_prod_links.p_nom * 8760
        n.madd(
            "Store",
            names=df.index,
            unit="MWh",
            suffix=" gas production",
            bus=df.index + " gas production",
            carrier="gas production",
            capital_cost=0,
            marginal_cost=0,
            e_cyclic=False,
            e_cyclic_per_period=False,
            e_nom=0,
            e_nom_extendable=True,
            e_nom_min=0,
            e_nom_max=e_nom_max_values,
            e_min_pu=-1,
            e_max_pu=0,
            lifetime=np.inf,
            build_year=n.investment_periods[0],
        )


class _GasPipelineCapacity(GasData):
    def __init__(
        self,
        year: int,
        interconnect: str,
        xlsx: str,
        api: str | None = None,
    ) -> None:
        self.xlsx = xlsx
        self.api = api
        super().__init__(year, interconnect)

    def read_data(self) -> pd.DataFrame:
        """Read in excel dataset."""
        return pd.read_excel(
            self.xlsx,
            sheet_name="Pipeline State2State Capacity",
            skiprows=1,
            index_col=0,
        )

    def get_states_in_model(self, n: pypsa.Network) -> list[str]:
        return n.buses[
            ~n.buses.carrier.isin(
                ["gas storage", "gas trade", "gas pipeline"],
            )
        ].reeds_state.unique()

    def filter_on_sate(
        self,
        n: pypsa.Network,
        df: pd.DataFrame,
        in_spatial_scope: bool,
    ) -> pd.DataFrame:
        states_in_model = self.get_states_in_model(n)

        if ("STATE_TO" and "STATE_FROM") not in df.columns:
            logger.debug(
                "Natual gas data not filtered due to incorrect data formatting",
            )
            return df

        if in_spatial_scope:
            df = df[(df.STATE_TO.isin(states_in_model)) & (df.STATE_FROM.isin(states_in_model))].copy()
        else:
            df = df[(df.STATE_TO.isin(states_in_model)) | (df.STATE_FROM.isin(states_in_model))].copy()

        return df[~(df.STATE_TO == df.STATE_FROM)].copy()

    def format_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Format pipeline data."""
        df = data.copy()
        df.columns = df.columns.str.strip()
        df = df[df.index == int(self.year)]
        df["Capacity (mmcfd)"] = df["Capacity (mmcfd)"] * MWH_2_MMCF / 24  # divide by 24 to get hourly
        df = df.rename(
            columns={
                "State From": "STATE_NAME_FROM",
                "County From": "COUNTRY_FROM",
                "State To": "STATE_NAME_TO",
                "County To": "COUNTRY_TO",
                "Capacity (mmcfd)": "CAPACITY_MW",
            },
        )
        df = (
            df.astype(
                {
                    "STATE_NAME_FROM": "str",
                    "COUNTRY_FROM": "str",
                    "STATE_NAME_TO": "str",
                    "COUNTRY_TO": "str",
                    "CAPACITY_MW": "float",
                },
            )[["STATE_NAME_FROM", "STATE_NAME_TO", "CAPACITY_MW"]]
            .groupby(["STATE_NAME_TO", "STATE_NAME_FROM"])
            .sum()
            .reset_index()
        )

        df = df[
            ~(
                (
                    df.STATE_NAME_TO.isin(
                        ["Gulf of Mexico", "Gulf of Mexico - Deepwater"],
                    )
                )
                | (
                    df.STATE_NAME_FROM.isin(
                        ["Gulf of Mexico", "Gulf of Mexico - Deepwater"],
                    )
                )
            )
        ]

        df = self.assign_pipeline_interconnects(df)

        if self.api:
            trade = self._get_capacity_based_on_trade_flows()
            df = self._merge_capacity_trade_data(df, trade, self.agg_pipeline)

        # # slight buffer for when building constraints
        # df["CAPACITY_MW"] = np.ceil(df["CAPACITY_MW"].mul(1.03))

        return self.extract_pipelines(df)

    def _get_capacity_based_on_trade_flows(self) -> pd.DataFrame:
        """Check that trade flows do not exceed design capacity.

        See Issue #487
        https://github.com/PyPSA/pypsa-usa/issues/487
        """
        df = pd.concat(
            [
                eia.Trade("gas", False, "exports", self.year, self.api).get_data(),
                eia.Trade("gas", True, "exports", self.year, self.api).get_data(),
            ],
        )
        df["STATE_FROM"] = df.state.map(lambda x: x.split("-")[0])
        df["STATE_TO"] = df.state.map(lambda x: x.split("-")[1])
        df["CAPACITY_MW"] = df.value.mul(MWH_2_MMCF).div(365).div(24)  # MMCF/year -> MW
        df = df.reset_index(drop=True).drop(
            columns=["series-description", "value", "units", "state"],
        )
        df["STATE_NAME_TO"] = df.STATE_TO.map(self.state_2_name)
        df["STATE_NAME_FROM"] = df.STATE_FROM.map(self.state_2_name)
        df["INTERCONNECT_TO"] = df.STATE_TO.map(self.state_2_interconnect)
        df["INTERCONNECT_FROM"] = df.STATE_FROM.map(self.state_2_interconnect)

        return df

    def _merge_capacity_trade_data(
        self,
        capacity: pd.DataFrame,
        trade: pd.DataFrame,
        agg_pipeline: bool,
    ) -> pd.DataFrame:
        """
        Merge capacity and trade data with different logic based on agg_pipeline setting.

        When agg_pipeline=True: Uses original logic (keep highest capacity per state pair)
        When agg_pipeline=False: Enhanced logic for individual pipelines:
            - Add missing state pairs from trade data
            - Scale existing pipelines if total capacity < trade capacity
        """
        capacity = capacity[capacity["CAPACITY_MW"] > 1]
        trade = trade[trade["CAPACITY_MW"] > 1]

        if agg_pipeline:
            # Original logic for aggregated pipelines
            df = pd.concat([capacity, trade])
            df = df.sort_values(by="CAPACITY_MW", ascending=False)
            df = df.drop_duplicates(
                subset=[
                    "STATE_NAME_TO",
                    "STATE_NAME_FROM",
                    "STATE_TO",
                    "STATE_FROM",
                    "INTERCONNECT_TO",
                    "INTERCONNECT_FROM",
                ],
                keep="first",
            )
            return df.sort_values("STATE_NAME_TO")

        else:
            # Create working copies
            capacity_copy = capacity.copy()
            trade_copy = trade.copy()

            # Create directional state pair identifiers (maintain direction: FROM -> TO)
            def create_directional_pair(row):
                return (row["STATE_NAME_FROM"], row["STATE_NAME_TO"])

            capacity_copy["directional_pair"] = capacity_copy.apply(create_directional_pair, axis=1)
            trade_copy["directional_pair"] = trade_copy.apply(create_directional_pair, axis=1)

            # Get directional state pairs in capacity and trade data
            capacity_directional_pairs = set(capacity_copy["directional_pair"])
            trade_directional_pairs = set(trade_copy["directional_pair"])

            # Directional pairs that exist only in trade data (need to add directly)
            missing_directional_pairs = trade_directional_pairs - capacity_directional_pairs

            # Directional pairs that exist in both (need to check and potentially scale)
            common_directional_pairs = capacity_directional_pairs.intersection(trade_directional_pairs)

            # Start with the original capacity data
            result_df = capacity_copy.copy()

            # Add missing directional pairs from trade data
            if missing_directional_pairs:
                missing_rows = trade_copy[trade_copy["directional_pair"].isin(missing_directional_pairs)]
                result_df = pd.concat([result_df, missing_rows], ignore_index=True)
                logger.info(f"Added {len(missing_rows)} pipeline records from trade data for missing directional pairs")

            # Scale existing directional pairs if needed
            scaling_applied = []
            for directional_pair in common_directional_pairs:
                # Get capacity total for this directional pair (FROM -> TO)
                capacity_mask = result_df["directional_pair"] == directional_pair
                capacity_rows = result_df[capacity_mask]
                capacity_total = capacity_rows["CAPACITY_MW"].sum()

                # Get trade total for this directional pair (FROM -> TO)
                trade_mask = trade_copy["directional_pair"] == directional_pair
                trade_rows = trade_copy[trade_mask]
                trade_total = trade_rows["CAPACITY_MW"].sum()

                # If capacity total < trade total, apply scaling factor
                if capacity_total < trade_total:
                    scaling_factor = trade_total / capacity_total
                    # Update the capacity values in result_df for this directional pair
                    result_df.loc[capacity_mask, "CAPACITY_MW"] *= scaling_factor
                    scaling_applied.append(
                        {
                            "directional_pair": directional_pair,
                            "original_total": capacity_total,
                            "trade_total": trade_total,
                            "scaling_factor": scaling_factor,
                            "pipelines_affected": capacity_mask.sum(),
                        }
                    )

            if scaling_applied:
                logger.info(f"Applied scaling to {len(scaling_applied)} directional pairs based on trade data")

            # Remove the temporary directional_pair column
            result_df = result_df.drop(columns=["directional_pair"])

            return result_df.sort_values("STATE_NAME_TO").reset_index(drop=True)

    @abstractmethod
    def build_infrastructure(self, n: pypsa.Network) -> None:
        pass

    @abstractmethod
    def extract_pipelines(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts pipelines for that region.

        Used in Format data
        """
        pass

    def assign_pipeline_interconnects(self, df: pd.DataFrame):
        """Adds interconnect labels to the pipelines."""
        df["STATE_TO"] = df.STATE_NAME_TO.map(self.name_2_state)
        df["STATE_FROM"] = df.STATE_NAME_FROM.map(self.name_2_state)

        df["INTERCONNECT_TO"] = df.STATE_TO.map(self.state_2_interconnect)
        df["INTERCONNECT_FROM"] = df.STATE_FROM.map(self.state_2_interconnect)

        assert not df.isna().any().any()

        return df


class InterconnectGasPipelineCapacity(_GasPipelineCapacity):
    """Pipeline capacity within the interconnect."""

    def __init__(
        self,
        year: int,
        interconnect: str,
        xlsx: str,
        api: str | None = None,
        agg_pipeline: bool = True,
    ) -> None:
        self.agg_pipeline = agg_pipeline
        super().__init__(year, interconnect, xlsx, api)

    def _calculate_distances(self, df: pd.DataFrame, n: pypsa.Network) -> pd.Series:
        """
        Calculate distances between state pairs.

        Args:
            df: Pipeline data with STATE_FROM and STATE_TO columns
            n: PyPSA network containing bus coordinates

        Returns
        -------
            Series of distances in kilometers indexed by pipeline names
        """
        # Get gas bus coordinates from the network
        gas_buses = n.buses[n.buses.carrier == "gas"].copy()
        state_coords = gas_buses.set_index("STATE")[["x", "y"]]

        # Calculate distances between state pairs using haversine formula
        distances = []
        for _, row in df.iterrows():
            state_from = row["STATE_FROM"]
            state_to = row["STATE_TO"]

            if state_from in state_coords.index and state_to in state_coords.index:
                # Get coordinates for both states (longitude, latitude)
                lon1, lat1 = state_coords.loc[state_from, ["x", "y"]]
                lon2, lat2 = state_coords.loc[state_to, ["x", "y"]]

                # Calculate distance using haversine formula
                distance_km = pypsa.geo.haversine([lon1, lat1], [lon2, lat2]).item()
                distances.append(distance_km)
            else:
                logger.warning(f"State coordinates not found for {state_from} or {state_to}")
                distances.append(0)

        return pd.Series(distances, index=df.index)

    def _calculate_capital_costs(
        self,
        df: pd.DataFrame,
        n: pypsa.Network,
        fom: float,
        investment: float,
        lifetime: float,
        length_factor: float,
        currency_year: int,
    ) -> pd.Series:
        """
        Calculate capital costs for gas pipelines based on distance and cost parameters.
        Capital cost = distance × (annualized_investment + fom) × EUR_2_USD (in $/MW)
        """
        distances = self._calculate_distances(df, n) * length_factor

        if lifetime == np.inf:
            annualized_investment = 0
        else:
            annualized_investment = calculate_annuity(lifetime, discount_rate)

        # Calculate total annualized cost (investment + FOM)
        conversion_factor = get_currency_conversion_factor(currency_year, "EUR")
        total_annual_cost = (annualized_investment + fom * 0.01) * investment * conversion_factor
        capital_costs = distances * total_annual_cost
        return capital_costs

    def _calculate_efficiency(
        self, df: pd.DataFrame, n: pypsa.Network, elec_input: float, length_factor: float, gas_type: str
    ) -> pd.Series:
        """
        Calculate transport efficiency
        """
        distances = self._calculate_distances(df, n) * length_factor

        # Leakage + Compression
        # Leakage rate: https://www.nature.com/articles/s41560-025-01752-6#Sec14
        # Compression drive efficiency: https://netl.doe.gov/sites/default/files/netl-file/Brun.pdf
        efficiency = (
            1 - 0.01 * distances / 1000 - elec_input * distances / 1000 / 0.325
            if gas_type == "gas"
            else 1 - 0.012 * distances / 1000 - elec_input * distances / 1000 / 0.325
        )

        return efficiency

    def format_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Format pipeline data."""
        df = data.copy()
        df.columns = df.columns.str.strip()
        df = df[df.index == int(self.year)]
        df["Capacity (mmcfd)"] = df["Capacity (mmcfd)"] * MWH_2_MMCF / 24  # divide by 24 to get hourly
        df = df.rename(
            columns={
                "State From": "STATE_NAME_FROM",
                "County From": "COUNTRY_FROM",
                "State To": "STATE_NAME_TO",
                "County To": "COUNTRY_TO",
                "Capacity (mmcfd)": "CAPACITY_MW",
            },
        )

        # Convert to appropriate data types
        df = df.astype(
            {
                "STATE_NAME_FROM": "str",
                "COUNTRY_FROM": "str",
                "STATE_NAME_TO": "str",
                "COUNTRY_TO": "str",
                "CAPACITY_MW": "float",
            },
        )[["STATE_NAME_FROM", "STATE_NAME_TO", "CAPACITY_MW"]]

        # Conditional aggregation based on agg_pipeline parameter
        if self.agg_pipeline:
            # Original aggregation logic
            df = df.groupby(["STATE_NAME_TO", "STATE_NAME_FROM"]).sum().reset_index()
        # If not aggregating, keep individual pipeline records

        # Filter out Gulf of Mexico connections
        df = df[
            ~(
                (
                    df.STATE_NAME_TO.isin(
                        ["Gulf of Mexico", "Gulf of Mexico - Deepwater"],
                    )
                )
                | (
                    df.STATE_NAME_FROM.isin(
                        ["Gulf of Mexico", "Gulf of Mexico - Deepwater"],
                    )
                )
            )
        ]

        df = self.assign_pipeline_interconnects(df)

        # Only use trade data for capacity correction when aggregating
        if self.api:
            trade = self._get_capacity_based_on_trade_flows()
            df = self._merge_capacity_trade_data(df, trade, self.agg_pipeline)
            # # Apply capacity buffer
            # df["CAPACITY_MW"] = np.ceil(df["CAPACITY_MW"].mul(1.03))

        return self.extract_pipelines(df)

    def extract_pipelines(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get piplines within the geographic scope."""
        df = data.copy()
        # for some reason drop duplicates is not wokring here and I cant figure out why :(
        # df = df.drop_duplicates(subset=["STATE_TO", "STATE_FROM"], keep=False).copy()
        df = df[~df.apply(lambda x: x.STATE_TO == x.STATE_FROM, axis=1)].copy()

        if self.interconnect != "usa":
            df = df[(df.INTERCONNECT_TO == self.interconnect) & (df.INTERCONNECT_FROM == self.interconnect)]
            if df.empty:
                logger.error(
                    f"Empty natural gas domestic pipelines for interconnect {self.interconnect}",
                )
        else:
            df = df[
                ~(
                    df[["INTERCONNECT_TO", "INTERCONNECT_FROM"]].isin(
                        ["canada", "mexico"],
                    )
                ).all(axis=1)
            ]

        df = df[df["CAPACITY_MW"] > 1]

        return df.reset_index(drop=True)

    def build_infrastructure(self, n: pypsa.Network, **kwargs) -> None:
        """Add pypsa components to network."""
        df = self.filter_on_sate(n, self.data, in_spatial_scope=True)
        retro_pipeline = kwargs.get("retro_pipeline", False)
        addi_costs = kwargs.get("addi_costs", None)
        pipeline_retro_factor = kwargs.get("pipeline_retro_factor", 0.6)
        length_factor = kwargs.get("length_factor", 1.0)

        gas_fom = addi_costs.loc["CH4 (g) pipeline", "FOM"]
        gas_investment = addi_costs.loc["CH4 (g) pipeline", "investment"]  # EUR/MW/km
        gas_lifetime = np.inf
        gas_elec_input = addi_costs.loc["CH4 (g) pipeline", "electricity-input"]  # MW_e/1000km/MW_CH4
        gas_currency_year = addi_costs.loc["CH4 (g) pipeline", "currency_year"]

        h2_fom = addi_costs.loc["H2 (g) pipeline repurposed", "FOM"]
        h2_investment = addi_costs.loc["H2 (g) pipeline repurposed", "investment"]  # EUR/MW/km
        h2_lifetime = addi_costs.loc["H2 (g) pipeline repurposed", "lifetime"]
        h2_elec_input = addi_costs.loc["H2 (g) pipeline repurposed", "electricity-input"]  # MW_e/1000km/MW_H2
        h2_currency_year = addi_costs.loc["H2 (g) pipeline repurposed", "currency_year"]

        if df.empty:
            # happens for single state models
            logger.info("No gas pipelines added within interconnect")
            return

        if "gas pipeline" not in n.carriers.index:
            n.add("Carrier", "gas pipeline", color="#d35050", nice_name="Gas Pipeline")

        if self.agg_pipeline:
            df.index = df.STATE_FROM + " " + df.STATE_TO

            distances = self._calculate_distances(df, n) * length_factor
            # Calculate capital costs and electricity efficiency for gas pipelines
            gas_capital_costs = self._calculate_capital_costs(
                df,
                n,
                gas_fom,
                gas_investment,
                gas_lifetime,
                length_factor,
                gas_currency_year,
            )
            efficiency = self._calculate_efficiency(df, n, gas_elec_input, length_factor, "gas")

            n.madd(
                "Link",
                names=df.index,
                suffix=" gas pipeline",
                carrier="gas pipeline",
                unit="MW",
                bus0=df.STATE_FROM + " gas",
                bus1=df.STATE_TO + " gas",
                efficiency=efficiency,
                p_nom=df.CAPACITY_MW,
                p_min_pu=0,
                p_max_pu=1,
                p_nom_extendable=True,
                p_nom_min=0,
                p_nom_max=df.CAPACITY_MW,
                capital_cost=gas_capital_costs,
                length=distances,
                lifetime=np.inf,
                build_year=n.investment_periods[0],
            )

            # Add H2 pipeline infrastructure if retro_pipeline is enabled
            if retro_pipeline:
                if "h2 pipeline retrofit" not in n.carriers.index:
                    n.add("Carrier", "h2 pipeline retrofit", color="#ea048a", nice_name="H2 Pipeline Retrofit")

                # Calculate capital costs and electricity efficiency for H2 pipelines
                h2_capital_costs = self._calculate_capital_costs(
                    df,
                    n,
                    h2_fom,
                    h2_investment,
                    h2_lifetime,
                    length_factor,
                    h2_currency_year,
                )
                efficiency = self._calculate_efficiency(df, n, h2_elec_input, length_factor, "h2")

                n.madd(
                    "Link",
                    names=df.index,
                    suffix=" h2 pipeline retrofit_fwd",
                    carrier="h2 pipeline retrofit",
                    unit="MW",
                    bus0=df.STATE_FROM + " h2",
                    bus1=df.STATE_TO + " h2",
                    efficiency=efficiency,
                    p_nom=0,
                    p_min_pu=0,
                    p_max_pu=1,
                    p_nom_extendable=True,
                    p_nom_min=0,
                    p_nom_max=df.CAPACITY_MW * pipeline_retro_factor,
                    capital_cost=h2_capital_costs / 2,
                    length=distances,
                    lifetime=h2_lifetime,
                    build_year=n.investment_periods[0],
                )

                n.madd(
                    "Link",
                    names=df.index,
                    suffix=" h2 pipeline retrofit_rev",
                    carrier="h2 pipeline retrofit",
                    unit="MW",
                    bus0=df.STATE_TO + " h2",
                    bus1=df.STATE_FROM + " h2",
                    efficiency=efficiency,
                    p_nom=0,
                    p_min_pu=0,
                    p_max_pu=1,
                    p_nom_extendable=True,
                    p_nom_min=0,
                    p_nom_max=df.CAPACITY_MW * pipeline_retro_factor,
                    capital_cost=h2_capital_costs / 2,
                    length=distances,
                    lifetime=h2_lifetime,
                    build_year=n.investment_periods[0],
                )
        else:
            # Individual pipeline logic with proper naming
            # Sort by state pair and capacity for consistent naming
            df_sorted = df.sort_values(["STATE_FROM", "STATE_TO", "CAPACITY_MW"], ascending=[True, True, False])

            # Create index for each pipeline within each state pair
            df_sorted["pipeline_index"] = df_sorted.groupby(["STATE_FROM", "STATE_TO"]).cumcount() + 1

            def create_pipeline_name(row):
                state_from = row["STATE_FROM"]
                state_to = row["STATE_TO"]
                index = row["pipeline_index"]
                return f"{state_from} {state_to} gas pipeline-{index}"

            df_sorted["pipeline_name"] = df_sorted.apply(create_pipeline_name, axis=1)
            df_sorted = df_sorted.set_index("pipeline_name")

            distances = self._calculate_distances(df_sorted, n) * length_factor
            # Calculate capital costs and electricity efficiency for individual gas pipelines
            gas_capital_costs = self._calculate_capital_costs(
                df_sorted,
                n,
                gas_fom,
                gas_investment,
                gas_lifetime,
                length_factor,
                gas_currency_year,
            )
            efficiency = self._calculate_efficiency(df_sorted, n, gas_elec_input, length_factor, "gas")

            n.madd(
                "Link",
                names=df_sorted.index,
                carrier="gas pipeline",
                unit="MW",
                bus0=df_sorted.STATE_FROM + " gas",
                bus1=df_sorted.STATE_TO + " gas",
                efficiency=efficiency,
                p_nom=df_sorted.CAPACITY_MW,
                p_min_pu=0,
                p_max_pu=1,
                p_nom_extendable=True,
                p_nom_min=0,
                p_nom_max=df_sorted.CAPACITY_MW,
                capital_cost=gas_capital_costs,
                length=distances,
                lifetime=np.inf,
                build_year=n.investment_periods[0],
            )

            # Add H2 pipeline infrastructure if retro_pipeline is enabled
            if retro_pipeline:
                if "h2 pipeline retrofit" not in n.carriers.index:
                    n.add("Carrier", "h2 pipeline retrofit", color="#ea048a", nice_name="H2 Pipeline Retrofit")

                # Create H2 pipeline names based on gas pipeline names
                def create_h2_pipeline_name(row):
                    state_from = row["STATE_FROM"]
                    state_to = row["STATE_TO"]
                    index = row["pipeline_index"]
                    return f"{state_from} {state_to} h2 pipeline retrofit-{index}"

                # Create a new dataframe for hydrogen pipelines
                df_h2 = df_sorted.copy()

                # Calculate H2 capacity BEFORE changing the index
                df_h2["H2_CAPACITY_MW"] = df_h2["CAPACITY_MW"] * pipeline_retro_factor

                # Create H2 pipeline names and set as index
                df_h2["h2_pipeline_name"] = df_h2.apply(create_h2_pipeline_name, axis=1)
                df_h2 = df_h2.set_index("h2_pipeline_name")

                distances = self._calculate_distances(df_h2, n) * length_factor
                # Calculate capital costs and electricity efficiency for individual H2 pipelines
                h2_capital_costs = self._calculate_capital_costs(
                    df_h2,
                    n,
                    h2_fom,
                    h2_investment,
                    h2_lifetime,
                    length_factor,
                    h2_currency_year,
                )
                efficiency = self._calculate_efficiency(df_h2, n, h2_elec_input, length_factor, "h2")

                n.madd(
                    "Link",
                    names=df_h2.index,
                    suffix="_fwd",
                    carrier="h2 pipeline retrofit",
                    unit="MW",
                    bus0=df_h2.STATE_FROM + " h2",
                    bus1=df_h2.STATE_TO + " h2",
                    efficiency=efficiency,
                    p_nom=0,
                    p_min_pu=0,
                    p_max_pu=1,
                    p_nom_extendable=True,
                    p_nom_min=0,
                    p_nom_max=df_h2.H2_CAPACITY_MW,
                    capital_cost=h2_capital_costs / 2,
                    length=distances,
                    lifetime=h2_lifetime,
                    build_year=n.investment_periods[0],
                )

                n.madd(
                    "Link",
                    names=df_h2.index,
                    suffix="_rev",
                    carrier="h2 pipeline retrofit",
                    unit="MW",
                    bus0=df_h2.STATE_TO + " h2",
                    bus1=df_h2.STATE_FROM + " h2",
                    efficiency=efficiency,
                    p_nom=0,
                    p_min_pu=0,
                    p_max_pu=1,
                    p_nom_extendable=True,
                    p_nom_min=0,
                    p_nom_max=df_h2.H2_CAPACITY_MW,
                    capital_cost=h2_capital_costs / 2,
                    length=distances,
                    lifetime=h2_lifetime,
                    build_year=n.investment_periods[0],
                )


class TradeGasPipelineCapacity(_GasPipelineCapacity):
    """Pipeline capcity connecting to the interconnect."""

    def __init__(
        self,
        year: int,
        interconnect: str,
        xlsx: str,
        api: str,
        domestic: bool = True,
    ) -> None:
        self.domestic = domestic
        super().__init__(year, interconnect, xlsx, api)

    def extract_pipelines(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get pipelines within geographic scope."""
        df = data.copy()
        if self.domestic:
            return self._get_domestic_pipeline_connections(df)
        else:
            return self._get_international_pipeline_connections(df)

    def _get_domestic_pipeline_connections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gets all pipelines within the usa that connect to the interconnect."""
        # get rid of international connections
        df = df[~((df.INTERCONNECT_TO.isin(["canada", "mexico"])) | (df.INTERCONNECT_FROM.isin(["canada", "mexico"])))]

        if self.interconnect == "usa":
            return df
        else:
            # get rid of pipelines that exist only in other interconnects
            return df[df["INTERCONNECT_TO"].eq(self.interconnect) | df["INTERCONNECT_FROM"].eq(self.interconnect)]

    def _get_international_pipeline_connections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gets all international pipeline connections."""
        df = df[(df.INTERCONNECT_TO.isin(["canada", "mexico"])) | (df.INTERCONNECT_FROM.isin(["canada", "mexico"]))]
        if self.interconnect == "usa":
            return df
        else:
            return df[(df.INTERCONNECT_TO == self.interconnect) | (df.INTERCONNECT_FROM == self.interconnect)]

    def _get_international_costs(
        self,
        direction: str,
        interpoloation_method: str = "zero",
    ) -> pd.DataFrame:
        """
        Gets timeseries of international costs in $/MWh.

        interpolation_method can be one of:
        - linear, zero
        """
        assert direction in ("imports", "exports")

        # fuel costs/profits at a national level
        costs = eia.FuelCosts("gas", self.year, self.api, industry=direction).get_data()

        # fuel costs come in MCF, so first convert to MMCF
        costs = costs[["value"]].astype("float")
        costs = costs / 1000 * MWH_2_MMCF

        return costs.resample("1h").asfreq().interpolate(method=interpoloation_method)

    def _expand_costs(self, n: pypsa.Network, costs: pd.DataFrame) -> pd.DataFrame:
        """Expands import/export costs over snapshots and investment periods."""
        expanded_costs = []
        for invesetment_period in n.investment_periods:
            # reindex to match any tsa
            cost = costs.copy()
            cost.index = cost.index.map(lambda x: x.replace(year=invesetment_period))
            cost = cost.reindex(n.snapshots.get_level_values(1), method="nearest")
            # set investment periods
            # https://stackoverflow.com/a/56278736/14961492
            old_idx = cost.index.to_frame()
            old_idx.insert(0, "period", invesetment_period)
            cost.index = pd.MultiIndex.from_frame(old_idx)
            expanded_costs.append(cost)
        return pd.concat(expanded_costs)

    def _add_zero_capacity_connections(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Will add a zero capacity link if a connection is missing due to no
        capacity.

        For example, the input data frame of...

        |   | STATE_NAME_TO    | STATE_NAME_FROM  | CAPACITY_MW | STATE_TO | STATE_FROM | INTERCONNECT_TO | INTERCONNECT_FROM |
        |---|------------------|------------------|-------------|----------|------------|-----------------|-------------------|
        | 0 | British Columbia | Washington       | 100         | BC       | WA         | canada          | western           |
        | 1 | Idaho            | British Columbia | 50          | ID       | BC         | western         | canada            |
        | 2 | Washington       | British Columbia | 120         | WA       | BC         | western         | canada            |

        Will get converted to...

        |   | STATE_NAME_TO    | STATE_NAME_FROM  | CAPACITY_MW | STATE_TO | STATE_FROM | INTERCONNECT_TO | INTERCONNECT_FROM |
        |---|------------------|------------------|-------------|----------|------------|-----------------|-------------------|
        | 0 | British Columbia | Washington       | 100         | BC       | WA         | canada          | western           |
        | 1 | Idaho            | British Columbia | 50          | ID       | BC         | western         | canada            |
        | 2 | Washington       | British Columbia | 120         | WA       | BC         | western         | canada            |
        | 3 | British Columbia | Idaho            | 0           | BC       | ID         | canada          | western           |
        """

        @staticmethod
        def missing_connections(df: pd.DataFrame) -> list[tuple[str, str]]:
            connections = set(
                map(tuple, df[["STATE_NAME_TO", "STATE_NAME_FROM"]].values),
            )
            missing_connections = []

            for conn in connections:
                reverse_conn = (conn[1], conn[0])
                if reverse_conn not in connections:
                    missing_connections.append(reverse_conn)

            return missing_connections

        connections = missing_connections(df)

        if not connections:
            return df

        state_2_code = df.set_index("STATE_NAME_TO")["STATE_TO"].to_dict()
        state_2_code.update(df.set_index("STATE_NAME_FROM")["STATE_FROM"].to_dict())

        state_2_interconnect = df.set_index("STATE_NAME_TO")["INTERCONNECT_TO"].to_dict()
        state_2_interconnect.update(
            df.set_index("STATE_NAME_FROM")["INTERCONNECT_FROM"].to_dict(),
        )

        zero_capacity = []
        for connection in connections:
            zero_capacity.append(
                [
                    connection[0],
                    connection[1],
                    0,
                    state_2_code[connection[0]],
                    state_2_code[connection[1]],
                    state_2_interconnect[connection[0]],
                    state_2_interconnect[connection[1]],
                ],
            )

        zero_df = pd.DataFrame(zero_capacity, columns=df.columns)

        return pd.concat([df, zero_df])

    def _get_marginal_costs(
        self,
        n: pypsa.Network,
        connections: pd.DataFrame,
        imports: bool,
    ) -> pd.DataFrame:
        """Gets time varrying import/export costs."""
        df = connections.copy()

        states_in_model = self.get_states_in_model(n)

        if imports:
            costs = self._get_international_costs("imports")
            df = df[df.STATE_TO.isin(states_in_model)]
        else:
            # multiple by -1 cause exporting makes money
            costs = self._get_international_costs("exports").mul(-1)
            df = df[df.STATE_FROM.isin(states_in_model)]

        for link in df.index:
            costs[link] = costs["value"]

        return costs.drop(columns=["value"])

    def _assign_country(self, n: pypsa.Network, template: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns country column.

        Country is always in model spatial scope.
        """
        df = template.copy()
        states_in_model = self.get_states_in_model(n)

        df["COUNTRY"] = np.where(
            df.STATE_FROM.isin(states_in_model),
            df.STATE_FROM,
            df.STATE_TO,
        )

        return df

    def _assign_link_buses(
        self,
        n: pypsa.Network,
        template: pd.DataFrame,
    ) -> pd.DataFrame:
        """Assigns bus names for links."""

        def assign_bus0_name(row) -> str:
            if row["STATE_FROM"] in states_in_model:
                return f"{row['STATE_FROM']} gas"
            else:
                return f"{row['STATE_FROM']} {row['STATE_TO']} gas trade"

        def assign_bus1_name(row) -> str:
            if row["STATE_TO"] in states_in_model:
                return f"{row['STATE_TO']} gas"
            else:
                return f"{row['STATE_FROM']} {row['STATE_TO']} gas trade"

        df = template.copy()
        states_in_model = self.get_states_in_model(n)

        df["bus0"] = df.apply(assign_bus0_name, axis=1)
        df["bus1"] = df.apply(assign_bus1_name, axis=1)

        return df

    def _assign_stores(self, template: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns if the associated store should be sink or source.

        If bus0 is a trading bus, energy will flow into the model (ie.
        imports) If bus0 is a state gas bus, energy will flow out of the
        model (ie. exports)
        """
        df = template.copy()

        df["store"] = df.bus0.map(
            lambda x: "import" if x.endswith(" trade") else "export",
        )

        return df

    def build_infrastructure(self, n: pypsa.Network) -> None:
        """
        Builds import and export bus+link+store to connect to.

        Dataframe must have a 'STATE_TO', 'STATE_FROM', 'INTERCONNECT_TO', and
        'INTERCONNECT_FROM' columns

        The function does the following
        - exisitng domestic buses are retained
        - new import export buses are created based on region
            - "WA BC gas trade"
            - "BC WA gas trade"
        - new one way links are added with capacity limits
            - "WA BC gas trade"
            - "BC WA gas trade"
        - stores are added WITHOUT energy limits
            - "WA BC gas trade"
            - "BC WA gas trade"
        """
        df = self.filter_on_sate(n, self.data, in_spatial_scope=False)

        df = self._add_zero_capacity_connections(df)

        template = df.copy()
        template["NAME"] = template.STATE_FROM + " " + template.STATE_TO
        template = template.set_index("NAME")

        template = self._assign_country(n, template)
        template = self._assign_link_buses(n, template)
        template = self._assign_stores(template)

        # remove any conections within geographic scope
        if self.domestic:
            template = template[
                ~(template.STATE_TO.isin(n.buses.reeds_state) & template.STATE_FROM.isin(n.buses.reeds_state))
            ]

        store_imports = template[template.store == "import"].copy()
        store_exports = template[template.store == "export"].copy()

        if not self.domestic:
            import_costs = self._get_marginal_costs(n, template, True)
            export_costs = self._get_marginal_costs(n, template, False)
            marginal_cost = pd.concat([import_costs, export_costs], axis=1)
            marginal_cost = self._expand_costs(n, marginal_cost)
        else:
            marginal_cost = 0

        if "gas trade" not in n.carriers.index:
            n.add("Carrier", "gas trade", color="#d35050", nice_name="Gas Trade")

        n.madd(
            "Bus",
            names=template.index,
            suffix=" gas trade",
            carrier="gas trade",
            unit="MWh",
            country=template.COUNTRY,
            interconnect=self.interconnect,
        )

        n.madd(
            "Link",
            names=template.index,
            suffix=" gas trade",
            carrier="gas trade",
            unit="MW",
            bus0=template.bus0,
            bus1=template.bus1,
            p_nom=template.CAPACITY_MW,
            p_min_pu=0,
            p_max_pu=1,
            p_nom_extendable=False,
            efficiency=1,  # must be 1 for proper cost accounting
            marginal_cost=marginal_cost,
            lifetime=np.inf,
            build_year=n.investment_periods[0],
        )

        n.madd(
            "Store",
            names=store_exports.index,
            suffix=" gas trade",
            unit="MWh",
            bus=store_exports.bus1,
            carrier="gas trade",
            capital_cost=0,
            marginal_cost=0,
            e_cyclic=False,
            e_cyclic_per_period=False,
            e_nom=0,
            e_nom_extendable=True,
            e_nom_min=0,
            e_nom_max=np.inf,
            e_min_pu=0,
            e_max_pu=1,
            lifetime=np.inf,
            build_year=n.investment_periods[0],
        )

        n.madd(
            "Store",
            names=store_imports.index,
            unit="MWh",
            suffix=" gas trade",
            bus=store_imports.bus0,
            carrier="gas trade",
            capital_cost=0,
            marginal_cost=0,
            e_cyclic=False,
            e_cyclic_per_period=False,
            e_nom=0,
            e_nom_extendable=True,
            e_nom_min=0,
            e_nom_max=np.inf,
            e_min_pu=-1,  # minus 1 for energy addition!
            e_max_pu=0,
            lifetime=np.inf,
            build_year=n.investment_periods[0],
        )


class PipelineLinepack(GasData):
    """Creator for linepack infrastructure."""

    def __init__(
        self,
        year: int,
        interconnect: str,
        counties: str,
        pipelines: str,
    ) -> None:
        self.counties = StateGeometry(counties)
        self.states = self.counties.states
        self.pipeline_geojson = pipelines
        super().__init__(year, interconnect)

    def read_data(self) -> gpd.GeoDataFrame:
        """Read in geojson pipe locations.

        https://atlas.eia.gov/apps/3652f0f1860d45beb0fed27dc8a6fc8d/explore.
        """
        return gpd.read_file(self.pipeline_geojson)

    def filter_on_sate(
        self,
        n: pypsa.Network,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Filter formatted data to only include states in geographic scope."""
        states_in_model = n.buses[
            ~n.buses.carrier.isin(
                ["gas storage", "gas trade", "gas pipeline"],
            )
        ].reeds_state.unique()

        if "STATE" not in df.columns:
            logger.debug(
                "Natual gas data not filtered due to incorrect data formatting",
            )
            return df

        df = df[df.STATE.isin(states_in_model)].copy()

        return df

    def format_data(self, data: gpd.GeoDataFrame) -> pd.DataFrame:
        """Format linepack data."""
        gdf = data.copy()
        states = self.states.copy()

        length_in_state = gpd.sjoin(
            gdf.to_crs("4269"),
            states,
            how="right",
            predicate="within",
        ).reset_index()
        length_in_state = (
            length_in_state[["STATE_NAME", "STATE", "TYPEPIPE", "Shape_Leng", "Shape__Length"]]
            .rename(columns={"Shape_Leng": "LENGTH_DEG", "Shape__Length": "LENGTH_M"})
            .groupby(by=["STATE_NAME", "STATE", "TYPEPIPE"])
            .sum()
            .reset_index()
        )

        # https://publications.anl.gov/anlpubs/2008/02/61034.pdf
        intrastate_radius = 12 * 0.0254  # inches in meters (24in dia)
        interstate_radius = 18 * 0.0254  # inches meters (36in dia)

        volumne_in_state = length_in_state.copy()
        volumne_in_state["RADIUS"] = volumne_in_state.TYPEPIPE.map(
            lambda x: interstate_radius if x == "Interstate" else intrastate_radius,
        )
        volumne_in_state["VOLUME_M3"] = volumne_in_state.LENGTH_M * pi * volumne_in_state.RADIUS**2
        volumne_in_state = volumne_in_state[["STATE_NAME", "STATE", "VOLUME_M3"]]
        volumne_in_state = volumne_in_state.groupby(by=["STATE_NAME", "STATE"]).sum()

        # https://publications.anl.gov/anlpubs/2008/02/61034.pdf
        max_pressure = 8000  # kPa
        min_pressure = 4000  # kPa

        # Energy content calculated using:
        # E_total = n * Cv * T = (PV/RT) * Cv * T = (PV/R) * Cv
        # E = PV * (R/Cv)
        # R = 8.314 J/(mol.K)
        # Cv_Methane = 35.7 J/(mol.K)

        r_cv = 8.314 / 35.7

        energy_in_state = volumne_in_state.copy()
        energy_in_state["MAX_ENERGY_kJ"] = energy_in_state.VOLUME_M3 * max_pressure * r_cv
        energy_in_state["MIN_ENERGY_kJ"] = energy_in_state.VOLUME_M3 * min_pressure * r_cv
        energy_in_state["NOMINAL_ENERGY_kJ"] = (energy_in_state.MAX_ENERGY_kJ + energy_in_state.MIN_ENERGY_kJ) / 2

        final = energy_in_state.copy()
        final["MAX_ENERGY_MWh"] = final.MAX_ENERGY_kJ * KJ_2_MWH
        final["MIN_ENERGY_MWh"] = final.MIN_ENERGY_kJ * KJ_2_MWH
        final["NOMINAL_ENERGY_MWh"] = final.NOMINAL_ENERGY_kJ * KJ_2_MWH

        final = final[["MAX_ENERGY_MWh", "MIN_ENERGY_MWh", "NOMINAL_ENERGY_MWh"]].reset_index()
        return self.filter_on_interconnect(final)

    def build_infrastructure(self, n: pypsa.Network, **kwargs) -> None:
        """Add pypsa components to network."""
        df = self.filter_on_sate(n, self.data)
        df = df.set_index("STATE")

        if "gas pipeline" not in n.carriers.index:
            n.add("Carrier", "gas pipeline", color="#d35050", nice_name="Gas Pipeline")

        cyclic_storage = kwargs.get("cyclic_storage", True)
        standing_loss = kwargs.get("standing_loss", 0)

        n.madd(
            "Store",
            names=df.index,
            unit="MWh_th",
            suffix=" linepack",
            bus=df.index + " gas",
            carrier="gas pipeline",
            e_nom=df.MAX_ENERGY_MWh,
            e_nom_extendable=False,
            e_nom_min=0,
            e_nom_max=np.inf,
            e_min_pu=df.MIN_ENERGY_MWh / df.MAX_ENERGY_MWh,
            e_max_pu=1,
            e_initial=df.NOMINAL_ENERGY_MWh,
            e_initial_per_period=False,
            e_cyclic=cyclic_storage,
            e_cyclic_per_period=True,
            p_set=0,
            marginal_cost=0,
            capital_cost=0.1,
            standing_loss=standing_loss,
            lifetime=np.inf,
            build_year=n.investment_periods[0],
        )


def _remove_marginal_costs(n: pypsa.Network):
    """Removes marginal costs of CCGT and OCGT plants."""
    links = n.links[n.links.carrier.str.contains("CCGT") | n.links.carrier.str.contains("OCGT")].index

    n.links.loc[links, "marginal_cost"] = 0


###
# MAIN FUNCTION TO EXECUTE
###


def build_natural_gas(
    n: pypsa.Network,
    years: dict[str, int],
    api: str,
    interconnect: str = "western",
    county_path: str = "../data/counties/cb_2020_us_county_500k.shp",
    pipelines_path: str = "../data/natural_gas/EIA-StatetoStateCapacity_Feb2024.xlsx",
    pipeline_shape_path: str = "../data/natural_gas/pipelines.geojson",
    storage_plant_path: str = "../repo_data/Natural gas storage plant.csv",
    options: dict[str, Any] | None = None,
    **kwargs,
) -> None:
    if not options:
        options = {}
    cost_factor = options.get("cost_factor", 1.0)
    cyclic_storage = options.get("cyclic_storage", True)
    standing_loss = options.get("standing_loss", 0)
    line_pack = options.get("line_pack", True)
    agg_storage = options.get("aggregate_storage", True)
    agg_pipeline = options.get("aggregate_pipeline", True)
    retro_storage_h2 = options.get("retro_storage_h2", False)
    retro_storage_acaes = options.get("retro_storage_acaes", False)
    retro_storage_type = options.get("retro_storage_type", None)
    retro_pipeline = options.get("retro_pipeline", False)
    storage_eng_retro_factor_h2 = options.get("storage_eng_retro_factor_h2", 0.2551)
    storage_pow_retro_factor_h2 = options.get("storage_pow_retro_factor_h2", 0.4)
    storage_eng_retro_factor_acaes = options.get("storage_eng_retro_factor_acaes", 0.018)
    length_factor = options.get("length_factor", 1.25)
    addi_costs = kwargs.get("addi_costs", None)

    # add state level natural gas processing facilities
    production = GasProcessing(years["production"], interconnect, api)
    production.build_infrastructure(n, capacity_multiplier=1, cost_factor=cost_factor)

    # add state level gas storage facilities
    storage = GasStorage(2024, interconnect, storage_plant_path, agg_storage)  # year is not really used for storage
    storage.build_infrastructure(
        n,
        cyclic_storage=cyclic_storage,
        retro_storage_h2=retro_storage_h2,
        retro_storage_acaes=retro_storage_acaes,
        addi_costs=addi_costs,
        storage_eng_retro_factor_h2=storage_eng_retro_factor_h2,
        storage_pow_retro_factor_h2=storage_pow_retro_factor_h2,
        storage_eng_retro_factor_acaes=storage_eng_retro_factor_acaes,
        retro_storage_type=retro_storage_type,
    )

    # add interconnect pipelines - capital costs calculated from bus coordinates
    pipelines = InterconnectGasPipelineCapacity(years["pipeline"], interconnect, pipelines_path, api, agg_pipeline)
    pipelines.build_infrastructure(n, retro_pipeline=retro_pipeline, addi_costs=addi_costs, length_factor=length_factor)

    # add pipelines for imports/exports
    # TODO: have trade pipelines share data to only instantiate once

    # pipelines_domestic = TradeGasPipelineCapacity( # Hawaii, DC
    #     years["pipeline"],
    #     interconnect,
    #     pipelines_path,
    #     api,
    #     domestic=True,
    # )
    # pipelines_domestic.build_infrastructure(n)
    # pipelines_international = TradeGasPipelineCapacity( # Canada, Mexico
    #     years["pipeline"],
    #     interconnect,
    #     pipelines_path,
    #     api,
    #     domestic=False,
    # )
    # pipelines_international.build_infrastructure(n)

    # add pipeline linepack
    if line_pack:
        linepack = PipelineLinepack(years["pipeline"], interconnect, county_path, pipeline_shape_path)
        linepack.build_infrastructure(
            n,
            cyclic_storage=cyclic_storage,
            standing_loss=standing_loss,
        )

    # _remove_marginal_costs(n)


if __name__ == "__main__":
    n = pypsa.Network("../resources/Washington/western/elec_s10_c4m_ec_lv1.0_3h.nc")
    year = 2018
    years = {"production": 2018, "storage": 2018, "pipeline": 2018}
    with open("./../config/config.api.yaml") as file:
        yaml_data = yaml.safe_load(file)
    api = yaml_data["api"]["eia"]

    pipelines = InterconnectGasPipelineCapacity(
        year,
        "western",
        "../data/natural_gas/EIA-StatetoStateCapacity_Feb2024.xlsx",
        api,
    )
    # pipelines.build_infrastructure(n)

    build_natural_gas(n=n, years=years, api=api)
