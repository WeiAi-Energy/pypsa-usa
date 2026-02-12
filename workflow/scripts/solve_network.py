"""
Solves optimal operation and capacity for a network with the option to
iteratively optimize while updating line reactances.

This script is used for optimizing the electrical network as well as the
sector coupled network.

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.

The optimization is based on the :func:`network.optimize` function.
Additionally, some extra constraints specified in :mod:`solve_network` are added.

.. note::

    The rules ``solve_elec_networks`` and ``solve_sector_networks`` run
    the workflow for all scenarios in the configuration file (``scenario:``)
    based on the rule :mod:`solve_network`.
"""

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
import yaml
from _helpers import (
    configure_logging,
    load_costs,
)
from constants import NG_MWH_2_MMCF, STATE_2_CODE
from eia import Trade
from pandas import Series
from pypsa.descriptors import get_switchable_as_dense as get_as_dense

logger = logging.getLogger(__name__)
# pypsa.pf.logger.setLevel(logging.WARNING)


def get_region_buses(n, region_list):
    return n.buses[
        (
            n.buses.country.isin(region_list)
            # | n.buses.reeds_zone.isin(region_list)
            | n.buses.reeds_state.isin(region_list)
            | n.buses.interconnect.str.lower().isin(region_list)
            # | n.buses.nerc_reg.isin(region_list)
            | (1 if "all" in region_list else 0)
        )
        & (n.buses.carrier == "AC")
    ]


def filter_components(
    n: pypsa.Network,
    component_type: str,
    planning_horizon: str | int,
    carrier_list: list[str],
    region_buses: pd.Index,
    extendable: bool,
):
    """
    Filter components based on common criteria.

    Parameters
    ----------
    - n: pypsa.Network
        The PyPSA network object.
    - component_type: str
        The type of component (e.g., "Generator", "StorageUnit").
    - planning_horizon: str or int
        The planning horizon to filter active assets.
    - carrier_list: list
        List of carriers to filter.
    - region_buses: pd.Index
        Index of region buses to filter.
    - extendable: bool, optional
        If specified, filters by extendable or non-extendable assets.

    Returns
    -------
    - pd.DataFrame
        Filtered assets.
    """
    component = n.df(component_type)
    if planning_horizon != "all":
        ph = int(planning_horizon)
        iv = n.investment_periods
        active_components = n.get_active_assets(component.index.name, iv[iv >= ph][0])
    else:
        active_components = component.index

    # Links will throw the following attribute error, as we must specify bus0
    # AttributeError: 'DataFrame' object has no attribute 'bus'. Did you mean: 'bus0'?
    bus_name = "bus0" if component_type.lower() == "link" else "bus"

    filtered = component.loc[
        active_components
        & component.carrier.isin(carrier_list)
        & component[bus_name].isin(region_buses)
        & (component.p_nom_extendable == extendable)
    ]

    return filtered


def _build_observation_matrix(multiplier_data_path: str) -> tuple[pd.DataFrame, pd.Index, pd.Index]:
    """
    Build sparse observation matrix for all technologies and states.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Index, pd.Index]
        - obs_matrix: DataFrame with shape (n_files, n_states), NaN for missing
        - file_index: Index of file names (technologies)
        - state_index: Index of state codes
    """
    multiplier_path = Path(multiplier_data_path)

    # Collect all observations
    observations = []
    for csv_file in multiplier_path.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if "Location Variation" not in df.columns:
                continue

            df = df[["State", "Location Variation"]].copy()
            df["State"] = df["State"].map(STATE_2_CODE)
            df = df.dropna(subset=["State", "Location Variation"])

            df_grouped = df.groupby("State")["Location Variation"].mean()

            observations.append(
                {
                    "file": csv_file.stem,
                    "data": df_grouped,
                }
            )
        except Exception:
            continue

    if not observations:
        return pd.DataFrame(), pd.Index([]), pd.Index([])

    # Build matrix
    all_states = set()
    for obs in observations:
        all_states.update(obs["data"].index)

    all_states = sorted(all_states)
    all_files = [obs["file"] for obs in observations]

    # Create sparse matrix
    obs_matrix = pd.DataFrame(
        index=pd.Index(all_files, name="file"),
        columns=pd.Index(all_states, name="state"),
        dtype=float,
    )

    for obs in observations:
        file_name = obs["file"]
        for state, value in obs["data"].items():
            obs_matrix.loc[file_name, state] = value

    return obs_matrix, obs_matrix.index, obs_matrix.columns


def _fit_two_way_fixed_effects(
    obs_matrix: pd.DataFrame,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[float, pd.Series, pd.Series]:
    """
    Fit two-way fixed effects model using Alternating Least Squares (ALS).

    Model: x_{f,s} ≈ μ + α_f + β_s

    Parameters
    ----------
    obs_matrix : pd.DataFrame
        Observation matrix (files x states) with NaN for missing values
    max_iter : int
        Maximum ALS iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    Tuple[float, pd.Series, pd.Series]
        - mu: global mean
        - alpha: file/tech effects (indexed by file)
        - beta: state effects (indexed by state)
    """
    # Initialize
    mu = obs_matrix.stack().mean()  # Global mean of observed values
    alpha = pd.Series(0.0, index=obs_matrix.index, name="tech_effect")
    beta = pd.Series(0.0, index=obs_matrix.columns, name="state_effect")

    for iteration in range(max_iter):
        alpha_old = alpha.copy()
        beta_old = beta.copy()

        # Update alpha (file effects): α_f = mean_s(x_{f,s} - μ - β_s)
        for file in obs_matrix.index:
            row = obs_matrix.loc[file]
            observed = row.dropna()
            if len(observed) > 0:
                residuals = observed - mu - beta[observed.index]
                alpha[file] = residuals.mean()

        # Update beta (state effects): β_s = mean_f(x_{f,s} - μ - α_f)
        for state in obs_matrix.columns:
            col = obs_matrix[state]
            observed = col.dropna()
            if len(observed) > 0:
                residuals = observed - mu - alpha[observed.index]
                beta[state] = residuals.mean()

        # Check convergence
        alpha_change = np.abs(alpha - alpha_old).max()
        beta_change = np.abs(beta - beta_old).max()

        if alpha_change < tol and beta_change < tol:
            logger.info(f"Two-way FE converged in {iteration + 1} iterations")
            break
    else:
        logger.warning(f"Two-way FE did not converge in {max_iter} iterations")

    return mu, alpha, beta


def _compute_z_scores_per_file(obs_matrix: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Compute z-scores for each state within each file.

    z_{f,s} = (x_{f,s} - μ_f) / σ_f

    Returns
    -------
    Dict[str, pd.Series]
        Dictionary mapping file name to Series of z-scores (indexed by state)
    """
    z_scores = {}

    for file in obs_matrix.index:
        row = obs_matrix.loc[file].dropna()
        if len(row) > 1:
            mean_f = row.mean()
            std_f = row.std(ddof=1)
            if std_f > 0:
                z_scores[file] = (row - mean_f) / std_f

    return z_scores


def _aggregate_state_z_scores(z_scores: dict[str, pd.Series]) -> pd.Series:
    """
    Aggregate z-scores across files to get average state z-score.

    Returns
    -------
    pd.Series
        Mean z-score for each state across all files (states with data)
    """
    # Collect all z-scores
    all_z = []
    for file, zs in z_scores.items():
        for state, z in zs.items():
            all_z.append({"state": state, "z": z})

    if not all_z:
        return pd.Series(dtype=float)

    df_z = pd.DataFrame(all_z)
    # Average z-score for each state
    mean_z = df_z.groupby("state")["z"].mean()
    return mean_z


def _compute_adaptive_weight(
    target_file: str,
    obs_matrix: pd.DataFrame,
    z_scores: dict[str, pd.Series],
) -> float:
    """
    Compute adaptive weight w ∈ [0, 1] for blending z-score and FE estimates.

    Higher w → trust z-score method more (more data overlap with other files)
    Lower w → trust FE method more (sparse data, rely on systematic effects)

    Parameters
    ----------
    target_file : str
        File name for which we're computing weight
    obs_matrix : pd.DataFrame
        Full observation matrix
    z_scores : Dict[str, pd.Series]
        Z-scores per file

    Returns
    -------
    float
        Weight w ∈ [0, 1]
    """
    if target_file not in obs_matrix.index:
        return 0.5  # Default

    # Count how many other files have overlapping states
    target_states = set(obs_matrix.loc[target_file].dropna().index)

    if len(target_states) == 0:
        return 0.0  # No data, rely purely on FE

    # Count states with z-score information from other files
    states_with_z = set()
    for file, zs in z_scores.items():
        if file != target_file:
            states_with_z.update(zs.index)

    overlap = len(target_states & states_with_z)

    # Weight based on overlap ratio
    # More overlap → higher confidence in z-score method
    w = overlap / len(target_states) if len(target_states) > 0 else 0.0

    return w


def _build_state_normalized_deviations_twoway(
    multiplier_data_path: str,
) -> tuple[Any, Any, float, Series, Series, Series]:
    """
    Build two-way fixed effects model and z-score information.

    Returns
    -------
    Tuple containing:
        - file_means: Dict[file] = μ_f (mean of each file)
        - file_stds: Dict[file] = σ_f (std of each file)
        - mu_global: Global mean
        - alpha: Tech/file effects
        - beta: State effects
        - mean_z_scores: Average z-score per state across files
    """
    # Step 1: Build observation matrix
    obs_matrix, file_idx, state_idx = _build_observation_matrix(multiplier_data_path)

    if obs_matrix.empty:
        return {}, {}, 1.0, pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    # Step 2: Fit two-way fixed effects
    logger.info("Fitting two-way fixed effects model...")
    mu_global, alpha, beta = _fit_two_way_fixed_effects(obs_matrix)

    logger.info(
        f"Global mean: {mu_global:.3f}, "
        f"Tech effect range: [{alpha.min():.3f}, {alpha.max():.3f}], "
        f"State effect range: [{beta.min():.3f}, {beta.max():.3f}]",
    )

    # Step 3: Compute z-scores per file
    logger.info("Computing z-scores per file...")
    z_scores = _compute_z_scores_per_file(obs_matrix)

    # Step 4: Aggregate z-scores across files
    mean_z_scores = _aggregate_state_z_scores(z_scores)

    # Step 5: Compute file statistics for z-score method
    file_means = obs_matrix.mean(axis=1).to_dict()
    file_stds = obs_matrix.std(axis=1, ddof=1).to_dict()

    return file_means, file_stds, mu_global, alpha, beta, mean_z_scores


def clean_locational_multiplier_twoway(
    df: pd.DataFrame,
    file_means: dict[str, float],
    file_stds: dict[str, float],
    mu_global: float,
    alpha: pd.Series,
    beta: pd.Series,
    mean_z_scores: pd.Series,
    obs_matrix: pd.DataFrame,
    csv_filename: str = None,
) -> pd.DataFrame:
    """
    Clean and fill missing states using adaptive blending of z-score and FE methods.

    Final estimate: x̂ = w·(μ_f + ẑ_s·σ_f) + (1-w)·(μ + α_f + β_s)
    """
    ALL_US_STATES = {
        "AL",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "FL",
        "GA",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
    }

    # Clean existing data
    df_clean = df[["State", "Location Variation"]].copy()
    df_clean["State"] = df_clean["State"].map(STATE_2_CODE)
    df_clean = df_clean.dropna(subset=["State"])
    df_grouped = df_clean.groupby("State").mean()

    states_with_data = set(df_grouped.dropna(subset=["Location Variation"]).index)
    missing_states = ALL_US_STATES - states_with_data

    if not missing_states:
        return df_grouped

    if csv_filename not in file_means or csv_filename not in alpha.index:
        # Fallback: no info for this file
        logger.warning(f"{csv_filename}: No FE info available, using simple mean")
        fallback = (
            df_grouped["Location Variation"].dropna().mean()
            if len(
                df_grouped["Location Variation"].dropna(),
            )
            > 0
            else 1.0
        )
        for state in missing_states:
            df_grouped.loc[state, "Location Variation"] = fallback
        return df_grouped

    # Get file statistics
    mu_f = file_means[csv_filename]
    sigma_f = file_stds.get(csv_filename, 0.0)
    alpha_f = alpha[csv_filename]

    # Compute adaptive weight
    w = _compute_adaptive_weight(
        csv_filename,
        obs_matrix,
        _compute_z_scores_per_file(obs_matrix),
    )

    filled_count = 0
    for state in missing_states:
        # Method 1: z-score estimate
        if state in mean_z_scores and sigma_f > 0:
            z_hat = mean_z_scores[state]
            x_hat_z = mu_f + z_hat * sigma_f
        else:
            x_hat_z = mu_f  # Fallback to file mean

        # Method 2: Fixed effects estimate
        if state in beta.index:
            beta_s = beta[state]
            x_hat_fe = mu_global + alpha_f + beta_s
        else:
            x_hat_fe = mu_global + alpha_f  # State effect unknown, use global + tech

        # Adaptive blend
        x_final = w * x_hat_z + (1 - w) * x_hat_fe

        df_grouped.loc[state, "Location Variation"] = x_final
        filled_count += 1

    if filled_count > 0:
        logger.info(
            f"{csv_filename}: Filled {filled_count} states with adaptive blend "
            f"(w={w:.2f}, μ_f={mu_f:.3f}, α_f={alpha_f:+.3f})",
        )

    return df_grouped


def apply_regional_cost_multipliers(
    n: pypsa.Network,
    multiplier_data_path: str = "repo_data/locational_multipliers/",
) -> None:
    """
    Apply regional multipliers using two-way fixed effects model.
    """
    logger.info("Applying regional cost multipliers with two-way FE model")

    # Build two-way fixed effects model and z-scores
    file_means, file_stds, mu_global, alpha, beta, mean_z_scores = _build_state_normalized_deviations_twoway(
        multiplier_data_path
    )

    # Build observation matrix for adaptive weighting
    obs_matrix, _, _ = _build_observation_matrix(multiplier_data_path)

    # Define carrier to multiplier mapping
    carrier_multiplier_map = {
        # Generator
        "hydro": ("hydro-100mw", ["Generator"]),
        "nuclear": ("nuclear-1117mw", ["Generator"]),
        "solar": ("spv-150mw", ["Generator"]),
        "onwind": ("onshore-wind-200mw", ["Generator"]),
        "offwind_floating": ("offshore-wind-40x10mw", ["Generator"]),
        # Gas-related generation and links (mostly Links after sector coupling)
        "OCGT": ("natural-gas-430mw-90ccs", ["Link"]),
        "CCGT": ("natural-gas-430mw-90ccs", ["Link"]),
        "CCGT-95CCS": ("natural-gas-430mw-90ccs", ["Link"]),
        "CCGT-97CCS": ("natural-gas-430mw-90ccs", ["Link"]),
        "acaes": ("natural-gas-430mw-90ccs", ["Link"]),
        "tes": ("natural-gas-430mw-90ccs", ["Link"]),
        "h2 turbine": ("natural-gas-430mw-90ccs", ["Link"]),
        # Biomass-related (mostly Links after sector coupling)
        "biomass": ("biomass-50mw", ["Link"]),
        "biomass-CCS": ("biomass-50mw", ["Link"]),
        # Coal-related (mostly Links after sector coupling)
        "coal": ("coal-ultra-supercritical", ["Link"]),
        "coal-95CCS": ("coal-ultra-supercritical-90ccs", ["Link"]),
        "coal-99CCS": ("coal-ultra-supercritical-90ccs", ["Link"]),
        # Battery storage (StorageUnits)
        "2hr_battery_storage": ("battery-storage-4hr-50mw", ["StorageUnit"]),
        "4hr_battery_storage": ("battery-storage-4hr-50mw", ["StorageUnit"]),
        "6hr_battery_storage": ("battery-storage-4hr-50mw", ["StorageUnit"]),
        "8hr_battery_storage": ("battery-storage-4hr-50mw", ["StorageUnit"]),
        "10hr_battery_storage": ("battery-storage-4hr-50mw", ["StorageUnit"]),
        "battery": ("battery-storage-4hr-50mw", ["StorageUnit"]),
        "battery storage": ("battery-storage-4hr-50mw", ["StorageUnit"]),
        # DAC and electrolyzer (Links)
        "dac": ("fuel-cell-10mw", ["Link"]),
        "h2 electrolysis": ("fuel-cell-10mw", ["Link"]),
        "gas methanation": ("fuel-cell-10mw", ["Link"]),
        "h2 smr": ("fuel-cell-10mw", ["Link"]),
        "h2 smr-cc": ("fuel-cell-10mw", ["Link"]),
        "h2 bio": ("fuel-cell-10mw", ["Link"]),
        "h2 bio-cc": ("fuel-cell-10mw", ["Link"]),
        "gas bio-cc": ("fuel-cell-10mw", ["Link"]),
        # Pumped hydro storage and hydro
        "8hr_PHS": ("hydro-100mw", ["StorageUnit"]),
        "10hr_PHS": ("hydro-100mw", ["StorageUnit"]),
        "12hr_PHS": ("hydro-100mw", ["StorageUnit"]),
        "PHS": ("hydro-100mw", ["StorageUnit"]),
    }

    # Get bus to state mapping
    bus_state_mapper = n.buses["STATE"].to_dict()

    # Process each carrier across all component types
    for carrier, (multiplier_file, component_types) in carrier_multiplier_map.items():
        multiplier_path = Path(multiplier_data_path) / f"{multiplier_file}.csv"
        if not multiplier_path.exists():
            logger.warning(f"Multiplier file {multiplier_path} not found. Skipping {carrier}")
            continue

        df_multiplier = pd.read_csv(multiplier_path)

        # Use two-way FE method to fill missing states
        df_multiplier = clean_locational_multiplier_twoway(
            df_multiplier,
            file_means=file_means,
            file_stds=file_stds,
            mu_global=mu_global,
            alpha=alpha,
            beta=beta,
            mean_z_scores=mean_z_scores,
            obs_matrix=obs_matrix,
            csv_filename=multiplier_file,
        )

        current_mean = df_multiplier["Location Variation"].mean()
        if current_mean > 0:  # Avoid division by zero
            df_multiplier["Location Variation"] = df_multiplier["Location Variation"] / current_mean
            logger.info(
                f"Normalized {carrier} multipliers: original mean={current_mean:.3f}, "
                f"new mean={df_multiplier['Location Variation'].mean():.3f}",
            )
        else:
            logger.warning(f"Cannot normalize {carrier} multipliers: mean is {current_mean}")

        # Process Generators
        if "Generator" in component_types:
            gens = n.generators[n.generators.carrier == carrier]
            if not gens.empty:
                gen_copy = gens.copy()
                gen_copy["state"] = gen_copy.bus.map(bus_state_mapper)
                gen_copy = gen_copy[gen_copy["state"].isin(df_multiplier.index)]

                for idx in gen_copy.index:
                    state = gen_copy.at[idx, "state"]
                    multiplier = df_multiplier.at[state, "Location Variation"]
                    current_cost = n.generators.at[idx, "capital_cost"]
                    n.generators.at[idx, "capital_cost"] = current_cost * multiplier

                logger.info(f"Applied regional multipliers to {len(gen_copy)} {carrier} generators")

        # Process Links
        if "Link" in component_types:
            links = n.links[n.links.carrier == carrier]
            if not links.empty:
                link_copy = links.copy()
                link_copy["state"] = link_copy.bus1.map(bus_state_mapper)
                link_copy = link_copy[link_copy["state"].isin(df_multiplier.index)]

                for idx in link_copy.index:
                    state = link_copy.at[idx, "state"]
                    multiplier = df_multiplier.at[state, "Location Variation"]
                    current_cost = n.links.at[idx, "capital_cost"]
                    n.links.at[idx, "capital_cost"] = current_cost * multiplier

                logger.info(f"Applied regional multipliers to {len(link_copy)} {carrier} links")

        # Process StorageUnits
        if "StorageUnit" in component_types:
            storage = n.storage_units[n.storage_units.carrier == carrier]
            if not storage.empty:
                storage_copy = storage.copy()
                storage_copy["state"] = storage_copy.bus.map(bus_state_mapper)
                storage_copy = storage_copy[storage_copy["state"].isin(df_multiplier.index)]

                for idx in storage_copy.index:
                    state = storage_copy.at[idx, "state"]
                    multiplier = df_multiplier.at[state, "Location Variation"]
                    current_cost = n.storage_units.at[idx, "capital_cost"]
                    n.storage_units.at[idx, "capital_cost"] = current_cost * multiplier

                logger.info(f"Applied regional multipliers to {len(storage_copy)} {carrier} storage units")


def recalculate_network_costs(n: pypsa.Network, costs: pd.DataFrame) -> None:
    """
    Recalculate capital costs for transmission lines, AC/DC links, gas pipelines, and H2 pipelines.

    For HVDC links:
    - Recalculates length and efficiency based on haversine distance
    - Merges duplicate HVDC links between same state pairs
    - Applies state multipliers ONLY to distance-related costs
    - Inverter costs are NOT affected by state multipliers

    For AC lines/links:
    - Applies state multipliers to all capital costs

    For gas, H2, CO2 pipelines:
    - Applies same state multiplier approach as transmission links

    Parameters
    ----------
    n : pypsa.Network
        Network object to modify
    costs : pd.DataFrame
        Cost data containing transmission cost parameters

    Reference
    ---------
    State multipliers based on: https://docs.nrel.gov/docs/fy21osti/78195.pdf Fig.26
    """
    logger.info("Recalculating network capital costs with state multipliers")

    # ========================================================================
    # State multiplier mapping
    # ========================================================================
    state_multipliers = {
        # 3.89 multiplier - Highest cost states (Northeast)
        "ME": 3.89,
        "NH": 3.89,
        "VT": 3.89,
        "MA": 3.89,
        "RI": 3.89,
        "CT": 3.89,
        "NJ": 3.89,
        "DE": 3.89,
        "MD": 3.89,
        # 3.06 multiplier - High cost states
        "CA": 3.06,
        "NY": 3.06,
        "PA": 3.06,
        # 1.94 multiplier - Moderate-high cost states
        "LA": 1.94,
        "AR": 1.94,
        "MS": 1.94,
        # 1.5 multiplier - Moderate cost states
        "AL": 1.5,
        "AZ": 1.5,
        "CO": 1.5,
        "FL": 1.5,
        "GA": 1.5,
        "ID": 1.5,
        "KY": 1.5,
        "MT": 1.5,
        "NM": 1.5,
        "NV": 1.5,
        "OH": 1.5,
        "OR": 1.5,
        "TX": 1.5,
        "UT": 1.5,
        "VA": 1.5,
        "WA": 1.5,
        "WV": 1.5,
        "WY": 1.5,
        # 1.17 multiplier - Low-moderate cost states
        "MI": 1.17,
        "WI": 1.17,
        "TN": 1.17,
        # 1.0 multiplier - Baseline (lowest cost states)
        "ND": 1.0,
        "SD": 1.0,
        "MN": 1.0,
        "NE": 1.0,
        "IA": 1.0,
        "IL": 1.0,
        "IN": 1.0,
        "MO": 1.0,
        "KS": 1.0,
        "OK": 1.0,
    }

    # Determine which column has state information
    state_col = "STATE"

    # ========================================================================
    # Part 1: Process HVDC Links
    # ========================================================================
    hvdc_links = n.links[n.links.carrier == "DC"]

    if not hvdc_links.empty:
        logger.info(f"Processing {len(hvdc_links)} HVDC links")

        # Ensure underwater_fraction column exists
        if "underwater_fraction" not in n.links.columns:
            n.links["underwater_fraction"] = 0.0

        # Get cost parameters
        hvdc_overhead_cost = costs.at["HVDC overhead", "annualized_capex_per_mw_km"]
        hvdc_submarine_cost = costs.at["HVDC submarine", "annualized_capex_per_mw_km"]
        hvdc_inverter_cost = costs.at["HVDC inverter pair", "annualized_capex_per_mw"]

        for link_idx in hvdc_links.index:
            link = n.links.loc[link_idx]

            # Get states and multipliers
            state0 = n.buses.at[link.bus0, state_col]
            state1 = n.buses.at[link.bus1, state_col]
            mult0 = state_multipliers.get(state0, 1.0)
            mult1 = state_multipliers.get(state1, 1.0)
            avg_multiplier = (mult0 + mult1) / 2.0

            # Calculate distance-related cost (subject to state multiplier)
            underwater_frac = link.underwater_fraction
            distance_cost_per_mw = link.length * (
                (1.0 - underwater_frac) * hvdc_overhead_cost + underwater_frac * hvdc_submarine_cost
            )
            distance_cost_adjusted = distance_cost_per_mw * avg_multiplier

            # Inverter cost (NOT subject to state multiplier)
            inverter_cost = hvdc_inverter_cost

            # Total capital cost (divide by 2 for bidirectional links)
            total_cost = (distance_cost_adjusted + inverter_cost) / 2

            n.links.at[link_idx, "capital_cost"] = total_cost

        logger.info(f"Recalculated costs for {len(hvdc_links)} HVDC links")

    # ========================================================================
    # Part 2: Process AC Lines
    # ========================================================================
    if not n.lines.empty:
        logger.info(f"Processing {len(n.lines)} AC lines")
        lines_updated = 0

        for line_idx in n.lines.index:
            line = n.lines.loc[line_idx]

            state0 = n.buses.at[line.bus0, state_col]
            state1 = n.buses.at[line.bus1, state_col]
            mult0 = state_multipliers.get(state0, 1.0)
            mult1 = state_multipliers.get(state1, 1.0)
            avg_multiplier = (mult0 + mult1) / 2.0

            if pd.notna(line.capital_cost) and line.capital_cost > 0:
                old_cost = line.capital_cost
                new_cost = old_cost * avg_multiplier
                n.lines.at[line_idx, "capital_cost"] = new_cost
                lines_updated += 1

        logger.info(f"Updated costs for {lines_updated} AC lines")

    # ========================================================================
    # Part 3: Process AC Links
    # ========================================================================
    ac_links = n.links[n.links.carrier == "AC"]

    if not ac_links.empty:
        logger.info(f"Processing {len(ac_links)} AC links")
        links_updated = 0

        for link_idx in ac_links.index:
            link = n.links.loc[link_idx]

            state0 = n.buses.at[link.bus0, state_col]
            state1 = n.buses.at[link.bus1, state_col]
            mult0 = state_multipliers.get(state0, 1.0)
            mult1 = state_multipliers.get(state1, 1.0)
            avg_multiplier = (mult0 + mult1) / 2.0

            if pd.notna(link.capital_cost) and link.capital_cost > 0:
                old_cost = link.capital_cost
                new_cost = old_cost * avg_multiplier
                n.links.at[link_idx, "capital_cost"] = new_cost
                links_updated += 1

        logger.info(f"Updated costs for {links_updated} AC links")

    # ========================================================================
    # Part 4: Process Gas Pipelines and H2 Pipelines Retrofit (No change)
    # ========================================================================
    # https://static1.squarespace.com/static/5b1032e545776e01e7058845/t/5cb37389c830257d563c0034/1555264398511/02%2BPhase%2BII.pdf Table 5.3

    # ========================================================================
    # Part 5: Process new H2/CO2 Pipelines
    # ========================================================================
    new_pipeline_carriers = ["h2 pipeline new", "co2 pipeline new"]
    new_pipelines = n.links[n.links.carrier.isin(new_pipeline_carriers)]

    if not new_pipelines.empty:
        logger.info(f"Processing {len(new_pipelines)} H2/CO2 pipelines")
        pipes_updated = 0

        for pipe_idx in new_pipelines.index:
            pipe = n.links.loc[pipe_idx]

            # Extract state from bus names (format: "STATE h2")
            bus0_parts = pipe.bus0.split()
            bus1_parts = pipe.bus1.split()

            state0 = bus0_parts[0] if len(bus0_parts) > 0 else None
            state1 = bus1_parts[0] if len(bus1_parts) > 0 else None

            if state0 is None or state1 is None:
                continue

            mult0 = state_multipliers.get(state0, 1.0)
            mult1 = state_multipliers.get(state1, 1.0)
            avg_multiplier = (mult0 + mult1) / 2.0

            if pd.notna(pipe.capital_cost) and pipe.capital_cost > 0:
                old_cost = pipe.capital_cost
                new_cost = old_cost * avg_multiplier
                n.links.at[pipe_idx, "capital_cost"] = new_cost
                pipes_updated += 1

        logger.info(f"Updated costs for {pipes_updated} H2/CO2 pipelines")


def add_land_use_constraints(n):
    """
    Adds constraint for land-use based on information from the generators
    table.

    Constraint is defined by land-use per carrier and land_region. The
    definition of land_region enables sub-bus level land-use
    constraints.
    """
    model = n.model
    generators = n.generators.query(
        "p_nom_extendable & land_region != '' ",
    ).rename_axis(index="Generator-ext")

    if generators.empty:
        return
    p_nom = n.model["Generator-p_nom"].loc[generators.index]

    grouper = pd.concat([generators.carrier, generators.land_region], axis=1)
    lhs = p_nom.groupby(grouper).sum()

    maximum = generators.groupby(["carrier", "land_region"])["p_nom_max"].max()
    maximum = maximum[np.isfinite(maximum)]

    rhs = xr.DataArray(maximum).rename(dim_0="group")
    index = rhs.indexes["group"].intersection(lhs.indexes["group"])

    if not index.empty:
        logger.info("Adding land-use constraints")
        model.add_constraints(
            lhs.sel(group=index) <= rhs.loc[index],
            name="land_use_constraint",
        )


def add_bidirectional_link_constraints(n):
    """
    Add constraints for bidirectional links (transmission and H2 pipelines).

    For pairs of extendable links with identical names except for 'fwd' and 'rev':
    Add constraint: fwd.p_nom_opt - fwd.p_nom = rev.p_nom_opt - rev.p_nom

    This ensures the two links model the same physical infrastructure.
    """
    # Get all extendable links
    extendable_links = n.links[n.links.p_nom_extendable].copy()

    # Find potential bidirectional link pairs
    # These are links that contain either '_fwd' or '_rev' at the end of their names
    bidirectional_candidates = extendable_links[
        extendable_links.index.str.contains(r"_fwd$|_rev$", regex=True, case=True)
    ]

    if bidirectional_candidates.empty:
        logger.info("No bidirectional link candidates found (no _fwd or _rev at the end of the names)")
        return

    # Group links by their base name (removing _fwd or _rev parts)
    link_pairs = {}

    for link_name in bidirectional_candidates.index:
        if "_fwd" in link_name:
            base_name = link_name.replace("_fwd", "", 1)
            if base_name not in link_pairs:
                link_pairs[base_name] = {}
            link_pairs[base_name]["fwd"] = link_name
        elif "_rev" in link_name:
            base_name = link_name.replace("_rev", "", 1)
            if base_name not in link_pairs:
                link_pairs[base_name] = {}
            link_pairs[base_name]["rev"] = link_name

    # Filter to only complete pairs (both fwd and rev exist)
    complete_pairs = {base_name: pair for base_name, pair in link_pairs.items() if "fwd" in pair and "rev" in pair}

    if not complete_pairs:
        logger.info("No complete bidirectional link pairs found")
        # Log the incomplete pairs for infoging
        incomplete_pairs = {k: v for k, v in link_pairs.items() if len(v) == 1}
        if incomplete_pairs:
            logger.info(f"Found {len(incomplete_pairs)} incomplete pairs:")
            for base_name, pair in incomplete_pairs.items():
                direction = list(pair.keys())[0]
                link_name = list(pair.values())[0]
                logger.info(f"  {base_name}: only {direction} link ({link_name})")
        return

    constraints_added = 0

    for base_name, pair in complete_pairs.items():
        fwd_link = pair["fwd"]
        rev_link = pair["rev"]

        # Get link properties
        fwd_p_nom = n.links.loc[fwd_link, "p_nom"]
        rev_p_nom = n.links.loc[rev_link, "p_nom"]

        # Get optimization variables
        fwd_p_nom_opt = n.model["Link-p_nom"].loc[fwd_link]
        rev_p_nom_opt = n.model["Link-p_nom"].loc[rev_link]

        # Add constraint: fwd.p_nom_opt - fwd.p_nom = rev.p_nom_opt - rev.p_nom
        constraint_name = f"bidirectional_link_{base_name.replace(' ', '_').replace('-', '_')}"
        lhs = fwd_p_nom_opt + rev_p_nom - fwd_p_nom - rev_p_nom_opt

        n.model.add_constraints(
            lhs == 0,
            name=constraint_name,
        )
        constraints_added += 1

    logger.info(f"Added {constraints_added} bidirectional link constraints")


def prepare_network(
    n,
    solve_opts=None,
):
    if "clip_p_max_pu" in solve_opts:
        clip_val = solve_opts["clip_p_max_pu"]

        n.generators_t.p_max_pu = n.generators_t.p_max_pu.where(
            n.generators_t.p_max_pu > clip_val,
            other=0.0,
        )

        n.generators_t.p_min_pu = n.generators_t.p_min_pu.where(
            n.generators_t.p_min_pu > clip_val,
            other=0.0,
        )

        n.storage_units_t.inflow = n.storage_units_t.inflow.where(
            n.storage_units_t.inflow > clip_val,
            other=0.0,
        )

    load_shedding = solve_opts.get("load_shedding")
    if load_shedding:
        # intersect between macroeconomic and surveybased willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
        # TODO: retrieve color and nice name from config
        logger.warning("Adding load shedding generators.")
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
        buses_i = n.buses.query("carrier == 'AC'").index
        if not np.isscalar(load_shedding):
            # TODO: do not scale via sign attribute (use Eur/MWh instead of Eur/kWh)
            load_shedding = 1e2  # Eur/kWh

        n.madd(
            "Generator",
            buses_i,
            " load",
            bus=buses_i,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=load_shedding,  # Eur/kWh
            p_nom=1e9,  # kW
        )

    if solve_opts.get("noisy_costs"):  ##random noise to costs of generators
        for t in n.iterate_components():
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (np.random.random(len(t.df)) - 0.5)

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    return n


def add_regional_co2limit(n, sns, config):
    """Adding regional regional CO2 Limits Specified in the config.yaml."""
    regional_co2_lims = pd.read_csv(
        config["electricity"]["regional_Co2_limits"],
        index_col=[0],
    )

    logger.info("Adding regional Co2 Limits.")

    # Filter the regional_co2_lims DataFrame based on the planning horizons present in the snapshots
    regional_co2_lims = regional_co2_lims[regional_co2_lims.planning_horizon.isin(sns.get_level_values(0))]
    weightings = n.snapshot_weightings.loc[n.snapshots]

    for idx, emmission_lim in regional_co2_lims.iterrows():
        region_list = [region.strip() for region in emmission_lim.regions.split(",")]
        region_buses = get_region_buses(n, region_list)

        emissions = n.carriers.co2_emissions.fillna(0)[lambda ds: ds != 0]
        region_gens = n.generators[n.generators.bus.isin(region_buses.index)]
        region_gens_em = region_gens.query("carrier in @emissions.index")

        if region_buses.empty or region_gens_em.empty:
            continue

        region_co2lim = emmission_lim.limit
        planning_horizon = emmission_lim.planning_horizon

        efficiency = get_as_dense(
            n,
            "Generator",
            "efficiency",
            inds=region_gens_em.index,
        )  # mw_elect/mw_th
        em_pu = region_gens_em.carrier.map(emissions) / efficiency  # tonnes_co2/mw_electrical
        em_pu = em_pu.multiply(weightings.generators, axis=0).loc[planning_horizon].fillna(0)

        # Emitting Gens
        p_em = n.model["Generator-p"].loc[:, region_gens_em.index].sel(period=planning_horizon)
        lhs = (p_em * em_pu).sum()
        rhs = region_co2lim

        n.model.add_constraints(
            lhs <= rhs,
            name=f"GlobalConstraint-{emmission_lim.name}_{planning_horizon}co2_limit",
        )

        logger.info(
            f"Adding regional Co2 Limit for {emmission_lim.name} in {planning_horizon}",
        )


def add_PRM_constraints(n, config):
    """
    Add Planning Reserve Margin (PRM) constraints for regional capacity adequacy.

    This function enforces that each region has sufficient firm capacity to meet
    peak demand plus a reserve margin. Only firm resources (not variable renewables
    or storage) contribute to meeting this requirement.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object
    config : dict
        Configuration dictionary containing PRM parameters
    """
    # Load regional PRM requirements
    regional_prm = _get_combined_prm_requirements(n, config)

    # Apply constraints for each region and planning horizon
    for _, prm in regional_prm.iterrows():
        region_list = [region_.strip() for region_ in prm.region.split(",")]
        region_buses = get_region_buses(n, region_list)

        # Calculate required reserve margin
        regional_demand = _get_regional_demand(n, prm.planning_horizon, region_buses)
        total_regional_demand = regional_demand.sum(axis=1)
        planning_reserve = total_regional_demand * (1.0 + prm.prm)

        # Get capacity contribution from resources
        total_extendable_contribution, total_nonextendable_contribution = _calculate_capacity_accreditation(
            n,
            prm.planning_horizon,
            region_buses,
        )

        # 2. StorageUnit Contribution (including nhr_battery_storage and PSH)
        active_su = n.get_active_assets("StorageUnit", prm.planning_horizon)
        region_su_mask = n.storage_units.bus.isin(region_buses.index)
        regional_active_su = n.storage_units[region_su_mask & active_su]

        extendable_su = regional_active_su[regional_active_su.p_nom_extendable]
        nonextendable_su = regional_active_su[~regional_active_su.p_nom_extendable]

        if not extendable_su.empty:
            p_nom_su = n.model["StorageUnit-p_nom"].loc[extendable_su.index]
            credit = pd.Series(1.0, index=extendable_su.index)
            batt_mask = extendable_su.carrier.str.contains("battery")
            credit.loc[batt_mask] = (extendable_su.loc[batt_mask, "max_hours"] / 4).clip(upper=1.0)
            hours = n.snapshots.get_level_values(1)
            credit_matrix = pd.DataFrame(
                np.outer(np.ones(len(hours)), credit),
                index=hours,
                columns=extendable_su.index,
            )
            credit_matrix.T.index.name = "StorageUnit-ext"
            total_extendable_contribution += (p_nom_su * credit_matrix).sum(dim="StorageUnit-ext")

        if not nonextendable_su.empty:
            p_nom_su = nonextendable_su.p_nom
            credit = pd.Series(1.0, index=nonextendable_su.index)
            batt_mask = nonextendable_su.carrier.str.contains("battery")
            credit.loc[batt_mask] = (nonextendable_su.loc[batt_mask, "max_hours"] / 4).clip(upper=1.0)
            credit.T.index.name = "StorageUnit-ext"
            total_nonextendable_contribution += (p_nom_su * credit).sum()

        # 3. Link Contribution (h2 turbine, acaes discharge, tes discharge)
        link_carriers = ["h2 turbine", "acaes retrofit discharge", "acaes new discharge", "tes discharge"]
        active_links = n.get_active_assets("Link", prm.planning_horizon)
        region_links_mask = n.links.bus1.isin(region_buses.index)
        carrier_mask = n.links.carrier.str.contains("|".join(link_carriers))
        regional_active_links = n.links[region_links_mask & carrier_mask & active_links]

        extendable_links = regional_active_links[regional_active_links.p_nom_extendable]
        nonextendable_links = regional_active_links[~regional_active_links.p_nom_extendable]

        # Process extendable links
        if not extendable_links.empty:
            p_nom_links = n.model["Link-p_nom"].loc[extendable_links.index]
            discharge_efficiency = extendable_links.efficiency

            # Check if links have time-varying p_max_pu
            if (
                hasattr(n, "links_t")
                and "p_max_pu" in n.links_t
                and any(link in n.links_t.p_max_pu.columns for link in extendable_links.index)
            ):
                # Get time-varying p_max_pu for links that have it
                p_max_pu_links = get_as_dense(n, "Link", "p_max_pu", inds=extendable_links.index)
                p_max_pu_links = p_max_pu_links.loc[prm.planning_horizon]

                # Multiply efficiency by time-varying p_max_pu
                discharge_efficiency_adj = p_max_pu_links.multiply(discharge_efficiency, axis=1)
                discharge_efficiency_adj.T.index.name = "Link-ext"
                total_extendable_contribution += (p_nom_links * discharge_efficiency_adj).sum(dim="Link-ext")
            else:
                # Use static efficiency if no time-varying p_max_pu
                discharge_efficiency.T.index.name = "Link-ext"
                total_extendable_contribution += (p_nom_links * discharge_efficiency).sum(dim="Link-ext")

        # Process non-extendable links
        if not nonextendable_links.empty:
            p_nom_links = nonextendable_links.p_nom
            discharge_efficiency = nonextendable_links.efficiency

            # Check if links have time-varying p_max_pu
            if (
                hasattr(n, "links_t")
                and "p_max_pu" in n.links_t
                and any(link in n.links_t.p_max_pu.columns for link in nonextendable_links.index)
            ):
                # Get time-varying p_max_pu for links that have it
                p_max_pu_links = get_as_dense(n, "Link", "p_max_pu", inds=nonextendable_links.index)
                p_max_pu_links = p_max_pu_links.loc[prm.planning_horizon]

                # Multiply efficiency by time-varying p_max_pu
                discharge_efficiency_adj = p_max_pu_links.multiply(discharge_efficiency, axis=1)
                total_nonextendable_contribution += (p_nom_links * discharge_efficiency_adj).sum(axis=1)
            else:
                # Use static efficiency if no time-varying p_max_pu
                total_nonextendable_contribution += (p_nom_links * discharge_efficiency).sum()

        # Add the constraint to the model
        n.model.add_constraints(
            total_extendable_contribution >= planning_reserve - total_nonextendable_contribution,
            name=f"GlobalConstraint-{prm.name}_{prm.planning_horizon}_PRM",
        )
        # snapshots = n.snapshots.get_level_values(1)[n.snapshots.get_level_values(0) == prm.planning_horizon]
        #
        # for i, snapshot in enumerate(snapshots):
        #     constraint_name = f"GlobalConstraint-{prm.name}_{prm.planning_horizon}_PRM_t{i:04d}"
        #     lhs_t = lhs_capacity.sel(timestep=snapshot)
        #
        #     rhs_t = planning_reserve.iloc[i] - (rhs_existing.iloc[i] if hasattr(rhs_existing, 'iloc') else rhs_existing)
        #
        #     n.model.add_constraints(
        #         lhs_t >= rhs_t,
        #         name=constraint_name
        #     )

        logger.info(
            f"Added PRM constraint for {prm.name} in {prm.planning_horizon} for all time steps.",
        )


def add_ERM_constraints(n, config):
    """
    Add Energy Reserve Margin (ERM) constraints for individual state buses.

    This function enforces that each state has sufficient capacity to meet
    demand plus a reserve margin at each timestep, considering storage
    state of charge constraints and time-varying p_max_pu.

    Modified to apply constraints at the state level rather than regional level.

    The constraint is:
    extendable_gen + storage_unit_reserve + nonstore_link_reserve + store_link_reserve
    >= planning_reserve - non_extendable_gen

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object
    config : dict
        Configuration dictionary containing ERM parameters
    """
    # Load regional ERM requirements (reuse PRM structure)
    regional_erm = _get_combined_prm_requirements(n, config)

    # Apply constraints for each region and planning horizon
    for _, erm in regional_erm.iterrows():
        region_list = [region_.strip() for region_ in erm.region.split(",")]
        region_buses = get_region_buses(n, region_list)

        # Get unique states within this region
        states_in_region = region_buses["reeds_state"].dropna().unique()

        logger.info(
            f"Adding ERM constraints for {len(states_in_region)} states in region {erm.name} for planning horizon {erm.planning_horizon}",
        )

        # Apply ERM constraint to each state individually
        for state in states_in_region:
            # Get buses for this specific state
            state_buses = region_buses[region_buses["reeds_state"] == state]

            # Create unique identifier for this state-horizon combination
            state_id = f"{state}_{erm.planning_horizon}".replace(" ", "_").replace("-", "_")

            # Calculate required reserve margin for each timestep for this state
            state_demand = _get_regional_demand(n, erm.planning_horizon, state_buses)
            planning_reserve = state_demand.sum(axis=1) * (1.0 + erm.prm)  # Using prm field for ERM ratio

            # Get capacity contribution from generators for this state
            total_extendable_contribution, total_nonextendable_contribution = _calculate_capacity_accreditation(
                n,
                erm.planning_horizon,
                state_buses,
            )

            # Get timesteps for this planning horizon
            snapshots = n.snapshots.get_level_values(1)[n.snapshots.get_level_values(0) == erm.planning_horizon]

            # 1. Storage Unit Reserve Variables and Constraints for this state
            active_su = n.get_active_assets("StorageUnit", erm.planning_horizon)
            state_su_mask = n.storage_units.bus.isin(state_buses.index)
            state_active_su = n.storage_units[state_su_mask & active_su]

            storage_unit_reserve_total = 0
            if not state_active_su.empty:
                # Add storage unit reserve variables with unique name for this state
                storage_reserve_coords = [state_active_su.index, snapshots]
                storage_var_name = f"storage_unit_reserve_{state_id}"
                n.model.add_variables(
                    coords=storage_reserve_coords,
                    name=storage_var_name,
                )
                storage_unit_reserve = n.model[storage_var_name]

                # Constraints for storage unit reserve
                for su in state_active_su.index:
                    if state_active_su.loc[su, "p_nom_extendable"]:
                        p_nom_su = n.model["StorageUnit-p_nom"].loc[su]
                    else:
                        p_nom_su = state_active_su.loc[su, "p_nom"]

                    # storage_unit_reserve <= p_nom_opt
                    n.model.add_constraints(
                        storage_unit_reserve.loc[su, :] <= p_nom_su,
                        name=f"ERM_storage_reserve_capacity_{su.replace(' ', '_')}_{state_id}",
                    )

                    # storage_unit_reserve / efficiency_dispatch <= e
                    e_su = n.model["StorageUnit-state_of_charge"].loc[erm.planning_horizon, su]
                    efficiency_dispatch = state_active_su.loc[su, "efficiency_dispatch"]

                    n.model.add_constraints(
                        storage_unit_reserve.loc[su, :] / efficiency_dispatch - e_su <= 0,
                        name=f"ERM_storage_reserve_soc_{su.replace(' ', '_')}_{state_id}",
                    )

                storage_unit_reserve_total = storage_unit_reserve.sum(dim="StorageUnit")

            # 2. Store Link Reserve Variables and Constraints (for ACAES/TES discharge links) for this state
            ldes_carriers = ["acaes retrofit", "acaes new", "tes"]
            active_links = n.get_active_assets("Link", erm.planning_horizon)
            # bus1 is in state and bus0 is not in state
            state_links_mask = n.links.bus1.isin(state_buses.index) & ~n.links.bus0.isin(state_buses.index)
            ldes_mask = n.links.carrier.isin(ldes_carriers)
            state_ldes_links = n.links[state_links_mask & ldes_mask & active_links]

            store_link_reserve_total = 0
            if not state_ldes_links.empty:
                # Add store link reserve variables with unique name for this state
                store_reserve_coords = [state_ldes_links.index, snapshots]
                store_var_name = f"store_link_reserve_{state_id}"
                n.model.add_variables(
                    coords=store_reserve_coords,
                    name=store_var_name,
                    lower=0,
                )
                store_link_reserve = n.model[store_var_name]

                # Constraints for store link reserve
                for link in state_ldes_links.index:
                    if state_ldes_links.loc[link, "p_nom_extendable"]:
                        p_nom_link = n.model["Link-p_nom"].loc[link]
                    else:
                        p_nom_link = state_ldes_links.loc[link, "p_nom"]

                    efficiency_link = state_ldes_links.loc[link, "efficiency"]

                    # store_link_reserve <= p_nom_opt * efficiency_link
                    n.model.add_constraints(
                        store_link_reserve.loc[link, :] <= p_nom_link * efficiency_link,
                        name=f"ERM_store_link_capacity_{link.replace(' ', '_')}_{state_id}",
                    )

                    # store_link_reserve / efficiency_link <= e
                    # Find the connected store from bus0
                    store_bus = state_ldes_links.loc[link, "bus0"]
                    # Find store connected to this bus
                    connected_stores = n.stores[n.stores.bus == store_bus]
                    if not connected_stores.empty:
                        store_name = connected_stores.index[0]  # Assume one store per bus
                        e_store = n.model["Store-e"].loc[erm.planning_horizon, store_name]

                        n.model.add_constraints(
                            store_link_reserve.loc[link, :] / efficiency_link - e_store <= 0,
                            name=f"ERM_store_link_soc_{link.replace(' ', '_')}_{state_id}",
                        )

                store_link_reserve_total = store_link_reserve.sum(dim="Link")

            # 3. Non-store Link Contributions for this state
            state_other_links = n.links[state_links_mask & ~ldes_mask & active_links]
            nonstore_link_reserve_total = 0
            if not state_other_links.empty:
                extendable_other_links = state_other_links[state_other_links.p_nom_extendable]
                nonextendable_other_links = state_other_links[~state_other_links.p_nom_extendable]

                # Process extendable links
                if not extendable_other_links.empty:
                    p_nom_other_links = n.model["Link-p_nom"].loc[extendable_other_links.index]
                    efficiency_other = extendable_other_links.efficiency
                    p_max_pu_other = extendable_other_links.p_max_pu

                    # Check for time-varying p_max_pu for extendable links
                    if hasattr(n, "links_t") and "p_max_pu" in n.links_t:
                        timevar_links = [
                            link for link in extendable_other_links.index if link in n.links_t.p_max_pu.columns
                        ]
                        static_links = [
                            link for link in extendable_other_links.index if link not in n.links_t.p_max_pu.columns
                        ]

                        total_contribution = 0

                        # Process time-varying links
                        if timevar_links:
                            timevar_links_index = pd.Index(timevar_links)
                            p_max_pu_timevar = get_as_dense(n, "Link", "p_max_pu", inds=timevar_links_index)
                            p_max_pu_timevar = p_max_pu_timevar.loc[erm.planning_horizon]

                            p_nom_timevar = p_nom_other_links.loc[timevar_links]
                            efficiency_timevar = efficiency_other.loc[timevar_links]

                            # Ensure proper indexing for vectorized operations
                            efficiency_timevar.index.name = "Link-ext"

                            # First calculate efficiency * p_max_pu for each link and timestep
                            # This creates a DataFrame with shape (timesteps, links)
                            efficiency_p_max_pu = p_max_pu_timevar.multiply(efficiency_timevar, axis=1)

                            # Convert to xarray with proper coordinates for linopy operations
                            efficiency_p_max_pu_xr = xr.DataArray(
                                efficiency_p_max_pu.values.T,  # Transpose to get (links, timesteps)
                                coords=[timevar_links, efficiency_p_max_pu.index],
                                dims=["Link-ext", "timestep"],
                            )

                            # Now multiply with p_nom_timevar and sum over links
                            timevar_contribution = (p_nom_timevar * efficiency_p_max_pu_xr).sum(dim="Link-ext")
                            total_contribution += timevar_contribution

                        # Process static links (no time-varying p_max_pu)
                        if static_links:
                            p_nom_static = p_nom_other_links.loc[static_links]
                            efficiency_static = efficiency_other.loc[static_links]
                            p_max_pu_static = p_max_pu_other.loc[static_links]
                            efficiency_static.index.name = "Link-ext"
                            p_max_pu_static.index.name = "Link-ext"
                            static_contribution = (p_nom_static * efficiency_static * p_max_pu_static).sum(
                                dim="Link-ext"
                            )
                            total_contribution += static_contribution

                        nonstore_link_reserve_total += total_contribution
                    else:
                        # Original logic if no time-varying p_max_pu exists
                        efficiency_other.index.name = "Link-ext"
                        p_max_pu_other.index.name = "Link-ext"
                        nonstore_link_reserve_total += (p_nom_other_links * efficiency_other * p_max_pu_other).sum(
                            dim="Link-ext"
                        )

                # Process non-extendable links
                if not nonextendable_other_links.empty:
                    p_nom_other_links = nonextendable_other_links.p_nom
                    efficiency_other = nonextendable_other_links.efficiency
                    p_max_pu_other = nonextendable_other_links.p_max_pu

                    # Check if links have time-varying p_max_pu
                    if hasattr(n, "links_t") and "p_max_pu" in n.links_t:
                        timevar_links = [
                            link for link in nonextendable_other_links.index if link in n.links_t.p_max_pu.columns
                        ]
                        static_links = [
                            link for link in nonextendable_other_links.index if link not in n.links_t.p_max_pu.columns
                        ]

                        # Process time-varying links
                        if timevar_links:
                            timevar_links_index = pd.Index(timevar_links)
                            p_max_pu_timevar = get_as_dense(n, "Link", "p_max_pu", inds=timevar_links_index)
                            p_max_pu_timevar = p_max_pu_timevar.loc[erm.planning_horizon]

                            p_nom_timevar = p_nom_other_links.loc[timevar_links]
                            efficiency_timevar = efficiency_other.loc[timevar_links]

                            # Multiply efficiency by time-varying p_max_pu for each timestep
                            efficiency_p_max_pu = p_max_pu_timevar.multiply(efficiency_timevar, axis=1)
                            timevar_contribution = (p_nom_timevar * efficiency_p_max_pu).sum(axis=1)
                            nonstore_link_reserve_total += timevar_contribution

                        # Process static links
                        if static_links:
                            p_nom_static = p_nom_other_links.loc[static_links]
                            efficiency_static = efficiency_other.loc[static_links]
                            p_max_pu_static = p_max_pu_other.loc[static_links]
                            static_contribution = (p_nom_static * efficiency_static * p_max_pu_static).sum()
                            nonstore_link_reserve_total += static_contribution
                    else:
                        # Original logic if no time-varying p_max_pu exists
                        nonstore_link_reserve_total += (p_nom_other_links * efficiency_other * p_max_pu_other).sum()

            # 4. Line Contributions for this state
            active_lines = n.get_active_assets("Line", erm.planning_horizon)

            # Lines that connect to the state (at least one end in state, one end potentially outside)
            supply_lines_mask = (n.lines.bus0.isin(state_buses.index) | n.lines.bus1.isin(state_buses.index)) & ~(
                n.lines.bus0.isin(state_buses.index) & n.lines.bus1.isin(state_buses.index)
            )

            state_supply_lines = n.lines[supply_lines_mask & active_lines]

            extendable_lines = state_supply_lines[state_supply_lines.s_nom_extendable]
            nonextendable_lines = state_supply_lines[~state_supply_lines.s_nom_extendable]

            line_extendable_contribution = 0
            line_nonextendable_contribution = 0

            # --- Extendable Lines ---
            if not extendable_lines.empty:
                ext_s_nom = n.model["Line-s_nom"].loc[extendable_lines.index]
                ext_s_max_pu = extendable_lines.s_max_pu
                ext_s_max_pu.index.name = "Line-ext"

                line_extendable_contribution = (ext_s_nom * ext_s_max_pu).sum(dim="Line-ext")

            # --- Non-Extendable Lines ---
            if not nonextendable_lines.empty:
                non_ext_s_nom = nonextendable_lines.s_nom
                non_ext_s_max_pu = nonextendable_lines.s_max_pu

                line_nonextendable_contribution = (non_ext_s_nom * non_ext_s_max_pu).sum()

            # 5. Add the main ERM constraint for this state for ALL timesteps at once (vectorized)
            constraint_name = f"ERM_Constraint_{state}_{erm.planning_horizon}"

            # Left hand side: all capacity contributions across all timesteps
            lhs_capacity = (
                total_extendable_contribution
                + line_extendable_contribution
                + storage_unit_reserve_total
                + store_link_reserve_total
                + nonstore_link_reserve_total
            )

            # Right hand side: demand plus reserve minus non-extendable capacity for all timesteps
            rhs = planning_reserve - total_nonextendable_contribution - line_nonextendable_contribution

            # Add vectorized constraint for all timesteps at once
            n.model.add_constraints(
                lhs_capacity >= rhs,
                name=constraint_name,
            )


STATE_TO_NERC_REGION = {
    # NPCC
    "ME": "NPCC_NE",
    "NH": "NPCC_NE",
    "VT": "NPCC_NE",
    "MA": "NPCC_NE",
    "RI": "NPCC_NE",
    "CT": "NPCC_NE",
    "NY": "NPCC_NY",
    # PJM
    "PA": "PJM",
    "NJ": "PJM",
    "DE": "PJM",
    "MD": "PJM",
    "OH": "PJM",
    "WV": "PJM",
    # MISO
    "MI": "MISO",
    "WI": "MISO",
    "MN": "MISO",
    "IA": "MISO",
    "IL": "MISO",
    "IN": "MISO",
    "MO": "MISO",
    "ND": "MISO",
    "SD": "MISO",
    # SERC
    "KY": "SERC",
    "VA": "SERC",
    "NC": "SERC",
    "SC": "SERC",
    "TN": "SERC",
    "GA": "SERC",
    "AL": "SERC",
    "MS": "SERC",
    "FL": "SERC",
    "AR": "SERC",
    "LA": "SERC",
    # SPP
    "KS": "SPP",
    "OK": "SPP",
    "NE": "SPP",
    # ERCOT
    "TX": "ERCOT",
    # WECC
    "WA": "WECC_NWPP",
    "OR": "WECC_NWPP",
    "ID": "WECC_NWPP",
    "MT": "WECC_NWPP",
    "WY": "WECC_NWPP",
    "UT": "WECC_NWPP",
    "CA": "WECC_CA",
    "NV": "WECC_SRSG",
    "AZ": "WECC_SRSG",
    "CO": "WECC_SRSG",
    "NM": "WECC_SRSG",
}


def _get_combined_prm_requirements(n, config):
    """
    Combine PRM requirements from different sources into a single dataframe.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Returns
    -------
    pd.DataFrame
        Combined PRM requirements with columns: name, region, prm, planning_horizon
    """
    # Load user-defined PRM requirements
    # regional_prm = pd.read_csv(
    #     config["electricity"]["SAFE_regional_reservemargins"],
    #     index_col=[0],
    # )

    # Process ReEDS PRM data if available
    try:
        reeds_prm = pd.read_csv(snakemake.input.safer_reeds, index_col=[0])

        buses_df = n.buses[n.buses.carrier == "AC"].copy()
        buses_df["nerc_reg"] = buses_df["STATE"].map(STATE_TO_NERC_REGION)
        nerc_to_states_map = (
            buses_df.groupby("nerc_reg")["STATE"].unique().apply(lambda x: ", ".join(sorted(x))).to_dict()
        )
        reeds_prm["region"] = reeds_prm.index.map(nerc_to_states_map)
        reeds_prm = reeds_prm.dropna(subset="region")
        reeds_prm = reeds_prm.drop(
            columns=["none", "ramp2025_20by50", "ramp2025_25by50", "ramp2025_30by50"],
        )
        reeds_prm = reeds_prm.rename(columns={"static": "prm", "t": "planning_horizon"})

        # Combine both data sources
        # regional_prm = pd.concat([regional_prm, reeds_prm])
        regional_prm = reeds_prm
    except (FileNotFoundError, AttributeError):
        logger.info("ReEDS PRM data not available, using only user-defined PRM values")

    # Filter for relevant planning horizons
    return regional_prm[regional_prm.planning_horizon.isin(n.investment_periods)]


def _get_regional_demand(n, planning_horizon, region_buses):
    """
    Calculate hourly demand for a specific region and planning horizon.

    Parameters
    ----------
    n : pypsa.Network
    planning_horizon : int or str
        Planning horizon year
    region_buses : pd.DataFrame
        DataFrame containing buses in the region

    Returns
    -------
    pd.Series
        Hourly demand series for the region
    """
    return n.loads_t.p_set.loc[
        planning_horizon,
        n.loads.bus.isin(region_buses.index),
    ]


def _calculate_capacity_accreditation(n, planning_horizon, region_buses):
    """
    Calculate capacity accreditation for generators, storage units, and links.
    """
    # Initialize total contributions as linopy expressions or floats
    extendable_contribution = 0
    nonextendable_contribution = 0

    # 1. Generator Contribution
    active_gens = n.get_active_assets("Generator", planning_horizon)
    region_gens_mask = n.generators.bus.isin(region_buses.index)
    regional_active_gens = n.generators[region_gens_mask & active_gens]

    extendable_gens = regional_active_gens[regional_active_gens.p_nom_extendable]
    nonextendable_gens = regional_active_gens[~regional_active_gens.p_nom_extendable]

    # --- Extendable Generators ---
    if not extendable_gens.empty:
        ext_p_nom = n.model["Generator-p_nom"].loc[extendable_gens.index]
        ext_p_max_pu = get_as_dense(n, "Generator", "p_max_pu", inds=extendable_gens.index)
        ext_p_max_pu = ext_p_max_pu.loc[planning_horizon]
        ext_p_max_pu.T.index.name = "Generator-ext"

        extendable_contribution += (ext_p_nom * ext_p_max_pu).sum(dim="Generator-ext")

    # --- Non-Extendable Generators ---
    if not nonextendable_gens.empty:
        non_ext_p_nom = nonextendable_gens.p_nom
        non_ext_p_max_pu = get_as_dense(n, "Generator", "p_max_pu", inds=nonextendable_gens.index)
        non_ext_p_max_pu = non_ext_p_max_pu.loc[planning_horizon]

        nonextendable_contribution += (non_ext_p_nom * non_ext_p_max_pu).sum(axis=1)

    return extendable_contribution, nonextendable_contribution


def add_co2_constraints(n, config, reference_network_path=None):
    """
    Adds national CO2 constraints based on NZA emission scenarios.

    When only_power is enabled, applies a carbon price (from reference network's
    co2 bus marginal prices) to the objective function instead of an emission cap.

    This is a simplified version of add_sector_co2_constraints that only
    considers national-level total carbon emission constraints.

    Parameters
    ----------
        n : pypsa.Network
        config : dict
        reference_network_path : str, optional
            Path to reference network for extracting carbon prices (only_power mode)
    """

    def apply_national_co2_limit(n: pypsa.Network, year: int, value: float):
        """
        Apply national CO2 limit for the specified year.
        For every snapshot, sum of co2 and ch4 must be less than limit.
        """
        # Find all CO2 and CH4 stores at national level (no state prefix filtering)
        stores = n.stores[(n.stores.index.str.endswith("co2 atmosphere")) | (n.stores.index.str.endswith("ch4"))].index

        name = f"co2_limit-{year}"
        log_statement = f"Adding national co2 Limit in {year} of"

        # Apply constraint to final snapshot (cumulative emissions)
        lhs = n.model["Store-e"].loc[:, stores].sel(snapshot=n.snapshots[-1]).sum(dim="Store")
        rhs = value  # value in T CO2

        n.model.add_constraints(lhs <= rhs, name=name)

        logger.info(f"{log_statement} {rhs * 1e-6} MMT CO2")

    def apply_carbon_price_from_reference_network(n: pypsa.Network, reference_network_path: str):
        """
        Extract carbon price from reference network and add penalties to objective function.

        This function:
        1. Loads the reference network
        2. Extracts carbon price from co2 bus marginal prices
        3. Adds carbon_price * final_co2_emissions to the objective function

        Parameters
        ----------
        n : pypsa.Network
            Current network to modify
        reference_network_path : str
            Path to the reference solved network
        """
        # Load the reference network
        n_ref = pypsa.Network(reference_network_path)

        # ========== Extract and apply carbon price ==========
        # Extract CO2 marginal prices from the reference network's co2 buses
        co2_buses = n_ref.buses[n_ref.buses.carrier == "co2"].index

        # Get co2 atmosphere buses
        co2_atmosphere_buses = [b for b in co2_buses if "atmosphere" in b]

        # Get the marginal price from buses_t.marginal_price
        co2_marginal_prices = n_ref.buses_t.marginal_price[co2_atmosphere_buses]

        # Take the mean across all states and time periods
        carbon_price = -co2_marginal_prices.mean().mean()

        # Find all CO2 atmosphere stores in current network
        co2_atmosphere_stores = n.stores[n.stores.index.str.endswith("co2 atmosphere")].index

        # Get final snapshot
        final_snapshot = n.snapshots[-1]

        # Get CO2 storage levels at final timestep
        final_co2_levels = n.model["Store-e"].loc[final_snapshot, co2_atmosphere_stores]

        # Add carbon price penalty to objective
        # Positive carbon_price penalizes emissions (positive storage in atmosphere)
        carbon_penalty = carbon_price * final_co2_levels.sum()

        # Modify objective function
        n.model.objective.expression += carbon_penalty

        logger.info(
            f"Added carbon price penalty of {carbon_price:.2f} USD/tCO2 to objective function "
            f"for {len(co2_atmosphere_stores)} CO2 atmosphere stores at final timestep",
        )

    # Check if only_power mode is enabled
    only_power_enabled = config.get("sector", {}).get("only_power", {}).get("enable", False)

    if only_power_enabled:
        logger.info("Only power mode enabled - applying carbon price to objective function instead of emission cap")
        apply_carbon_price_from_reference_network(n, reference_network_path)
        return

    # Try to get the CO2 policy file path from config
    try:
        base_policy_path = config["sector"]["co2"]["policy"]

        # If only_power is enabled, use the power-specific emissions file
        if only_power_enabled:
            # Replace nza_emissions.csv with nza_emissions_power.csv
            f = base_policy_path.replace("nza_emissions.csv", "nza_emissions_power.csv")
            logger.info(f"Only power mode enabled - using power-specific emission caps: {f}")
        else:
            f = base_policy_path

    except KeyError:
        logger.error("No co2 policy constraint file found")
        return

    # Read the NZA emissions CSV file
    try:
        df = pd.read_csv(f)
    except FileNotFoundError:
        logger.error(f"CO2 policy file not found: {f}")
        return

    if df.empty:
        logger.warning("No co2 policies applied - CSV file is empty")
        return

    # Get the NZA scenario from config
    try:
        nza_scenario = config["scenario"]["nza_scenario"]
    except KeyError:
        logger.error("No nza_scenario specified in config. Please add config['scenario']['nza_scenario']")
        logger.info(f"Available scenarios: {df.scenario.unique().tolist()}")
        return

    # Filter for the specified scenario
    df_scenario = df[df.scenario == nza_scenario]

    if df_scenario.empty:
        logger.error(f"No data found for scenario: {nza_scenario}")
        logger.info(f"Available scenarios: {df.scenario.unique().tolist()}")
        return

    # Filter for years that exist in the network's investment periods
    # For 2050 specifically as requested
    target_years = [year for year in df_scenario.year.unique() if year in n.investment_periods]

    if not target_years:
        logger.warning("No matching years found between scenario data and network investment periods")
        logger.info(f"Scenario years: {df_scenario.year.unique().tolist()}")
        logger.info(f"Network investment periods: {n.investment_periods}")
        return

    # Get emission cap reduction from adj_scenario (if any)
    emission_cap_reduction = getattr(n, "adj_scenario_emission_reduction", 0.0)

    # Apply constraints for each year
    for year in target_years:
        df_year = df_scenario[df_scenario.year == year].reset_index(drop=True)

        if df_year.empty:
            continue

        # Get the emission cap (in MMT CO2)
        emission_cap = df_year.loc[0, "emission_cap"]

        # Apply adjustment from adj_scenario
        # emission_cap_reduction is in ton CO2, convert to match units
        adjusted_emission_cap = emission_cap * 1e6 - emission_cap_reduction

        if emission_cap_reduction > 0:
            logger.info(
                f"Adjusting emission cap for {year}: "
                f"{emission_cap * 1e6:.2f} - {emission_cap_reduction:.2f} = {adjusted_emission_cap:.2f} ton CO2",
            )

        # Apply the national constraint with adjusted cap
        apply_national_co2_limit(n, year, adjusted_emission_cap)

    logger.info(f"Applied CO2 constraints for scenario '{nza_scenario}' using data from {f}")


def add_ng_import_export_limits(n, config):
    def _format_link_name(s: str) -> str:
        states = s.split("-")
        return f"{states[0]} {states[1]} gas"

    def _format_data(
        prod: pd.DataFrame,
        link_suffix: str | None = None,
    ) -> pd.DataFrame:
        df = prod.copy()
        df["link"] = df.state.map(_format_link_name)
        if link_suffix:
            df["link"] = df.link + link_suffix

        # convert mmcf to MWh
        df["value"] = df["value"] * NG_MWH_2_MMCF

        return df[["link", "value"]].rename(columns={"value": "rhs"}).set_index("link")

    def add_import_limits(n, data, constraint, multiplier=None):
        """Sets gas import limit over each year."""
        assert constraint in ("max", "min")

        if not multiplier:
            multiplier = 1

        weights = n.snapshot_weightings.objective

        links = n.links[(n.links.carrier == "gas trade") & (n.links.bus0.str.endswith(" gas trade"))].index.to_list()

        for year in n.investment_periods:
            for link in links:
                try:
                    rhs = data.at[link, "rhs"] * multiplier
                except KeyError:
                    # logger.warning(f"Can not set gas import limit for {link}")
                    continue
                lhs = n.model["Link-p"].mul(weights).sel(snapshot=year, Link=link).sum()

                if constraint == "min":
                    n.model.add_constraints(
                        lhs >= rhs,
                        name=f"ng_limit_import_min-{year}-{link}",
                    )
                else:
                    n.model.add_constraints(
                        lhs <= rhs,
                        name=f"ng_limit_import_max-{year}-{link}",
                    )

    def add_export_limits(n, data, constraint, multiplier=None):
        """Sets maximum export limit over the year."""
        assert constraint in ("max", "min")

        if not multiplier:
            multiplier = 1

        weights = n.snapshot_weightings.objective

        links = n.links[(n.links.carrier == "gas trade") & (n.links.bus0.str.endswith(" gas"))].index.to_list()

        for year in n.investment_periods:
            for link in links:
                try:
                    rhs = data.at[link, "rhs"] * multiplier
                except KeyError:
                    # logger.warning(f"Can not set gas import limit for {link}")
                    continue
                lhs = n.model["Link-p"].mul(weights).sel(snapshot=year, Link=link).sum()

                if constraint == "min":
                    n.model.add_constraints(
                        lhs >= rhs,
                        name=f"ng_limit_export_min-{year}-{link}",
                    )
                else:
                    n.model.add_constraints(
                        lhs <= rhs,
                        name=f"ng_limit_export_max-{year}-{link}",
                    )

    api = config["api"]["eia"]
    year = pd.to_datetime(config["snapshots"]["start"]).year

    # get limits

    import_min = config["sector"]["natural_gas"]["imports"].get("min", 1)
    import_max = config["sector"]["natural_gas"]["imports"].get("max", 1)
    export_min = config["sector"]["natural_gas"]["exports"].get("min", 1)
    export_max = config["sector"]["natural_gas"]["exports"].get("max", 1)

    # to avoid numerical issues, ensure there is a gap between min/max constraints
    if import_max == "inf":
        pass
    elif abs(import_max - import_min) < 0.0001:
        import_min -= 0.001
        import_max += 0.001
        if import_min < 0:
            import_min = 0

    if export_max == "inf":
        pass
    elif abs(export_max - export_min) < 0.0001:
        export_min -= 0.001
        export_max += 0.001
        if export_min < 0:
            export_min = 0

    # import and export dataframes contain the same information, just in different formats
    # ie. imports from one S1 -> S2 are the same as exports from S2 -> S1
    # we use the exports direction to set limits

    # add domestic limits

    trade = Trade("gas", False, "exports", year, api).get_data()
    trade = _format_data(trade, " trade")

    add_import_limits(n, trade, "min", import_min)
    add_export_limits(n, trade, "min", export_min)

    if not import_max == "inf":
        add_import_limits(n, trade, "max", import_max)
    if not export_max == "inf":
        add_export_limits(n, trade, "max", export_max)

    # add international limits

    trade = Trade("gas", True, "exports", year, api).get_data()
    trade = _format_data(trade, " trade")

    add_import_limits(n, trade, "min", import_min)
    add_export_limits(n, trade, "min", export_min)

    if not import_max == "inf":
        add_import_limits(n, trade, "max", import_max)
    if not export_max == "inf":
        add_export_limits(n, trade, "max", export_max)


def add_gas_storage_retrofit_constraints(n, config):
    """
    Add constraints for retrofit of gas infrastructure.
    Updated to handle continuous mode only with new mathematical constraints.
    """
    # Get retrofit parameters from config
    ng_options = config.get("sector", {}).get("natural_gas", {})
    retro_storage_h2 = ng_options.get("retro_storage_h2", False)
    retro_storage_acaes = ng_options.get("retro_storage_acaes", False)

    # If neither H2 nor ACAES retrofit is enabled, return early
    if not retro_storage_h2 and not retro_storage_acaes:
        return

    # Get retrofit factors
    storage_eng_retro_factor_h2 = ng_options.get("storage_eng_retro_factor_h2", 0.2551)  # α_eng,H2
    storage_pow_retro_factor_h2 = ng_options.get("storage_pow_retro_factor_h2", 0.4)  # α_pow,H2
    storage_eng_retro_factor_acaes = ng_options.get("storage_eng_retro_factor_acaes", 0.0093)  # α_eng,air

    logger.info("Adding storage retrofit constraints")

    # Get aggregation level (0, 1)
    agg_storage = ng_options.get("aggregate_storage", 1)

    # Identify retrofit gas storage
    gas_stores = n.stores[n.stores.carrier.str.contains("gas storage") & ~n.stores.carrier.str.contains("non")]

    # Identify H2 and ACAES storage if enabled
    h2_stores = None
    acaes_stores = None

    if retro_storage_h2:
        h2_stores = n.stores[n.stores.carrier.str.contains("h2 storage") & n.stores.carrier.str.contains("retrofit")]

    if retro_storage_acaes:
        acaes_stores = n.stores[n.stores.carrier.str.contains("acaes") & n.stores.carrier.str.contains("retrofit")]

    # Build storage pairs based on aggregation level
    storage_triples = []  # (state, gas_store, h2_store, acaes_store)

    if agg_storage == 1:  # State and type aggregation
        for gas_store in gas_stores.index:
            # Format: "STATE gas storage FIELD_TYPE retrofit"
            parts = gas_store.split(" gas storage ")
            if len(parts) >= 2:
                state = parts[0]
                field_type_and_suffix = parts[1]  # "FIELD_TYPE retrofit"
                field_type = field_type_and_suffix.replace(" retrofit", "")

                h2_store = None
                acaes_store = None

                if retro_storage_h2 and h2_stores is not None:
                    h2_store_name = f"{state} h2 storage {field_type} retrofit"
                    if h2_store_name in h2_stores.index:
                        h2_store = h2_store_name

                if retro_storage_acaes and acaes_stores is not None:
                    acaes_store_name = f"{state} acaes {field_type} retrofit"
                    if acaes_store_name in acaes_stores.index:
                        acaes_store = acaes_store_name

                # Only add if at least one retrofit storage exists
                if h2_store is not None or acaes_store is not None:
                    storage_triples.append((state, gas_store, h2_store, acaes_store))

    else:  # agg_storage == 0, individual facilities
        for gas_store in gas_stores.index:
            parts = gas_store.split(" gas storage ")
            state = parts[0]
            facility_suffix = parts[1]

            h2_store = None
            acaes_store = None

            if retro_storage_h2 and h2_stores is not None:
                h2_store_name = f"{state} h2 storage {facility_suffix}"
                if h2_store_name in h2_stores.index:
                    h2_store = h2_store_name

            if retro_storage_acaes and acaes_stores is not None:
                acaes_store_name = f"{state} acaes {facility_suffix}"
                if acaes_store_name in acaes_stores.index:
                    acaes_store = acaes_store_name

            # Only add if at least one retrofit storage exists
            if h2_store is not None or acaes_store is not None:
                storage_triples.append((state, gas_store, h2_store, acaes_store))

    # Add constraints for each storage triple
    for state, gas_store, h2_store, acaes_store in storage_triples:
        # Get original gas storage capacity
        original_e_nom = gas_stores.loc[gas_store, "e_nom_max"]

        # Get extendable storage capacities
        gas_e_nom = n.model["Store-e_nom"].loc[gas_store]

        # Initialize storage variables
        h2_e_nom = None
        acaes_e_nom = None

        if h2_store is not None:
            h2_e_nom = n.model["Store-e_nom"].loc[h2_store]
        if acaes_store is not None:
            acaes_e_nom = n.model["Store-e_nom"].loc[acaes_store]

        # Constraint 1: Energy capacity constraint
        # E_gas^nom + E_H2^nom/α_eng,H2 + E_air^nom/α_eng,air ≤ E_original^nom
        energy_constraint_terms = [gas_e_nom]

        if h2_e_nom is not None:
            energy_constraint_terms.append(h2_e_nom / storage_eng_retro_factor_h2)
        if acaes_e_nom is not None:
            energy_constraint_terms.append(acaes_e_nom / storage_eng_retro_factor_acaes)

        if len(energy_constraint_terms) > 1:  # Only add constraint if there are retrofit options
            n.model.add_constraints(
                sum(energy_constraint_terms) <= original_e_nom,
                name=f"storage_retrofit_energy_capacity_{gas_store.replace(' ', '_')}",
            )

        # Add power capacity constraints for charge/discharge links
        # Determine link names based on aggregation level
        gas_charge = f"{gas_store} charge"
        gas_discharge = f"{gas_store} discharge"

        h2_charge = None
        h2_discharge = None
        if h2_store is not None:
            h2_charge = f"{h2_store} charge"
            h2_discharge = f"{h2_store} discharge"

        # Check if discharge links exist before adding power constraints
        if gas_discharge in n.links.index:
            original_p_discharge = n.links.loc[gas_discharge, "p_nom_max"]
            gas_p_discharge = n.model["Link-p_nom"].loc[gas_discharge]

            # Constraint 2: Power capacity constraint for discharging
            # P_gas,dis^nom + P_H2,dis^nom/α_pow,H2 ≤ P_original,dis^nom
            if h2_discharge is not None and h2_discharge in n.links.index:
                h2_p_discharge = n.model["Link-p_nom"].loc[h2_discharge]

                n.model.add_constraints(
                    gas_p_discharge + h2_p_discharge / storage_pow_retro_factor_h2 <= original_p_discharge,
                    name=f"storage_retrofit_power_discharge_{gas_store.replace(' ', '_')}",
                )

                # Constraint 3a: Internal proportionality for gas discharge
                # P_gas,dis^nom / P_original,dis^nom = E_gas^nom / E_original^nom
                # Rearranged: P_gas,dis^nom * E_original^nom / P_original,dis^nom = E_gas^nom
                n.model.add_constraints(
                    gas_p_discharge * original_e_nom / original_p_discharge == gas_e_nom,
                    name=f"storage_gas_discharge_energy_proportion_{gas_store.replace(' ', '_')}",
                )

                # Constraint 3b: Internal proportionality for H2 discharge
                # P_H2,dis^nom / α_pow,H2 / P_original,dis^nom = E_H2^nom / α_eng,H2 / E_original^nom
                # Rearranged: P_H2,dis^nom * α_eng,H2 / α_pow,H2 * E_original^nom / P_original,dis^nom = E_H2^nom
                if h2_e_nom is not None:
                    n.model.add_constraints(
                        h2_p_discharge
                        * storage_eng_retro_factor_h2
                        / storage_pow_retro_factor_h2
                        * original_e_nom
                        / original_p_discharge
                        == h2_e_nom,
                        name=f"storage_h2_discharge_energy_proportion_{gas_store.replace(' ', '_')}",
                    )

        # Check if charge links exist and add similar proportionality constraints for gas
        if gas_charge in n.links.index:
            original_p_charge = n.links.loc[gas_charge, "p_nom_max"]
            gas_p_charge = n.model["Link-p_nom"].loc[gas_charge]

            # Constraint 4a: Internal proportionality for gas charge
            # P_gas,char^nom * E_original^nom / P_original,char^nom <= E_gas^nom
            n.model.add_constraints(
                gas_p_charge * original_e_nom / original_p_charge <= gas_e_nom,
                name=f"storage_gas_charge_energy_proportion_{gas_store.replace(' ', '_')}",
            )

            if h2_charge is not None and h2_charge in n.links.index and h2_e_nom is not None:
                h2_p_charge = n.model["Link-p_nom"].loc[h2_charge]
                # Constraint 4b: Internal proportionality for H2 charge
                n.model.add_constraints(
                    h2_p_charge
                    * storage_eng_retro_factor_h2
                    / storage_pow_retro_factor_h2
                    * original_e_nom
                    / original_p_charge
                    <= h2_e_nom,
                    name=f"storage_h2_charge_energy_proportion_{gas_store.replace(' ', '_')}",
                )

    # Add proportionality constraints for non-retrofit gas storage
    gas_stores_nonretro = n.stores[
        n.stores.carrier.str.contains("gas storage") & n.stores.carrier.str.contains("nonretrofit")
    ]

    if not gas_stores_nonretro.empty:
        for store_name in gas_stores_nonretro.index:
            # Define corresponding link names
            charge_link = f"{store_name} charge"
            discharge_link = f"{store_name} discharge"

            # Check if links exist
            if charge_link not in n.links.index or discharge_link not in n.links.index:
                logger.warning(f"Links not found for non-retrofit storage {store_name}")
                continue

            # Get original capacities
            original_e_nom = n.stores.loc[store_name, "e_nom_max"]
            original_p_discharge = n.links.loc[discharge_link, "p_nom_max"]
            original_p_charge = n.links.loc[charge_link, "p_nom_max"]

            # Get optimization variables
            e_nom = n.model["Store-e_nom"].loc[store_name]
            p_nom_discharge = n.model["Link-p_nom"].loc[discharge_link]
            p_nom_charge = n.model["Link-p_nom"].loc[charge_link]

            # Add discharge proportionality constraint
            # p_nom_discharge * original_e_nom / original_p_discharge == e_nom
            n.model.add_constraints(
                p_nom_discharge * original_e_nom / original_p_discharge == e_nom,
                name=f"nonretrofit_gas_discharge_energy_proportion_{store_name.replace(' ', '_')}",
            )

            # Add charge proportionality constraint
            # p_nom_charge * original_e_nom / original_p_charge <= e_nom
            n.model.add_constraints(
                p_nom_charge * original_e_nom / original_p_charge <= e_nom,
                name=f"nonretrofit_gas_charge_energy_proportion_{store_name.replace(' ', '_')}",
            )


def add_gas_pipeline_retrofit_constraints(n, config):
    """
    Add constraints for hydrogen retrofit of gas infrastructure.
    """
    # Get retrofit parameters from config
    ng_options = config.get("sector", {}).get("natural_gas", {})
    retro_pipeline = ng_options.get("retro_pipeline", False)

    # Get retrofit factors
    pipeline_retro_factor = ng_options.get("pipeline_retro_factor", 0.6)

    # Check if binary mode is enabled
    binary_pipeline = ng_options.get("binary_pipeline", False)

    if retro_pipeline:
        logger.info("Adding pipeline retrofit constraints")
        # Identify retrofit gas and H2 pipeline pairs
        gas_pipes = n.links[n.links.carrier.str.contains("gas pipeline") & ~n.links.carrier.str.contains("non")]
        h2_pipes = n.links[n.links.carrier.str.contains("h2 pipeline") & n.links.carrier.str.contains("retrofit")]

        agg_pipeline = ng_options.get("aggregate_pipeline", True)

        if agg_pipeline:
            # For aggregated pipelines
            pipeline_pairs = []
            for gas_pipe in gas_pipes.index:
                h2_pipe_fwd = gas_pipe.replace(" gas pipeline", " h2 pipeline retrofit")
                h2_pipe_rev = gas_pipe.replace(" gas pipeline", " h2 pipeline retrofit reverse")
                if h2_pipe_fwd in h2_pipes.index:
                    if h2_pipe_rev in h2_pipes.index:
                        pipeline_pairs.append((gas_pipe, h2_pipe_fwd, h2_pipe_rev))
                    else:
                        pipeline_pairs.append((gas_pipe, h2_pipe_fwd))
        else:
            # For individual pipelines
            pipeline_pairs = []
            for gas_pipe in gas_pipes.index:
                parts = gas_pipe.split(" gas pipeline ")
                if len(parts) == 2:
                    prefix = parts[0]
                    suffix = parts[1]
                    h2_pipe_fwd = f"{prefix} h2 pipeline retrofit {suffix}"
                    h2_pipe_rev = f"{prefix} h2 pipeline retrofit reverse {suffix}"
                    if h2_pipe_fwd in h2_pipes.index:
                        if h2_pipe_rev in h2_pipes.index:
                            pipeline_pairs.append((gas_pipe, h2_pipe_fwd, h2_pipe_rev))
                        else:
                            pipeline_pairs.append((gas_pipe, h2_pipe_fwd))

        if binary_pipeline and pipeline_pairs:
            # Add binary variables for pipeline retrofit decision (x1)
            gas_pipes_list = [pair[0] for pair in pipeline_pairs]
            gas_pipes_index = pd.Index(gas_pipes_list, name="pipeline")

            n.model.add_variables(
                coords=[gas_pipes_index],
                name="pipeline_retrofit_binary",
                binary=True,
            )
            retrofit_vars = n.model["pipeline_retrofit_binary"]

            # Add binary variables for pipeline retirement decision (x2)
            n.model.add_variables(
                coords=[gas_pipes_index],
                name="pipeline_retirement_binary",
                binary=True,
            )
            retirement_vars = n.model["pipeline_retirement_binary"]

        for pair in pipeline_pairs:
            gas_pipe = pair[0]
            h2_pipe_fwd = pair[1]
            h2_pipe_rev = pair[2] if len(pair) > 2 else None

            # Get original gas pipeline capacity
            original_p_nom = gas_pipes.loc[gas_pipe, "p_nom_max"]
            gas_p_nom = n.model["Link-p_nom"].loc[gas_pipe]

            # Process H2 pipeline capacity
            h2_p_nom = n.model["Link-p_nom"].loc[h2_pipe_fwd]

            if binary_pipeline:
                retrofit_var = retrofit_vars.loc[gas_pipe]  # x1
                retirement_var = retirement_vars.loc[gas_pipe]  # x2

                # Mutual exclusivity constraint: x1 + x2 <= 1
                n.model.add_constraints(
                    retrofit_var + retirement_var <= 1,
                    name=f"pipeline_mutual_exclusivity_{gas_pipe.replace(' ', '_')}",
                )

                # gas_p_nom <= (1 - x1 - x2) * original_p_nom
                n.model.add_constraints(
                    gas_p_nom + retrofit_var * original_p_nom + retirement_var * original_p_nom <= original_p_nom,
                    name=f"pipeline_retrofit_gas_{gas_pipe.replace(' ', '_')}",
                )
                # h2_p_nom <= x1 * original_p_nom * pipeline_retro_factor
                n.model.add_constraints(
                    h2_p_nom <= retrofit_var * original_p_nom * pipeline_retro_factor,
                    name=f"pipeline_retrofit_h2_{gas_pipe.replace(' ', '_')}",
                )
            else:
                n.model.add_constraints(
                    h2_p_nom / pipeline_retro_factor + gas_p_nom <= original_p_nom,
                    name=f"pipeline_retrofit_continuous_{gas_pipe.replace(' ', '_')}",
                )


def add_new_h2_storage_constraints(n, config):
    """
    Add constraints for new H2 storage components.

    This function adds constraints for new H2 storage (carrier: "h2 storage new") including:
    1. discharge_capacity / energy_capacity <= 5e-3
    # 2. charge_capacity / discharge_capacity = 132 / 359

    These constraints ensure proper sizing relationships between energy and power capacities
    for new hydrogen storage infrastructure.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object
    config : dict
        Configuration dictionary containing hydrogen options
    """
    # Get hydrogen configuration options
    h2_options = config.get("sector", {}).get("hydrogen", {})

    # Check if new_storage is enabled
    if not h2_options.get("new_storage", True):
        logger.info("New H2 storage is disabled, skipping constraints")
        return

    logger.info("Adding new H2 storage capacity constraints")

    # Define constraint ratios
    DISCHARGE_ENERGY_RATIO = 5e-3  # discharge capacity / energy capacity
    # CHARGE_DISCHARGE_RATIO = 132 / 359  # charge capacity / discharge capacity

    # Identify new H2 storage stores
    h2_storage_new_stores = n.stores[n.stores.carrier == "h2 storage new"]

    if h2_storage_new_stores.empty:
        logger.info("No new H2 storage found in the network")
        return

    # Process each new H2 storage store
    constraints_added = 0
    for store_name in h2_storage_new_stores.index:
        # Extract state from store name (format: "{state} h2 storage new")
        state = store_name.replace(" h2 storage new", "")

        # Define corresponding link names
        charge_link = f"{state} charge h2 storage new"
        discharge_link = f"{state} discharge h2 storage new"

        # Check if corresponding links exist
        if charge_link not in n.links.index:
            logger.warning(f"Charge link {charge_link} not found for storage {store_name}")
            continue
        if discharge_link not in n.links.index:
            logger.warning(f"Discharge link {discharge_link} not found for storage {store_name}")
            continue

        # Get optimization variables
        store_e_nom = n.model["Store-e_nom"].loc[store_name]  # Energy capacity (MWh)
        charge_p_nom = n.model["Link-p_nom"].loc[charge_link]  # Charge power capacity (MW)
        discharge_p_nom = n.model["Link-p_nom"].loc[discharge_link]  # Discharge power capacity (MW)

        # Constraint 1: discharge_p_nom <= store_e_nom * DISCHARGE_ENERGY_RATIO
        constraint_name_1 = f"h2_storage_new_discharge_energy_ratio_{state.replace(' ', '_')}"
        n.model.add_constraints(
            discharge_p_nom <= store_e_nom * DISCHARGE_ENERGY_RATIO,
            name=constraint_name_1,
        )

        # # Constraint 2: charge_p_nom = discharge_p_nom * CHARGE_DISCHARGE_RATIO
        # constraint_name_2 = f"h2_storage_new_charge_discharge_ratio_{state.replace(' ', '_')}"
        # n.model.add_constraints(
        #     charge_p_nom == discharge_p_nom * CHARGE_DISCHARGE_RATIO,
        #     name=constraint_name_2
        # )

        # constraints_added += 2
        constraints_added += 1

    logger.info(
        f"Added {constraints_added} new H2 storage capacity constraints for {len(h2_storage_new_stores)} storage units",
    )


def add_power_soc_constraints(n):
    """
    Add power-SOC constraints for gas and H2 underground storage.

    These constraints model the physical relationship between storage state of charge (SOC)
    and maximum discharge rate. As SOC decreases, pressure differential decreases,
    leading to reduced maximum discharge capacity.

    Constraint: discharge_power_t <= energy_t * (p_nom / e_nom)

    where:
    - discharge_power_t: discharge link power at time t
    - energy_t: storage energy level at time t
    - (p_nom / e_nom): constant power-to-energy ratio for each storage

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object
    config : dict
        Configuration dictionary containing natural gas options
    """
    logger.info("Adding power-SOC constraints for gas and H2 storage")

    # Identify gas and H2 storage components
    gas_stores = n.stores[n.stores.carrier.str.contains("gas storage")]
    h2_stores = n.stores[n.stores.carrier.str.contains("h2 storage")]

    # Combine all storage units to process
    all_stores = pd.concat([gas_stores, h2_stores])

    # Process each storage unit
    for store_name in all_stores.index:
        # Determine carrier type
        if "gas storage" in store_name:
            carrier_type = "gas"
            discharge_link = f"{store_name} discharge"
        elif "h2 storage" in store_name:
            carrier_type = "h2"
            discharge_link = f"{store_name} discharge"

        # Check if discharge link exists
        if discharge_link not in n.links.index:
            logger.warning(f"Discharge link {discharge_link} not found for storage {store_name}")
            continue

        # Calculate power-to-energy ratio (p_nom / e_nom)
        # Get nominal capacities from the network components
        store_e_nom_max = n.stores.loc[store_name, "e_nom_max"]
        link_p_nom_max = n.links.loc[discharge_link, "p_nom_max"]

        # Calculate the constant ratio
        power_energy_ratio = link_p_nom_max / store_e_nom_max

        # Add constraints for each investment period
        for period in n.investment_periods:
            # Get optimization variables
            discharge_power = n.model["Link-p"].loc[period, discharge_link]
            storage_energy = n.model["Store-e"].loc[period, store_name]

            # Add constraint: discharge_power_t <= energy_t * power_energy_ratio
            # This constraint applies to all timesteps in the investment period
            lhs = discharge_power
            rhs = storage_energy * power_energy_ratio

            constraint_name = f"power_soc_{carrier_type}_{store_name.replace(' ', '_')}_{period}"

            n.model.add_constraints(
                lhs <= rhs,
                name=constraint_name,
            )


def add_h2_electrolysis_average_constraints(n, averaging_period="week"):
    """
    Add constraints to ensure each h2 electrolysis link has equal average power over specified time periods.

    For each h2 electrolysis link:
    - If averaging_period is 'hour': Set p_max_pu = p_min_pu = 1 (constant operation)
    - Otherwise: Calculate average power across all hours in each complete time block
      and set constraint that all block averages are equal
    - Partial blocks at the end are not constrained

    This ensures consistent hydrogen production patterns across time periods for each electrolyzer.
    Different electrolyzers are not constrained relative to each other.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object
    averaging_period : str
        Time period for averaging: 'hour', 'day', 'week', 'month', 'season'
        - 'hour': Constant operation (p_max_pu = p_min_pu = 1)
        - 'day': 24-hour blocks
        - 'week': 168-hour blocks (7 days)
        - 'month': 720-hour blocks (30 days)
        - 'season': 2190-hour blocks (91.25 days)
    """
    # Find all h2 electrolysis links
    h2_elec_links = n.links[n.links.carrier == "h2 electrolysis"].index

    if h2_elec_links.empty:
        logger.info("No h2 electrolysis links found")
        return

    # Handle 'hour' case: set constant operation
    if averaging_period == "hour":
        logger.info("Setting h2 electrolysis to constant operation (p_max_pu = p_min_pu = 1)")

        # Set p_max_pu and p_min_pu to 1 for all h2 electrolysis links
        n.links.loc[h2_elec_links, "p_max_pu"] = 1.0
        n.links.loc[h2_elec_links, "p_min_pu"] = 1.0

        logger.info(f"Set constant operation for {len(h2_elec_links)} h2 electrolysis links")
        return

    # Define block sizes in HOURS for each averaging period
    period_hours = {
        "day": 24,
        "week": 168,  # 7 * 24
        "month": 720,  # 30 * 24
        "season": 2190,  # 91.25 * 24
    }

    if averaging_period not in period_hours:
        logger.warning(
            f"Invalid averaging_period '{averaging_period}'. "
            f"Must be one of {list(period_hours.keys()) + ['hour']}. Defaulting to 'week'.",
        )
        averaging_period = "week"

    block_hours = period_hours[averaging_period]

    logger.info(f"Adding {averaging_period}ly average constraints for h2 electrolysis")

    # Process each h2 electrolysis link separately
    for link in h2_elec_links:
        # Process each investment period separately
        for period in n.investment_periods:
            # Get snapshots for this period (just the timestamps, level 1)
            timestamps = n.snapshots.get_level_values(1)[n.snapshots.get_level_values(0) == period]

            # Calculate the actual time resolution (timestep in hours)
            if len(timestamps) > 1:
                time_deltas = pd.to_datetime(timestamps[1:]).values - pd.to_datetime(timestamps[:-1]).values
                # Get the most common timestep (in case of irregular spacing)
                timestep_hours = pd.Series(time_deltas).mode()[0] / np.timedelta64(1, "h")
            else:
                logger.warning(f"Only one timestamp found for period {period}, skipping")
                continue

            # Check if block_hours is divisible by timestep_hours
            if block_hours % timestep_hours != 0:
                logger.warning(
                    f"Block size {block_hours}h is not divisible by timestep {timestep_hours}h. "
                    f"Adjusting block size to nearest multiple: {int(block_hours / timestep_hours) * timestep_hours}h",
                )
                # Adjust block size to nearest multiple of timestep
                block_hours_adjusted = int(block_hours / timestep_hours) * timestep_hours
            else:
                block_hours_adjusted = block_hours

            # Calculate number of snapshots per block
            snapshots_per_block = int(block_hours_adjusted / timestep_hours)

            # Group timestamps into blocks
            time_blocks = []
            for i in range(0, len(timestamps), snapshots_per_block):
                block_timestamps = timestamps[i : i + snapshots_per_block]
                # Only include complete blocks
                if len(block_timestamps) == snapshots_per_block:
                    time_blocks.append(block_timestamps)

            # Skip if less than 2 complete blocks (need at least 2 blocks to constrain)
            if len(time_blocks) < 2:
                logger.info(
                    f"Skipping {link} in period {period}: "
                    f"insufficient complete {averaging_period}s ({len(time_blocks)} blocks)",
                )
                continue

            # Get the Link-p variable for this link and period
            link_p = n.model["Link-p"].loc[period, link]

            # Calculate block averages
            block_averages = []
            for block_idx, block_timestamps in enumerate(time_blocks):
                # Sum power for these timestamps
                block_power_sum = link_p.loc[block_timestamps].sum()

                # Calculate average (sum / number of snapshots in that block)
                block_avg = block_power_sum / len(block_timestamps)
                block_averages.append(block_avg)

            # Add constraints: block_avg[1] = block_avg[0], block_avg[2] = block_avg[0], ...
            # This ensures all time blocks have the same average power
            for i in range(1, len(block_averages)):
                constraint_name = f"h2_elec_{averaging_period}ly_avg_{link.replace(' ', '_')}_{period}_block{i}"

                n.model.add_constraints(
                    block_averages[i] == block_averages[0],
                    name=constraint_name,
                )

    logger.info(
        f"Added {averaging_period}ly average constraints for {len(h2_elec_links)} h2 electrolysis links",
    )


def ban_fossil(n):
    """
    Ban certain gas and coal-fired generation technologies from the network by
    setting their p_nom_min and p_nom_max to 0
    """
    # Define carriers to ban
    banned_carriers = [
        "OCGT",
        "CCGT",
        "CCGT-95CCS",
        "CCGT-97CCS",
        "coal",
        "coal-95CCS",
        "coal-99CCS",
        "h2 smr",
        "h2 smr-cc",
    ]

    # Find links with banned carriers
    all_banned_links = n.links[n.links.carrier.isin(banned_carriers)]

    # Set p_nom_min to 0 (minimum capacity)
    n.links.loc[all_banned_links.index, "p_nom_min"] = 0
    n.links.loc[all_banned_links.index, "p_nom_max"] = 0

    logger.info(
        f"Successfully banned {len(all_banned_links)} fossil fuel links",
    )


def ban_gasccs(n):
    # Define carriers to ban
    banned_carriers = ["CCGT-95CCS", "CCGT-97CCS"]

    # Find links with banned carriers
    all_banned_links = n.links[n.links.carrier.isin(banned_carriers)]

    # Set p_nom_min to 0 (minimum capacity)
    n.links.loc[all_banned_links.index, "p_nom_min"] = 0
    n.links.loc[all_banned_links.index, "p_nom_max"] = 0

    logger.info(
        "Successfully banned Gas-CCS",
    )


def add_gas_turbine_retrofit_constraints(n, config):
    """
    Add constraints for H2 retrofit of gas turbines (CCGT/OCGT).

    For each original CCGT/OCGT link with a corresponding retrofit link:
    original_capacity >= gas_p_nom + h2_p_nom

    This ensures the sum of gas and H2 operation doesn't exceed original capacity.
    """
    h2_options = config.get("sector", {}).get("hydrogen", {})
    retro_turbine = h2_options.get("retro_turbine", False)

    if not retro_turbine:
        return

    logger.info("Adding gas turbine retrofit constraints")

    # Find all CCGT and OCGT links (without CCS) with build_year < 2050
    gas_turbines = n.links[
        (n.links.carrier.isin(["CCGT", "OCGT"]))  # Only non-CCS turbines
        & (n.links.build_year < 2050)
    ]

    constraints_added = 0

    for turbine_idx in gas_turbines.index:
        retrofit_idx = f"{turbine_idx} retrofit"

        # Check if corresponding retrofit link exists
        if retrofit_idx not in n.links.index:
            continue

        # Get original capacity
        original_capacity = n.links.loc[turbine_idx, "p_nom"]

        # Get optimization variables
        turbine_p_nom = n.model["Link-p_nom"].loc[turbine_idx]
        retrofit_p_nom = n.model["Link-p_nom"].loc[retrofit_idx]

        # Add constraint: turbine_p_nom + retrofit_p_nom <= original_capacity
        constraint_name = f"turbine_retrofit_capacity_{turbine_idx.replace(' ', '_').replace('-', '_')}"

        n.model.add_constraints(
            turbine_p_nom + retrofit_p_nom <= original_capacity,
            name=constraint_name,
        )

        constraints_added += 1

    logger.info(f"Added {constraints_added} gas turbine retrofit capacity constraints")


def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.optimization.optimize``.

    If you want to enforce additional custom constraints, this is a good
    location to add them. The arguments ``opts`` and
    ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    if "PRM" in opts and n.generators.p_nom_extendable.any():
        add_PRM_constraints(n, config)
    if "ERM" in opts and n.generators.p_nom_extendable.any():
        add_ERM_constraints(n, config)

    if "SimpSec" in opts:
        if config["sector"]["co2"].get("policy", {}):
            # Pass reference network path if available
            reference_network_path = getattr(snakemake.input, "reference_network", None)
            # Handle case where reference_network might be empty list
            if isinstance(reference_network_path, list) and len(reference_network_path) == 0:
                reference_network_path = None
            add_co2_constraints(n, config, reference_network_path)
        # if config["sector"]["natural_gas"].get("imports", False):
        #     add_ng_import_export_limits(n, config)
        if config["sector"]["natural_gas"].get("retro_storage_h2", False) or config["sector"]["natural_gas"].get(
            "retro_storage_acaes", False
        ):
            add_gas_storage_retrofit_constraints(n, config)
        if config["sector"]["natural_gas"].get("retro_pipeline", False):
            add_gas_pipeline_retrofit_constraints(n, config)
        h2_options = config.get("sector", {}).get("hydrogen", {})
        if h2_options.get("new_storage", True):
            add_new_h2_storage_constraints(n, config)
        if config["sector"]["natural_gas"].get("varying_discharge", True):
            add_power_soc_constraints(n)
        h2_averaging = config.get("sector", {}).get("hydrogen", {}).get("constant_average", None)
        if h2_averaging:
            # h2_averaging can be 'day', 'week', 'month', or 'season'
            add_h2_electrolysis_average_constraints(n, averaging_period=h2_averaging)
        if h2_options.get("retro_turbine", False):
            add_gas_turbine_retrofit_constraints(n, config)

    add_land_use_constraints(n)
    add_bidirectional_link_constraints(n)


def run_optimize(n, rolling_horizon, skip_iterations, cf_solving, **kwargs):
    """Initiate the correct type of pypsa.optimize function."""
    if rolling_horizon:
        kwargs["horizon"] = cf_solving.get("horizon", 365)
        kwargs["overlap"] = cf_solving.get("overlap", 0)
        n.optimize.optimize_with_rolling_horizon(**kwargs)
        status, condition = "", ""
    elif skip_iterations:
        status, condition = n.optimize(**kwargs)
    else:
        kwargs["track_iterations"] = (cf_solving.get("track_iterations", False),)
        kwargs["min_iterations"] = cf_solving.get("min_iterations", 4)
        kwargs["max_iterations"] = cf_solving.get("max_iterations", 6)
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            **kwargs,
        )

    if status != "ok" and not rolling_horizon:
        logger.warning(
            f"Solving status '{status}' with termination condition '{condition}'",
        )
    if "infeasible" in condition:
        # n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'")


def solve_network(n, config, solving, opts="", **kwargs):
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    foresight = config["foresight"]
    kwargs["multi_investment_periods"] = foresight == "perfect"

    kwargs["solver_options"] = solving["solver_options"][set_of_options] if set_of_options else {}
    kwargs["solver_name"] = solving["solver"]["name"]
    kwargs["extra_functionality"] = extra_functionality
    kwargs["transmission_losses"] = cf_solving.get("transmission_losses", False)
    kwargs["linearized_unit_commitment"] = cf_solving.get(
        "linearized_unit_commitment",
        False,
    )
    kwargs["assign_all_duals"] = cf_solving.get("assign_all_duals", False)

    rolling_horizon = cf_solving.pop("rolling_horizon", False)
    skip_iterations = cf_solving.pop("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    match foresight:
        case "perfect":
            run_optimize(n, rolling_horizon, skip_iterations, cf_solving, **kwargs)
        case "myopic":
            for i, planning_horizon in enumerate(n.investment_periods):
                # planning_horizons = snakemake.params.planning_horizons
                sns_horizon = n.snapshots[n.snapshots.get_level_values(0) == planning_horizon]

                # add sns_horizon to kwarg
                kwargs["snapshots"] = sns_horizon

                run_optimize(n, rolling_horizon, skip_iterations, cf_solving, **kwargs)

                if i == len(n.investment_periods) - 1:
                    logger.info(f"Final time horizon {planning_horizon}")
                    continue
                logger.info(f"Preparing brownfield from {planning_horizon}")

                # electric transmission grid set optimised capacities of previous as minimum
                n.lines.s_nom_min = n.lines.s_nom_opt  # for lines
                dc_i = n.links[n.links.carrier == "DC"].index
                n.links.loc[dc_i, "p_nom_min"] = n.links.loc[dc_i, "p_nom_opt"]  # for links

                for c in n.iterate_components(["Generator", "Link", "StorageUnit"]):
                    nm = c.name
                    # limit our components that we remove/modify to those prior to this time horizon
                    c_lim = c.df.loc[n.get_active_assets(nm, planning_horizon)]

                    logger.info(f"Preparing brownfield for the component {nm}")
                    # attribute selection for naming convention
                    attr = "p"
                    # copy over asset sizing from previous period
                    c_lim[f"{attr}_nom"] = c_lim[f"{attr}_nom_opt"]
                    c_lim[f"{attr}_nom_extendable"] = False
                    df = copy.deepcopy(c_lim)
                    time_df = copy.deepcopy(c.pnl)

                    for c_idx in c_lim.index:
                        n.remove(nm, c_idx)

                    for df_idx in df.index:
                        if nm == "Generator":
                            n.madd(
                                nm,
                                [df_idx],
                                carrier=df.loc[df_idx].carrier,
                                bus=df.loc[df_idx].bus,
                                p_nom_min=df.loc[df_idx].p_nom_min,
                                p_nom=df.loc[df_idx].p_nom,
                                p_nom_max=df.loc[df_idx].p_nom_max,
                                p_nom_extendable=df.loc[df_idx].p_nom_extendable,
                                ramp_limit_up=df.loc[df_idx].ramp_limit_up,
                                ramp_limit_down=df.loc[df_idx].ramp_limit_down,
                                efficiency=df.loc[df_idx].efficiency,
                                marginal_cost=df.loc[df_idx].marginal_cost,
                                capital_cost=df.loc[df_idx].capital_cost,
                                build_year=df.loc[df_idx].build_year,
                                lifetime=df.loc[df_idx].lifetime,
                                heat_rate=df.loc[df_idx].heat_rate,
                                fuel_cost=df.loc[df_idx].fuel_cost,
                                vom_cost=df.loc[df_idx].vom_cost,
                                carrier_base=df.loc[df_idx].carrier_base,
                                p_min_pu=df.loc[df_idx].p_min_pu,
                                p_max_pu=df.loc[df_idx].p_max_pu,
                                land_region=df.loc[df_idx].land_region,
                            )
                        else:
                            n.add(nm, df_idx, **df.loc[df_idx])
                    logger.info(n.consistency_check())

                    # copy time-dependent
                    selection = n.component_attrs[nm].type.str.contains(
                        "series",
                    )

                    for tattr in n.component_attrs[nm].index[selection]:
                        n.import_series_from_dataframe(time_df[tattr], nm, tattr)

                # roll over the last snapshot of time varying storage state of charge to be the state_of_charge_initial for the next time period
                n.storage_units.loc[:, "state_of_charge_initial"] = n.storage_units_t.state_of_charge.loc[
                    planning_horizon
                ].iloc[-1]

        case _:
            raise ValueError(f"Invalid foresight option: '{foresight}'. Must be 'perfect' or 'myopic'.")

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_network",
            case="HighE_new_h2storage_tes",
            transmission_network="reeds",
        )
    configure_logging(snakemake)
    # update_config_from_wildcards(snakemake.config, snakemake.wildcards)
    # if "sector_opts" in snakemake.wildcards.keys():
    #     update_config_with_sector_opts(
    #         snakemake.config,
    #         snakemake.wildcards.sector_opts,
    #     )

    opts = snakemake.params.opts
    # opts = "4h"
    # if "sector_opts" in snakemake.wildcards.keys():
    #     opts += "-" + snakemake.wildcards.sector_opts
    opts = [o for o in opts.split("-") if o != ""]
    solve_opts = snakemake.params.solving["options"]

    # sector specific co2 options
    # if snakemake.wildcards.sector != "E":
    # sector co2 limits applied via config file, not through Co2L
    opts = [x for x in opts if not x.startswith("Co2L")]
    # if snakemake.wildcards.sector != "SimpSec":
    #     opts.append("sector")
    # else:
    opts.append("SimpSec")

    sector_config = snakemake.params.sector
    solving_config = snakemake.params.solving
    complete_config = {
        "foresight": snakemake.params.foresight,
        "sector": sector_config,
        "scenario": {
            "planning_horizons": snakemake.params.planning_horizons,
            "opts": snakemake.params.opts,
            "nza_scenario": getattr(snakemake.params, "nza_scenario", ""),
        },
        "solving": solving_config,
        "co2_sequestration_potential": snakemake.params.co2_sequestration_potential,
    }

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)

    # line_data = pd.read_csv("repo_data/lines.csv")
    # line_data["Line"] = line_data["Line"].astype(str)
    # line_data = line_data.set_index("Line")
    # common = n.lines.index.intersection(line_data.index)
    # n.lines.loc[common, "x"] = line_data.loc[common, "x"]
    # n.lines.loc[common, "r"] = line_data.loc[common, "r"]
    # n.calculate_dependent_values()

    first_year = snakemake.params.planning_horizons[0]
    costs = load_costs(f"resources/costs/costs_{first_year}.csv")
    # Apply regional multipliers to generators and links
    apply_regional_cost_multipliers(n, "repo_data/locational_multipliers/")
    # Recalculate network infrastructure costs
    recalculate_network_costs(n, costs)

    n.links.loc[n.links[n.links.carrier == "coal"].index, "p_nom_min"] = 0
    n.links.loc[n.links[n.links.carrier == "coal"].index, "p_nom_max"] = 0
    if sector_config["fossil"].get("ban", False):
        ban_fossil(n)
    if sector_config["fossil"].get("ban_gasccs", False):
        ban_gasccs(n)

    n = prepare_network(
        n,
        solve_opts,
    )
    n.lines.s_nom_opt = n.lines.s_nom
    n = solve_network(
        n,
        config=complete_config,
        solving=solving_config,
        opts=opts,
        log_fn=snakemake.log.solver,
    )
    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
    with open(snakemake.output.config, "w") as file:
        yaml.dump(
            n.meta,
            file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
