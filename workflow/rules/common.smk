# SPDX-FileCopyrightText: : 2023-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

import copy
import re
from functools import partial, lru_cache
from collections.abc import Mapping

import os, sys, glob

path = workflow.source_path("../scripts/_helpers.py")
sys.path.insert(0, os.path.dirname(path))

from _helpers import validate_checksum, update_config_from_wildcards
from snakemake.utils import update_config


# ============================================================================
# LEGACY CONFIG PROVIDER (for rules without case wildcard)
# ============================================================================


def static_getter(wildcards, keys, default):
    """Getter function for static config values (no case wildcard)"""
    config_with_wildcards = update_config_from_wildcards(
        config, wildcards, inplace=False
    )
    return get_config(config_with_wildcards, keys, default)


def dynamic_getter(wildcards, keys, default):
    """Getter function for dynamic config values based on scenario (no case wildcard)"""
    if "run" not in wildcards.keys():
        return get_config(config, keys, default)
    scenario_name = wildcards.run
    if scenario_name not in scenarios:
        raise ValueError(
            f"Scenario {scenario_name} not found in file {config['run']['scenario']['file']}."
        )
    config_with_scenario = scenario_config(scenario_name)
    config_with_wildcards = update_config_from_wildcards(
        config_with_scenario, wildcards, inplace=False
    )
    return get_config(config_with_wildcards, keys, default)


def config_provider(*keys, default=None):
    """
    Legacy config provider for rules WITHOUT case wildcard.

    Use this for early-stage rules that don't have case-specific configs yet.
    For rules with {case} wildcard, use case_config_provider() instead.

    Usage in Snakemake rules:
        params:
            my_param=config_provider("key1", "key2", default="some_default_value")
    """
    # Using functools.partial to freeze certain arguments
    if config["run"].get("scenarios", {}).get("enable", False):
        return partial(dynamic_getter, keys=keys, default=default)
    else:
        return partial(static_getter, keys=keys, default=default)


def get_config(config, keys, default=None):
    """Retrieve a nested value from a dictionary using a tuple of keys."""
    value = config
    for key in keys:
        if isinstance(value, list):
            value = value[key]
        else:
            value = value.get(key, default)
        if value == default:
            return default
    return value


def merge_configs(base_config, scenario_config):
    """Merge base config with a specific scenario without modifying the original."""
    merged = copy.deepcopy(base_config)
    update_config(merged, scenario_config)
    return merged


@lru_cache
def scenario_config(scenario_name):
    """Retrieve a scenario config based on the overrides from the scenario file."""
    return merge_configs(config, scenarios[scenario_name])


def deep_merge(base_dict, override_dict):
    """
    Recursively merge dictionaries. Values in override_dict will override base_dict.

    Args:
        base_dict: Base configuration dictionary
        override_dict: Override configuration dictionary

    Returns:
        dict: Merged dictionary
    """
    result = copy.deepcopy(base_dict)

    for key, value in override_dict.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def build_case_configs(global_config, cases):
    """
    Build complete configuration dictionary for each case.
    Case-specific configuration will override global configuration.

    Args:
        global_config: Global configuration from config files
        cases: Dictionary of case-specific configurations

    Returns:
        dict: {case_name: merged_config}
    """
    case_configs = {}

    for case_name, case_specific in cases.items():
        # Deep merge global config with case config
        merged = deep_merge(global_config, case_specific)
        case_configs[case_name] = merged

    return case_configs


# ============================================================================
# NEW UNIFIED CONFIG SYSTEM
# ============================================================================


def get_case_config(wildcards, apply_wildcards=True):
    """
    Get complete configuration for a case with proper override hierarchy.

    Hierarchy: config.default.yaml < config.sector.yaml < config.yaml cases[case_name]

    Args:
        wildcards: Snakemake wildcards object
        apply_wildcards: Whether to apply wildcard-based config updates

    Returns:
        dict: Complete merged configuration
    """
    # Start with preprocessed case config (already merged during workflow init)
    if hasattr(wildcards, "case") and wildcards.case in CASE_CONFIGS:
        merged_config = copy.deepcopy(CASE_CONFIGS[wildcards.case])
    else:
        # Fallback to global config
        merged_config = copy.deepcopy(config)

    # Apply wildcard-based updates if requested
    if apply_wildcards:
        merged_config = update_config_from_wildcards(
            merged_config, wildcards, inplace=False
        )

    return merged_config


def case_config_provider(*keys, default=None):
    """
    Provide config values with case-specific overrides.

    This is the PRIMARY function to use in rule params.
    It ensures proper config hierarchy and wildcard application.

    Usage in rules:
        params:
            planning_horizons=case_config_provider("scenario", "planning_horizons"),
            sector=case_config_provider("sector"),
            opts=case_config_provider("scenario", "opts", default="3h"),

    Args:
        *keys: Configuration key path
        default: Default value if key not found

    Returns:
        Partial function that extracts config value from wildcards
    """

    def getter(wildcards):
        case_cfg = get_case_config(wildcards, apply_wildcards=True)

        # Traverse key path
        value = case_cfg
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
                if value == default:
                    return default
            elif isinstance(value, list):
                # Handle list indexing if key is integer
                try:
                    value = value[int(key)]
                except (ValueError, IndexError, TypeError):
                    return default
            else:
                return default

        return value if value is not None else default

    return getter


# Convenience functions for common config sections
def get_case_scenario(wildcards):
    """Get scenario config for a case"""
    return get_case_config(wildcards).get("scenario", {})


def get_case_sector(wildcards):
    """Get sector config for a case"""
    return get_case_config(wildcards).get("sector", {})


def get_case_electricity(wildcards):
    """Get electricity config for a case"""
    return get_case_config(wildcards).get("electricity", {})


def get_case_solving(wildcards):
    """Get solving config for a case"""
    return get_case_config(wildcards).get("solving", {})


def get_case_costs(wildcards):
    """Get costs config for a case"""
    return get_case_config(wildcards).get("costs", {})


def get_case_clustering(wildcards):
    """Get clustering config for a case"""
    return get_case_config(wildcards).get("clustering", {})


def get_transmission_network(wildcards):
    """
    Get transmission network with proper fallback hierarchy.
    Priority: wildcard > case config > global config

    Args:
        wildcards: Snakemake wildcards object

    Returns:
        str: Transmission network name
    """
    # 1. If wildcard directly has transmission_network, use it
    if hasattr(wildcards, "transmission_network"):
        return wildcards.transmission_network

    # 2. Get from case config using case_config_provider
    case_cfg = get_case_config(wildcards, apply_wildcards=False)
    transmission_network = case_cfg.get("model_topology", {}).get(
        "transmission_network"
    )

    if transmission_network:
        return transmission_network

    # 3. Use global default
    return "tamu"


# ============================================================================
# HELPER FUNCTIONS THAT USE THE NEW SYSTEM
# ============================================================================


def solver_threads(w):
    """Get solver threads from case-specific config"""
    case_cfg = get_case_config(w)
    solver_options = case_cfg.get("solving", {}).get("solver_options", {})
    option_set = (
        case_cfg.get("solving", {}).get("solver", {}).get("options", "gurobi-default")
    )
    threads = solver_options.get(option_set, {}).get("threads", 4)
    return threads


def memory(w):
    """Calculate memory requirements based on case config"""
    case_cfg = get_case_config(w)

    factor = 4.0
    opts = case_cfg.get("scenario", {}).get("opts", "")

    for o in str(opts).split("-"):
        m = re.match(r"^(\d+)h$", o, re.IGNORECASE)
        if m is not None:
            factor /= int(m.group(1))
            break

    for o in str(opts).split("-"):
        m = re.match(r"^(\d+)seg$", o, re.IGNORECASE)
        if m is not None:
            factor *= int(m.group(1)) / 8760
            break

    clusters = case_cfg.get("scenario", {}).get("clusters", [53])[0]
    simpl = case_cfg.get("scenario", {}).get("simpl", [53])[0]

    if str(clusters).endswith("m") or str(clusters).endswith("c"):
        val = int(factor * (50000 + 30 * int(simpl) + 195 * int(str(clusters)[:-1])))
    elif clusters == "all":
        val = int(factor * (18000 + 180 * 4000))
    else:
        val = int(factor * (15000 + 195 * int(clusters)))

    planning_horizons = case_cfg.get("scenario", {}).get("planning_horizons", [2050])
    return int(val * len(planning_horizons))


def input_custom_extra_functionality(w):
    """Get custom extra functionality path from case config"""
    case_cfg = get_case_config(w)
    path = (
        case_cfg.get("solving", {})
        .get("options", {})
        .get("custom_extra_functionality", False)
    )
    if path:
        return os.path.join(os.path.dirname(workflow.snakefile), path)
    return []


def solved_previous_horizon(w):
    """Get path to previous horizon network"""
    planning_horizons = case_config_provider("scenario", "planning_horizons")(w)
    i = planning_horizons.index(int(w.planning_horizons))
    planning_horizon_p = str(planning_horizons[i - 1])

    transmission_network = get_transmission_network(w)

    return (
        RESULTS
        + f"{transmission_network}/"
        + f"{w.case}/networks/solved_network_"
        + planning_horizon_p
        + ".nc"
    )


# Check if the workflow has access to the internet
def has_internet_access(url="www.zenodo.org") -> bool:
    import http.client as http_client

    conn = http_client.HTTPConnection(url, timeout=5)
    try:
        conn.request("HEAD", "/")
        return True
    except:
        return False
    finally:
        conn.close()
