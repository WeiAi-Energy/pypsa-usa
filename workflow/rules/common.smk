# SPDX-FileCopyrightText: : 2023-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

import copy
import logging
import re
from functools import partial, lru_cache

import os, sys, glob

smk_logger = logging.getLogger("snakemake")

path = workflow.source_path("../scripts/_helpers.py")
sys.path.insert(0, os.path.dirname(path))

from _helpers import validate_checksum, update_config_from_wildcards
from snakemake.utils import update_config


# opts tokens each scenario.decarbonization level requires/forbids in scenario.opts.
# Required tokens are ordered so auto-inserted tokens come out in a stable, readable order.
DECARBONIZATION_OPTS = {
    "BAU": (("TCT", "RPS"), ("REM",)),
    "95Emission": (("TCT", "RPS", "REM"), ()),
    "100VRE": (("TCT", "RPS"), ("REM",)),
}


def _apply_decarbonization_opts(opt_string, required, forbidden):
    """Add any missing required tokens and strip forbidden ones from one opts string."""
    tokens = [token for token in str(opt_string).split("-") if token]
    tokens = [token for token in tokens if token not in forbidden]
    missing = [token for token in required if token not in tokens]
    return "-".join([*missing, *tokens])


def normalize_electricity_config(config_obj):
    """Normalize scenario defaults and reject sector-coupled configurations."""
    scenario = config_obj.setdefault("scenario", {})
    sector = scenario.get("sector", "")
    if sector not in ("", "E"):
        raise ValueError(f"This workflow supports electricity plus electrolyzers only; got sector={sector!r}.")
    scenario["sector"] = "E"

    planning_horizons = scenario.get("planning_horizons", [])
    if planning_horizons and len(planning_horizons) != 1:
        raise ValueError(
            "This workflow supports a single planning horizon only; got "
            f"scenario.planning_horizons={planning_horizons!r}."
        )

    interconnect = scenario.get("interconnect", [])
    if interconnect and len(interconnect) != 1:
        raise ValueError(
            "This workflow supports a single interconnect only; got "
            f"scenario.interconnect={interconnect!r}."
        )

    if scenario.get("demand_level", "High") not in {"Low", "Mid", "High"}:
        raise ValueError("scenario.demand_level must be Low, Mid, or High.")

    decarbonization = scenario.get("decarbonization")
    if decarbonization is not None:
        if decarbonization not in DECARBONIZATION_OPTS:
            raise ValueError(
                f"scenario.decarbonization must be one of {sorted(DECARBONIZATION_OPTS)}; got {decarbonization!r}.",
            )
        required, forbidden = DECARBONIZATION_OPTS[decarbonization]
        fixed_opts = [
            _apply_decarbonization_opts(opt_string, required, forbidden)
            for opt_string in scenario.get("opts", [])
        ]
        if fixed_opts != scenario.get("opts", []):
            smk_logger.info(
                "scenario.decarbonization=%r adjusted scenario.opts from %r to %r.",
                decarbonization,
                scenario.get("opts", []),
                fixed_opts,
            )
        scenario["opts"] = fixed_opts
    return config_obj


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
    return normalize_electricity_config(merged)


@lru_cache
def case_config(case_name):
    """Retrieve a case config based on the overrides in config/cases_backup.yaml."""
    cases = config.get("cases", {})
    if case_name not in cases:
        raise ValueError(f"Case '{case_name}' not found in config/cases_backup.yaml.")
    cfg = merge_configs(config, cases[case_name] or {})
    scenario = cfg.setdefault("scenario", {})
    for level in ("Low", "Mid", "High"):
        if f"_{level}Dmd_" in case_name:
            scenario["demand_level"] = level
            break
    return cfg


def _wildcard_matches_config(config_value, wildcard_value):
    if isinstance(config_value, list):
        return str(wildcard_value) in {str(value) for value in config_value}
    return config_value is not None and str(config_value) == str(wildcard_value)


def infer_case_from_wildcards(wildcards):
    """Infer a case config for shared-resource rules that do not carry a case wildcard."""
    if "case" in wildcards.keys():
        return wildcards.case

    cases = config.get("cases", {})
    if not cases:
        return None

    candidates = []
    for case_name in cases:
        cfg = case_config(case_name)
        scenario_cfg = cfg.get("scenario", {})

        if "opts" in wildcards.keys():
            if not _wildcard_matches_config(scenario_cfg.get("opts"), wildcards.opts):
                continue

        if "ll" in wildcards.keys():
            if not _wildcard_matches_config(scenario_cfg.get("ll"), wildcards.ll):
                continue

        if "sector" in wildcards.keys():
            if not _wildcard_matches_config(scenario_cfg.get("sector"), wildcards.sector):
                continue

        candidates.append(case_name)

    return candidates[0] if candidates else None


def config_for_wildcards(wildcards):
    """Return the effective config for the given wildcards."""
    inferred_case = infer_case_from_wildcards(wildcards)
    if inferred_case is not None:
        base_config = case_config(inferred_case)
    else:
        base_config = copy.deepcopy(config)
    return update_config_from_wildcards(base_config, wildcards, inplace=False)


def case_resource_dir(wildcards):
    """Return the case-specific resource root for the current wildcards."""
    if "case" in wildcards.keys():
        return CASE_RESOURCES.format(case=wildcards.case)
    return CASE_RESOURCES


def case_results_dir(wildcards):
    """Return the case-specific results root for the current wildcards."""
    if "case" in wildcards.keys():
        return CASE_RESULTS.format(case=wildcards.case)
    return CASE_RESULTS


def case_logs_dir(wildcards):
    """Return the case-specific log root for the current wildcards."""
    if "case" in wildcards.keys():
        return CASE_LOGS.format(case=wildcards.case)
    return CASE_LOGS


def case_benchmarks_dir(wildcards):
    """Return the case-specific benchmark root for the current wildcards."""
    if "case" in wildcards.keys():
        return CASE_BENCHMARKS.format(case=wildcards.case)
    return CASE_BENCHMARKS


def static_getter(wildcards, keys, default):
    """Getter function for static config values."""
    return get_config(config_for_wildcards(wildcards), keys, default)


def dynamic_getter(wildcards, keys, default):
    """Getter function for dynamic config values based on case."""
    return get_config(config_for_wildcards(wildcards), keys, default)


def config_provider(*keys, default=None):
    """Dynamically provide config values based on case-specific overrides.

    Usage in Snakemake rules would look something like:
    params:
        my_param=config_provider("key1", "key2", default="some_default_value")
    """
    if config.get("cases"):
        return partial(dynamic_getter, keys=keys, default=default)
    else:
        return partial(static_getter, keys=keys, default=default)


class WildcardLookup(dict):
    """Dictionary with attribute-style access for Snakemake path helpers."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


def scenario_value_for_case(wildcards, key, cfg=None):
    """Return a single scenario value for case-scoped post-processing rules."""
    if cfg is None:
        cfg = config_for_wildcards(wildcards)

    value = cfg.get("scenario", {}).get(key)
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(
                f"Case-scoped result paths require scenario.{key} to contain exactly one value; got {value}.",
            )
        return value[0]
    return value


def scenario_lookup(wildcards, cfg=None, keys=None):
    """Build a wildcard mapping from a case config for case-scoped post-processing."""
    if cfg is None:
        cfg = config_for_wildcards(wildcards)

    if keys is None:
        keys = ("interconnect", "ll", "opts", "sector")

    values = WildcardLookup(
        {key: scenario_value_for_case(wildcards, key, cfg=cfg) for key in keys},
    )

    if "case" in wildcards.keys():
        values["case"] = wildcards.case

    return values


def resolved_result_network_path(wildcards, suffix=""):
    """Resolve the stable case-scoped solved-network path."""
    values = {"case": wildcards.case} if "case" in wildcards.keys() else {}
    return result_network_path(suffix).format(**values)


def resolved_result_network_path_for_case(case_name, suffix=""):
    """Resolve the stable solved-network path for an arbitrary case."""
    return result_network_path(suffix).format(case=case_name)


def get_cost_year(wildcards=None, cfg=None):
    """Return the configured cost data year, falling back to the first planning horizon."""
    if cfg is None:
        cfg = config_for_wildcards(wildcards) if wildcards is not None else config

    scenario_cfg = cfg.get("scenario", {})
    cost_year = scenario_cfg.get("cost_year")
    if cost_year is not None:
        return int(cost_year)

    planning_horizons = scenario_cfg.get("planning_horizons", [])
    if not planning_horizons:
        raise ValueError("scenario.cost_year or scenario.planning_horizons must be configured.")
    return int(planning_horizons[0])


def get_interconnect(wildcards=None, cfg=None):
    """Return the single configured interconnect."""
    if cfg is None:
        cfg = config_for_wildcards(wildcards) if wildcards is not None else config

    interconnect = cfg.get("scenario", {}).get("interconnect", [])
    if not interconnect:
        raise ValueError("scenario.interconnect must be configured.")
    if isinstance(interconnect, list):
        return str(interconnect[0])
    return str(interconnect)


def representative_periods_config():
    """
    Return the shared representative-period settings.

    These are deliberately read from the top-level config rather than per case:
    ``add_electricity`` and ``simplify_network`` write to shared ``RESOURCES``, so
    their snapshots -- and therefore the selection driving them -- must be
    identical across cases. ``validate_shared_representative_periods`` warns if a
    case tries to override them.
    """
    return config.get("clustering", {}).get("temporal", {}).get("representative_periods", {}) or {}


def representative_periods_enabled():
    """Return whether representative-period selection is active."""
    return bool(representative_periods_config().get("enable", False))


def representative_periods_planning_horizons():
    """
    Return the shared planning horizon driving representative-period selection.

    Read from the top-level config, not per case, for the same reason as
    ``representative_periods_config``: the selection is shared across cases.
    """
    return config.get("scenario", {}).get("planning_horizons", [])


def validate_shared_representative_periods():
    """Warn when a case config overrides the shared representative-period settings."""
    shared = representative_periods_config()
    for case_name in config.get("cases", {}):
        case_setting = (
            case_config(case_name)
            .get("clustering", {})
            .get("temporal", {})
            .get("representative_periods", {})
            or {}
        )
        if case_setting != shared:
            smk_logger.warning(
                "Case %s overrides clustering.temporal.representative_periods, but the setting is "
                "shared across cases and the override is ignored. Shared value: %s",
                case_name,
                shared,
            )


def demand_level_for_wildcards(wildcards, cfg=None):
    """Resolve the demand level: use the rule's own wildcard if it carries one,
    otherwise infer it from the case config (for case-scoped rules downstream
    of ``add_extra_components`` that no longer carry a `demand_level` wildcard)."""
    if "demand_level" in wildcards.keys():
        return wildcards.demand_level
    if cfg is None:
        cfg = config_for_wildcards(wildcards)
    return cfg.get("scenario", {}).get("demand_level", "High")


def representative_periods_dir(wildcards, cfg=None):
    """Return the demand-level-scoped representative-period output directory."""
    level = demand_level_for_wildcards(wildcards, cfg=cfg)
    return RESOURCES + f"{level}Dmd/representative_periods/"


def representative_snapshots_input(wildcards):
    """Snapshot-definition input, empty when representative periods are disabled."""
    if not representative_periods_enabled():
        return {}
    return {"representative_snapshots": representative_periods_dir(wildcards) + "snapshots.csv"}


def representative_metadata_input(wildcards):
    """Snapshot definition plus JSON metadata, empty when disabled."""
    if not representative_periods_enabled():
        return {}
    return {
        "representative_metadata": representative_periods_dir(wildcards) + "metadata.json",
    }


def region_temperature_path(wildcards, cfg=None):
    """Population-weighted region temperature aligned to the representative snapshots."""
    level = demand_level_for_wildcards(wildcards, cfg=cfg)
    return RESOURCES + f"{level}Dmd/region_temperature.csv"


def temperature_derate_input(wildcards):
    """
    Inputs for temperature-dependent capacity derates.

    Gated on representative periods because the temperature table is built for the
    selected source hours only. With representative periods disabled there is no
    derate at all -- the old static summer/winter derate has been removed.
    """
    if not representative_periods_enabled():
        return {}
    return {
        "region_temperature": region_temperature_path(wildcards),
        "outage_forced_temperature": "repo_data/forced_outage/outage_forced_temperature_murphy2019.csv",
    }


def get_cost_file(wildcards=None, cfg=None):
    """Return the configured build_cost_data output path."""
    cost_year = get_cost_year(wildcards=wildcards, cfg=cfg)
    return RESOURCES + f"costs/costs_{cost_year}.csv"


def get_cost_files_for_planning_horizons(wildcards=None, cfg=None):
    """Repeat the configured cost file once per planning horizon for multi-horizon inputs."""
    if cfg is None:
        cfg = config_for_wildcards(wildcards) if wildcards is not None else config
    cost_file = get_cost_file(cfg=cfg)
    return [cost_file] * len(cfg.get("scenario", {}).get("planning_horizons", []))


def elec_network_path(wildcards, cfg=None):
    """Final mandatory topology-reduced electricity network."""
    return clustered_network_path(wildcards, cfg=cfg)


def elec_dem_path(wildcards, cfg=None):
    """`add_demand` output for the case's demand level (shared across cases with the same level)."""
    level = demand_level_for_wildcards(wildcards, cfg=cfg)
    return RESOURCES + f"{level}Dmd/elec_dem.nc"


def electricity_attached_network_path(wildcards, cfg=None):
    """`add_electricity` output before topology reduction (shared per demand level)."""
    level = demand_level_for_wildcards(wildcards, cfg=cfg)
    return RESOURCES + f"{level}Dmd/elec_pp.pkl"


def clustered_network_path(wildcards, cfg=None):
    """`simplify_network` output (shared per demand level)."""
    level = demand_level_for_wildcards(wildcards, cfg=cfg)
    return RESOURCES + f"{level}Dmd/elec.nc"


def clustered_region_path(wildcards, region="onshore", cfg=None):
    level = demand_level_for_wildcards(wildcards, cfg=cfg)
    return RESOURCES + f"{level}Dmd/Geospatial/regions_{region}_elec.geojson"


def clustered_busmap_path(wildcards, cfg=None):
    level = demand_level_for_wildcards(wildcards, cfg=cfg)
    return RESOURCES + f"{level}Dmd/busmap.csv"


def solver_threads(w):
    solver_options = config_provider("solving", "solver_options")(w)
    option_set = config_provider("solving", "solver", "options")(w)
    threads = solver_options[option_set].get("threads", 4)
    return threads


def memory(w):
    factor = 4.0
    for o in w.opts.split("-"):
        m = re.match(r"^(\d+)h$", o, re.IGNORECASE)
        if m is not None:
            factor /= int(m.group(1))
            break
    for o in w.opts.split("-"):
        m = re.match(r"^(\d+)seg$", o, re.IGNORECASE)
        if m is not None:
            factor *= int(m.group(1)) / 8760
            break
    # Topology reduction is mandatory. Retain a conservative fixed
    # electricity-network estimate.
    val = int(factor * (18000 + 180 * 4000))
    return int(val * len(config_provider("scenario", "planning_horizons")(w)))


def input_custom_extra_functionality(w):
    path = config_provider(
        "solving", "options", "custom_extra_functionality", default=False
    )(w)
    if path:
        return os.path.join(os.path.dirname(workflow.snakefile), path)
    return []


def input_size_bytes(inputs):
    """
    Return the total size in bytes for Snakemake inputs.

    In normal workflow execution `inputs` is typically a Snakemake object with a
    `.size` property. During local script debugging some inputs may degrade to
    plain strings, so fall back to filesystem stat calls when needed.
    """
    try:
        return inputs.size
    except AttributeError:
        pass

    def item_size(item):
        if item is None:
            return 0
        if isinstance(item, (list, tuple, set)):
            return sum(item_size(child) for child in item)

        try:
            return item.size
        except AttributeError:
            pass
        except OSError:
            return 0

        path = os.fspath(item)
        return os.path.getsize(path) if os.path.exists(path) else 0

    return sum(item_size(item) for item in inputs)


# Check if the workflow has access to the internet by trying to access the HEAD of specified url
def has_internet_access(url="www.zenodo.org") -> bool:
    import http.client as http_client

    # based on answer and comments from
    # https://stackoverflow.com/a/29854274/11318472
    conn = http_client.HTTPConnection(url, timeout=5)  # need access to zenodo anyway
    try:
        conn.request("HEAD", "/")
        return True
    except:
        return False
    finally:
        conn.close()
