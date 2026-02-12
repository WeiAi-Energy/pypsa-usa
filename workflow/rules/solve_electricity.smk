# Rules to Optimize/Solve Network

def is_milp(wildcards):
    case_cfg = get_case_config(wildcards)  # Get case-specific config
    ng_options = case_cfg.get("sector",{}).get("natural_gas",{})
    binary_storage = ng_options.get("binary_storage",False)
    binary_pipeline = ng_options.get("binary_pipeline",False)

    return binary_storage or binary_pipeline


def agg_level(wildcards):
    case_cfg = get_case_config(wildcards)  # Get case-specific config
    ng_options = case_cfg.get("sector",{}).get("natural_gas",{})
    aggregate_storage = ng_options.get("aggregate_storage",False)
    aggregate_pipeline = ng_options.get("aggregate_pipeline",False)

    return float(aggregate_storage) + float(aggregate_pipeline)


def memory(wildcards, input, attempt):
    file_size_sum = sum(getattr(f,'size',0) for f in input if hasattr(f,'size'))

    factor = 3 - agg_level(wildcards)

    if is_milp(wildcards):
        calculated_memory = file_size_sum // 100000 * attempt * 200 * factor
        return max(10000,calculated_memory)
    else:
        calculated_memory = file_size_sum // 100000 * attempt * 160 * factor
        return max(5000,calculated_memory)


rule solve_network:
    params:
        # Use case_config_provider for all parameters
        opts=case_config_provider("scenario", "opts"),
        planning_horizons=case_config_provider("scenario", "planning_horizons"),
        solving=get_case_solving,  # Use convenience function
        sector=get_case_sector,    # Use convenience function
        foresight=case_config_provider("foresight"),
        nza_scenario=case_config_provider("scenario","nza_scenario"),
        co2_sequestration_potential=case_config_provider(
            "sector", "co2", "sequestration_potential", default=200
        ),
        transmission_network=get_transmission_network,
    input:
        network=RESOURCES + "{transmission_network}/{case}/sector_network.nc",
        flowgates="repo_data/ReEDS_Constraints/transmission/transmission_capacity_init_AC_ba_NARIS2024.csv",
        safer_reeds="config/policy_constraints/reeds/prm_annual.csv",
        rps_reeds="config/policy_constraints/reeds/rps_fraction.csv",
        ces_reeds="config/policy_constraints/reeds/ces_fraction.csv",
        reference_network=get_reference_network_for_only_power,
        **{
            f"gen_cost_mult_{Path(x).stem}": f"repo_data/locational_multipliers/{Path(x).name}"
            for x in Path("repo_data/locational_multipliers/").glob("*")
        },
    output:
        network=RESULTS + "{transmission_network}/{case}/networks/solved_network.nc",
        config=RESULTS + "{transmission_network}/{case}/configs/config.yaml",
    log:
        solver=normpath(LOGS + "{transmission_network}/{case}/solve_network_solver.log"),
        python=LOGS + "{transmission_network}/{case}/solve_network_python.log",
    threads: solver_threads
    resources:
        mem_mb=lambda wildcards, input, attempt: memory(wildcards, input, attempt),
        walltime=case_config_provider("solving", "walltime", default="12:00:00"),
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network.py"
