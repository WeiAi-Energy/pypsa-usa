# Rules to Optimize/Solve Network


def solve_network_input(wildcards):
    resolved = scenario_lookup(wildcards)
    return (
        case_resource_dir(wildcards)
        + f"elec_ec_l{resolved.ll}_{resolved.opts}.nc"
    )


rule solve_network:
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        planning_horizons=config_provider("scenario", "planning_horizons"),
        transmission_network=config_provider("model_topology", "transmission_network"),
        opts=config_provider("scenario", "opts"),
    input:
        network=solve_network_input,
        sector_costs="repo_data/costs/simple_sector_costs.csv",
        hydrogen_demand_share="repo_data/ReEDS_Constraints/hydrogen_demand_share.csv",
        flowgates="repo_data/ReEDS_Constraints/transmission/transmission_capacity_init_AC_ba_NARIS2024.csv",
        safer_reeds="config/policy_constraints/reeds/prm_annual.csv",
        rps_reeds="config/policy_constraints/reeds/rps_fraction.csv",
        ces_reeds="config/policy_constraints/reeds/ces_fraction.csv",
    output:
        network=result_network_path(),
        config=result_config_path(),
    log:
        solver=normpath(CASE_LOGS + "solve_network/solver.log"),
        python=CASE_LOGS + "solve_network/python.log",
    benchmark:
        CASE_BENCHMARKS + "solve_network"
    threads:
        lambda wildcards: config_provider("solving", "solver_options")(wildcards)[
            config_provider("solving", "solver", "options")(wildcards)
        ].get("threads", 4)
    resources:
        walltime=config_provider("solving", "walltime", default="24:00:00"),
        mem_mb=config_provider("solving", "mem", default=28000),
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network.py"
