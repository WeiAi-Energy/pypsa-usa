"""Rules for post procesing solved networks"""


rule plot_simpsec_capacity:
    params:
        sector=get_case_sector,
    input:
        network=RESULTS + "{transmission_network}/{case}/networks/solved_network.nc",
    output:
        generation_capacities=RESULTS
        + "{transmission_network}/{case}/figures/capacity/generation_capacities.png",
        transport_capacities=RESULTS
        + "{transmission_network}/{case}/figures/capacity/transport_capacities.png",
        storage_energy_capacities=RESULTS
        + "{transmission_network}/{case}/figures/capacity/storage_energy_capacities.png",
        storage_power_capacities=RESULTS
        + "{transmission_network}/{case}/figures/capacity/storage_power_capacities.png",
    log:
        LOGS + "{transmission_network}/{case}/plot_capacity.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_simpsec_capacity.py"


rule plot_simpsec_dispatch:
    input:
        network=RESULTS + "{transmission_network}/{case}/networks/solved_network.nc",
    output:
        electricity_dispatch=RESULTS
        + "{transmission_network}/{case}/figures/dispatch/electricity_dispatch.png",
        gas_dispatch=RESULTS
        + "{transmission_network}/{case}/figures/dispatch/gas_dispatch.png",
        hydrogen_dispatch=RESULTS
        + "{transmission_network}/{case}/figures/dispatch/hydrogen_dispatch.png",
        storage_soc=RESULTS
        + "{transmission_network}/{case}/figures/dispatch/storage_soc.png",
    log:
        LOGS + "{transmission_network}/{case}/plot_dispatch.log",
    threads: 1
    resources:
        mem_mb=8000,
    script:
        "../scripts/plot_simpsec_dispatch.py"


rule plot_simpsec_network:
    params:
        sector=get_case_sector,
        line_max_extension=config["lines"]["max_extension"],
    input:
        network=RESULTS + "{transmission_network}/{case}/networks/solved_network.nc",
        regions_onshore=RESOURCES
        + "{transmission_network}/Geospatial/regions_onshore_clustered.geojson",
    output:
        elec_cap_network=RESULTS
        + "{transmission_network}/{case}/figures/maps/elec_cap_network.png",
        gas_cap_network=RESULTS
        + "{transmission_network}/{case}/figures/maps/gas_cap_network.png",
        h2_cap_network=RESULTS
        + "{transmission_network}/{case}/figures/maps/h2_cap_network.png",
        large_storage_cap_network=RESULTS
        + "{transmission_network}/{case}/figures/maps/large_storage_cap_network.png",
        small_storage_cap_network=RESULTS
        + "{transmission_network}/{case}/figures/maps/small_storage_cap_network.png",
        demand_cap_network=RESULTS
        + "{transmission_network}/{case}/figures/maps/demand_cap_network.png",
    log:
        LOGS + "{transmission_network}/{case}/plot_network.log",
    threads: 1
    resources:
        mem_mb=8000,
    script:
        "../scripts/plot_simpsec_network.py"


rule plot_simpsec_cost:
    input:
        network=RESULTS + "{transmission_network}/{case}/networks/solved_network.nc",
    output:
        costs=RESULTS + "{transmission_network}/{case}/figures/cost/system_costs.png",
        lmps=RESULTS + "{transmission_network}/{case}/figures/cost/lmps.png",
    log:
        LOGS + "{transmission_network}/{case}/plot_cost.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_simpsec_cost.py"


rule plot_simpsec_flexibility:
    input:
        network=RESULTS + "{transmission_network}/{case}/networks/solved_network.nc",
    output:
        flexibility_interconnection_week=RESULTS
        + "{transmission_network}/{case}/figures/flexibility/flexibility_contribution_Interconnection_week.png",
        flexibility_interconnection_month=RESULTS
        + "{transmission_network}/{case}/figures/flexibility/flexibility_contribution_Interconnection_month.png",
        flexibility_interconnection_season=RESULTS
        + "{transmission_network}/{case}/figures/flexibility/flexibility_contribution_Interconnection_season.png",
    log:
        LOGS + "{transmission_network}/{case}/plot_flexibility.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_simpsec_flexibility.py"
