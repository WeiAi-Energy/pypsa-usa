"""Rules for post procesing solved networks"""


def postprocess_regions_onshore(wildcards):
    resolved = scenario_lookup(wildcards)
    cfg = config_for_wildcards(wildcards)
    if cfg["custom_files"].get("activate", False):
        return (
            cfg["custom_files"]["files_path"]
            + "regions_onshore.geojson"
        )
    return clustered_region_path(resolved, "onshore", cfg=cfg)


def postprocess_regions_offshore(wildcards):
    resolved = scenario_lookup(wildcards)
    cfg = config_for_wildcards(wildcards)
    if cfg["custom_files"].get("activate", False):
        return (
            cfg["custom_files"]["files_path"]
            + "regions_offshore.geojson"
        )
    return clustered_region_path(resolved, "offshore", cfg=cfg)


rule plot_network_maps:
    input:
        network=resolved_result_network_path,
        regions_onshore=postprocess_regions_onshore,
        regions_offshore=postprocess_regions_offshore,
    params:
        electricity=config_provider("electricity"),
        plotting=config_provider("plotting"),
        retirement=config_provider("electricity", "retirement", default="technical"),
    output:
        **{
            fig: result_figure_path("maps", figure=fig)
            for fig in FIGURES_MAPS
        },
    log:
        CASE_LOGS + "plot_figures/plot_network_maps.log",
    threads: 1
    resources:
        mem_mb=7000,
        walltime="00:30:00",
    script:
        "../scripts/plot_network_maps.py"


rule plot_statistics:
    input:
        network=resolved_result_network_path,
        regions_onshore=postprocess_regions_onshore,
        regions_offshore=postprocess_regions_offshore,
    params:
        electricity=config_provider("electricity"),
        plotting=config_provider("plotting"),
        retirement=config_provider("electricity", "retirement", default="technical"),
    output:
        **{
            fig: result_figure_path("emissions", figure=fig)
            for fig in FIGURES_EMISSIONS
        },
        **{
            fig: result_figure_path("production", figure=fig)
            for fig in FIGURES_PRODUCTION
        },
        **{
            fig: result_figure_path("system", figure=fig)
            for fig in FIGURES_SYSTEM
        },
        statistics_summary=result_figure_path("statistics", figure="statistics.csv"),
        statistics_dissaggregated=result_figure_path(
            "statistics",
            figure="statistics_dissaggregated.csv",
        ),
        cost_breakdown=result_figure_path("statistics", figure="cost_breakdown.csv"),
        generators=result_figure_path("statistics", figure="generators.csv"),
        storage_units=result_figure_path("statistics", figure="storage_units.csv"),
        links=result_figure_path("statistics", figure="links.csv"),
        buses=result_figure_path("statistics", figure="buses.csv"),
        lines=result_figure_path("statistics", figure="lines.csv"),
        stores=result_figure_path("statistics", figure="stores.csv"),
        global_constraints=result_figure_path(
            "statistics",
            figure="global_constraints.csv",
        ),
    log:
        CASE_LOGS + "plot_figures/plot_statistics.log",
    threads: 1
    resources:
        mem_mb=5000,
        walltime="00:30:00",
    script:
        "../scripts/plot_statistics.py"
