import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.legend import Legend
from matplotlib.patches import FancyBboxPatch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from plot_network_maps import (
    LEGEND_FONT_SIZE,
    LEGEND_TITLE_SIZE,
    MAP_BACKGROUND_COLOR,
    MAP_REGION_BOUNDARY_ALPHA,
    MAP_REGION_BOUNDARY_COLOR,
    MAP_REGION_BOUNDARY_WIDTH,
    NO_PIE_TRANSMISSION_LEGEND_Y_OFFSET,
    SSSC_MARKER_AREA_SCALE,
    _add_technology_block,
    _create_right_legend_panel,
    add_capacity_map_legends,
    aggregate_bus_values_to_zone,
    build_reeds_zone_capacity_network,
    create_title,
    draw_model_region_background,
    get_adaptive_legend_values,
    get_capacity_map_boundaries,
    get_capacity_map_bus_values,
    get_capacity_size_max_value,
    get_map_boundaries,
    get_zone_region_boundaries,
    get_line_x_midpoint_values,
    get_line_x_sssc_new_plot_values,
    get_line_x_sssc_plot_values,
    get_plot_interconnect,
    get_model_region_background,
    _scale_sssc_marker_area,
    get_line_plot_values,
    plot_capacity_map,
    get_plot_branch_components,
    get_transmission_link_values,
    resolve_capacity_map_plot_inputs,
)


def make_network():
    return SimpleNamespace(
        lines=pd.DataFrame(
            {
                "s_nom": [100.0],
                "s_nom_opt": [120.0],
                "bus0": ["b1"],
                "bus1": ["b2"],
                "carrier": ["AC"],
            },
            index=["line1"],
        ),
        line_xs=pd.DataFrame(
            {
                "s_nom": [80.0],
                "s_nom_opt": [95.0],
                "sssc_nom_opt": [20.0],
                "bus0": ["b2"],
                "bus1": ["b3"],
                "carrier": ["AC"],
            },
            index=["lx1"],
        ),
        links=pd.DataFrame(
            {
                "carrier": ["AC", "DC", "battery discharger", "electrolysis", "CCGT-95CCS", "tes", "tes"],
                "p_nom": [50.0, 75.0, 10.0, 20.0, 15.0, 30.0, 35.0],
                "p_nom_opt": [60.0, 90.0, 12.0, 25.0, 18.0, 32.0, 40.0],
                "bus0": ["b1", "b2", "b1", "b1", "b2", "b1", "tes"],
                "bus1": ["b2", "b3", "b3", "h2", "b3", "tes", "b2"],
                "efficiency": [1.0, 1.0, 0.9, 0.7, 0.85, 0.99, 0.5],
            },
            index=[
                "ac_link",
                "dc_link",
                "battery_link",
                "electrolysis_link",
                "gas_ccs_link",
                "tes_charge_link",
                "tes_discharge_link",
            ],
        ),
        buses=pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0, 1.5],
                "y": [0.0, 1.0, 2.0, 3.0, 1.5],
                "carrier": ["AC", "AC", "AC", "H2", "tes"],
                "reeds_zone": ["Z1", "Z1", "Z2", "Z2", "Z1"],
            },
            index=["b1", "b2", "b3", "h2", "tes"],
        ),
        generators=pd.DataFrame(columns=["bus", "carrier", "p_nom", "p_nom_opt"]),
        storage_units=pd.DataFrame(
            {
                "bus": ["b2"],
                "carrier": ["battery"],
                "p_nom": [5.0],
                "p_nom_opt": [11.0],
            },
            index=["battery_storage_unit"],
        ),
        carriers=pd.DataFrame(
            {
                "nice_name": {
                    "battery": "Battery Storage",
                    "electrolysis": "Electrolyzer",
                    "CCGT-95CCS": "Gas w/ CCS",
                    "tes": "Thermal Energy Storage",
                },
                "color": {
                    "battery": "#1f77b4",
                    "electrolysis": "#7FD12C",
                    "CCGT-95CCS": "#D2B48C",
                    "tes": "#8c564b",
                    "AC": "#70af1d",
                    "DC": "#8a1caf",
                },
            },
        ),
        components={"Line": {}, "LineX": {}, "Link": {}, "Transformer": {}},
    )



def test_get_plot_branch_components_includes_linex_when_available():
    n = make_network()

    assert get_plot_branch_components(n) == ["Line", "LineX", "Link", "Transformer"]


def test_get_line_plot_values_uses_line_capacity_fields_for_linex():
    n = make_network()

    base_values = get_line_plot_values(n, "s_nom")
    opt_values = get_line_plot_values(n, "s_nom_opt")

    assert base_values.to_dict() == {"line1": 100.0, "lx1": 80.0}
    assert opt_values.to_dict() == {"line1": 120.0, "lx1": 95.0}


def test_get_transmission_link_values_includes_dc_but_excludes_non_transmission_links():
    n = make_network()

    base_values = get_transmission_link_values(n, "p_nom")
    opt_values = get_transmission_link_values(n, "p_nom_opt")

    assert base_values.to_dict() == {"ac_link": 50.0, "dc_link": 75.0}
    assert opt_values.to_dict() == {"ac_link": 60.0, "dc_link": 90.0}


def test_get_line_x_sssc_plot_values_returns_optimized_sssc_capacity():
    n = make_network()

    sssc_values = get_line_x_sssc_plot_values(n, "sssc_nom_opt")

    assert sssc_values.to_dict() == {"lx1": 20.0}


def test_get_line_x_sssc_new_plot_values_returns_opt_minus_base_clipped_at_zero():
    n = make_network()
    n.line_xs["sssc_nom"] = [5.0]
    n.line_xs["sssc_nom_opt"] = [20.0]

    sssc_new_values = get_line_x_sssc_new_plot_values(n)

    assert sssc_new_values.to_dict() == {"lx1": 15.0}


def test_scale_sssc_marker_area_uses_fixed_scale():
    scaled = _scale_sssc_marker_area(pd.Series({"lx1": 10.0}, dtype=float))

    assert scaled.to_dict() == {"lx1": 10.0 * SSSC_MARKER_AREA_SCALE}


def test_get_line_x_midpoint_values_keeps_only_installed_line_x_assets():
    n = make_network()
    values = pd.Series({"lx1": 20.0, "missing": 30.0}, dtype=float)

    midpoint_values = get_line_x_midpoint_values(n, values, min_value=0.0)

    assert list(midpoint_values.columns) == ["x", "y", "value"]
    assert midpoint_values.index.tolist() == ["lx1"]
    assert midpoint_values.at["lx1", "x"] == 1.5
    assert midpoint_values.at["lx1", "y"] == 1.5
    assert midpoint_values.at["lx1", "value"] == 20.0


def test_aggregate_bus_values_to_zone_sums_capacities_within_each_zone():
    n = make_network()
    bus_values = pd.Series(
        [10.0, 5.0, 7.0],
        index=pd.MultiIndex.from_tuples(
            [("b1", "solar"), ("b2", "solar"), ("b3", "solar")],
            names=["bus", "carrier"],
        ),
    )

    zone_values = aggregate_bus_values_to_zone(n, bus_values)

    assert zone_values.to_dict() == {("Z1", "solar"): 15.0, ("Z2", "solar"): 7.0}


def test_build_reeds_zone_capacity_network_aggregates_transmission_between_zones_and_drops_intra_zone_branches():
    n = make_network()
    line_values = get_line_plot_values(n, "s_nom")
    link_values = get_transmission_link_values(n, "p_nom")

    zone_n, zone_line_values, zone_link_values = build_reeds_zone_capacity_network(n, line_values, link_values)

    assert set(zone_n.buses.index) == {"Z1", "Z2"}
    assert zone_line_values.to_dict() == {"Z1~Z2": 80.0}
    assert zone_n.lines.loc["Z1~Z2", ["bus0", "bus1"]].tolist() == ["Z1", "Z2"]
    assert list(zone_link_values.index) == ["DC::Z1~Z2"]
    assert zone_link_values.iloc[0] == 75.0


def test_build_reeds_zone_capacity_network_counts_complete_dc_fwd_rev_pair_once():
    n = SimpleNamespace(
        lines=pd.DataFrame(columns=["s_nom", "s_nom_opt", "bus0", "bus1", "carrier"]),
        line_xs=pd.DataFrame(columns=["s_nom", "s_nom_opt", "bus0", "bus1", "carrier"]),
        links=pd.DataFrame(
            {
                "carrier": ["DC", "DC"],
                "p_nom": [75.0, 75.0],
                "bus0": ["b1", "b3"],
                "bus1": ["b3", "b1"],
            },
            index=["dc1_fwd", "dc1_rev"],
        ),
        buses=pd.DataFrame(
            {
                "x": [0.0, 2.0],
                "y": [0.0, 2.0],
                "carrier": ["AC", "AC"],
                "reeds_zone": ["Z1", "Z2"],
            },
            index=["b1", "b3"],
        ),
        carriers=pd.DataFrame({"nice_name": {}, "color": {}}),
    )
    link_values = get_transmission_link_values(n, "p_nom")

    _zone_n, _zone_line_values, zone_link_values = build_reeds_zone_capacity_network(
        n,
        pd.Series(dtype=float),
        link_values,
    )

    assert list(zone_link_values.index) == ["DC::Z1~Z2"]
    assert zone_link_values.iloc[0] == 75.0


def test_resolve_capacity_map_plot_inputs_aggregates_to_zone_and_drops_sssc_when_pie_shown():
    n = make_network()
    bus_values = pd.Series(
        [10.0],
        index=pd.MultiIndex.from_tuples([("b1", "solar")], names=["bus", "carrier"]),
    )
    line_values = get_line_plot_values(n, "s_nom")
    link_values = get_transmission_link_values(n, "p_nom")
    sssc_values = pd.Series({"lx1": 20.0}, dtype=float)

    plot_n, plot_bus_values, _plot_line_values, _plot_link_values, plot_sssc_values, original_n = (
        resolve_capacity_map_plot_inputs(n, bus_values, line_values, link_values, sssc_values, True)
    )

    assert plot_n is not n
    assert plot_bus_values.to_dict() == {("Z1", "solar"): 10.0}
    assert plot_sssc_values is None
    assert original_n is n


def test_resolve_capacity_map_plot_inputs_keeps_full_resolution_when_pie_hidden():
    n = make_network()
    sssc_values = pd.Series({"lx1": 20.0}, dtype=float)

    plot_n, _plot_bus_values, _plot_line_values, _plot_link_values, plot_sssc_values, original_n = (
        resolve_capacity_map_plot_inputs(
            n,
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            sssc_values,
            False,
        )
    )

    assert plot_n is n
    assert plot_sssc_values is sssc_values
    assert original_n is None


def test_get_plot_interconnect_prefers_case_config_over_missing_wildcard():
    config = {"scenario": {"interconnect": ["usa"]}}

    assert get_plot_interconnect(config=config, wildcards={"case": "demo"}) == "usa"


def test_get_plot_interconnect_falls_back_to_wildcards_when_config_missing():
    assert get_plot_interconnect(config={}, wildcards={"interconnect": "western"}) == "western"


def test_create_title_omits_empty_parentheses():
    assert create_title("Optimal Network Capacities") == "Optimal Network Capacities"
    assert create_title("Optimal Network Capacities", interconnect="usa") == "Optimal Network Capacities\ninterconnect = usa"


def test_plot_capacity_map_omits_default_axis_title_when_none_provided():
    n = make_network()
    n.plot = lambda *args, **kwargs: None
    regions = SimpleNamespace(total_bounds=pd.Series([-125.0, 24.0, -66.0, 50.0]))

    with (
        patch("plot_network_maps.draw_model_region_background"),
        patch("plot_network_maps.get_map_boundaries", return_value=[-125.0, -66.0, 24.0, 50.0]),
    ):
        fig, ax = plot_capacity_map(
            n=n,
            bus_values=pd.Series(dtype=float),
            line_values=pd.Series(dtype=float),
            link_values=pd.Series(dtype=float),
            regions=regions,
            title=None,
        )

    assert ax.get_title() == ""
    plt.close(fig)


def test_plot_capacity_map_hides_bus_pies_when_show_capacity_pie_is_false():
    n = make_network()
    plot_calls = []

    def capture_plot(*args, **kwargs):
        plot_calls.append(kwargs)

    n.plot = capture_plot
    regions = SimpleNamespace(total_bounds=pd.Series([-125.0, 24.0, -66.0, 50.0]))
    bus_values = pd.Series(
        [50.0],
        index=pd.MultiIndex.from_tuples([("b1", "Battery Storage")], names=["bus", "carrier"]),
        dtype=float,
    )

    with (
        patch("plot_network_maps.draw_model_region_background"),
        patch("plot_network_maps.get_map_boundaries", return_value=[-125.0, -66.0, 24.0, 50.0]),
    ):
        fig, _ = plot_capacity_map(
            n=n,
            bus_values=bus_values,
            line_values=pd.Series(dtype=float),
            link_values=pd.Series(dtype=float),
            regions=regions,
            show_capacity_pie=False,
        )

    assert plot_calls[0]["bus_sizes"] == 0
    assert plot_calls[0]["boundaries"] == (-125.0, -66.5, 24.0, 50.0)
    plt.close(fig)


def test_add_capacity_map_legends_skips_technology_block_and_enlarges_transmission_and_sssc_legends_without_pies():
    n = make_network()
    fig, ax = plt.subplots()
    legend_calls = []

    def capture_legend(*args, **kwargs):
        legend_calls.append(kwargs)
        return None

    with (
        patch("plot_network_maps._add_technology_block") as technology_mock,
        patch("plot_network_maps._add_legend_block", side_effect=capture_legend),
    ):
        add_capacity_map_legends(
            ax=ax,
            n=n,
            bus_values=pd.Series(
                [50.0],
                index=pd.MultiIndex.from_tuples([("b1", "Battery Storage")], names=["bus", "carrier"]),
                dtype=float,
            ),
            bus_scale=1.0,
            bus_colors=pd.Series({"Battery Storage": "#1f77b4"}),
            nice_names=pd.Series({"Battery Storage": "Battery Storage"}),
            line_values=pd.Series({"line1": 3000.0}, dtype=float),
            link_values=pd.Series({"dc_link": 2000.0}, dtype=float),
            line_scale=1000.0,
            show_dc_link=True,
            sssc_values=pd.Series({"lx1": 1000.0}, dtype=float),
            show_capacity_pie=False,
        )

    technology_mock.assert_not_called()
    assert [call["title"] for call in legend_calls] == ["Transmission", "SSSC"]
    assert legend_calls[0]["anchor"] == (0.05, 0.28 + NO_PIE_TRANSMISSION_LEGEND_Y_OFFSET)
    assert legend_calls[1]["anchor"][1] == 0.28
    for call in legend_calls:
        assert call["fontsize"] == LEGEND_FONT_SIZE + 5
        assert call["title_fontsize"] == LEGEND_TITLE_SIZE + 5
    plt.close(fig)


def test_get_map_boundaries_returns_pypsa_boundary_order_with_default_shrink():
    regions = SimpleNamespace(total_bounds=pd.Series([-125.0, 24.0, -66.0, 50.0]))

    boundaries = get_map_boundaries(regions)

    assert boundaries == (-123.0, -67.0, 24.0, 50.0)


def test_get_map_boundaries_can_skip_shrink():
    regions = SimpleNamespace(total_bounds=pd.Series([-125.0, 24.0, -66.0, 50.0]))

    boundaries = get_map_boundaries(regions, shrink=False)

    assert boundaries == (-125.0, -66.0, 24.0, 50.0)


def test_get_capacity_map_boundaries_tightens_max_longitude_without_pies():
    regions = SimpleNamespace(total_bounds=pd.Series([-125.0, 24.0, -66.0, 50.0]))

    boundaries = get_capacity_map_boundaries(regions, show_capacity_pie=False)

    assert boundaries == (-123.0, -67.5, 24.0, 50.0)


def test_get_capacity_map_boundaries_keeps_default_bounds_with_pies():
    regions = SimpleNamespace(total_bounds=pd.Series([-125.0, 24.0, -66.0, 50.0]))

    boundaries = get_capacity_map_boundaries(regions, show_capacity_pie=True)

    assert boundaries == (-123.0, -67.0, 24.0, 50.0)


def test_get_capacity_map_bus_values_uses_requested_capacity_attr_for_base_capacities():
    n = make_network()

    values = get_capacity_map_bus_values(
        n,
        carriers=["Battery Storage", "Electrolyzer", "Gas w/ CCS", "Thermal Energy Storage"],
        capacity_attr="p_nom",
    )

    assert values.round(3).to_dict() == {
        ("b1", "Electrolyzer"): 20.0,
        ("b2", "Battery Storage"): 5.0,
        ("b2", "Thermal Energy Storage"): 17.5,
        ("b3", "Battery Storage"): 9.0,
        ("b3", "Gas w/ CCS"): 12.75,
    }


def test_get_capacity_map_bus_values_uses_requested_capacity_attr_for_optimized_capacities():
    n = make_network()

    values = get_capacity_map_bus_values(
        n,
        carriers=["Battery Storage", "Electrolyzer", "Gas w/ CCS", "Thermal Energy Storage"],
        capacity_attr="p_nom_opt",
    )

    assert values.round(3).to_dict() == {
        ("b1", "Electrolyzer"): 25.0,
        ("b2", "Battery Storage"): 11.0,
        ("b2", "Thermal Energy Storage"): 20.0,
        ("b3", "Battery Storage"): 10.8,
        ("b3", "Gas w/ CCS"): 15.3,
    }


def test_get_capacity_map_bus_values_excludes_tes_charge_links():
    n = make_network()

    values = get_capacity_map_bus_values(
        n,
        carriers=["Thermal Energy Storage"],
        capacity_attr="p_nom_opt",
    )

    assert values.round(3).to_dict() == {
        ("b2", "Thermal Energy Storage"): 20.0,
    }

    assert ("b1", "Thermal Energy Storage") not in values.index


def test_get_capacity_map_bus_values_keeps_other_link_carriers_on_requested_capacity_attr():
    n = make_network()

    values = get_capacity_map_bus_values(
        n,
        carriers=["Battery Storage", "Electrolyzer", "Gas w/ CCS"],
        capacity_attr="p_nom",
    )

    assert values.round(3).to_dict() == {
        ("b1", "Electrolyzer"): 20.0,
        ("b2", "Battery Storage"): 5.0,
        ("b3", "Battery Storage"): 9.0,
        ("b3", "Gas w/ CCS"): 12.75,
    }


def test_draw_model_region_background_overlays_semitransparent_boundaries():
    class FakeGeoFrame:
        def __init__(self):
            self.calls = []

        def plot(self, **kwargs):
            self.calls.append(kwargs)

    background = FakeGeoFrame()
    boundaries = FakeGeoFrame()
    ax = object()

    with (
        patch("plot_network_maps.get_model_region_background", return_value=background),
        patch("plot_network_maps.get_zone_region_boundaries", return_value=boundaries),
    ):
        draw_model_region_background(ax, regions=object(), n=object())

    assert len(background.calls) == 1
    assert background.calls[0]["ax"] is ax
    assert background.calls[0]["facecolor"] == MAP_BACKGROUND_COLOR
    assert background.calls[0]["edgecolor"] == "none"
    assert background.calls[0]["zorder"] == 0

    assert len(boundaries.calls) == 1
    assert boundaries.calls[0]["ax"] is ax
    assert boundaries.calls[0]["facecolor"] == "none"
    assert boundaries.calls[0]["edgecolor"] == MAP_REGION_BOUNDARY_COLOR
    assert boundaries.calls[0]["linewidth"] == MAP_REGION_BOUNDARY_WIDTH
    assert boundaries.calls[0]["alpha"] == MAP_REGION_BOUNDARY_ALPHA
    assert boundaries.calls[0]["zorder"] == 1


def test_get_zone_region_boundaries_dissolves_regions_by_bus_reeds_zone():
    regions = pd.DataFrame(
        {
            "name": ["b1", "b2", "b3"],
            "geometry": ["g1", "g2", "g3"],
        },
    )
    regions.crs = "EPSG:4326"
    network = SimpleNamespace(
        buses=pd.DataFrame(
            {
                "reeds_zone": ["R1", "R1", "R2"],
            },
            index=["b1", "b2", "b3"],
        ),
    )

    cleaned = object()

    class FakeDissolved(pd.DataFrame):
        @property
        def _constructor(self):
            return FakeDissolved

    class FakeGeoFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return FakeGeoFrame

        def dissolve(self, by=None):
            assert by == "reeds_zone"
            dissolved = FakeDissolved({"geometry": ["merged-r1", "merged-r2"]}, index=["R1", "R2"])
            dissolved.index.name = "reeds_zone"
            return dissolved

    fake_regions = FakeGeoFrame(regions)
    fake_regions.crs = regions.crs

    with patch("plot_network_maps._clean_model_regions", return_value=cleaned) as clean_mock:
        result = get_zone_region_boundaries(network, fake_regions)

    assert result is cleaned
    dissolved_input = clean_mock.call_args.args[0]
    assert list(dissolved_input["reeds_zone"]) == ["R1", "R2"]



def test_get_model_region_background_does_not_reapply_merge_tolerance_after_cleaning():
    class FakeGeoFrame:
        def __init__(self):
            self.dissolve_called = 0
            self.explode_called = 0
            self.reset_called = 0

        def __getitem__(self, key):
            assert key == ["geometry"]
            return self

        def copy(self):
            return self

        def dissolve(self):
            self.dissolve_called += 1
            return self

        def explode(self, index_parts=False):
            assert index_parts is False
            self.explode_called += 1
            return self

        def reset_index(self, drop=True):
            assert drop is True
            self.reset_called += 1
            return self

    cleaned = FakeGeoFrame()

    with (
        patch("plot_network_maps._clean_model_regions", return_value=cleaned) as clean_mock,
        patch("plot_network_maps._apply_region_merge_tolerance") as tolerance_mock,
    ):
        result = get_model_region_background(object())

    assert result is cleaned
    clean_mock.assert_called_once()
    tolerance_mock.assert_not_called()
    assert cleaned.dissolve_called == 1
    assert cleaned.explode_called == 1
    assert cleaned.reset_called == 1


def test_get_adaptive_legend_values_keeps_lower_tiers_when_max_exceeds_one_point_five_times_previous():
    values = get_adaptive_legend_values([100, 500, 1000, 2500, 5000], 3200)

    assert values == [100.0, 500.0, 1000.0, 3200.0]


def test_get_adaptive_legend_values_drops_last_default_tier_when_max_is_close_to_previous():
    values = get_adaptive_legend_values([100, 500, 1000, 2500, 5000], 1400)

    assert values == [100.0, 500.0, 1400.0]


def test_get_capacity_size_max_value_sums_multiindex_bus_slices_before_taking_max():
    bus_values = pd.Series(
        [600.0, 500.0, 700.0],
        index=pd.MultiIndex.from_tuples(
            [("b1", "solar"), ("b1", "wind"), ("b2", "solar")],
            names=["bus", "carrier"],
        ),
    )

    assert get_capacity_size_max_value(bus_values) == 1100.0


def test_add_technology_block_places_capacity_between_heading_and_names_without_background_fill():
    fig, ax = plt.subplots()
    legend_ax = _create_right_legend_panel(fig)
    bus_colors = pd.Series({"solar": "#f4d03f", "onwind": "#b7cde8"})
    nice_names = pd.Series({"solar": "Solar", "onwind": "Onshore Wind"})

    _add_technology_block(
        legend_ax,
        bus_colors=bus_colors,
        nice_names=nice_names,
        bus_legend_values=[10000.0, 30000.0],
        bus_scale=1000.0,
        panel_title="Technology",
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axis_legends = [artist for artist in legend_ax.get_children() if isinstance(artist, Legend)]
    figure_legends = [artist for artist in fig.artists if isinstance(artist, Legend)]
    heading = next(text for text in legend_ax.texts if text.get_text() == "Technology")
    capacity_background = next(
        patch for patch in legend_ax.patches if isinstance(patch, FancyBboxPatch)
    )
    technology_legend = next(
        legend
        for legend in axis_legends
        if [text.get_text() for text in legend.get_texts()] == ["Solar", "Onshore Wind"]
    )

    heading_bbox = heading.get_window_extent(renderer=renderer)
    capacity_bbox = capacity_background.get_window_extent(renderer=renderer)
    technology_bbox = technology_legend.get_window_extent(renderer=renderer)
    figure_bbox = fig.get_window_extent(renderer=renderer)

    assert figure_legends == []
    assert capacity_bbox.y0 <= heading_bbox.y0 <= capacity_bbox.y1
    assert capacity_bbox.y0 > technology_bbox.y1
    assert capacity_background.get_facecolor()[-1] == 0.0
    assert technology_bbox.x0 >= figure_bbox.x0
    assert technology_bbox.x1 <= figure_bbox.x1
    assert technology_legend.get_frame().get_visible() is False

    plt.close(fig)
