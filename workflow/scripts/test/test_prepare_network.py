import os
import sys
from types import SimpleNamespace

import pandas as pd
import pypsa
import pytest

if not hasattr(pypsa.components, "convert_lines_to_line_x"):
    pypsa.components.convert_lines_to_line_x = lambda *args, **kwargs: None

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from _helpers import calculate_annuity
from prepare_network import (
    _apply_bidirectional_transmission_link_volume_correction,
    _get_line_x_conversion_candidates,
    _get_line_x_conversion_config,
    _get_phase_shifting_transformer_annualized_capex_per_mw,
    _get_sssc_annualized_capex_per_mw,
    _infer_timestep_hours,
    _rescale_representative_metadata_steps,
    average_every_nhours,
    set_line_nom_max,
)


def _make_network():
    return SimpleNamespace(
        links=pd.DataFrame(
            {
                "carrier": ["DC", "DC"],
                "length": [100.0, 100.0],
            },
            index=["dc1_fwd", "dc1_rev"],
        ),
    )


def test_average_every_nhours_preserves_investment_period_for_weather_timestamps():
    n = pypsa.Network()
    weather_timesteps = pd.date_range("2012-01-01", periods=6, freq="h")
    n.set_snapshots(
        pd.MultiIndex.from_product(
            [[2050], weather_timesteps],
            names=["period", "timestep"],
        ),
    )
    n.set_investment_periods([2050])

    averaged = average_every_nhours(n, "3h")

    assert averaged.investment_periods.equals(pd.Index([2050]))
    assert averaged.snapshots.get_level_values("period").unique().equals(pd.Index([2050]))
    assert list(averaged.snapshots.get_level_values("timestep")) == list(weather_timesteps[[0, 3]])


def test_infer_timestep_hours_reads_multiindex_snapshots():
    n = pypsa.Network()
    weather_timesteps = pd.date_range("2012-01-01", periods=4, freq="3h")
    n.set_snapshots(
        pd.MultiIndex.from_product(
            [[2050], weather_timesteps],
            names=["period", "timestep"],
        ),
    )

    assert _infer_timestep_hours(n) == 3.0


def test_rescale_representative_metadata_steps_divides_by_resolution_factor():
    metadata = {
        "2050": {
            "periods": [
                {"period_id": 0, "kind": "representative", "steps": 24},
                {"period_id": 1, "kind": "extreme", "steps": 24},
            ],
        },
    }

    rescaled = _rescale_representative_metadata_steps(metadata, scale_factor=3.0)

    assert [entry["steps"] for entry in rescaled["2050"]["periods"]] == [8, 8]


def test_rescale_representative_metadata_steps_noop_for_unit_factor():
    metadata = {"2050": {"periods": [{"period_id": 0, "steps": 24}]}}

    rescaled = _rescale_representative_metadata_steps(metadata, scale_factor=1.0)

    assert rescaled["2050"]["periods"][0]["steps"] == 24


def test_rescale_representative_metadata_steps_rejects_incompatible_resolution():
    metadata = {"2050": {"periods": [{"period_id": 0, "steps": 24}]}}

    with pytest.raises(ValueError, match="incompatible"):
        _rescale_representative_metadata_steps(metadata, scale_factor=5.0)


def test_bidirectional_link_volume_correction_applies_to_directional_links():
    n = _make_network()

    _apply_bidirectional_transmission_link_volume_correction(n)

    assert n.links["length"].tolist() == [50.0, 50.0]


def test_line_x_conversion_config_reads_convert_lines_to_line_x():
    assert _get_line_x_conversion_config(
        {
            "convert_lines_to_line_x": {
                "enable": True,
            },
        },
    ) == {
        "enable": True,
    }


def test_line_x_conversion_candidates_exclude_tree_connected_lines():
    n = pypsa.Network()
    for bus_name in ["b1", "b2", "b3"]:
        n.add("Bus", bus_name)

    n.add("Line", "meshed", bus0="b1", bus1="b2", x=0.1, r=0.01, s_nom=100)
    n.add("Line", "tree", bus0="b2", bus1="b3", x=0.1, r=0.01, s_nom=100)

    line_names, excluded_tree_lines = _get_line_x_conversion_candidates(
        n,
        pd.Index(["b3"]),
    )

    assert list(line_names) == ["meshed"]
    assert excluded_tree_lines == 1


def test_phase_shifting_transformer_annualized_capex_per_mw_uses_config():
    annualized = _get_phase_shifting_transformer_annualized_capex_per_mw(
        {
            "phase_shifting_transformer": {
                "capex_per_kw": 40,
                "cost_recovery_period_years": 60,
                "wacc_real": 0.044,
            },
        },
    )

    assert annualized == pytest.approx(calculate_annuity(60, 0.044) * 40 * 1e3)


def test_sssc_annualized_capex_per_mw_reads_line_x_config():
    annualized = _get_sssc_annualized_capex_per_mw(
        {
            "convert_lines_to_line_x": {
                "capex_per_kva": 61.5,
                "cost_recovery_period_years": 30,
                "wacc_real": 0.044,
            },
        },
    )

    assert annualized == pytest.approx(calculate_annuity(30, 0.044) * 61.5 * 1e3)


def test_set_line_nom_max_disables_line_expansion_when_extension_limit_is_zero():
    n = SimpleNamespace(
        lines=pd.DataFrame(
            {
                "s_nom": [100.0],
                "s_nom_max": [500.0],
                "s_nom_extendable": [True],
            },
            index=["line1"],
        ),
        line_xs=pd.DataFrame(
            {
                "s_nom": [80.0],
                "s_nom_max": [400.0],
                "s_nom_extendable": [True],
            },
            index=["line_x1"],
        ),
        links=pd.DataFrame(
            {
                "carrier": ["DC"],
                "p_nom": [50.0],
                "p_nom_max": [150.0],
                "p_nom_extendable": [True],
            },
            index=["dc1"],
        ),
    )

    set_line_nom_max(n, s_nom_max_ext=0)

    assert bool(n.lines.at["line1", "s_nom_extendable"]) is False
    assert n.lines.at["line1", "s_nom_max"] == 100.0
    assert bool(n.line_xs.at["line_x1", "s_nom_extendable"]) is False
    assert n.line_xs.at["line_x1", "s_nom_max"] == 80.0


def test_set_line_nom_max_disables_dc_link_expansion_when_extension_limit_is_zero():
    n = SimpleNamespace(
        lines=pd.DataFrame(
            {
                "s_nom": [100.0],
                "s_nom_max": [500.0],
                "s_nom_extendable": [True],
            },
            index=["line1"],
        ),
        line_xs=pd.DataFrame(),
        links=pd.DataFrame(
            {
                "carrier": ["DC", "AC"],
                "p_nom": [50.0, 60.0],
                "p_nom_max": [150.0, 160.0],
                "p_nom_extendable": [True, True],
            },
            index=["dc1", "ac1"],
        ),
    )

    set_line_nom_max(n, p_nom_max_ext=0)

    assert bool(n.links.at["dc1", "p_nom_extendable"]) is False
    assert n.links.at["dc1", "p_nom_max"] == 50.0
    assert bool(n.links.at["ac1", "p_nom_extendable"]) is True
    assert n.links.at["ac1", "p_nom_max"] == 160.0
