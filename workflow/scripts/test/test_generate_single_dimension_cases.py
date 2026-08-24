import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from generate_single_dimension_cases import generate_cases


def _make_2030_template() -> dict:
    return {
        "scenario": {
            "planning_horizons": [2030],
            "ll": ["v1.07"],
            "demand_level": "Mid",
        },
        "electricity": {},
        "links": {
            "new_dc_network": False,
        },
        "lines": {
            "convert_lines_to_line_x": {
                "enable": True,
                "sssc_tot_max": float("inf"),
            },
        },
        "flexible_electrolysis": {
            "enable": False,
            "annual_hydrogen_twh": 1512,
        },
    }


def _make_2050_template() -> dict:
    return {
        "scenario": {
            "planning_horizons": [2050],
            "ll": ["v1.30"],
            "demand_level": "Mid",
        },
        "electricity": {},
        "links": {
            "new_dc_network": False,
        },
        "lines": {
            "convert_lines_to_line_x": {
                "enable": True,
                "sssc_tot_max": float("inf"),
            },
        },
        "flexible_electrolysis": {
            "enable": True,
            "annual_hydrogen_twh": 1512,
        },
    }


def test_generate_cases_changes_one_dimension_at_a_time():
    generated_cases = generate_cases(
        {
            2030: _make_2030_template(),
            2050: _make_2050_template(),
        },
    )

    assert "2030_MidDmd_v1.07_NoDCNet_SSSC_InfGVA" in generated_cases
    assert "2030_MidDmd_v1.0_NoDCNet_SSSC_InfGVA" in generated_cases
    assert "2030_LowDmd_v1.07_NoDCNet_SSSC_InfGVA" in generated_cases
    assert "2030_MidDmd_v1.07_NoDCNet_NoSSSC" in generated_cases
    assert "2030_MidDmd_v1.07_DCNet_SSSC_InfGVA" not in generated_cases
    assert "2030_MidDmd_v1.07_OffWind_NoDCNet_SSSC_InfGVA" not in generated_cases
    assert "2030_MidDmd_v1.07_NoDCNet_NoSSSC_InfGVA" not in generated_cases

    one_gva_2030 = generated_cases["2030_MidDmd_v1.07_NoDCNet_SSSC_1GVA"]
    assert one_gva_2030["lines"]["convert_lines_to_line_x"]["sssc_tot_max"] == 1000

    high_cost_2030 = generated_cases["2030_MidDmd_v1.07_NoDCNet_SSSC_InfGVA_HighCost"]
    assert high_cost_2030["lines"]["convert_lines_to_line_x"]["capex_per_kva"] == 80
    assert high_cost_2030["lines"]["convert_lines_to_line_x"]["cost_recovery_period_years"] == 15
    assert high_cost_2030["lines"]["convert_lines_to_line_x"]["sssc_tot_max"] == float("inf")

    assert "2050_MidDmd_v1.30_NoDCNet_SSSC_InfGVA" in generated_cases
    assert "2050_MidDmd_v1.30_NoDCNet_SSSC_1GVA" in generated_cases
    assert "2050_MidDmd_v1.30_NoDCNet_SSSC_5GVA" in generated_cases
    assert "2050_MidDmd_v1.30_NoDCNet_SSSC_10GVA" in generated_cases
    assert "2050_MidDmd_v1.30_NoDCNet_SSSC_50GVA" in generated_cases
    assert "2050_MidDmd_v1.30_DCNet_SSSC_InfGVA" in generated_cases
    assert "2050_MidDmd_v1.60_NoDCNet_SSSC_InfGVA" in generated_cases
    assert "2050_MidDmd_v1.30_NoDCNet_SSSC_InfGVA_HighCost" in generated_cases
    assert not any("ReAddSSSC" in case_name for case_name in generated_cases)

    baseline_2050 = generated_cases["2050_MidDmd_v1.30_NoDCNet_SSSC_InfGVA"]
    assert "max_extension" not in baseline_2050["links"]
    assert baseline_2050["lines"]["convert_lines_to_line_x"]["sssc_tot_max"] == float("inf")

    one_gva_2050 = generated_cases["2050_MidDmd_v1.30_NoDCNet_SSSC_1GVA"]
    assert one_gva_2050["flexible_electrolysis"]["annual_hydrogen_twh"] == 1512
    assert one_gva_2050["lines"]["convert_lines_to_line_x"]["sssc_tot_max"] == 1000

    high_cost_2050 = generated_cases["2050_MidDmd_v1.30_NoDCNet_SSSC_InfGVA_HighCost"]
    assert high_cost_2050["lines"]["convert_lines_to_line_x"]["capex_per_kva"] == 80
    assert high_cost_2050["lines"]["convert_lines_to_line_x"]["cost_recovery_period_years"] == 15

    assert "2050_HighDmd_vinf_OffWind_DCNet_NoSSSC" not in generated_cases
