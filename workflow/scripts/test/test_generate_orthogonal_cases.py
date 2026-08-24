import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from generate_orthogonal_cases import generate_cases


def _make_template(year: int) -> dict:
    return {
        "scenario": {
            "planning_horizons": [year],
            "ll": ["template-ll"],
        },
        "electricity": {},
        "links": {"new_dc_network": False},
        "lines": {
            "convert_lines_to_line_x": {},
        },
        "flexible_electrolysis": {
            "enable": False,
            "annual_hydrogen_twh": 1512,
        },
    }


def test_generate_cases_covers_demand_ll_dc_sssc_dimensions():
    generated_cases = generate_cases(
        {
            2030: _make_template(2030),
            2050: _make_template(2050),
        },
    )

    assert len(generated_cases) == 180
    assert not any(name.startswith("2030_") for name in generated_cases)

    sssc_case = generated_cases["2050_MidDmd_v1.30_DCNet_SSSC_InfGVA"]
    assert sssc_case["scenario"]["demand_level"] == "Mid"
    assert sssc_case["flexible_electrolysis"] == {"enable": False, "annual_hydrogen_twh": 1512}
    assert sssc_case["lines"]["convert_lines_to_line_x"]["sssc_tot_max"] == float("inf")
    assert "max_extension" not in sssc_case["links"]

    tot_max_case = generated_cases["2050_MidDmd_v1.30_DCNet_SSSC_10GVA"]
    assert tot_max_case["lines"]["convert_lines_to_line_x"]["sssc_tot_max"] == 10000

    no_sssc_case = generated_cases["2050_MidDmd_v1.30_NoDCNet_NoSSSC"]
    assert no_sssc_case["lines"]["convert_lines_to_line_x"]["enable"] is False
    assert "sssc_tot_max" not in no_sssc_case["lines"]["convert_lines_to_line_x"]
    assert "max_extension" not in no_sssc_case["links"]
