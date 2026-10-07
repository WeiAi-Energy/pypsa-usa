"""ATB model case selection across ATB vintages."""

import pandas as pd
import pytest

from build_cost_data import (
    ATB_MONETARY_PARAMETERS,
    atb_model_cases,
    restate_atb_in_usd2022,
    select_model_case,
)


@pytest.mark.parametrize(
    ("report_year", "cost_case", "tax_credits", "expected"),
    [
        (2024, "R&D", True, ["Market"]),
        (2024, "Exp", False, ["R&D"]),
        (2025, "R&D", False, ["R&D"]),
        (2025, "R&D", True, ["R&D + TC"]),
        (2025, "Exp", False, ["Exp", "R&D"]),
        (2025, "Exp", True, ["Exp + TC", "R&D + TC"]),
    ],
)
def test_atb_model_cases(report_year, cost_case, tax_credits, expected):
    assert atb_model_cases(report_year, cost_case, tax_credits) == expected


def test_select_model_case_falls_back_per_technology():
    # Only wind, PV and batteries carry an Expanded case in the 2025 ATB.
    atb = pd.DataFrame(
        {
            "pypsa-name": ["solar", "solar", "solar", "CCGT", "CCGT"],
            "model_case_nrelatb": ["Exp", "R&D", "Exp + TC", "R&D", "R&D + TC"],
        },
    )

    selected = select_model_case(atb, ["Exp", "R&D"])

    assert selected.set_index("pypsa-name")["model_case_nrelatb"].to_dict() == {
        "solar": "Exp",
        "CCGT": "R&D",
    }


def test_restate_atb_in_usd2022_scales_money_only():
    atb = pd.DataFrame({column: [100.0] for column in ATB_MONETARY_PARAMETERS})
    atb["wacc_real"] = 0.05
    atb["cost_recovery_period_years"] = 30.0
    atb["heat_rate_mmbtu_per_mwh"] = 6.5

    restated = restate_atb_in_usd2022(atb, 2025)

    # The 2025 ATB is in 2023 USD: CPI 100.0 (2022) / 104.1 (2023).
    for column in ATB_MONETARY_PARAMETERS:
        assert restated.at[0, column] == pytest.approx(100.0 / 1.041)
    assert restated.at[0, "wacc_real"] == 0.05
    assert restated.at[0, "cost_recovery_period_years"] == 30.0
    assert restated.at[0, "heat_rate_mmbtu_per_mwh"] == 6.5
    # 2024 ATB is already in 2022 USD.
    assert restate_atb_in_usd2022(atb, 2024).at[0, "capex_per_kw"] == pytest.approx(100.0)


def test_restate_atb_in_usd2022_rejects_unknown_vintage():
    with pytest.raises(ValueError, match="dollar year"):
        restate_atb_in_usd2022(pd.DataFrame({c: [1.0] for c in ATB_MONETARY_PARAMETERS}), 2019)
