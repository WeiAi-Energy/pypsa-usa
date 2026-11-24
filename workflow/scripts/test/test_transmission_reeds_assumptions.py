"""ReEDS transmission assumptions: LCC converter cost, transmission FOM and HVDC losses."""

import pytest
from build_base_network import hvdc_efficiency
from build_cost_data import TRANSMISSION_TECHS, get_transmission_data
from money import to_dollars
from regional_cost import transmission_annual_factor


def _value(data, tech, parameter):
    row = (data["pypsa-name"] == tech) & (data["parameter"] == parameter)
    assert row.sum() == 1
    return float(data.loc[row, "value"].iloc[0])


@pytest.mark.fast
def test_converter_pair_is_two_lcc_stations_in_2004_dollars():
    data = get_transmission_data(2023)
    expected = to_dollars(2 * 150.0, year=2004, to_year=2023)
    assert _value(data, "HVDC inverter pair", "capex_per_kw") == pytest.approx(expected)
    assert expected == pytest.approx(484, rel=0.01)
    # other costs are taken as given in the model's dollar year
    assert _value(data, "HVAC overhead", "capex_per_mw_km") == 1541


@pytest.mark.fast
def test_every_transmission_row_has_reeds_fom():
    data = get_transmission_data(2023)
    for tech in TRANSMISSION_TECHS:
        assert _value(data, tech, "fom_fraction") == 0.015


@pytest.mark.fast
def test_annual_factor_is_capital_recovery_plus_fom():
    data = get_transmission_data(2023)
    costs = data.pivot(index="pypsa-name", columns="parameter", values="value")
    assert transmission_annual_factor(costs, "HVAC overhead") == pytest.approx(0.04756 + 0.015, abs=1e-5)
    # a cost table without fom_fraction annualizes at the capital recovery factor alone
    assert transmission_annual_factor(costs.drop(columns="fom_fraction"), "HVAC overhead") == pytest.approx(
        0.04756,
        abs=1e-5,
    )


@pytest.mark.fast
def test_hvdc_efficiency_follows_reeds():
    # back-to-back tie: the two LCC converter stations only
    assert hvdc_efficiency(0.0) == pytest.approx(1 - 2 * 0.007)
    # 1,000 miles of line: 1.4% at the converters plus 5% on the line
    assert hvdc_efficiency(1000 * 1.609344) == pytest.approx(1 - 0.014 - 0.05)
