"""Voltage- and region-resolved transmission line costs.

The real ReEDS tables are read here rather than mocked: the whole point of the
feature is that the numbers come from those files, and the derivations (voltage
ratios, DC/AC ratio) are cheap assertions that catch a data refresh changing shape.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

from _helpers import (
    LINK_FIXED_COST_COL,
    LINK_UNIT_COST_COL,
    get_complete_bidirectional_link_pairs,
    get_currency_conversion_factor,
    recompute_link_transmission_costs,
)
from regional_cost import (
    NEAR_NEIGHBOUR_PAIRS,
    REEDS_TRANSMISSION_COST_YEAR,
    REFERENCE_VOLTAGE_KV,
    county_unit_cost_field,
    dc_ac_line_cost_ratio,
    load_transmission_basecost,
    load_transmission_pair_costs,
    pair_unit_costs,
    transmission_unit_costs,
    voltage_cost_anchors,
    voltage_cost_exponent,
    voltage_ratio,
)

TRANSMISSION = Path(__file__).parents[2] / "repo_data" / "ReEDS_Constraints" / "transmission"
BASECOST = TRANSMISSION / "rev_transmission_basecost.csv"
DISTANCE_COST = TRANSMISSION / "transmission_distance_cost_500kVac_county.csv"


@pytest.fixture(scope="module")
def basecost():
    return load_transmission_basecost(str(BASECOST))


@pytest.fixture(scope="module")
def pair_table():
    return load_transmission_pair_costs(str(DISTANCE_COST))


# --------------------------------------------------------------------------
# Voltage and DC/AC relationships from rev_transmission_basecost
# --------------------------------------------------------------------------


def test_voltage_anchors_are_region_medians_of_within_region_ratios(basecost):
    anchors = voltage_cost_anchors(basecost)

    assert list(anchors.index) == [69.0, 138.0, 230.0, 500.0]
    assert anchors.loc[500.0] == pytest.approx(1.0)
    # Ratios must be taken inside each region before the median across regions:
    # the county cost table already carries the regional level, so a median of raw
    # USD/MW-mile would fold that level in a second time.
    regions = ["TEPPC", "SCE", "MISO", "Southeast"]
    for kv, label in [(69.0, "69ACsingle"), (138.0, "138ACsingle"), (230.0, "230ACsingle")]:
        within = basecost.loc[label, regions] / basecost.loc["500ACsingle", regions]
        assert anchors.loc[kv] == pytest.approx(float(within.median()))

    assert anchors.loc[69.0] == pytest.approx(4.7151, abs=1e-4)
    assert anchors.loc[138.0] == pytest.approx(3.0104, abs=1e-4)
    assert anchors.loc[230.0] == pytest.approx(2.1102, abs=1e-4)


def test_voltage_anchors_are_monotonically_decreasing(basecost):
    anchors = voltage_cost_anchors(basecost)
    assert (anchors.diff().dropna() < 0).all()


def test_dc_ac_ratio_is_one_number_across_all_reeds_cost_regions(basecost):
    # The four regions agree to 7e-6 relative -- rounding noise in the published
    # USD/MW-mile figures, not a regional spread -- so one scalar is faithful.
    assert dc_ac_line_cost_ratio(basecost) == pytest.approx(623.10 / 1506.22, rel=1e-5)


def test_dc_ac_ratio_rejects_a_regional_spread(basecost):
    tampered = basecost.copy()
    tampered.loc["500DCbipole", "MISO"] *= 1.5
    with pytest.raises(ValueError, match="differs across ReEDS cost regions"):
        dc_ac_line_cost_ratio(tampered)


def test_voltage_cost_exponent_is_anchored_at_the_reference_voltage(basecost):
    exponent = voltage_cost_exponent(voltage_cost_anchors(basecost))

    # The county cost field is quoted at 500 kV AC, so the law must pass through
    # 1.0 there exactly -- a free intercept would rescale every line in the network.
    assert voltage_ratio([REFERENCE_VOLTAGE_KV], exponent) == pytest.approx([1.0])
    assert exponent == pytest.approx(-0.8200, abs=1e-4)


def test_voltage_cost_exponent_stays_within_the_documented_anchor_error(basecost):
    anchors = voltage_cost_anchors(basecost)
    fitted = voltage_ratio(anchors.index.to_numpy(), voltage_cost_exponent(anchors))
    residuals = fitted / anchors.to_numpy() - 1.0

    # The anchors are not collinear in log-log, so one exponent cannot reproduce
    # them; this pins the trade the power law makes (worst -10.4% at 230 kV).
    assert np.abs(residuals).max() == pytest.approx(0.104, abs=5e-3)
    assert dict(zip(anchors.index, np.abs(residuals)))[230.0] == pytest.approx(0.104, abs=5e-3)


def test_voltage_ratio_prices_the_tamu_voltage_classes_on_one_power_law(basecost):
    exponent = voltage_cost_exponent(voltage_cost_anchors(basecost))
    # The synthetic TAMU network carries these classes and ReEDS publishes none of
    # them; the power law covers them with no interpolate-vs-extrapolate seam.
    expected = {100.0: 3.7426, 115.0: 3.3373, 161.0: 2.5326, 345.0: 1.3557, 765.0: 0.7056}
    got = dict(zip(expected, voltage_ratio(list(expected), exponent)))
    for kv, ratio in expected.items():
        assert got[kv] == pytest.approx(ratio, abs=1e-3)

    # Strictly decreasing in voltage across the whole span, tails included.
    span = [69.0, 100.0, 115.0, 138.0, 161.0, 230.0, 345.0, 500.0, 765.0]
    ratios = voltage_ratio(span, exponent)
    assert (np.diff(ratios) < 0).all()


def test_voltage_ratio_is_scale_free(basecost):
    exponent = voltage_cost_exponent(voltage_cost_anchors(basecost))
    # A power law depends only on the voltage ratio, so doubling both voltages
    # leaves their cost ratio unchanged. Piecewise interpolation did not do this.
    low, high = voltage_ratio([115.0, 230.0], exponent)
    low2, high2 = voltage_ratio([230.0, 460.0], exponent)
    assert low / high == pytest.approx(low2 / high2)


def test_voltage_ratio_prices_missing_voltage_at_the_reference(basecost):
    exponent = voltage_cost_exponent(voltage_cost_anchors(basecost))
    assert voltage_ratio([np.nan, 0.0], exponent) == pytest.approx([1.0, 1.0])


# --------------------------------------------------------------------------
# County-pair costs and the county field
# --------------------------------------------------------------------------


def test_pair_table_is_symmetric_and_priced_per_great_circle_km(pair_table):
    raw = pd.read_csv(DISTANCE_COST)
    assert len(pair_table) == 2 * len(raw)

    lookup = pair_unit_costs(pair_table)
    first = raw.iloc[0]
    expected = first.USD2004perMW / (first.length_miles * 1.609344)
    assert lookup.loc[(first.r, first.rr)] == pytest.approx(expected)
    assert lookup.loc[(first.rr, first.r)] == pytest.approx(expected)


def test_county_field_uses_only_the_nearest_pairs():
    table = pd.DataFrame(
        {
            "r": ["pA"] * 4 + ["pB"] * 4,
            "rr": ["pB", "pC", "pD", "pE"] * 2,
            "length_km": [10.0, 20.0, 300.0, 400.0] * 2,
            "unit_cost": [1000.0, 1100.0, 5000.0, 6000.0] * 2,
        },
    )
    near = county_unit_cost_field(table, n_nearest=2)
    # The two 300/400 km pairs are five times as expensive and must not enter.
    assert near.loc["pA"] == pytest.approx(1050.0)
    # Falling back to every incident pair is what the K cut-off exists to avoid.
    assert county_unit_cost_field(table, n_nearest=4).loc["pA"] == pytest.approx(3050.0)


def test_county_field_covers_every_county_in_the_reeds_table(pair_table):
    field = county_unit_cost_field(pair_table)
    assert set(field.index) == set(pair_table["r"])
    assert field.notna().all()
    assert (field > 0).all()


def test_county_field_default_k_is_eight():
    # Chosen from leave-one-out error against the pair values on pairs under 60 km
    # (5.5% MAPE / 0.874 R2 for all incident pairs against 2.7% / 0.954 at K=8).
    assert NEAR_NEIGHBOUR_PAIRS == 8


# --------------------------------------------------------------------------
# Four-level fallback
# --------------------------------------------------------------------------


def _fallback_inputs():
    pair_costs = pd.Series(
        {("p06001", "p06003"): 2000.0, ("p06003", "p06001"): 2000.0},
    )
    field = pd.Series({"p06001": 1000.0, "p06003": 1400.0, "p09110": 3000.0, "p09120": 3400.0})
    return pair_costs, field


def test_fallback_prefers_the_exact_county_pair():
    pair_costs, field = _fallback_inputs()
    unit = transmission_unit_costs(
        pd.Series({"l": "p06001"}),
        pd.Series({"l": "p06003"}),
        pair_costs,
        field,
    )
    assert unit.loc["l"] == pytest.approx(2000.0)


def test_fallback_averages_the_endpoint_county_fields():
    pair_costs, field = _fallback_inputs()
    # Same-county line: ReEDS publishes no r == rr pair, and 77.6% of TAMU lines
    # look like this, so this is the main path rather than a rare fallback.
    same = transmission_unit_costs(pd.Series({"l": "p06001"}), pd.Series({"l": "p06001"}), pair_costs, field)
    assert same.loc["l"] == pytest.approx(1000.0)
    # Cross-county but uncovered pair -> mean of the two fields.
    uncovered = transmission_unit_costs(pd.Series({"l": "p06001"}), pd.Series({"l": "p09110"}), pair_costs, field)
    assert uncovered.loc["l"] == pytest.approx(2000.0)


def test_fallback_uses_the_state_median_for_legacy_connecticut_fips():
    pair_costs, field = _fallback_inputs()
    # ReEDS uses the post-2022 Connecticut planning regions (p09110-p09190); the
    # census county shapes here use the legacy FIPS (p09001-p09015), so only the
    # state matches.
    unit = transmission_unit_costs(
        pd.Series({"l": "p09001"}),
        pd.Series({"l": "p09003"}),
        pair_costs,
        field,
    )
    assert unit.loc["l"] == pytest.approx(3200.0)


def test_fallback_uses_the_national_median_without_a_county():
    pair_costs, field = _fallback_inputs()
    unit = transmission_unit_costs(
        pd.Series({"l": np.nan}, dtype=object),
        pd.Series({"l": np.nan}, dtype=object),
        pair_costs,
        field,
    )
    assert unit.loc["l"] == pytest.approx(float(field.median()))


def test_fallback_resolves_every_branch(pair_table):
    field = county_unit_cost_field(pair_table)
    counties = pd.Series(list(field.index[:50]) + ["p09001", None], dtype=object)
    counties.index = [f"l{i}" for i in range(len(counties))]
    unit = transmission_unit_costs(counties, counties, pair_unit_costs(pair_table), field)
    assert unit.notna().all()
    assert (unit > 0).all()


# --------------------------------------------------------------------------
# Currency
# --------------------------------------------------------------------------


def test_reeds_2004_usd_converts_to_2022_usd():
    assert REEDS_TRANSMISSION_COST_YEAR == 2004
    assert get_currency_conversion_factor(2004, "USD") == pytest.approx(1.549, abs=2e-3)


def test_no_eur_exchange_rate_for_2004():
    with pytest.raises(ValueError, match="No EUR/USD exchange rate"):
        get_currency_conversion_factor(2004, "EUR")


# --------------------------------------------------------------------------
# Link cost recomputation after aggregation
# --------------------------------------------------------------------------


def _link_network():
    n = pypsa.Network()
    n.add("Bus", "a", x=-100.0, y=40.0, carrier="AC")
    n.add("Bus", "b", x=-99.0, y=40.0, carrier="AC")
    n.add("Bus", "c", x=-98.0, y=40.0, carrier="AC")
    n.add("Link", "dc0_fwd", bus0="a", bus1="b", carrier="DC", p_nom=100)
    n.add("Link", "dc0_rev", bus0="b", bus1="a", carrier="DC", p_nom=100)
    n.add("Link", "dc1_fwd", bus0="a", bus1="c", carrier="DC", p_nom=100)
    n.add("Link", "h2", bus0="a", bus1="c", carrier="H2", p_nom=100, capital_cost=7.0)
    n.links[LINK_UNIT_COST_COL] = 0.0
    n.links[LINK_FIXED_COST_COL] = 0.0
    n.links.loc[["dc0_fwd", "dc0_rev", "dc1_fwd"], LINK_UNIT_COST_COL] = 10.0
    n.links.loc[["dc0_fwd", "dc0_rev", "dc1_fwd"], LINK_FIXED_COST_COL] = 1000.0
    return n


def test_paired_links_split_the_corridor_cost_and_singles_carry_it_whole():
    from pypsa.geo import haversine_pts

    n = _link_network()
    recompute_link_transmission_costs(n)

    coords = n.buses[["x", "y"]].astype(float)
    d_ab = float(haversine_pts(coords.loc[["a"]].to_numpy(), coords.loc[["b"]].to_numpy())[0])
    d_ac = float(haversine_pts(coords.loc[["a"]].to_numpy(), coords.loc[["c"]].to_numpy())[0])

    # opts.bidirectional_link forces the two directions to expand together, so
    # half each sums to one corridor.
    assert n.links.at["dc0_fwd", "capital_cost"] == pytest.approx((10.0 * d_ab + 1000.0) / 2)
    assert n.links.at["dc0_rev", "capital_cost"] == pytest.approx((10.0 * d_ab + 1000.0) / 2)
    # dc1_rev was never built (or was dropped as a self-loop), so dc1_fwd is the
    # whole corridor.
    assert n.links.at["dc1_fwd", "capital_cost"] == pytest.approx(10.0 * d_ac + 1000.0)


def test_recompute_ignores_non_transmission_links():
    n = _link_network()
    recompute_link_transmission_costs(n)
    assert n.links.at["h2", "capital_cost"] == pytest.approx(7.0)


def test_recompute_follows_the_buses_when_aggregation_moves_them():
    n = _link_network()
    recompute_link_transmission_costs(n)
    before = n.links.at["dc1_fwd", "capital_cost"]

    # pypsa clustering re-points Links at cluster centroids without rescaling their
    # cost; this is the step that keeps the two consistent.
    n.buses.loc["c", "x"] = -90.0
    recompute_link_transmission_costs(n)
    assert n.links.at["dc1_fwd", "capital_cost"] > before


def test_recompute_is_a_no_op_without_the_stored_unit_costs():
    n = _link_network()
    n.links = n.links.drop(columns=[LINK_UNIT_COST_COL])
    n.links["capital_cost"] = 3.0
    recompute_link_transmission_costs(n)
    assert (n.links["capital_cost"] == 3.0).all()


def test_bidirectional_pairing_needs_both_directions():
    links = pd.DataFrame(index=["x_fwd", "x_rev", "y_fwd", "z"])
    pairs = get_complete_bidirectional_link_pairs(links)
    assert pairs == {"x": {"fwd": "x_fwd", "rev": "x_rev"}}


# --------------------------------------------------------------------------
# Magnitude regression against the national value this replaces
# --------------------------------------------------------------------------


def test_reference_voltage_cost_corroborates_the_national_value_it_replaces(pair_table, basecost):
    """The median county priced at 500 kV, against the 1541 USD/MW-km it replaces.

    1541 (NREL fy21osti/78195) used to price every line at every voltage, and is
    effectively a 500 kV figure -- so the reference voltage is where the two methods
    are comparable at all, and the divergence at every other voltage is the point of
    the change. The two are independent sources (ReEDS reV county costs against an
    NREL generic figure), so they are expected to corroborate, not to match: the
    measured 1892 is 23% above 1541.

    Pinned tightly on purpose: this is the tripwire for a ReEDS data refresh or a
    change in the field/voltage derivation quietly moving the cost level.
    """
    field = county_unit_cost_field(pair_table)
    usd2022 = get_currency_conversion_factor(REEDS_TRANSMISSION_COST_YEAR, "USD")
    exponent = voltage_cost_exponent(voltage_cost_anchors(basecost))
    median_500kv = float(field.median()) * usd2022 * float(voltage_ratio([500.0], exponent)[0])

    assert median_500kv == pytest.approx(1892, rel=0.02), (
        f"500 kV median capex moved to {median_500kv:.0f} USD2022/MW/km"
    )
    # Same order as the value it replaces -- a factor-of-two gap would mean the
    # voltage ratios are anchored at the wrong voltage.
    assert 1 / 1.5 < median_500kv / 1541 < 1.5

    # Lower voltages must come out materially more expensive per MW; that spread is
    # the whole reason for the change.
    assert float(field.median()) * usd2022 * float(voltage_ratio([138.0], exponent)[0]) > 2 * median_500kv


# --------------------------------------------------------------------------
# Survival through the aggregation passes in simplify_network
# --------------------------------------------------------------------------


def _priced_network():
    """Four buses in a line, three corridors priced at three different unit costs."""
    n = pypsa.Network()
    for name, x in [("a", -100.0), ("b", -99.0), ("c", -98.0), ("d", -97.0)]:
        n.add("Bus", name, x=x, y=40.0, v_nom=230.0, carrier="AC")
    for name, (b0, b1), s_nom, unit in [
        ("ab", ("a", "b"), 100.0, 5000.0),
        ("ab2", ("a", "b"), 300.0, 2000.0),
        ("bc", ("b", "c"), 400.0, 3000.0),
        ("cd", ("c", "d"), 500.0, 1000.0),
    ]:
        length = 85.0
        n.add(
            "Line",
            name,
            bus0=b0,
            bus1=b1,
            s_nom=s_nom,
            r=0.1,
            x=1.0,
            length=length,
            capital_cost=unit * length,
        )
    return n


def test_line_unit_cost_is_capacity_weighted_through_pypsa_clustering():
    from simplify_network import clustering_from_busmap

    n = _priced_network()
    unit_before = n.lines.capital_cost / n.lines.length

    # Collapse b and c together: 'bc' disappears inside the cluster, and the two
    # parallel a-b corridors merge into one.
    busmap = pd.Series({"a": "a", "b": "bc", "c": "bc", "d": "d"})
    clustered = clustering_from_busmap(n, busmap, line_length_factor=1.25).network

    unit_after = clustered.lines.capital_cost / clustered.lines.length
    merged = unit_after.index[clustered.lines.bus0.isin(["a", "bc"]) & clustered.lines.bus1.isin(["a", "bc"])]

    # ab (100 MW at 5000) and ab2 (300 MW at 2000) -> (5000 + 3*2000)/4 = 2750.
    assert float(unit_after.loc[merged].iloc[0]) == pytest.approx(2750.0)
    # cd is untouched by the busmap and must keep its own unit cost exactly.
    single = unit_after.index[clustered.lines.bus0.isin(["bc", "d"]) & clustered.lines.bus1.isin(["bc", "d"])]
    assert float(unit_after.loc[single].iloc[0]) == pytest.approx(float(unit_before.loc["cd"]))


def test_link_unit_cost_columns_survive_pypsa_clustering():
    from simplify_network import clustering_from_busmap

    n = _priced_network()
    n.add("Link", "dc_fwd", bus0="a", bus1="d", carrier="DC", p_nom=100)
    n.add("Link", "dc_rev", bus0="d", bus1="a", carrier="DC", p_nom=100)
    n.links[LINK_UNIT_COST_COL] = 40.0
    n.links[LINK_FIXED_COST_COL] = 18000.0
    recompute_link_transmission_costs(n)
    before = n.links.at["dc_fwd", "capital_cost"]

    busmap = pd.Series({"a": "a", "b": "bc", "c": "bc", "d": "d"})
    clustered = clustering_from_busmap(n, busmap, line_length_factor=1.25).network

    # pypsa keeps extra Link columns, which is what lets the cost be rebuilt rather
    # than re-derived from endpoint counties that no longer exist post-aggregation.
    assert LINK_UNIT_COST_COL in clustered.links.columns
    assert clustered.links[LINK_UNIT_COST_COL].notna().all()

    recompute_link_transmission_costs(clustered)
    # a and d were not merged, so the corridor length is unchanged.
    assert clustered.links.at["dc_fwd", "capital_cost"] == pytest.approx(before)
