"""
Test the cap of build_dynamic_line_rating.

Dynamic ratings are capped at DLR_CAP on each base line before the s_nom-weighted
aggregation onto the model branches (``trunk(rel, 1.3)`` of the DLR calculation
scripts, per physical line), not on the aggregated branch rating; without a cap the
ratings are aggregated as read.
"""

import os
import sys

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_dynamic_line_rating import DLR_CAP, aggregate_dynamic_line_rating
from constants import HOURS_PER_YEAR


@pytest.fixture
def setup(tmp_path):
    base = pypsa.Network()
    base.add("Bus", ["a", "b", "c"])
    # two parallel lines a-b merge into one branch; b-c stays one line
    base.add("Line", ["0", "1", "2"], bus0=["a", "a", "b"], bus1=["b", "b", "c"], s_nom=[100, 300, 100], x=0.1)
    model = pypsa.Network()
    model.add("Bus", ["a", "b", "c"])
    model.add("Line", ["ab", "bc"], bus0=["a", "b"], bus1=["b", "c"], s_nom=400, x=0.1)

    busmaps = []
    for name in ["busmap_b", "busmap_s", "busmap_c"]:
        fn = tmp_path / f"{name}.csv"
        pd.Series(["a", "b", "c"], index=pd.Index(["a", "b", "c"], name="Bus"), name="busmap").to_csv(fn)
        busmaps.append(str(fn))

    hours = pd.date_range("2013-01-01", periods=HOURS_PER_YEAR, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    ratings = pd.DataFrame(
        {
            "0": rng.uniform(0.8, 2.0, HOURS_PER_YEAR).round(3),
            "1": rng.uniform(0.8, 1.2, HOURS_PER_YEAR).round(3),
            "2": rng.uniform(0.8, 2.0, HOURS_PER_YEAR).round(3),
        },
        index=pd.Index(hours.astype(str), name="hour"),
    )
    csv = tmp_path / "dlr_2013.csv"
    ratings.to_csv(csv)
    return str(csv), base, busmaps, model, ratings


def test_cap_applies_before_aggregation(setup):
    csv, base, busmaps, model, ratings = setup
    capped = aggregate_dynamic_line_rating(csv, base, busmaps, model, cap=DLR_CAP)
    uncapped = aggregate_dynamic_line_rating(csv, base, busmaps, model)

    r0, r1, r2 = (ratings[c].to_numpy(np.float32) for c in ["0", "1", "2"])
    ab = (100 * r0 + 300 * r1) / 400
    np.testing.assert_allclose(uncapped["ab"], ab, atol=6e-5)
    np.testing.assert_allclose(uncapped["bc"], r2, atol=6e-5)
    ab_capped = (100 * np.minimum(r0, DLR_CAP) + 300 * np.minimum(r1, DLR_CAP)) / 400
    np.testing.assert_allclose(capped["ab"], ab_capped, atol=6e-5)
    np.testing.assert_allclose(capped["bc"], np.minimum(r2, DLR_CAP), atol=6e-5)

    assert DLR_CAP == 1.3
    assert capped.to_numpy().max() <= DLR_CAP
    assert uncapped.to_numpy().max() > DLR_CAP
    # capping the aggregated rating instead would have kept "ab" higher wherever line "0"
    # exceeds the cap while the weighted mean stays below it
    assert (np.minimum(ab, DLR_CAP) > capped["ab"].to_numpy() + 1e-3).any()


def test_row_numbers_instead_of_timestamps(setup, tmp_path):
    """slr_2012.csv numbers its rows 0 ... 8759 under an empty header instead of naming the hours."""
    csv, base, busmaps, model, ratings = setup
    numbered = tmp_path / "slr_2013.csv"
    ratings.reset_index(drop=True).to_csv(numbered)  # header ",0,1,2", rows "0,...", "1,..."
    assert open(numbered).readline().startswith(",0,")

    by_number = aggregate_dynamic_line_rating(str(numbered), base, busmaps, model, year=2013)
    by_time = aggregate_dynamic_line_rating(csv, base, busmaps, model, year=2013)
    pd.testing.assert_frame_equal(by_number, by_time)

    with pytest.raises(ValueError, match="weather year is needed"):
        aggregate_dynamic_line_rating(str(numbered), base, busmaps, model)
    with pytest.raises(ValueError, match="not in the weather year 2012"):
        aggregate_dynamic_line_rating(csv, base, busmaps, model, year=2012)
