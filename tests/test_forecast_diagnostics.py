from __future__ import annotations

import numpy as np
import pandas as pd

from flightdelaybench.forecast_diagnostics import (
    _group_score_sums,
    _reliability_bins,
    _weather_strata,
)


def test_weather_strata_are_mutually_exclusive_with_extreme_precedence() -> None:
    frame = pd.DataFrame(
        {
            "origin_forecast24_missing": [0, 0, 0, 1],
            "dest_forecast24_missing": [0, 0, 0, 0],
            "origin_forecast24_prcp_sum": [0.0, 12.0, 25.0, np.nan],
            "dest_forecast24_prcp_sum": [0.0, 0.0, 0.0, 0.0],
            "origin_forecast24_gust_max": [5.0, 5.0, 5.0, np.nan],
            "dest_forecast24_gust_max": [5.0, 5.0, 5.0, 5.0],
            "origin_forecast24_cape_max": [1.0, 1.0, 1.0, np.nan],
            "dest_forecast24_cape_max": [1.0, 1.0, 1.0, 1.0],
        }
    )
    thresholds = {
        "prcp_sum": {"q90": 10.0, "q99": 20.0},
        "gust_max": {"q90": 30.0, "q99": 50.0},
        "cape_max": {"q90": 100.0, "q99": 200.0},
    }
    assert _weather_strata(frame, thresholds).tolist() == [
        "typical",
        "adverse",
        "extreme",
        "missing",
    ]


def test_group_score_sums_preserve_counts_and_favour_better_forecast() -> None:
    frame = pd.DataFrame(
        {
            "Reporting_Airline": ["AA", "AA", "BB", "BB"],
            "label": [0, 1, 0, 1],
            "prob_baseline": [0.5, 0.5, 0.5, 0.5],
            "prob_forecast_residual": [0.1, 0.9, 0.1, 0.9],
        }
    )
    result = _group_score_sums(frame, "Reporting_Airline")
    assert result["n"].sum() == 4
    assert result["positives"].sum() == 2
    assert (result["forecast_log"] < result["baseline_log"]).all()


def test_reliability_bins_cover_every_row() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    bins = _reliability_bins(labels, probabilities, bins=2)
    assert sum(record["n"] for record in bins) == 4
    assert bins[0]["observed_rate"] == 0.0
    assert bins[1]["observed_rate"] == 1.0
