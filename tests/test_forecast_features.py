from __future__ import annotations

import json

import numpy as np
import pandas as pd

from flightdelaybench.forecast_acquisition import HOURLY_VARIABLES
from flightdelaybench.forecast_features import _aggregate_airport_year


def test_daily_forecast_aggregation_preserves_missing_days(tmp_path) -> None:
    times = pd.date_range("2024-07-01", periods=48, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    hourly: dict[str, list[float] | list[str]] = {"time": times}
    for variable in HOURLY_VARIABLES:
        hourly[variable] = [1.0] * 48
    precipitation = "precipitation_previous_day1"
    hourly[precipitation] = [0.5] * 24 + [1.0] * 24
    path = tmp_path / "response.json"
    path.write_text(json.dumps({"hourly": hourly}), encoding="utf-8")

    result = _aggregate_airport_year(path, "JFK", 2024)
    assert len(result) == 366
    first = result.loc[result["FlightDate"].eq(pd.Timestamp("2024-07-01"))].iloc[0]
    assert np.isclose(first["forecast24_prcp_sum"], 12.0)
    assert first["forecast24_missing"] == 0
    missing = result.loc[result["FlightDate"].eq(pd.Timestamp("2024-01-01"))].iloc[0]
    assert missing["forecast24_missing"] == 1
    assert missing["forecast24_min_variable_coverage"] == 0.0
