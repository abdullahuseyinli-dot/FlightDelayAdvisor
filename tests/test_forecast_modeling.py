from __future__ import annotations

import numpy as np
import pandas as pd

from flightdelaybench.contracts import FORECAST24_DAILY_FEATURES, RECENT_OPERATIONAL_FEATURES
from flightdelaybench.forecast_features import DAILY_FORECAST_COLUMNS
from flightdelaybench.forecast_modeling import (
    attach_daily_forecasts,
    engineer_forecast_residual_features,
    engineer_weather_transfer_features,
    fit_forecast_residual_catboost,
)
from flightdelaybench.modeling import (
    CATEGORICAL_FEATURES,
    CLIMATOLOGY_FEATURES,
    PRIOR_RATE_FEATURES,
    PRIOR_SUPPORT_FEATURES,
    REGISTERED_MODEL_INPUTS,
)
from flightdelaybench.oracle import ORACLE_WEATHER_FEATURES


def _transfer_frame(rows: int = 3) -> pd.DataFrame:
    index = np.arange(rows)
    values: dict[str, object] = {}
    for column in REGISTERED_MODEL_INPUTS:
        if column in CATEGORICAL_FEATURES:
            values[column] = np.where(index % 2, "B", "A")
        elif column in PRIOR_RATE_FEATURES:
            values[column] = 0.1 + index * 0.01
        elif column in PRIOR_SUPPORT_FEATURES:
            values[column] = 20.0 + index
        elif column in CLIMATOLOGY_FEATURES and column.endswith("_missing"):
            values[column] = 0
        else:
            values[column] = 1.0 + index
    for feature_index, name in enumerate(RECENT_OPERATIONAL_FEATURES):
        values[name] = (
            0.1 + (feature_index % 5) * 0.01
            if "_rate_" in name
            else np.log1p(20 + feature_index + index)
        )
    for feature_index, name in enumerate(FORECAST24_DAILY_FEATURES):
        values[name] = (
            np.zeros(rows)
            if name.endswith("_missing")
            else np.ones(rows) * (2.0 + feature_index)
        )
    for feature_index, name in enumerate(ORACLE_WEATHER_FEATURES):
        values[name] = np.ones(rows) * (3.0 + feature_index)
    return pd.DataFrame(values)


def test_observed_and_forecast_transfer_share_identical_model_schema() -> None:
    frame = _transfer_frame()
    observed = engineer_weather_transfer_features(
        frame,
        source="observed_training",
        include_cross_direction=False,
    )
    forecast = engineer_weather_transfer_features(
        frame,
        source="forecast24_inference",
        include_cross_direction=False,
    )
    assert observed.columns.tolist() == forecast.columns.tolist()
    assert np.isfinite(forecast.select_dtypes(include=[np.number]).to_numpy()).all()


def test_daily_forecast_join_preserves_flight_order(tmp_path) -> None:
    forecast_dir = tmp_path / "forecast"
    forecast_dir.mkdir()
    rows: list[dict[str, object]] = []
    for airport, marker in (("JFK", 10.0), ("LAX", 20.0)):
        row: dict[str, object] = {
            "Airport": airport,
            "FlightDate": pd.Timestamp("2024-07-01"),
        }
        for column in DAILY_FORECAST_COLUMNS:
            row[column] = 0 if column.endswith("_missing") else marker
        rows.append(row)
    pd.DataFrame(rows).to_parquet(forecast_dir / "year=2024.parquet", index=False)
    flights = pd.DataFrame(
        {
            "sample_id": ["second", "first"],
            "Year": [2024, 2024],
            "FlightDate": pd.to_datetime(["2024-07-01", "2024-07-01"]),
            "Origin": ["LAX", "JFK"],
            "Dest": ["JFK", "LAX"],
        }
    )
    result = attach_daily_forecasts(flights, forecast_dir)
    assert result["sample_id"].tolist() == ["second", "first"]
    assert result["forecast24_origin_tavg"].tolist() == [20.0, 10.0]
    assert result["forecast24_dest_tavg"].tolist() == [10.0, 20.0]


def test_residual_weather_features_have_an_operational_only_ablation() -> None:
    frame = _transfer_frame(rows=12)
    operational = engineer_forecast_residual_features(
        frame,
        include_cross_direction=False,
        include_weather=False,
    )
    weather = engineer_forecast_residual_features(
        frame,
        include_cross_direction=False,
        include_weather=True,
    )

    assert not any(name.startswith("forecast24_") for name in operational)
    assert "forecast24_origin_temperature_anomaly" in weather
    assert "forecast24_endpoints_gust_max_max" in weather
    assert weather.shape[1] > operational.shape[1]


class _ConstantBaseline:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(frame), 0.2, dtype=np.float64)
        return np.column_stack([1.0 - positive, positive])


def test_residual_model_runs_as_an_additive_baseline_correction() -> None:
    frame = _transfer_frame(rows=24)
    labels = np.asarray([0, 1] * 12, dtype=np.int64)
    model = fit_forecast_residual_catboost(
        frame.iloc[:16].reset_index(drop=True),
        labels[:16],
        baseline_model=_ConstantBaseline(),
        task="delay",
        include_cross_direction=False,
        include_weather=True,
        params={"iterations": 3, "depth": 3, "task_type": "CPU"},
        validation_frame=frame.iloc[16:].reset_index(drop=True),
        validation_labels=labels[16:],
    )

    probabilities = model.predict_proba(frame.iloc[16:].reset_index(drop=True))
    assert probabilities.shape == (8, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
