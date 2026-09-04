from __future__ import annotations

import numpy as np
import pandas as pd

from flightdelaybench.census_modeling import (
    CENSUS_BASE_CATEGORICAL_FEATURES,
    CENSUS_BASE_SCHEDULE_FEATURES,
    CENSUS_RICH_NUMERIC_FEATURES,
    engineer_census_features,
    fit_census_catboost,
)
from flightdelaybench.contracts import (
    CENSUS_FLIGHT_RECENT_FEATURES,
    CENSUS_GRAPH_MESSAGE_FEATURES,
    RECENT_OPERATIONAL_FEATURES,
    SCHEDULE_CONTEXT_FEATURES,
)


def _census_frame(rows: int = 24) -> pd.DataFrame:
    index = np.arange(rows)
    values: dict[str, object] = {}
    for name in CENSUS_BASE_CATEGORICAL_FEATURES:
        values[name] = np.where(index % 2, "B", "A")
    for offset, name in enumerate(CENSUS_BASE_SCHEDULE_FEATURES):
        values[name] = 1.0 + index + offset
    values["Year"] = np.full(rows, 2023)
    values["Month"] = 1 + index % 12
    values["DayOfWeek"] = 1 + index % 7
    values["DayOfYear"] = 1 + index
    values["DepHour"] = index % 24
    values["Distance"] = 250.0 + index * 10
    for offset, name in enumerate(CENSUS_RICH_NUMERIC_FEATURES):
        values[name] = 20.0 + index + offset
    values["ScheduledFlightId"] = np.where(index % 2, "A_1", "B_2")
    for offset, name in enumerate(RECENT_OPERATIONAL_FEATURES):
        values[name] = (
            0.05 + (index + offset) % 10 / 100
            if "_rate_" in name
            else np.log1p(100 + index + offset)
        )
    for offset, name in enumerate(CENSUS_FLIGHT_RECENT_FEATURES):
        values[name] = (
            0.08 + (index + offset) % 5 / 100
            if "_rate_" in name
            else np.log1p(10 + index + offset)
        )
    for offset, name in enumerate(SCHEDULE_CONTEXT_FEATURES):
        values[name] = 0.1 + index + offset
    for offset, name in enumerate(CENSUS_GRAPH_MESSAGE_FEATURES):
        values[name] = (
            0.05 + (index + offset) % 5 / 100
            if "partner_count" not in name
            else np.log1p(2 + index % 3)
        )
    return pd.DataFrame(values)


def test_census_feature_ablation_and_finite_engineering() -> None:
    frame = _census_frame()
    core = engineer_census_features(
        frame,
        include_cross_direction=False,
        include_rich_schedule=False,
        include_flight_history=False,
        include_schedule_context=False,
    )
    full = engineer_census_features(
        frame,
        include_cross_direction=True,
        include_rich_schedule=True,
        include_flight_history=True,
        include_schedule_context=True,
        include_graph_pressure=True,
    )

    assert "ScheduledFlightId" not in core
    assert "ScheduledFlightId" in full
    assert "scheduled_block_physics_residual" in full
    assert "recent_flight_delay_excess_global_7d" in full
    assert "census_schedule_global_surge_vs_7d" in full
    assert "graph_origin_delay_mean_minus_local_7d" in full
    assert full.shape[1] > core.shape[1]
    assert np.isfinite(full.select_dtypes(include=[np.number]).to_numpy()).all()


def test_census_catboost_cpu_smoke() -> None:
    frame = _census_frame()
    labels = np.asarray([0, 1] * 12, dtype=np.int64)
    model = fit_census_catboost(
        frame.iloc[:16].reset_index(drop=True),
        labels[:16],
        task="delay",
        params={"iterations": 3, "depth": 3, "task_type": "CPU"},
        include_cross_direction=False,
        include_rich_schedule=True,
        include_flight_history=True,
        include_schedule_context=True,
        validation_frame=frame.iloc[16:].reset_index(drop=True),
        validation_labels=labels[16:],
    )
    probabilities = model.predict_proba(frame.iloc[16:].reset_index(drop=True))
    assert probabilities.shape == (8, 2)
    assert np.isfinite(probabilities).all()
