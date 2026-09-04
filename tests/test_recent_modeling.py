from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flightdelaybench.contracts import (
    RECENT_OPERATIONAL_FEATURES,
    AvailabilityHorizon,
    validate_predictors,
)
from flightdelaybench.modeling import (
    CATEGORICAL_FEATURES,
    CLIMATOLOGY_FEATURES,
    PRIOR_RATE_FEATURES,
    PRIOR_SUPPORT_FEATURES,
    REGISTERED_MODEL_INPUTS,
)
from flightdelaybench.recent_modeling import engineer_recent_model_features


def _recent_frame(rows: int = 8) -> pd.DataFrame:
    index = np.arange(rows)
    values: dict[str, object] = {}
    for column in REGISTERED_MODEL_INPUTS:
        if column in CATEGORICAL_FEATURES:
            values[column] = np.where(index % 2, "B", "A")
        elif column in PRIOR_RATE_FEATURES:
            values[column] = 0.1 + (index % 3) * 0.03
        elif column in PRIOR_SUPPORT_FEATURES:
            values[column] = 10.0 + index
        elif column in CLIMATOLOGY_FEATURES and column.endswith("_missing"):
            values[column] = index % 2
        else:
            values[column] = 1.0 + index
    frame = pd.DataFrame(values)
    for feature_index, name in enumerate(RECENT_OPERATIONAL_FEATURES):
        if "_rate_" in name:
            frame[name] = 0.05 + 0.01 * ((index + feature_index) % 10)
        else:
            frame[name] = np.log1p(10 + index + feature_index)
    return frame


def test_recent_features_have_later_horizon_and_finite_engineering() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        validate_predictors(
            [RECENT_OPERATIONAL_FEATURES[0]],
            AvailabilityHorizon.SCHEDULE_CLIMATOLOGY,
        )
    validate_predictors(
        RECENT_OPERATIONAL_FEATURES,
        AvailabilityHorizon.FORECAST_24H,
    )
    frame = _recent_frame()
    direct = engineer_recent_model_features(frame, include_cross_direction=False)
    network = engineer_recent_model_features(frame, include_cross_direction=True)
    assert "recent_route_delay_momentum_7v90" in direct
    assert "recent_origin_flow_pressure_delay_7d" not in direct
    assert "recent_origin_flow_pressure_delay_7d" in network
    assert network.shape[1] > direct.shape[1]
    assert np.isfinite(network.select_dtypes(include=[np.number]).to_numpy()).all()
