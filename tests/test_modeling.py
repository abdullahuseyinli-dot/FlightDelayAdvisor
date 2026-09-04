from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flightdelaybench.modeling import (
    CATEGORICAL_FEATURES,
    CLIMATOLOGY_FEATURES,
    MODEL_FEATURES,
    PRIOR_RATE_FEATURES,
    PRIOR_SUPPORT_FEATURES,
    REGISTERED_MODEL_INPUTS,
    CategoryCodec,
    engineer_model_features,
    task_view,
    temporal_sample_weights,
)


def _model_frame() -> pd.DataFrame:
    rows = 3
    values: dict[str, object] = {}
    for column in REGISTERED_MODEL_INPUTS:
        if column in CATEGORICAL_FEATURES:
            values[column] = ["A", "B", "A"]
        elif column in PRIOR_RATE_FEATURES:
            values[column] = [0.1, 0.2, 0.3]
        elif column in PRIOR_SUPPORT_FEATURES:
            values[column] = [10.0, 20.0, 30.0]
        elif column in CLIMATOLOGY_FEATURES and column.endswith("_missing"):
            values[column] = [0, 0, 1]
        else:
            values[column] = [1.0, 2.0, 3.0]
    frame = pd.DataFrame(values)
    frame["sample_id"] = ["a", "b", "c"]
    frame["FlightDate"] = pd.date_range("2020-01-01", periods=rows)
    frame["ArrDel15"] = [0.0, 1.0, np.nan]
    frame["Cancelled"] = [0, 0, 1]
    frame["delay_label_observed"] = [1, 1, 0]
    frame["joint_label_observed"] = [1, 1, 1]
    frame["disruption_state"] = [0, 1, 2]
    return frame


def test_feature_algebra_is_complete_and_finite() -> None:
    result = engineer_model_features(_model_frame())
    assert tuple(result.columns) == MODEL_FEATURES
    assert np.isfinite(result.select_dtypes(include=[np.number]).to_numpy()).all()
    assert result["route_delay_excess"].eq(0).all()


def test_category_codec_maps_unseen_level_to_negative_one() -> None:
    frame = _model_frame()
    codec = CategoryCodec.fit(frame.iloc[:2])
    transformed = codec.transform(frame)
    assert transformed.loc[2, "Reporting_Airline"] == 0
    unseen = frame.iloc[[0]].copy()
    unseen["Reporting_Airline"] = "UNSEEN"
    assert codec.transform(unseen).iloc[0]["Reporting_Airline"] == -1


def test_task_views_keep_correct_populations() -> None:
    frame = _model_frame()
    delay_frame, delay = task_view(frame, "delay")
    cancel_frame, cancel = task_view(frame, "cancellation")
    joint_frame, joint = task_view(frame, "joint")
    assert len(delay_frame) == 2 and delay.tolist() == [0, 1]
    assert len(cancel_frame) == 3 and cancel.tolist() == [0, 0, 1]
    assert len(joint_frame) == 3 and joint.tolist() == [0, 1, 2]


def test_temporal_weights_are_normalized_and_decay_by_age() -> None:
    weights = temporal_sample_weights(
        pd.Series([2023, 2021]), prediction_year=2024, half_life_years=2.0
    )
    assert weights is not None
    assert weights.mean() == pytest.approx(1.0)
    assert weights[0] / weights[1] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="strictly precede"):
        temporal_sample_weights(pd.Series([2024]), prediction_year=2024, half_life_years=2.0)
