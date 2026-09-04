from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flightdelaybench.flare_modeling import (
    select_usable_extra_features,
    task_frame,
)


def test_feature_selection_uses_availability_not_labels() -> None:
    base = pd.DataFrame(
        {
            "flare24_origin_weather_missing": [0.0, 1.0, 0.0, 1.0],
            "flare24_origin_departure_ceiling": [np.nan, np.nan, np.nan, np.nan],
            "ArrDel15": [0, 0, 1, 1],
        }
    )
    permuted = base.copy()
    permuted["ArrDel15"] = list(reversed(permuted["ArrDel15"].tolist()))
    candidates = (
        "flare24_origin_weather_missing",
        "flare24_origin_departure_ceiling",
    )
    assert select_usable_extra_features(base, candidates) == select_usable_extra_features(
        permuted, candidates
    )
    assert select_usable_extra_features(base, candidates) == (
        "flare24_origin_weather_missing",
    )


def test_hurdle_task_frame_excludes_cancelled_from_delay() -> None:
    frame = pd.DataFrame(
        {
            "Cancelled": [0, 1, 0],
            "delay_label_observed": [1, 0, 1],
            "ArrDel15": [1, np.nan, 0],
        }
    )
    delay_frame, delay_labels = task_frame(frame, "delay")
    assert len(delay_frame) == 2
    np.testing.assert_array_equal(delay_labels, [1, 0])
    cancel_frame, cancel_labels = task_frame(frame, "cancellation")
    assert len(cancel_frame) == 3
    np.testing.assert_array_equal(cancel_labels, [0, 1, 0])


def test_unknown_extra_feature_is_rejected() -> None:
    with pytest.raises(ValueError, match="unregistered"):
        select_usable_extra_features(pd.DataFrame({"invented": [1, 2]}), ("invented",))
