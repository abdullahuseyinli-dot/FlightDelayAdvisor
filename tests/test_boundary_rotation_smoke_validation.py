from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flightdelaybench.boundary_rotation_smoke_validation import (
    NON_ROTATION_FEATURES,
    ROTATION_MESSAGE_FEATURES,
    compare_rotation_feature_frames,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    values: dict[str, object] = {"sample_id": ["a", "b", "c"]}
    values.update(
        {
            feature: np.full(3, index + 1, dtype=np.float32)
            for index, feature in enumerate(NON_ROTATION_FEATURES)
        }
    )
    values.update(
        {
            feature: np.array([0.0, 0.5, np.nan], dtype=np.float32)
            for feature in ROTATION_MESSAGE_FEATURES
        }
    )
    reference = pd.DataFrame(values)
    candidate = reference.copy()
    candidate.loc[1, ROTATION_MESSAGE_FEATURES[0]] = np.float32(0.75)
    candidate.loc[2, ROTATION_MESSAGE_FEATURES[-1]] = np.float32(0.2)
    return reference, candidate


def test_rotation_smoke_comparison_isolates_material_changes() -> None:
    reference, candidate = _frames()
    result = compare_rotation_feature_frames(reference, candidate)
    assert result["target_id_set_match"] is True
    assert result["non_rotation_feature_equality"] is True
    assert result["any_rotation_feature_changed_rows"] == 2
    assert result["any_rotation_feature_changed_row_fraction"] == pytest.approx(2 / 3)


def test_rotation_smoke_comparison_rejects_non_rotation_drift() -> None:
    reference, candidate = _frames()
    candidate.loc[0, NON_ROTATION_FEATURES[0]] += np.float32(1.0)
    with pytest.raises(ValueError, match="non-rotation"):
        compare_rotation_feature_frames(reference, candidate)
