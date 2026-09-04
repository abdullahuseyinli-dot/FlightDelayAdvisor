from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flightdelaybench.flare_boundary_contracts import (
    BOUNDARY_FULL_BY_SOURCE,
    BOUNDARY_FULL_FEATURES,
    BOUNDARY_OBSERVATION_FEATURES,
    BOUNDARY_RESIDUAL_BY_SOURCE,
)
from flightdelaybench.flare_boundary_modeling import (
    attach_boundary_views,
    boundary_model_input_columns,
)
from flightdelaybench.flare_capacity_contracts import (
    CAPACITY_ALL_FEATURES,
    CAPACITY_STATIC_FEATURES,
)
from flightdelaybench.flare_modeling import FLARE24_BASE_EXTRA_FEATURES


def _capacity_frame() -> pd.DataFrame:
    values = {feature: [1.0, 2.0] for feature in CAPACITY_ALL_FEATURES}
    values["ccrth_rotation_predecessor_capacity_shadow_price"] = [np.nan, 2.0]
    return pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "Year": [2024, 2024],
            "Month": [1, 1],
            **values,
        }
    )


def test_boundary_views_are_aligned_and_expose_new_rotation_state(tmp_path: Path) -> None:
    frame = _capacity_frame()
    partition = frame.loc[:, ["sample_id", *CAPACITY_ALL_FEATURES]].copy()
    source = "ccrth_origin_movement_demand_60m_log1p"
    partition[source] += 0.5
    partition["ccrth_rotation_predecessor_capacity_shadow_price"] = [0.7, 2.0]
    path = tmp_path / "year=2024" / "month=01.parquet"
    path.parent.mkdir(parents=True)
    partition.to_parquet(path, index=False)

    result = attach_boundary_views(frame, boundary_feature_dir=tmp_path)
    assert np.allclose(result[BOUNDARY_FULL_BY_SOURCE[source]], [1.5, 2.5])
    assert np.allclose(result[BOUNDARY_RESIDUAL_BY_SOURCE[source]], [0.5, 0.5])
    rotation_source = "ccrth_rotation_predecessor_capacity_shadow_price"
    assert np.allclose(result[BOUNDARY_RESIDUAL_BY_SOURCE[rotation_source]], [0.7, 0.0])
    assert result[BOUNDARY_OBSERVATION_FEATURES[0]].tolist() == [1.0, 0.0]
    assert result[BOUNDARY_OBSERVATION_FEATURES[1]].tolist() == [0.0, 0.0]


def test_boundary_views_reject_static_drift(tmp_path: Path) -> None:
    frame = _capacity_frame()
    partition = frame.loc[:, ["sample_id", *CAPACITY_ALL_FEATURES]].copy()
    partition[CAPACITY_STATIC_FEATURES[0]] += 1.0
    path = tmp_path / "year=2024" / "month=01.parquet"
    path.parent.mkdir(parents=True)
    partition.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="static feature"):
        attach_boundary_views(frame, boundary_feature_dir=tmp_path)


def test_boundary_model_input_projection_is_unique_and_complete() -> None:
    flare = (FLARE24_BASE_EXTRA_FEATURES[0],)
    capacity = (CAPACITY_STATIC_FEATURES[0],)
    boundary = (BOUNDARY_FULL_FEATURES[0],)
    columns = boundary_model_input_columns(
        flare_features=flare,
        capacity_features=capacity,
        boundary_features=boundary,
    )
    assert len(columns) == len(set(columns))
    assert set((*flare, *capacity, *boundary)).issubset(columns)
    assert "Origin" in columns
    assert "ScheduledFlightId" in columns
