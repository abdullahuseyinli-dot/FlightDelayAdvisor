from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flightdelaybench.flare_capacity_contracts import CAPACITY_ALL_FEATURES
from flightdelaybench.flare_capacity_modeling import (
    CAPACITY_CANDIDATE_FEATURES,
    attach_capacity_feature_partitions,
    capacity_candidate_profile,
    select_usable_capacity_features,
)


def test_capacity_candidates_are_strictly_nested() -> None:
    profile = capacity_candidate_profile()
    assert profile["nested"] is True
    assert tuple(CAPACITY_CANDIDATE_FEATURES) == (
        "flare24",
        "raw_demand",
        "normalized_capacity",
        "queue_shadow",
        "hypergraph",
    )


def test_capacity_partition_join_preserves_order(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    partition_path = feature_dir / "year=2024" / "month=01.parquet"
    partition_path.parent.mkdir(parents=True)
    features = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            **{
                feature: np.asarray([index, index + 1], dtype=np.float32)
                for index, feature in enumerate(CAPACITY_ALL_FEATURES)
            },
        }
    )
    features.to_parquet(partition_path, index=False)
    frame = pd.DataFrame(
        {
            "sample_id": ["b", "a"],
            "Year": [2024, 2024],
            "Month": [1, 1],
        }
    )
    result = attach_capacity_feature_partitions(frame, feature_dir=feature_dir)
    assert result["sample_id"].tolist() == ["b", "a"]
    assert result[CAPACITY_ALL_FEATURES[0]].tolist() == [1.0, 0.0]


def test_capacity_feature_selection_uses_covariates_only() -> None:
    candidates = CAPACITY_ALL_FEATURES[:3]
    frame = pd.DataFrame(
        {
            candidates[0]: [0.0, 1.0, 2.0],
            candidates[1]: [1.0, 1.0, 1.0],
            candidates[2]: [np.nan, np.nan, np.nan],
            "Cancelled": [0, 1, 0],
        }
    )
    assert select_usable_capacity_features(frame, candidates) == (candidates[0],)
