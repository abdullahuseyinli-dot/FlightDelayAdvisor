from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from flightdelaybench.flare_capacity_modeling import CAPACITY_CANDIDATE_FEATURES
from flightdelaybench.flare_capacity_study import CAPACITY_CANDIDATES
from flightdelaybench.flare_capacity_validation import (
    _validate_nested_feature_sets,
    _validate_prediction_probabilities,
    _validate_probability_normalization_audit,
)


def test_feature_set_validation_ignores_canonical_json_key_order() -> None:
    canonical_key_order = sorted(CAPACITY_CANDIDATES)
    feature_sets = {
        candidate: list(CAPACITY_CANDIDATE_FEATURES[candidate])
        for candidate in canonical_key_order
    }

    validated = _validate_nested_feature_sets(feature_sets)

    assert tuple(validated) == CAPACITY_CANDIDATES


def _normalization_record(method: str, rows: int) -> dict[str, object]:
    return {
        "method": method,
        "rows": rows,
        "acceptance_bound": 1e-6,
        "maximum_absolute_row_sum_error_before_normalization": 4.5e-8,
        "rows_over_acceptance_bound": 0,
        "maximum_absolute_probability_adjustment": 3.4e-8,
        "normalization": "divide each accepted three-state row by its float64 row sum",
    }


def test_probability_normalization_audit_covers_all_partitions_and_methods() -> None:
    selection = _normalization_record("flare24", 100)
    references = [_normalization_record("flare24", 10) for _ in range(12)]
    methods = [
        *CAPACITY_CANDIDATES,
        "global_simplex",
        "capacity_gated_simplex",
    ]
    assembled = [_normalization_record(method, 120) for method in methods]
    report = {
        "baseline_selection": {"probability_normalization_audit": selection},
        "baseline_audit": {"probability_normalization_audits": references},
        "probability_normalization_audit": {
            "selection_reference": selection,
            "retrospective_reference_partitions": references,
            "retrospective_assembled_methods": assembled,
        },
    }

    result = _validate_probability_normalization_audit(report)

    assert result["retrospective_rows"] == 120
    assert result["assembled_methods"] == 7


def test_joint_prediction_probability_validation(tmp_path: Path) -> None:
    path = tmp_path / "retrospective_2025_01.parquet"
    pd.DataFrame(
        {
            "prob_flare24_on_time": [0.7, 0.4],
            "prob_flare24_delayed": [0.2, 0.5],
            "prob_flare24_cancelled": [0.1, 0.1],
        }
    ).to_parquet(path, index=False)
    audit = _validate_prediction_probabilities(path)
    assert audit == {"rows": 2, "methods": ["flare24"]}


def test_joint_prediction_probability_validation_rejects_bad_simplex(
    tmp_path: Path,
) -> None:
    path = tmp_path / "retrospective_2025_01.parquet"
    pd.DataFrame(
        {
            "prob_flare24_on_time": [0.7],
            "prob_flare24_delayed": [0.7],
            "prob_flare24_cancelled": [0.1],
        }
    ).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="simplex"):
        _validate_prediction_probabilities(path)
