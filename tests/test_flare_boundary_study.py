from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flightdelaybench.flare_boundary_contracts import (
    BOUNDARY_OBSERVATION_FEATURES,
    BOUNDARY_RESIDUAL_FEATURES,
)
from flightdelaybench.flare_boundary_study import (
    BOUNDARY_ANY_RESIDUAL_COLUMN,
    BOUNDARY_GATING_FEATURE,
    ENSEMBLE_MEMBERS,
    _apply_boundary_gated_simplex,
    _apply_simplex,
    _boundary_signal_summary,
    _cluster_binary_rate_difference,
    _decision_metrics,
    _join_schedule_reference,
    _projected_task_frame,
    _select_accuracy_bias,
    _select_boundary_gated_simplex,
    _select_simplex,
)
from flightdelaybench.hashing import sha256_file


def test_boundary_simplex_prefers_the_perfect_member() -> None:
    labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    perfect = np.eye(3)[labels] * 0.98 + 0.02 / 3.0
    weak = np.full((len(labels), 3), 1.0 / 3.0)
    probabilities = {
        ENSEMBLE_MEMBERS[0]: weak,
        ENSEMBLE_MEMBERS[1]: perfect,
        ENSEMBLE_MEMBERS[2]: weak,
    }
    selection = _select_simplex(labels, probabilities)
    blended = _apply_simplex(probabilities, selection["weights"])
    assert selection["weights"][ENSEMBLE_MEMBERS[1]] > 0.99
    assert np.allclose(blended.sum(axis=1), 1.0)


def test_projected_task_frame_copies_only_model_columns() -> None:
    frame = pd.DataFrame(
        {
            "Cancelled": [0, 1, 0, 0],
            "delay_label_observed": [1, 0, 1, 1],
            "ArrDel15": [1.0, np.nan, 0.0, 1.0],
            "needed_a": [10.0, 20.0, 30.0, 40.0],
            "needed_b": [1.0, 2.0, 3.0, 4.0],
            **{f"unused_{index}": np.ones(4) for index in range(100)},
        }
    )
    projected, labels = _projected_task_frame(
        frame,
        "delay",
        ("needed_a", "needed_b"),
        eligibility_mask=np.array([True, True, False, True]),
    )
    assert projected.columns.tolist() == ["needed_a", "needed_b"]
    assert projected["needed_a"].tolist() == [10.0, 40.0]
    assert labels.tolist() == [1, 1]


def test_accuracy_bias_is_selected_only_from_supplied_labels() -> None:
    labels = np.array([0, 1, 1, 2], dtype=np.int64)
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.5, 0.4, 0.1],
            [0.5, 0.4, 0.1],
            [0.5, 0.1, 0.4],
        ]
    )
    selection = _select_accuracy_bias(labels, probabilities)
    metrics = _decision_metrics(
        labels,
        probabilities,
        log_bias=(
            selection["selected_delay_log_bias"],
            selection["selected_cancellation_log_bias"],
        ),
    )
    assert metrics["accuracy"] == 1.0


def test_boundary_gated_simplex_preserves_probability_simplex() -> None:
    labels = np.tile(np.array([0, 1, 2], dtype=np.int64), 20)
    weak = np.full((len(labels), 3), 1.0 / 3.0)
    strong = np.eye(3)[labels] * 0.9 + 0.1 / 3.0
    probabilities = {
        ENSEMBLE_MEMBERS[0]: weak,
        ENSEMBLE_MEMBERS[1]: strong,
        ENSEMBLE_MEMBERS[2]: weak,
    }
    gate = np.linspace(-1.0, 1.0, len(labels))
    selection = _select_boundary_gated_simplex(
        labels,
        probabilities,
        gate,
        cutpoint_values=gate,
        minimum_rows=3,
    )
    output = _apply_boundary_gated_simplex(probabilities, gate, selection)
    assert np.allclose(output.sum(axis=1), 1.0)
    assert (output.argmax(axis=1) == labels).all()


def test_boundary_signal_summary_counts_rows_not_feature_cells() -> None:
    values = {
        feature: np.zeros(3, dtype=np.float32) for feature in BOUNDARY_RESIDUAL_FEATURES
    }
    frame = pd.DataFrame(values)
    frame.loc[1, BOUNDARY_GATING_FEATURE] = np.float32(0.25)
    frame[BOUNDARY_OBSERVATION_FEATURES[0]] = np.array([0.0, 1.0, 0.0])
    frame[BOUNDARY_OBSERVATION_FEATURES[1]] = np.array([0.0, 0.0, 1.0])
    summary = _boundary_signal_summary(frame)
    assert summary["rows_with_any_nonzero_boundary_residual"] == 1
    assert summary["fraction_with_any_nonzero_boundary_residual"] == 1 / 3
    assert summary["rows_with_newly_observed_rotation_predecessor_state"] == 1
    assert summary["rows_with_lost_rotation_predecessor_state"] == 1


def test_boundary_signal_summary_accepts_compact_prediction_contract() -> None:
    frame = pd.DataFrame(
        {
            BOUNDARY_ANY_RESIDUAL_COLUMN: [0, 1, 1],
            BOUNDARY_OBSERVATION_FEATURES[0]: [0.0, 1.0, 0.0],
            BOUNDARY_OBSERVATION_FEATURES[1]: [0.0, 0.0, 1.0],
        }
    )
    summary = _boundary_signal_summary(frame)
    assert summary["rows_with_any_nonzero_boundary_residual"] == 2
    assert summary["rows_with_newly_observed_rotation_predecessor_state"] == 1
    assert summary["rows_with_lost_rotation_predecessor_state"] == 1


def test_schedule_baseline_reference_is_keyed_and_normalized(tmp_path: Path) -> None:
    path = tmp_path / "schedule.parquet"
    pd.DataFrame(
        {
            "sample_id": ["b", "a"],
            "prob_baseline_on_time": [0.6, 0.8],
            "prob_baseline_delayed": [0.3, 0.1],
            "prob_baseline_cancelled": [0.1, 0.1],
        }
    ).to_parquet(path, index=False)
    result = _join_schedule_reference(
        pd.DataFrame({"sample_id": ["a", "b"]}),
        {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)},
    )
    assert result["prob_schedule_baseline_on_time"].tolist() == pytest.approx([0.8, 0.6])
    probabilities = result[
        [
            "prob_schedule_baseline_on_time",
            "prob_schedule_baseline_delayed",
            "prob_schedule_baseline_cancelled",
        ]
    ].to_numpy()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_cluster_binary_rate_difference_bootstraps_whole_dates() -> None:
    outcomes = np.array([0, 1, 0, 1, 1, 1, 0, 0], dtype=np.int64)
    exposed = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.bool_)
    clusters = np.array(["d1", "d1", "d1", "d1", "d2", "d2", "d2", "d2"])
    result = _cluster_binary_rate_difference(
        outcomes,
        exposed,
        clusters,
        repetitions=200,
        seed=9,
    )
    assert result["clusters"] == 2
    assert result["inference_status"] == "ESTIMABLE"
    assert result["exposed_rows"] == 4
    assert result["unexposed_rows"] == 4
    assert result["rate_difference_exposed_minus_unexposed"] == pytest.approx(-0.5)


def test_cluster_binary_rate_difference_retains_saturated_nonestimable_contrast() -> None:
    outcomes = np.array([0, 1, 0, 1, 1, 0], dtype=np.int64)
    exposed = np.array([0, 1, 1, 1, 1, 1], dtype=np.bool_)
    clusters = np.array(["d1", "d1", "d2", "d2", "d3", "d3"])
    result = _cluster_binary_rate_difference(
        outcomes,
        exposed,
        clusters,
        repetitions=200,
        seed=11,
    )
    assert result["inference_status"] == (
        "NOT_ESTIMABLE_INSUFFICIENT_TEMPORAL_CONTRAST_SUPPORT"
    )
    assert result["cluster_bootstrap_lower"] is None
    assert result["cluster_bootstrap_upper"] is None
    assert result["unexposed_rows"] == 1
