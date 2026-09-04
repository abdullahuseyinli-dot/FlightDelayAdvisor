from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flightdelaybench.flare_capacity_factorized import (
    _frozen_factorized_implementation_files,
    _load_protocol,
    apply_task_factorized_capacity_gated,
    apply_task_factorized_global,
    component_pair_grid,
    select_binary_probability_simplex,
    select_task_factorized_capacity_gated,
    select_task_factorized_global,
)
from flightdelaybench.flare_capacity_factorized_validation import (
    _assert_factorized_selections,
)
from flightdelaybench.flare_capacity_study import BLEND_CANDIDATES, GATING_FEATURE
from flightdelaybench.flare_evaluation import joint_loss_rows
from flightdelaybench.flare_reconciliation import hurdle_joint_probabilities


def _frame(rows: int = 600) -> pd.DataFrame:
    index = np.arange(rows)
    cancelled = (index % 7 == 0).astype(int)
    delayed = (index % 3 == 0).astype(int)
    disruption = np.where(cancelled == 1, 2, delayed)
    return pd.DataFrame(
        {
            "FlightDate": pd.Timestamp("2024-10-01")
            + pd.to_timedelta(index % 60, unit="D"),
            "Cancelled": cancelled,
            "ArrDel15": delayed,
            "delay_label_observed": (cancelled == 0).astype(int),
            "joint_label_observed": 1,
            "disruption_state": disruption,
            GATING_FEATURE: index.astype(float),
        }
    )


def _binary_predictions(labels: np.ndarray, *, accuracy: float) -> np.ndarray:
    return np.where(labels == 1, accuracy, 1.0 - accuracy).astype(np.float64)


def _factorized_experts(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    cancelled = frame["Cancelled"].to_numpy(dtype=int)
    delayed = frame["ArrDel15"].to_numpy(dtype=int)
    weak_cancel = _binary_predictions(cancelled, accuracy=0.58)
    weak_delay = _binary_predictions(delayed, accuracy=0.58)
    result = {
        name: hurdle_joint_probabilities(weak_cancel, weak_delay)
        for name in BLEND_CANDIDATES
    }
    result["raw_demand"] = hurdle_joint_probabilities(
        _binary_predictions(cancelled, accuracy=0.92), weak_delay
    )
    result["normalized_capacity"] = hurdle_joint_probabilities(
        weak_cancel, _binary_predictions(delayed, accuracy=0.90)
    )
    return result


def test_binary_simplex_selects_the_informative_component() -> None:
    labels = np.tile([0, 1], 300)
    weak = _binary_predictions(labels, accuracy=0.55)
    strong = _binary_predictions(labels, accuracy=0.90)
    probabilities = {name: weak.copy() for name in BLEND_CANDIDATES}
    probabilities["queue_shadow"] = strong

    selection = select_binary_probability_simplex(labels, probabilities)

    assert selection["weights"]["queue_shadow"] > 0.99
    assert selection["binary_log_loss"] < 0.2


def test_global_factorization_uses_different_task_experts() -> None:
    frame = _frame()
    probabilities = _factorized_experts(frame)

    selection = select_task_factorized_global(frame, probabilities)
    factorized = apply_task_factorized_global(probabilities, selection)

    assert selection["components"]["cancellation"]["weights"]["raw_demand"] > 0.99
    assert (
        selection["components"]["delay_given_operated"]["weights"][
            "normalized_capacity"
        ]
        > 0.99
    )
    assert np.allclose(factorized.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)


def test_capacity_gated_factorization_learns_regime_specific_experts() -> None:
    frame = _frame(900)
    cancelled = frame["Cancelled"].to_numpy(dtype=int)
    delayed = frame["ArrDel15"].to_numpy(dtype=int)
    weak_cancel = _binary_predictions(cancelled, accuracy=0.52)
    weak_delay = _binary_predictions(delayed, accuracy=0.52)
    probabilities = {
        name: hurdle_joint_probabilities(weak_cancel, weak_delay)
        for name in BLEND_CANDIDATES
    }
    for positions, expert in (
        (slice(0, 300), "flare24"),
        (slice(300, 600), "raw_demand"),
        (slice(600, 900), "hypergraph"),
    ):
        joint = probabilities[expert].copy()
        joint[positions] = hurdle_joint_probabilities(
            _binary_predictions(cancelled[positions], accuracy=0.92),
            _binary_predictions(delayed[positions], accuracy=0.90),
        )
        probabilities[expert] = joint
    global_selection = select_task_factorized_global(frame, probabilities)
    gated_selection = select_task_factorized_capacity_gated(
        frame,
        probabilities,
        cutpoints=[299.5, 599.5],
        minimum_rows=50,
        global_selection=global_selection,
    )

    global_joint = apply_task_factorized_global(probabilities, global_selection)
    gated_joint = apply_task_factorized_capacity_gated(
        probabilities,
        frame[GATING_FEATURE],
        gated_selection,
    )
    labels = frame["disruption_state"].to_numpy(dtype=int)
    global_loss, _ = joint_loss_rows(labels, global_joint)
    gated_loss, _ = joint_loss_rows(labels, gated_joint)

    assert gated_loss.mean() < global_loss.mean()
    assert np.allclose(gated_joint.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)


def test_component_grid_is_complete_and_finds_cross_task_pair() -> None:
    frame = _frame()
    records = component_pair_grid(frame, _factorized_experts(frame))

    assert len(records) == 25
    assert len({record["method"] for record in records}) == 25
    winner = min(records, key=lambda record: record["joint_log_loss"])
    assert winner["cancellation_source"] == "raw_demand"
    assert winner["delay_given_operated_source"] == "normalized_capacity"


def test_factorized_protocol_matches_executable_contract() -> None:
    protocol, record = _load_protocol(
        Path("configs/flare24_ccrth_factorized_v1.toml"),
        repetitions=2_000,
        seed=20260904,
    )

    assert protocol["identity"]["method"] == "TF-CC-RTH-v1"
    assert len(record["sha256"]) == 64


def test_factorized_lock_binds_all_scoring_sources() -> None:
    paths = _frozen_factorized_implementation_files()
    names = {path.name for path in paths}

    assert len(names) == len(paths)
    assert all(path.is_file() for path in paths)
    assert {
        "flare_capacity_factorized.py",
        "flare_capacity_study.py",
        "flare_evaluation.py",
        "flare_reconciliation.py",
        "bootstrap.py",
        "metrics.py",
        "hashing.py",
        "provenance.py",
    }.issubset(names)


def test_factorized_validator_ignores_canonical_json_mapping_order() -> None:
    weights = {name: float(name == "flare24") for name in BLEND_CANDIDATES}
    task_selection = {"weights": weights, "rows": 100, "fallback_to_global": False}
    global_selection = {
        "components": {
            "cancellation": task_selection,
            "delay_given_operated": task_selection,
        }
    }
    canonical_order_regimes = {
        name: {
            "components": {
                "cancellation": task_selection,
                "delay_given_operated": task_selection,
            }
        }
        for name in sorted(("low", "elevated", "severe", "missing"))
    }

    _assert_factorized_selections(
        {
            "gating_feature": GATING_FEATURE,
            "global": global_selection,
            "regimes": canonical_order_regimes,
        },
        gated=True,
    )
