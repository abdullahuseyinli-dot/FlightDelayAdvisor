from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flightdelaybench.flare_capacity_modeling import CapacityCatBoostModel
from flightdelaybench.flare_capacity_study import (
    AUGMENTED_CANDIDATES,
    BLEND_CANDIDATES,
    TASKS,
    _atomic_joblib,
    _frozen_capacity_implementation_files,
    _load_capacity_model,
    _load_protocol,
    _load_recovery_model_record,
    _normalize_persisted_joint,
    _verify_implementation_unchanged,
    apply_capacity_gated_simplex,
    blend_probabilities,
    capacity_regimes,
    select_capacity_gated_simplex,
    select_probability_simplex,
    select_purged_forward_calibration,
)
from flightdelaybench.flare_evaluation import joint_loss_rows
from flightdelaybench.hashing import canonical_json_sha256


def _synthetic_probabilities(labels: np.ndarray) -> dict[str, np.ndarray]:
    observed = np.eye(3, dtype=np.float64)[labels]
    good = 0.85 * observed + 0.15 / 3.0
    weak = 0.30 * observed + 0.70 / 3.0
    return {
        name: good.copy() if name == "hypergraph" else weak.copy()
        for name in BLEND_CANDIDATES
    }


def test_persisted_joint_normalization_repairs_only_bounded_float32_drift() -> None:
    persisted = np.asarray(
        [[0.1, 0.2, 0.7], [0.73, 0.26, 0.01]],
        dtype=np.float32,
    )

    normalized, audit = _normalize_persisted_joint(persisted, method="flare24")

    assert np.allclose(normalized.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)
    assert audit["maximum_absolute_row_sum_error_before_normalization"] <= 1e-6
    assert audit["rows_over_acceptance_bound"] == 0
    assert audit["maximum_absolute_probability_adjustment"] <= 1e-6


def test_persisted_joint_normalization_rejects_material_incoherence() -> None:
    with pytest.raises(ValueError, match="exceeds the float32 acceptance bound"):
        _normalize_persisted_joint(
            np.asarray([[0.1, 0.2, 0.6]], dtype=np.float32),
            method="flare24",
        )


def test_probability_simplex_prefers_informative_candidate() -> None:
    labels = np.tile(np.arange(3), 100)
    probabilities = _synthetic_probabilities(labels)
    selection = select_probability_simplex(labels, probabilities)
    assert selection["weights"]["hypergraph"] > 0.99
    blended = blend_probabilities(probabilities, selection["weights"])
    assert np.allclose(blended.sum(axis=1), 1.0)


def test_capacity_gated_simplex_can_learn_regime_specific_experts() -> None:
    labels = np.tile(np.arange(3), 200)
    observed = np.eye(3, dtype=np.float64)[labels]
    good = 0.85 * observed + 0.15 / 3.0
    poor = 0.05 * observed + 0.95 / 3.0
    probabilities = {
        name: np.full((len(labels), 3), 1.0 / 3.0, dtype=np.float64)
        for name in BLEND_CANDIDATES
    }
    probabilities["flare24"][:300] = good[:300]
    probabilities["flare24"][300:] = poor[300:]
    probabilities["hypergraph"][:300] = poor[:300]
    probabilities["hypergraph"][300:] = good[300:]
    gate = np.arange(len(labels), dtype=np.float64)
    selection = select_capacity_gated_simplex(
        labels,
        probabilities,
        gate,
        minimum_rows=50,
    )
    gated = apply_capacity_gated_simplex(probabilities, gate, selection)
    global_blend = blend_probabilities(
        probabilities,
        selection["global"]["weights"],
    )
    gated_loss, _ = joint_loss_rows(labels, gated)
    global_loss, _ = joint_loss_rows(labels, global_blend)
    assert gated_loss.mean() < global_loss.mean()
    assert np.allclose(gated.sum(axis=1), 1.0)


def test_capacity_regimes_preserve_missing_values() -> None:
    regimes = capacity_regimes([0.0, 1.0, 2.0, np.nan], [0.5, 1.5])
    assert regimes.tolist() == ["low", "elevated", "severe", "missing"]


def test_frozen_capacity_protocol_matches_executable_constants() -> None:
    protocol, record = _load_protocol(
        Path("configs/flare24_ccrth_v1.toml"),
        rows_per_train_month=125_000,
        validation_limit=250_000,
        bootstrap_repetitions=2_000,
        seed=20260903,
    )
    assert protocol["identity"]["method"] == "CC-RTH-v1"
    assert len(record["sha256"]) == 64


def test_capacity_method_lock_binds_transitive_scoring_sources() -> None:
    paths = _frozen_capacity_implementation_files()
    names = {path.name for path in paths}
    assert len(names) == len(paths)
    assert all(path.is_file() for path in paths)
    assert {
        "flare_capacity_study.py",
        "flare_capacity_modeling.py",
        "calibration.py",
        "bootstrap.py",
        "flare_evaluation.py",
        "flare_reconciliation.py",
        "hashing.py",
        "provenance.py",
    }.issubset(names)


def test_capacity_study_detects_source_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    started = {"git_head": "abc", "source_files": [{"path": "a.py", "sha256": "1"}]}
    monkeypatch.setattr(
        "flightdelaybench.flare_capacity_study.capture_provenance",
        lambda _paths: {
            "git_head": "abc",
            "source_files": [{"path": "a.py", "sha256": "2"}],
        },
    )
    with pytest.raises(RuntimeError, match="provenance changed before method lock"):
        _verify_implementation_unchanged(started, phase="method lock")


def test_capacity_model_is_checksum_reloaded_one_artifact_at_a_time(
    tmp_path: Path,
) -> None:
    model = CapacityCatBoostModel(
        estimator={"diagnostic": True},
        task="delay",
        flare_features=(),
        capacity_features=(),
    )
    record = _atomic_joblib(model, tmp_path / "model.joblib")
    loaded = _load_capacity_model(record)
    assert loaded.task == "delay"
    assert loaded.estimator == {"diagnostic": True}


def test_recovery_admits_only_complete_checksum_bound_2024_models(
    tmp_path: Path,
) -> None:
    protocol_sha = "a" * 64
    manifest_sha = "b" * 64
    artifacts = []
    for candidate in AUGMENTED_CANDIDATES:
        for task in TASKS:
            artifact = _atomic_joblib(
                {"candidate": candidate, "task": task},
                tmp_path / f"{candidate}_{task}.joblib",
            )
            artifacts.append(
                {
                    "candidate": candidate,
                    "task": task,
                    "early_stopping_best_iteration": 9,
                    "refit_iterations": 10,
                    **artifact,
                }
            )
    record_path = tmp_path / "recovery.json"
    payload = {
        "disposition": (
            "RETAINED_FAILED_RUN_2024_ONLY_MODELS_APPROVED_FOR_CHECKSUM_RECOVERY"
        ),
        "run_directory": "D:/retained-failed-run",
        "boundary_status": {
            "2025_outcomes_accessed": False,
            "2026_outcomes_accessed": False,
            "method_lock_written": False,
            "study_report_written": False,
        },
        "protocol": {"sha256": protocol_sha},
        "capacity_manifest": {"self_hash": manifest_sha},
        "completed_model_artifacts": artifacts,
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    models, provenance = _load_recovery_model_record(
        record_path,
        protocol_sha256=protocol_sha,
        capacity_manifest_self_hash=manifest_sha,
    )

    assert len(models) == 8
    assert provenance is not None
    assert provenance["model_count"] == 8
    assert provenance["2025_outcomes_accessed"] is False


def test_recovery_rejects_an_incomplete_model_set(tmp_path: Path) -> None:
    artifact = _atomic_joblib({}, tmp_path / "only_one.joblib")
    record_path = tmp_path / "incomplete.json"
    payload = {
        "disposition": (
            "RETAINED_FAILED_RUN_2024_ONLY_MODELS_APPROVED_FOR_CHECKSUM_RECOVERY"
        ),
        "boundary_status": {
            "2025_outcomes_accessed": False,
            "2026_outcomes_accessed": False,
            "method_lock_written": False,
            "study_report_written": False,
        },
        "protocol": {"sha256": "a" * 64},
        "capacity_manifest": {"self_hash": "b" * 64},
        "completed_model_artifacts": [
            {
                "candidate": AUGMENTED_CANDIDATES[0],
                "task": TASKS[0],
                "early_stopping_best_iteration": 9,
                "refit_iterations": 10,
                **artifact,
            }
        ],
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not contain all eight final models"):
        _load_recovery_model_record(
            record_path,
            protocol_sha256="a" * 64,
            capacity_manifest_self_hash="b" * 64,
        )


def test_forward_calibration_embargoes_each_boundary() -> None:
    dates = pd.date_range("2024-10-01", "2024-12-31", freq="D").repeat(4)
    labels = np.tile([0, 1, 0, 1], len(dates) // 4)
    frame = pd.DataFrame(
        {
            "FlightDate": dates,
            "Cancelled": labels,
            "delay_label_observed": 1,
            "ArrDel15": labels,
        }
    )
    raw = np.where(labels == 1, 0.7, 0.3)
    selected = select_purged_forward_calibration(
        frame,
        raw,
        task="cancellation",
        methods=("identity",),
    )
    scored_dates = pd.Series(dates[selected.crossfit_mask]).dt.normalize()
    assert not scored_dates.isin(
        pd.to_datetime(["2024-10-14", "2024-10-15"])
    ).any()
    purged = [
        fold["purged_dates"]
        for fold in selected.candidate_records[0]["folds"]
    ]
    assert purged == [
        ["2024-10-14", "2024-10-15"],
        ["2024-10-30", "2024-10-31"],
        ["2024-11-29", "2024-11-30"],
    ]
    assert selected.final_fit_through == "2024-12-31"
