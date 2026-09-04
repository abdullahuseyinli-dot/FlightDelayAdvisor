"""Recover a completed BC-POT-R study after a reporting-only failure.

The recovery path never refits a model and never regenerates a flight prediction.  It
revalidates the original method lock, models, calibrators, Q4 selection predictions,
and twelve 2025 retrospective prediction artifacts; reconstructs all selections and
statistics from those immutable artifacts; and records the reporting-only change.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .flare_boundary_contracts import (
    BOUNDARY_FULL_FEATURES,
    BOUNDARY_OBSERVATION_FEATURES,
    BOUNDARY_RESIDUAL_FEATURES,
)
from .flare_boundary_modeling import BoundaryCatBoostModel
from .flare_boundary_study import (
    BOUNDARY_GATING_FEATURE,
    BREAKTHROUGH_MINIMUM,
    BREAKTHROUGH_STRETCH,
    CANDIDATES,
    ENSEMBLE_MEMBERS,
    PROBABILITY_STATES,
    SELECTION_MONTHS,
    _accuracy_comparisons,
    _airport_scores,
    _apply_boundary_gated_simplex,
    _apply_simplex,
    _artifact_record,
    _boundary_signal_outcome_analysis,
    _boundary_signal_summary,
    _decision_metrics,
    _implementation_files,
    _load_protocol,
    _monthly_scores,
    _probability_frame,
    _residual_regime_scores,
    _select_accuracy_bias,
    _select_boundary_gated_simplex,
    _select_simplex,
    _self_hashed,
    _verified_record,
)
from .flare_capacity_contracts import CAPACITY_ALL_FEATURES, CAPACITY_STATIC_FEATURES
from .flare_capacity_study import (
    PRIMARY_AUDIT_END,
    PRIMARY_AUDIT_START,
    select_purged_forward_calibration,
)
from .flare_evaluation import evaluate_joint_probabilities, joint_loss_rows
from .flare_reconciliation import hurdle_joint_probabilities
from .flare_study import TASKS
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

RECOVERY_STATUS = "RECOVERED_FROM_V6_REPORTING_ONLY_FAILURE_NO_RETRAINING"
STUDY_STATUS = "COMPLETE_BCPOTR_2024_SELECTION_2025_RETROSPECTIVE_REDEVELOPMENT"
METHOD_NAMES = (
    "schedule_baseline",
    "flare24",
    "capacity_gated_simplex",
    "meta_current",
    *CANDIDATES,
    "boundary_ensemble",
    "boundary_gated_ensemble",
)
PROBABILITY_REPRODUCTION_ATOL = 2e-6


def _recovery_implementation_files() -> tuple[Path, ...]:
    return (Path(__file__), *_implementation_files())


def _assert_source_unchanged(started: dict[str, Any]) -> dict[str, Any]:
    current = capture_provenance(_recovery_implementation_files())
    if current.get("git_head") != started.get("git_head") or current.get(
        "source_files"
    ) != started.get("source_files"):
        raise RuntimeError("BC-POT-R recovery source changed during execution")
    return current


def _verify_self_hashed_record(
    record: dict[str, Any],
    *,
    role: str,
) -> tuple[dict[str, Any], Path]:
    path = _verified_record(record, role=role)
    payload, hash_key = _self_hashed(path)
    if record.get("self_hash_key") not in (None, hash_key) or record.get(
        "self_hash"
    ) != payload[hash_key]:
        raise ValueError(f"{role} self-hash binding failed: {path}")
    return payload, path


def _inventory_completed_run(
    run_dir: Path,
) -> dict[tuple[str, str], Path]:
    """Require the exact 25-file v6 completed-prediction inventory."""

    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    expected: set[Path] = {run_dir / "method_lock.json"}
    for candidate in CANDIDATES:
        for task in TASKS:
            expected.add(run_dir / "models" / f"{candidate}_{task}.joblib")
    calibrators: dict[tuple[str, str], Path] = {}
    calibrator_dir = run_dir / "calibrators"
    for candidate in CANDIDATES:
        for task in TASKS:
            matches = sorted(calibrator_dir.glob(f"{candidate}_{task}_*.joblib"))
            if len(matches) != 1:
                raise ValueError(
                    f"recovery requires one calibrator for {candidate}/{task}; "
                    f"found {len(matches)}"
                )
            calibrators[(candidate, task)] = matches[0]
            expected.add(matches[0])
    prediction_dir = run_dir / "predictions"
    expected.update(
        prediction_dir / f"raw_selection_2024_{month:02d}.parquet"
        for month in SELECTION_MONTHS
    )
    expected.add(prediction_dir / "selection_2024_crossfit_joint.parquet")
    expected.update(
        prediction_dir / f"retrospective_2025_{month:02d}.parquet"
        for month in range(1, 13)
    )
    observed = {path for path in run_dir.rglob("*") if path.is_file()}
    if observed != expected:
        missing = sorted(path.as_posix() for path in expected - observed)
        unexpected = sorted(path.as_posix() for path in observed - expected)
        raise ValueError(
            f"completed BC-POT-R artifact inventory differs; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return calibrators


def _verify_failure_record(
    path: Path,
    *,
    run_dir: Path,
    lock_path: Path,
) -> dict[str, Any]:
    failure: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    files = [item for item in run_dir.rglob("*") if item.is_file()]
    if (
        failure.get("status")
        != "RETAINED_POST_PREDICTION_SATURATED_CONTRAST_REPORTING_FAILURE"
        or failure.get("run_id") != run_dir.name
        or failure.get("final_report_written") is not False
        or failure.get("breakthrough_claimed") is not False
        or failure.get("2026_outcomes_accessed") is not False
        or int(failure.get("completed_models", -1)) != 4
        or int(failure.get("completed_calibrators", -1)) != 4
        or int(failure.get("completed_prediction_artifacts", -1)) != 16
        or int(failure.get("completed_2025_months", -1)) != 12
        or int(failure.get("preserved_run_file_count", -1)) != len(files)
        or int(failure.get("preserved_run_bytes", -1))
        != sum(item.stat().st_size for item in files)
        or failure.get("method_lock_sha256") != sha256_file(lock_path)
    ):
        raise ValueError("BC-POT-R v6 failure record does not bind the completed run")
    log_path = Path(str(failure.get("log", "")))
    if not log_path.is_file() or failure.get("log_sha256") != sha256_file(log_path):
        raise ValueError("BC-POT-R v6 failure log binding failed")
    return failure


def _validate_locked_model_parameters(
    model: BoundaryCatBoostModel,
    *,
    task: str,
    lock: dict[str, Any],
) -> dict[str, Any]:
    actual = dict(model.estimator.get_params())
    expected = dict(lock["model_parameters"][task])
    for name, expected_value in expected.items():
        if name == "iterations":
            continue
        if actual.get(name) != expected_value:
            raise ValueError(f"locked model parameter differs: {model.candidate}/{task}/{name}")
    tree_count = int(model.estimator.tree_count_)
    fixed = {
        "iterations": tree_count,
        "random_seed": 20260903,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "bootstrap_type": "Bayesian",
        "task_type": "GPU",
        "devices": "0",
        "verbose": False,
        "allow_writing_files": False,
    }
    for name, expected_value in fixed.items():
        if actual.get(name) != expected_value:
            raise ValueError(
                f"locked model execution parameter differs: "
                f"{model.candidate}/{task}/{name}"
            )
    return actual


def _recover_models(
    run_dir: Path,
    *,
    lock: dict[str, Any],
    expected_flare_features: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, tuple[str, ...]]]]:
    records: list[dict[str, Any]] = []
    feature_sets: dict[str, dict[str, tuple[str, ...]]] = {}
    for task in TASKS:
        for candidate in CANDIDATES:
            path = run_dir / "models" / f"{candidate}_{task}.joblib"
            model = joblib.load(path)
            if (
                not isinstance(model, BoundaryCatBoostModel)
                or model.task != task
                or model.candidate != candidate
            ):
                raise TypeError(f"invalid recovered boundary model contract: {path}")
            if model.flare_features != expected_flare_features:
                raise ValueError(f"recovered model FLARE feature lock differs: {candidate}/{task}")
            if candidate == "boundary_only":
                if not set(model.capacity_features).issubset(CAPACITY_STATIC_FEATURES):
                    raise ValueError("boundary-only model contains induced dynamic capacity")
                if not set(model.boundary_features).issubset(BOUNDARY_FULL_FEATURES):
                    raise ValueError("boundary-only model contains residual boundary features")
            else:
                allowed = {*BOUNDARY_RESIDUAL_FEATURES, *BOUNDARY_OBSERVATION_FEATURES}
                if not set(model.capacity_features).issubset(CAPACITY_ALL_FEATURES):
                    raise ValueError("residual model exceeds capacity feature contract")
                if not set(model.boundary_features).issubset(allowed):
                    raise ValueError("residual model exceeds boundary feature contract")
            settings = {
                "capacity": model.capacity_features,
                "boundary": model.boundary_features,
            }
            previous = feature_sets.setdefault(candidate, settings)
            if previous != settings:
                raise ValueError(f"recovered feature set differs by task: {candidate}")
            parameters = _validate_locked_model_parameters(
                model,
                task=task,
                lock=lock,
            )
            importance = np.asarray(
                model.estimator.get_feature_importance(), dtype=np.float64
            )
            names = list(model.estimator.feature_names_)
            if importance.shape != (len(names),):
                raise ValueError(f"model feature importance is malformed: {candidate}/{task}")
            order = np.argsort(-importance)
            records.append(
                {
                    "candidate": candidate,
                    "task": task,
                    **_artifact_record(path),
                    "refit_iterations": int(model.estimator.tree_count_),
                    "flare_feature_count": len(model.flare_features),
                    "capacity_feature_count": len(model.capacity_features),
                    "boundary_feature_count": len(model.boundary_features),
                    "estimator_parameters_sha256": canonical_json_sha256(parameters),
                    "training_rows": None,
                    "positive_rows": None,
                    "fit_seconds": None,
                    "recovery_note": (
                        "Training counts and elapsed fit time were not persisted before the "
                        "reporting failure and are intentionally not reconstructed."
                    ),
                    "top_30_feature_importance": [
                        {
                            "feature": names[index],
                            "importance": float(importance[index]),
                        }
                        for index in order[:30]
                    ],
                }
            )
            del model
            gc.collect()
    return records, feature_sets


def _assert_probability_reproduction(
    expected: NDArray[np.float64],
    persisted: NDArray[np.float64],
    *,
    role: str,
) -> float:
    if expected.shape != persisted.shape:
        raise ValueError(f"{role} probability shapes differ")
    maximum = float(np.max(np.abs(expected - persisted), initial=0.0))
    if maximum > PROBABILITY_REPRODUCTION_ATOL:
        raise ValueError(f"{role} predictions differ by {maximum:.3g}")
    return maximum


def _recover_calibration(
    run_dir: Path,
    calibrator_paths: dict[tuple[str, str], Path],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    NDArray[np.object_],
    dict[str, NDArray[np.float64]],
]:
    columns = [
        "sample_id",
        "FlightDate",
        "ArrDel15",
        "Cancelled",
        "delay_label_observed",
        *[
            f"raw_{candidate}_{task}"
            for candidate in CANDIDATES
            for task in TASKS
        ],
    ]
    parts: list[pd.DataFrame] = []
    raw_records: list[dict[str, Any]] = []
    for month in SELECTION_MONTHS:
        path = run_dir / "predictions" / f"raw_selection_2024_{month:02d}.parquet"
        frame = pd.read_parquet(path, columns=columns)
        if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ValueError(f"invalid raw selection IDs: {path}")
        dates = pd.to_datetime(frame["FlightDate"], errors="raise")
        if not dates.dt.year.eq(2024).all() or not dates.dt.month.eq(month).all():
            raise ValueError(f"raw selection period differs: {path}")
        parts.append(frame)
        raw_records.append({"year": 2024, "month": month, **_artifact_record(path), "rows": len(frame)})
    raw = pd.concat(parts, ignore_index=True)
    del parts
    if raw["sample_id"].duplicated().any():
        raise ValueError("raw selection IDs overlap across months")

    common_crossfit = np.ones(len(raw), dtype=np.bool_)
    crossfit: dict[tuple[str, str], NDArray[np.float32]] = {}
    records: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for task in TASKS:
            column = f"raw_{candidate}_{task}"
            values = raw[column].to_numpy(dtype=np.float64)
            selected = select_purged_forward_calibration(raw, values, task=task)
            path = calibrator_paths[(candidate, task)]
            persisted = joblib.load(path)
            if type(persisted) is not type(selected.final_calibrator):
                raise TypeError(f"persisted calibrator type differs: {candidate}/{task}")
            expected_values = selected.final_calibrator.predict(values)
            persisted_values = persisted.predict(values)
            maximum = _assert_probability_reproduction(
                expected_values,
                persisted_values,
                role=f"{candidate}/{task} final calibrator",
            )
            if not path.stem.endswith(f"_{selected.method}"):
                raise ValueError(f"calibrator filename and method differ: {path}")
            common_crossfit &= selected.crossfit_mask
            crossfit[(candidate, task)] = selected.crossfit_probabilities.astype("float32")
            records.append(
                {
                    "candidate": candidate,
                    "task": task,
                    "method": selected.method,
                    "final_fit_through": selected.final_fit_through,
                    "candidate_methods": list(selected.candidate_records),
                    "maximum_absolute_reproduction_error": maximum,
                    **_artifact_record(path),
                }
            )
            del persisted_values, expected_values, persisted
            gc.collect()
    ids = raw.loc[common_crossfit, "sample_id"].astype(object).to_numpy()
    candidate_probabilities: dict[str, NDArray[np.float64]] = {}
    for candidate in CANDIDATES:
        candidate_probabilities[candidate] = hurdle_joint_probabilities(
            crossfit[(candidate, "cancellation")][common_crossfit],
            crossfit[(candidate, "delay")][common_crossfit],
        )
    del raw, crossfit
    gc.collect()
    return records, raw_records, ids, candidate_probabilities


def _recover_selection(
    run_dir: Path,
    *,
    expected_ids: NDArray[np.object_],
    candidate_probabilities: dict[str, NDArray[np.float64]],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
    dict[str, Any],
    dict[str, Any],
]:
    path = run_dir / "predictions" / "selection_2024_crossfit_joint.parquet"
    frame = pd.read_parquet(path)
    ids = frame["sample_id"].astype(object).to_numpy()
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("selection cross-fit IDs are invalid")
    if not np.array_equal(ids, expected_ids):
        raise ValueError("selection cross-fit IDs differ from reconstructed calibration mask")
    dates = pd.to_datetime(frame["FlightDate"], errors="raise")
    if not dates.dt.year.eq(2024).all():
        raise ValueError("selection cross-fit period is not 2024")

    member_reproduction: dict[str, float] = {}
    methods: dict[str, NDArray[np.float64]] = {
        "flare24": _probability_frame(frame, "flare24"),
        "capacity_gated_simplex": _probability_frame(frame, "capacity_gated_simplex"),
    }
    for candidate in CANDIDATES:
        persisted = _probability_frame(frame, candidate)
        member_reproduction[candidate] = _assert_probability_reproduction(
            candidate_probabilities[candidate],
            persisted,
            role=f"selection {candidate}",
        )
        methods[candidate] = candidate_probabilities[candidate]
    observed = frame["joint_label_observed"].eq(1).to_numpy()
    labels = frame.loc[observed, "disruption_state"].to_numpy(dtype=np.int64)
    ensemble = _select_simplex(
        labels,
        {name: methods[name][observed] for name in ENSEMBLE_MEMBERS},
    )
    boundary = _apply_simplex(
        {name: methods[name] for name in ENSEMBLE_MEMBERS}, ensemble["weights"]
    )
    member_reproduction["boundary_ensemble"] = _assert_probability_reproduction(
        boundary,
        _probability_frame(frame, "boundary_ensemble"),
        role="selection boundary ensemble",
    )
    methods["boundary_ensemble"] = boundary
    gated = _select_boundary_gated_simplex(
        labels,
        {name: methods[name][observed] for name in ENSEMBLE_MEMBERS},
        frame.loc[observed, BOUNDARY_GATING_FEATURE].to_numpy(dtype=np.float64),
        cutpoint_values=frame[BOUNDARY_GATING_FEATURE].to_numpy(dtype=np.float64),
    )
    boundary_gated = _apply_boundary_gated_simplex(
        {name: methods[name] for name in ENSEMBLE_MEMBERS},
        frame[BOUNDARY_GATING_FEATURE].to_numpy(dtype=np.float64),
        gated,
    )
    member_reproduction["boundary_gated_ensemble"] = _assert_probability_reproduction(
        boundary_gated,
        _probability_frame(frame, "boundary_gated_ensemble"),
        role="selection boundary-gated ensemble",
    )
    methods["boundary_gated_ensemble"] = boundary_gated
    scores: dict[str, Any] = {}
    for name, probabilities in methods.items():
        log_rows, brier_rows = joint_loss_rows(labels, probabilities[observed])
        scores[name] = {
            "joint_log_loss": float(log_rows.mean()),
            "multiclass_brier": float(brier_rows.mean()),
            **_decision_metrics(labels, probabilities[observed]),
        }
    selected_method = min(
        ("boundary_ensemble", "boundary_gated_ensemble"),
        key=lambda name: (float(scores[name]["joint_log_loss"]), name),
    )
    accuracy_bias = _select_accuracy_bias(labels, methods[selected_method][observed])
    artifact = {**_artifact_record(path), "rows": len(frame)}
    verification = {
        "rows": len(frame),
        "exact_reconstructed_sample_id_order": True,
        "maximum_absolute_prediction_errors": member_reproduction,
        "probability_reproduction_atol": PROBABILITY_REPRODUCTION_ATOL,
    }
    del frame, methods, candidate_probabilities, boundary, boundary_gated
    gc.collect()
    return artifact, scores, ensemble, selected_method, accuracy_bias, {
        "selection": gated,
        "verification": verification,
    }


def _load_and_verify_audit(
    run_dir: Path,
    *,
    ensemble: dict[str, Any],
    gated: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    maximum_global = 0.0
    maximum_gated = 0.0
    for month in range(1, 13):
        path = run_dir / "predictions" / f"retrospective_2025_{month:02d}.parquet"
        frame = pd.read_parquet(path)
        if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ValueError(f"invalid retrospective IDs: {path}")
        dates = pd.to_datetime(frame["FlightDate"], errors="raise")
        if not dates.dt.year.eq(2025).all() or not dates.dt.month.eq(month).all():
            raise ValueError(f"retrospective period differs: {path}")
        inputs = {
            name: _probability_frame(frame, name) for name in ENSEMBLE_MEMBERS
        }
        reproduced = _apply_simplex(inputs, ensemble["weights"])
        maximum_global = max(
            maximum_global,
            _assert_probability_reproduction(
                reproduced,
                _probability_frame(frame, "boundary_ensemble"),
                role=f"2025-{month:02d} boundary ensemble",
            ),
        )
        reproduced_gated = _apply_boundary_gated_simplex(
            inputs,
            frame[BOUNDARY_GATING_FEATURE].to_numpy(dtype=np.float64),
            gated,
        )
        maximum_gated = max(
            maximum_gated,
            _assert_probability_reproduction(
                reproduced_gated,
                _probability_frame(frame, "boundary_gated_ensemble"),
                role=f"2025-{month:02d} boundary-gated ensemble",
            ),
        )
        parts.append(frame)
        records.append({"year": 2025, "month": month, **_artifact_record(path), "rows": len(frame)})
        del inputs, reproduced, reproduced_gated
        gc.collect()
    audit = pd.concat(parts, ignore_index=True)
    del parts
    if audit["sample_id"].duplicated().any():
        raise ValueError("retrospective IDs overlap across months")
    return audit, records, {
        "months": 12,
        "rows": len(audit),
        "global_ensemble_maximum_absolute_error": maximum_global,
        "gated_ensemble_maximum_absolute_error": maximum_gated,
        "probability_reproduction_atol": PROBABILITY_REPRODUCTION_ATOL,
        "all_predictions_immutable_and_reproduced": True,
    }


def recover_boundary_study(
    *,
    source_run_dir: Path,
    failure_manifest_path: Path,
    recovery_run_dir: Path,
    output_path: Path,
    bootstrap_repetitions: int = 2_000,
    seed: int = 20260903,
) -> dict[str, Any]:
    """Recover the final study report without fitting or predicting anything."""

    if recovery_run_dir.exists():
        raise FileExistsError(f"refusing to reuse recovery directory: {recovery_run_dir}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite recovered report: {output_path}")
    if bootstrap_repetitions < 100 or seed != 20260903:
        raise ValueError("BC-POT-R recovery settings differ from the frozen protocol")
    started_at = time.perf_counter()
    provenance_at_start = capture_provenance(_recovery_implementation_files())
    calibrator_paths = _inventory_completed_run(source_run_dir)
    lock_path = source_run_dir / "method_lock.json"
    lock, lock_hash_key = _self_hashed(lock_path)
    if (
        lock.get("status")
        != "LOCKED_BCPOTR_RETROSPECTIVE_REDEVELOPMENT_BEFORE_NEW_OUTCOME_LOAD"
        or lock.get("previous_2025_aggregate_results_known") is not True
        or lock.get("new_2025_boundary_predictions_or_labels_loaded_before_lock") is not False
        or lock.get("confirmation_2026_opened") is not False
        or tuple(lock.get("candidates", ())) != CANDIDATES
        or tuple(lock.get("ensemble_members", ())) != ENSEMBLE_MEMBERS
    ):
        raise ValueError("source BC-POT-R method lock is invalid")
    failure = _verify_failure_record(
        failure_manifest_path,
        run_dir=source_run_dir,
        lock_path=lock_path,
    )
    if failure.get("method_lock_self_hash") != lock[lock_hash_key]:
        raise ValueError("failure record does not bind the method-lock self-hash")

    protocol_path = _verified_record(lock["protocol"], role="locked BC-POT-R protocol")
    protocol = _load_protocol(
        protocol_path,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    boundary_manifest, boundary_manifest_path = _verify_self_hashed_record(
        lock["boundary_inputs"]["manifest"], role="locked boundary manifest"
    )
    _verify_self_hashed_record(
        lock["boundary_inputs"]["validation"], role="locked boundary validation"
    )
    _verify_self_hashed_record(
        lock["boundary_rotation_outcome_blind_gate"], role="locked rotation gate"
    )
    _parent, _ = _verify_self_hashed_record(
        lock["parent_report"], role="locked parent report"
    )
    meta, _ = _verify_self_hashed_record(lock["meta_report"], role="locked meta report")
    baseline, _ = _verify_self_hashed_record(
        lock["baseline_selection_report"], role="locked baseline selection report"
    )
    baseline_audit, _ = _verify_self_hashed_record(
        lock["baseline_audit_report"], role="locked baseline audit report"
    )
    del meta, baseline_audit, boundary_manifest_path

    expected_flare = tuple(baseline["feature_sets"]["rotation_structural"])
    model_records, feature_sets = _recover_models(
        source_run_dir,
        lock=lock,
        expected_flare_features=expected_flare,
    )
    calibration_records, raw_records, expected_ids, candidate_probabilities = (
        _recover_calibration(source_run_dir, calibrator_paths)
    )
    (
        selection_artifact,
        selection_scores,
        ensemble_selection,
        selected_method,
        accuracy_bias,
        gated_bundle,
    ) = _recover_selection(
        source_run_dir,
        expected_ids=expected_ids,
        candidate_probabilities=candidate_probabilities,
    )
    del expected_ids
    gated_selection = gated_bundle["selection"]
    audit, audit_records, audit_verification = _load_and_verify_audit(
        source_run_dir,
        ensemble=ensemble_selection,
        gated=gated_selection,
    )

    dates = pd.to_datetime(audit["FlightDate"], errors="raise").dt.normalize()
    primary_mask = dates.between(
        pd.Timestamp(PRIMARY_AUDIT_START), pd.Timestamp(PRIMARY_AUDIT_END)
    ).to_numpy()
    primary = audit.loc[primary_mask].reset_index(drop=True)
    primary_methods = {name: _probability_frame(primary, name) for name in METHOD_NAMES}
    evaluation = evaluate_joint_probabilities(
        primary,
        cast(Any, primary_methods),
        reference_method="meta_current",
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=seed,
    )
    accuracy = _accuracy_comparisons(
        primary,
        primary_methods,
        reference="meta_current",
        repetitions=bootstrap_repetitions,
        seed=seed,
    )
    selected_accuracy_vs_schedule = _accuracy_comparisons(
        primary,
        {
            "schedule_baseline": primary_methods["schedule_baseline"],
            selected_method: primary_methods[selected_method],
        },
        reference="schedule_baseline",
        repetitions=bootstrap_repetitions,
        seed=seed,
    )[selected_method]
    full_methods = {name: _probability_frame(audit, name) for name in METHOD_NAMES}
    full_observed = audit["joint_label_observed"].eq(1).to_numpy()
    full_labels = audit.loc[full_observed, "disruption_state"].to_numpy(dtype=np.int64)
    full_scores: dict[str, Any] = {}
    for name, probabilities in full_methods.items():
        log_rows, brier_rows = joint_loss_rows(full_labels, probabilities[full_observed])
        full_scores[name] = {
            "joint_log_loss": float(log_rows.mean()),
            "multiclass_brier": float(brier_rows.mean()),
            **_decision_metrics(full_labels, probabilities[full_observed]),
        }
    selected_bias = (
        float(accuracy_bias["selected_delay_log_bias"]),
        float(accuracy_bias["selected_cancellation_log_bias"]),
    )
    primary_observed = primary["joint_label_observed"].eq(1).to_numpy()
    primary_labels = primary.loc[primary_observed, "disruption_state"].to_numpy(
        dtype=np.int64
    )
    biased_decision = _decision_metrics(
        primary_labels,
        primary_methods[selected_method][primary_observed],
        log_bias=selected_bias,
    )
    selected_accuracy = float(accuracy[selected_method]["accuracy"])
    reference_accuracy = float(accuracy["meta_current"]["accuracy"])
    absolute_gain = selected_accuracy - reference_accuracy
    breakthrough_gate = {
        "metric": "primary-period standard argmax accuracy",
        "reference": "meta_current",
        "reference_accuracy": reference_accuracy,
        "selected_method": selected_method,
        "selected_method_accuracy": selected_accuracy,
        "absolute_gain": absolute_gain,
        "minimum_required_absolute_gain": BREAKTHROUGH_MINIMUM,
        "stretch_required_absolute_gain": BREAKTHROUGH_STRETCH,
        "minimum_gate_passed": absolute_gain >= BREAKTHROUGH_MINIMUM,
        "stretch_gate_passed": absolute_gain >= BREAKTHROUGH_STRETCH,
        "no_relative_percentage_substitution": True,
    }
    monthly_scores = _monthly_scores(audit, full_methods)
    residual_regime_scores = _residual_regime_scores(
        primary, primary_methods, gated_selection
    )
    airport_scores = _airport_scores(
        primary, primary_methods, selected_method=selected_method
    )
    primary_prevalence = {
        state: float((primary_labels == index).mean())
        for index, state in enumerate(PROBABILITY_STATES)
    }
    signal_summary = _boundary_signal_summary(primary)
    signal_associations = _boundary_signal_outcome_analysis(
        primary,
        gated_selection,
        repetitions=bootstrap_repetitions,
        seed=seed,
    )
    saturated = [
        record
        for record in signal_associations
        if record["signal"] == "any_nonzero_boundary_residual"
    ]
    if not saturated or any(
        record["inference_status"]
        != "NOT_ESTIMABLE_INSUFFICIENT_TEMPORAL_CONTRAST_SUPPORT"
        for record in saturated
    ):
        raise RuntimeError("recovery did not preserve the saturated-contrast diagnosis")

    final_provenance = _assert_source_unchanged(provenance_at_start)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": STUDY_STATUS,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": protocol["identity"]["expanded_name"],
        "epistemic_status": protocol["epistemic_status"],
        "protocol": _artifact_record(protocol_path),
        "method_lock": {
            **_artifact_record(lock_path),
            "self_hash": lock[lock_hash_key],
        },
        "recovery": {
            "status": RECOVERY_STATUS,
            "source_run_directory": source_run_dir.as_posix(),
            "recovery_run_directory": recovery_run_dir.as_posix(),
            "failure_manifest": _artifact_record(failure_manifest_path),
            "models_refit": False,
            "calibrators_refit_for_prediction": False,
            "predictions_regenerated": False,
            "selection_reconstructed_from_locked_q4_predictions": True,
            "metrics_recomputed_from_locked_2025_predictions": True,
            "reporting_change": (
                "A date-cluster interval with inadequate temporal contrast support is "
                "reported as non-estimable; predictive artifacts are unchanged."
            ),
            "selection_artifact_verification": gated_bundle["verification"],
            "retrospective_artifact_verification": audit_verification,
        },
        "boundary_inputs": lock["boundary_inputs"],
        "boundary_rotation_outcome_blind_gate": lock[
            "boundary_rotation_outcome_blind_gate"
        ],
        "boundary_context_totals": {
            "rows": boundary_manifest["rows"],
            "resource_nodes_computed_not_persisted": boundary_manifest["resource_nodes"],
            "rotation_edges_computed_not_persisted": boundary_manifest["rotation_edges"],
        },
        "parent_report": lock["parent_report"],
        "meta_report": lock["meta_report"],
        "baseline_selection_report": lock["baseline_selection_report"],
        "baseline_audit_report": lock["baseline_audit_report"],
        "feature_sets": {
            candidate: {name: list(values) for name, values in settings.items()}
            for candidate, settings in feature_sets.items()
        },
        "model_artifacts": model_records,
        "raw_selection_prediction_artifacts": raw_records,
        "calibration_artifacts": calibration_records,
        "selection_crossfit_prediction_artifact": selection_artifact,
        "selection_scores": selection_scores,
        "ensemble_selection": ensemble_selection,
        "boundary_gated_ensemble_selection": gated_selection,
        "selected_method_by_2024_forward_score": selected_method,
        "accuracy_bias_selection": accuracy_bias,
        "retrospective_prediction_artifacts": audit_records,
        "primary_evaluation_period": [PRIMARY_AUDIT_START, PRIMARY_AUDIT_END],
        "primary_evaluation": evaluation,
        "primary_argmax_decision_metrics": accuracy,
        "primary_selected_argmax_comparison_vs_schedule": selected_accuracy_vs_schedule,
        "secondary_q4_locked_biased_selected_method_decision": biased_decision,
        "primary_class_prevalence": primary_prevalence,
        "retrospective_monthly_scores": monthly_scores,
        "retrospective_boundary_residual_regime_scores": residual_regime_scores,
        "primary_boundary_signal_summary": signal_summary,
        "primary_boundary_signal_outcome_associations": signal_associations,
        "retrospective_airport_scores": airport_scores,
        "full_year_descriptive_scores": full_scores,
        "breakthrough_gate": breakthrough_gate,
        "outcomes_accessed": {
            "training_and_selection_years": [2024],
            "retrospective_evaluation_years": [2025],
            "2025_results_known_before_architecture": True,
            "2026_accessed": False,
        },
        "confirmation_gate": {"year": 2026, "opened": False},
        "claim_limits": [
            "This is retrospective redevelopment; earlier aggregate 2025 results informed the hypothesis.",
            "The boundary residual is a computational contrast, not a causal effect.",
            "BTS final schedules are a retrospective T-24 proxy, not archived schedule snapshots.",
            "Capacity, queue, recovery, and shadow-price fields are model states, not observed airport operations.",
            "No active runway, gate, crew, tail assignment, TFM restriction, or realised queue is claimed.",
            "The any-boundary-change contrast is saturated and its date-cluster interval is non-estimable.",
            "The 2026 confirmation gate remains unopened.",
        ],
        "versions": {
            "python": platform.python_version(),
            "catboost": version("catboost"),
            "joblib": version("joblib"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
        },
        "provenance": final_provenance,
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_path, report)
    recovery_run_dir.mkdir(parents=True)
    write_canonical_json(recovery_run_dir / "run_manifest.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--failure-manifest", type=Path, required=True)
    parser.add_argument("--recovery-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = recover_boundary_study(
        source_run_dir=args.source_run_dir,
        failure_manifest_path=args.failure_manifest,
        recovery_run_dir=args.recovery_run_dir,
        output_path=args.output,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "recovery": result["recovery"]["status"],
                "breakthrough_gate": result["breakthrough_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
