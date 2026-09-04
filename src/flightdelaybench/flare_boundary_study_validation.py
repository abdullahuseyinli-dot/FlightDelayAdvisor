"""Validate the BC-POT-R retrospective study and recompute its primary results."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .flare_boundary_contracts import (
    BOUNDARY_FULL_FEATURES,
    BOUNDARY_OBSERVATION_FEATURES,
    BOUNDARY_RESIDUAL_FEATURES,
    validate_boundary_predictors,
)
from .flare_boundary_study import (
    BREAKTHROUGH_MINIMUM,
    BREAKTHROUGH_STRETCH,
    CANDIDATES,
    _decision_metrics,
    _normalize_joint,
)
from .flare_capacity_contracts import (
    CAPACITY_ALL_FEATURES,
    CAPACITY_STATIC_FEATURES,
    validate_capacity_predictors,
)
from .flare_capacity_study import PRIMARY_AUDIT_END, PRIMARY_AUDIT_START, _joint_columns
from .flare_evaluation import joint_loss_rows
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

METHODS = (
    "schedule_baseline",
    "flare24",
    "capacity_gated_simplex",
    "meta_current",
    *CANDIDATES,
    "boundary_ensemble",
    "boundary_gated_ensemble",
)
SELECTION_METHODS = tuple(
    method for method in METHODS if method not in {"schedule_baseline", "meta_current"}
)
RECOVERY_STATUS = "RECOVERED_FROM_V6_REPORTING_ONLY_FAILURE_NO_RETRAINING"


def _self_hashed(path: Path) -> tuple[dict[str, Any], str]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    keys = [
        key for key in ("report_sha256", "manifest_sha256", "validation_sha256") if key in payload
    ]
    if len(keys) != 1:
        raise ValueError(f"invalid self-hash keys in {path}")
    key = keys[0]
    body = {name: value for name, value in payload.items() if name != key}
    if canonical_json_sha256(body) != payload[key]:
        raise ValueError(f"self-hash validation failed: {path}")
    return payload, key


def _verified_file(record: dict[str, Any], *, role: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{role} checksum failed: {path}")
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{role} byte count failed: {path}")
    return path


def _audit_probability_frame(
    path: Path,
    *,
    methods: tuple[str, ...] = METHODS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [
        "sample_id",
        "FlightDate",
        "joint_label_observed",
        "disruption_state",
        *[column for method in methods for column in _joint_columns(method)],
    ]
    frame = pd.read_parquet(path, columns=columns)
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError(f"boundary prediction artifact has invalid IDs: {path}")
    maximum_error = 0.0
    for method in methods:
        raw = frame.loc[:, list(_joint_columns(method))].to_numpy(dtype=np.float64)
        sums = raw.sum(axis=1)
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(sums - 1.0), initial=0.0)),
        )
        _normalize_joint(raw, role=f"validation {method}")
    return frame, {
        "path": path.as_posix(),
        "rows": len(frame),
        "maximum_absolute_probability_sum_error": maximum_error,
    }


def _validate_recovery_contract(
    report: dict[str, Any],
    *,
    report_path: Path,
    lock_path: Path,
    selection_rows: int,
    audit_rows: int,
) -> dict[str, Any] | None:
    recovery = report.get("recovery")
    if recovery is None:
        return None
    if not isinstance(recovery, dict):
        raise ValueError("BC-POT-R recovery contract is malformed")
    if (
        recovery.get("status") != RECOVERY_STATUS
        or recovery.get("models_refit") is not False
        or recovery.get("calibrators_refit_for_prediction") is not False
        or recovery.get("predictions_regenerated") is not False
        or recovery.get("selection_reconstructed_from_locked_q4_predictions") is not True
        or recovery.get("metrics_recomputed_from_locked_2025_predictions") is not True
    ):
        raise ValueError("BC-POT-R recovery changed a predictive artifact")
    failure_path = _verified_file(
        recovery["failure_manifest"], role="BC-POT-R recovery failure manifest"
    )
    source_lock = Path(str(recovery.get("source_run_directory", ""))) / "method_lock.json"
    if source_lock.resolve() != lock_path.resolve():
        raise ValueError("BC-POT-R recovery source directory differs from the method lock")
    duplicate = (
        Path(str(recovery.get("recovery_run_directory", ""))) / "run_manifest.json"
    )
    if not duplicate.is_file() or sha256_file(duplicate) != sha256_file(report_path):
        raise ValueError("BC-POT-R recovered run manifest is not an exact report duplicate")

    selection = recovery.get("selection_artifact_verification", {})
    retrospective = recovery.get("retrospective_artifact_verification", {})
    try:
        selection_atol = float(selection["probability_reproduction_atol"])
        audit_atol = float(retrospective["probability_reproduction_atol"])
        selection_errors = [
            float(value)
            for value in selection["maximum_absolute_prediction_errors"].values()
        ]
        audit_errors = [
            float(retrospective["global_ensemble_maximum_absolute_error"]),
            float(retrospective["gated_ensemble_maximum_absolute_error"]),
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("BC-POT-R recovery numerical audit is malformed") from error
    if (
        selection.get("exact_reconstructed_sample_id_order") is not True
        or int(selection.get("rows", -1)) != selection_rows
        or int(retrospective.get("months", -1)) != 12
        or int(retrospective.get("rows", -1)) != audit_rows
        or retrospective.get("all_predictions_immutable_and_reproduced") is not True
        or not 0.0 < selection_atol <= 1e-5
        or not 0.0 < audit_atol <= 1e-5
        or not selection_errors
        or any(not np.isfinite(value) or value > selection_atol for value in selection_errors)
        or any(not np.isfinite(value) or value > audit_atol for value in audit_errors)
    ):
        raise ValueError("BC-POT-R recovery probability reproduction failed")
    return {
        "status": RECOVERY_STATUS,
        "failure_manifest": failure_path.as_posix(),
        "selection_rows": selection_rows,
        "retrospective_rows": audit_rows,
        "maximum_selection_prediction_error": max(selection_errors),
        "maximum_retrospective_prediction_error": max(audit_errors),
        "models_refit": False,
        "predictions_regenerated": False,
    }


def validate_boundary_study(report_path: Path) -> dict[str, Any]:
    report, hash_key = _self_hashed(report_path)
    if report.get("status") != ("COMPLETE_BCPOTR_2024_SELECTION_2025_RETROSPECTIVE_REDEVELOPMENT"):
        raise ValueError("BC-POT-R report is not complete")
    if (
        report.get("outcomes_accessed", {}).get("2026_accessed") is not False
        or report.get("confirmation_gate", {}).get("opened") is not False
        or report.get("epistemic_status", {}).get("previous_2025_aggregate_results_known")
        is not True
    ):
        raise ValueError("BC-POT-R report misstates its epistemic boundary")

    protocol_path = _verified_file(report["protocol"], role="BC-POT-R protocol")
    del protocol_path
    lock_path = _verified_file(report["method_lock"], role="BC-POT-R method lock")
    lock, lock_hash_key = _self_hashed(lock_path)
    if (
        lock[lock_hash_key] != report["method_lock"]["self_hash"]
        or lock.get("status") != "LOCKED_BCPOTR_RETROSPECTIVE_REDEVELOPMENT_BEFORE_NEW_OUTCOME_LOAD"
        or lock.get("previous_2025_aggregate_results_known") is not True
        or lock.get("new_2025_boundary_predictions_or_labels_loaded_before_lock") is not False
        or lock.get("confirmation_2026_opened") is not False
    ):
        raise ValueError("BC-POT-R method lock is invalid")

    for role in (
        "parent_report",
        "meta_report",
        "baseline_selection_report",
        "baseline_audit_report",
    ):
        _verified_file(report[role], role=role)
    boundary_validation_path = _verified_file(
        report["boundary_inputs"]["validation"], role="boundary feature validation"
    )
    boundary_validation, boundary_validation_key = _self_hashed(boundary_validation_path)
    if (
        boundary_validation[boundary_validation_key]
        != report["boundary_inputs"]["validation"]["self_hash"]
        or boundary_validation.get("status") != "PASS_BOUNDARY_OPERATIONS_TWIN_FEATURE_VALIDATION"
    ):
        raise ValueError("BC-POT-R feature validation binding failed")
    rotation_gate_path = _verified_file(
        report["boundary_rotation_outcome_blind_gate"],
        role="boundary rotation outcome-blind gate",
    )
    rotation_gate, rotation_gate_key = _self_hashed(rotation_gate_path)
    if (
        rotation_gate[rotation_gate_key]
        != report["boundary_rotation_outcome_blind_gate"]["self_hash"]
        or rotation_gate.get("status")
        != "PASS_BOUNDARY_ROTATION_OUTCOME_BLIND_MATERIALITY_GATE"
        or rotation_gate.get("gate_passed") is not True
        or rotation_gate.get("outcomes_read") is not False
    ):
        raise ValueError("BC-POT-R rotation materiality gate binding failed")

    feature_sets = report.get("feature_sets", {})
    if set(feature_sets) != set(CANDIDATES):
        raise ValueError("BC-POT-R feature sets have unexpected candidates")
    expected_boundary = {
        "boundary_only": set(BOUNDARY_FULL_FEATURES),
        "counterfactual_residual": {
            *BOUNDARY_RESIDUAL_FEATURES,
            *BOUNDARY_OBSERVATION_FEATURES,
        },
    }
    for candidate in CANDIDATES:
        settings = feature_sets[candidate]
        capacity = list(settings.get("capacity", ()))
        boundary = list(settings.get("boundary", ()))
        validate_capacity_predictors(capacity)
        validate_boundary_predictors(boundary)
        if not set(capacity).issubset(CAPACITY_ALL_FEATURES) or not set(boundary).issubset(
            expected_boundary[candidate]
        ):
            raise ValueError(f"BC-POT-R {candidate} feature contract is invalid")
    if not set(feature_sets["boundary_only"]["capacity"]).issubset(CAPACITY_STATIC_FEATURES):
        raise ValueError("boundary-only candidate includes induced dynamic features")

    model_keys: set[tuple[str, str]] = set()
    for record in report.get("model_artifacts", []):
        key = (str(record.get("candidate")), str(record.get("task")))
        if key in model_keys:
            raise ValueError(f"duplicate BC-POT-R model: {key}")
        model_keys.add(key)
        _verified_file(record, role=f"BC-POT-R model {key}")
    if model_keys != {
        (candidate, task) for candidate in CANDIDATES for task in ("delay", "cancellation")
    }:
        raise ValueError("BC-POT-R model set is incomplete")
    for record in report.get("calibration_artifacts", []):
        _verified_file(record, role="BC-POT-R calibrator")
        if record.get("final_fit_through") != "2024-12-31":
            raise ValueError("BC-POT-R calibrator crosses its declared period")

    selection_path = _verified_file(
        report["selection_crossfit_prediction_artifact"],
        role="BC-POT-R selection predictions",
    )
    selection, selection_audit = _audit_probability_frame(
        selection_path,
        methods=SELECTION_METHODS,
    )
    if not pd.to_datetime(selection["FlightDate"], errors="raise").dt.year.eq(2024).all():
        raise ValueError("BC-POT-R selection prediction year is invalid")

    audit_frames: list[pd.DataFrame] = []
    artifact_audits: list[dict[str, Any]] = []
    seen_months: set[int] = set()
    for record in report.get("retrospective_prediction_artifacts", []):
        if int(record.get("year", -1)) != 2025:
            raise ValueError("BC-POT-R retrospective artifact has invalid year")
        month = int(record["month"])
        if month in seen_months:
            raise ValueError(f"duplicate BC-POT-R retrospective month: {month}")
        seen_months.add(month)
        path = _verified_file(record, role=f"BC-POT-R retrospective {month:02d}")
        frame, audit = _audit_probability_frame(path)
        audit_frames.append(frame)
        artifact_audits.append({"month": month, **audit})
    if seen_months != set(range(1, 13)):
        raise ValueError("BC-POT-R retrospective artifacts do not cover 2025")
    audit = pd.concat(audit_frames, ignore_index=True)
    recovery_audit = _validate_recovery_contract(
        report,
        report_path=report_path,
        lock_path=lock_path,
        selection_rows=len(selection),
        audit_rows=len(audit),
    )
    dates = pd.to_datetime(audit["FlightDate"], errors="raise").dt.normalize()
    primary_mask = dates.between(
        pd.Timestamp(PRIMARY_AUDIT_START), pd.Timestamp(PRIMARY_AUDIT_END)
    ).to_numpy()
    primary = audit.loc[primary_mask].reset_index(drop=True)
    observed = primary["joint_label_observed"].eq(1).to_numpy()
    labels = primary.loc[observed, "disruption_state"].to_numpy(dtype=np.int64)
    recomputed: dict[str, Any] = {}
    for method in METHODS:
        probabilities = _normalize_joint(
            primary.loc[:, list(_joint_columns(method))].to_numpy(dtype=np.float64),
            role=f"recomputed {method}",
        )[observed]
        log_rows, brier_rows = joint_loss_rows(labels, probabilities)
        decision = _decision_metrics(labels, probabilities)
        reported_joint = report["primary_evaluation"]["methods"][method]["joint"]
        reported_decision = report["primary_argmax_decision_metrics"][method]
        if (
            not np.isclose(
                float(log_rows.mean()),
                float(reported_joint["log_loss"]),
                rtol=0.0,
                atol=2e-8,
            )
            or not np.isclose(
                float(brier_rows.mean()),
                float(reported_joint["multiclass_brier"]),
                rtol=0.0,
                atol=2e-8,
            )
            or not np.isclose(
                decision["accuracy"],
                float(reported_decision["accuracy"]),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(f"BC-POT-R reported metrics do not reproduce: {method}")
        recomputed[method] = {
            "n": len(labels),
            "joint_log_loss": float(log_rows.mean()),
            "multiclass_brier": float(brier_rows.mean()),
            "accuracy": decision["accuracy"],
        }

    gate = report.get("breakthrough_gate", {})
    selected_method = str(report.get("selected_method_by_2024_forward_score", ""))
    if selected_method not in {"boundary_ensemble", "boundary_gated_ensemble"}:
        raise ValueError("BC-POT-R selected method is invalid")
    gain = recomputed[selected_method]["accuracy"] - recomputed["meta_current"]["accuracy"]
    gain_vs_schedule = (
        recomputed[selected_method]["accuracy"] - recomputed["schedule_baseline"]["accuracy"]
    )
    reported_vs_schedule = report.get(
        "primary_selected_argmax_comparison_vs_schedule", {}
    ).get("accuracy_difference_vs_reference", {})
    if (
        not np.isclose(float(gate.get("absolute_gain", np.nan)), gain)
        or gate.get("selected_method") != selected_method
        or gate.get("minimum_gate_passed") is not (gain >= BREAKTHROUGH_MINIMUM)
        or gate.get("stretch_gate_passed") is not (gain >= BREAKTHROUGH_STRETCH)
        or gate.get("no_relative_percentage_substitution") is not True
        or not np.isclose(float(reported_vs_schedule.get("estimate", np.nan)), gain_vs_schedule)
    ):
        raise ValueError("BC-POT-R breakthrough gate is inconsistent")
    associations = report.get("primary_boundary_signal_outcome_associations", [])
    if (
        len(associations) != 9
        or {record.get("signal") for record in associations}
        != {
            "any_nonzero_boundary_residual",
            "newly_observed_boundary_predecessor",
            "severe_vs_low_route_shadow_residual",
        }
        or {record.get("outcome") for record in associations}
        != {"joint_disruption", "cancellation", "delay_given_operated"}
        or any(record.get("causal_effect_claimed") is not False for record in associations)
    ):
        raise ValueError("BC-POT-R boundary-signal association analysis is incomplete")
    for record in report.get("provenance", {}).get("source_files", []):
        _verified_file(record, role="BC-POT-R source provenance")

    return {
        "status": "PASS_BCPOTR_STUDY_VALIDATION",
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "report": {
            "path": report_path.as_posix(),
            "sha256": sha256_file(report_path),
            "self_hash_key": hash_key,
            "self_hash": report[hash_key],
        },
        "method_lock": {
            "path": lock_path.as_posix(),
            "sha256": sha256_file(lock_path),
            "self_hash": lock[lock_hash_key],
        },
        "selection_prediction_audit": selection_audit,
        "retrospective_prediction_audits": artifact_audits,
        "primary_period": [PRIMARY_AUDIT_START, PRIMARY_AUDIT_END],
        "recomputed_primary_metrics": recomputed,
        "breakthrough_gate_recomputed": {
            "absolute_accuracy_gain": gain,
            "minimum_gate_passed": gain >= BREAKTHROUGH_MINIMUM,
            "stretch_gate_passed": gain >= BREAKTHROUGH_STRETCH,
        },
        "selected_absolute_accuracy_gain_vs_schedule": gain_vs_schedule,
        "epistemic_status": "retrospective redevelopment; earlier 2025 aggregates known",
        "confirmation_outcomes_accessed": False,
        "recovery_audit": recovery_audit,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = validate_boundary_study(args.report)
    result["validator_provenance"] = capture_provenance(
        (
            Path(__file__),
            Path(__file__).with_name("flare_boundary_study.py"),
            Path(__file__).with_name("flare_boundary_contracts.py"),
            Path(__file__).with_name("hashing.py"),
        )
    )
    result["validation_sha256"] = canonical_json_sha256(result)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite BC-POT-R validation: {args.output}")
    write_canonical_json(args.output, result)
    print(json.dumps({"status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
