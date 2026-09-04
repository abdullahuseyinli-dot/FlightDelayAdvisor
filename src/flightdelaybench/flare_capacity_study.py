"""Leakage-safe ablation study for the capacity-conditioned FLARE hypergraph.

The experiment keeps the published FLARE-24 final forecast as its exact reference,
fits four nested CC-RTH extensions on January--September 2024, selects calibration
and blending only from forward predictions in Q4 2024, writes a method lock, and
then performs a retrospective (not confirmatory) 2025 evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .bootstrap import paired_cluster_mean_difference
from .calibration import (
    BinaryCalibrator,
    CalibrationMethod,
    fit_binary_calibrator,
)
from .flare_capacity_contracts import CAPACITY_ALL_FEATURES
from .flare_capacity_modeling import (
    CAPACITY_CANDIDATE_FEATURES,
    CapacityCandidate,
    CapacityCatBoostModel,
    attach_capacity_feature_partitions,
    capacity_candidate_profile,
    fit_capacity_catboost,
    select_usable_capacity_features,
)
from .flare_evaluation import evaluate_joint_probabilities, joint_loss_rows
from .flare_modeling import task_frame
from .flare_reconciliation import hurdle_joint_probabilities
from .flare_study import (
    CALIBRATION_METHODS,
    MODEL_PARAMETERS,
    PREDICTION_ID_COLUMNS,
    TASKS,
    load_enriched_month,
)
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

CAPACITY_CANDIDATES: tuple[CapacityCandidate, ...] = tuple(CAPACITY_CANDIDATE_FEATURES)
AUGMENTED_CANDIDATES: tuple[CapacityCandidate, ...] = tuple(
    candidate for candidate in CAPACITY_CANDIDATES if candidate != "flare24"
)
BLEND_CANDIDATES = CAPACITY_CANDIDATES
GATING_FEATURE = "ccrth_route_sum_shadow_price"
GATING_QUANTILES = (0.50, 0.90)
GATING_LABELS = ("low", "elevated", "severe", "missing")
PROBABILITY_STATES = ("on_time", "delayed", "cancelled")
PERSISTED_FLOAT32_SIMPLEX_ACCEPTANCE = 1e-6
PURGE_DAYS = 2
EARLY_STOP_TRAIN_END = "2024-08-29"
FINAL_CALIBRATION_START = "2024-10-03"
FINAL_CALIBRATION_END = "2024-12-31"
PRIMARY_AUDIT_START = "2025-01-03"
PRIMARY_AUDIT_END = "2025-12-29"
RECOVERY_DISPOSITION = (
    "RETAINED_FAILED_RUN_2024_ONLY_MODELS_APPROVED_FOR_CHECKSUM_RECOVERY"
)
RECOVERY_ARTIFACT_ORIGIN = "CHECKSUM_VERIFIED_2024_ONLY_FAILED_RUN_RECOVERY"
PURGED_CALIBRATION_FOLDS = (
    (FINAL_CALIBRATION_START, "2024-10-13", "2024-10-16", "2024-10-31"),
    (FINAL_CALIBRATION_START, "2024-10-29", "2024-11-01", "2024-11-30"),
    (FINAL_CALIBRATION_START, "2024-11-28", "2024-12-01", "2024-12-31"),
)


def _frozen_capacity_implementation_files() -> tuple[Path, ...]:
    """Return every in-repository source that can affect a locked CC-RTH score."""

    directory = Path(__file__).parent
    return tuple(
        directory / name
        for name in (
            "flare_capacity_study.py",
            "flare_capacity_modeling.py",
            "flare_capacity.py",
            "flare_capacity_contracts.py",
            "flare_study.py",
            "flare_modeling.py",
            "flare_evaluation.py",
            "flare_reconciliation.py",
            "calibration.py",
            "bootstrap.py",
            "metrics.py",
            "census_modeling.py",
            "census_recent.py",
            "census_graph.py",
            "schedule_context.py",
            "recent.py",
            "recent_modeling.py",
            "modeling.py",
            "contracts.py",
            "hashing.py",
            "provenance.py",
        )
    )


def _verify_implementation_unchanged(
    started: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    """Fail if score-producing source bytes or the checked-out commit changed in-run."""

    current = capture_provenance(_frozen_capacity_implementation_files())
    if (
        current.get("git_head") != started.get("git_head")
        or current.get("source_files") != started.get("source_files")
    ):
        raise RuntimeError(
            f"CC-RTH implementation provenance changed before {phase}; retain this "
            "partial run and restart in a new directory"
        )
    return current


def _read_self_hashed(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    keys = [
        key
        for key in ("report_sha256", "manifest_sha256", "validation_sha256")
        if key in payload
    ]
    if len(keys) != 1:
        raise ValueError(f"artifact must have exactly one recognised self-hash: {path}")
    key = keys[0]
    recorded = str(payload[key])
    body = {name: value for name, value in payload.items() if name != key}
    if canonical_json_sha256(body) != recorded:
        raise ValueError(f"artifact self-hash failed: {path}")
    return payload, key


def _verified_artifact(record: dict[str, Any], *, role: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{role} checksum failed: {path}")
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{role} byte count failed: {path}")
    return path


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite CC-RTH artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated CC-RTH partial exists: {partial}")
    frame.to_parquet(partial, index=False, compression="zstd")
    partial.replace(path)
    return {
        "path": path.as_posix(),
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _atomic_joblib(value: Any, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite CC-RTH model: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated CC-RTH partial exists: {partial}")
    joblib.dump(value, partial)
    partial.replace(path)
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_capacity_model(record: dict[str, Any]) -> CapacityCatBoostModel:
    """Checksum-load one serialized model so training need not retain all estimators."""

    path = _verified_artifact(record, role="CC-RTH serialized model")
    value = joblib.load(path)
    if not isinstance(value, CapacityCatBoostModel):
        raise TypeError(f"unexpected CC-RTH model artifact type: {type(value)!r}")
    return value


def _load_recovery_model_record(
    path: Path | None,
    *,
    protocol_sha256: str,
    capacity_manifest_self_hash: str,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any] | None]:
    """Admit only checksum-bound, 2024-only models from a retained failed run."""

    if path is None:
        return {}, None
    if not path.is_file():
        raise FileNotFoundError(path)
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded_self_hash = payload.get("record_sha256")
    if not isinstance(recorded_self_hash, str) or canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "record_sha256"}
    ) != recorded_self_hash:
        raise ValueError("CC-RTH recovery record self-hash failed")
    boundary = payload.get("boundary_status", {})
    if (
        payload.get("disposition") != RECOVERY_DISPOSITION
        or boundary.get("2025_outcomes_accessed") is not False
        or boundary.get("2026_outcomes_accessed") is not False
        or boundary.get("method_lock_written") is not False
        or boundary.get("study_report_written") is not False
        or payload.get("protocol", {}).get("sha256") != protocol_sha256
        or payload.get("capacity_manifest", {}).get("self_hash")
        != capacity_manifest_self_hash
    ):
        raise ValueError("CC-RTH recovery record violates the frozen 2024-only boundary")
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for value in payload.get("completed_model_artifacts", []):
        record = dict(value)
        candidate = str(record.get("candidate", ""))
        task = str(record.get("task", ""))
        key = (candidate, task)
        if (
            candidate not in AUGMENTED_CANDIDATES
            or task not in TASKS
            or key in records
            or int(record.get("refit_iterations", 0)) <= 0
            or int(record.get("early_stopping_best_iteration", -1))
            != int(record["refit_iterations"]) - 1
        ):
            raise ValueError(f"invalid CC-RTH recovery model record: {key}")
        _verified_artifact(record, role=f"CC-RTH recovery model {candidate}/{task}")
        records[key] = record
    expected = {
        (candidate, task)
        for candidate in AUGMENTED_CANDIDATES
        for task in TASKS
    }
    if set(records) != expected:
        raise ValueError("CC-RTH recovery record does not contain all eight final models")
    return records, {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "source_failed_run": payload.get("run_directory"),
        "model_count": len(records),
        "2025_outcomes_accessed": False,
        "2026_outcomes_accessed": False,
    }


def _verify_capacity_manifest(
    manifest_path: Path,
    *,
    feature_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest, hash_key = _read_self_hashed(manifest_path)
    if manifest.get("status") != "COMPLETE_COVARIATE_GRAPH_NO_TARGET_OUTCOMES_ACCESSED":
        raise ValueError("CC-RTH study requires a complete covariate-only graph manifest")
    if set(manifest.get("target_years", [])) != {2024, 2025} or set(
        manifest.get("target_months", [])
    ) != set(range(1, 13)):
        raise ValueError("CC-RTH manifest must cover every month of 2024 and 2025")
    if tuple(manifest.get("features", ())) != CAPACITY_ALL_FEATURES:
        raise ValueError("CC-RTH manifest feature order differs from the registered contract")
    if manifest.get("outcome_columns_read") != [] or manifest.get(
        "target_tail_number_read"
    ) is not False:
        raise ValueError("CC-RTH manifest violates the covariate-only information boundary")
    root = feature_dir.resolve()
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for month_record in manifest.get("outputs", []):
        key = (int(month_record["year"]), int(month_record["month"]))
        if key in seen:
            raise ValueError(f"duplicate CC-RTH feature output: {key}")
        feature_record = dict(month_record["features"])
        path = _verified_artifact(feature_record, role=f"CC-RTH feature partition {key}")
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError(f"CC-RTH feature partition escapes supplied root: {path}") from error
        expected = feature_dir / f"year={key[0]}" / f"month={key[1]:02d}.parquet"
        if path.resolve() != expected.resolve():
            raise ValueError(f"CC-RTH feature partition has unexpected path: {path}")
        seen.add(key)
        records.append({"year": key[0], "month": key[1], **feature_record})
    expected_periods = {(year, month) for year in (2024, 2025) for month in range(1, 13)}
    if seen != expected_periods:
        raise ValueError("CC-RTH feature outputs do not cover the required periods")
    return manifest, [
        {
            "path": manifest_path.as_posix(),
            "sha256": sha256_file(manifest_path),
            "self_hash_key": hash_key,
            "self_hash": manifest[hash_key],
            "verified_feature_partitions": len(records),
        }
    ]


def _load_protocol(
    path: Path,
    *,
    rows_per_train_month: int,
    validation_limit: int,
    bootstrap_repetitions: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    models = protocol.get("models", {})
    calibration = protocol.get("calibration", {})
    blending = protocol.get("blending", {})
    evaluation = protocol.get("evaluation", {})
    boundary = protocol.get("information_boundary", {})
    if (
        protocol.get("identity", {}).get("method") != "CC-RTH-v1"
        or tuple(models.get("candidates", ())) != CAPACITY_CANDIDATES
        or int(models.get("rows_per_training_month", -1)) != rows_per_train_month
        or int(models.get("early_stopping_max_rows", -1)) != validation_limit
        or int(models.get("sampling_seed", -1)) != seed
        or models.get("delay_parameters") != MODEL_PARAMETERS["delay"]
        or models.get("cancellation_parameters") != MODEL_PARAMETERS["cancellation"]
        or models.get("early_stopping_training_end_after_purge")
        != EARLY_STOP_TRAIN_END
        or tuple(calibration.get("families", ())) != CALIBRATION_METHODS
        or int(calibration.get("purge_days", -1)) != PURGE_DAYS
        or tuple(tuple(fold) for fold in calibration.get("folds", ()))
        != PURGED_CALIBRATION_FOLDS
        or calibration.get("final_fit_through") != FINAL_CALIBRATION_END
        or calibration.get("training_starts_after_model_purge")
        != FINAL_CALIBRATION_START
        or blending.get("gating_feature") != GATING_FEATURE
        or tuple(float(value) for value in blending.get("gating_quantiles", ()))
        != GATING_QUANTILES
        or int(blending.get("minimum_regime_rows", -1)) != 10_000
        or int(evaluation.get("bootstrap_repetitions", -1)) != bootstrap_repetitions
        or int(evaluation.get("bootstrap_seed", -1)) != seed
        or tuple(evaluation.get("primary_evaluation_dates", ()))
        != (PRIMARY_AUDIT_START, PRIMARY_AUDIT_END)
        or int(boundary.get("training_outcome_year", -1)) != 2024
        or int(boundary.get("retrospective_evaluation_year", -1)) != 2025
        or int(boundary.get("confirmation_year", -1)) != 2026
        or boundary.get("confirmation_gate_opened") is not False
        or boundary.get("target_outcome_columns_allowed_in_graph") is not False
        or boundary.get("target_tail_number_allowed_in_graph") is not False
    ):
        raise ValueError("CC-RTH executable settings differ from the frozen protocol")
    return protocol, {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "status": protocol["identity"]["status"],
    }


def _verify_capacity_validation(
    path: Path,
    *,
    capacity_manifest_path: Path,
    capacity_manifest: dict[str, Any],
) -> dict[str, Any]:
    validation, hash_key = _read_self_hashed(path)
    if validation.get("status") != "PASS_CCRTH_GRAPH_VALIDATION":
        raise ValueError("CC-RTH modeling requires a passing graph validation gate")
    source = validation.get("manifest", {})
    if (
        Path(str(source.get("path", ""))).resolve() != capacity_manifest_path.resolve()
        or source.get("sha256") != sha256_file(capacity_manifest_path)
        or source.get("self_hash") != capacity_manifest["manifest_sha256"]
    ):
        raise ValueError("CC-RTH graph validation is not bound to the supplied manifest")
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "self_hash_key": hash_key,
        "self_hash": validation[hash_key],
        "status": validation["status"],
        "advisories": validation.get("advisories", []),
    }


def select_probability_simplex(
    labels: ArrayLike,
    probabilities: dict[str, ArrayLike],
) -> dict[str, Any]:
    """Fit a deterministic non-negative probability blend by convex log loss."""

    if tuple(probabilities) != BLEND_CANDIDATES:
        raise ValueError(f"blend candidates must be ordered as {BLEND_CANDIDATES}")
    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1 or y.size == 0 or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("blend labels must be a non-empty three-state vector")
    matrices = [np.asarray(probabilities[name], dtype=np.float64) for name in BLEND_CANDIDATES]
    if any(matrix.shape != (len(y), 3) for matrix in matrices):
        raise ValueError("blend probability matrices must align")
    if any(
        not np.isfinite(matrix).all()
        or (matrix < 0.0).any()
        or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8)
        for matrix in matrices
    ):
        raise ValueError("blend probabilities must be valid simplices")
    true_class = np.column_stack(
        [matrix[np.arange(len(y)), y] for matrix in matrices]
    )

    def objective(weights: NDArray[np.float64]) -> float:
        mixture = np.clip(true_class @ weights, 1e-12, 1.0)
        return float(-np.log(mixture).mean())

    def gradient(weights: NDArray[np.float64]) -> NDArray[np.float64]:
        mixture = np.clip(true_class @ weights, 1e-12, 1.0)
        return np.asarray(-(true_class / mixture[:, None]).mean(axis=0), dtype=np.float64)

    initial = np.full(len(BLEND_CANDIDATES), 1.0 / len(BLEND_CANDIDATES))
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(initial),
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    if not result.success:
        raise RuntimeError(f"CC-RTH simplex selection failed: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    weights /= weights.sum()
    blended = sum(
        weight * matrix for weight, matrix in zip(weights, matrices, strict=True)
    )
    log_rows, brier_rows = joint_loss_rows(y, blended)
    return {
        "selection_metric": "joint_log_loss",
        "optimizer": "convex SLSQP with analytic gradient",
        "success": True,
        "iterations": int(result.nit),
        "weights": {
            name: float(weight)
            for name, weight in zip(BLEND_CANDIDATES, weights, strict=True)
        },
        "joint_log_loss": float(log_rows.mean()),
        "multiclass_brier": float(brier_rows.mean()),
    }


def blend_probabilities(
    probabilities: dict[str, NDArray[np.float64]],
    weights: dict[str, float],
) -> NDArray[np.float64]:
    if tuple(probabilities) != BLEND_CANDIDATES or set(weights) != set(BLEND_CANDIDATES):
        raise ValueError("CC-RTH blend inputs have an unexpected candidate set")
    weight_array = np.asarray([weights[name] for name in BLEND_CANDIDATES], dtype=np.float64)
    if (weight_array < 0.0).any() or not np.isclose(weight_array.sum(), 1.0, atol=1e-8):
        raise ValueError("CC-RTH blend weights must be a probability simplex")
    result = sum(weights[name] * probabilities[name] for name in BLEND_CANDIDATES)
    output = np.asarray(result, dtype=np.float64)
    if not np.isfinite(output).all() or not np.allclose(output.sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("CC-RTH blend produced invalid joint probabilities")
    return output


def capacity_regimes(
    values: ArrayLike,
    cutpoints: ArrayLike,
) -> NDArray[np.str_]:
    numeric = np.asarray(values, dtype=np.float64)
    cuts = np.asarray(cutpoints, dtype=np.float64)
    if numeric.ndim != 1 or cuts.shape != (2,) or not np.isfinite(cuts).all():
        raise ValueError("capacity regimes require a vector and two finite cutpoints")
    if cuts[0] > cuts[1]:
        raise ValueError("capacity-regime cutpoints must be ordered")
    output = np.full(len(numeric), "missing", dtype="<U8")
    observed = np.isfinite(numeric)
    output[observed & (numeric <= cuts[0])] = "low"
    output[observed & (numeric > cuts[0]) & (numeric <= cuts[1])] = "elevated"
    output[observed & (numeric > cuts[1])] = "severe"
    return output


def select_capacity_gated_simplex(
    labels: ArrayLike,
    probabilities: dict[str, ArrayLike],
    gating_values: ArrayLike,
    *,
    cutpoint_values: ArrayLike | None = None,
    minimum_rows: int = 10_000,
) -> dict[str, Any]:
    """Select outcome-blind stress cutpoints and outcome-trained weights on Q4 only."""

    y = np.asarray(labels, dtype=np.int64)
    gate = np.asarray(gating_values, dtype=np.float64)
    if gate.shape != (len(y),):
        raise ValueError("gating values must align with labels")
    cutpoint_source = (
        gate if cutpoint_values is None else np.asarray(cutpoint_values, dtype=np.float64)
    )
    if cutpoint_source.ndim != 1:
        raise ValueError("capacity cutpoint values must be one-dimensional")
    observed_gate = cutpoint_source[np.isfinite(cutpoint_source)]
    if observed_gate.size == 0:
        raise ValueError("capacity-gated simplex has no observed gating covariate")
    cutpoints = np.quantile(observed_gate, GATING_QUANTILES).astype(np.float64)
    regimes = capacity_regimes(gate, cutpoints)
    global_selection = select_probability_simplex(y, probabilities)
    selections: dict[str, Any] = {}
    for regime in GATING_LABELS:
        mask = regimes == regime
        if int(mask.sum()) < minimum_rows or np.unique(y[mask]).size < 3:
            selections[regime] = {
                **global_selection,
                "rows": int(mask.sum()),
                "fallback_to_global": True,
            }
            continue
        selected = select_probability_simplex(
            y[mask],
            {
                name: np.asarray(values, dtype=np.float64)[mask]
                for name, values in probabilities.items()
            },
        )
        selections[regime] = {
            **selected,
            "rows": int(mask.sum()),
            "fallback_to_global": False,
        }
    return {
        "gating_feature": GATING_FEATURE,
        "cutpoint_quantiles": list(GATING_QUANTILES),
        "cutpoints": cutpoints.tolist(),
        "minimum_rows": minimum_rows,
        "global": global_selection,
        "regimes": selections,
    }


def apply_capacity_gated_simplex(
    probabilities: dict[str, NDArray[np.float64]],
    gating_values: ArrayLike,
    selection: dict[str, Any],
) -> NDArray[np.float64]:
    gate = np.asarray(gating_values, dtype=np.float64)
    first = next(iter(probabilities.values()))
    if gate.shape != (len(first),):
        raise ValueError("gating values must align with prediction matrices")
    regimes = capacity_regimes(gate, selection["cutpoints"])
    output = np.empty_like(first, dtype=np.float64)
    for regime in GATING_LABELS:
        mask = regimes == regime
        if mask.any():
            subset = {name: values[mask] for name, values in probabilities.items()}
            output[mask] = blend_probabilities(
                subset,
                cast(dict[str, float], selection["regimes"][regime]["weights"]),
            )
    if not np.isfinite(output).all() or not np.allclose(output.sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("capacity-gated simplex produced invalid probabilities")
    return output


def _binary_eligible(
    frame: pd.DataFrame,
    task: str,
) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
    if task == "cancellation":
        mask = frame["Cancelled"].isin([0, 1]).to_numpy(dtype=np.bool_)
        labels = frame["Cancelled"].fillna(-1).to_numpy(dtype=np.int64)
    elif task == "delay":
        mask = (
            frame["Cancelled"].eq(0) & frame["delay_label_observed"].eq(1)
        ).to_numpy(dtype=np.bool_)
        labels = frame["ArrDel15"].fillna(-1).to_numpy(dtype=np.int64)
    else:
        raise ValueError(f"invalid calibration task: {task}")
    if not np.isin(labels[mask], [0, 1]).all():
        raise ValueError(f"invalid {task} calibration labels")
    return mask, labels


def _binary_loss(labels: ArrayLike, probabilities: ArrayLike) -> NDArray[np.float64]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if y.shape != p.shape or y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("binary loss inputs must be aligned and valid")
    return np.asarray(-(y * np.log(p) + (1 - y) * np.log1p(-p)), dtype=np.float64)


@dataclass(slots=True)
class PurgedCalibrationSelection:
    task: str
    method: CalibrationMethod
    crossfit_probabilities: NDArray[np.float64]
    crossfit_mask: NDArray[np.bool_]
    final_calibrator: BinaryCalibrator
    final_fit_through: str
    candidate_records: tuple[dict[str, Any], ...]


def select_purged_forward_calibration(
    frame: pd.DataFrame,
    raw_probabilities: ArrayLike,
    *,
    task: str,
    methods: tuple[CalibrationMethod, ...] = CALIBRATION_METHODS,
) -> PurgedCalibrationSelection:
    """Select calibration with a two-date embargo at every forward boundary."""

    raw = np.asarray(raw_probabilities, dtype=np.float64)
    if raw.shape != (len(frame),) or not np.isfinite(raw).all():
        raise ValueError("raw calibration probabilities must align and be finite")
    dates = pd.to_datetime(frame["FlightDate"], errors="raise").dt.normalize()
    eligible, labels = _binary_eligible(frame, task)
    validation_union = np.zeros(len(frame), dtype=np.bool_)
    records: list[dict[str, Any]] = []
    crossfit_by_method: dict[CalibrationMethod, NDArray[np.float64]] = {}
    for method in methods:
        crossfit = np.full(len(frame), np.nan, dtype=np.float64)
        fold_records: list[dict[str, Any]] = []
        for fold_index, (train_start, train_end, valid_start, valid_end) in enumerate(
            PURGED_CALIBRATION_FOLDS,
            start=1,
        ):
            gap_days = (pd.Timestamp(valid_start) - pd.Timestamp(train_end)).days - 1
            if gap_days != PURGE_DAYS:
                raise RuntimeError("calibration fold does not implement the declared purge")
            train_date = dates.between(pd.Timestamp(train_start), pd.Timestamp(train_end)).to_numpy()
            valid_date = dates.between(pd.Timestamp(valid_start), pd.Timestamp(valid_end)).to_numpy()
            train = train_date & eligible
            valid = valid_date & eligible
            if not train.any() or not valid.any() or np.unique(labels[train]).size != 2:
                raise ValueError(f"purged calibration fold {fold_index} for {task} lacks support")
            calibrator = fit_binary_calibrator(method, raw[train], labels[train])
            crossfit[valid_date] = calibrator.predict(raw[valid_date])
            fold_loss = _binary_loss(labels[valid], crossfit[valid])
            fold_records.append(
                {
                    "fold": fold_index,
                    "train_dates": [train_start, train_end],
                    "purged_dates": [
                        (pd.Timestamp(train_end) + pd.Timedelta(days=1)).date().isoformat(),
                        (pd.Timestamp(valid_start) - pd.Timedelta(days=1)).date().isoformat(),
                    ],
                    "validation_dates": [valid_start, valid_end],
                    "training_rows": int(train.sum()),
                    "validation_rows": int(valid.sum()),
                    "log_loss": float(fold_loss.mean()),
                    "brier": float(np.mean(np.square(crossfit[valid] - labels[valid]))),
                }
            )
            validation_union |= valid_date
        scored = validation_union & eligible
        pooled_loss = _binary_loss(labels[scored], crossfit[scored])
        records.append(
            {
                "method": method,
                "folds": fold_records,
                "pooled_rows": int(scored.sum()),
                "pooled_log_loss": float(pooled_loss.mean()),
                "pooled_brier": float(np.mean(np.square(crossfit[scored] - labels[scored]))),
            }
        )
        crossfit_by_method[method] = crossfit
    order = {method: index for index, method in enumerate(methods)}
    selected_record = min(
        records,
        key=lambda record: (
            float(record["pooled_log_loss"]),
            order[cast(CalibrationMethod, record["method"])],
        ),
    )
    selected_method = cast(CalibrationMethod, selected_record["method"])
    final_date = dates.between(
        pd.Timestamp(FINAL_CALIBRATION_START), pd.Timestamp(FINAL_CALIBRATION_END)
    ).to_numpy()
    final_fit = eligible & final_date
    if not final_fit.any() or np.unique(labels[final_fit]).size != 2:
        raise ValueError(f"final purged calibration fit for {task} lacks support")
    final = fit_binary_calibrator(selected_method, raw[final_fit], labels[final_fit])
    return PurgedCalibrationSelection(
        task=task,
        method=selected_method,
        crossfit_probabilities=crossfit_by_method[selected_method],
        crossfit_mask=validation_union,
        final_calibrator=final,
        final_fit_through=FINAL_CALIBRATION_END,
        candidate_records=tuple(records),
    )


def _load_capacity_month(
    *,
    year: int,
    month: int,
    limit: int | None,
    capacity_feature_dir: Path,
    loader_arguments: dict[str, Any],
) -> pd.DataFrame:
    if year not in {2024, 2025}:
        raise ValueError(f"CC-RTH experiment refuses outcome year {year}")
    frame = load_enriched_month(
        year=year,
        month=month,
        limit=limit,
        **loader_arguments,
    )
    return attach_capacity_feature_partitions(frame, feature_dir=capacity_feature_dir)


def _load_training_period(
    *,
    months: tuple[int, ...],
    rows_per_month: int,
    capacity_feature_dir: Path,
    loader_arguments: dict[str, Any],
) -> pd.DataFrame:
    return pd.concat(
        [
            _load_capacity_month(
                year=2024,
                month=month,
                limit=rows_per_month,
                capacity_feature_dir=capacity_feature_dir,
                loader_arguments=loader_arguments,
            )
            for month in months
        ],
        ignore_index=True,
    )


def _joint_columns(name: str) -> tuple[str, str, str]:
    return tuple(f"prob_{name}_{state}" for state in PROBABILITY_STATES)  # type: ignore[return-value]


def _normalize_persisted_joint(
    values: ArrayLike,
    *,
    method: str,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Normalize bounded float32 storage drift and expose the complete adjustment."""

    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or len(probabilities) == 0:
        raise ValueError(f"persisted CC-RTH {method} probabilities must have shape (n, 3)")
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError(
            f"persisted CC-RTH {method} probabilities must be finite and non-negative"
        )
    row_sums = probabilities.sum(axis=1)
    absolute_error = np.abs(row_sums - 1.0)
    maximum_error = float(absolute_error.max())
    if (
        maximum_error > PERSISTED_FLOAT32_SIMPLEX_ACCEPTANCE
        or (row_sums <= 0.0).any()
    ):
        raise ValueError(
            f"persisted CC-RTH {method} probability drift exceeds the float32 "
            f"acceptance bound: {maximum_error}"
        )
    normalized = probabilities / row_sums[:, None]
    maximum_adjustment = float(np.max(np.abs(normalized - probabilities)))
    return np.asarray(normalized, dtype=np.float64), {
        "method": method,
        "rows": len(probabilities),
        "acceptance_bound": PERSISTED_FLOAT32_SIMPLEX_ACCEPTANCE,
        "maximum_absolute_row_sum_error_before_normalization": maximum_error,
        "rows_over_original_1e_8_absolute_tolerance": int(
            (absolute_error > 1e-8).sum()
        ),
        "rows_over_acceptance_bound": int(
            (absolute_error > PERSISTED_FLOAT32_SIMPLEX_ACCEPTANCE).sum()
        ),
        "maximum_absolute_probability_adjustment": maximum_adjustment,
        "normalization": "divide each accepted three-state row by its float64 row sum",
    }


def _set_joint_columns(
    frame: pd.DataFrame,
    name: str,
    probabilities: NDArray[np.float64],
) -> None:
    if probabilities.shape != (len(frame), 3):
        raise ValueError(f"joint probabilities for {name} do not align")
    for index, column in enumerate(_joint_columns(name)):
        frame[column] = probabilities[:, index].astype("float32")


def _get_joint_columns(
    frame: pd.DataFrame,
    name: str,
    *,
    audit_records: list[dict[str, Any]] | None = None,
    role: str = "assembled joint probability columns",
) -> NDArray[np.float64]:
    columns = list(_joint_columns(name))
    storage_dtypes = [str(dtype) for dtype in frame.loc[:, columns].dtypes]
    normalized, audit = _normalize_persisted_joint(
        frame.loc[:, columns].to_numpy(dtype=np.float64),
        method=name,
    )
    if audit_records is not None:
        audit_records.append(
            {
                "role": role,
                "storage_dtypes": storage_dtypes,
                **audit,
            }
        )
    return normalized


def _baseline_selection_reference(
    report_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    report, hash_key = _read_self_hashed(report_path)
    if report.get("status") != "COMPLETE_2024_FLARE24_SELECTION_NOT_CONFIRMATORY":
        raise ValueError("CC-RTH baseline must be the completed 2024 FLARE-24 selection")
    if report.get("outcomes_accessed", {}).get("2025_accessed_by_this_run") is not False:
        raise ValueError("FLARE-24 selection baseline crossed the 2024 information boundary")
    weights = report.get("ensemble_selection", {}).get("selected", {}).get("weights", {})
    if weights.get("rotation_structural") != 1.0 or any(
        float(value) != (1.0 if name == "rotation_structural" else 0.0)
        for name, value in weights.items()
    ):
        raise ValueError(
            "CC-RTH v1 expects the measured FLARE-24 final blend to equal rotation_structural"
        )
    if report.get("selected_variance_multiplier") is not None:
        raise ValueError("CC-RTH v1 expects the measured FLARE-24 no-alignment selection")
    record = dict(report["crossfit_prediction_artifact"])
    path = _verified_artifact(record, role="FLARE-24 2024 cross-fit reference")
    columns = ["sample_id", *_joint_columns("reconciled")]
    reference = pd.read_parquet(path, columns=columns).rename(
        columns={
            old: new
            for old, new in zip(
                _joint_columns("reconciled"),
                _joint_columns("flare24"),
                strict=True,
            )
        }
    )
    if reference["sample_id"].isna().any() or reference["sample_id"].duplicated().any():
        raise ValueError("FLARE-24 cross-fit reference has invalid sample ids")
    normalization_audits: list[dict[str, Any]] = []
    probabilities = _get_joint_columns(
        reference,
        "flare24",
        audit_records=normalization_audits,
        role="checksum-loaded FLARE-24 2024 cross-fit reference",
    )
    for index, column in enumerate(_joint_columns("flare24")):
        reference[column] = probabilities[:, index]
    provenance = {
        "path": report_path.as_posix(),
        "sha256": sha256_file(report_path),
        "self_hash_key": hash_key,
        "self_hash": report[hash_key],
        "crossfit_predictions": record,
        "reference_method": "selected FLARE-24 ensemble/reconciled forecast",
        "probability_normalization_audit": normalization_audits[0],
    }
    return report, reference, provenance


def _bind_baseline_roots(
    report: dict[str, Any],
    roots: dict[str, Path],
) -> list[dict[str, Any]]:
    """Ensure augmented models use the same feature vintages as frozen FLARE-24."""

    records_by_role = {
        str(record.get("role")): dict(record)
        for record in report.get("input_artifacts", [])
    }
    bound: list[dict[str, Any]] = []
    for role, root in roots.items():
        if role not in records_by_role:
            raise ValueError(f"FLARE-24 baseline does not bind required input role: {role}")
        record = records_by_role[role]
        manifest_path = _verified_artifact(record, role=f"FLARE-24 {role} manifest")
        if Path(str(record.get("output_root", ""))).resolve() != root.resolve():
            raise ValueError(f"supplied {role} root differs from frozen FLARE-24 input")
        bound.append(
            {
                "role": role,
                "root": root.as_posix(),
                "manifest": manifest_path.as_posix(),
                "manifest_sha256": record["sha256"],
            }
        )
    return bound


def _baseline_audit_references(
    report_path: Path,
) -> tuple[dict[int, pd.DataFrame], dict[str, Any]]:
    report, hash_key = _read_self_hashed(report_path)
    if report.get("status") != "COMPLETE_2025_FLARE24_RETROSPECTIVE_AUDIT_NOT_BLIND_CONFIRMATION":
        raise ValueError("CC-RTH baseline must be the completed 2025 FLARE-24 audit")
    if int(report.get("evaluation_year", -1)) != 2025:
        raise ValueError("FLARE-24 audit baseline has the wrong year")
    references: dict[int, pd.DataFrame] = {}
    records: list[dict[str, Any]] = []
    normalization_audits: list[dict[str, Any]] = []
    for record_value in report.get("prediction_artifacts", []):
        record = dict(record_value)
        if int(record.get("year", -1)) != 2025:
            continue
        month = int(record["month"])
        path = _verified_artifact(record, role=f"FLARE-24 2025-{month:02d} reference")
        reference = pd.read_parquet(
            path,
            columns=["sample_id", *_joint_columns("reconciled")],
        ).rename(
            columns={
                old: new
                for old, new in zip(
                    _joint_columns("reconciled"),
                    _joint_columns("flare24"),
                    strict=True,
                )
            }
        )
        if reference["sample_id"].isna().any() or reference["sample_id"].duplicated().any():
            raise ValueError(f"FLARE-24 2025-{month:02d} reference has invalid ids")
        probabilities = _get_joint_columns(
            reference,
            "flare24",
            audit_records=normalization_audits,
            role=f"checksum-loaded FLARE-24 2025-{month:02d} reference",
        )
        for index, column in enumerate(_joint_columns("flare24")):
            reference[column] = probabilities[:, index]
        references[month] = reference
        records.append(record)
    if set(references) != set(range(1, 13)):
        raise ValueError("FLARE-24 audit reference does not cover all 2025 months")
    return references, {
        "path": report_path.as_posix(),
        "sha256": sha256_file(report_path),
        "self_hash_key": hash_key,
        "self_hash": report[hash_key],
        "prediction_artifacts": records,
        "probability_normalization_audits": normalization_audits,
        "reference_joint_metrics": report["primary_evaluation"]["methods"]["reconciled"][
            "joint"
        ],
    }


def _join_reference(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    role: str,
) -> pd.DataFrame:
    overlap = sorted(set(_joint_columns("flare24")) & set(frame.columns))
    if overlap:
        raise ValueError(f"{role} frame already contains reference columns: {overlap}")
    result = frame.merge(
        reference,
        on="sample_id",
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if not result["_merge"].eq("both").all():
        missing = int(result["_merge"].ne("both").sum())
        raise ValueError(f"{role} reference omits {missing} requested flights")
    result = result.drop(columns="_merge")
    probabilities = _get_joint_columns(result, "flare24")
    for index, column in enumerate(_joint_columns("flare24")):
        result[column] = probabilities[:, index]
    return result


def _feature_importance_record(model: CapacityCatBoostModel) -> dict[str, Any]:
    names = list(model.estimator.feature_names_)
    values = np.asarray(model.estimator.get_feature_importance(), dtype=np.float64)
    if values.shape != (len(names),):
        raise RuntimeError("CatBoost feature importance did not align with feature names")
    order = np.argsort(-values)
    capacity_mask = np.asarray([name.startswith("ccrth_") for name in names])
    return {
        "importance_type": "CatBoost PredictionValuesChange; descriptive, not causal",
        "total_features": len(names),
        "capacity_feature_importance_fraction": float(values[capacity_mask].sum() / values.sum()),
        "top_40": [
            {"feature": names[index], "importance": float(values[index])}
            for index in order[:40]
        ],
        "top_capacity_40": [
            {"feature": names[index], "importance": float(values[index])}
            for index in order
            if capacity_mask[index]
        ][:40],
    }


def _incremental_comparisons(
    frame: pd.DataFrame,
    methods: dict[str, NDArray[np.float64]],
    *,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    observed = frame["joint_label_observed"].eq(1).to_numpy()
    labels = frame.loc[observed, "disruption_state"].to_numpy(dtype=np.int64)
    clusters = frame.loc[observed, "FlightDate"].to_numpy()
    losses = {
        name: joint_loss_rows(labels, probabilities[observed])
        for name, probabilities in methods.items()
    }
    comparisons: list[dict[str, Any]] = []
    pairs: list[tuple[str, str]] = [
        (right, left)
        for left, right in pairwise(CAPACITY_CANDIDATES)
    ]
    pairs.extend((name, "flare24") for name in ("global_simplex", "capacity_gated_simplex"))
    for candidate, reference in pairs:
        candidate_log, candidate_brier = losses[candidate]
        reference_log, reference_brier = losses[reference]
        log_interval = paired_cluster_mean_difference(
            candidate_log,
            reference_log,
            clusters,
            repetitions=repetitions,
            seed=seed,
        )
        brier_interval = paired_cluster_mean_difference(
            candidate_brier,
            reference_brier,
            clusters,
            repetitions=repetitions,
            seed=seed,
        )
        comparisons.append(
            {
                "candidate": candidate,
                "reference": reference,
                "joint_log_loss_difference": asdict(log_interval),
                "multiclass_brier_difference": asdict(brier_interval),
                "negative_favors_candidate": True,
            }
        )
    return comparisons


def _monthly_scores(
    frame: pd.DataFrame,
    methods: dict[str, NDArray[np.float64]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    months = pd.to_numeric(frame["Month"], errors="raise").astype(int)
    observed = frame["joint_label_observed"].eq(1).to_numpy()
    labels = frame["disruption_state"].to_numpy(dtype=np.int64)
    for month in sorted(months.unique()):
        mask = months.eq(month).to_numpy() & observed
        metrics: dict[str, Any] = {}
        for name, probabilities in methods.items():
            log_rows, brier_rows = joint_loss_rows(labels[mask], probabilities[mask])
            metrics[name] = {
                "joint_log_loss": float(log_rows.mean()),
                "multiclass_brier": float(brier_rows.mean()),
            }
        records.append({"month": int(month), "n": int(mask.sum()), "methods": metrics})
    return records


def _joint_point_scores(
    frame: pd.DataFrame,
    methods: dict[str, NDArray[np.float64]],
) -> dict[str, Any]:
    observed = frame["joint_label_observed"].eq(1).to_numpy()
    labels = frame.loc[observed, "disruption_state"].to_numpy(dtype=np.int64)
    result: dict[str, Any] = {}
    for name, probabilities in methods.items():
        log_rows, brier_rows = joint_loss_rows(labels, probabilities[observed])
        result[name] = {
            "n": len(labels),
            "joint_log_loss": float(log_rows.mean()),
            "multiclass_brier": float(brier_rows.mean()),
        }
    return result


def _regime_scores(
    frame: pd.DataFrame,
    methods: dict[str, NDArray[np.float64]],
    selection: dict[str, Any],
) -> list[dict[str, Any]]:
    regimes = capacity_regimes(frame[GATING_FEATURE], selection["cutpoints"])
    observed = frame["joint_label_observed"].eq(1).to_numpy()
    labels = frame["disruption_state"].to_numpy(dtype=np.int64)
    records: list[dict[str, Any]] = []
    for regime in GATING_LABELS:
        mask = (regimes == regime) & observed
        if not mask.any():
            continue
        metrics: dict[str, Any] = {}
        for name, probabilities in methods.items():
            log_rows, brier_rows = joint_loss_rows(labels[mask], probabilities[mask])
            metrics[name] = {
                "joint_log_loss": float(log_rows.mean()),
                "multiclass_brier": float(brier_rows.mean()),
            }
        records.append({"regime": regime, "n": int(mask.sum()), "methods": metrics})
    return records


def _airport_scores(
    frame: pd.DataFrame,
    methods: dict[str, NDArray[np.float64]],
) -> list[dict[str, Any]]:
    observed = frame["joint_label_observed"].eq(1).to_numpy()
    labels = frame.loc[observed, "disruption_state"].to_numpy(dtype=np.int64)
    records: list[dict[str, Any]] = []
    for role, column in (("origin", "Origin"), ("destination", "Dest")):
        airports = frame.loc[observed, column].astype(str).reset_index(drop=True)
        grouped_by_method: dict[str, pd.DataFrame] = {}
        for name, probabilities in methods.items():
            log_rows, brier_rows = joint_loss_rows(labels, probabilities[observed])
            values = pd.DataFrame(
                {
                    "airport": airports,
                    "joint_log_loss": log_rows,
                    "multiclass_brier": brier_rows,
                }
            )
            grouped_by_method[name] = (
                values.groupby("airport", observed=True, sort=True)
                .agg(
                    n=("joint_log_loss", "size"),
                    joint_log_loss=("joint_log_loss", "mean"),
                    multiclass_brier=("multiclass_brier", "mean"),
                )
                .reset_index()
            )
        baseline = grouped_by_method["flare24"].set_index("airport")
        for name, grouped in grouped_by_method.items():
            for row in grouped.itertuples(index=False):
                records.append(
                    {
                        "role": role,
                        "airport": row.airport,
                        "method": name,
                        "n": int(row.n),
                        "joint_log_loss": float(row.joint_log_loss),
                        "joint_log_loss_delta_vs_flare24": float(
                            row.joint_log_loss
                            - baseline.loc[row.airport, "joint_log_loss"]
                        ),
                        "multiclass_brier": float(row.multiclass_brier),
                        "multiclass_brier_delta_vs_flare24": float(
                            row.multiclass_brier
                            - baseline.loc[row.airport, "multiclass_brier"]
                        ),
                    }
                )
    return records


def run_capacity_hypergraph_study(
    *,
    protocol_path: Path,
    baseline_selection_report_path: Path,
    baseline_audit_report_path: Path,
    capacity_manifest_path: Path,
    capacity_validation_path: Path,
    census_dir: Path,
    recent_dir: Path,
    flight_recent_dir: Path,
    graph_dir: Path,
    weather_feature_dir: Path,
    rotation_feature_dir: Path,
    capacity_feature_dir: Path,
    run_dir: Path,
    output_path: Path,
    recovery_model_record_path: Path | None = None,
    rows_per_train_month: int = 125_000,
    validation_limit: int = 250_000,
    bootstrap_repetitions: int = 2_000,
    seed: int = 20260903,
) -> dict[str, Any]:
    """Fit, lock, and retrospectively evaluate the complete CC-RTH evolution."""

    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse CC-RTH run directory: {run_dir}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite CC-RTH report: {output_path}")
    if rows_per_train_month < 10_000 or validation_limit < 10_000:
        raise ValueError("CC-RTH research samples must contain at least 10,000 rows per period")
    if bootstrap_repetitions < 100:
        raise ValueError("CC-RTH requires at least 100 date-cluster bootstrap repetitions")
    started = time.perf_counter()
    implementation_provenance_at_start = capture_provenance(
        _frozen_capacity_implementation_files()
    )
    protocol, protocol_record = _load_protocol(
        protocol_path,
        rows_per_train_month=rows_per_train_month,
        validation_limit=validation_limit,
        bootstrap_repetitions=bootstrap_repetitions,
        seed=seed,
    )
    baseline_report, baseline_crossfit, baseline_selection_provenance = (
        _baseline_selection_reference(baseline_selection_report_path)
    )
    if int(baseline_report.get("rows_per_train_month", -1)) != rows_per_train_month:
        raise ValueError("CC-RTH training sample differs from the frozen FLARE-24 baseline")
    if int(baseline_report.get("validation_limit", -1)) != validation_limit:
        raise ValueError("CC-RTH validation sample differs from the frozen FLARE-24 baseline")
    if int(baseline_report.get("seed", -1)) != seed:
        raise ValueError("CC-RTH sampling seed differs from the frozen FLARE-24 baseline")
    capacity_manifest, capacity_manifest_records = _verify_capacity_manifest(
        capacity_manifest_path,
        feature_dir=capacity_feature_dir,
    )
    capacity_validation_record = _verify_capacity_validation(
        capacity_validation_path,
        capacity_manifest_path=capacity_manifest_path,
        capacity_manifest=capacity_manifest,
    )
    recovery_models, recovery_record = _load_recovery_model_record(
        recovery_model_record_path,
        protocol_sha256=protocol_record["sha256"],
        capacity_manifest_self_hash=capacity_manifest["manifest_sha256"],
    )
    bound_baseline_inputs = _bind_baseline_roots(
        baseline_report,
        {
            "census": census_dir,
            "closed-left HMOP": recent_dir,
            "closed-left flight history": flight_recent_dir,
            "CL-SGMP": graph_dir,
            "FLARE cutoff-coherent weather": weather_feature_dir,
            "FLARE latent rotations": rotation_feature_dir,
        },
    )
    run_dir.mkdir(parents=True)
    loader_arguments: dict[str, Any] = {
        "census_dir": census_dir,
        "recent_dir": recent_dir,
        "flight_recent_dir": flight_recent_dir,
        "graph_dir": graph_dir,
        "weather_feature_dir": weather_feature_dir,
        "rotation_feature_dir": rotation_feature_dir,
        "seed": seed,
    }

    print("loading CC-RTH January-August 2024 training sample", flush=True)
    train = _load_training_period(
        months=tuple(range(1, 9)),
        rows_per_month=rows_per_train_month,
        capacity_feature_dir=capacity_feature_dir,
        loader_arguments=loader_arguments,
    )
    print("loading CC-RTH September 2024 early-stopping sample", flush=True)
    validation = _load_capacity_month(
        year=2024,
        month=9,
        limit=validation_limit,
        capacity_feature_dir=capacity_feature_dir,
        loader_arguments=loader_arguments,
    )
    if pd.to_datetime(train["FlightDate"]).max() >= pd.to_datetime(
        validation["FlightDate"]
    ).min():
        raise AssertionError("CC-RTH training and early-stopping periods overlap")
    pilot_train = train.loc[
        pd.to_datetime(train["FlightDate"], errors="raise").le(
            pd.Timestamp(EARLY_STOP_TRAIN_END)
        )
    ].reset_index(drop=True)
    pilot_gap_days = (
        pd.to_datetime(validation["FlightDate"], errors="raise").min()
        - pd.to_datetime(pilot_train["FlightDate"], errors="raise").max()
    ).days - 1
    if pilot_gap_days != PURGE_DAYS:
        raise RuntimeError("CC-RTH early-stopping boundary does not have its declared purge")
    flare_features = tuple(baseline_report["feature_sets"]["rotation_structural"])
    missing_flare = sorted(set(flare_features) - set(train.columns))
    if missing_flare:
        raise ValueError(f"CC-RTH training data omit frozen FLARE features: {missing_flare}")
    usable_capacity = select_usable_capacity_features(pilot_train, CAPACITY_ALL_FEATURES)
    feature_sets: dict[CapacityCandidate, tuple[str, ...]] = {
        candidate: tuple(
            feature
            for feature in CAPACITY_CANDIDATE_FEATURES[candidate]
            if feature in usable_capacity
        )
        for candidate in CAPACITY_CANDIDATES
    }
    if any(
        not set(feature_sets[left]).issubset(feature_sets[right])
        for left, right in pairwise(CAPACITY_CANDIDATES)
    ):
        raise RuntimeError("usable CC-RTH feature sets are not nested")

    model_artifacts_by_key: dict[
        CapacityCandidate, dict[str, dict[str, Any]]
    ] = {
        candidate: {} for candidate in AUGMENTED_CANDIDATES
    }
    model_records: list[dict[str, Any]] = []
    importance_records: list[dict[str, Any]] = []
    for task in TASKS:
        pilot_train_task, pilot_train_labels = task_frame(pilot_train, task)
        refit_train_task, refit_train_labels = task_frame(train, task)
        valid_task, valid_labels = task_frame(validation, task)
        for candidate in AUGMENTED_CANDIDATES:
            fit_started = time.perf_counter()
            parameters = {**MODEL_PARAMETERS[task], "random_seed": seed}
            recovery_key = (candidate, task)
            if recovery_models:
                recovered = recovery_models[recovery_key]
                model = _load_capacity_model(recovered)
                if (
                    model.task != task
                    or model.flare_features != flare_features
                    or model.capacity_features != feature_sets[candidate]
                ):
                    raise ValueError(
                        f"recovered CC-RTH model contract differs: {candidate}/{task}"
                    )
                refit_iterations = int(recovered["refit_iterations"])
                if int(model.estimator.tree_count_) != refit_iterations:
                    raise ValueError(
                        f"recovered CC-RTH tree count differs: {candidate}/{task}"
                    )
                refit_parameters = {**parameters, "iterations": refit_iterations}
                artifact = {
                    key: recovered[key] for key in ("path", "bytes", "sha256")
                }
                model_artifacts_by_key[candidate][task] = artifact
                model_records.append(
                    {
                        "candidate": candidate,
                        "task": task,
                        "early_stopping_training_rows": len(pilot_train_task),
                        "early_stopping_validation_rows": len(valid_task),
                        "early_stopping_positive_training_rows": int(
                            pilot_train_labels.sum()
                        ),
                        "early_stopping_positive_validation_rows": int(
                            valid_labels.sum()
                        ),
                        "early_stopping_best_iteration": int(
                            recovered["early_stopping_best_iteration"]
                        ),
                        "refit_training_rows": len(refit_train_task) + len(valid_task),
                        "refit_positive_rows": int(
                            refit_train_labels.sum() + valid_labels.sum()
                        ),
                        "refit_through_date": "2024-09-30",
                        "refit_iterations": refit_iterations,
                        "flare_feature_count": len(flare_features),
                        "capacity_feature_count": len(feature_sets[candidate]),
                        "fit_seconds": None,
                        "parameters": refit_parameters,
                        "artifact_origin": RECOVERY_ARTIFACT_ORIGIN,
                        **artifact,
                    }
                )
                importance_records.append(
                    {
                        "candidate": candidate,
                        "task": task,
                        **_feature_importance_record(model),
                    }
                )
                print(
                    f"recovered checksum-verified CC-RTH {candidate} {task} "
                    f"with {refit_iterations} trees",
                    flush=True,
                )
                del model
                gc.collect()
                continue
            print(f"early-stopping CC-RTH {candidate} {task}", flush=True)
            pilot = fit_capacity_catboost(
                pilot_train_task,
                pilot_train_labels,
                task=task,
                flare_features=flare_features,
                capacity_features=feature_sets[candidate],
                params=parameters,
                validation_frame=valid_task,
                validation_labels=valid_labels,
            )
            best_iteration = int(pilot.estimator.get_best_iteration())
            refit_iterations = (
                int(MODEL_PARAMETERS[task]["iterations"])
                if best_iteration < 0
                else best_iteration + 1
            )
            del pilot
            gc.collect()
            refit_frame = pd.concat([refit_train_task, valid_task], ignore_index=True)
            refit_labels = np.concatenate([refit_train_labels, valid_labels])
            refit_parameters = {**parameters, "iterations": refit_iterations}
            print(
                f"refitting CC-RTH {candidate} {task} through September "
                f"for {refit_iterations} iterations",
                flush=True,
            )
            model = fit_capacity_catboost(
                refit_frame,
                refit_labels,
                task=task,
                flare_features=flare_features,
                capacity_features=feature_sets[candidate],
                params=refit_parameters,
            )
            artifact = _atomic_joblib(
                model,
                run_dir / "models" / f"{candidate}_{task}.joblib",
            )
            model_artifacts_by_key[candidate][task] = artifact
            model_records.append(
                {
                    "candidate": candidate,
                    "task": task,
                    "early_stopping_training_rows": len(pilot_train_task),
                    "early_stopping_validation_rows": len(valid_task),
                    "early_stopping_positive_training_rows": int(
                        pilot_train_labels.sum()
                    ),
                    "early_stopping_positive_validation_rows": int(valid_labels.sum()),
                    "early_stopping_best_iteration": best_iteration,
                    "refit_training_rows": len(refit_frame),
                    "refit_positive_rows": int(refit_labels.sum()),
                    "refit_through_date": "2024-09-30",
                    "refit_iterations": refit_iterations,
                    "flare_feature_count": len(flare_features),
                    "capacity_feature_count": len(feature_sets[candidate]),
                    "fit_seconds": time.perf_counter() - fit_started,
                    "parameters": refit_parameters,
                    **artifact,
                }
            )
            importance_records.append(
                {
                    "candidate": candidate,
                    "task": task,
                    **_feature_importance_record(model),
                }
            )
            del model, refit_frame, refit_labels
            gc.collect()
        del (
            pilot_train_task,
            pilot_train_labels,
            refit_train_task,
            refit_train_labels,
            valid_task,
            valid_labels,
        )
        gc.collect()
    training_rows = len(train)
    early_stopping_training_rows = len(pilot_train)
    validation_rows = len(validation)
    del train, pilot_train, validation
    gc.collect()

    raw_selection_records: list[dict[str, Any]] = []
    selection_parts: list[pd.DataFrame] = []
    for month in (10, 11, 12):
        print(f"predicting CC-RTH 2024-{month:02d} selection partition", flush=True)
        frame = _load_capacity_month(
            year=2024,
            month=month,
            limit=None,
            capacity_feature_dir=capacity_feature_dir,
            loader_arguments=loader_arguments,
        )
        prediction = frame.loc[:, list(PREDICTION_ID_COLUMNS)].copy()
        prediction[GATING_FEATURE] = pd.to_numeric(
            frame[GATING_FEATURE], errors="raise"
        ).astype("float32")
        for candidate in AUGMENTED_CANDIDATES:
            for task in TASKS:
                model = _load_capacity_model(model_artifacts_by_key[candidate][task])
                prediction[f"raw_{candidate}_{task}"] = model.predict_proba(frame).astype(
                    "float32"
                )
                del model
                gc.collect()
        artifact = _atomic_parquet(
            prediction,
            run_dir / "predictions" / f"raw_selection_2024_{month:02d}.parquet",
        )
        raw_selection_records.append({"year": 2024, "month": month, **artifact})
        selection_parts.append(prediction)
        del frame, prediction
        gc.collect()
    selection = pd.concat(selection_parts, ignore_index=True)
    del selection_parts

    calibrators: dict[CapacityCandidate, dict[str, BinaryCalibrator]] = {
        candidate: {} for candidate in AUGMENTED_CANDIDATES
    }
    calibration_records: list[dict[str, Any]] = []
    common_crossfit = np.ones(len(selection), dtype=np.bool_)
    for candidate in AUGMENTED_CANDIDATES:
        for task in TASKS:
            selected = select_purged_forward_calibration(
                selection,
                selection[f"raw_{candidate}_{task}"].to_numpy(dtype=np.float64),
                task=task,
            )
            calibrators[candidate][task] = selected.final_calibrator
            selection[f"xfit_{candidate}_{task}"] = selected.crossfit_probabilities.astype(
                "float32"
            )
            common_crossfit &= selected.crossfit_mask
            artifact = _atomic_joblib(
                selected.final_calibrator,
                run_dir / "calibrators" / f"{candidate}_{task}_{selected.method}.joblib",
            )
            calibration_records.append(
                {
                    "candidate": candidate,
                    "task": task,
                    "selected_method": selected.method,
                    "final_fit_through": selected.final_fit_through,
                    "candidate_methods": list(selected.candidate_records),
                    **artifact,
                }
            )
    if not common_crossfit.any():
        raise RuntimeError("CC-RTH calibration produced no common forward predictions")
    crossfit = selection.loc[common_crossfit].reset_index(drop=True)
    crossfit = _join_reference(crossfit, baseline_crossfit, role="Q4 cross-fit")
    methods: dict[str, NDArray[np.float64]] = {
        "flare24": _get_joint_columns(crossfit, "flare24")
    }
    for candidate in AUGMENTED_CANDIDATES:
        methods[candidate] = hurdle_joint_probabilities(
            crossfit[f"xfit_{candidate}_cancellation"],
            crossfit[f"xfit_{candidate}_delay"],
        )
    observed = crossfit["joint_label_observed"].eq(1).to_numpy()
    labels = crossfit.loc[observed, "disruption_state"].to_numpy(dtype=np.int64)
    simplex_selection = select_probability_simplex(
        labels,
        {candidate: methods[candidate][observed] for candidate in BLEND_CANDIDATES},
    )
    methods["global_simplex"] = blend_probabilities(
        {candidate: methods[candidate] for candidate in BLEND_CANDIDATES},
        cast(dict[str, float], simplex_selection["weights"]),
    )
    gated_selection = select_capacity_gated_simplex(
        labels,
        {candidate: methods[candidate][observed] for candidate in BLEND_CANDIDATES},
        crossfit.loc[observed, GATING_FEATURE].to_numpy(dtype=np.float64),
        cutpoint_values=crossfit[GATING_FEATURE].to_numpy(dtype=np.float64),
    )
    methods["capacity_gated_simplex"] = apply_capacity_gated_simplex(
        {candidate: methods[candidate] for candidate in BLEND_CANDIDATES},
        crossfit[GATING_FEATURE].to_numpy(dtype=np.float64),
        gated_selection,
    )
    selection_evaluation = evaluate_joint_probabilities(
        crossfit,
        cast(dict[str, ArrayLike], methods),
        reference_method="flare24",
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=seed,
    )
    selection_incremental = _incremental_comparisons(
        crossfit,
        methods,
        repetitions=bootstrap_repetitions,
        seed=seed,
    )
    selected_method = min(
        selection_evaluation["methods"],
        key=lambda name: (
            float(selection_evaluation["methods"][name]["joint"]["log_loss"]),
            list(selection_evaluation["methods"]).index(name),
        ),
    )
    crossfit_artifact_frame = crossfit.loc[
        :, [*PREDICTION_ID_COLUMNS, GATING_FEATURE]
    ].copy()
    for name, values in methods.items():
        _set_joint_columns(crossfit_artifact_frame, name, values)
    crossfit_artifact = _atomic_parquet(
        crossfit_artifact_frame,
        run_dir / "predictions" / "selection_2024_crossfit_joint.parquet",
    )

    implementation_provenance_at_lock = _verify_implementation_unchanged(
        implementation_provenance_at_start,
        phase="method lock",
    )
    method_lock: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED_CCRTH_V1_BEFORE_THIS_RUN_OPENED_2025_OUTCOMES_NOT_BLIND_CONFIRMATION",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "CC-RTH-v1",
        "protocol": protocol_record,
        "capacity_graph_validation_gate": capacity_validation_record,
        "baseline": baseline_selection_provenance,
        "bound_baseline_inputs": bound_baseline_inputs,
        "feature_sets": {name: list(values) for name, values in feature_sets.items()},
        "flare_features": list(flare_features),
        "model_artifacts": model_records,
        "model_recovery": recovery_record,
        "calibrator_artifacts": calibration_records,
        "global_simplex": simplex_selection,
        "capacity_gated_simplex": gated_selection,
        "selected_method_by_2024_forward_score": selected_method,
        "graph_boundary_purge_days": PURGE_DAYS,
        "periods": {
            "loaded_training": ["2024-01-01", "2024-08-31"],
            "early_stopping_training": ["2024-01-01", EARLY_STOP_TRAIN_END],
            "early_stopping_boundary_purge": ["2024-08-30", "2024-08-31"],
            "early_stopping": ["2024-09-01", "2024-09-30"],
            "fixed_iteration_refit": ["2024-01-01", "2024-09-30"],
            "calibration_and_selection": ["2024-10-01", "2024-12-31"],
            "model_to_calibration_boundary_purge": ["2024-10-01", "2024-10-02"],
            "purged_calibration_folds": [list(fold) for fold in PURGED_CALIBRATION_FOLDS],
            "final_calibration_fit_through": FINAL_CALIBRATION_END,
            "primary_retrospective_evaluation": [PRIMARY_AUDIT_START, PRIMARY_AUDIT_END],
        },
        "outcomes_accessed_by_this_extension_before_lock": {
            "years": [2024],
            "maximum_date": "2024-12-31",
            "2025_loaded": False,
            "2026_loaded": False,
        },
        "epistemic_status": (
            "The implementation is locked before this execution opens 2025 outcomes, but "
            "the research direction was formed after earlier 2025 FLARE results were known; "
            "the 2025 evaluation is therefore retrospective, not blind confirmation."
        ),
        "versions": {
            "python": platform.python_version(),
            "catboost": version("catboost"),
            "joblib": version("joblib"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
        },
        "provenance": implementation_provenance_at_lock,
    }
    audit_gap_days = (
        pd.Timestamp(PRIMARY_AUDIT_START) - pd.Timestamp(FINAL_CALIBRATION_END)
    ).days - 1
    if audit_gap_days != PURGE_DAYS:
        raise RuntimeError("CC-RTH year boundary does not have its declared purge")
    method_lock["manifest_sha256"] = canonical_json_sha256(method_lock)
    method_lock_path = run_dir / "method_lock.json"
    write_canonical_json(method_lock_path, method_lock)
    print(
        f"locked CC-RTH method after 2024 selection; selected={selected_method}",
        flush=True,
    )

    # This is the first point at which the extension opens any 2025 outcome-bearing artifact.
    baseline_audit, baseline_audit_provenance = _baseline_audit_references(
        baseline_audit_report_path
    )
    audit_records: list[dict[str, Any]] = []
    audit_parts: list[pd.DataFrame] = []
    for month in range(1, 13):
        print(f"predicting and scoring CC-RTH retrospective 2025-{month:02d}", flush=True)
        frame = _load_capacity_month(
            year=2025,
            month=month,
            limit=None,
            capacity_feature_dir=capacity_feature_dir,
            loader_arguments=loader_arguments,
        )
        prediction = frame.loc[:, list(PREDICTION_ID_COLUMNS)].copy()
        prediction[GATING_FEATURE] = pd.to_numeric(
            frame[GATING_FEATURE], errors="raise"
        ).astype("float32")
        prediction = _join_reference(
            prediction,
            baseline_audit[month],
            role=f"2025-{month:02d}",
        )
        month_methods: dict[str, NDArray[np.float64]] = {
            "flare24": _get_joint_columns(prediction, "flare24")
        }
        for candidate in AUGMENTED_CANDIDATES:
            raw: dict[str, NDArray[np.float64]] = {}
            calibrated: dict[str, NDArray[np.float64]] = {}
            for task in TASKS:
                model = _load_capacity_model(model_artifacts_by_key[candidate][task])
                raw[task] = model.predict_proba(frame)
                del model
                gc.collect()
                calibrated[task] = calibrators[candidate][task].predict(raw[task])
                prediction[f"raw_{candidate}_{task}"] = raw[task].astype("float32")
                prediction[f"calibrated_{candidate}_{task}"] = calibrated[task].astype(
                    "float32"
                )
            month_methods[candidate] = hurdle_joint_probabilities(
                calibrated["cancellation"], calibrated["delay"]
            )
            _set_joint_columns(prediction, candidate, month_methods[candidate])
        month_methods["global_simplex"] = blend_probabilities(
            {candidate: month_methods[candidate] for candidate in BLEND_CANDIDATES},
            cast(dict[str, float], simplex_selection["weights"]),
        )
        month_methods["capacity_gated_simplex"] = apply_capacity_gated_simplex(
            {candidate: month_methods[candidate] for candidate in BLEND_CANDIDATES},
            prediction[GATING_FEATURE].to_numpy(dtype=np.float64),
            gated_selection,
        )
        _set_joint_columns(prediction, "global_simplex", month_methods["global_simplex"])
        _set_joint_columns(
            prediction,
            "capacity_gated_simplex",
            month_methods["capacity_gated_simplex"],
        )
        artifact = _atomic_parquet(
            prediction,
            run_dir / "predictions" / f"retrospective_2025_{month:02d}.parquet",
        )
        audit_records.append({"year": 2025, "month": month, **artifact})
        audit_parts.append(prediction)
        del frame, prediction, month_methods
        gc.collect()
    audit_full = pd.concat(audit_parts, ignore_index=True)
    del audit_parts
    full_year_probability_normalization: list[dict[str, Any]] = []
    audit_methods_full = {
        name: _get_joint_columns(
            audit_full,
            name,
            audit_records=full_year_probability_normalization,
            role="assembled persisted 2025 monthly prediction columns",
        )
        for name in (
            *CAPACITY_CANDIDATES,
            "global_simplex",
            "capacity_gated_simplex",
        )
    }
    full_year_scores = _joint_point_scores(audit_full, audit_methods_full)
    audit_dates = pd.to_datetime(audit_full["FlightDate"], errors="raise")
    primary_mask = audit_dates.between(
        pd.Timestamp(PRIMARY_AUDIT_START), pd.Timestamp(PRIMARY_AUDIT_END)
    ).to_numpy()
    audit = audit_full.loc[primary_mask].reset_index(drop=True)
    audit_methods = {
        name: probabilities[primary_mask]
        for name, probabilities in audit_methods_full.items()
    }
    audit_evaluation = evaluate_joint_probabilities(
        audit,
        cast(dict[str, ArrayLike], audit_methods),
        reference_method="flare24",
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=seed + 10_000,
    )
    audit_incremental = _incremental_comparisons(
        audit,
        audit_methods,
        repetitions=bootstrap_repetitions,
        seed=seed + 10_000,
    )
    reference_expected = baseline_audit_provenance["reference_joint_metrics"]
    reference_observed = {
        "n": full_year_scores["flare24"]["n"],
        "log_loss": full_year_scores["flare24"]["joint_log_loss"],
        "multiclass_brier": full_year_scores["flare24"]["multiclass_brier"],
    }
    for metric in ("log_loss", "multiclass_brier"):
        if not np.isclose(
            float(reference_observed[metric]),
            float(reference_expected[metric]),
            rtol=0.0,
            atol=2e-8,
        ):
            raise RuntimeError(f"CC-RTH baseline reproduction failed for {metric}")
    del audit_full, audit_methods_full
    gc.collect()

    implementation_provenance_at_report = _verify_implementation_unchanged(
        implementation_provenance_at_start,
        phase="final report",
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_CCRTH_2024_SELECTION_2025_RETROSPECTIVE_EVALUATION",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "Capacity-Conditioned Resource-Time Flight Hypergraph (CC-RTH-v1)",
        "protocol": protocol_record,
        "candidate_profile": capacity_candidate_profile(),
        "periods": method_lock["periods"],
        "primary_evaluation_period": [PRIMARY_AUDIT_START, PRIMARY_AUDIT_END],
        "full_year_descriptive_period": ["2025-01-01", "2025-12-31"],
        "loaded_training_rows": training_rows,
        "early_stopping_training_rows_after_purge": early_stopping_training_rows,
        "loaded_validation_rows": validation_rows,
        "rows_per_train_month": rows_per_train_month,
        "validation_limit": validation_limit,
        "feature_sets": {name: list(values) for name, values in feature_sets.items()},
        "feature_counts": {name: len(values) for name, values in feature_sets.items()},
        "flare_feature_count": len(flare_features),
        "capacity_manifest": capacity_manifest_records[0],
        "capacity_graph_validation_gate": capacity_validation_record,
        "capacity_graph_totals": {
            "flights": int(capacity_manifest["rows"]),
            "resource_nodes": int(capacity_manifest["resource_nodes"]),
            "incidence_edges": int(capacity_manifest["incidence_edges"]),
            "rotation_edges": int(capacity_manifest["rotation_edges"]),
        },
        "capacity_feature_nonmissing_fraction": capacity_manifest[
            "feature_nonmissing_fraction"
        ],
        "capacity_graph_diagnostics": capacity_manifest["diagnostics"],
        "bound_baseline_inputs": bound_baseline_inputs,
        "baseline_selection": baseline_selection_provenance,
        "baseline_audit": baseline_audit_provenance,
        "probability_normalization_audit": {
            "selection_reference": baseline_selection_provenance[
                "probability_normalization_audit"
            ],
            "retrospective_reference_partitions": baseline_audit_provenance[
                "probability_normalization_audits"
            ],
            "retrospective_assembled_methods": full_year_probability_normalization,
        },
        "model_recovery": recovery_record,
        "model_artifacts": model_records,
        "feature_importance": importance_records,
        "raw_selection_prediction_artifacts": raw_selection_records,
        "calibration_artifacts": calibration_records,
        "selection_crossfit_prediction_artifact": crossfit_artifact,
        "global_simplex_selection": simplex_selection,
        "capacity_gated_simplex_selection": gated_selection,
        "selected_method_by_2024_forward_score": selected_method,
        "method_lock": {
            "path": method_lock_path.as_posix(),
            "sha256": sha256_file(method_lock_path),
            "self_hash": method_lock["manifest_sha256"],
        },
        "selection_evaluation": selection_evaluation,
        "selection_incremental_comparisons": selection_incremental,
        "retrospective_prediction_artifacts": audit_records,
        "retrospective_evaluation": audit_evaluation,
        "retrospective_full_year_descriptive_joint_scores": full_year_scores,
        "retrospective_incremental_comparisons": audit_incremental,
        "retrospective_monthly_scores": _monthly_scores(audit, audit_methods),
        "retrospective_capacity_regime_scores": _regime_scores(
            audit, audit_methods, gated_selection
        ),
        "retrospective_airport_scores": _airport_scores(audit, audit_methods),
        "baseline_reproduction": {
            "expected": reference_expected,
            "observed": reference_observed,
            "absolute_tolerance": 2e-8,
            "passed": True,
        },
        "bootstrap_repetitions": bootstrap_repetitions,
        "seed": seed,
        "outcomes_accessed": {
            "selection_years": [2024],
            "retrospective_evaluation_years": [2025],
            "2026_accessed": False,
        },
        "confirmation_gate": {
            "year": 2026,
            "opened": False,
        },
        "epistemic_status": method_lock["epistemic_status"],
        "frozen_protocol_status": protocol["identity"]["status"],
        "claim_limits": [
            "The 2025 evidence is retrospective and not a blind confirmation.",
            "BTS final schedules proxy a T-24 schedule snapshot.",
            "Prior-year frontiers are empirical scheduling envelopes, not physical AAR/ADR.",
            "NASR describes static runway geometry, not realised active configurations.",
            "Queue, overload, recovery, marginal-load, and shadow-price values are model proxies, not observed airport states.",
            "Feature importance and stress-stratum differences are descriptive, not causal.",
            "The primary 2025 score spans January 3 through December 29: two dates are embargoed at the left boundary and two dates are excluded at the right boundary because 2026 schedule context remains unopened. Full-year point scores are descriptive continuity values.",
        ],
        "versions": {
            "python": platform.python_version(),
            "catboost": version("catboost"),
            "joblib": version("joblib"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
        },
        "provenance": implementation_provenance_at_report,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(output_path, report)
    write_canonical_json(run_dir / "run_manifest.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline-selection-report", type=Path, required=True)
    parser.add_argument("--baseline-audit-report", type=Path, required=True)
    parser.add_argument("--capacity-manifest", type=Path, required=True)
    parser.add_argument("--capacity-validation", type=Path, required=True)
    parser.add_argument("--census-dir", type=Path, required=True)
    parser.add_argument("--recent-dir", type=Path, required=True)
    parser.add_argument("--flight-recent-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--weather-feature-dir", type=Path, required=True)
    parser.add_argument("--rotation-feature-dir", type=Path, required=True)
    parser.add_argument("--capacity-feature-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--recovery-model-record",
        type=Path,
        help=(
            "checksum-bound failure record for a complete set of 2024-only final "
            "models; omitted for a fresh fit"
        ),
    )
    parser.add_argument("--rows-per-train-month", type=int, default=125_000)
    parser.add_argument("--validation-limit", type=int, default=250_000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_capacity_hypergraph_study(
        protocol_path=args.protocol,
        baseline_selection_report_path=args.baseline_selection_report,
        baseline_audit_report_path=args.baseline_audit_report,
        capacity_manifest_path=args.capacity_manifest,
        capacity_validation_path=args.capacity_validation,
        census_dir=args.census_dir,
        recent_dir=args.recent_dir,
        flight_recent_dir=args.flight_recent_dir,
        graph_dir=args.graph_dir,
        weather_feature_dir=args.weather_feature_dir,
        rotation_feature_dir=args.rotation_feature_dir,
        capacity_feature_dir=args.capacity_feature_dir,
        run_dir=args.run_dir,
        output_path=args.output,
        recovery_model_record_path=args.recovery_model_record,
        rows_per_train_month=args.rows_per_train_month,
        validation_limit=args.validation_limit,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_method": result["selected_method_by_2024_forward_score"],
                "retrospective_joint_scores": {
                    name: metrics["joint"]
                    for name, metrics in result["retrospective_evaluation"][
                        "methods"
                    ].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
