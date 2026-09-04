"""Run the BC-POT-R two-view retrospective research experiment.

The experiment fits only on 2024, selects calibration, ensembling, and an optional
decision-only accuracy rule on purged forward Q4 2024 predictions, and then evaluates
2025 retrospectively. Earlier aggregate 2025 results were known when this architecture
was designed, so this is explicitly redevelopment evidence rather than confirmation.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
import tomllib
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .bootstrap import paired_cluster_mean_difference
from .flare_boundary_contracts import (
    BOUNDARY_FULL_FEATURES,
    BOUNDARY_OBSERVATION_FEATURES,
    BOUNDARY_RESIDUAL_BY_SOURCE,
    BOUNDARY_RESIDUAL_FEATURES,
)
from .flare_boundary_modeling import (
    BOUNDARY_GPU_TRAINING_PARAMETERS,
    BoundaryCandidate,
    BoundaryCatBoostModel,
    attach_boundary_views,
    boundary_model_input_columns,
    fit_boundary_catboost,
    select_usable_boundary_features,
)
from .flare_capacity_contracts import CAPACITY_ALL_FEATURES, CAPACITY_STATIC_FEATURES
from .flare_capacity_modeling import (
    attach_capacity_feature_partitions,
    select_usable_capacity_features,
)
from .flare_capacity_study import (
    EARLY_STOP_TRAIN_END,
    PRIMARY_AUDIT_END,
    PRIMARY_AUDIT_START,
    PURGE_DAYS,
    _joint_columns,
    select_purged_forward_calibration,
)
from .flare_evaluation import evaluate_joint_probabilities, joint_loss_rows
from .flare_reconciliation import hurdle_joint_probabilities
from .flare_study import MODEL_PARAMETERS, TASKS, load_enriched_month
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .modeling import TaskName
from .provenance import capture_provenance

CANDIDATES: tuple[BoundaryCandidate, ...] = (
    "boundary_only",
    "counterfactual_residual",
)
ENSEMBLE_MEMBERS = (
    "capacity_gated_simplex",
    "boundary_only",
    "counterfactual_residual",
)
PROBABILITY_STATES = ("on_time", "delayed", "cancelled")
CONTEXTUAL_REFERENCES = (
    "schedule_baseline",
    "flare24",
    "capacity_gated_simplex",
    "meta_current",
)
SELECTION_MONTHS = (10, 11, 12)
BREAKTHROUGH_MINIMUM = 0.05
BREAKTHROUGH_STRETCH = 0.10
BOUNDARY_GATING_FEATURE = BOUNDARY_RESIDUAL_BY_SOURCE["ccrth_route_sum_shadow_price"]
BOUNDARY_GATING_QUANTILES = (0.50, 0.90)
BOUNDARY_GATING_LABELS = ("low", "elevated", "severe", "missing")
BOUNDARY_ANALYSIS_SIGNALS = (
    "any_nonzero_boundary_residual",
    "newly_observed_boundary_predecessor",
    "severe_vs_low_route_shadow_residual",
)
BOUNDARY_ANALYSIS_OUTCOMES = (
    "joint_disruption",
    "cancellation",
    "delay_given_operated",
)
BOUNDARY_ANY_RESIDUAL_COLUMN = "bcpot_any_nonzero_boundary_residual"


def _projected_task_frame(
    frame: pd.DataFrame,
    task: TaskName,
    feature_columns: tuple[str, ...],
    *,
    eligibility_mask: ArrayLike | None = None,
) -> tuple[pd.DataFrame, NDArray[np.int64]]:
    """Create a hurdle view while copying only columns consumed by the model.

    The joined BC-POT-R source has hundreds of diagnostic columns. Projecting before
    the row copy prevents pandas from consolidating gigabytes of unused float blocks.
    """

    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("projected boundary feature columns must be unique")
    required = {"Cancelled", "delay_label_observed", "ArrDel15", *feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"projected boundary task frame is missing columns: {missing}")
    eligible = (
        np.ones(len(frame), dtype=np.bool_)
        if eligibility_mask is None
        else np.asarray(eligibility_mask, dtype=np.bool_)
    )
    if eligible.shape != (len(frame),):
        raise ValueError("boundary task eligibility mask is not row-aligned")
    if task == "cancellation":
        mask = eligible & frame["Cancelled"].isin([0, 1]).to_numpy()
        labels = frame.loc[mask, "Cancelled"]
    elif task == "delay":
        mask = (
            eligible
            & frame["Cancelled"].eq(0).to_numpy()
            & frame["delay_label_observed"].eq(1).to_numpy()
        )
        labels = frame.loc[mask, "ArrDel15"]
    else:
        raise ValueError(f"BC-POT-R does not fit direct task: {task}")
    numeric = pd.to_numeric(labels, errors="raise").astype("int64")
    if not numeric.isin([0, 1]).all():
        raise ValueError(f"invalid labels for BC-POT-R {task}")
    projected = frame.loc[mask, list(feature_columns)].reset_index(drop=True)
    return projected, numeric.to_numpy(dtype=np.int64)


def _implementation_files() -> tuple[Path, ...]:
    directory = Path(__file__).parent
    return tuple(
        directory / name
        for name in (
            "flare_boundary_study.py",
            "flare_boundary_modeling.py",
            "flare_boundary_contracts.py",
            "flare_boundary_validation.py",
            "flare_capacity_study.py",
            "flare_capacity_modeling.py",
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


def _self_hashed(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    keys = [
        key for key in ("report_sha256", "manifest_sha256", "validation_sha256") if key in payload
    ]
    if len(keys) != 1:
        raise ValueError(f"artifact must have exactly one self-hash: {path}")
    key = keys[0]
    recorded = str(payload[key])
    body = {name: value for name, value in payload.items() if name != key}
    if canonical_json_sha256(body) != recorded:
        raise ValueError(f"artifact self-hash failed: {path}")
    return payload, key


def _verified_record(record: dict[str, Any], *, role: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{role} checksum failed: {path}")
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{role} byte count failed: {path}")
    return path


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _atomic_joblib(value: Any, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite boundary model: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated boundary model partial exists: {partial}")
    joblib.dump(value, partial)
    partial.replace(path)
    return _artifact_record(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite boundary prediction: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated boundary prediction partial exists: {partial}")
    frame.to_parquet(partial, index=False, compression="zstd")
    partial.replace(path)
    return {**_artifact_record(path), "rows": len(frame)}


def _verify_unchanged(started: dict[str, Any], *, phase: str) -> dict[str, Any]:
    current = capture_provenance(_implementation_files())
    if current.get("git_head") != started.get("git_head") or current.get(
        "source_files"
    ) != started.get("source_files"):
        raise RuntimeError(
            f"boundary study implementation changed before {phase}; retain the partial "
            "run and restart with a new run identifier"
        )
    return current


def _load_protocol(path: Path, *, bootstrap_repetitions: int) -> dict[str, Any]:
    protocol: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    models = protocol.get("models", {})
    ensemble = protocol.get("ensemble", {})
    evaluation = protocol.get("evaluation", {})
    descriptive = protocol.get("descriptive_boundary_analysis", {})
    gate = protocol.get("breakthrough_gate", {})
    if (
        protocol.get("identity", {}).get("method") != "BC-POT-R-v1"
        or tuple(models.get("candidates", ())) != CANDIDATES
        or int(models.get("rows_per_training_month", -1)) != 125_000
        or int(models.get("early_stopping_max_rows", -1)) != 250_000
        or int(models.get("sampling_seed", -1)) != 20260903
        or models.get("boosting_type")
        != BOUNDARY_GPU_TRAINING_PARAMETERS["boosting_type"]
        or int(models.get("max_ctr_complexity", -1))
        != BOUNDARY_GPU_TRAINING_PARAMETERS["max_ctr_complexity"]
        or models.get("gpu_cat_features_storage")
        != BOUNDARY_GPU_TRAINING_PARAMETERS["gpu_cat_features_storage"]
        or float(models.get("gpu_ram_part", -1.0))
        != BOUNDARY_GPU_TRAINING_PARAMETERS["gpu_ram_part"]
        or models.get("pinned_memory_size")
        != BOUNDARY_GPU_TRAINING_PARAMETERS["pinned_memory_size"]
        or models.get("early_stopping_training_end_after_purge") != EARLY_STOP_TRAIN_END
        or int(evaluation.get("bootstrap_repetitions", -1)) != bootstrap_repetitions
        or tuple(ensemble.get("members", ())) != ENSEMBLE_MEMBERS
        or ensemble.get("boundary_residual_gating_feature") != BOUNDARY_GATING_FEATURE
        or tuple(float(value) for value in ensemble.get("boundary_residual_gating_quantiles", ()))
        != BOUNDARY_GATING_QUANTILES
        or int(ensemble.get("minimum_regime_rows", -1)) != 10_000
        or tuple(ensemble.get("final_candidates", ()))
        != ("boundary_ensemble", "boundary_gated_ensemble")
        or tuple(evaluation.get("primary_evaluation_dates", ()))
        != (PRIMARY_AUDIT_START, PRIMARY_AUDIT_END)
        or tuple(evaluation.get("contextual_references", ())) != CONTEXTUAL_REFERENCES
        or tuple(descriptive.get("signals", ())) != BOUNDARY_ANALYSIS_SIGNALS
        or tuple(descriptive.get("outcomes", ())) != BOUNDARY_ANALYSIS_OUTCOMES
        or descriptive.get("causal_effect_claimed") is not False
        or float(gate.get("minimum_absolute_gain", -1.0)) != BREAKTHROUGH_MINIMUM
        or float(gate.get("stretch_absolute_gain", -1.0)) != BREAKTHROUGH_STRETCH
        or protocol.get("information_boundary", {}).get("confirmation_gate_opened") is not False
        or protocol.get("epistemic_status", {}).get("previous_2025_aggregate_results_known")
        is not True
    ):
        raise ValueError("boundary executable settings differ from the frozen protocol")
    return protocol


def _verify_boundary_inputs(
    manifest_path: Path,
    validation_path: Path,
    *,
    boundary_feature_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, manifest_key = _self_hashed(manifest_path)
    if (
        manifest.get("status") != "COMPLETE_COVARIATE_FEATURES_NO_TARGET_OUTCOMES_ACCESSED"
        or manifest.get("method") != "FLARE-24-BOUNDARY-COMPLETE-PROBABILISTIC-OPERATIONS-TWIN"
        or manifest.get("confirmation_outcomes_accessed") is not False
        or set(manifest.get("target_years", ())) != {2024, 2025}
        or set(manifest.get("target_months", ())) != set(range(1, 13))
    ):
        raise ValueError("boundary study requires a complete covariate-only manifest")
    root = boundary_feature_dir.resolve()
    for output in manifest.get("outputs", []):
        path = _verified_record(output["features"], role="boundary feature partition")
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError(f"boundary feature partition escapes root: {path}") from error
    validation, validation_key = _self_hashed(validation_path)
    source = validation.get("manifest", {})
    if (
        validation.get("status") != "PASS_BOUNDARY_OPERATIONS_TWIN_FEATURE_VALIDATION"
        or Path(str(source.get("path", ""))).resolve() != manifest_path.resolve()
        or source.get("sha256") != sha256_file(manifest_path)
        or source.get("self_hash") != manifest[manifest_key]
        or validation.get("confirmation_outcomes_accessed") is not False
    ):
        raise ValueError("boundary validation is not bound to the supplied manifest")
    return (
        manifest,
        {
            "manifest": {
                "path": manifest_path.as_posix(),
                "sha256": sha256_file(manifest_path),
                "self_hash_key": manifest_key,
                "self_hash": manifest[manifest_key],
            },
            "validation": {
                "path": validation_path.as_posix(),
                "sha256": sha256_file(validation_path),
                "self_hash_key": validation_key,
                "self_hash": validation[validation_key],
                "status": validation["status"],
            },
        },
    )


def _verify_parent_report(
    path: Path,
    *,
    expected_status: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report, key = _self_hashed(path)
    if report.get("status") != expected_status:
        raise ValueError(f"parent report has unexpected status: {path}")
    return report, {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "self_hash_key": key,
        "self_hash": report[key],
    }


def _verify_boundary_rotation_gate(
    path: Path,
    *,
    boundary_manifest: dict[str, Any],
) -> dict[str, Any]:
    gate, gate_key = _self_hashed(path)
    if (
        gate.get("status") != "PASS_BOUNDARY_ROTATION_OUTCOME_BLIND_MATERIALITY_GATE"
        or gate.get("gate_passed") is not True
        or gate.get("outcomes_read") is not False
        or gate.get("target_tail_number_read") is not False
        or gate.get("confirmation_outcomes_accessed") is not False
    ):
        raise ValueError("boundary study requires a passing outcome-blind rotation gate")
    validation_path = _verified_record(
        gate["rotation_validation"], role="boundary rotation validation"
    )
    validation, validation_key = _self_hashed(validation_path)
    if (
        validation.get("status") != "PASS_BOUNDARY_ROTATION_MODEL_VALIDATION"
        or validation[validation_key] != gate["rotation_validation"].get("self_hash")
        or validation.get("confirmation_outcomes_accessed") is not False
    ):
        raise ValueError("boundary rotation validation binding failed")
    rotation_manifest = validation.get("manifest", {})
    expected = boundary_manifest.get("rotation_manifest", {})
    if (
        Path(str(rotation_manifest.get("path", ""))).resolve()
        != Path(str(expected.get("path", ""))).resolve()
        or rotation_manifest.get("sha256") != expected.get("sha256")
    ):
        raise ValueError("boundary features do not use the smoke-gated rotation manifest")
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "self_hash_key": gate_key,
        "self_hash": gate[gate_key],
        "status": gate["status"],
        "changed_rotation_feature_row_fraction": gate["comparison"][
            "any_rotation_feature_changed_row_fraction"
        ],
    }


def _verify_loader_roots(parent: dict[str, Any], roots: dict[str, Path]) -> None:
    records = {
        str(record["role"]): Path(str(record["root"]))
        for record in parent.get("bound_baseline_inputs", [])
    }
    for role, root in roots.items():
        if role not in records or records[role].resolve() != root.resolve():
            raise ValueError(f"supplied {role} root differs from the parent experiment")


def _load_month(
    *,
    year: int,
    month: int,
    limit: int | None,
    capacity_feature_dir: Path,
    boundary_feature_dir: Path,
    loader_arguments: dict[str, Any],
) -> pd.DataFrame:
    if year not in {2024, 2025}:
        raise ValueError(f"boundary experiment refuses outcome year {year}")
    frame = load_enriched_month(
        year=year,
        month=month,
        limit=limit,
        **loader_arguments,
    )
    frame = attach_capacity_feature_partitions(
        frame,
        feature_dir=capacity_feature_dir,
    )
    return attach_boundary_views(frame, boundary_feature_dir=boundary_feature_dir)


def _load_training(
    *,
    capacity_feature_dir: Path,
    boundary_feature_dir: Path,
    loader_arguments: dict[str, Any],
) -> pd.DataFrame:
    return pd.concat(
        [
            _load_month(
                year=2024,
                month=month,
                limit=125_000,
                capacity_feature_dir=capacity_feature_dir,
                boundary_feature_dir=boundary_feature_dir,
                loader_arguments=loader_arguments,
            )
            for month in range(1, 9)
        ],
        ignore_index=True,
    )


def _load_model(record: dict[str, Any]) -> BoundaryCatBoostModel:
    path = _verified_record(record, role="boundary model")
    model = joblib.load(path)
    if not isinstance(model, BoundaryCatBoostModel):
        raise TypeError(f"unexpected boundary model type: {type(model)!r}")
    return model


def _normalize_joint(values: ArrayLike, *, role: str) -> NDArray[np.float64]:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError(f"{role} probabilities must have three columns")
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError(f"{role} probabilities are invalid")
    sums = probabilities.sum(axis=1)
    if (sums <= 0.0).any() or float(np.max(np.abs(sums - 1.0))) > 1e-6:
        raise ValueError(f"{role} persisted probability drift is too large")
    return np.asarray(probabilities / sums[:, None], dtype=np.float64)


def _probability_frame(
    frame: pd.DataFrame,
    method: str,
) -> NDArray[np.float64]:
    return _normalize_joint(
        frame.loc[:, list(_joint_columns(method))].to_numpy(dtype=np.float64),
        role=method,
    )


def _set_probability_frame(
    frame: pd.DataFrame,
    method: str,
    probabilities: NDArray[np.float64],
) -> None:
    if probabilities.shape != (len(frame), 3):
        raise ValueError(f"{method} probabilities do not align")
    for index, column in enumerate(_joint_columns(method)):
        frame[column] = probabilities[:, index].astype("float32")


def _join_reference(
    frame: pd.DataFrame,
    record: dict[str, Any],
    *,
    methods: tuple[str, ...],
    role: str,
) -> pd.DataFrame:
    path = _verified_record(record, role=role)
    columns = ["sample_id"]
    for method in methods:
        columns.extend(_joint_columns(method))
    reference = pd.read_parquet(path, columns=columns)
    if reference["sample_id"].isna().any() or reference["sample_id"].duplicated().any():
        raise ValueError(f"{role} has invalid sample ids")
    result = frame.merge(
        reference,
        on="sample_id",
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if not result["_merge"].eq("both").all():
        raise ValueError(f"{role} omits requested sample ids")
    result = result.drop(columns="_merge")
    for method in methods:
        normalized = _probability_frame(result, method)
        for index, column in enumerate(_joint_columns(method)):
            result[column] = normalized[:, index]
    return result


def _join_meta_reference(
    frame: pd.DataFrame,
    record: dict[str, Any],
) -> pd.DataFrame:
    path = _verified_record(record, role="meta-stack retrospective prediction")
    prefix = "tf_ccrth_cancellation_logit_stack"
    source_columns = tuple(f"prob_{prefix}_{state}" for state in PROBABILITY_STATES)
    reference = pd.read_parquet(path, columns=["sample_id", *source_columns]).rename(
        columns={
            source: target
            for source, target in zip(
                source_columns,
                _joint_columns("meta_current"),
                strict=True,
            )
        }
    )
    result = frame.merge(
        reference,
        on="sample_id",
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if not result["_merge"].eq("both").all():
        raise ValueError("meta-stack retrospective prediction omits sample ids")
    result = result.drop(columns="_merge")
    normalized = _probability_frame(result, "meta_current")
    for index, column in enumerate(_joint_columns("meta_current")):
        result[column] = normalized[:, index]
    return result


def _join_schedule_reference(
    frame: pd.DataFrame,
    record: dict[str, Any],
) -> pd.DataFrame:
    path = _verified_record(record, role="original schedule-baseline prediction")
    source_columns = tuple(f"prob_baseline_{state}" for state in PROBABILITY_STATES)
    reference = pd.read_parquet(path, columns=["sample_id", *source_columns]).rename(
        columns={
            source: target
            for source, target in zip(
                source_columns,
                _joint_columns("schedule_baseline"),
                strict=True,
            )
        }
    )
    if reference["sample_id"].isna().any() or reference["sample_id"].duplicated().any():
        raise ValueError("schedule-baseline prediction has invalid sample ids")
    result = frame.merge(
        reference,
        on="sample_id",
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if not result["_merge"].eq("both").all():
        raise ValueError("schedule-baseline prediction omits sample ids")
    result = result.drop(columns="_merge")
    normalized = _probability_frame(result, "schedule_baseline")
    for index, column in enumerate(_joint_columns("schedule_baseline")):
        result[column] = normalized[:, index]
    return result


def _select_simplex(
    labels: ArrayLike,
    probabilities: dict[str, NDArray[np.float64]],
) -> dict[str, Any]:
    if tuple(probabilities) != ENSEMBLE_MEMBERS:
        raise ValueError(f"ensemble members must be ordered as {ENSEMBLE_MEMBERS}")
    y = np.asarray(labels, dtype=np.int64)
    matrices = [probabilities[name] for name in ENSEMBLE_MEMBERS]
    if any(matrix.shape != (len(y), 3) for matrix in matrices):
        raise ValueError("ensemble probability matrices do not align")
    truth = np.column_stack([matrix[np.arange(len(y)), y] for matrix in matrices])

    def objective(weights: NDArray[np.float64]) -> float:
        return float(-np.log(np.clip(truth @ weights, 1e-12, 1.0)).mean())

    def gradient(weights: NDArray[np.float64]) -> NDArray[np.float64]:
        mixture = np.clip(truth @ weights, 1e-12, 1.0)
        return np.asarray(-(truth / mixture[:, None]).mean(axis=0), dtype=np.float64)

    initial = np.full(len(matrices), 1.0 / len(matrices))
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(initial),
        constraints={"type": "eq", "fun": lambda values: float(values.sum() - 1.0)},
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    if not result.success:
        raise RuntimeError(f"boundary simplex selection failed: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    weights /= weights.sum()
    return {
        "weights": {
            name: float(weight) for name, weight in zip(ENSEMBLE_MEMBERS, weights, strict=True)
        },
        "selection_metric": "joint_log_loss",
        "joint_log_loss": objective(weights),
        "iterations": int(result.nit),
    }


def _apply_simplex(
    probabilities: dict[str, NDArray[np.float64]],
    weights: dict[str, float],
) -> NDArray[np.float64]:
    if tuple(probabilities) != ENSEMBLE_MEMBERS or set(weights) != set(ENSEMBLE_MEMBERS):
        raise ValueError("boundary simplex inputs differ from the lock")
    output = sum(weights[name] * probabilities[name] for name in ENSEMBLE_MEMBERS)
    result = np.asarray(output, dtype=np.float64)
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("boundary simplex produced invalid probabilities")
    return result


def _boundary_regimes(
    values: ArrayLike,
    cutpoints: ArrayLike,
) -> NDArray[np.str_]:
    numeric = np.asarray(values, dtype=np.float64)
    cuts = np.asarray(cutpoints, dtype=np.float64)
    if numeric.ndim != 1 or cuts.shape != (2,) or not np.isfinite(cuts).all():
        raise ValueError("boundary regimes require a vector and two finite cutpoints")
    if cuts[0] > cuts[1]:
        raise ValueError("boundary residual cutpoints must be ordered")
    output = np.full(len(numeric), "missing", dtype="<U8")
    observed = np.isfinite(numeric)
    output[observed & (numeric <= cuts[0])] = "low"
    output[observed & (numeric > cuts[0]) & (numeric <= cuts[1])] = "elevated"
    output[observed & (numeric > cuts[1])] = "severe"
    return output


def _select_boundary_gated_simplex(
    labels: ArrayLike,
    probabilities: dict[str, NDArray[np.float64]],
    gating_values: ArrayLike,
    *,
    cutpoint_values: ArrayLike,
    minimum_rows: int = 10_000,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    gate = np.asarray(gating_values, dtype=np.float64)
    cutpoint_source = np.asarray(cutpoint_values, dtype=np.float64)
    if gate.shape != (len(y),) or cutpoint_source.ndim != 1:
        raise ValueError("boundary gating values do not align")
    observed = cutpoint_source[np.isfinite(cutpoint_source)]
    if observed.size == 0:
        raise ValueError("boundary gating feature has no observed values")
    cutpoints = np.quantile(observed, BOUNDARY_GATING_QUANTILES)
    regimes = _boundary_regimes(gate, cutpoints)
    global_selection = _select_simplex(y, probabilities)
    selections: dict[str, Any] = {}
    for regime in BOUNDARY_GATING_LABELS:
        mask = regimes == regime
        if int(mask.sum()) < minimum_rows or np.unique(y[mask]).size < 3:
            selections[regime] = {
                **global_selection,
                "rows": int(mask.sum()),
                "fallback_to_global": True,
            }
        else:
            selected = _select_simplex(
                y[mask],
                {name: values[mask] for name, values in probabilities.items()},
            )
            selections[regime] = {
                **selected,
                "rows": int(mask.sum()),
                "fallback_to_global": False,
            }
    return {
        "gating_feature": BOUNDARY_GATING_FEATURE,
        "cutpoint_quantiles": list(BOUNDARY_GATING_QUANTILES),
        "cutpoints": cutpoints.tolist(),
        "minimum_rows": minimum_rows,
        "global": global_selection,
        "regimes": selections,
    }


def _apply_boundary_gated_simplex(
    probabilities: dict[str, NDArray[np.float64]],
    gating_values: ArrayLike,
    selection: dict[str, Any],
) -> NDArray[np.float64]:
    first = next(iter(probabilities.values()))
    gate = np.asarray(gating_values, dtype=np.float64)
    if gate.shape != (len(first),):
        raise ValueError("boundary gating values do not align with predictions")
    regimes = _boundary_regimes(gate, selection["cutpoints"])
    output = np.empty_like(first, dtype=np.float64)
    for regime in BOUNDARY_GATING_LABELS:
        mask = regimes == regime
        if mask.any():
            output[mask] = _apply_simplex(
                {name: values[mask] for name, values in probabilities.items()},
                selection["regimes"][regime]["weights"],
            )
    if not np.isfinite(output).all() or not np.allclose(output.sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("boundary-gated simplex produced invalid probabilities")
    return output


def _decision_metrics(
    labels: ArrayLike,
    probabilities: NDArray[np.float64],
    *,
    log_bias: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    scores = probabilities.copy()
    scores[:, 1] *= np.exp(log_bias[0])
    scores[:, 2] *= np.exp(log_bias[1])
    predicted = scores.argmax(axis=1)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (y, predicted), 1)
    recall = np.divide(
        np.diag(confusion),
        confusion.sum(axis=1),
        out=np.zeros(3, dtype=np.float64),
        where=confusion.sum(axis=1) > 0,
    )
    return {
        "n": len(y),
        "accuracy": float((predicted == y).mean()),
        "balanced_accuracy": float(recall.mean()),
        "class_recall": {
            state: float(recall[index]) for index, state in enumerate(PROBABILITY_STATES)
        },
        "confusion_matrix_true_by_predicted": confusion.tolist(),
        "log_bias_delay": float(log_bias[0]),
        "log_bias_cancellation": float(log_bias[1]),
    }


def _select_accuracy_bias(
    labels: ArrayLike,
    probabilities: NDArray[np.float64],
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    delay_grid = np.linspace(-1.0, 2.0, 25)
    cancellation_grid = np.linspace(-1.0, 3.0, 25)
    records: list[dict[str, float]] = []
    for delay_bias in delay_grid:
        delay_score = probabilities[:, 1] * np.exp(delay_bias)
        for cancellation_bias in cancellation_grid:
            cancellation_score = probabilities[:, 2] * np.exp(cancellation_bias)
            predicted = np.where(
                probabilities[:, 0] >= np.maximum(delay_score, cancellation_score),
                0,
                np.where(delay_score >= cancellation_score, 1, 2),
            )
            records.append(
                {
                    "delay": float(delay_bias),
                    "cancellation": float(cancellation_bias),
                    "accuracy": float((predicted == y).mean()),
                }
            )
    selected = min(
        records,
        key=lambda record: (
            -record["accuracy"],
            abs(record["delay"]) + abs(record["cancellation"]),
            record["delay"],
            record["cancellation"],
        ),
    )
    return {
        "selection_metric": "three_state_accuracy",
        "delay_log_bias_grid": [-1.0, 2.0, 25],
        "cancellation_log_bias_grid": [-1.0, 3.0, 25],
        "selected_delay_log_bias": selected["delay"],
        "selected_cancellation_log_bias": selected["cancellation"],
        "selected_accuracy": selected["accuracy"],
    }


def _accuracy_comparisons(
    frame: pd.DataFrame,
    methods: dict[str, NDArray[np.float64]],
    *,
    reference: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    observed = frame["joint_label_observed"].eq(1).to_numpy()
    labels = frame.loc[observed, "disruption_state"].to_numpy(dtype=np.int64)
    clusters = frame.loc[observed, "FlightDate"].to_numpy()
    reference_correct = (methods[reference][observed].argmax(axis=1) == labels).astype(np.float64)
    records: dict[str, Any] = {}
    for name, probabilities in methods.items():
        scored = probabilities[observed]
        correct = (scored.argmax(axis=1) == labels).astype(np.float64)
        metrics = _decision_metrics(labels, scored)
        if name != reference:
            interval = paired_cluster_mean_difference(
                correct,
                reference_correct,
                clusters,
                repetitions=repetitions,
                seed=seed,
            )
            metrics["accuracy_difference_vs_reference"] = asdict(interval)
        records[name] = metrics
    return records


def _records_by_month(report: dict[str, Any], key: str) -> dict[int, dict[str, Any]]:
    records = {
        int(record["month"]): dict(record)
        for record in report.get(key, [])
        if int(record.get("year", -1)) == 2025
    }
    if set(records) != set(range(1, 13)):
        raise ValueError(f"parent report {key} does not cover 2025")
    return records


def _point_scores(
    frame: pd.DataFrame,
    methods: dict[str, NDArray[np.float64]],
    mask: NDArray[np.bool_],
) -> dict[str, Any]:
    observed = frame["joint_label_observed"].eq(1).to_numpy() & mask
    labels = frame.loc[observed, "disruption_state"].to_numpy(dtype=np.int64)
    scores: dict[str, Any] = {}
    for name, probabilities in methods.items():
        selected = probabilities[observed]
        log_rows, brier_rows = joint_loss_rows(labels, selected)
        scores[name] = {
            "n": len(labels),
            "joint_log_loss": float(log_rows.mean()),
            "multiclass_brier": float(brier_rows.mean()),
            "accuracy": float((selected.argmax(axis=1) == labels).mean()),
        }
    return scores


def _monthly_scores(
    frame: pd.DataFrame,
    methods: dict[str, NDArray[np.float64]],
) -> list[dict[str, Any]]:
    months = pd.to_numeric(frame["Month"], errors="raise").astype(int)
    return [
        {
            "month": int(month),
            "methods": _point_scores(
                frame,
                methods,
                months.eq(month).to_numpy(),
            ),
        }
        for month in sorted(months.unique())
    ]


def _residual_regime_scores(
    frame: pd.DataFrame,
    methods: dict[str, NDArray[np.float64]],
    selection: dict[str, Any],
) -> list[dict[str, Any]]:
    regimes = _boundary_regimes(
        frame[BOUNDARY_GATING_FEATURE].to_numpy(dtype=np.float64),
        selection["cutpoints"],
    )
    records: list[dict[str, Any]] = []
    observed = frame["joint_label_observed"].eq(1).to_numpy()
    labels = frame["disruption_state"].to_numpy(dtype=np.int64)
    gating_values = frame[BOUNDARY_GATING_FEATURE].to_numpy(dtype=np.float64)
    for regime in BOUNDARY_GATING_LABELS:
        mask = regimes == regime
        if not mask.any():
            continue
        scored = mask & observed
        prevalence = {
            state: (float((labels[scored] == index).mean()) if scored.any() else None)
            for index, state in enumerate(PROBABILITY_STATES)
        }
        values = gating_values[mask & np.isfinite(gating_values)]
        records.append(
            {
                "regime": regime,
                "rows": int(mask.sum()),
                "observed_rows": int(scored.sum()),
                "class_prevalence": prevalence,
                "gating_feature_mean": float(values.mean()) if values.size else None,
                "gating_feature_p90": float(np.quantile(values, 0.90)) if values.size else None,
                "methods": _point_scores(frame, methods, mask),
            }
        )
    return records


def _boundary_any_residual_mask(frame: pd.DataFrame) -> NDArray[np.bool_]:
    if BOUNDARY_ANY_RESIDUAL_COLUMN in frame:
        values = pd.to_numeric(frame[BOUNDARY_ANY_RESIDUAL_COLUMN], errors="raise")
        return np.asarray(values.to_numpy(dtype=np.float64) > 0.5, dtype=np.bool_)
    missing = sorted(set(BOUNDARY_RESIDUAL_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"boundary residual summary is missing columns: {missing}")
    any_residual = np.zeros(len(frame), dtype=np.bool_)
    for feature in BOUNDARY_RESIDUAL_FEATURES:
        values = pd.to_numeric(frame[feature], errors="raise").to_numpy(dtype=np.float32)
        any_residual |= np.isfinite(values) & (np.abs(values) > np.float32(1e-7))
    return any_residual


def _boundary_signal_summary(frame: pd.DataFrame) -> dict[str, Any]:
    any_residual = _boundary_any_residual_mask(frame)
    new_predecessor = frame[
        "bcpot_rotation_predecessor_state_newly_observed"
    ].to_numpy(dtype=np.float64) > 0.5
    lost_predecessor = frame[
        "bcpot_rotation_predecessor_state_lost"
    ].to_numpy(dtype=np.float64) > 0.5
    return {
        "rows": len(frame),
        "rows_with_any_nonzero_boundary_residual": int(any_residual.sum()),
        "fraction_with_any_nonzero_boundary_residual": float(any_residual.mean()),
        "rows_with_newly_observed_rotation_predecessor_state": int(new_predecessor.sum()),
        "fraction_with_newly_observed_rotation_predecessor_state": float(
            new_predecessor.mean()
        ),
        "rows_with_lost_rotation_predecessor_state": int(lost_predecessor.sum()),
        "fraction_with_lost_rotation_predecessor_state": float(lost_predecessor.mean()),
    }


def _cluster_binary_rate_difference(
    outcomes: ArrayLike,
    exposed: ArrayLike,
    clusters: ArrayLike,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    y = np.asarray(outcomes, dtype=np.int64)
    treatment = np.asarray(exposed, dtype=np.bool_)
    cluster = np.asarray(clusters)
    if (
        y.ndim != 1
        or treatment.shape != y.shape
        or cluster.shape != y.shape
        or not np.isin(y, [0, 1]).all()
        or len(y) == 0
        or repetitions < 100
    ):
        raise ValueError("clustered binary-rate inputs are invalid")
    exposed_rows = int(treatment.sum())
    unexposed_rows = int((~treatment).sum())
    unique_clusters = np.unique(cluster)
    exposed_clusters = int(np.unique(cluster[treatment]).size) if exposed_rows else 0
    unexposed_clusters = int(np.unique(cluster[~treatment]).size) if unexposed_rows else 0
    common = {
        "exposed_rows": exposed_rows,
        "unexposed_rows": unexposed_rows,
        "clusters": int(unique_clusters.size),
        "clusters_with_exposed": exposed_clusters,
        "clusters_with_unexposed": unexposed_clusters,
        "repetitions": repetitions,
        "seed": seed,
    }
    if exposed_rows == 0 or unexposed_rows == 0:
        return {
            **common,
            "exposed_rate": float(y[treatment].mean()) if exposed_rows else None,
            "unexposed_rate": float(y[~treatment].mean()) if unexposed_rows else None,
            "rate_difference_exposed_minus_unexposed": None,
            "cluster_bootstrap_lower": None,
            "cluster_bootstrap_upper": None,
            "valid_bootstrap_repetitions": 0,
            "bootstrap_valid_fraction": 0.0,
            "inference_status": "NOT_ESTIMABLE_EMPTY_CONTRAST_GROUP",
        }
    table = pd.DataFrame({"cluster": cluster, "exposed": treatment, "outcome": y})
    grouped = (
        table.groupby(["cluster", "exposed"], sort=True, observed=True)["outcome"]
        .agg(["sum", "count"])
        .reset_index()
    )
    cluster_ids = sorted(grouped["cluster"].unique())
    lookup = {value: index for index, value in enumerate(cluster_ids)}
    successes = np.zeros((len(cluster_ids), 2), dtype=np.float64)
    counts = np.zeros((len(cluster_ids), 2), dtype=np.float64)
    for row in grouped.itertuples(index=False):
        cluster_index = lookup[row.cluster]
        group_index = int(bool(row.exposed))
        successes[cluster_index, group_index] = float(row.sum)
        counts[cluster_index, group_index] = float(row.count)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cluster_ids), size=(repetitions, len(cluster_ids)))
    sampled_successes = successes[draws].sum(axis=1)
    sampled_counts = counts[draws].sum(axis=1)
    valid = (sampled_counts > 0.0).all(axis=1)
    differences = (
        sampled_successes[valid, 1] / sampled_counts[valid, 1]
        - sampled_successes[valid, 0] / sampled_counts[valid, 0]
    )
    valid_repetitions = len(differences)
    exposed_rate = float(y[treatment].mean())
    unexposed_rate = float(y[~treatment].mean())
    enough_support = valid_repetitions >= int(np.ceil(0.99 * repetitions))
    lower: float | None = None
    upper: float | None = None
    if enough_support:
        quantiles = np.quantile(differences, [0.025, 0.975])
        lower, upper = float(quantiles[0]), float(quantiles[1])
    return {
        **common,
        "exposed_rate": exposed_rate,
        "unexposed_rate": unexposed_rate,
        "rate_difference_exposed_minus_unexposed": exposed_rate - unexposed_rate,
        "cluster_bootstrap_lower": lower,
        "cluster_bootstrap_upper": upper,
        "valid_bootstrap_repetitions": valid_repetitions,
        "bootstrap_valid_fraction": valid_repetitions / repetitions,
        "inference_status": (
            "ESTIMABLE"
            if enough_support
            else "NOT_ESTIMABLE_INSUFFICIENT_TEMPORAL_CONTRAST_SUPPORT"
        ),
    }


def _boundary_signal_outcome_analysis(
    frame: pd.DataFrame,
    gating_selection: dict[str, Any],
    *,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    any_residual = _boundary_any_residual_mask(frame)
    new_predecessor = frame[
        "bcpot_rotation_predecessor_state_newly_observed"
    ].to_numpy(dtype=np.float64) > 0.5
    regimes = _boundary_regimes(
        frame[BOUNDARY_GATING_FEATURE].to_numpy(dtype=np.float64),
        gating_selection["cutpoints"],
    )
    signal_definitions = (
        (
            "any_nonzero_boundary_residual",
            np.ones(len(frame), dtype=np.bool_),
            any_residual,
            "any nonzero boundary residual versus none",
        ),
        (
            "newly_observed_boundary_predecessor",
            np.ones(len(frame), dtype=np.bool_),
            new_predecessor,
            "newly observed context-only predecessor state versus not newly observed",
        ),
        (
            "severe_vs_low_route_shadow_residual",
            np.isin(regimes, ["low", "severe"]),
            regimes == "severe",
            "severe versus low using Q4-2024-locked residual cutpoints",
        ),
    )
    joint_observed = frame["joint_label_observed"].eq(1).to_numpy()
    cancellation_observed = frame["Cancelled"].isin([0, 1]).to_numpy()
    delay_observed = (
        frame["Cancelled"].eq(0) & frame["delay_label_observed"].eq(1)
    ).to_numpy()
    outcome_definitions = (
        (
            "joint_disruption",
            joint_observed,
            frame["disruption_state"].fillna(-1).to_numpy(dtype=np.int64) != 0,
        ),
        (
            "cancellation",
            cancellation_observed,
            frame["Cancelled"].fillna(-1).to_numpy(dtype=np.int64) == 1,
        ),
        (
            "delay_given_operated",
            delay_observed,
            frame["ArrDel15"].fillna(-1).to_numpy(dtype=np.int64) == 1,
        ),
    )
    clusters = frame["FlightDate"].to_numpy()
    records: list[dict[str, Any]] = []
    for signal_index, (signal, signal_scope, exposure, contrast) in enumerate(
        signal_definitions
    ):
        for outcome_index, (outcome, outcome_scope, labels) in enumerate(
            outcome_definitions
        ):
            eligible = signal_scope & outcome_scope
            comparison = _cluster_binary_rate_difference(
                labels[eligible],
                exposure[eligible],
                clusters[eligible],
                repetitions=repetitions,
                seed=seed + 100 * signal_index + outcome_index,
            )
            records.append(
                {
                    "signal": signal,
                    "contrast": contrast,
                    "outcome": outcome,
                    **comparison,
                    "causal_effect_claimed": False,
                }
            )
    return records


def _airport_scores(
    frame: pd.DataFrame,
    methods: dict[str, NDArray[np.float64]],
    *,
    selected_method: str,
) -> list[dict[str, Any]]:
    observed = frame["joint_label_observed"].eq(1).to_numpy()
    labels = frame.loc[observed, "disruption_state"].to_numpy(dtype=np.int64)
    records: list[dict[str, Any]] = []
    for role, column in (("origin", "Origin"), ("destination", "Dest")):
        airports = frame.loc[observed, column].astype(str).reset_index(drop=True)
        per_method: dict[str, pd.DataFrame] = {}
        for name in ("meta_current", selected_method):
            probabilities = methods[name][observed]
            log_rows, brier_rows = joint_loss_rows(labels, probabilities)
            values = pd.DataFrame(
                {
                    "airport": airports,
                    "log_loss": log_rows,
                    "brier": brier_rows,
                    "correct": (probabilities.argmax(axis=1) == labels).astype(np.float64),
                }
            )
            per_method[name] = values.groupby("airport", sort=True, observed=True).agg(
                n=("log_loss", "size"),
                joint_log_loss=("log_loss", "mean"),
                multiclass_brier=("brier", "mean"),
                accuracy=("correct", "mean"),
            )
        reference = per_method["meta_current"]
        selected = per_method[selected_method]
        for airport in selected.index:
            records.append(
                {
                    "role": role,
                    "airport": airport,
                    "n": int(selected.loc[airport, "n"]),
                    "selected_method": selected_method,
                    "joint_log_loss": float(selected.loc[airport, "joint_log_loss"]),
                    "joint_log_loss_delta_vs_meta": float(
                        selected.loc[airport, "joint_log_loss"]
                        - reference.loc[airport, "joint_log_loss"]
                    ),
                    "multiclass_brier": float(selected.loc[airport, "multiclass_brier"]),
                    "multiclass_brier_delta_vs_meta": float(
                        selected.loc[airport, "multiclass_brier"]
                        - reference.loc[airport, "multiclass_brier"]
                    ),
                    "accuracy": float(selected.loc[airport, "accuracy"]),
                    "accuracy_delta_vs_meta": float(
                        selected.loc[airport, "accuracy"] - reference.loc[airport, "accuracy"]
                    ),
                }
            )
    return records


def run_boundary_study(
    *,
    protocol_path: Path,
    boundary_manifest_path: Path,
    boundary_validation_path: Path,
    boundary_rotation_gate_path: Path,
    parent_report_path: Path,
    meta_report_path: Path,
    baseline_selection_report_path: Path,
    baseline_audit_report_path: Path,
    census_dir: Path,
    recent_dir: Path,
    flight_recent_dir: Path,
    graph_dir: Path,
    weather_feature_dir: Path,
    rotation_feature_dir: Path,
    capacity_feature_dir: Path,
    boundary_feature_dir: Path,
    run_dir: Path,
    output_path: Path,
    bootstrap_repetitions: int = 2_000,
    seed: int = 20260903,
) -> dict[str, Any]:
    """Fit, lock, and evaluate the paired boundary-complete operations twin."""

    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse boundary run directory: {run_dir}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite boundary report: {output_path}")
    if bootstrap_repetitions < 100 or seed != 20260903:
        raise ValueError("boundary study bootstrap or sampling settings differ from protocol")
    started_at = time.perf_counter()
    provenance_at_start = capture_provenance(_implementation_files())
    protocol = _load_protocol(protocol_path, bootstrap_repetitions=bootstrap_repetitions)
    boundary_manifest, boundary_inputs = _verify_boundary_inputs(
        boundary_manifest_path,
        boundary_validation_path,
        boundary_feature_dir=boundary_feature_dir,
    )
    boundary_rotation_gate = _verify_boundary_rotation_gate(
        boundary_rotation_gate_path,
        boundary_manifest=boundary_manifest,
    )
    parent, parent_record = _verify_parent_report(
        parent_report_path,
        expected_status="COMPLETE_CCRTH_2024_SELECTION_2025_RETROSPECTIVE_EVALUATION",
    )
    meta, meta_record = _verify_parent_report(
        meta_report_path,
        expected_status="COMPLETE_TF_CCRTH_METASTACK_POST_HOC_2025_2026_UNOPENED",
    )
    baseline, baseline_record = _verify_parent_report(
        baseline_selection_report_path,
        expected_status="COMPLETE_2024_FLARE24_SELECTION_NOT_CONFIRMATORY",
    )
    baseline_audit, baseline_audit_record = _verify_parent_report(
        baseline_audit_report_path,
        expected_status="COMPLETE_2025_FLARE24_RETROSPECTIVE_AUDIT_NOT_BLIND_CONFIRMATION",
    )
    roots = {
        "census": census_dir,
        "closed-left HMOP": recent_dir,
        "closed-left flight history": flight_recent_dir,
        "CL-SGMP": graph_dir,
        "FLARE cutoff-coherent weather": weather_feature_dir,
        "FLARE latent rotations": rotation_feature_dir,
    }
    _verify_loader_roots(parent, roots)
    if (
        int(parent.get("rows_per_train_month", -1)) != 125_000
        or int(parent.get("validation_limit", -1)) != 250_000
    ):
        raise ValueError("parent CC-RTH sampling contract differs")
    if int(parent.get("seed", -1)) != seed:
        raise ValueError("parent CC-RTH sampling seed differs")
    if (
        Path(str(parent["capacity_manifest"]["path"])).resolve()
        != Path("manifests/flare24_ccrth_features_v3.json").resolve()
    ):
        raise ValueError("boundary study requires the frozen induced-graph parent")
    if tuple(baseline.get("feature_sets", {}).get("rotation_structural", ())) == ():
        raise ValueError("baseline FLARE feature set is missing")

    run_dir.mkdir(parents=True)
    method_lock: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED_BCPOTR_RETROSPECTIVE_REDEVELOPMENT_BEFORE_NEW_OUTCOME_LOAD",
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "path": protocol_path.as_posix(),
            "bytes": protocol_path.stat().st_size,
            "sha256": sha256_file(protocol_path),
            "identity": protocol["identity"],
        },
        "boundary_inputs": boundary_inputs,
        "boundary_rotation_outcome_blind_gate": boundary_rotation_gate,
        "parent_report": parent_record,
        "meta_report": meta_record,
        "baseline_selection_report": baseline_record,
        "baseline_audit_report": baseline_audit_record,
        "candidates": list(CANDIDATES),
        "ensemble_members": list(ENSEMBLE_MEMBERS),
        "boundary_gating_feature": BOUNDARY_GATING_FEATURE,
        "boundary_gating_quantiles": list(BOUNDARY_GATING_QUANTILES),
        "model_parameters": {
            task: {**parameters, **BOUNDARY_GPU_TRAINING_PARAMETERS}
            for task, parameters in MODEL_PARAMETERS.items()
        },
        "feature_selection": "training-covariate availability and variation only",
        "previous_2025_aggregate_results_known": True,
        "new_2025_boundary_predictions_or_labels_loaded_before_lock": False,
        "confirmation_2026_opened": False,
        "breakthrough_gate": {
            "reference": "meta_current",
            "metric": "primary-period standard argmax accuracy",
            "minimum_absolute_gain": BREAKTHROUGH_MINIMUM,
            "stretch_absolute_gain": BREAKTHROUGH_STRETCH,
        },
        "provenance": provenance_at_start,
    }
    method_lock["manifest_sha256"] = canonical_json_sha256(method_lock)
    lock_path = run_dir / "method_lock.json"
    write_canonical_json(lock_path, method_lock)

    # load_enriched_month uses argument names rather than evidence-role labels.
    loader_arguments: dict[str, Any] = {
        "census_dir": census_dir,
        "recent_dir": recent_dir,
        "flight_recent_dir": flight_recent_dir,
        "graph_dir": graph_dir,
        "weather_feature_dir": weather_feature_dir,
        "rotation_feature_dir": rotation_feature_dir,
        "seed": seed,
    }
    print("loading BC-POT-R January-August 2024 training sample", flush=True)
    train = _load_training(
        capacity_feature_dir=capacity_feature_dir,
        boundary_feature_dir=boundary_feature_dir,
        loader_arguments=loader_arguments,
    )
    print("loading BC-POT-R September 2024 early-stopping sample", flush=True)
    validation = _load_month(
        year=2024,
        month=9,
        limit=250_000,
        capacity_feature_dir=capacity_feature_dir,
        boundary_feature_dir=boundary_feature_dir,
        loader_arguments=loader_arguments,
    )
    train_dates = pd.to_datetime(train["FlightDate"], errors="raise")
    pilot_mask = train_dates.le(pd.Timestamp(EARLY_STOP_TRAIN_END)).to_numpy()
    gap_days = (
        pd.to_datetime(validation["FlightDate"], errors="raise").min()
        - train_dates.loc[pilot_mask].max()
    ).days - 1
    if gap_days != PURGE_DAYS:
        raise RuntimeError("boundary early-stopping split lacks the declared purge")

    flare_features = tuple(baseline["feature_sets"]["rotation_structural"])
    pilot_capacity = train.loc[pilot_mask, list(CAPACITY_ALL_FEATURES)]
    usable_capacity = set(select_usable_capacity_features(pilot_capacity, CAPACITY_ALL_FEATURES))
    del pilot_capacity
    gc.collect()
    pilot_boundary = train.loc[pilot_mask, list((*BOUNDARY_FULL_FEATURES, *BOUNDARY_RESIDUAL_FEATURES, *BOUNDARY_OBSERVATION_FEATURES))]
    usable_boundary = set(select_usable_boundary_features(pilot_boundary))
    del pilot_boundary
    gc.collect()
    feature_sets: dict[BoundaryCandidate, dict[str, tuple[str, ...]]] = {
        "boundary_only": {
            "capacity": tuple(
                feature for feature in CAPACITY_STATIC_FEATURES if feature in usable_capacity
            ),
            "boundary": tuple(
                feature for feature in BOUNDARY_FULL_FEATURES if feature in usable_boundary
            ),
        },
        "counterfactual_residual": {
            "capacity": tuple(
                feature for feature in CAPACITY_ALL_FEATURES if feature in usable_capacity
            ),
            "boundary": tuple(
                feature
                for feature in (
                    *BOUNDARY_RESIDUAL_FEATURES,
                    *BOUNDARY_OBSERVATION_FEATURES,
                )
                if feature in usable_boundary
            ),
        },
    }
    if (
        not feature_sets["boundary_only"]["boundary"]
        or not feature_sets["counterfactual_residual"]["boundary"]
    ):
        raise RuntimeError("boundary feature selection produced an empty candidate")

    model_records: list[dict[str, Any]] = []
    models: dict[BoundaryCandidate, dict[str, dict[str, Any]]] = {
        candidate: {} for candidate in CANDIDATES
    }
    for task in TASKS:
        for candidate in CANDIDATES:
            fit_started = time.perf_counter()
            settings = feature_sets[candidate]
            model_columns = boundary_model_input_columns(
                flare_features=flare_features,
                capacity_features=settings["capacity"],
                boundary_features=settings["boundary"],
            )
            pilot_frame, pilot_labels = _projected_task_frame(
                train,
                task,
                model_columns,
                eligibility_mask=pilot_mask,
            )
            valid_frame, valid_labels = _projected_task_frame(
                validation,
                task,
                model_columns,
            )
            parameters = {
                **MODEL_PARAMETERS[task],
                **BOUNDARY_GPU_TRAINING_PARAMETERS,
                "random_seed": seed,
            }
            print(f"early-stopping BC-POT-R {candidate} {task}", flush=True)
            pilot_model = fit_boundary_catboost(
                pilot_frame,
                pilot_labels,
                task=cast(Any, task),
                candidate=candidate,
                capacity_features=settings["capacity"],
                boundary_features=settings["boundary"],
                params=parameters,
                flare_features=flare_features,
                validation_frame=valid_frame,
                validation_labels=valid_labels,
            )
            refit_iterations = max(int(pilot_model.estimator.tree_count_), 1)
            del pilot_model, pilot_frame, pilot_labels
            gc.collect()
            refit_train, refit_train_labels = _projected_task_frame(
                train,
                task,
                model_columns,
            )
            refit_frame = pd.concat([refit_train, valid_frame], ignore_index=True)
            refit_labels = np.concatenate([refit_train_labels, valid_labels])
            del refit_train, refit_train_labels, valid_frame, valid_labels
            gc.collect()
            print(
                f"refitting BC-POT-R {candidate} {task} for {refit_iterations} iterations",
                flush=True,
            )
            model = fit_boundary_catboost(
                refit_frame,
                refit_labels,
                task=cast(Any, task),
                candidate=candidate,
                capacity_features=settings["capacity"],
                boundary_features=settings["boundary"],
                params={**parameters, "iterations": refit_iterations},
                flare_features=flare_features,
            )
            artifact = _atomic_joblib(
                model,
                run_dir / "models" / f"{candidate}_{task}.joblib",
            )
            models[candidate][task] = artifact
            importance = np.asarray(model.estimator.get_feature_importance(), dtype=np.float64)
            names = list(model.estimator.feature_names_)
            order = np.argsort(-importance)
            model_records.append(
                {
                    "candidate": candidate,
                    "task": task,
                    **artifact,
                    "refit_iterations": refit_iterations,
                    "training_rows": len(refit_frame),
                    "positive_rows": int(refit_labels.sum()),
                    "capacity_feature_count": len(settings["capacity"]),
                    "boundary_feature_count": len(settings["boundary"]),
                    "fit_seconds": time.perf_counter() - fit_started,
                    "top_30_feature_importance": [
                        {
                            "feature": names[index],
                            "importance": float(importance[index]),
                        }
                        for index in order[:30]
                    ],
                }
            )
            del model, refit_frame, refit_labels
            gc.collect()
    del train, validation, train_dates, pilot_mask
    gc.collect()

    print("predicting BC-POT-R Q4 2024 selection period", flush=True)
    selection_parts: list[pd.DataFrame] = []
    raw_selection_records: list[dict[str, Any]] = []
    prediction_source_columns = [
        "sample_id",
        "FlightDate",
        "Year",
        "Month",
        "Origin",
        "Dest",
        "ArrDel15",
        "Cancelled",
        "delay_label_observed",
        "joint_label_observed",
        "disruption_state",
        BOUNDARY_GATING_FEATURE,
        *BOUNDARY_OBSERVATION_FEATURES,
    ]
    prediction_columns = [*prediction_source_columns, BOUNDARY_ANY_RESIDUAL_COLUMN]
    for month in SELECTION_MONTHS:
        frame = _load_month(
            year=2024,
            month=month,
            limit=None,
            capacity_feature_dir=capacity_feature_dir,
            boundary_feature_dir=boundary_feature_dir,
            loader_arguments=loader_arguments,
        )
        prediction = frame.loc[:, prediction_source_columns].copy()
        prediction[BOUNDARY_ANY_RESIDUAL_COLUMN] = _boundary_any_residual_mask(frame)
        for candidate in CANDIDATES:
            for task in TASKS:
                model = _load_model(models[candidate][task])
                prediction[f"raw_{candidate}_{task}"] = model.predict_proba(frame).astype("float32")
                del model
                gc.collect()
        raw_selection_records.append(
            {
                "year": 2024,
                "month": month,
                **_atomic_parquet(
                    prediction,
                    run_dir / "predictions" / f"raw_selection_2024_{month:02d}.parquet",
                ),
            }
        )
        selection_parts.append(prediction)
        del frame, prediction
        gc.collect()
    selection = pd.concat(selection_parts, ignore_index=True)
    del selection_parts
    gc.collect()

    calibrators: dict[BoundaryCandidate, dict[str, Any]] = {
        candidate: {} for candidate in CANDIDATES
    }
    calibration_records: list[dict[str, Any]] = []
    common_crossfit = np.ones(len(selection), dtype=np.bool_)
    for candidate in CANDIDATES:
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
                    "method": selected.method,
                    "final_fit_through": selected.final_fit_through,
                    "candidate_methods": list(selected.candidate_records),
                    **artifact,
                }
            )
    crossfit = selection.loc[common_crossfit].reset_index(drop=True)
    parent_selection_record = dict(parent["selection_crossfit_prediction_artifact"])
    crossfit = _join_reference(
        crossfit,
        parent_selection_record,
        methods=("flare24", "capacity_gated_simplex"),
        role="parent Q4 cross-fit predictions",
    )
    selection_methods: dict[str, NDArray[np.float64]] = {
        "flare24": _probability_frame(crossfit, "flare24"),
        "capacity_gated_simplex": _probability_frame(crossfit, "capacity_gated_simplex"),
    }
    for candidate in CANDIDATES:
        probabilities = hurdle_joint_probabilities(
            crossfit[f"xfit_{candidate}_cancellation"],
            crossfit[f"xfit_{candidate}_delay"],
        )
        selection_methods[candidate] = probabilities
        _set_probability_frame(crossfit, candidate, probabilities)
    selection_observed = crossfit["joint_label_observed"].eq(1).to_numpy()
    selection_labels = crossfit.loc[selection_observed, "disruption_state"].to_numpy(dtype=np.int64)
    ensemble_selection = _select_simplex(
        selection_labels,
        {name: selection_methods[name][selection_observed] for name in ENSEMBLE_MEMBERS},
    )
    boundary_ensemble = _apply_simplex(
        {name: selection_methods[name] for name in ENSEMBLE_MEMBERS},
        ensemble_selection["weights"],
    )
    selection_methods["boundary_ensemble"] = boundary_ensemble
    _set_probability_frame(crossfit, "boundary_ensemble", boundary_ensemble)
    boundary_gated_selection = _select_boundary_gated_simplex(
        selection_labels,
        {name: selection_methods[name][selection_observed] for name in ENSEMBLE_MEMBERS},
        crossfit.loc[selection_observed, BOUNDARY_GATING_FEATURE].to_numpy(dtype=np.float64),
        cutpoint_values=crossfit[BOUNDARY_GATING_FEATURE].to_numpy(dtype=np.float64),
    )
    boundary_gated_ensemble = _apply_boundary_gated_simplex(
        {name: selection_methods[name] for name in ENSEMBLE_MEMBERS},
        crossfit[BOUNDARY_GATING_FEATURE].to_numpy(dtype=np.float64),
        boundary_gated_selection,
    )
    selection_methods["boundary_gated_ensemble"] = boundary_gated_ensemble
    _set_probability_frame(
        crossfit,
        "boundary_gated_ensemble",
        boundary_gated_ensemble,
    )
    selection_scores: dict[str, Any] = {}
    for name, probabilities in selection_methods.items():
        log_rows, brier_rows = joint_loss_rows(selection_labels, probabilities[selection_observed])
        selection_scores[name] = {
            "joint_log_loss": float(log_rows.mean()),
            "multiclass_brier": float(brier_rows.mean()),
            **_decision_metrics(selection_labels, probabilities[selection_observed]),
        }
    selected_method = min(
        ("boundary_ensemble", "boundary_gated_ensemble"),
        key=lambda name: (float(selection_scores[name]["joint_log_loss"]), name),
    )
    accuracy_bias = _select_accuracy_bias(
        selection_labels,
        selection_methods[selected_method][selection_observed],
    )
    selection_artifact = _atomic_parquet(
        crossfit.loc[
            :,
            [
                *prediction_columns,
                *[
                    column
                    for name in (
                        "flare24",
                        "capacity_gated_simplex",
                        *CANDIDATES,
                        "boundary_ensemble",
                        "boundary_gated_ensemble",
                    )
                    for column in _joint_columns(name)
                ],
            ],
        ],
        run_dir / "predictions" / "selection_2024_crossfit_joint.parquet",
    )
    del selection, crossfit, selection_methods, boundary_ensemble
    del boundary_gated_ensemble
    gc.collect()

    _verify_unchanged(provenance_at_start, phase="2025 retrospective evaluation")
    print("evaluating locked BC-POT-R design on retrospective 2025", flush=True)
    parent_audit = _records_by_month(parent, "retrospective_prediction_artifacts")
    meta_audit = _records_by_month(meta, "retrospective_prediction_artifacts")
    schedule_audit = _records_by_month(baseline_audit, "prediction_artifacts")
    audit_parts: list[pd.DataFrame] = []
    audit_records: list[dict[str, Any]] = []
    for month in range(1, 13):
        frame = _load_month(
            year=2025,
            month=month,
            limit=None,
            capacity_feature_dir=capacity_feature_dir,
            boundary_feature_dir=boundary_feature_dir,
            loader_arguments=loader_arguments,
        )
        prediction = frame.loc[:, prediction_source_columns].copy()
        prediction[BOUNDARY_ANY_RESIDUAL_COLUMN] = _boundary_any_residual_mask(frame)
        month_methods: dict[str, NDArray[np.float64]] = {}
        for candidate in CANDIDATES:
            raw: dict[str, NDArray[np.float64]] = {}
            for task in TASKS:
                model = _load_model(models[candidate][task])
                raw_values = model.predict_proba(frame)
                calibrator = calibrators[candidate][task]
                raw[task] = calibrator.predict(raw_values)
                del model
                gc.collect()
            month_methods[candidate] = hurdle_joint_probabilities(raw["cancellation"], raw["delay"])
            _set_probability_frame(prediction, candidate, month_methods[candidate])
        prediction = _join_reference(
            prediction,
            parent_audit[month],
            methods=("flare24", "capacity_gated_simplex"),
            role=f"parent retrospective 2025-{month:02d}",
        )
        prediction = _join_meta_reference(prediction, meta_audit[month])
        prediction = _join_schedule_reference(prediction, schedule_audit[month])
        ensemble_inputs = {
            "capacity_gated_simplex": _probability_frame(prediction, "capacity_gated_simplex"),
            "boundary_only": month_methods["boundary_only"],
            "counterfactual_residual": month_methods["counterfactual_residual"],
        }
        month_methods["boundary_ensemble"] = _apply_simplex(
            ensemble_inputs,
            ensemble_selection["weights"],
        )
        _set_probability_frame(
            prediction,
            "boundary_ensemble",
            month_methods["boundary_ensemble"],
        )
        month_methods["boundary_gated_ensemble"] = _apply_boundary_gated_simplex(
            ensemble_inputs,
            prediction[BOUNDARY_GATING_FEATURE].to_numpy(dtype=np.float64),
            boundary_gated_selection,
        )
        _set_probability_frame(
            prediction,
            "boundary_gated_ensemble",
            month_methods["boundary_gated_ensemble"],
        )
        audit_records.append(
            {
                "year": 2025,
                "month": month,
                **_atomic_parquet(
                    prediction,
                    run_dir / "predictions" / f"retrospective_2025_{month:02d}.parquet",
                ),
            }
        )
        audit_parts.append(prediction)
        del frame, prediction, month_methods
        gc.collect()
    audit = pd.concat(audit_parts, ignore_index=True)
    del audit_parts
    gc.collect()
    dates = pd.to_datetime(audit["FlightDate"], errors="raise").dt.normalize()
    primary_mask = dates.between(
        pd.Timestamp(PRIMARY_AUDIT_START), pd.Timestamp(PRIMARY_AUDIT_END)
    ).to_numpy()
    primary = audit.loc[primary_mask].reset_index(drop=True)
    method_names = (
        "schedule_baseline",
        "flare24",
        "capacity_gated_simplex",
        "meta_current",
        *CANDIDATES,
        "boundary_ensemble",
        "boundary_gated_ensemble",
    )
    primary_methods = {name: _probability_frame(primary, name) for name in method_names}
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
    full_methods = {name: _probability_frame(audit, name) for name in method_names}
    full_scores: dict[str, Any] = {}
    full_labels = audit.loc[audit["joint_label_observed"].eq(1), "disruption_state"].to_numpy(
        dtype=np.int64
    )
    full_observed = audit["joint_label_observed"].eq(1).to_numpy()
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
    primary_labels = primary.loc[primary_observed, "disruption_state"].to_numpy(dtype=np.int64)
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
        primary,
        primary_methods,
        boundary_gated_selection,
    )
    airport_scores = _airport_scores(
        primary,
        primary_methods,
        selected_method=selected_method,
    )
    primary_prevalence = {
        state: float((primary_labels == index).mean())
        for index, state in enumerate(PROBABILITY_STATES)
    }

    final_provenance = _verify_unchanged(provenance_at_start, phase="final report")
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_BCPOTR_2024_SELECTION_2025_RETROSPECTIVE_REDEVELOPMENT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": protocol["identity"]["expanded_name"],
        "epistemic_status": protocol["epistemic_status"],
        "protocol": {
            "path": protocol_path.as_posix(),
            "bytes": protocol_path.stat().st_size,
            "sha256": sha256_file(protocol_path),
        },
        "method_lock": {
            **_artifact_record(lock_path),
            "self_hash": method_lock["manifest_sha256"],
        },
        "boundary_inputs": boundary_inputs,
        "boundary_rotation_outcome_blind_gate": boundary_rotation_gate,
        "boundary_context_totals": {
            "rows": boundary_manifest["rows"],
            "resource_nodes_computed_not_persisted": boundary_manifest["resource_nodes"],
            "rotation_edges_computed_not_persisted": boundary_manifest["rotation_edges"],
        },
        "parent_report": parent_record,
        "meta_report": meta_record,
        "baseline_selection_report": baseline_record,
        "baseline_audit_report": baseline_audit_record,
        "feature_sets": {
            candidate: {name: list(values) for name, values in settings.items()}
            for candidate, settings in feature_sets.items()
        },
        "model_artifacts": model_records,
        "raw_selection_prediction_artifacts": raw_selection_records,
        "calibration_artifacts": calibration_records,
        "selection_crossfit_prediction_artifact": selection_artifact,
        "selection_scores": selection_scores,
        "ensemble_selection": ensemble_selection,
        "boundary_gated_ensemble_selection": boundary_gated_selection,
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
        "primary_boundary_signal_summary": _boundary_signal_summary(primary),
        "primary_boundary_signal_outcome_associations": _boundary_signal_outcome_analysis(
            primary,
            boundary_gated_selection,
            repetitions=bootstrap_repetitions,
            seed=seed,
        ),
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
    write_canonical_json(output_path, report)
    write_canonical_json(run_dir / "run_manifest.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--boundary-manifest", type=Path, required=True)
    parser.add_argument("--boundary-validation", type=Path, required=True)
    parser.add_argument("--boundary-rotation-gate", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--meta-report", type=Path, required=True)
    parser.add_argument("--baseline-selection-report", type=Path, required=True)
    parser.add_argument("--baseline-audit-report", type=Path, required=True)
    parser.add_argument("--census-dir", type=Path, required=True)
    parser.add_argument("--recent-dir", type=Path, required=True)
    parser.add_argument("--flight-recent-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--weather-feature-dir", type=Path, required=True)
    parser.add_argument("--rotation-feature-dir", type=Path, required=True)
    parser.add_argument("--capacity-feature-dir", type=Path, required=True)
    parser.add_argument("--boundary-feature-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_boundary_study(
        protocol_path=args.protocol,
        boundary_manifest_path=args.boundary_manifest,
        boundary_validation_path=args.boundary_validation,
        boundary_rotation_gate_path=args.boundary_rotation_gate,
        parent_report_path=args.parent_report,
        meta_report_path=args.meta_report,
        baseline_selection_report_path=args.baseline_selection_report,
        baseline_audit_report_path=args.baseline_audit_report,
        census_dir=args.census_dir,
        recent_dir=args.recent_dir,
        flight_recent_dir=args.flight_recent_dir,
        graph_dir=args.graph_dir,
        weather_feature_dir=args.weather_feature_dir,
        rotation_feature_dir=args.rotation_feature_dir,
        capacity_feature_dir=args.capacity_feature_dir,
        boundary_feature_dir=args.boundary_feature_dir,
        run_dir=args.run_dir,
        output_path=args.output,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "breakthrough_gate": report["breakthrough_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
