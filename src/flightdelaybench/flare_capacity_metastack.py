"""Post-hoc cancellation logit meta-stack for TF-CC-RTH.

Coefficients are fit on Q4 2024 forward predictions.  A fixed ridge grid is
selected on January--June 2025, July 1--2 are embargoed, and July 3--December
29 are scored held-forward.  Earlier inspection of 2025 results motivated this
track, so neither half-year is described as epistemically blind.  The selected
method is frozen for genuinely unopened 2026 confirmation.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
import tomllib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression

from .flare_capacity_factorized import (
    _atomic_json,
    _load_prediction_frame,
    _probability_methods,
    _verify_parent_inputs,
)
from .flare_capacity_study import (
    BLEND_CANDIDATES,
    GATING_FEATURE,
    PRIMARY_AUDIT_END,
    PRIMARY_AUDIT_START,
    _airport_scores,
    _atomic_joblib,
    _atomic_parquet,
    _joint_columns,
    _joint_point_scores,
    _monthly_scores,
    _read_self_hashed,
    _regime_scores,
    _set_joint_columns,
)
from .flare_evaluation import evaluate_joint_probabilities, joint_loss_rows
from .flare_reconciliation import (
    hurdle_joint_probabilities,
    joint_binary_marginals,
)
from .flare_study import PREDICTION_ID_COLUMNS
from .hashing import canonical_json_sha256, sha256_file
from .provenance import capture_provenance

META_METHOD = "tf_ccrth_cancellation_logit_stack"
META_PROBABILITY_COLUMN = "prob_tf_ccrth_cancellation_logit_stack_cancellation"
SELECTION_START = "2025-01-03"
SELECTION_END = "2025-06-30"
H2_EMBARGO = ("2025-07-01", "2025-07-02")
HELD_FORWARD_START = "2025-07-03"
HELD_FORWARD_END = "2025-12-29"
CONFIRMATION_YEAR = 2026
EXPECTED_FACTORIZED_STATUS = "COMPLETE_TF_CCRTH_POST_HOC_2025_ANALYSIS_2026_UNOPENED"
EXPECTED_FACTORIZED_VALIDATION_STATUS = "PASS_TF_CCRTH_STUDY_VALIDATION"


def _frozen_metastack_implementation_files() -> tuple[Path, ...]:
    directory = Path(__file__).parent
    return tuple(
        directory / name
        for name in (
            "flare_capacity_metastack.py",
            "flare_capacity_factorized.py",
            "flare_capacity_study.py",
            "flare_evaluation.py",
            "flare_reconciliation.py",
            "flare_study.py",
            "bootstrap.py",
            "metrics.py",
            "hashing.py",
            "provenance.py",
        )
    )


def _verify_implementation_unchanged(
    started: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    current = capture_provenance(_frozen_metastack_implementation_files())
    if current.get("git_head") != started.get("git_head") or current.get(
        "source_files"
    ) != started.get("source_files"):
        raise RuntimeError(
            "TF-CC-RTH meta-stack implementation changed before "
            f"{phase}; retain this run and restart with a new version"
        )
    return current


def _load_protocol(
    path: Path,
    *,
    repetitions: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open("rb") as handle:
        protocol: dict[str, Any] = tomllib.load(handle)
    identity = protocol.get("identity", {})
    boundary = protocol.get("information_boundary", {})
    meta = protocol.get("meta_model", {})
    selection = protocol.get("selection", {})
    evaluation = protocol.get("evaluation", {})
    if (
        identity.get("method") != "TF-CC-RTH-LogitStack-v1"
        or identity.get("status") != "POST_HOC_2025_DEVELOPMENT_FOR_UNOPENED_2026_CONFIRMATION"
        or int(boundary.get("coefficient_fit_year", -1)) != 2024
        or int(boundary.get("regularization_selection_year", -1)) != 2025
        or int(boundary.get("held_forward_evaluation_year", -1)) != 2025
        or int(boundary.get("confirmation_year", -1)) != CONFIRMATION_YEAR
        or boundary.get("prior_2025_results_informed_design") is not True
        or boundary.get("confirmation_outcomes_accessed") is not False
        or meta.get("task") != "cancellation"
        or tuple(meta.get("input_candidates", ())) != BLEND_CANDIDATES
        or float(meta.get("probability_clip", -1.0)) != 1e-6
        or meta.get("family") != "L2-regularized logistic regression"
        or meta.get("solver") != "lbfgs"
        or int(meta.get("maximum_iterations", -1)) != 800
        or float(meta.get("tolerance", -1.0)) != 1e-9
        or tuple(meta.get("regularization_c_grid", ()))
        != (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
        or tuple(meta.get("coefficient_fit_dates", ())) != ("2024-10-03", "2024-12-31")
        or tuple(selection.get("dates", ())) != (SELECTION_START, SELECTION_END)
        or tuple(selection.get("h2_embargo_dates", ())) != H2_EMBARGO
        or tuple(evaluation.get("held_forward_dates", ())) != (HELD_FORWARD_START, HELD_FORWARD_END)
        or tuple(evaluation.get("descriptive_full_primary_dates", ()))
        != (PRIMARY_AUDIT_START, PRIMARY_AUDIT_END)
        or int(evaluation.get("bootstrap_repetitions", -1)) != repetitions
        or int(evaluation.get("bootstrap_seed", -1)) != seed
    ):
        raise ValueError("meta-stack executable settings differ from the frozen protocol")
    return protocol, {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "status": identity["status"],
    }


def _verify_factorized_context(
    report_path: Path,
    validation_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report, report_key = _read_self_hashed(report_path)
    validation, validation_key = _read_self_hashed(validation_path)
    if (
        report.get("status") != EXPECTED_FACTORIZED_STATUS
        or validation.get("status") != EXPECTED_FACTORIZED_VALIDATION_STATUS
        or report.get("outcomes_accessed", {}).get("2026_accessed") is not False
        or validation.get("confirmation_outcomes_accessed") is not False
    ):
        raise ValueError("meta-stack requires validated TF-CC-RTH context")
    bound = validation.get("report", {})
    if (
        Path(str(bound.get("path", ""))).resolve() != report_path.resolve()
        or bound.get("sha256") != sha256_file(report_path)
        or bound.get("self_hash") != report[report_key]
    ):
        raise ValueError("TF-CC-RTH validation is not bound to its report")
    return {
        "path": report_path.as_posix(),
        "bytes": report_path.stat().st_size,
        "sha256": sha256_file(report_path),
        "self_hash_key": report_key,
        "self_hash": report[report_key],
    }, {
        "path": validation_path.as_posix(),
        "bytes": validation_path.stat().st_size,
        "sha256": sha256_file(validation_path),
        "self_hash_key": validation_key,
        "self_hash": validation[validation_key],
        "status": validation["status"],
    }


def cancellation_logit_design(
    probabilities: Mapping[str, NDArray[np.float64]],
    *,
    clip: float = 1e-6,
) -> NDArray[np.float64]:
    """Return registered candidate cancellation logits in frozen column order."""

    if tuple(probabilities) != BLEND_CANDIDATES:
        raise ValueError("meta-stack probability candidates are out of order")
    if not 0.0 < clip < 0.5:
        raise ValueError("meta-stack clipping must lie in (0, 0.5)")
    columns: list[NDArray[np.float64]] = []
    row_count: int | None = None
    for name in BLEND_CANDIDATES:
        joint = np.asarray(probabilities[name], dtype=np.float64)
        if joint.ndim != 2 or joint.shape[1] != 3:
            raise ValueError(f"meta-stack {name} must be an n-by-3 joint matrix")
        if row_count is None:
            row_count = len(joint)
        if len(joint) != row_count:
            raise ValueError("meta-stack joint matrices do not align")
        view = joint_binary_marginals(joint)
        values = np.clip(view["cancel_probability"], clip, 1.0 - clip)
        columns.append(np.log(values / (1.0 - values)))
    return np.column_stack(columns)


def fit_cancellation_metastack(
    design: ArrayLike,
    labels: ArrayLike,
    *,
    regularization_c: float,
    maximum_iterations: int = 800,
    tolerance: float = 1e-9,
) -> LogisticRegression:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if (
        x.ndim != 2
        or x.shape != (len(y), len(BLEND_CANDIDATES))
        or len(y) == 0
        or not np.isfinite(x).all()
        or not np.isin(y, [0, 1]).all()
        or np.unique(y).size != 2
        or not np.isfinite(regularization_c)
        or regularization_c <= 0.0
    ):
        raise ValueError("meta-stack fit inputs are invalid")
    model = LogisticRegression(
        C=float(regularization_c),
        solver="lbfgs",
        max_iter=maximum_iterations,
        tol=tolerance,
    )
    model.fit(x, y)
    if int(model.n_iter_[0]) >= maximum_iterations:
        raise RuntimeError("meta-stack logistic fit reached its iteration limit")
    return model


def _model_record(
    model: LogisticRegression,
    *,
    regularization_c: float,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    return {
        "regularization_c": regularization_c,
        "feature_order": list(BLEND_CANDIDATES),
        "intercept": float(model.intercept_[0]),
        "coefficients": {
            name: float(value) for name, value in zip(BLEND_CANDIDATES, model.coef_[0], strict=True)
        },
        "iterations": int(model.n_iter_[0]),
        **artifact,
    }


def _c_key(value: float) -> str:
    return f"c_{value:.4g}".replace(".", "p")


def _metastack_joint(
    model: LogisticRegression,
    base: Mapping[str, NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    design = cancellation_logit_design(base)
    cancel = np.asarray(model.predict_proba(design)[:, 1], dtype=np.float64)
    flare_views = joint_binary_marginals(base["flare24"])
    joint = hurdle_joint_probabilities(
        cancel,
        flare_views["delay_given_operated_probability"],
    )
    return cancel, joint


def _grid_point_scores(
    frame: pd.DataFrame,
    predictions: Mapping[str, NDArray[np.float64]],
    flare: NDArray[np.float64],
) -> dict[str, dict[str, float | int]]:
    observed = frame["joint_label_observed"].eq(1).to_numpy(dtype=np.bool_)
    labels = frame.loc[observed, "disruption_state"].to_numpy(dtype=np.int64)
    result: dict[str, dict[str, float | int]] = {}
    for name, values in predictions.items():
        log_rows, brier_rows = joint_loss_rows(labels, values[observed])
        result[name] = {
            "n": len(labels),
            "joint_log_loss": float(log_rows.mean()),
            "multiclass_brier": float(brier_rows.mean()),
        }
    flare_log, flare_brier = joint_loss_rows(labels, flare[observed])
    result["flare24"] = {
        "n": len(labels),
        "joint_log_loss": float(flare_log.mean()),
        "multiclass_brier": float(flare_brier.mean()),
    }
    return result


def run_cancellation_metastack_study(
    *,
    protocol_path: Path,
    parent_report_path: Path,
    parent_validation_path: Path,
    factorized_report_path: Path,
    factorized_validation_path: Path,
    run_dir: Path,
    grid_lock_path: Path,
    held_forward_lock_path: Path,
    confirmation_lock_path: Path,
    output_path: Path,
    bootstrap_repetitions: int = 2_000,
    seed: int = 20260905,
) -> dict[str, Any]:
    started = time.perf_counter()
    destinations = (
        run_dir,
        grid_lock_path,
        held_forward_lock_path,
        confirmation_lock_path,
        output_path,
    )
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite meta-stack artifacts: {existing}")
    protocol, protocol_record = _load_protocol(
        protocol_path, repetitions=bootstrap_repetitions, seed=seed
    )
    parent_report, parent_lock, parent_record, parent_validation_record = _verify_parent_inputs(
        parent_report_path, parent_validation_path
    )
    factorized_record, factorized_validation_record = _verify_factorized_context(
        factorized_report_path, factorized_validation_path
    )
    implementation_at_start = capture_provenance(_frozen_metastack_implementation_files())
    run_dir.mkdir(parents=True)

    selection_frame, q4_input_record = _load_prediction_frame(
        dict(parent_report["selection_crossfit_prediction_artifact"]),
        role="meta-stack Q4 coefficient-fit predictions",
        expected_year=2024,
        expected_month=None,
    )
    selection_dates = pd.to_datetime(selection_frame["FlightDate"], errors="raise")
    fit_mask = selection_dates.between("2024-10-03", "2024-12-31").to_numpy()
    if not fit_mask.any():
        raise ValueError("meta-stack Q4 coefficient fit has no rows")
    q4_normalization: list[dict[str, Any]] = []
    q4_all = _probability_methods(
        selection_frame,
        normalization_audits=q4_normalization,
        role="meta-stack checksum-loaded Q4 probabilities",
    )
    q4_base: dict[str, NDArray[np.float64]] = {
        name: q4_all[name][fit_mask] for name in BLEND_CANDIDATES
    }
    q4_frame = selection_frame.loc[fit_mask].reset_index(drop=True)
    design = cancellation_logit_design(q4_base)
    labels = q4_frame["Cancelled"].to_numpy(dtype=np.int64)
    c_grid = tuple(float(value) for value in protocol["meta_model"]["regularization_c_grid"])
    models: dict[str, LogisticRegression] = {}
    model_records: list[dict[str, Any]] = []
    q4_predictions: dict[str, NDArray[np.float64]] = {}
    for regularization_c in c_grid:
        name = _c_key(regularization_c)
        model = fit_cancellation_metastack(
            design,
            labels,
            regularization_c=regularization_c,
            maximum_iterations=int(protocol["meta_model"]["maximum_iterations"]),
            tolerance=float(protocol["meta_model"]["tolerance"]),
        )
        models[name] = model
        artifact = _atomic_joblib(
            model,
            run_dir / "models" / f"cancellation_logit_stack_{name}.joblib",
        )
        model_records.append(
            _model_record(
                model,
                regularization_c=regularization_c,
                artifact=artifact,
            )
        )
        _, q4_predictions[name] = _metastack_joint(model, q4_base)
    q4_scores = _grid_point_scores(q4_frame, q4_predictions, q4_base["flare24"])
    grid_lock: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED_METASTACK_CANDIDATE_GRID_BEFORE_FORMAL_2025_REANALYSIS",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "TF-CC-RTH-LogitStack-v1",
        "protocol": protocol_record,
        "parent_report": parent_record,
        "parent_validation": parent_validation_record,
        "factorized_context": factorized_record,
        "factorized_validation": factorized_validation_record,
        "q4_input": q4_input_record,
        "model_artifacts": model_records,
        "q4_fit_scores": q4_scores,
        "information_boundary": {
            "2024_row_level_predictions_loaded": True,
            "prior_2025_results_known_from_preliminary_exploration": True,
            "formal_2025_partitions_loaded_by_this_run": False,
            "2026_outcomes_accessed": False,
        },
        "epistemic_status": (
            "The exact grid is fixed before this formal rerun loads 2025 partitions, "
            "but it was designed after exploratory 2025 inspection. This is a workflow "
            "lock, not a claim of epistemic blindness."
        ),
        "provenance": _verify_implementation_unchanged(
            implementation_at_start, phase="meta-stack grid lock"
        ),
    }
    grid_lock["manifest_sha256"] = canonical_json_sha256(grid_lock)
    _atomic_json(grid_lock_path, grid_lock)
    print("locked meta-stack grid; no 2026 outcomes accessed", flush=True)

    retrospective_records = sorted(
        (dict(record) for record in parent_report["retrospective_prediction_artifacts"]),
        key=lambda record: int(record["month"]),
    )
    if [int(record["month"]) for record in retrospective_records] != list(range(1, 13)):
        raise ValueError("meta-stack parent does not contain all 2025 months")
    first_half_parts: list[pd.DataFrame] = []
    input_records: list[dict[str, Any]] = []
    for record in retrospective_records[:6]:
        month = int(record["month"])
        print(f"meta-stack H1 selection source 2025-{month:02d}", flush=True)
        frame, verified = _load_prediction_frame(
            record,
            role=f"meta-stack H1 2025-{month:02d}",
            expected_year=2025,
            expected_month=month,
        )
        normalization: list[dict[str, Any]] = []
        all_methods = _probability_methods(
            frame,
            normalization_audits=normalization,
            role=f"meta-stack H1 2025-{month:02d} base probabilities",
        )
        base: dict[str, NDArray[np.float64]] = {
            name: all_methods[name] for name in BLEND_CANDIDATES
        }
        reduced = frame.loc[:, list(PREDICTION_ID_COLUMNS)].copy()
        reduced[GATING_FEATURE] = frame[GATING_FEATURE].to_numpy(dtype=np.float32)
        _set_joint_columns(reduced, "flare24", base["flare24"])
        for name, model in models.items():
            cancel, joint = _metastack_joint(model, base)
            reduced[f"meta_cancel_{name}"] = cancel
            _set_joint_columns(reduced, name, joint)
        first_half_parts.append(reduced)
        input_records.append({**verified, "probability_normalization": normalization})
        del frame, all_methods, base, reduced
        gc.collect()
    first_half = pd.concat(first_half_parts, ignore_index=True)
    del first_half_parts
    h1_dates = pd.to_datetime(first_half["FlightDate"], errors="raise")
    h1_mask = h1_dates.between(SELECTION_START, SELECTION_END).to_numpy()
    h1 = first_half.loc[h1_mask].reset_index(drop=True)
    h1_flare = np.column_stack(
        [h1[column].to_numpy(dtype=np.float64) for column in _joint_columns("flare24")]
    )
    h1_flare /= h1_flare.sum(axis=1, keepdims=True)
    h1_predictions: dict[str, NDArray[np.float64]] = {}
    for regularization_c in c_grid:
        name = _c_key(regularization_c)
        values = np.column_stack(
            [h1[column].to_numpy(dtype=np.float64) for column in _joint_columns(name)]
        )
        values /= values.sum(axis=1, keepdims=True)
        h1_predictions[name] = values
    h1_scores = _grid_point_scores(h1, h1_predictions, h1_flare)
    c_by_name = {_c_key(value): value for value in c_grid}
    selected_name = min(
        c_by_name,
        key=lambda name: (
            float(h1_scores[name]["joint_log_loss"]),
            c_grid.index(c_by_name[name]),
        ),
    )
    selected_c = c_by_name[selected_name]
    selected_model = models[selected_name]
    held_forward_lock: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED_METASTACK_AFTER_H1_BEFORE_FORMAL_H2_PARTITION_LOAD",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "TF-CC-RTH-LogitStack-v1",
        "grid_lock": {
            "path": grid_lock_path.as_posix(),
            "sha256": sha256_file(grid_lock_path),
            "self_hash": grid_lock["manifest_sha256"],
        },
        "h1_selection_dates": [SELECTION_START, SELECTION_END],
        "h1_candidate_scores": h1_scores,
        "selected_model": selected_name,
        "selected_regularization_c": selected_c,
        "selected_model_artifact": next(
            record for record in model_records if record["regularization_c"] == selected_c
        ),
        "embargo_dates": list(H2_EMBARGO),
        "held_forward_dates": [HELD_FORWARD_START, HELD_FORWARD_END],
        "information_boundary": {
            "formal_h1_2025_partitions_loaded": True,
            "formal_h2_2025_partitions_loaded_before_lock": False,
            "prior_full_2025_results_known_from_preliminary_exploration": True,
            "2026_outcomes_accessed": False,
        },
        "epistemic_status": (
            "H2 is execution-held-forward but not epistemically blind because earlier "
            "exploration had already inspected 2025. The split measures temporal "
            "stability and cannot replace unopened-2026 confirmation."
        ),
        "provenance": _verify_implementation_unchanged(
            implementation_at_start, phase="meta-stack H2 lock"
        ),
    }
    held_forward_lock["manifest_sha256"] = canonical_json_sha256(held_forward_lock)
    _atomic_json(held_forward_lock_path, held_forward_lock)
    print(
        f"locked meta-stack H2 method; selected={selected_name}; no 2026 outcomes accessed",
        flush=True,
    )

    # Retain only the selected candidate before loading H2.  This bounds peak
    # memory without changing any prediction or recorded H1 grid score.
    first_half[META_PROBABILITY_COLUMN] = first_half[f"meta_cancel_{selected_name}"]
    for target, source in zip(
        _joint_columns(META_METHOD), _joint_columns(selected_name), strict=True
    ):
        first_half[target] = first_half[source]
    retained_columns = [
        *PREDICTION_ID_COLUMNS,
        GATING_FEATURE,
        *_joint_columns("flare24"),
        META_PROBABILITY_COLUMN,
        *_joint_columns(META_METHOD),
    ]
    first_half = first_half.loc[:, retained_columns].copy()
    del h1, h1_predictions
    gc.collect()

    all_parts = [first_half]
    for record in retrospective_records[6:]:
        month = int(record["month"])
        print(f"meta-stack H2 held-forward source 2025-{month:02d}", flush=True)
        frame, verified = _load_prediction_frame(
            record,
            role=f"meta-stack H2 2025-{month:02d}",
            expected_year=2025,
            expected_month=month,
        )
        normalization = []
        all_methods = _probability_methods(
            frame,
            normalization_audits=normalization,
            role=f"meta-stack H2 2025-{month:02d} base probabilities",
        )
        base = {name: all_methods[name] for name in BLEND_CANDIDATES}
        cancel, joint = _metastack_joint(selected_model, base)
        reduced = frame.loc[:, list(PREDICTION_ID_COLUMNS)].copy()
        reduced[GATING_FEATURE] = frame[GATING_FEATURE].to_numpy(dtype=np.float32)
        _set_joint_columns(reduced, "flare24", base["flare24"])
        reduced[META_PROBABILITY_COLUMN] = cancel
        _set_joint_columns(reduced, META_METHOD, joint)
        all_parts.append(reduced)
        input_records.append({**verified, "probability_normalization": normalization})
        del frame, all_methods, base, reduced
        gc.collect()
    audit_full = pd.concat(all_parts, ignore_index=True)
    del all_parts, first_half
    if audit_full["sample_id"].duplicated().any():
        raise ValueError("meta-stack 2025 source repeats sample ids")
    if audit_full[[META_PROBABILITY_COLUMN, *_joint_columns(META_METHOD)]].isna().any().any():
        raise RuntimeError("meta-stack assembled predictions contain missing values")
    flare_full = np.column_stack(
        [audit_full[column].to_numpy(dtype=np.float64) for column in _joint_columns("flare24")]
    )
    flare_full /= flare_full.sum(axis=1, keepdims=True)
    meta_full = np.column_stack(
        [audit_full[column].to_numpy(dtype=np.float64) for column in _joint_columns(META_METHOD)]
    )
    meta_full /= meta_full.sum(axis=1, keepdims=True)
    dates = pd.to_datetime(audit_full["FlightDate"], errors="raise")
    full_primary_mask = dates.between(PRIMARY_AUDIT_START, PRIMARY_AUDIT_END).to_numpy()
    held_forward_mask = dates.between(HELD_FORWARD_START, HELD_FORWARD_END).to_numpy()
    full_primary = audit_full.loc[full_primary_mask].reset_index(drop=True)
    held_forward = audit_full.loc[held_forward_mask].reset_index(drop=True)
    full_primary_methods = {
        "flare24": flare_full[full_primary_mask],
        META_METHOD: meta_full[full_primary_mask],
    }
    held_forward_methods = {
        "flare24": flare_full[held_forward_mask],
        META_METHOD: meta_full[held_forward_mask],
    }
    held_forward_evaluation = evaluate_joint_probabilities(
        held_forward,
        cast(dict[str, ArrayLike], held_forward_methods),
        reference_method="flare24",
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=seed,
    )
    descriptive_full_evaluation = evaluate_joint_probabilities(
        full_primary,
        cast(dict[str, ArrayLike], full_primary_methods),
        reference_method="flare24",
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=seed + 10_000,
    )
    output_prediction_records: list[dict[str, Any]] = []
    output_columns = [
        "sample_id",
        "FlightDate",
        "Year",
        "Month",
        META_PROBABILITY_COLUMN,
        *_joint_columns(META_METHOD),
    ]
    for month in range(1, 13):
        month_mask = pd.to_numeric(audit_full["Month"], errors="raise").eq(month)
        artifact = _atomic_parquet(
            audit_full.loc[month_mask, output_columns].reset_index(drop=True),
            run_dir / "predictions" / f"retrospective_2025_{month:02d}.parquet",
        )
        output_prediction_records.append({"year": 2025, "month": month, **artifact})

    selected_model_record = next(
        record for record in model_records if record["regularization_c"] == selected_c
    )
    confirmation_lock: dict[str, Any] = {
        "schema_version": 2,
        "status": "LOCKED_TF_CCRTH_LOGIT_STACK_FOR_UNOPENED_2026_CONFIRMATION",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "TF-CC-RTH-LogitStack-v1",
        "locked_candidate": {
            "name": META_METHOD,
            "task": "cancellation",
            "feature_order": list(BLEND_CANDIDATES),
            "probability_transform": "logit after clipping to [1e-6, 1-1e-6]",
            "regularization_c": selected_c,
            "intercept": selected_model_record["intercept"],
            "coefficients": selected_model_record["coefficients"],
            "delay_given_operated_source": "flare24",
            "hurdle_recombination": True,
        },
        "selection_disclosure": {
            "coefficient_fit_period": ["2024-10-03", "2024-12-31"],
            "regularization_selection_period": [SELECTION_START, SELECTION_END],
            "held_forward_period": [HELD_FORWARD_START, HELD_FORWARD_END],
            "2025_results_informed_architecture_and_grid": True,
            "2025_intervals_are_not_independent_confirmation": True,
        },
        "confirmation_protocol": {
            "year": CONFIRMATION_YEAR,
            "primary_dates": ["2026-01-03", "2026-12-29"],
            "reference": "exact FLARE-24 pipeline under the same T-24 contract",
            "primary_metric": "paired date-cluster joint log-loss difference",
            "co_primary_guardrail": "paired date-cluster multiclass Brier difference",
            "success_rule": "both 95% paired date-cluster upper bounds below zero",
            "bootstrap_repetitions": bootstrap_repetitions,
            "bootstrap_seed": seed + 20_000,
            "no_2026_refit_or_reselection": True,
        },
        "grid_lock": {
            "path": grid_lock_path.as_posix(),
            "sha256": sha256_file(grid_lock_path),
            "self_hash": grid_lock["manifest_sha256"],
        },
        "held_forward_lock": {
            "path": held_forward_lock_path.as_posix(),
            "sha256": sha256_file(held_forward_lock_path),
            "self_hash": held_forward_lock["manifest_sha256"],
        },
        "information_boundary": {
            "2024_outcomes_accessed": True,
            "2025_outcomes_accessed_for_development_and_retrospective_evaluation": True,
            "2026_outcomes_accessed": False,
        },
        "provenance": _verify_implementation_unchanged(
            implementation_at_start, phase="meta-stack 2026 confirmation lock"
        ),
    }
    confirmation_lock["manifest_sha256"] = canonical_json_sha256(confirmation_lock)
    _atomic_json(confirmation_lock_path, confirmation_lock)

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_TF_CCRTH_METASTACK_POST_HOC_2025_2026_UNOPENED",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "Task-Factorized CC-RTH Cancellation Logit Meta-Stack",
        "protocol": protocol_record,
        "parent_report": parent_record,
        "parent_validation": parent_validation_record,
        "factorized_context": factorized_record,
        "factorized_validation": factorized_validation_record,
        "q4_input": q4_input_record,
        "q4_probability_normalization": q4_normalization,
        "model_artifacts": model_records,
        "q4_fit_scores": q4_scores,
        "grid_lock": {
            "path": grid_lock_path.as_posix(),
            "bytes": grid_lock_path.stat().st_size,
            "sha256": sha256_file(grid_lock_path),
            "self_hash": grid_lock["manifest_sha256"],
        },
        "h1_selection_period": [SELECTION_START, SELECTION_END],
        "h1_candidate_scores": h1_scores,
        "selected_model": selected_name,
        "selected_regularization_c": selected_c,
        "held_forward_lock": {
            "path": held_forward_lock_path.as_posix(),
            "bytes": held_forward_lock_path.stat().st_size,
            "sha256": sha256_file(held_forward_lock_path),
            "self_hash": held_forward_lock["manifest_sha256"],
        },
        "h2_embargo_dates": list(H2_EMBARGO),
        "held_forward_evaluation_period": [HELD_FORWARD_START, HELD_FORWARD_END],
        "held_forward_evaluation": held_forward_evaluation,
        "descriptive_full_primary_period": [PRIMARY_AUDIT_START, PRIMARY_AUDIT_END],
        "descriptive_full_primary_evaluation": descriptive_full_evaluation,
        "descriptive_full_primary_joint_scores": _joint_point_scores(
            full_primary, full_primary_methods
        ),
        "descriptive_monthly_scores": _monthly_scores(full_primary, full_primary_methods),
        "descriptive_capacity_regime_scores": _regime_scores(
            full_primary,
            full_primary_methods,
            {"cutpoints": parent_lock["capacity_gated_simplex"]["cutpoints"]},
        ),
        "descriptive_airport_scores": _airport_scores(full_primary, full_primary_methods),
        "retrospective_source_artifacts": input_records,
        "retrospective_prediction_artifacts": output_prediction_records,
        "confirmation_lock": {
            "path": confirmation_lock_path.as_posix(),
            "bytes": confirmation_lock_path.stat().st_size,
            "sha256": sha256_file(confirmation_lock_path),
            "self_hash": confirmation_lock["manifest_sha256"],
            "2026_outcomes_accessed": False,
        },
        "outcomes_accessed": {
            "coefficient_fit_years": [2024],
            "development_and_retrospective_years": [2025],
            "2026_accessed": False,
        },
        "claim_limits": [
            "The architecture and regularization grid were designed after exploratory 2025 inspection.",
            "H2 was held forward by execution order but was not epistemically blind to the researchers.",
            "The H2 interval is not adjusted for prior exploratory architecture choices.",
            "Independent confirmation is reserved for the unopened 2026 lock.",
            "No causal, physical-capacity, safety, production, or state-of-the-art claim is made.",
        ],
        "versions": {
            "python": platform.python_version(),
            "joblib": version("joblib"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
        },
        "provenance": _verify_implementation_unchanged(
            implementation_at_start, phase="meta-stack final report"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    _atomic_json(output_path, report)
    _atomic_json(run_dir / "run_manifest.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--parent-validation", type=Path, required=True)
    parser.add_argument("--factorized-report", type=Path, required=True)
    parser.add_argument("--factorized-validation", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--grid-lock", type=Path, required=True)
    parser.add_argument("--held-forward-lock", type=Path, required=True)
    parser.add_argument("--confirmation-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_cancellation_metastack_study(
        protocol_path=args.protocol,
        parent_report_path=args.parent_report,
        parent_validation_path=args.parent_validation,
        factorized_report_path=args.factorized_report,
        factorized_validation_path=args.factorized_validation,
        run_dir=args.run_dir,
        grid_lock_path=args.grid_lock,
        held_forward_lock_path=args.held_forward_lock,
        confirmation_lock_path=args.confirmation_lock,
        output_path=args.output,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_model": result["selected_model"],
                "selected_regularization_c": result["selected_regularization_c"],
                "2026_accessed": result["outcomes_accessed"]["2026_accessed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
