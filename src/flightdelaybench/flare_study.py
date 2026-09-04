"""Chronological FLARE-24 selection and retrospective-audit orchestration.

The selection phase fits only January-August 2024, uses September for early
stopping, and reserves Q4 for forward calibration, ensemble, and reconciliation
selection.  The audit phase (implemented below) freezes those choices before it
opens 2025 outcomes.  No path in this module accepts a 2026 partition.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any, Literal, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .calibration import BinaryCalibrator, CalibrationMethod, fit_binary_calibrator
from .census_graph import attach_census_graph_features
from .census_modeling import (
    CENSUS_BASE_CATEGORICAL_FEATURES,
    CENSUS_FLIGHT_CATEGORICAL_FEATURES,
    CENSUS_LOAD_COLUMNS,
)
from .census_recent import attach_census_recent_features
from .flare_aggregate import HierarchicalAggregateForecaster
from .flare_evaluation import (
    evaluate_joint_probabilities,
    joint_loss_rows,
    reconcile_joint_by_date,
)
from .flare_modeling import (
    FLARE24_ALL_EXTRA_FEATURES,
    FLARE24_BASE_EXTRA_FEATURES,
    FlareCatBoostModel,
    attach_flare_feature_partitions,
    fit_flare_catboost,
    select_usable_extra_features,
    task_frame,
)
from .flare_reconciliation import hurdle_joint_probabilities
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance
from .schedule_context import attach_schedule_context

Task = Literal["delay", "cancellation"]
Candidate = Literal["baseline", "weather", "rotation_structural", "rotation_risk"]

CANDIDATES: tuple[Candidate, ...] = (
    "baseline",
    "weather",
    "rotation_structural",
    "rotation_risk",
)
TASKS: tuple[Task, ...] = ("delay", "cancellation")
CALIBRATION_METHODS: tuple[CalibrationMethod, ...] = (
    "identity",
    "intercept",
    "platt",
    "beta",
    "isotonic",
)
CALIBRATION_FOLDS = (
    ("2024-10-01", "2024-10-15", "2024-10-16", "2024-10-31"),
    ("2024-10-01", "2024-10-31", "2024-11-01", "2024-11-30"),
    ("2024-10-01", "2024-11-30", "2024-12-01", "2024-12-31"),
)
MODEL_PARAMETERS: dict[Task, dict[str, Any]] = {
    "delay": {
        "iterations": 1_200,
        "learning_rate": 0.020911799420764387,
        "depth": 8,
        "l2_leaf_reg": 17.721143984084343,
        "random_strength": 0.20984487415833059,
        "bagging_temperature": 0.11862865926623616,
        "border_count": 64,
    },
    "cancellation": {
        "iterations": 1_500,
        "learning_rate": 0.04266275558194561,
        "depth": 10,
        "l2_leaf_reg": 1.8439374710684058,
        "random_strength": 1.8211594669189084,
        "bagging_temperature": 1.1653117632683447,
        "border_count": 64,
    },
}

PREDICTION_ID_COLUMNS = (
    "sample_id",
    "FlightDate",
    "Year",
    "Month",
    "Origin",
    "Dest",
    "DepHour",
    "ArrHour",
    "Reporting_Airline",
    "Route",
    "ArrDel15",
    "Cancelled",
    "delay_label_observed",
    "joint_label_observed",
    "disruption_state",
)
AGGREGATE_COLUMNS = (
    "FlightDate",
    "Origin",
    "Dest",
    "DepHour",
    "ArrHour",
    "Reporting_Airline",
    "Route",
)
AGGREGATE_HISTORY_COLUMNS = (*AGGREGATE_COLUMNS, "joint_label_observed", "disruption_state")


def _frozen_implementation_files() -> tuple[Path, ...]:
    directory = Path(__file__).parent
    return tuple(
        directory / name
        for name in (
            "flare_study.py",
            "flare_modeling.py",
            "flare_aggregate.py",
            "flare_reconciliation.py",
            "flare_evaluation.py",
            "calibration.py",
            "bootstrap.py",
            "metrics.py",
            "census_modeling.py",
            "census_recent.py",
            "census_graph.py",
            "census_normalization.py",
            "schedule_context.py",
            "recent.py",
            "recent_modeling.py",
            "modeling.py",
            "contracts.py",
            "hashing.py",
            "provenance.py",
        )
    )


def _self_hashed_payload(path: Path) -> tuple[dict[str, Any], str]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    key = "manifest_sha256" if "manifest_sha256" in payload else "report_sha256"
    if key not in payload:
        raise ValueError(f"artifact has no self-hash: {path}")
    recorded = str(payload[key])
    body = {name: value for name, value in payload.items() if name != key}
    if canonical_json_sha256(body) != recorded:
        raise ValueError(f"artifact self-hash failed: {path}")
    return payload, key


def _verify_manifest_output_root(
    manifest: dict[str, Any],
    *,
    root: Path,
    years: set[int],
    role: str,
) -> int:
    """Bind command-line data roots to checksummed manifest outputs."""

    resolved_root = root.resolve()
    records = [
        record
        for record in manifest.get("outputs", [])
        if int(record.get("year", -1)) in years
    ]
    if not records:
        raise ValueError(f"{role} manifest has no outputs for years {sorted(years)}")
    for record in records:
        path = Path(str(record.get("path", "")))
        try:
            path.resolve().relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"{role} output escapes its supplied root: {path}") from error
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"{role} output checksum failed: {path}")
        if "bytes" in record and path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"{role} output byte count failed: {path}")
    return len(records)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite FLARE-24 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated FLARE-24 partial exists: {partial}")
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
        raise FileExistsError(f"refusing to overwrite FLARE-24 model: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated FLARE-24 model partial exists: {partial}")
    joblib.dump(value, partial)
    partial.replace(path)
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _census_path(census_dir: Path, year: int, month: int) -> Path:
    if year not in {2023, 2024, 2025}:
        raise ValueError(f"FLARE-24 orchestration refuses census year {year}")
    path = census_dir / f"year={year}" / f"month={month:02d}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing census partition: {path}")
    return path


def load_enriched_month(
    *,
    census_dir: Path,
    recent_dir: Path,
    flight_recent_dir: Path,
    graph_dir: Path,
    weather_feature_dir: Path,
    rotation_feature_dir: Path,
    year: int,
    month: int,
    limit: int | None,
    seed: int,
) -> pd.DataFrame:
    """Load one full schedule context, then sample and attach keyed covariates."""

    path = _census_path(census_dir, year, month)
    frame = pd.read_parquet(path, columns=list(CENSUS_LOAD_COLUMNS))
    if not frame["Year"].eq(year).all() or not frame["Month"].eq(month).all():
        raise ValueError(f"census partition period mismatch: {path}")
    contextual = attach_schedule_context(frame)
    if limit is not None and len(contextual) > limit:
        contextual = contextual.sample(
            n=limit,
            replace=False,
            random_state=seed + year * 100 + month,
        )
    contextual = contextual.reset_index(drop=True)
    for column in CENSUS_BASE_CATEGORICAL_FEATURES + CENSUS_FLIGHT_CATEGORICAL_FEATURES:
        contextual[column] = contextual[column].astype("string").fillna("__MISSING__")
    enriched = attach_census_recent_features(
        contextual,
        recent_dir=recent_dir,
        flight_recent_dir=flight_recent_dir,
    )
    enriched = attach_census_graph_features(enriched, graph_dir=graph_dir)
    enriched = attach_flare_feature_partitions(
        enriched,
        feature_dir=weather_feature_dir,
        feature_columns=FLARE24_BASE_EXTRA_FEATURES,
    )
    enriched = attach_flare_feature_partitions(
        enriched,
        feature_dir=rotation_feature_dir,
        feature_columns=tuple(
            feature
            for feature in FLARE24_ALL_EXTRA_FEATURES
            if feature not in FLARE24_BASE_EXTRA_FEATURES
        ),
    )
    return enriched


def _load_period_sample(
    *,
    months: tuple[int, ...],
    limit_per_month: int | None,
    loader_arguments: dict[str, Any],
) -> pd.DataFrame:
    return pd.concat(
        [
            load_enriched_month(
                year=2024,
                month=month,
                limit=limit_per_month,
                **loader_arguments,
            )
            for month in months
        ],
        ignore_index=True,
    )


def _binary_eligible(frame: pd.DataFrame, task: Task) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
    if task == "cancellation":
        mask = frame["Cancelled"].isin([0, 1]).to_numpy(dtype=np.bool_)
        labels = frame["Cancelled"].fillna(-1).to_numpy(dtype=np.int64)
    else:
        mask = (
            frame["Cancelled"].eq(0) & frame["delay_label_observed"].eq(1)
        ).to_numpy(dtype=np.bool_)
        labels = frame["ArrDel15"].fillna(-1).to_numpy(dtype=np.int64)
    if not np.isin(labels[mask], [0, 1]).all():
        raise ValueError(f"invalid {task} labels in eligible calibration rows")
    return mask, labels


def _binary_loss(labels: ArrayLike, probabilities: ArrayLike) -> NDArray[np.float64]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not np.isin(y, [0, 1]).all():
        raise ValueError("binary loss inputs must be aligned and valid")
    return np.asarray(-(y * np.log(p) + (1 - y) * np.log1p(-p)), dtype=np.float64)


@dataclass(slots=True)
class CalibrationSelection:
    task: Task
    method: CalibrationMethod
    crossfit_probabilities: NDArray[np.float64]
    crossfit_mask: NDArray[np.bool_]
    final_calibrator: BinaryCalibrator
    candidate_records: tuple[dict[str, Any], ...]


def select_forward_calibration(
    frame: pd.DataFrame,
    raw_probabilities: ArrayLike,
    *,
    task: Task,
    methods: tuple[CalibrationMethod, ...] = CALIBRATION_METHODS,
) -> CalibrationSelection:
    """Select calibration using expanding forward folds within Q4 2024."""

    raw = np.asarray(raw_probabilities, dtype=np.float64)
    if raw.shape != (len(frame),) or not np.isfinite(raw).all():
        raise ValueError("raw calibration probabilities must align and be finite")
    dates = pd.to_datetime(frame["FlightDate"], errors="raise").dt.normalize()
    eligible, labels = _binary_eligible(frame, task)
    records: list[dict[str, Any]] = []
    crossfit_by_method: dict[CalibrationMethod, NDArray[np.float64]] = {}
    validation_union = np.zeros(len(frame), dtype=np.bool_)
    for method in methods:
        crossfit = np.full(len(frame), np.nan, dtype=np.float64)
        fold_records: list[dict[str, Any]] = []
        for fold_index, (train_start, train_end, valid_start, valid_end) in enumerate(
            CALIBRATION_FOLDS,
            start=1,
        ):
            train_date = dates.between(pd.Timestamp(train_start), pd.Timestamp(train_end)).to_numpy()
            valid_date = dates.between(pd.Timestamp(valid_start), pd.Timestamp(valid_end)).to_numpy()
            train = train_date & eligible
            valid = valid_date & eligible
            if not train.any() or not valid.any() or np.unique(labels[train]).size != 2:
                raise ValueError(f"calibration fold {fold_index} for {task} lacks support")
            calibrator = fit_binary_calibrator(method, raw[train], labels[train])
            crossfit[valid_date] = calibrator.predict(raw[valid_date])
            fold_loss = _binary_loss(labels[valid], crossfit[valid])
            fold_records.append(
                {
                    "fold": fold_index,
                    "train_dates": [train_start, train_end],
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
    final = fit_binary_calibrator(selected_method, raw[eligible], labels[eligible])
    return CalibrationSelection(
        task=task,
        method=selected_method,
        crossfit_probabilities=crossfit_by_method[selected_method],
        crossfit_mask=validation_union,
        final_calibrator=final,
        candidate_records=tuple(records),
    )


def select_simplex_ensemble(
    labels: ArrayLike,
    probabilities: dict[str, ArrayLike],
    *,
    step: float = 0.05,
) -> dict[str, Any]:
    """Select a convex blend over all prespecified candidates using joint log loss."""

    if tuple(probabilities) != CANDIDATES:
        raise ValueError(f"ensemble candidates must be ordered as {CANDIDATES}")
    y = np.asarray(labels, dtype=np.int64)
    matrices = {
        name: np.asarray(values, dtype=np.float64) for name, values in probabilities.items()
    }
    if any(matrix.shape != (len(y), 3) for matrix in matrices.values()):
        raise ValueError("ensemble joint probability matrices must align")
    divisions = round(1.0 / step)
    if divisions < 1 or not np.isclose(divisions * step, 1.0):
        raise ValueError("ensemble step must divide one exactly")
    records: list[dict[str, Any]] = []
    for prefix in product(range(divisions + 1), repeat=len(CANDIDATES) - 1):
        allocated = sum(prefix)
        if allocated > divisions:
            continue
        units = (*prefix, divisions - allocated)
        weights = {
            name: unit / divisions
            for name, unit in zip(CANDIDATES, units, strict=True)
        }
        blended = sum(weights[name] * matrices[name] for name in CANDIDATES)
        log_rows, brier_rows = joint_loss_rows(y, blended)
        records.append(
            {
                "weights": weights,
                "joint_log_loss": float(log_rows.mean()),
                "multiclass_brier": float(brier_rows.mean()),
            }
        )
    selected = min(
        records,
        key=lambda record: (
            float(record["joint_log_loss"]),
            *(
                -float(cast(dict[str, float], record["weights"])[name])
                for name in reversed(CANDIDATES[1:])
            ),
        ),
    )
    return {
        "selection_metric": "joint_log_loss",
        "grid_step": step,
        "selected": selected,
        "grid": records,
    }


def _blend_joint(
    methods: dict[str, NDArray[np.float64]],
    weights: dict[str, float],
) -> NDArray[np.float64]:
    blended = sum(float(weights[name]) * methods[name] for name in CANDIDATES)
    result = np.asarray(blended, dtype=np.float64)
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-9):
        raise RuntimeError("FLARE-24 ensemble left the probability simplex")
    return result


def weather_severity_index(frame: pd.DataFrame) -> NDArray[np.float64]:
    """Outcome-blind composite used only for prespecified diagnostic strata."""

    columns = (
        "flare24_route_endpoint_convective_max",
        "flare24_route_endpoint_icing_max",
        "flare24_route_endpoint_crosswind_max",
        "flare24_route_endpoint_visibility_hazard_max",
    )
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"weather severity input is missing columns: {missing}")
    convective = np.log1p(
        np.maximum(
            pd.to_numeric(frame[columns[0]], errors="coerce").to_numpy(dtype=np.float64),
            0.0,
        )
    ) / 5.0
    icing = pd.to_numeric(frame[columns[1]], errors="coerce").to_numpy(dtype=np.float64)
    crosswind = (
        pd.to_numeric(frame[columns[2]], errors="coerce").to_numpy(dtype=np.float64)
        / 30.0
    )
    visibility = (
        pd.to_numeric(frame[columns[3]], errors="coerce").to_numpy(dtype=np.float64)
        / 3.0
    )
    components = np.column_stack((convective, icing, crosswind, visibility))
    valid = np.isfinite(components)
    severity = np.max(np.where(valid, components, -np.inf), axis=1)
    severity[~valid.any(axis=1)] = np.nan
    return np.asarray(severity, dtype=np.float64)


def _aggregate_history(
    census_dir: Path,
    *,
    year: int,
    months: tuple[int, ...],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for month in months:
        path = _census_path(census_dir, year, month)
        frame = pd.read_parquet(path, columns=list(AGGREGATE_HISTORY_COLUMNS))
        frames.append(frame.loc[frame["joint_label_observed"].eq(1)].reset_index(drop=True))
        records.append(
            {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "columns_read": list(AGGREGATE_HISTORY_COLUMNS),
                "maximum_outcome_date": pd.to_datetime(frame["FlightDate"]).max().date().isoformat(),
            }
        )
    history = pd.concat(frames, ignore_index=True).drop(columns="joint_label_observed")
    return history, records


def _reconciliation_scale_selection(
    flights: pd.DataFrame,
    base_probabilities: NDArray[np.float64],
    aggregate_forecasts: pd.DataFrame,
    *,
    candidates: tuple[float, ...],
) -> dict[str, Any]:
    observed = flights["joint_label_observed"].eq(1).to_numpy()
    labels = flights.loc[observed, "disruption_state"].to_numpy(dtype=np.int64)
    base_log_rows, base_brier_rows = joint_loss_rows(labels, base_probabilities[observed])
    records: list[dict[str, Any]] = [
        {
            "mode": "identity_no_alignment",
            "variance_multiplier": None,
            "joint_log_loss": float(base_log_rows.mean()),
            "multiclass_brier": float(base_brier_rows.mean()),
            "mean_absolute_probability_shift": 0.0,
        }
    ]
    for multiplier in candidates:
        reconciled = reconcile_joint_by_date(
            flights,
            base_probabilities,
            aggregate_forecasts,
            variance_multiplier=multiplier,
        )
        log_rows, brier_rows = joint_loss_rows(labels, reconciled.probabilities[observed])
        records.append(
            {
                "mode": "soft_marginal_alignment",
                "variance_multiplier": multiplier,
                "joint_log_loss": float(log_rows.mean()),
                "multiclass_brier": float(brier_rows.mean()),
                "mean_absolute_probability_shift": float(
                    np.mean(np.abs(reconciled.probabilities - base_probabilities))
                ),
            }
        )
    selected = min(
        records,
        key=lambda record: (
            float(record["joint_log_loss"]),
            0 if record["mode"] == "identity_no_alignment" else 1,
            float(record["variance_multiplier"] or 0.0),
        ),
    )
    return {
        "selection_metric": "joint_log_loss",
        "identity_candidate_included": True,
        "selected_mode": selected["mode"],
        "selected_variance_multiplier": selected["variance_multiplier"],
        "candidates": records,
    }


def run_flare24_selection(
    *,
    protocol_path: Path,
    census_dir: Path,
    census_manifest: Path,
    recent_dir: Path,
    recent_manifest: Path,
    flight_recent_dir: Path,
    flight_recent_manifest: Path,
    graph_dir: Path,
    graph_manifest: Path,
    weather_feature_dir: Path,
    weather_feature_manifest: Path,
    rotation_feature_dir: Path,
    rotation_feature_manifest: Path,
    run_dir: Path,
    output_path: Path,
    rows_per_train_month: int = 125_000,
    validation_limit: int = 250_000,
    bootstrap_repetitions: int = 2_000,
    seed: int = 20260903,
) -> dict[str, Any]:
    """Fit and freeze the complete method using 2024 outcomes only."""

    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse FLARE-24 selection run: {run_dir}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite FLARE-24 selection report: {output_path}")
    if rows_per_train_month < 10_000 or validation_limit < 10_000:
        raise ValueError("FLARE-24 selection samples are too small for the research protocol")
    if bootstrap_repetitions < 100:
        raise ValueError("FLARE-24 selection requires at least 100 bootstrap repetitions")
    started = time.perf_counter()
    protocol: dict[str, Any] = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_models = protocol.get("models", {})
    protocol_evaluation = protocol.get("evaluation", {})
    if (
        protocol.get("identity", {}).get("method") != "FLARE-24"
        or protocol.get("information_boundary", {}).get("confirmation_gate_opened") is not False
        or protocol.get("reconciliation", {}).get("include_identity_no_alignment_control")
        is not True
        or tuple(protocol.get("calibration", {}).get("families", ()))
        != CALIBRATION_METHODS
        or float(protocol.get("ensemble", {}).get("grid_step", np.nan)) != 0.05
        or tuple(protocol.get("reconciliation", {}).get("variance_multiplier_candidates", ()))
        != (0.5, 1.0, 2.0, 4.0, 8.0)
        or int(protocol_models.get("rows_per_training_month", -1))
        != rows_per_train_month
        or int(protocol_models.get("early_stopping_max_rows", -1)) != validation_limit
        or protocol_models.get("refit_at_selected_iteration") is not True
        or protocol_models.get("fixed_iteration_refit_dates")
        != ["2024-01-01", "2024-09-30"]
        or protocol_models.get("delay_parameters") != MODEL_PARAMETERS["delay"]
        or protocol_models.get("cancellation_parameters")
        != MODEL_PARAMETERS["cancellation"]
        or int(protocol_evaluation.get("bootstrap_repetitions", -1))
        != bootstrap_repetitions
        or int(protocol_evaluation.get("bootstrap_seed", -1)) != seed
    ):
        raise ValueError("FLARE-24 executable settings differ from the frozen protocol")
    input_artifacts: list[dict[str, Any]] = [
        {
            "role": "frozen FLARE-24 protocol",
            "path": protocol_path.as_posix(),
            "sha256": sha256_file(protocol_path),
            "bytes": protocol_path.stat().st_size,
            "status": protocol.get("identity", {}).get("status"),
        }
    ]
    for role, path, root in (
        ("census", census_manifest, census_dir),
        ("closed-left HMOP", recent_manifest, recent_dir),
        ("closed-left flight history", flight_recent_manifest, flight_recent_dir),
        ("CL-SGMP", graph_manifest, graph_dir),
        ("FLARE cutoff-coherent weather", weather_feature_manifest, weather_feature_dir),
        ("FLARE latent rotations", rotation_feature_manifest, rotation_feature_dir),
    ):
        payload, self_hash_key = _self_hashed_payload(path)
        verified_outputs = _verify_manifest_output_root(
            payload,
            root=root,
            years={2024},
            role=role,
        )
        input_artifacts.append(
            {
                "role": role,
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "self_hash_key": self_hash_key,
                "self_hash": payload[self_hash_key],
                "status": payload.get("status"),
                "output_root": root.as_posix(),
                "verified_output_years": [2024],
                "verified_output_files": verified_outputs,
            }
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
    print("loading FLARE-24 January-August 2024 training sample", flush=True)
    train = _load_period_sample(
        months=tuple(range(1, 9)),
        limit_per_month=rows_per_train_month,
        loader_arguments=loader_arguments,
    )
    print("loading FLARE-24 September 2024 early-stopping sample", flush=True)
    validation = _load_period_sample(
        months=(9,),
        limit_per_month=validation_limit,
        loader_arguments=loader_arguments,
    )
    if pd.to_datetime(train["FlightDate"]).max() >= pd.to_datetime(validation["FlightDate"]).min():
        raise AssertionError("FLARE-24 training and early-stopping periods overlap")
    weather_features = select_usable_extra_features(
        train,
        FLARE24_BASE_EXTRA_FEATURES,
        minimum_nonmissing_fraction=0.01,
    )
    rotation_structural_features = select_usable_extra_features(
        train,
        tuple(
            feature
            for feature in FLARE24_ALL_EXTRA_FEATURES
            if feature != "flare24_rotation_inbound_disruption_risk"
        ),
        minimum_nonmissing_fraction=0.01,
    )
    rotation_risk_features = select_usable_extra_features(
        train,
        FLARE24_ALL_EXTRA_FEATURES,
        minimum_nonmissing_fraction=0.01,
    )
    feature_sets: dict[Candidate, tuple[str, ...]] = {
        "baseline": (),
        "weather": weather_features,
        "rotation_structural": rotation_structural_features,
        "rotation_risk": rotation_risk_features,
    }
    models: dict[Candidate, dict[Task, FlareCatBoostModel]] = {
        candidate: {} for candidate in CANDIDATES
    }
    model_records: list[dict[str, Any]] = []
    for task in TASKS:
        train_task, train_labels = task_frame(train, task)
        valid_task, valid_labels = task_frame(validation, task)
        for candidate in CANDIDATES:
            fit_started = time.perf_counter()
            print(f"early-stopping FLARE-24 {candidate} {task}", flush=True)
            pilot_model = fit_flare_catboost(
                train_task,
                train_labels,
                task=task,
                extra_features=feature_sets[candidate],
                params=MODEL_PARAMETERS[task],
                validation_frame=valid_task,
                validation_labels=valid_labels,
                include_cross_direction=True,
                include_rich_schedule=True,
                include_flight_history=True,
                include_schedule_context=True,
                include_graph_pressure=True,
            )
            early_stopping_best_iteration = int(
                pilot_model.estimator.get_best_iteration()
            )
            refit_iterations = (
                int(MODEL_PARAMETERS[task]["iterations"])
                if early_stopping_best_iteration < 0
                else early_stopping_best_iteration + 1
            )
            refit_frame = pd.concat([train_task, valid_task], ignore_index=True)
            refit_labels = np.concatenate([train_labels, valid_labels])
            refit_parameters = {**MODEL_PARAMETERS[task], "iterations": refit_iterations}
            print(
                f"refitting FLARE-24 {candidate} {task} through September "
                f"for {refit_iterations} iterations",
                flush=True,
            )
            model = fit_flare_catboost(
                refit_frame,
                refit_labels,
                task=task,
                extra_features=feature_sets[candidate],
                params=refit_parameters,
                include_cross_direction=True,
                include_rich_schedule=True,
                include_flight_history=True,
                include_schedule_context=True,
                include_graph_pressure=True,
            )
            models[candidate][task] = model
            artifact = _atomic_joblib(
                model,
                run_dir / "models" / f"{candidate}_{task}.joblib",
            )
            model_records.append(
                {
                    "candidate": candidate,
                    "task": task,
                    "early_stopping_training_rows": len(train_task),
                    "early_stopping_validation_rows": len(valid_task),
                    "early_stopping_positive_training_rows": int(train_labels.sum()),
                    "early_stopping_positive_validation_rows": int(valid_labels.sum()),
                    "early_stopping_best_iteration": early_stopping_best_iteration,
                    "refit_training_rows": len(refit_frame),
                    "refit_positive_rows": int(refit_labels.sum()),
                    "refit_through_date": "2024-09-30",
                    "refit_iterations": refit_iterations,
                    "extra_feature_count": len(feature_sets[candidate]),
                    "fit_seconds": time.perf_counter() - fit_started,
                    "early_stopping_parameters": MODEL_PARAMETERS[task],
                    "refit_parameters": refit_parameters,
                    **artifact,
                }
            )
            del pilot_model, refit_frame, refit_labels
            gc.collect()
        del train_task, train_labels, valid_task, valid_labels
        gc.collect()
    training_rows = len(train)
    validation_rows = len(validation)
    del train, validation
    gc.collect()

    raw_prediction_records: list[dict[str, Any]] = []
    selection_parts: list[pd.DataFrame] = []
    for month in (10, 11, 12):
        print(f"predicting FLARE-24 2024-{month:02d} selection partition", flush=True)
        frame = load_enriched_month(
            year=2024,
            month=month,
            limit=None,
            **loader_arguments,
        )
        prediction = frame.loc[:, list(PREDICTION_ID_COLUMNS)].copy()
        prediction["weather_severity_index"] = weather_severity_index(frame).astype(
            "float32"
        )
        prediction["weather_covariate_available"] = (
            frame["flare24_origin_variable_coverage"].gt(0.0)
            & frame["flare24_dest_variable_coverage"].gt(0.0)
        ).astype("int8")
        for candidate in CANDIDATES:
            prediction[f"raw_{candidate}_cancellation"] = models[candidate][
                "cancellation"
            ].predict_proba(frame).astype("float32")
            prediction[f"raw_{candidate}_delay"] = models[candidate][
                "delay"
            ].predict_proba(frame).astype("float32")
        artifact = _atomic_parquet(
            prediction,
            run_dir / "predictions" / f"raw_selection_2024_{month:02d}.parquet",
        )
        raw_prediction_records.append({"year": 2024, "month": month, **artifact})
        selection_parts.append(prediction)
        del frame, prediction
        gc.collect()
    selection = pd.concat(selection_parts, ignore_index=True)
    del selection_parts

    calibrators: dict[Candidate, dict[Task, BinaryCalibrator]] = {
        candidate: {} for candidate in CANDIDATES
    }
    calibration_records: list[dict[str, Any]] = []
    common_crossfit = np.ones(len(selection), dtype=np.bool_)
    for candidate in CANDIDATES:
        for task in TASKS:
            selected = select_forward_calibration(
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
                    "candidate_methods": list(selected.candidate_records),
                    **artifact,
                }
            )
    if not common_crossfit.any():
        raise RuntimeError("FLARE-24 calibration produced no common forward predictions")
    selection_crossfit = selection.loc[common_crossfit].reset_index(drop=True)
    severity_values = selection_crossfit["weather_severity_index"].dropna().to_numpy(
        dtype=np.float64
    )
    if severity_values.size == 0:
        raise RuntimeError("FLARE-24 selection has no weather severity values")
    severity_cutpoints = np.quantile(severity_values, [0.25, 0.5, 0.75]).tolist()
    candidate_joint: dict[str, NDArray[np.float64]] = {}
    for candidate in CANDIDATES:
        candidate_joint[candidate] = hurdle_joint_probabilities(
            selection_crossfit[f"xfit_{candidate}_cancellation"],
            selection_crossfit[f"xfit_{candidate}_delay"],
        )
    joint_observed = selection_crossfit["joint_label_observed"].eq(1).to_numpy()
    joint_labels = selection_crossfit.loc[
        joint_observed, "disruption_state"
    ].to_numpy(dtype=np.int64)
    ensemble_selection = select_simplex_ensemble(
        joint_labels,
        {
            candidate: candidate_joint[candidate][joint_observed]
            for candidate in CANDIDATES
        },
    )
    ensemble_weights = cast(
        dict[str, float],
        cast(dict[str, Any], ensemble_selection["selected"])["weights"],
    )
    ensemble_joint = _blend_joint(candidate_joint, ensemble_weights)

    print("fitting independent aggregate forecaster through September 2024", flush=True)
    aggregate_history, aggregate_history_records = _aggregate_history(
        census_dir,
        year=2024,
        months=tuple(range(1, 10)),
    )
    aggregate_model = HierarchicalAggregateForecaster().fit(aggregate_history)
    del aggregate_history
    gc.collect()
    aggregate_model_artifact = _atomic_joblib(
        aggregate_model,
        run_dir / "aggregate" / "through_2024_09.joblib",
    )
    aggregate_forecasts = aggregate_model.predict(
        selection_crossfit.loc[:, list(AGGREGATE_COLUMNS)]
    )
    aggregate_forecast_artifact = _atomic_parquet(
        aggregate_forecasts,
        run_dir / "aggregate" / "selection_2024_q4_forecasts.parquet",
    )
    print("selecting uncertainty scale for probabilistic reconciliation", flush=True)
    reconciliation_selection = _reconciliation_scale_selection(
        selection_crossfit,
        ensemble_joint,
        aggregate_forecasts,
        candidates=(0.5, 1.0, 2.0, 4.0, 8.0),
    )
    selected_multiplier_value = reconciliation_selection["selected_variance_multiplier"]
    selected_variance_multiplier = (
        None if selected_multiplier_value is None else float(selected_multiplier_value)
    )
    if selected_variance_multiplier is None:
        reconciled_probabilities = ensemble_joint.copy()
        reconciliation_diagnostics: list[dict[str, Any]] = []
    else:
        reconciled_result = reconcile_joint_by_date(
            selection_crossfit,
            ensemble_joint,
            aggregate_forecasts,
            variance_multiplier=selected_variance_multiplier,
        )
        reconciled_probabilities = reconciled_result.probabilities
        reconciliation_diagnostics = list(reconciled_result.date_diagnostics)
    evaluation_methods: dict[str, ArrayLike] = {
        **candidate_joint,
        "ensemble": ensemble_joint,
        "reconciled": reconciled_probabilities,
    }
    evaluation = evaluate_joint_probabilities(
        selection_crossfit,
        evaluation_methods,
        reference_method="baseline",
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=seed,
    )
    crossfit_artifact_frame = selection_crossfit.loc[
        :,
        [
            *PREDICTION_ID_COLUMNS,
            "weather_severity_index",
            "weather_covariate_available",
        ],
    ].copy()
    for candidate in CANDIDATES:
        crossfit_artifact_frame[f"prob_{candidate}_on_time"] = candidate_joint[
            candidate
        ][:, 0].astype("float32")
        crossfit_artifact_frame[f"prob_{candidate}_delayed"] = candidate_joint[
            candidate
        ][:, 1].astype("float32")
        crossfit_artifact_frame[f"prob_{candidate}_cancelled"] = candidate_joint[
            candidate
        ][:, 2].astype("float32")
    for name, values in (
        ("ensemble", ensemble_joint),
        ("reconciled", reconciled_probabilities),
    ):
        crossfit_artifact_frame[f"prob_{name}_on_time"] = values[:, 0].astype("float32")
        crossfit_artifact_frame[f"prob_{name}_delayed"] = values[:, 1].astype("float32")
        crossfit_artifact_frame[f"prob_{name}_cancelled"] = values[:, 2].astype("float32")
    crossfit_artifact = _atomic_parquet(
        crossfit_artifact_frame,
        run_dir / "predictions" / "selection_2024_crossfit_joint.parquet",
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_2024_FLARE24_SELECTION_NOT_CONFIRMATORY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "FLARE-24",
        "working_name": "flight-level aggregate-reconciled ensemble at 24 hours",
        "input_artifacts": input_artifacts,
        "periods": {
            "early_stopping_training": ["2024-01-01", "2024-08-31"],
            "early_stopping_validation": ["2024-09-01", "2024-09-30"],
            "fixed_iteration_refit": ["2024-01-01", "2024-09-30"],
            "forward_calibration_and_selection": ["2024-10-01", "2024-12-31"],
            "crossfit_scoring": ["2024-10-16", "2024-12-31"],
        },
        "loaded_training_rows": training_rows,
        "loaded_validation_rows": validation_rows,
        "rows_per_train_month": rows_per_train_month,
        "validation_limit": validation_limit,
        "feature_sets": {
            candidate: list(features) for candidate, features in feature_sets.items()
        },
        "model_artifacts": model_records,
        "raw_prediction_artifacts": raw_prediction_records,
        "calibration": {
            "folds": [list(fold) for fold in CALIBRATION_FOLDS],
            "artifacts": calibration_records,
        },
        "ensemble_selection": ensemble_selection,
        "aggregate_model": {
            "model_card": aggregate_model.model_card().as_dict(),
            "history_inputs": aggregate_history_records,
            **aggregate_model_artifact,
        },
        "aggregate_forecasts": aggregate_forecast_artifact,
        "reconciliation_selection": reconciliation_selection,
        "selected_variance_multiplier": selected_variance_multiplier,
        "reconciliation_diagnostics": reconciliation_diagnostics,
        "diagnostic_weather_severity": {
            "formula": "max(log1p(endpoint convection)/5, endpoint icing, endpoint gust-crosswind knots/30, endpoint visibility category/3)",
            "selection_quartile_cutpoints": severity_cutpoints,
            "role": "outcome-blind descriptive stratification only",
        },
        "crossfit_prediction_artifact": crossfit_artifact,
        "selection_evaluation": evaluation,
        "bootstrap_repetitions": bootstrap_repetitions,
        "seed": seed,
        "outcomes_accessed": {
            "years": [2024],
            "maximum_calendar_date": "2024-12-31",
            "2025_accessed_by_this_run": False,
            "2026_accessed": False,
        },
        "confirmation_gate": {
            "year": 2026,
            "opened": False,
            "rule": "No 2026 outcome may be acquired, loaded, tuned, or scored.",
        },
        "versions": {
            "python": platform.python_version(),
            "catboost": version("catboost"),
            "joblib": version("joblib"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
        },
        "provenance": capture_provenance(_frozen_implementation_files()),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "All scores in this report are method-selection evidence from 2024, not an "
            "independent audit or confirmation. BTS schedules are retrospective proxies for "
            "D-24 schedule snapshots."
        ),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(output_path, report)
    write_canonical_json(run_dir / "run_manifest.json", report)
    return report


def _verified_joblib(record: dict[str, Any]) -> Any:
    path = Path(str(record["path"]))
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError(f"FLARE-24 serialized artifact checksum failed: {path}")
    return joblib.load(path)


def write_flare24_method_lock(
    selection_report_path: Path,
    *,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze every selected FLARE-24 choice before the 2025 audit is opened."""

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite FLARE-24 method lock: {output_path}")
    selection, self_hash_key = _self_hashed_payload(selection_report_path)
    if selection.get("status") != "COMPLETE_2024_FLARE24_SELECTION_NOT_CONFIRMATORY":
        raise ValueError("FLARE-24 lock requires a completed 2024 selection report")
    boundary = selection.get("outcomes_accessed", {})
    if boundary.get("maximum_calendar_date") != "2024-12-31" or boundary.get(
        "2025_accessed_by_this_run"
    ):
        raise ValueError("FLARE-24 selection outcome boundary is invalid")
    feature_sets = selection.get("feature_sets", {})
    if set(feature_sets) != set(CANDIDATES):
        raise ValueError("FLARE-24 selection feature sets are incomplete")
    expected_pairs = {(candidate, task) for candidate in CANDIDATES for task in TASKS}
    model_artifacts = list(selection.get("model_artifacts", []))
    calibrator_artifacts = list(selection.get("calibration", {}).get("artifacts", []))
    for role, records in (
        ("model", model_artifacts),
        ("calibrator", calibrator_artifacts),
    ):
        pairs = {(str(record.get("candidate")), str(record.get("task"))) for record in records}
        if pairs != expected_pairs or len(records) != len(expected_pairs):
            raise ValueError(f"FLARE-24 selection {role} artifacts are incomplete")
        for record in records:
            path = Path(str(record.get("path", "")))
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                raise ValueError(f"FLARE-24 selection {role} artifact changed: {path}")
    weights = selection.get("ensemble_selection", {}).get("selected", {}).get("weights", {})
    if set(weights) != set(CANDIDATES) or not np.isclose(
        sum(float(value) for value in weights.values()), 1.0
    ):
        raise ValueError("FLARE-24 selection ensemble weights are invalid")
    reconciliation = selection.get("reconciliation_selection", {})
    if reconciliation.get("identity_candidate_included") is not True:
        raise ValueError("FLARE-24 selection omitted the no-alignment control")
    selected_multiplier = selection.get("selected_variance_multiplier")
    if (reconciliation.get("selected_mode") == "identity_no_alignment") != (
        selected_multiplier is None
    ):
        raise ValueError("FLARE-24 reconciliation choice is internally inconsistent")
    lock: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED_FLARE24_METHOD_BEFORE_2025_RETROSPECTIVE_AUDIT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_report": {
            "path": selection_report_path.as_posix(),
            "sha256": sha256_file(selection_report_path),
            "self_hash_key": self_hash_key,
            "self_hash": selection[self_hash_key],
        },
        "frozen_choices": {
            "feature_sets": feature_sets,
            "model_artifacts": model_artifacts,
            "calibrator_artifacts": calibrator_artifacts,
            "ensemble_weights": weights,
            "reconciliation_enabled": selected_multiplier is not None,
            "reconciliation_variance_multiplier": selection[
                "selected_variance_multiplier"
            ],
            "weather_severity_cutpoints": selection[
                "diagnostic_weather_severity"
            ]["selection_quartile_cutpoints"],
        },
        "outcomes_accessed_by_lock": {
            "years": [2024],
            "maximum_calendar_date": "2024-12-31",
            "2025": False,
            "2026": False,
        },
        "confirmation_gate": {
            "year": 2026,
            "opened": False,
        },
        "provenance": capture_provenance((Path(__file__),)),
    }
    lock["manifest_sha256"] = canonical_json_sha256(lock)
    write_canonical_json(output_path, lock)
    return lock


def _monthly_joint_scores(
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


def _stratified_joint_scores(
    frame: pd.DataFrame,
    methods: dict[str, NDArray[np.float64]],
    groups: pd.Series,
) -> list[dict[str, Any]]:
    if len(groups) != len(frame):
        raise ValueError("diagnostic groups must align with the evaluation frame")
    records: list[dict[str, Any]] = []
    observed = frame["joint_label_observed"].eq(1).to_numpy()
    labels = frame["disruption_state"].to_numpy(dtype=np.int64)
    for group in sorted(groups.astype(str).unique()):
        mask = groups.astype(str).eq(group).to_numpy() & observed
        if not mask.any():
            continue
        metrics: dict[str, Any] = {}
        for name, probabilities in methods.items():
            log_rows, brier_rows = joint_loss_rows(labels[mask], probabilities[mask])
            metrics[name] = {
                "joint_log_loss": float(log_rows.mean()),
                "multiclass_brier": float(brier_rows.mean()),
            }
        records.append(
            {
                "group": group,
                "n": int(mask.sum()),
                "metrics": metrics,
            }
        )
    return records


def run_flare24_audit(
    method_lock_path: Path,
    *,
    census_dir: Path,
    recent_dir: Path,
    flight_recent_dir: Path,
    graph_dir: Path,
    weather_feature_dir: Path,
    rotation_feature_dir: Path,
    run_dir: Path,
    output_path: Path,
    bootstrap_repetitions: int = 2_000,
    seed: int = 20260903,
) -> dict[str, Any]:
    """Apply the frozen 2024 method once to the pre-existing 2025 audit year."""

    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse FLARE-24 audit run: {run_dir}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite FLARE-24 audit report: {output_path}")
    if bootstrap_repetitions < 100:
        raise ValueError("FLARE-24 audit requires at least 100 bootstrap repetitions")
    method_lock, lock_hash_key = _self_hashed_payload(method_lock_path)
    if method_lock.get("status") != (
        "LOCKED_FLARE24_METHOD_BEFORE_2025_RETROSPECTIVE_AUDIT"
    ):
        raise ValueError("FLARE-24 audit requires a completed pre-audit method lock")
    lock_boundary = method_lock.get("outcomes_accessed_by_lock", {})
    if lock_boundary.get("2025") is not False or lock_boundary.get("2026") is not False:
        raise ValueError("FLARE-24 method lock outcome boundary is invalid")
    selection_record = method_lock.get("selection_report", {})
    selection_report_path = Path(str(selection_record.get("path", "")))
    if (
        not selection_report_path.is_file()
        or sha256_file(selection_report_path) != selection_record.get("sha256")
    ):
        raise ValueError("FLARE-24 locked selection report changed")
    selection, self_hash_key = _self_hashed_payload(selection_report_path)
    if selection[self_hash_key] != selection_record.get("self_hash"):
        raise ValueError("FLARE-24 locked selection self-hash changed")
    if selection.get("status") != "COMPLETE_2024_FLARE24_SELECTION_NOT_CONFIRMATORY":
        raise ValueError("FLARE-24 audit requires a completed 2024 selection report")
    boundary = selection.get("outcomes_accessed", {})
    if boundary.get("maximum_calendar_date") != "2024-12-31" or boundary.get(
        "2025_accessed_by_this_run"
    ):
        raise ValueError("FLARE-24 selection outcome boundary is invalid")
    expected_frozen_choices = {
        "feature_sets": selection["feature_sets"],
        "model_artifacts": selection["model_artifacts"],
        "calibrator_artifacts": selection["calibration"]["artifacts"],
        "ensemble_weights": selection["ensemble_selection"]["selected"]["weights"],
        "reconciliation_enabled": selection["selected_variance_multiplier"] is not None,
        "reconciliation_variance_multiplier": selection["selected_variance_multiplier"],
        "weather_severity_cutpoints": selection["diagnostic_weather_severity"][
            "selection_quartile_cutpoints"
        ],
    }
    if method_lock.get("frozen_choices") != expected_frozen_choices:
        raise ValueError("FLARE-24 method lock does not exactly freeze the selected method")
    audit_roots: dict[str, tuple[Path, set[int]]] = {
        "census": (census_dir, {2024, 2025}),
        "closed-left HMOP": (recent_dir, {2025}),
        "closed-left flight history": (flight_recent_dir, {2025}),
        "CL-SGMP": (graph_dir, {2025}),
        "FLARE cutoff-coherent weather": (weather_feature_dir, {2025}),
        "FLARE latent rotations": (rotation_feature_dir, {2025}),
    }
    observed_roles: set[str] = set()
    for record in selection.get("input_artifacts", []):
        path = Path(str(record["path"]))
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"FLARE-24 selection input changed: {path}")
        role = str(record.get("role", ""))
        if role in audit_roots:
            manifest, _ = _self_hashed_payload(path)
            root, years = audit_roots[role]
            _verify_manifest_output_root(
                manifest,
                root=root,
                years=years,
                role=f"audit {role}",
            )
            observed_roles.add(role)
    if observed_roles != set(audit_roots):
        raise ValueError("FLARE-24 audit cannot bind every required input root")
    for record in selection.get("provenance", {}).get("source_files", []):
        path = Path(str(record["path"]))
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"FLARE-24 frozen implementation changed before audit: {path}")
    started = time.perf_counter()
    run_dir.mkdir(parents=True)
    models: dict[Candidate, dict[Task, FlareCatBoostModel]] = {
        candidate: {} for candidate in CANDIDATES
    }
    for record in selection["model_artifacts"]:
        candidate = cast(Candidate, str(record["candidate"]))
        task = cast(Task, str(record["task"]))
        models[candidate][task] = cast(FlareCatBoostModel, _verified_joblib(record))
    calibrators: dict[Candidate, dict[Task, BinaryCalibrator]] = {
        candidate: {} for candidate in CANDIDATES
    }
    for record in selection["calibration"]["artifacts"]:
        candidate = cast(Candidate, str(record["candidate"]))
        task = cast(Task, str(record["task"]))
        calibrators[candidate][task] = cast(BinaryCalibrator, _verified_joblib(record))
    if any(set(models[candidate]) != set(TASKS) for candidate in CANDIDATES):
        raise ValueError("FLARE-24 selection report has an incomplete model set")
    if any(set(calibrators[candidate]) != set(TASKS) for candidate in CANDIDATES):
        raise ValueError("FLARE-24 selection report has an incomplete calibrator set")
    ensemble_weights = {
        str(name): float(value)
        for name, value in selection["ensemble_selection"]["selected"]["weights"].items()
    }
    if set(ensemble_weights) != set(CANDIDATES) or not np.isclose(
        sum(ensemble_weights.values()), 1.0
    ):
        raise ValueError("FLARE-24 frozen ensemble weights are invalid")
    selected_multiplier_value = selection["selected_variance_multiplier"]
    selected_variance_multiplier = (
        None if selected_multiplier_value is None else float(selected_multiplier_value)
    )
    severity_cutpoints = np.asarray(
        selection["diagnostic_weather_severity"]["selection_quartile_cutpoints"],
        dtype=np.float64,
    )
    if severity_cutpoints.shape != (3,) or not np.isfinite(severity_cutpoints).all():
        raise ValueError("FLARE-24 weather diagnostic cutpoints are invalid")

    print("fitting independent aggregate forecaster on observed 2024 history", flush=True)
    aggregate_history, aggregate_history_records = _aggregate_history(
        census_dir,
        year=2024,
        months=tuple(range(1, 13)),
    )
    aggregate_model = HierarchicalAggregateForecaster().fit(aggregate_history)
    del aggregate_history
    gc.collect()
    aggregate_model_artifact = _atomic_joblib(
        aggregate_model,
        run_dir / "aggregate" / "history_2024_for_2025.joblib",
    )
    loader_arguments: dict[str, Any] = {
        "census_dir": census_dir,
        "recent_dir": recent_dir,
        "flight_recent_dir": flight_recent_dir,
        "graph_dir": graph_dir,
        "weather_feature_dir": weather_feature_dir,
        "rotation_feature_dir": rotation_feature_dir,
        "seed": seed,
    }
    prediction_records: list[dict[str, Any]] = []
    aggregate_records: list[dict[str, Any]] = []
    monthly_records: list[dict[str, Any]] = []
    reconciliation_diagnostics: list[dict[str, Any]] = []
    evaluation_frames: list[pd.DataFrame] = []
    probability_parts: dict[str, list[NDArray[np.float32]]] = {
        **{candidate: [] for candidate in CANDIDATES},
        "ensemble": [],
        "reconciled": [],
    }
    for month in range(1, 13):
        print(f"auditing frozen FLARE-24 on 2025-{month:02d}", flush=True)
        frame = load_enriched_month(
            year=2025,
            month=month,
            limit=None,
            **loader_arguments,
        )
        raw_binary: dict[Candidate, dict[Task, NDArray[np.float64]]] = {
            candidate: {} for candidate in CANDIDATES
        }
        calibrated_binary: dict[Candidate, dict[Task, NDArray[np.float64]]] = {
            candidate: {} for candidate in CANDIDATES
        }
        candidate_joint: dict[str, NDArray[np.float64]] = {}
        for candidate in CANDIDATES:
            for task in TASKS:
                raw = models[candidate][task].predict_proba(frame)
                raw_binary[candidate][task] = raw
                calibrated_binary[candidate][task] = calibrators[candidate][task].predict(
                    raw
                )
            candidate_joint[candidate] = hurdle_joint_probabilities(
                calibrated_binary[candidate]["cancellation"],
                calibrated_binary[candidate]["delay"],
            )
        ensemble_joint = _blend_joint(candidate_joint, ensemble_weights)
        aggregate_forecasts = aggregate_model.predict(
            frame.loc[:, list(AGGREGATE_COLUMNS)]
        )
        aggregate_artifact = _atomic_parquet(
            aggregate_forecasts,
            run_dir / "aggregate" / f"forecasts_2025_{month:02d}.parquet",
        )
        aggregate_records.append({"year": 2025, "month": month, **aggregate_artifact})
        if selected_variance_multiplier is None:
            reconciled_probabilities = ensemble_joint.copy()
            month_reconciliation_diagnostics: tuple[dict[str, Any], ...] = ()
        else:
            reconciled_result = reconcile_joint_by_date(
                frame,
                ensemble_joint,
                aggregate_forecasts,
                variance_multiplier=selected_variance_multiplier,
            )
            reconciled_probabilities = reconciled_result.probabilities
            month_reconciliation_diagnostics = reconciled_result.date_diagnostics
        methods = {
            **candidate_joint,
            "ensemble": ensemble_joint,
            "reconciled": reconciled_probabilities,
        }
        monthly_records.append(
            {
                "year": 2025,
                "month": month,
                "rows": len(frame),
                "proper_scores": _monthly_joint_scores(frame, methods),
            }
        )
        reconciliation_diagnostics.extend(month_reconciliation_diagnostics)
        prediction = frame.loc[:, list(PREDICTION_ID_COLUMNS)].copy()
        prediction["weather_severity_index"] = weather_severity_index(frame).astype(
            "float32"
        )
        prediction["weather_covariate_available"] = (
            frame["flare24_origin_variable_coverage"].gt(0.0)
            & frame["flare24_dest_variable_coverage"].gt(0.0)
        ).astype("int8")
        for candidate in CANDIDATES:
            for task in TASKS:
                prediction[f"raw_{candidate}_{task}"] = raw_binary[candidate][task].astype(
                    "float32"
                )
                prediction[f"calibrated_{candidate}_{task}"] = calibrated_binary[candidate][
                    task
                ].astype("float32")
        for name, values in methods.items():
            prediction[f"prob_{name}_on_time"] = values[:, 0].astype("float32")
            prediction[f"prob_{name}_delayed"] = values[:, 1].astype("float32")
            prediction[f"prob_{name}_cancelled"] = values[:, 2].astype("float32")
            probability_parts[name].append(values.astype("float32"))
        artifact = _atomic_parquet(
            prediction,
            run_dir / "predictions" / f"audit_2025_{month:02d}.parquet",
        )
        prediction_records.append({"year": 2025, "month": month, **artifact})
        evaluation_frames.append(
            prediction.loc[
                :,
                [
                    *PREDICTION_ID_COLUMNS,
                    "weather_severity_index",
                    "weather_covariate_available",
                ],
            ].copy()
        )
        del (
            frame,
            raw_binary,
            calibrated_binary,
            candidate_joint,
            ensemble_joint,
            aggregate_forecasts,
            reconciled_probabilities,
            methods,
            prediction,
        )
        gc.collect()

    evaluation_frame = pd.concat(evaluation_frames, ignore_index=True)
    del evaluation_frames
    evaluation_probabilities: dict[str, NDArray[np.float64]] = {
        name: np.concatenate(parts).astype(np.float64)
        for name, parts in probability_parts.items()
    }
    evaluation_inputs: dict[str, ArrayLike] = dict(evaluation_probabilities)
    evaluation = evaluate_joint_probabilities(
        evaluation_frame,
        evaluation_inputs,
        reference_method="baseline",
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=seed + 10_000,
    )
    severity = pd.Series("missing", index=evaluation_frame.index, dtype="string")
    available_severity = evaluation_frame["weather_severity_index"].notna()
    severity.loc[available_severity] = pd.cut(
        evaluation_frame.loc[available_severity, "weather_severity_index"],
        bins=[-np.inf, *severity_cutpoints.tolist(), np.inf],
        labels=["Q1_low", "Q2", "Q3", "Q4_high"],
        include_lowest=True,
    ).astype("string")
    weather_strata = _stratified_joint_scores(
        evaluation_frame,
        evaluation_probabilities,
        severity,
    )
    availability_strata = _stratified_joint_scores(
        evaluation_frame,
        evaluation_probabilities,
        evaluation_frame["weather_covariate_available"].map(
            {0: "weather_missing", 1: "weather_available"}
        ),
    )
    top_origins = set(evaluation_frame["Origin"].value_counts().head(20).index.astype(str))
    origin_groups = evaluation_frame["Origin"].astype(str).where(
        evaluation_frame["Origin"].astype(str).isin(top_origins),
        "OTHER",
    )
    origin_strata = _stratified_joint_scores(
        evaluation_frame,
        evaluation_probabilities,
        origin_groups,
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_2025_FLARE24_RETROSPECTIVE_AUDIT_NOT_BLIND_CONFIRMATION",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "FLARE-24",
        "selection_report": {
            "path": selection_report_path.as_posix(),
            "sha256": sha256_file(selection_report_path),
            "self_hash_key": self_hash_key,
            "self_hash": selection[self_hash_key],
        },
        "method_lock": {
            "path": method_lock_path.as_posix(),
            "sha256": sha256_file(method_lock_path),
            "self_hash_key": lock_hash_key,
            "self_hash": method_lock[lock_hash_key],
        },
        "frozen_method": {
            "model_refit": False,
            "calibrator_refit": False,
            "ensemble_weights": ensemble_weights,
            "reconciliation_enabled": selected_variance_multiplier is not None,
            "reconciliation_variance_multiplier": selected_variance_multiplier,
            "feature_sets": selection["feature_sets"],
            "model_artifacts": selection["model_artifacts"],
            "calibrator_artifacts": selection["calibration"]["artifacts"],
        },
        "aggregate_refit": {
            "permitted_role": "2024 outcomes are historical before every 2025 target date",
            "model_card": aggregate_model.model_card().as_dict(),
            "history_inputs": aggregate_history_records,
            **aggregate_model_artifact,
        },
        "aggregate_forecast_artifacts": aggregate_records,
        "prediction_artifacts": prediction_records,
        "evaluation_year": 2025,
        "evaluation_rows": len(evaluation_frame),
        "primary_evaluation": evaluation,
        "monthly_proper_scores": monthly_records,
        "diagnostics": {
            "weather_severity_selection_cutpoints": severity_cutpoints.tolist(),
            "weather_severity_strata": weather_strata,
            "weather_availability_strata": availability_strata,
            "top_20_origin_strata_plus_other": origin_strata,
            "interpretation": "post-freeze descriptive heterogeneity; not additional model selection",
        },
        "reconciliation_diagnostics": reconciliation_diagnostics,
        "bootstrap_repetitions": bootstrap_repetitions,
        "seed": seed,
        "outcomes_accessed": {
            "historical_aggregate_fit_year": 2024,
            "retrospective_audit_year": 2025,
            "maximum_calendar_date": "2025-12-31",
            "2026_accessed": False,
        },
        "confirmation_gate": {
            "year": 2026,
            "opened": False,
            "rule": "No 2026 outcome may be acquired, loaded, tuned, or scored.",
        },
        "versions": {
            "python": platform.python_version(),
            "catboost": version("catboost"),
            "joblib": version("joblib"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
        },
        "provenance": capture_provenance(_frozen_implementation_files()),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "This is a pre-existing-outcome retrospective audit, not a never-seen confirmation. "
            "It supports comparative predictive claims on the BTS top-100 cohort only; it does "
            "not establish causal, operational, real-time, fairness, or safety effects."
        ),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(output_path, report)
    write_canonical_json(run_dir / "run_manifest.json", report)
    return report


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--census-dir", type=Path, required=True)
    parser.add_argument("--recent-dir", type=Path, required=True)
    parser.add_argument("--flight-recent-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--weather-feature-dir", type=Path, required=True)
    parser.add_argument("--rotation-feature-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260903)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    selection = subparsers.add_parser("selection")
    _add_data_arguments(selection)
    selection.add_argument("--census-manifest", type=Path, required=True)
    selection.add_argument("--protocol", type=Path, required=True)
    selection.add_argument("--recent-manifest", type=Path, required=True)
    selection.add_argument("--flight-recent-manifest", type=Path, required=True)
    selection.add_argument("--graph-manifest", type=Path, required=True)
    selection.add_argument("--weather-feature-manifest", type=Path, required=True)
    selection.add_argument("--rotation-feature-manifest", type=Path, required=True)
    selection.add_argument("--rows-per-train-month", type=int, default=125_000)
    selection.add_argument("--validation-limit", type=int, default=250_000)
    lock = subparsers.add_parser("lock")
    lock.add_argument("--selection-report", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    _add_data_arguments(audit)
    audit.add_argument("--method-lock", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.phase == "lock":
        result = write_flare24_method_lock(
            args.selection_report,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "output": args.output.as_posix(),
                    "manifest_sha256": result["manifest_sha256"],
                },
                indent=2,
            )
        )
        return
    common = {
        "census_dir": args.census_dir,
        "recent_dir": args.recent_dir,
        "flight_recent_dir": args.flight_recent_dir,
        "graph_dir": args.graph_dir,
        "weather_feature_dir": args.weather_feature_dir,
        "rotation_feature_dir": args.rotation_feature_dir,
        "run_dir": args.run_dir,
        "output_path": args.output,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
    }
    if args.phase == "selection":
        result = run_flare24_selection(
            protocol_path=args.protocol,
            census_manifest=args.census_manifest,
            recent_manifest=args.recent_manifest,
            flight_recent_manifest=args.flight_recent_manifest,
            graph_manifest=args.graph_manifest,
            weather_feature_manifest=args.weather_feature_manifest,
            rotation_feature_manifest=args.rotation_feature_manifest,
            rows_per_train_month=args.rows_per_train_month,
            validation_limit=args.validation_limit,
            **common,
        )
    else:
        result = run_flare24_audit(
            args.method_lock,
            **common,
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": args.output.as_posix(),
                "report_sha256": result["report_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
