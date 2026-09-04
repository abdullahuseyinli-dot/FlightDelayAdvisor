"""Recover FLARE-24 audit reporting from immutable completed month partitions.

This module is intentionally separate from the method-locked selection and audit
implementation.  It is an evidence-recovery path for a run where all monthly
inference artifacts were written before report assembly rejected harmless float32
simplex round-off.  It never fits or applies a flight model.
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
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.typing import ArrayLike, NDArray

from .flare_evaluation import evaluate_joint_probabilities
from .flare_study import (
    AGGREGATE_HISTORY_COLUMNS,
    CANDIDATES,
    PREDICTION_ID_COLUMNS,
    _frozen_implementation_files,
    _monthly_joint_scores,
    _self_hashed_payload,
    _stratified_joint_scores,
)
from .flare_validation import validate_method_lock_artifact
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import PROJECT_ROOT, capture_provenance

AUDIT_METHODS = (*CANDIDATES, "ensemble", "reconciled")
JOINT_STATES = ("on_time", "delayed", "cancelled")
FLOAT32_SIMPLEX_ACCEPTANCE = 1e-6


def normalize_persisted_joint(
    values: ArrayLike,
    *,
    method: str,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Normalize harmless float32 simplex drift and return its complete audit."""

    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or len(probabilities) == 0:
        raise ValueError(f"persisted {method} probabilities must have shape (n, 3)")
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError(f"persisted {method} probabilities are not finite and non-negative")
    row_sums = probabilities.sum(axis=1)
    absolute_error = np.abs(row_sums - 1.0)
    maximum_error = float(absolute_error.max())
    if maximum_error > FLOAT32_SIMPLEX_ACCEPTANCE or (row_sums <= 0.0).any():
        raise ValueError(
            f"persisted {method} probability drift exceeds the float32 recovery bound: "
            f"{maximum_error}"
        )
    normalized = probabilities / row_sums[:, None]
    maximum_adjustment = float(np.max(np.abs(normalized - probabilities)))
    return np.asarray(normalized, dtype=np.float64), {
        "storage_dtype": "float32",
        "evaluation_dtype": "float64",
        "rows": len(probabilities),
        "maximum_absolute_row_sum_error_before_normalization": maximum_error,
        "rows_over_original_1e_8_absolute_tolerance": int((absolute_error > 1e-8).sum()),
        "rows_over_recovery_1e_6_absolute_tolerance": int(
            (absolute_error > FLOAT32_SIMPLEX_ACCEPTANCE).sum()
        ),
        "maximum_absolute_probability_adjustment": maximum_adjustment,
        "normalization": "divide each three-state row by its float64 row sum",
    }


def _artifact_record(path: Path, *, year: int | None = None, month: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing preserved FLARE-24 audit artifact: {path}")
    record: dict[str, Any] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".parquet":
        record["rows"] = pq.ParquetFile(path).metadata.num_rows
    if year is not None:
        record["year"] = year
    if month is not None:
        record["month"] = month
    return record


def _verified_relative_record(record: dict[str, Any], *, role: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file() or sha256_file(path) != str(record.get("sha256", "")):
        raise ValueError(f"{role} checksum failed: {path}")
    return path


def _verify_failed_run_record(failure_record_path: Path, run_dir: Path) -> dict[str, Any]:
    failure: dict[str, Any] = json.loads(failure_record_path.read_text(encoding="utf-8"))
    if failure.get("status") != "FAILED_AFTER_ALL_MONTHLY_INFERENCE_ARTIFACTS_WRITTEN":
        raise ValueError("recovery requires the adjudicated FLARE-24 audit failure record")
    preserved = failure.get("preserved_evidence", {})
    if Path(str(preserved.get("run_directory", ""))).resolve() != run_dir.resolve():
        raise ValueError("failure record points to a different FLARE-24 run directory")
    _verified_relative_record(preserved["console_log"], role="failed-audit console log")
    _verified_relative_record(
        preserved["deterministic_reproduction_log"],
        role="failed-audit deterministic reproduction log",
    )
    return failure


def _verify_frozen_sources(selection: dict[str, Any]) -> None:
    records = list(selection.get("provenance", {}).get("source_files", []))
    if not records:
        raise ValueError("locked FLARE-24 selection has no frozen source records")
    for record in records:
        _verified_relative_record(record, role="frozen FLARE-24 implementation")


def _aggregate_history_records(census_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for month in range(1, 13):
        path = census_dir / "year=2024" / f"month={month:02d}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing 2024 aggregate-history partition: {path}")
        dates = pd.read_parquet(path, columns=["FlightDate"])["FlightDate"]
        parsed = pd.to_datetime(dates, errors="raise")
        if parsed.empty or not parsed.dt.year.eq(2024).all() or not parsed.dt.month.eq(month).all():
            raise ValueError(f"aggregate-history period mismatch: {path}")
        records.append(
            {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "columns_read": list(AGGREGATE_HISTORY_COLUMNS),
                "maximum_outcome_date": parsed.max().date().isoformat(),
            }
        )
    return records


def _expected_partition_paths(run_dir: Path) -> tuple[list[Path], list[Path], Path]:
    predictions = [
        run_dir / "predictions" / f"audit_2025_{month:02d}.parquet"
        for month in range(1, 13)
    ]
    aggregates = [
        run_dir / "aggregate" / f"forecasts_2025_{month:02d}.parquet"
        for month in range(1, 13)
    ]
    aggregate_model = run_dir / "aggregate" / "history_2024_for_2025.joblib"
    expected = {*predictions, *aggregates, aggregate_model}
    actual = set(run_dir.rglob("*.parquet")) | set(run_dir.rglob("*.joblib"))
    if actual != expected:
        missing = sorted(path.as_posix() for path in expected - actual)
        unexpected = sorted(path.as_posix() for path in actual - expected)
        raise ValueError(
            f"preserved audit artifact inventory mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    partials = sorted(run_dir.rglob("*.part"))
    if partials:
        raise ValueError(f"unadjudicated partial audit artifacts remain: {partials}")
    return predictions, aggregates, aggregate_model


def _load_evaluation_evidence(
    prediction_paths: list[Path],
) -> tuple[
    pd.DataFrame,
    dict[str, NDArray[np.float64]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    evaluation_frames: list[pd.DataFrame] = []
    probability_parts: dict[str, list[NDArray[np.float32]]] = {
        method: [] for method in AUDIT_METHODS
    }
    prediction_records: list[dict[str, Any]] = []
    evidence_columns = [
        *PREDICTION_ID_COLUMNS,
        "weather_severity_index",
        "weather_covariate_available",
    ]
    probability_columns = [
        f"prob_{method}_{state}" for method in AUDIT_METHODS for state in JOINT_STATES
    ]
    for month, path in enumerate(prediction_paths, start=1):
        frame = pd.read_parquet(path, columns=[*evidence_columns, *probability_columns])
        if (
            frame.empty
            or not frame["Year"].eq(2025).all()
            or not frame["Month"].eq(month).all()
        ):
            raise ValueError(f"preserved FLARE-24 audit period mismatch: {path}")
        dates = pd.to_datetime(frame["FlightDate"], errors="raise")
        if not dates.dt.year.eq(2025).all() or not dates.dt.month.eq(month).all():
            raise ValueError(f"preserved FLARE-24 audit date mismatch: {path}")
        evaluation_frames.append(frame.loc[:, evidence_columns].copy())
        for method in AUDIT_METHODS:
            columns = [f"prob_{method}_{state}" for state in JOINT_STATES]
            probability_parts[method].append(frame.loc[:, columns].to_numpy(dtype=np.float32))
        record = _artifact_record(path, year=2025, month=month)
        if int(record["rows"]) != len(frame):
            raise ValueError(f"preserved FLARE-24 audit row count mismatch: {path}")
        prediction_records.append(record)
        del frame
        gc.collect()

    evaluation_frame = pd.concat(evaluation_frames, ignore_index=True)
    del evaluation_frames
    evaluation_probabilities: dict[str, NDArray[np.float64]] = {}
    normalization: dict[str, Any] = {}
    for method, parts in probability_parts.items():
        concatenated = np.concatenate(parts, axis=0)
        normalized, audit = normalize_persisted_joint(concatenated, method=method)
        evaluation_probabilities[method] = normalized
        normalization[method] = audit
        del concatenated, parts
        gc.collect()
    del probability_parts
    return evaluation_frame, evaluation_probabilities, prediction_records, normalization


def _monthly_scores(
    frame: pd.DataFrame,
    probabilities: dict[str, NDArray[np.float64]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for month in range(1, 13):
        mask = frame["Month"].eq(month).to_numpy()
        month_frame = frame.loc[mask].reset_index(drop=True)
        month_probabilities = {name: values[mask] for name, values in probabilities.items()}
        records.append(
            {
                "year": 2025,
                "month": month,
                "rows": len(month_frame),
                "proper_scores": _monthly_joint_scores(month_frame, month_probabilities),
            }
        )
    return records


def recover_flare24_audit_report(
    method_lock_path: Path,
    *,
    failed_run_dir: Path,
    census_dir: Path,
    failure_record_path: Path,
    output_path: Path,
    bootstrap_repetitions: int = 2_000,
    seed: int = 20260903,
) -> dict[str, Any]:
    """Finalize a failed audit from its complete immutable monthly inference evidence."""

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite recovered FLARE-24 audit: {output_path}")
    recovered_manifest_path = failed_run_dir / "recovered_run_manifest.json"
    if recovered_manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite recovered FLARE-24 run manifest: {recovered_manifest_path}"
        )
    if bootstrap_repetitions < 100:
        raise ValueError("FLARE-24 recovery requires at least 100 bootstrap repetitions")
    validate_method_lock_artifact(method_lock_path)
    method_lock, lock_hash_key = _self_hashed_payload(method_lock_path)
    selection_record = method_lock["selection_report"]
    selection_path = Path(str(selection_record["path"]))
    selection, selection_hash_key = _self_hashed_payload(selection_path)
    if sha256_file(selection_path) != selection_record["sha256"]:
        raise ValueError("locked FLARE-24 selection report changed before recovery")
    _verify_frozen_sources(selection)
    failure = _verify_failed_run_record(failure_record_path, failed_run_dir)
    prediction_paths, aggregate_paths, aggregate_model_path = _expected_partition_paths(
        failed_run_dir
    )
    started = time.perf_counter()
    print("loading and normalizing preserved 2025 monthly predictions", flush=True)
    evaluation_frame, evaluation_probabilities, prediction_records, normalization = (
        _load_evaluation_evidence(prediction_paths)
    )
    print("evaluating frozen FLARE-24 probabilities on the full 2025 audit", flush=True)
    evaluation_inputs: dict[str, ArrayLike] = dict(evaluation_probabilities)
    evaluation = evaluate_joint_probabilities(
        evaluation_frame,
        evaluation_inputs,
        reference_method="baseline",
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=seed + 10_000,
    )
    print("computing monthly and prespecified heterogeneity diagnostics", flush=True)
    monthly_records = _monthly_scores(evaluation_frame, evaluation_probabilities)

    choices = method_lock["frozen_choices"]
    severity_cutpoints = np.asarray(choices["weather_severity_cutpoints"], dtype=np.float64)
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

    print("verifying preserved aggregate and 2024 history evidence", flush=True)
    aggregate_model_record = _artifact_record(aggregate_model_path)
    aggregate_model = joblib.load(aggregate_model_path)
    model_card = aggregate_model.model_card().as_dict()
    if model_card.get("maximum_history_date") != "2024-12-31":
        raise ValueError("preserved aggregate model does not stop at 2024-12-31")
    aggregate_records: list[dict[str, Any]] = []
    for month, path in enumerate(aggregate_paths, start=1):
        dates = pd.to_datetime(
            pd.read_parquet(path, columns=["FlightDate"])["FlightDate"], errors="raise"
        )
        if dates.empty or not dates.dt.year.eq(2025).all() or not dates.dt.month.eq(month).all():
            raise ValueError(f"preserved aggregate forecast period mismatch: {path}")
        aggregate_records.append(_artifact_record(path, year=2025, month=month))

    selected_multiplier = choices["reconciliation_variance_multiplier"]
    if choices["reconciliation_enabled"] is not False or selected_multiplier is not None:
        raise ValueError("this recovery path requires the locked identity reconciliation control")
    failure_record = {
        "path": failure_record_path.as_posix(),
        "sha256": sha256_file(failure_record_path),
        "status": failure["status"],
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_2025_FLARE24_RETROSPECTIVE_AUDIT_NOT_BLIND_CONFIRMATION",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "FLARE-24",
        "selection_report": {
            "path": selection_path.as_posix(),
            "sha256": sha256_file(selection_path),
            "self_hash_key": selection_hash_key,
            "self_hash": selection[selection_hash_key],
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
            "ensemble_weights": choices["ensemble_weights"],
            "reconciliation_enabled": False,
            "reconciliation_variance_multiplier": None,
            "feature_sets": choices["feature_sets"],
            "model_artifacts": choices["model_artifacts"],
            "calibrator_artifacts": choices["calibrator_artifacts"],
        },
        "aggregate_refit": {
            "permitted_role": "2024 outcomes are historical before every 2025 target date",
            "model_card": model_card,
            "history_inputs": _aggregate_history_records(census_dir),
            **aggregate_model_record,
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
        "reconciliation_diagnostics": [],
        "bootstrap_repetitions": bootstrap_repetitions,
        "seed": seed,
        "recovery": {
            "kind": "report_assembly_only_from_immutable_monthly_predictions",
            "failed_run_record": failure_record,
            "flight_model_refit": False,
            "calibrator_refit": False,
            "reprediction": False,
            "method_selection_changed": False,
            "probability_normalization_audit": normalization,
            "acceptance_bound": FLOAT32_SIMPLEX_ACCEPTANCE,
            "reason": (
                "Persisted float32 rows differed from one by at most normal float32 "
                "round-off but the original finalizer imposed a 1e-8 zero-relative-tolerance "
                "simplex check. Each row was normalized in float64 before applying the frozen "
                "evaluation functions."
            ),
        },
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
        "provenance": capture_provenance((*_frozen_implementation_files(), Path(__file__))),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "This is a pre-existing-outcome retrospective audit, not a never-seen "
            "confirmation. Report assembly was transparently recovered from immutable "
            "monthly prediction artifacts after bounded float32 normalization. It supports "
            "comparative predictive claims on the BTS top-100 cohort only; it does not "
            "establish causal, operational, real-time, fairness, or safety effects."
        ),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(output_path, report)
    write_canonical_json(recovered_manifest_path, report)
    print("recovered FLARE-24 audit report written", flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-lock", type=Path, required=True)
    parser.add_argument("--failed-run-dir", type=Path, required=True)
    parser.add_argument("--census-dir", type=Path, required=True)
    parser.add_argument("--failure-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = recover_flare24_audit_report(
        args.method_lock,
        failed_run_dir=args.failed_run_dir,
        census_dir=args.census_dir,
        failure_record_path=args.failure_record,
        output_path=args.output,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": args.output.as_posix(),
                "report_sha256": report["report_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
