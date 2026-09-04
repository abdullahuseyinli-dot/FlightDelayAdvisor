"""Run the locked PAFRA models on the 2025 retrospective audit cohort."""

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
from numpy.typing import NDArray

from .bootstrap import paired_cluster_mean_difference
from .forecast_lock import validate_forecast_method_lock
from .forecast_modeling import attach_daily_forecasts
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .metrics import binary_metrics, clip_probabilities
from .modeling import CATEGORICAL_FEATURES, LOAD_COLUMNS
from .provenance import capture_provenance
from .recent import attach_recent_features
from .rolling import _atomic_parquet

AUDIT_YEAR = 2025


def _require_audit_year(year: int) -> None:
    if year != AUDIT_YEAR:
        raise PermissionError("this command is restricted to the locked 2025 retrospective audit")


def _verify_self_hashed_json(path: Path, field: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get(field)
    body = {key: value for key, value in payload.items() if key != field}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"self-hash failed for {path}")
    return payload


def _verify_bound_report(record: dict[str, Any], *, self_hash_field: str) -> dict[str, Any]:
    path = Path(record["path"])
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError(f"locked report checksum failed: {path}")
    payload = _verify_self_hashed_json(path, self_hash_field)
    if payload[self_hash_field] != record["self_hash"]:
        raise ValueError(f"locked report self-hash changed: {path}")
    return payload


def _verify_manifest_bound_to_selection(
    selection: dict[str, Any],
    *,
    name: str,
    self_hash_field: str = "manifest_sha256",
) -> tuple[Path, dict[str, Any]]:
    path = Path(selection[f"{name}_manifest"])
    if sha256_file(path) != selection[f"{name}_manifest_sha256"]:
        raise ValueError(f"{name} manifest no longer matches the method selection")
    payload = _verify_self_hashed_json(path, self_hash_field)
    expected_self_hash = selection.get(f"{name}_manifest_self_hash")
    if expected_self_hash is not None and payload[self_hash_field] != expected_self_hash:
        raise ValueError(f"{name} manifest self-hash no longer matches selection")
    return path, payload


def _records_for_year(
    manifest: dict[str, Any],
    *,
    year: int,
    expected: int,
) -> list[dict[str, Any]]:
    records = sorted(
        (record for record in manifest["outputs"] if int(record["year"]) == year),
        key=lambda record: (int(record.get("month", 0)), str(record.get("level", ""))),
    )
    if len(records) != expected:
        raise ValueError(f"expected {expected} manifest outputs for {year}, found {len(records)}")
    for record in records:
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"audit input checksum failed: {path}")
    return records


def _assert_records_under(records: list[dict[str, Any]], directory: Path) -> None:
    root = directory.resolve()
    if any(not Path(record["path"]).resolve().is_relative_to(root) for record in records):
        raise ValueError(f"manifest output is outside requested evidence directory: {directory}")


def _frozen_artifact(
    lock: dict[str, Any],
    *,
    task: str,
    kind: str,
    candidate: str | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = lock["frozen_artifacts"]
    matches = [
        record
        for record in records
        if record.get("task") == task
        and record.get("kind") == kind
        and (candidate is None or record.get("candidate") == candidate)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one locked {kind} artifact for {task}/{candidate}")
    return matches[0]


def _load_feature_partition(record: dict[str, Any], *, month: int) -> pd.DataFrame:
    frame = pd.read_parquet(Path(record["path"]), columns=list(LOAD_COLUMNS))
    if not frame["Year"].eq(AUDIT_YEAR).all() or not frame["Month"].eq(month).all():
        raise ValueError(f"feature partition does not match {AUDIT_YEAR}-{month:02d}")
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("string").fillna("__MISSING__")
    return frame.reset_index(drop=True)


def _eligible_mask_and_labels(
    frame: pd.DataFrame, task: str
) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
    if task == "delay":
        mask = np.asarray(
            frame["Cancelled"].eq(0) & frame["delay_label_observed"].eq(1),
            dtype=np.bool_,
        )
        label_column = "ArrDel15"
    elif task == "cancellation":
        mask = np.asarray(frame["Cancelled"].isin([0, 1]), dtype=np.bool_)
        label_column = "Cancelled"
    else:
        raise ValueError(f"unsupported audit task: {task}")
    labels = np.asarray(frame.loc[mask, label_column], dtype=np.int64)
    if not mask.any() or np.unique(labels).size != 2:
        raise ValueError(f"audit partition for {task} lacks both classes")
    return mask, labels


def _probability(model: Any, frame: pd.DataFrame) -> NDArray[np.float64]:
    probabilities = np.asarray(model.predict_proba(frame), dtype=np.float64)
    if probabilities.shape != (len(frame), 2):
        raise ValueError("locked binary model returned an unexpected probability shape")
    return clip_probabilities(probabilities[:, 1])


def _losses(labels: NDArray[np.int64], probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    values = clip_probabilities(probabilities)
    return np.asarray(
        -(labels * np.log(values) + (1 - labels) * np.log1p(-values)),
        dtype=np.float64,
    )


def _interval(interval: Any) -> dict[str, Any]:
    return {
        "estimate": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
        "confidence": interval.confidence,
        "clusters": interval.clusters,
        "repetitions": interval.repetitions,
        "seed": interval.seed,
    }


def _paired_probability_intervals(
    labels: NDArray[np.int64],
    candidate: NDArray[np.float64],
    reference: NDArray[np.float64],
    dates: NDArray[Any],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    log_interval = paired_cluster_mean_difference(
        _losses(labels, candidate),
        _losses(labels, reference),
        dates,
        repetitions=repetitions,
        seed=seed,
    )
    brier_interval = paired_cluster_mean_difference(
        np.square(candidate - labels),
        np.square(reference - labels),
        dates,
        repetitions=repetitions,
        seed=seed + 1,
    )
    return {
        "direction": "candidate_minus_reference; negative favours candidate",
        "log_loss": _interval(log_interval),
        "brier": _interval(brier_interval),
    }


def _prediction_frame(
    frame: pd.DataFrame,
    mask: NDArray[np.bool_],
    labels: NDArray[np.int64],
    probabilities: dict[str, NDArray[np.float32]],
) -> pd.DataFrame:
    output = frame.loc[
        mask,
        ["sample_id", "FlightDate", "Year", "Month", "Reporting_Airline", "Origin", "Dest", "Route"],
    ].reset_index(drop=True)
    output["label"] = labels.astype("int8")
    for name, values in probabilities.items():
        output[f"prob_{name}"] = values[mask]
    return output


def run_locked_forecast_audit(
    method_lock_path: Path,
    *,
    feature_dir: Path,
    recent_dir: Path,
    forecast_dir: Path,
    run_dir: Path,
    report_path: Path,
    evaluation_year: int = AUDIT_YEAR,
    bootstrap_repetitions: int = 2_000,
    seed: int = 20260903,
) -> dict[str, Any]:
    """Apply exact frozen artifacts month-wise; never fit on 2025 outcomes."""

    _require_audit_year(evaluation_year)
    if bootstrap_repetitions != 2_000:
        raise ValueError("the locked audit requires exactly 2000 bootstrap repetitions")
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse retrospective audit directory: {run_dir}")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite retrospective audit report: {report_path}")
    lock = validate_forecast_method_lock(method_lock_path)
    selection = _verify_bound_report(lock["selection_report"], self_hash_field="report_sha256")
    _verify_bound_report(lock["calibration_report"], self_hash_field="report_sha256")
    feature_manifest_path, feature_manifest = _verify_manifest_bound_to_selection(
        selection, name="feature"
    )
    recent_manifest_path, recent_manifest = _verify_manifest_bound_to_selection(
        selection, name="recent"
    )
    forecast_manifest_path, forecast_manifest = _verify_manifest_bound_to_selection(
        selection, name="forecast"
    )
    feature_records = _records_for_year(feature_manifest, year=AUDIT_YEAR, expected=12)
    recent_records = _records_for_year(recent_manifest, year=AUDIT_YEAR, expected=5)
    forecast_records = _records_for_year(forecast_manifest, year=AUDIT_YEAR, expected=1)
    _assert_records_under(feature_records, feature_dir)
    _assert_records_under(recent_records, recent_dir)
    _assert_records_under(forecast_records, forecast_dir)
    if [int(record["month"]) for record in feature_records] != list(range(1, 13)):
        raise ValueError("audit feature manifest does not contain exactly months 1 through 12")
    run_dir.mkdir(parents=True)

    model_records: dict[str, dict[str, dict[str, Any]]] = {}
    models: dict[str, dict[str, Any]] = {}
    calibrators: dict[str, Any] = {}
    for task in ("delay", "cancellation"):
        model_records[task] = {}
        models[task] = {}
        for candidate in ("baseline", "operational_residual", "forecast_residual"):
            record = _frozen_artifact(
                lock,
                task=task,
                kind="model",
                candidate=candidate,
            )
            model_records[task][candidate] = record
            models[task][candidate] = joblib.load(Path(record["path"]))
        calibrator_record = _frozen_artifact(lock, task=task, kind="calibrator")
        calibrators[task] = joblib.load(Path(calibrator_record["path"]))

    started = time.perf_counter()
    artifacts: list[dict[str, Any]] = []
    accumulators: dict[str, dict[str, list[np.ndarray]]] = {
        task: {
            "labels": [],
            "dates": [],
            "baseline": [],
            "operational_residual": [],
            "forecast_residual": [],
            "prelocked_calibration": [],
        }
        for task in ("delay", "cancellation")
    }
    monthly_results: dict[str, list[dict[str, Any]]] = {
        "delay": [],
        "cancellation": [],
    }
    input_rows = 0
    for record in feature_records:
        month = int(record["month"])
        frame = _load_feature_partition(record, month=month)
        input_rows += len(frame)
        frame = attach_recent_features(frame, recent_dir=recent_dir)
        frame = attach_daily_forecasts(frame, forecast_dir=forecast_dir)
        for task in ("delay", "cancellation"):
            mask, labels = _eligible_mask_and_labels(frame, task)
            probabilities64 = {
                candidate: _probability(models[task][candidate], frame)
                for candidate in ("baseline", "operational_residual", "forecast_residual")
            }
            probabilities64["prelocked_calibration"] = clip_probabilities(
                calibrators[task].predict(probabilities64["forecast_residual"])
            )
            monthly_probabilities = {
                name: np.asarray(values, dtype=np.float32)
                for name, values in probabilities64.items()
            }
            prediction = _prediction_frame(frame, mask, labels, monthly_probabilities)
            artifact = _atomic_parquet(
                prediction,
                run_dir / "predictions" / task / f"month={month:02d}.parquet",
            )
            artifacts.append(
                {
                    "kind": "predictions",
                    "task": task,
                    "year": AUDIT_YEAR,
                    "month": month,
                    **artifact,
                }
            )
            selected_probabilities = {
                name: np.asarray(values[mask], dtype=np.float32)
                for name, values in monthly_probabilities.items()
            }
            monthly_results[task].append(
                {
                    "month": month,
                    "n": len(labels),
                    "prevalence": float(labels.mean()),
                    "metrics": {
                        name: binary_metrics(labels, values).as_dict()
                        for name, values in selected_probabilities.items()
                    },
                }
            )
            accumulators[task]["labels"].append(labels.astype("int8"))
            accumulators[task]["dates"].append(prediction["FlightDate"].to_numpy())
            for name, values in selected_probabilities.items():
                accumulators[task][name].append(values)
            del prediction, monthly_probabilities, probabilities64, selected_probabilities
        del frame
        gc.collect()

    task_results: list[dict[str, Any]] = []
    for task_index, task in enumerate(("delay", "cancellation")):
        labels = np.concatenate(accumulators[task]["labels"]).astype(np.int64)
        dates = np.concatenate(accumulators[task]["dates"])
        annual_probabilities = {
            name: np.concatenate(accumulators[task][name]).astype(np.float64)
            for name in (
                "baseline",
                "operational_residual",
                "forecast_residual",
                "prelocked_calibration",
            )
        }
        task_seed = seed + task_index * 100
        task_results.append(
            {
                "task": task,
                "n": len(labels),
                "prevalence": float(labels.mean()),
                "metrics": {
                    name: binary_metrics(labels, values).as_dict()
                    for name, values in annual_probabilities.items()
                },
                "paired_forecast_minus_baseline": _paired_probability_intervals(
                    labels,
                    annual_probabilities["forecast_residual"],
                    annual_probabilities["baseline"],
                    dates,
                    repetitions=bootstrap_repetitions,
                    seed=task_seed,
                ),
                "paired_forecast_minus_operational_residual": _paired_probability_intervals(
                    labels,
                    annual_probabilities["forecast_residual"],
                    annual_probabilities["operational_residual"],
                    dates,
                    repetitions=bootstrap_repetitions,
                    seed=task_seed + 10,
                ),
                "paired_prelocked_calibration_minus_raw": _paired_probability_intervals(
                    labels,
                    annual_probabilities["prelocked_calibration"],
                    annual_probabilities["forecast_residual"],
                    dates,
                    repetitions=bootstrap_repetitions,
                    seed=task_seed + 20,
                ),
                "monthly": monthly_results[task],
            }
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_2025_RETROSPECTIVE_AUDIT_NOT_CONFIRMATORY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "prior_anchored_forecast_residual_adaptation",
        "working_acronym": "PAFRA",
        "method_lock": method_lock_path.as_posix(),
        "method_lock_sha256": sha256_file(method_lock_path),
        "method_lock_self_hash": lock["lock_sha256"],
        "evaluation_year": AUDIT_YEAR,
        "evaluation_dates": ["2025-01-01", "2025-12-31"],
        "input_rows": input_rows,
        "feature_manifest": feature_manifest_path.as_posix(),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "recent_manifest": recent_manifest_path.as_posix(),
        "recent_manifest_sha256": sha256_file(recent_manifest_path),
        "forecast_manifest": forecast_manifest_path.as_posix(),
        "forecast_manifest_sha256": sha256_file(forecast_manifest_path),
        "frozen_model_artifacts": model_records,
        "model_refit": False,
        "calibrator_refit": False,
        "bootstrap_repetitions": bootstrap_repetitions,
        "seed": seed,
        "task_results": task_results,
        "artifacts": artifacts,
        "versions": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
            "catboost": version("catboost"),
        },
        "provenance": capture_provenance(
            (
                Path(__file__),
                Path(__file__).with_name("forecast_lock.py"),
                Path(__file__).with_name("forecast_modeling.py"),
            )
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "outcomes_accessed": {"maximum_calendar_date": "2025-12-31"},
        "claim_limit": lock["blinding_status"],
        "confirmation_gate": lock["confirmation_gate"],
    }
    result["report_sha256"] = canonical_json_sha256(result)
    write_canonical_json(report_path, result)
    write_canonical_json(run_dir / "run_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method_lock", type=Path)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--recent-dir", type=Path, required=True)
    parser.add_argument("--forecast-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--evaluation-year", type=int, default=AUDIT_YEAR)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_locked_forecast_audit(
        args.method_lock,
        feature_dir=args.feature_dir,
        recent_dir=args.recent_dir,
        forecast_dir=args.forecast_dir,
        run_dir=args.run_dir,
        report_path=args.report,
        evaluation_year=args.evaluation_year,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "report": args.report.as_posix(),
                "status": result["status"],
                "metrics": {
                    item["task"]: item["metrics"]["forecast_residual"]
                    for item in result["task_results"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
