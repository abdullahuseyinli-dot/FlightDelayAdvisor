"""Select baseline-anchored forecast residual adapters within calendar 2024."""

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

import numpy as np
import pandas as pd

from .forecast_benchmark import _paired_intervals, _verify_feature_manifest
from .forecast_modeling import (
    fit_forecast_residual_catboost,
    forecast_residual_profile,
    load_forecast_recent_year,
)
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .modeling import TaskName, load_feature_years, task_view
from .provenance import capture_provenance
from .recent import attach_recent_features
from .recent_modeling import fit_recent_catboost, recent_model_profile
from .recent_rolling import _frozen_parameters, _verify_tuning_report
from .rolling import _atomic_joblib, _atomic_parquet, _score_task

ADAPTER_TRAIN_END = pd.Timestamp("2024-08-31")
EARLY_STOP_START = pd.Timestamp("2024-09-01")
EARLY_STOP_END = pd.Timestamp("2024-09-30")
SELECTION_START = pd.Timestamp("2024-10-01")
SELECTION_END = pd.Timestamp("2024-12-31")


def _split_selection_year(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(frame["FlightDate"], errors="raise")
    train = frame.loc[dates.le(ADAPTER_TRAIN_END)].reset_index(drop=True)
    validation = frame.loc[
        dates.ge(EARLY_STOP_START) & dates.le(EARLY_STOP_END)
    ].reset_index(drop=True)
    selection = frame.loc[
        dates.ge(SELECTION_START) & dates.le(SELECTION_END)
    ].reset_index(drop=True)
    if any(part.empty for part in (train, validation, selection)):
        raise ValueError("2024 adaptation split produced an empty partition")
    date_sets = [set(pd.to_datetime(part["FlightDate"]).unique()) for part in (train, validation, selection)]
    if date_sets[0] & date_sets[1] or date_sets[0] & date_sets[2] or date_sets[1] & date_sets[2]:
        raise AssertionError("forecast adaptation date partitions overlap")
    return train, validation, selection


def _selection_prediction_frame(
    frame: pd.DataFrame,
    *,
    baseline: np.ndarray,
    operational: np.ndarray,
    forecast: np.ndarray,
) -> pd.DataFrame:
    columns = [
        "sample_id",
        "FlightDate",
        "Year",
        "Month",
        "Reporting_Airline",
        "Origin",
        "Dest",
        "Route",
        "ArrDel15",
        "Cancelled",
        "delay_label_observed",
        "joint_label_observed",
        "disruption_state",
    ]
    result = frame.loc[:, columns].copy()
    result["prob_baseline"] = np.asarray(baseline, dtype=np.float32)
    result["prob_operational_residual"] = np.asarray(operational, dtype=np.float32)
    result["prob_forecast_residual"] = np.asarray(forecast, dtype=np.float32)
    return result


def run_forecast_adaptation_selection(
    *,
    feature_dir: Path,
    feature_manifest: Path,
    recent_dir: Path,
    recent_manifest: Path,
    forecast_dir: Path,
    forecast_manifest: Path,
    tuning_report_paths: tuple[Path, ...],
    run_dir: Path,
    report_path: Path,
    rows_per_history_year: int = 50_000,
    bootstrap_repetitions: int = 2_000,
    seed: int = 20260903,
) -> dict[str, Any]:
    """Use predeclared within-2024 dates; never load a 2025 or 2026 outcome."""

    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse forecast adaptation directory: {run_dir}")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite forecast adaptation report: {report_path}")
    tuning_payloads = [_verify_tuning_report(path) for path in tuning_report_paths]
    by_task = {str(payload["task"]): payload for payload in tuning_payloads}
    if set(by_task) != {"delay", "cancellation"} or len(tuning_payloads) != 2:
        raise ValueError("exactly one delay and one cancellation tuning report are required")
    if any(payload.get("schedule_context") for payload in tuning_payloads):
        raise ValueError("forecast residual selection requires strict non-proxy core reports")
    forecast_payload = _verify_feature_manifest(forecast_manifest)
    run_dir.mkdir(parents=True)

    started = time.perf_counter()
    history = load_feature_years(
        feature_dir,
        tuple(range(2011, 2024)),
        rows_per_year=rows_per_history_year,
        seed=seed,
    )
    history = attach_recent_features(history, recent_dir=recent_dir)
    selection_year = load_forecast_recent_year(
        feature_dir,
        recent_dir,
        forecast_dir,
        2024,
        limit=None,
        seed=seed,
    )
    adapter_train, early_stop, selection = _split_selection_year(selection_year)

    task_results: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for task in ("delay", "cancellation"):
        task_name: TaskName = task
        task_started = time.perf_counter()
        payload = by_task[task]
        parameters = _frozen_parameters(payload)
        include_cross = payload["candidate"] == "network"
        history_task, history_labels = task_view(history, task_name)
        base_model = fit_recent_catboost(
            history_task,
            history_labels,
            task=task_name,
            params=parameters,
            include_cross_direction=include_cross,
        )
        train_task, train_labels = task_view(adapter_train, task_name)
        valid_task, valid_labels = task_view(early_stop, task_name)
        operational_model = fit_forecast_residual_catboost(
            train_task,
            train_labels,
            baseline_model=base_model,
            task=task_name,
            include_cross_direction=include_cross,
            include_weather=False,
            validation_frame=valid_task,
            validation_labels=valid_labels,
        )
        forecast_model = fit_forecast_residual_catboost(
            train_task,
            train_labels,
            baseline_model=base_model,
            task=task_name,
            include_cross_direction=include_cross,
            include_weather=True,
            validation_frame=valid_task,
            validation_labels=valid_labels,
        )

        baseline_probabilities = np.asarray(
            base_model.predict_proba(selection),
            dtype=np.float64,
        )[:, 1]
        operational_probabilities = np.asarray(
            operational_model.predict_proba(selection),
            dtype=np.float64,
        )[:, 1]
        forecast_probabilities = np.asarray(
            forecast_model.predict_proba(selection),
            dtype=np.float64,
        )[:, 1]
        metrics = {
            "baseline": _score_task(selection, baseline_probabilities, task_name),
            "operational_residual": _score_task(
                selection,
                operational_probabilities,
                task_name,
            ),
            "forecast_residual": _score_task(
                selection,
                forecast_probabilities,
                task_name,
            ),
        }
        selected_candidate = min(
            metrics,
            key=lambda name: float(metrics[name]["log_loss"]),
        )
        model_records = {
            "baseline": _atomic_joblib(
                base_model,
                run_dir / "models" / f"{task}_baseline.joblib",
            ),
            "operational_residual": _atomic_joblib(
                operational_model,
                run_dir / "models" / f"{task}_operational_residual.joblib",
            ),
            "forecast_residual": _atomic_joblib(
                forecast_model,
                run_dir / "models" / f"{task}_forecast_residual.joblib",
            ),
        }
        artifacts.extend(
            {"kind": "model", "task": task, "candidate": name, **record}
            for name, record in model_records.items()
        )
        artifacts.append(
            {
                "kind": "predictions",
                "task": task,
                **_atomic_parquet(
                    _selection_prediction_frame(
                        selection,
                        baseline=baseline_probabilities,
                        operational=operational_probabilities,
                        forecast=forecast_probabilities,
                    ),
                    run_dir / "predictions" / f"{task}_2024_q4.parquet",
                ),
            }
        )
        task_results.append(
            {
                "task": task,
                "historical_training_rows": len(history_task),
                "adapter_training_rows": len(train_task),
                "early_stopping_rows": len(valid_task),
                "selection_rows": len(selection),
                "parameters": parameters,
                "base_profile": recent_model_profile(include_cross),
                "operational_residual_profile": forecast_residual_profile(
                    include_weather=False
                ),
                "forecast_residual_profile": forecast_residual_profile(
                    include_weather=True
                ),
                "operational_best_iteration": int(
                    operational_model.estimator.get_best_iteration()
                ),
                "forecast_best_iteration": int(forecast_model.estimator.get_best_iteration()),
                "metrics": metrics,
                "selected_by_log_loss": selected_candidate,
                "paired_forecast_minus_baseline": _paired_intervals(
                    selection,
                    baseline_probabilities,
                    forecast_probabilities,
                    task=task_name,
                    repetitions=bootstrap_repetitions,
                    seed=seed,
                ),
                "paired_forecast_minus_operational_residual": _paired_intervals(
                    selection,
                    operational_probabilities,
                    forecast_probabilities,
                    task=task_name,
                    repetitions=bootstrap_repetitions,
                    seed=seed + 11,
                ),
                "elapsed_seconds": time.perf_counter() - task_started,
            }
        )
        del base_model, operational_model, forecast_model
        del baseline_probabilities, operational_probabilities, forecast_probabilities
        del history_task, history_labels, train_task, train_labels, valid_task, valid_labels
        gc.collect()

    feature_payload: dict[str, Any] = json.loads(feature_manifest.read_text(encoding="utf-8"))
    recent_payload: dict[str, Any] = json.loads(recent_manifest.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_2024_SELECTION_NOT_CONFIRMATORY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "candidate_name": "prior_anchored_forecast_residual_adaptation",
        "feature_variant": feature_payload["feature_variant"],
        "feature_manifest": feature_manifest.as_posix(),
        "feature_manifest_sha256": sha256_file(feature_manifest),
        "recent_manifest": recent_manifest.as_posix(),
        "recent_manifest_sha256": sha256_file(recent_manifest),
        "recent_manifest_self_hash": recent_payload["manifest_sha256"],
        "forecast_manifest": forecast_manifest.as_posix(),
        "forecast_manifest_sha256": sha256_file(forecast_manifest),
        "forecast_manifest_self_hash": forecast_payload["manifest_sha256"],
        "tuning_reports": [
            {"path": path.as_posix(), "sha256": sha256_file(path)}
            for path in tuning_report_paths
        ],
        "history_years": list(range(2011, 2024)),
        "rows_per_history_year": rows_per_history_year,
        "adapter_train_dates": ["2024-01-01", ADAPTER_TRAIN_END.date().isoformat()],
        "early_stopping_dates": [
            EARLY_STOP_START.date().isoformat(),
            EARLY_STOP_END.date().isoformat(),
        ],
        "selection_dates": [
            SELECTION_START.date().isoformat(),
            SELECTION_END.date().isoformat(),
        ],
        "loaded_rows": {
            "history": len(history),
            "adapter_train": len(adapter_train),
            "early_stopping": len(early_stop),
            "selection": len(selection),
        },
        "selection_rule": "minimum Q4 2024 log loss within each task",
        "outcomes_accessed": {"maximum_calendar_date": "2024-12-31"},
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
                Path(__file__).with_name("forecast_modeling.py"),
                Path(__file__).with_name("recent_modeling.py"),
            )
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "Candidate selection uses 2024 only. A weather benefit requires superiority to "
            "both the frozen base and operational-only residual on proper scores and must be "
            "audited on a later untouched year."
        ),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(report_path, report)
    write_canonical_json(run_dir / "run_manifest.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--recent-dir", type=Path, required=True)
    parser.add_argument("--recent-manifest", type=Path, required=True)
    parser.add_argument("--forecast-dir", type=Path, required=True)
    parser.add_argument("--forecast-manifest", type=Path, required=True)
    parser.add_argument("--tuning-reports", type=Path, nargs=2, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rows-per-history-year", type=int, default=50_000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_forecast_adaptation_selection(
        feature_dir=args.feature_dir,
        feature_manifest=args.feature_manifest,
        recent_dir=args.recent_dir,
        recent_manifest=args.recent_manifest,
        forecast_dir=args.forecast_dir,
        forecast_manifest=args.forecast_manifest,
        tuning_report_paths=tuple(args.tuning_reports),
        run_dir=args.run_dir,
        report_path=args.report,
        rows_per_history_year=args.rows_per_history_year,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "report": args.report.as_posix(),
                "selected": {
                    item["task"]: item["selected_by_log_loss"]
                    for item in result["task_results"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
