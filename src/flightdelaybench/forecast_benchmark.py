"""Evaluate observed-to-forecast transfer on the 2024 selection year."""

from __future__ import annotations

import argparse
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

from .bootstrap import paired_cluster_mean_difference
from .forecast_modeling import (
    fit_weather_transfer_catboost,
    load_forecast_recent_year,
    load_observed_recent_years,
    weather_transfer_profile,
)
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .metrics import clip_probabilities
from .modeling import TaskName, task_view, temporal_sample_weights
from .provenance import capture_provenance
from .recent_modeling import fit_recent_catboost, recent_model_profile
from .recent_rolling import _frozen_parameters, _verify_tuning_report
from .rolling import _atomic_joblib, _atomic_parquet, _score_task


def _verify_feature_manifest(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("manifest_sha256")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"forecast feature manifest self-hash failed: {path}")
    if int(payload.get("fixed_lead_hours", -1)) != 24:
        raise ValueError("forecast benchmark requires fixed 24-hour covariates")
    return payload


def _paired_intervals(
    evaluation: pd.DataFrame,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    task: TaskName,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    eligible, labels = task_view(evaluation, task)
    if task == "delay":
        mask = evaluation["Cancelled"].eq(0) & evaluation["delay_label_observed"].eq(1)
    else:
        mask = evaluation["Cancelled"].isin([0, 1])
    p_reference = clip_probabilities(reference[np.asarray(mask)])
    p_candidate = clip_probabilities(candidate[np.asarray(mask)])
    y = labels.astype(np.float64)
    log_reference = -(y * np.log(p_reference) + (1.0 - y) * np.log1p(-p_reference))
    log_candidate = -(y * np.log(p_candidate) + (1.0 - y) * np.log1p(-p_candidate))
    brier_reference = np.square(p_reference - y)
    brier_candidate = np.square(p_candidate - y)
    clusters = eligible["FlightDate"].to_numpy()
    log_interval = paired_cluster_mean_difference(
        log_candidate,
        log_reference,
        clusters,
        repetitions=repetitions,
        seed=seed,
    )
    brier_interval = paired_cluster_mean_difference(
        brier_candidate,
        brier_reference,
        clusters,
        repetitions=repetitions,
        seed=seed + 1,
    )
    return {
        "direction": "weather_transfer_minus_recent; negative favours weather transfer",
        "log_loss": {
            "estimate": log_interval.estimate,
            "lower": log_interval.lower,
            "upper": log_interval.upper,
            "confidence": log_interval.confidence,
            "clusters": log_interval.clusters,
            "repetitions": log_interval.repetitions,
            "seed": log_interval.seed,
        },
        "brier": {
            "estimate": brier_interval.estimate,
            "lower": brier_interval.lower,
            "upper": brier_interval.upper,
            "confidence": brier_interval.confidence,
            "clusters": brier_interval.clusters,
            "repetitions": brier_interval.repetitions,
            "seed": brier_interval.seed,
        },
    }


def run_forecast_benchmark(
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
    train_years: tuple[int, ...] = tuple(range(2011, 2024)),
    evaluation_year: int = 2024,
    rows_per_train_year: int = 50_000,
    evaluation_limit: int | None = 300_000,
    temporal_weight_half_life: float | None = None,
    bootstrap_repetitions: int = 2_000,
    seed: int = 20260903,
) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse forecast benchmark directory: {run_dir}")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite forecast benchmark report: {report_path}")
    if evaluation_year != 2024 or max(train_years) >= evaluation_year:
        raise ValueError("this selection experiment must train before and evaluate on 2024")
    tuning_payloads = [_verify_tuning_report(path) for path in tuning_report_paths]
    by_task = {str(payload["task"]): payload for payload in tuning_payloads}
    if set(by_task) != {"delay", "cancellation"} or len(tuning_payloads) != 2:
        raise ValueError("exactly one delay and one cancellation tuning report are required")
    forecast_payload = _verify_feature_manifest(forecast_manifest)
    run_dir.mkdir(parents=True)

    started = time.perf_counter()
    train = load_observed_recent_years(
        feature_dir,
        recent_dir,
        train_years,
        rows_per_year=rows_per_train_year,
        seed=seed,
    )
    evaluation = load_forecast_recent_year(
        feature_dir,
        recent_dir,
        forecast_dir,
        evaluation_year,
        limit=evaluation_limit,
        seed=seed,
    )
    task_results: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for task in ("delay", "cancellation"):
        task_name: TaskName = task
        task_started = time.perf_counter()
        payload = by_task[task]
        parameters = _frozen_parameters(payload)
        include_cross = payload["candidate"] == "network"
        train_task, train_labels = task_view(train, task_name)
        weights = temporal_sample_weights(
            train_task["Year"],
            prediction_year=evaluation_year,
            half_life_years=temporal_weight_half_life,
        )
        recent_model = fit_recent_catboost(
            train_task,
            train_labels,
            task=task_name,
            params=parameters,
            include_cross_direction=include_cross,
            sample_weight=weights,
        )
        recent_probabilities = recent_model.predict_proba(evaluation)[:, 1]
        transfer_model = fit_weather_transfer_catboost(
            train_task,
            train_labels,
            task=task_name,
            params=parameters,
            include_cross_direction=include_cross,
            sample_weight=weights,
        )
        transfer_probabilities = transfer_model.predict_proba(evaluation)[:, 1]
        for candidate, model in (("recent", recent_model), ("weather_transfer", transfer_model)):
            artifacts.append(
                {
                    "kind": "model",
                    "task": task,
                    "candidate": candidate,
                    **_atomic_joblib(
                        model,
                        run_dir / "models" / f"{task}_{candidate}_{evaluation_year}.joblib",
                    ),
                }
            )
        prediction_frame = evaluation.loc[
            :,
            [
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
            ],
        ].copy()
        prediction_frame["prob_recent"] = recent_probabilities.astype("float32")
        prediction_frame["prob_weather_transfer"] = transfer_probabilities.astype("float32")
        artifacts.append(
            {
                "kind": "predictions",
                "task": task,
                "year": evaluation_year,
                **_atomic_parquet(
                    prediction_frame,
                    run_dir / "predictions" / f"{task}_{evaluation_year}.parquet",
                ),
            }
        )
        task_results.append(
            {
                "task": task,
                "training_rows": len(train_task),
                "evaluation_rows": len(evaluation),
                "parameters": parameters,
                "recent_profile": recent_model_profile(include_cross),
                "weather_transfer_profile": weather_transfer_profile(),
                "recent_metrics": _score_task(evaluation, recent_probabilities, task_name),
                "weather_transfer_metrics": _score_task(
                    evaluation,
                    transfer_probabilities,
                    task_name,
                ),
                "paired_date_cluster_intervals": _paired_intervals(
                    evaluation,
                    recent_probabilities,
                    transfer_probabilities,
                    task=task_name,
                    repetitions=bootstrap_repetitions,
                    seed=seed,
                ),
                "elapsed_seconds": time.perf_counter() - task_started,
            }
        )

    feature_payload: dict[str, Any] = json.loads(feature_manifest.read_text(encoding="utf-8"))
    recent_payload: dict[str, Any] = json.loads(recent_manifest.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_2024_SELECTION_NOT_CONFIRMATORY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "candidate_name": "fixed24h_gfs_observed_to_forecast_transfer",
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
        "train_years": list(train_years),
        "evaluation_year": evaluation_year,
        "rows_per_train_year": rows_per_train_year,
        "evaluation_limit": evaluation_limit,
        "temporal_weight_half_life": temporal_weight_half_life,
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
                Path(__file__).with_name("forecast_features.py"),
                Path(__file__).with_name("recent_modeling.py"),
            )
        ),
        "elapsed_seconds": time.perf_counter() - started,
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
    parser.add_argument("--train-start", type=int, default=2011)
    parser.add_argument("--train-end", type=int, default=2023)
    parser.add_argument("--evaluation-year", type=int, default=2024)
    parser.add_argument("--rows-per-train-year", type=int, default=50_000)
    parser.add_argument("--evaluation-limit", type=int, default=300_000)
    parser.add_argument("--temporal-weight-half-life", type=float)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_forecast_benchmark(
        feature_dir=args.feature_dir,
        feature_manifest=args.feature_manifest,
        recent_dir=args.recent_dir,
        recent_manifest=args.recent_manifest,
        forecast_dir=args.forecast_dir,
        forecast_manifest=args.forecast_manifest,
        tuning_report_paths=tuple(args.tuning_reports),
        run_dir=args.run_dir,
        report_path=args.report,
        train_years=tuple(range(args.train_start, args.train_end + 1)),
        evaluation_year=args.evaluation_year,
        rows_per_train_year=args.rows_per_train_year,
        evaluation_limit=args.evaluation_limit,
        temporal_weight_half_life=args.temporal_weight_half_life,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "report": args.report.as_posix(),
                "task_results": result["task_results"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
