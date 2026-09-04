"""Rolling-origin evaluation of frozen prior-day operational models."""

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

import pandas as pd

from .contracts import RECENT_OPERATIONAL_FEATURES
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .modeling import TaskName, task_view, temporal_sample_weights
from .provenance import capture_provenance
from .recent_modeling import (
    fit_recent_catboost,
    load_recent_feature_year,
    load_recent_feature_years,
    recent_model_profile,
)
from .rolling import (
    _atomic_joblib,
    _atomic_parquet,
    _prediction_frame,
    _score_task,
    _training_years,
)
from .schedule_context import (
    load_context_recent_feature_year,
    load_context_recent_feature_years,
    schedule_context_profile,
)

RECENT_PREDICTION_FEATURES = tuple(
    name for name in RECENT_OPERATIONAL_FEATURES if "_rate_" in name
)


def _verify_tuning_report(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("result_sha256")
    body = {key: value for key, value in payload.items() if key != "result_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"tuning report self-hash failed: {path}")
    if int(payload.get("validation_year", -1)) != 2018:
        raise ValueError(f"rolling methods must be frozen from the 2018 fold: {path}")
    if payload.get("task") not in {"delay", "cancellation", "joint"}:
        raise ValueError(f"unsupported tuning task in {path}")
    return payload


def _frozen_parameters(payload: dict[str, Any]) -> dict[str, Any]:
    best_iteration = int(payload["best_iteration"])
    if best_iteration < 0:
        raise ValueError("best_iteration must be non-negative")
    parameters: dict[str, Any] = dict(payload["best_parameters"])
    parameters["iterations"] = best_iteration + 1
    parameters.setdefault("bootstrap_type", "Bayesian")
    return parameters


def run_recent_rolling(
    *,
    candidate_name: str,
    feature_dir: Path,
    feature_manifest: Path,
    recent_dir: Path,
    recent_manifest: Path,
    tuning_report_paths: tuple[Path, ...],
    run_dir: Path,
    report_path: Path,
    target_years: tuple[int, ...],
    rows_per_train_year: int,
    evaluation_limit: int | None,
    recent_window_years: int | None = None,
    temporal_weight_half_life: float | None = None,
    seed: int = 20260903,
) -> dict[str, Any]:
    """Evaluate task-specific methods frozen before any rolling outcome is read."""

    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse rolling run directory: {run_dir}")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite rolling report: {report_path}")
    if not target_years or min(target_years) < 2019 or max(target_years) > 2024:
        raise ValueError("recent rolling targets must be within 2019-2024")
    if tuple(sorted(set(target_years))) != target_years:
        raise ValueError("target years must be unique and chronological")

    tuning_payloads = [_verify_tuning_report(path) for path in tuning_report_paths]
    by_task = {str(payload["task"]): payload for payload in tuning_payloads}
    if not {"delay", "cancellation"}.issubset(by_task) or len(by_task) != len(tuning_payloads):
        raise ValueError("unique delay/cancellation reports and at most one joint report are required")
    if len(by_task) not in {2, 3}:
        raise ValueError("rolling evaluation accepts two binary reports and an optional joint report")
    task_order: tuple[TaskName, ...] = ("delay", "cancellation", "joint")
    tasks: tuple[TaskName, ...] = tuple(task for task in task_order if task in by_task)
    include_context = any(bool(payload.get("schedule_context")) for payload in tuning_payloads)
    run_dir.mkdir(parents=True)

    started = time.perf_counter()
    fold_results: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for target_year in target_years:
        years = _training_years(
            target_year,
            minimum_year=2011,
            recent_window_years=recent_window_years,
        )
        if include_context:
            train = load_context_recent_feature_years(
                feature_dir,
                recent_dir,
                years,
                rows_per_year=rows_per_train_year,
                seed=seed,
            )
            evaluation = load_context_recent_feature_year(
                feature_dir,
                recent_dir,
                target_year,
                limit=evaluation_limit,
                seed=seed,
            )
        else:
            train = load_recent_feature_years(
                feature_dir,
                recent_dir,
                years,
                rows_per_year=rows_per_train_year,
                seed=seed,
            )
            evaluation = load_recent_feature_year(
                feature_dir,
                recent_dir,
                target_year,
                limit=evaluation_limit,
                seed=seed,
            )
        for task in tasks:
            task_name: TaskName = task
            payload = by_task[task]
            task_started = time.perf_counter()
            train_task, train_labels = task_view(train, task_name)
            weights = temporal_sample_weights(
                train_task["Year"],
                prediction_year=target_year,
                half_life_years=temporal_weight_half_life,
            )
            parameters = _frozen_parameters(payload)
            cross_direction = payload["candidate"] == "network"
            task_context = bool(payload.get("schedule_context"))
            model = fit_recent_catboost(
                train_task,
                train_labels,
                task=task_name,
                params=parameters,
                include_cross_direction=cross_direction,
                include_schedule_context=task_context,
                sample_weight=weights,
            )
            probabilities = model.predict_proba(evaluation)
            model_record = _atomic_joblib(
                model,
                run_dir / "models" / f"{task}_{target_year}.joblib",
            )
            prediction_frame = pd.concat(
                [
                    _prediction_frame(evaluation, probabilities, task=task_name),
                    evaluation.loc[:, list(RECENT_PREDICTION_FEATURES)].astype("float32"),
                ],
                axis=1,
                copy=False,
            )
            prediction_record = _atomic_parquet(
                prediction_frame,
                run_dir / "predictions" / f"{task}_{target_year}.parquet",
            )
            artifacts.extend(
                [
                    {"kind": "model", "task": task, "year": target_year, **model_record},
                    {
                        "kind": "predictions",
                        "task": task,
                        "year": target_year,
                        **prediction_record,
                    },
                ]
            )
            fold_results.append(
                {
                    "target_year": target_year,
                    "task": task,
                    "training_years": list(years),
                    "loaded_training_rows": len(train),
                    "task_training_rows": len(train_task),
                    "evaluation_rows": len(evaluation),
                    "tuning_report": next(
                        path.as_posix()
                        for path, candidate in zip(
                            tuning_report_paths,
                            tuning_payloads,
                            strict=True,
                        )
                        if candidate["task"] == task
                    ),
                    "parameters": parameters,
                    "feature_profile": recent_model_profile(
                        cross_direction,
                        task_context,
                    ),
                    "metrics": _score_task(evaluation, probabilities, task_name),
                    "elapsed_seconds": time.perf_counter() - task_started,
                }
            )
            del model, probabilities, train_task, train_labels, weights
            gc.collect()
        del train, evaluation
        gc.collect()

    feature_payload: dict[str, Any] = json.loads(feature_manifest.read_text(encoding="utf-8"))
    recent_payload: dict[str, Any] = json.loads(recent_manifest.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_DEVELOPMENT_AND_SELECTION_NOT_CONFIRMATORY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "candidate_name": candidate_name,
        "family": "catboost_prior_day_operational",
        "feature_variant": feature_payload["feature_variant"],
        "feature_manifest": feature_manifest.as_posix(),
        "feature_manifest_sha256": sha256_file(feature_manifest),
        "recent_manifest": recent_manifest.as_posix(),
        "recent_manifest_sha256": sha256_file(recent_manifest),
        "recent_manifest_self_hash": recent_payload["manifest_sha256"],
        "tuning_reports": [
            {"path": path.as_posix(), "sha256": sha256_file(path)}
            for path in tuning_report_paths
        ],
        "target_years": list(target_years),
        "tasks": list(tasks),
        "rows_per_train_year": rows_per_train_year,
        "evaluation_limit": evaluation_limit,
        "recent_window_years": recent_window_years,
        "temporal_weight_half_life": temporal_weight_half_life,
        "schedule_context": schedule_context_profile() if include_context else None,
        "seed": seed,
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
                Path(__file__).with_name("recent_modeling.py"),
                Path(__file__).with_name("schedule_context.py"),
                Path(__file__).with_name("modeling.py"),
            )
        ),
        "fold_results": fold_results,
        "artifacts": artifacts,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(report_path, report)
    write_canonical_json(run_dir / "run_manifest.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--recent-dir", type=Path, required=True)
    parser.add_argument("--recent-manifest", type=Path, required=True)
    parser.add_argument("--tuning-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-years", type=int, nargs="+", default=list(range(2019, 2025)))
    parser.add_argument("--rows-per-train-year", type=int, default=75_000)
    parser.add_argument("--evaluation-limit", type=int, default=200_000)
    parser.add_argument("--recent-window-years", type=int)
    parser.add_argument("--temporal-weight-half-life", type=float)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_recent_rolling(
        candidate_name=args.candidate_name,
        feature_dir=args.feature_dir,
        feature_manifest=args.feature_manifest,
        recent_dir=args.recent_dir,
        recent_manifest=args.recent_manifest,
        tuning_report_paths=tuple(args.tuning_reports),
        run_dir=args.run_dir,
        report_path=args.report,
        target_years=tuple(args.target_years),
        rows_per_train_year=args.rows_per_train_year,
        evaluation_limit=args.evaluation_limit,
        recent_window_years=args.recent_window_years,
        temporal_weight_half_life=args.temporal_weight_half_life,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "candidate": report["candidate_name"],
                "folds": len(report["fold_results"]),
                "elapsed_seconds": report["elapsed_seconds"],
                "report": args.report.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
