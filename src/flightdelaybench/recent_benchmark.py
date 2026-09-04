"""Screen prior-day multi-timescale CatBoost candidates on one temporal fold."""

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

from .benchmark import _binary_baselines, _metric_payload
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .modeling import TaskName, task_view
from .recent_modeling import (
    fit_recent_catboost,
    load_recent_feature_year,
    load_recent_feature_years,
    recent_model_profile,
)
from .schedule_context import (
    load_context_recent_feature_year,
    load_context_recent_feature_years,
    schedule_context_profile,
)

RECENT_CATBOOST_PARAMS: dict[TaskName, dict[str, Any]] = {
    "delay": {
        "iterations": 1_200,
        "learning_rate": 0.04,
        "depth": 8,
        "l2_leaf_reg": 6.0,
    },
    "cancellation": {
        "iterations": 1_000,
        "learning_rate": 0.04,
        "depth": 9,
        "l2_leaf_reg": 5.0,
    },
    "joint": {
        "iterations": 1_200,
        "learning_rate": 0.04,
        "depth": 8,
        "l2_leaf_reg": 6.0,
    },
}


def run_recent_benchmark(
    *,
    feature_dir: Path,
    feature_manifest: Path,
    recent_dir: Path,
    recent_manifest: Path,
    output_path: Path,
    train_years: tuple[int, ...],
    validation_year: int,
    rows_per_train_year: int,
    validation_limit: int | None,
    tasks: tuple[TaskName, ...],
    candidates: tuple[str, ...],
    include_schedule_context: bool = False,
    seed: int = 20260903,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite recent benchmark: {output_path}")
    if max(train_years) >= validation_year:
        raise ValueError("all recent benchmark training years must precede validation")
    unknown = sorted(set(candidates) - {"direct", "network"})
    if unknown:
        raise ValueError(f"unknown recent candidates: {unknown}")

    started = time.perf_counter()
    if include_schedule_context:
        train = load_context_recent_feature_years(
            feature_dir,
            recent_dir,
            train_years,
            rows_per_year=rows_per_train_year,
            seed=seed,
        )
        validation = load_context_recent_feature_year(
            feature_dir,
            recent_dir,
            validation_year,
            limit=validation_limit,
            seed=seed,
        )
    else:
        train = load_recent_feature_years(
            feature_dir,
            recent_dir,
            train_years,
            rows_per_year=rows_per_train_year,
            seed=seed,
        )
        validation = load_recent_feature_year(
            feature_dir,
            recent_dir,
            validation_year,
            limit=validation_limit,
            seed=seed,
        )
    results: list[dict[str, Any]] = []
    for task in tasks:
        train_task, train_labels = task_view(train, task)
        valid_task, valid_labels = task_view(validation, task)
        if task != "joint":
            for name, probabilities in _binary_baselines(valid_task, task).items():
                results.append(
                    {
                        "task": task,
                        "candidate": name,
                        "family": "strict_earlier_year_baseline",
                        "fit_seconds": 0.0,
                        "metrics": _metric_payload(valid_labels, probabilities, task),
                    }
                )
        for candidate in candidates:
            include_cross = candidate == "network"
            fit_started = time.perf_counter()
            model = fit_recent_catboost(
                train_task,
                train_labels,
                task=task,
                params=RECENT_CATBOOST_PARAMS[task],
                include_cross_direction=include_cross,
                include_schedule_context=include_schedule_context,
                validation_frame=valid_task,
                validation_labels=valid_labels,
            )
            probabilities = model.predict_proba(valid_task)
            results.append(
                {
                    "task": task,
                    "candidate": (
                        f"catboost_recent_context_{candidate}"
                        if include_schedule_context
                        else f"catboost_recent_{candidate}"
                    ),
                    "family": "catboost",
                    "fit_seconds": time.perf_counter() - fit_started,
                    "parameters": RECENT_CATBOOST_PARAMS[task],
                    "best_iteration": int(model.estimator.get_best_iteration()),
                    "feature_profile": recent_model_profile(
                        include_cross,
                        include_schedule_context,
                    ),
                    "metrics": _metric_payload(valid_labels, probabilities, task),
                }
            )

    feature_payload: dict[str, Any] = json.loads(feature_manifest.read_text(encoding="utf-8"))
    recent_payload: dict[str, Any] = json.loads(recent_manifest.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_24H_OPERATIONAL_PROXY_SCREENING_NOT_CONFIRMATORY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "feature_manifest": feature_manifest.as_posix(),
        "feature_manifest_sha256": sha256_file(feature_manifest),
        "feature_variant": feature_payload["feature_variant"],
        "recent_manifest": recent_manifest.as_posix(),
        "recent_manifest_sha256": sha256_file(recent_manifest),
        "recent_manifest_self_hash": recent_payload["manifest_sha256"],
        "train_years": list(train_years),
        "validation_year": validation_year,
        "rows_per_train_year": rows_per_train_year,
        "validation_limit": validation_limit,
        "loaded_train_rows": len(train),
        "loaded_validation_rows": len(validation),
        "seed": seed,
        "tasks": list(tasks),
        "candidates_requested": list(candidates),
        "schedule_context": (
            schedule_context_profile() if include_schedule_context else None
        ),
        "versions": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
            "catboost": version("catboost"),
        },
        "candidates": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--recent-dir", type=Path, required=True)
    parser.add_argument("--recent-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-start", type=int, default=2011)
    parser.add_argument("--train-end", type=int, default=2017)
    parser.add_argument("--validation-year", type=int, default=2018)
    parser.add_argument("--rows-per-train-year", type=int, default=100_000)
    parser.add_argument("--validation-limit", type=int, default=300_000)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["delay", "cancellation", "joint"],
        default=["delay", "cancellation"],
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=["direct", "network"],
        default=["direct", "network"],
    )
    parser.add_argument(
        "--include-schedule-context",
        action="store_true",
        help="Add target-day schedule-census proxy features before sampling.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_recent_benchmark(
        feature_dir=args.feature_dir,
        feature_manifest=args.feature_manifest,
        recent_dir=args.recent_dir,
        recent_manifest=args.recent_manifest,
        output_path=args.output,
        train_years=tuple(range(args.train_start, args.train_end + 1)),
        validation_year=args.validation_year,
        rows_per_train_year=args.rows_per_train_year,
        validation_limit=args.validation_limit,
        tasks=tuple(args.tasks),
        candidates=tuple(args.candidates),
        include_schedule_context=args.include_schedule_context,
    )
    print(
        json.dumps(
            {"output": args.output.as_posix(), "candidates": report["candidates"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
