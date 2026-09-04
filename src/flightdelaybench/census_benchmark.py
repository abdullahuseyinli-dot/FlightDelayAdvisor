"""Screen rich-schedule and flight-history ablations on the 2023 census fold."""

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

from .benchmark import _metric_payload
from .census_modeling import (
    census_model_profile,
    fit_census_catboost,
    load_census_year,
    load_census_years,
)
from .census_normalization import _verify_json_self_hash
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .metrics import binary_metrics
from .modeling import TaskName, task_view
from .provenance import capture_provenance

CENSUS_CANDIDATES: dict[str, dict[str, bool]] = {
    "hmop_direct": {
        "include_cross_direction": False,
        "include_rich_schedule": False,
        "include_flight_history": False,
        "include_schedule_context": False,
        "include_graph_pressure": False,
    },
    "rich_schedule_direct": {
        "include_cross_direction": False,
        "include_rich_schedule": True,
        "include_flight_history": False,
        "include_schedule_context": False,
        "include_graph_pressure": False,
    },
    "flight_history_direct": {
        "include_cross_direction": False,
        "include_rich_schedule": True,
        "include_flight_history": True,
        "include_schedule_context": False,
        "include_graph_pressure": False,
    },
    "full_census_direct": {
        "include_cross_direction": False,
        "include_rich_schedule": True,
        "include_flight_history": True,
        "include_schedule_context": True,
        "include_graph_pressure": False,
    },
    "full_census_network": {
        "include_cross_direction": True,
        "include_rich_schedule": True,
        "include_flight_history": True,
        "include_schedule_context": True,
        "include_graph_pressure": False,
    },
    "graph_census_direct": {
        "include_cross_direction": False,
        "include_rich_schedule": True,
        "include_flight_history": True,
        "include_schedule_context": True,
        "include_graph_pressure": True,
    },
    "graph_census_network": {
        "include_cross_direction": True,
        "include_rich_schedule": True,
        "include_flight_history": True,
        "include_schedule_context": True,
        "include_graph_pressure": True,
    },
}

SCREEN_PARAMETERS: dict[str, Any] = {
    "iterations": 1_000,
    "learning_rate": 0.045,
    "depth": 8,
    "l2_leaf_reg": 8.0,
    "random_strength": 0.25,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.4,
    "border_count": 128,
}


def _recent_baseline(frame: Any, task: TaskName) -> np.ndarray:
    outcome = "delay" if task == "delay" else "cancel"
    columns = [
        f"recent_{view}_{outcome}_rate_{window}d"
        for view in ("route", "airline", "origin_outbound", "dest_inbound")
        for window in (7, 28, 90)
    ]
    probabilities = np.asarray(frame.loc[:, columns], dtype=np.float64)
    clipped = np.clip(probabilities, 1e-5, 1.0 - 1e-5)
    logits = np.log(clipped / (1.0 - clipped))
    return np.asarray(1.0 / (1.0 + np.exp(-logits.mean(axis=1))), dtype=np.float64)


def run_census_benchmark(
    *,
    census_dir: Path,
    census_manifest: Path,
    recent_dir: Path,
    recent_manifest: Path,
    flight_recent_dir: Path,
    flight_recent_manifest: Path,
    graph_dir: Path | None,
    graph_manifest: Path | None,
    output_path: Path,
    train_years: tuple[int, ...] = (2019, 2020, 2021, 2022),
    validation_year: int = 2023,
    rows_per_train_year: int = 100_000,
    validation_limit: int = 300_000,
    tasks: tuple[TaskName, ...] = ("delay", "cancellation"),
    candidates: tuple[str, ...] = tuple(CENSUS_CANDIDATES),
    seed: int = 20260903,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite census benchmark: {output_path}")
    if max(train_years) >= validation_year or validation_year != 2023:
        raise ValueError("census development must train before the 2023 validation fold")
    unknown = sorted(set(candidates) - set(CENSUS_CANDIDATES))
    if unknown:
        raise ValueError(f"unknown census candidates: {unknown}")
    if any(task not in {"delay", "cancellation"} for task in tasks):
        raise ValueError("the census screening stage currently accepts binary tasks only")
    needs_graph = any(CENSUS_CANDIDATES[name]["include_graph_pressure"] for name in candidates)
    if needs_graph and (graph_dir is None or graph_manifest is None):
        raise ValueError("graph candidates require both graph directory and manifest")
    input_payloads = {
        "census": _verify_json_self_hash(census_manifest),
        "recent": _verify_json_self_hash(recent_manifest),
        "flight_recent": _verify_json_self_hash(flight_recent_manifest),
    }
    if graph_manifest is not None:
        input_payloads["graph"] = _verify_json_self_hash(graph_manifest)

    started = time.perf_counter()
    train = load_census_years(
        census_dir,
        recent_dir,
        flight_recent_dir,
        train_years,
        rows_per_year=rows_per_train_year,
        graph_dir=graph_dir,
        seed=seed,
    )
    validation = load_census_year(
        census_dir,
        recent_dir,
        flight_recent_dir,
        validation_year,
        limit=validation_limit,
        graph_dir=graph_dir,
        seed=seed,
    )
    results: list[dict[str, Any]] = []
    for task in tasks:
        train_task, train_labels = task_view(train, task)
        valid_task, valid_labels = task_view(validation, task)
        baseline_probabilities = _recent_baseline(valid_task, task)
        results.append(
            {
                "task": task,
                "candidate": "hierarchical_recent_logit_mean",
                "family": "closed_left_prior_baseline",
                "metrics": binary_metrics(valid_labels, baseline_probabilities).as_dict(),
            }
        )
        for candidate in candidates:
            flags = CENSUS_CANDIDATES[candidate]
            candidate_started = time.perf_counter()
            model = fit_census_catboost(
                train_task,
                train_labels,
                task=task,
                params=SCREEN_PARAMETERS,
                validation_frame=valid_task,
                validation_labels=valid_labels,
                **flags,
            )
            probabilities = np.asarray(model.predict_proba(valid_task), dtype=np.float64)
            results.append(
                {
                    "task": task,
                    "candidate": candidate,
                    "family": "catboost_census",
                    "parameters": SCREEN_PARAMETERS,
                    "best_iteration": int(model.estimator.get_best_iteration()),
                    "feature_profile": census_model_profile(**flags),
                    "metrics": _metric_payload(valid_labels, probabilities, task),
                    "fit_seconds": time.perf_counter() - candidate_started,
                }
            )
            del model, probabilities
            gc.collect()

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_2023_CENSUS_DEVELOPMENT_SCREEN_NOT_CONFIRMATORY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "census_manifest": census_manifest.as_posix(),
        "census_manifest_sha256": sha256_file(census_manifest),
        "census_manifest_self_hash": input_payloads["census"]["manifest_sha256"],
        "recent_manifest": recent_manifest.as_posix(),
        "recent_manifest_sha256": sha256_file(recent_manifest),
        "recent_manifest_self_hash": input_payloads["recent"]["manifest_sha256"],
        "flight_recent_manifest": flight_recent_manifest.as_posix(),
        "flight_recent_manifest_sha256": sha256_file(flight_recent_manifest),
        "flight_recent_manifest_self_hash": input_payloads["flight_recent"][
            "manifest_sha256"
        ],
        "graph_manifest": graph_manifest.as_posix() if graph_manifest else None,
        "graph_manifest_sha256": sha256_file(graph_manifest) if graph_manifest else None,
        "graph_manifest_self_hash": (
            input_payloads["graph"]["manifest_sha256"] if graph_manifest else None
        ),
        "train_years": list(train_years),
        "validation_year": validation_year,
        "rows_per_train_year": rows_per_train_year,
        "validation_limit": validation_limit,
        "loaded_train_rows": len(train),
        "loaded_validation_rows": len(validation),
        "tasks": list(tasks),
        "candidates_requested": list(candidates),
        "seed": seed,
        "candidates": results,
        "versions": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "catboost": version("catboost"),
        },
        "provenance": capture_provenance(
            (
                Path(__file__),
                Path(__file__).with_name("census_modeling.py"),
                Path(__file__).with_name("census_recent.py"),
            )
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "selection_boundary": "No outcome after 2023 was loaded by this development screen.",
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-dir", type=Path, required=True)
    parser.add_argument("--census-manifest", type=Path, required=True)
    parser.add_argument("--recent-dir", type=Path, required=True)
    parser.add_argument("--recent-manifest", type=Path, required=True)
    parser.add_argument("--flight-recent-dir", type=Path, required=True)
    parser.add_argument("--flight-recent-manifest", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path)
    parser.add_argument("--graph-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-train-year", type=int, default=100_000)
    parser.add_argument("--validation-limit", type=int, default=300_000)
    parser.add_argument("--tasks", nargs="+", choices=["delay", "cancellation"], default=["delay", "cancellation"])
    parser.add_argument("--candidates", nargs="+", choices=list(CENSUS_CANDIDATES), default=list(CENSUS_CANDIDATES))
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_census_benchmark(
        census_dir=args.census_dir,
        census_manifest=args.census_manifest,
        recent_dir=args.recent_dir,
        recent_manifest=args.recent_manifest,
        flight_recent_dir=args.flight_recent_dir,
        flight_recent_manifest=args.flight_recent_manifest,
        graph_dir=args.graph_dir,
        graph_manifest=args.graph_manifest,
        output_path=args.output,
        rows_per_train_year=args.rows_per_train_year,
        validation_limit=args.validation_limit,
        tasks=tuple(args.tasks),
        candidates=tuple(args.candidates),
        seed=args.seed,
    )
    print(json.dumps({"output": args.output.as_posix(), "candidates": result["candidates"]}, indent=2))


if __name__ == "__main__":
    main()
