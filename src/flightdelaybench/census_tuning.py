"""Tune a selected official-census candidate on the 2023 development fold."""

from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import log_loss

from .census_benchmark import CENSUS_CANDIDATES
from .census_modeling import (
    census_model_profile,
    fit_census_catboost,
    load_census_year,
    load_census_years,
)
from .census_normalization import _verify_json_self_hash
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .modeling import TaskName, task_view
from .provenance import capture_provenance


def tune_census_catboost(
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
    task: TaskName,
    candidate: str,
    trials: int = 12,
    rows_per_train_year: int = 100_000,
    validation_limit: int = 300_000,
    seed: int = 20260903,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite census tuning evidence: {output_path}")
    if task not in {"delay", "cancellation"}:
        raise ValueError("census tuning currently accepts binary tasks only")
    if candidate not in CENSUS_CANDIDATES:
        raise ValueError(f"unknown census candidate: {candidate}")
    if CENSUS_CANDIDATES[candidate]["include_graph_pressure"] and (
        graph_dir is None or graph_manifest is None
    ):
        raise ValueError("graph candidate tuning requires graph directory and manifest")
    if trials < 2:
        raise ValueError("at least two tuning trials are required")
    input_payloads = {
        "census": _verify_json_self_hash(census_manifest),
        "recent": _verify_json_self_hash(recent_manifest),
        "flight_recent": _verify_json_self_hash(flight_recent_manifest),
    }
    if graph_manifest is not None:
        input_payloads["graph"] = _verify_json_self_hash(graph_manifest)
    train_years = (2019, 2020, 2021, 2022)
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
        2023,
        limit=validation_limit,
        graph_dir=graph_dir,
        seed=seed,
    )
    train_task, train_labels = task_view(train, task)
    valid_task, valid_labels = task_view(validation, task)
    flags = CENSUS_CANDIDATES[candidate]
    started = time.perf_counter()

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        parameters: dict[str, Any] = {
            "iterations": 1_400,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.09, log=True),
            "depth": trial.suggest_int("depth", 7, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.02, 3.0, log=True),
            "bootstrap_type": "Bayesian",
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.5),
            "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        }
        trial_started = time.perf_counter()
        model = fit_census_catboost(
            train_task,
            train_labels,
            task=task,
            params=parameters,
            validation_frame=valid_task,
            validation_labels=valid_labels,
            **flags,
        )
        probabilities = np.asarray(model.predict_proba(valid_task), dtype=np.float64)[:, 1]
        score = float(
            log_loss(
                valid_labels,
                np.clip(probabilities, 1e-6, 1.0 - 1e-6),
                labels=[0, 1],
            )
        )
        trial.set_user_attr("best_iteration", int(model.estimator.get_best_iteration()))
        trial.set_user_attr("elapsed_seconds", time.perf_counter() - trial_started)
        del model, probabilities
        gc.collect()
        return score

    study.optimize(objective, n_trials=trials, n_jobs=1, gc_after_trial=True)
    best = study.best_trial
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_2023_CENSUS_TUNING_NOT_CONFIRMATORY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "task": task,
        "candidate": candidate,
        "feature_profile": census_model_profile(**flags),
        "training_years": list(train_years),
        "validation_year": 2023,
        "rows_per_train_year": rows_per_train_year,
        "validation_limit": validation_limit,
        "loaded_training_rows": len(train_task),
        "loaded_validation_rows": len(valid_task),
        "seed": seed,
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
        "objective": "binary_log_loss",
        "selection_boundary": "No outcome after 2023 was loaded or used by this search.",
        "best_value": best.value,
        "best_parameters": best.params,
        "best_iteration": best.user_attrs["best_iteration"],
        "trials": [
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "parameters": trial.params,
                "best_iteration": trial.user_attrs.get("best_iteration"),
                "elapsed_seconds": trial.user_attrs.get("elapsed_seconds"),
            }
            for trial in study.trials
        ],
        "provenance": capture_provenance(
            (
                Path(__file__),
                Path(__file__).with_name("census_modeling.py"),
                Path(__file__).with_name("census_benchmark.py"),
            )
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["result_sha256"] = canonical_json_sha256(report)
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
    parser.add_argument("--task", choices=["delay", "cancellation"], required=True)
    parser.add_argument("--candidate", choices=list(CENSUS_CANDIDATES), required=True)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--rows-per-train-year", type=int, default=100_000)
    parser.add_argument("--validation-limit", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = tune_census_catboost(
        census_dir=args.census_dir,
        census_manifest=args.census_manifest,
        recent_dir=args.recent_dir,
        recent_manifest=args.recent_manifest,
        flight_recent_dir=args.flight_recent_dir,
        flight_recent_manifest=args.flight_recent_manifest,
        graph_dir=args.graph_dir,
        graph_manifest=args.graph_manifest,
        output_path=args.output,
        task=args.task,
        candidate=args.candidate,
        trials=args.trials,
        rows_per_train_year=args.rows_per_train_year,
        validation_limit=args.validation_limit,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "task": report["task"],
                "candidate": report["candidate"],
                "best_value": report["best_value"],
                "best_iteration": report["best_iteration"],
                "output": args.output.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
