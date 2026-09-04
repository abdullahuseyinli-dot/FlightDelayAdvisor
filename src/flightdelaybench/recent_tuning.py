"""Tune the prior-day operational CatBoost family on the sealed 2018 dev fold."""

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

from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .modeling import TaskName, task_view
from .provenance import capture_provenance
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


def tune_recent_catboost(
    *,
    feature_dir: Path,
    feature_manifest: Path,
    recent_dir: Path,
    recent_manifest: Path,
    output_path: Path,
    task: TaskName,
    include_cross_direction: bool,
    include_schedule_context: bool = False,
    trials: int = 12,
    rows_per_train_year: int = 100_000,
    validation_limit: int = 300_000,
    seed: int = 20260903,
) -> dict[str, Any]:
    """Search hyperparameters without reading any outcome after 2018."""

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite tuning evidence: {output_path}")
    if trials < 2:
        raise ValueError("at least two tuning trials are required")

    if include_schedule_context:
        train = load_context_recent_feature_years(
            feature_dir,
            recent_dir,
            tuple(range(2011, 2018)),
            rows_per_year=rows_per_train_year,
            seed=seed,
        )
        validation = load_context_recent_feature_year(
            feature_dir,
            recent_dir,
            2018,
            limit=validation_limit,
            seed=seed,
        )
    else:
        train = load_recent_feature_years(
            feature_dir,
            recent_dir,
            tuple(range(2011, 2018)),
            rows_per_year=rows_per_train_year,
            seed=seed,
        )
        validation = load_recent_feature_year(
            feature_dir,
            recent_dir,
            2018,
            limit=validation_limit,
            seed=seed,
        )
    train_task, train_labels = task_view(train, task)
    valid_task, valid_labels = task_view(validation, task)
    started = time.perf_counter()

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        parameters: dict[str, Any] = {
            "iterations": 1_200,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.09, log=True),
            "depth": trial.suggest_int("depth", 7, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 25.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.02, 3.0, log=True),
            "bootstrap_type": "Bayesian",
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.5),
            "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        }
        trial_started = time.perf_counter()
        model = fit_recent_catboost(
            train_task,
            train_labels,
            task=task,
            params=parameters,
            include_cross_direction=include_cross_direction,
            include_schedule_context=include_schedule_context,
            validation_frame=valid_task,
            validation_labels=valid_labels,
        )
        probabilities = np.asarray(model.predict_proba(valid_task), dtype=np.float64)
        if task == "joint":
            clipped = np.clip(probabilities, 1e-6, 1.0)
            clipped /= clipped.sum(axis=1, keepdims=True)
            score = float(log_loss(valid_labels, clipped, labels=[0, 1, 2]))
        else:
            binary = np.clip(probabilities[:, 1], 1e-6, 1.0 - 1e-6)
            score = float(log_loss(valid_labels, binary, labels=[0, 1]))
        trial.set_user_attr("best_iteration", int(model.estimator.get_best_iteration()))
        trial.set_user_attr("elapsed_seconds", time.perf_counter() - trial_started)
        del model, probabilities
        gc.collect()
        return score

    study.optimize(objective, n_trials=trials, n_jobs=1, gc_after_trial=True)
    trial_records = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "parameters": trial.params,
            "best_iteration": trial.user_attrs.get("best_iteration"),
            "elapsed_seconds": trial.user_attrs.get("elapsed_seconds"),
        }
        for trial in study.trials
    ]
    best = study.best_trial
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_PRE_ROLLING_TUNING_2018_ONLY_NOT_CONFIRMATORY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "task": task,
        "candidate": "network" if include_cross_direction else "direct",
        "feature_profile": recent_model_profile(
            include_cross_direction,
            include_schedule_context,
        ),
        "schedule_context": schedule_context_profile() if include_schedule_context else None,
        "training_years": list(range(2011, 2018)),
        "validation_year": 2018,
        "rows_per_train_year": rows_per_train_year,
        "validation_limit": validation_limit,
        "loaded_training_rows": len(train_task),
        "loaded_validation_rows": len(valid_task),
        "seed": seed,
        "feature_manifest": feature_manifest.as_posix(),
        "feature_manifest_sha256": sha256_file(feature_manifest),
        "recent_manifest": recent_manifest.as_posix(),
        "recent_manifest_sha256": sha256_file(recent_manifest),
        "objective": "multiclass_log_loss" if task == "joint" else "binary_log_loss",
        "selection_boundary": "No outcome after 2018 was loaded or used by this search.",
        "best_value": best.value,
        "best_parameters": best.params,
        "best_iteration": best.user_attrs["best_iteration"],
        "trials": trial_records,
        "provenance": capture_provenance(
            (Path(__file__), Path(__file__).with_name("recent_modeling.py"))
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    result["result_sha256"] = canonical_json_sha256(result)
    write_canonical_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--recent-dir", type=Path, required=True)
    parser.add_argument("--recent-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=["delay", "cancellation", "joint"], required=True)
    parser.add_argument("--candidate", choices=["direct", "network"], required=True)
    parser.add_argument("--include-schedule-context", action="store_true")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--rows-per-train-year", type=int, default=100_000)
    parser.add_argument("--validation-limit", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = tune_recent_catboost(
        feature_dir=args.feature_dir,
        feature_manifest=args.feature_manifest,
        recent_dir=args.recent_dir,
        recent_manifest=args.recent_manifest,
        output_path=args.output,
        task=args.task,
        include_cross_direction=args.candidate == "network",
        include_schedule_context=args.include_schedule_context,
        trials=args.trials,
        rows_per_train_year=args.rows_per_train_year,
        validation_limit=args.validation_limit,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "task": result["task"],
                "candidate": result["candidate"],
                "best_value": result["best_value"],
                "best_parameters": result["best_parameters"],
                "best_iteration": result["best_iteration"],
                "output": args.output.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
