"""Run a create-only temporal benchmark fold for candidate screening."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import log_loss

from .frontier import (
    fit_chimeraboost,
    fit_tabm,
    frontier_model_features,
)
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .metrics import binary_metrics, multiclass_brier
from .modeling import (
    TaskName,
    fit_catboost,
    fit_lightgbm,
    load_feature_year,
    load_feature_years,
    task_view,
    temporal_sample_weights,
)
from .oracle import (
    ORACLE_WEATHER_FEATURES,
    fit_oracle_catboost,
    oracle_feature_profile,
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-values)), dtype=np.float64)


def _hierarchical_prior(frame: Any, outcome: str) -> np.ndarray:
    columns = [
        f"prior_{level}_{outcome}_rate" for level in ("route", "airline", "origin", "dest", "slot")
    ]
    probabilities = np.asarray(frame.loc[:, columns], dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-5, 1.0 - 1e-5)
    logits = np.log(probabilities / (1.0 - probabilities))
    return _sigmoid(logits.mean(axis=1))


def _binary_baselines(frame: Any, task: TaskName) -> dict[str, np.ndarray]:
    outcome = "delay" if task == "delay" else "cancel"
    return {
        "prior_global": np.asarray(frame[f"prior_global_{outcome}_rate"], dtype=np.float64),
        "prior_route": np.asarray(frame[f"prior_route_{outcome}_rate"], dtype=np.float64),
        "prior_slot": np.asarray(frame[f"prior_slot_{outcome}_rate"], dtype=np.float64),
        "prior_hierarchical_logit_mean": _hierarchical_prior(frame, outcome),
    }


def _metric_payload(
    labels: np.ndarray, probabilities: np.ndarray, task: TaskName
) -> dict[str, Any]:
    if task == "joint":
        return {
            "n": len(labels),
            "log_loss": float(log_loss(labels, probabilities, labels=[0, 1, 2])),
            "multiclass_brier": multiclass_brier(labels, probabilities),
            "class_prevalence": {str(value): float(np.mean(labels == value)) for value in range(3)},
        }
    return binary_metrics(
        labels, probabilities[:, 1] if probabilities.ndim == 2 else probabilities
    ).as_dict()


def run_benchmark(
    *,
    feature_dir: Path,
    feature_manifest: Path,
    output_path: Path,
    train_years: tuple[int, ...],
    validation_year: int,
    rows_per_train_year: int,
    validation_limit: int | None,
    tasks: tuple[TaskName, ...],
    families: tuple[str, ...],
    seed: int = 20260903,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite benchmark evidence: {output_path}")
    if max(train_years) >= validation_year:
        raise ValueError("all training years must precede the validation year")

    started = time.perf_counter()
    extra_columns = ORACLE_WEATHER_FEATURES if "catboost_oracle" in families else ()
    train = load_feature_years(
        feature_dir,
        train_years,
        rows_per_year=rows_per_train_year,
        seed=seed,
        extra_columns=extra_columns,
    )
    validation = load_feature_year(
        feature_dir,
        validation_year,
        limit=validation_limit,
        seed=seed,
        extra_columns=extra_columns,
    )
    candidates: list[dict[str, Any]] = []
    model_params: dict[str, dict[str, Any]] = {
        "lightgbm": {
            "n_estimators": 900,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 250,
            "max_bin": 127,
            "reg_alpha": 0.2,
            "reg_lambda": 2.0,
        },
        "catboost": {
            "iterations": 900,
            "learning_rate": 0.06,
            "depth": 8,
            "l2_leaf_reg": 6.0,
        },
        "catboost_oracle": {
            "iterations": 900,
            "learning_rate": 0.06,
            "depth": 8,
            "l2_leaf_reg": 6.0,
        },
        "chimeraboost": {
            "quality": 3,
        },
        "chimeraboost_ensemble": {
            "quality": 4,
        },
        "tabm": {
            "max_epochs": 25,
            "patience": 6,
            "batch_size": 1_024,
            "eval_batch_size": 4_096,
            "embedding": "piecewise",
            "n_bins": 48,
            "d_embedding": 8,
            "arch_type": "tabm",
            "k": 16,
            "n_blocks": 2,
            "d_block": 256,
            "verbose": True,
        },
        "tabm_periodic": {
            "max_epochs": 25,
            "patience": 6,
            "batch_size": 1_024,
            "eval_batch_size": 4_096,
            "embedding": "periodic",
            "d_embedding": 8,
            "n_frequencies": 24,
            "arch_type": "tabm",
            "k": 16,
            "n_blocks": 2,
            "d_block": 256,
            "verbose": True,
        },
        "tabm_route": {
            "max_epochs": 25,
            "patience": 6,
            "batch_size": 512,
            "eval_batch_size": 2_048,
            "learning_rate": 1e-3,
            "embedding": "piecewise",
            "n_bins": 64,
            "d_embedding": 12,
            "arch_type": "tabm",
            "k": 16,
            "n_blocks": 2,
            "d_block": 384,
            "include_route": True,
            "verbose": True,
        },
    }

    for task in tasks:
        train_task, train_labels = task_view(train, task)
        valid_task, valid_labels = task_view(validation, task)
        if task != "joint":
            for name, probabilities in _binary_baselines(valid_task, task).items():
                candidates.append(
                    {
                        "task": task,
                        "candidate": name,
                        "family": "strict_prior_baseline",
                        "fit_seconds": 0.0,
                        "metrics": _metric_payload(valid_labels, probabilities, task),
                    }
                )

        for family in (name for name in families if name != "lightgbm_temporal_4y"):
            if family not in model_params:
                raise ValueError(f"unsupported model family: {family}")
            family_started = time.perf_counter()
            if family == "lightgbm":
                lightgbm_model = fit_lightgbm(
                    train_task,
                    train_labels,
                    task=task,
                    params=model_params[family],
                    validation_frame=valid_task,
                    validation_labels=valid_labels,
                )
                detail: dict[str, Any] = {"best_iteration": lightgbm_model.best_iteration}
                probabilities = lightgbm_model.predict_proba(valid_task)
            elif family == "catboost":
                catboost_model = fit_catboost(
                    train_task,
                    train_labels,
                    task=task,
                    params=model_params[family],
                    validation_frame=valid_task,
                    validation_labels=valid_labels,
                )
                detail = {"best_iteration": int(catboost_model.estimator.get_best_iteration())}
                probabilities = catboost_model.predict_proba(valid_task)
            elif family == "catboost_oracle":
                oracle_model = fit_oracle_catboost(
                    train_task,
                    train_labels,
                    task=task,
                    params=model_params[family],
                    validation_frame=valid_task,
                    validation_labels=valid_labels,
                )
                detail = {
                    "best_iteration": int(oracle_model.estimator.get_best_iteration()),
                    "feature_profile": oracle_feature_profile(),
                }
                probabilities = oracle_model.predict_proba(valid_task)
            elif family.startswith("chimeraboost"):
                chimera_model = fit_chimeraboost(
                    train_task,
                    train_labels,
                    task=task,
                    params=model_params[family],
                    validation_frame=valid_task,
                    validation_labels=valid_labels,
                )
                detail = {
                    "best_iteration": int(chimera_model.estimator.best_iteration_),
                    "temperature": float(chimera_model.estimator.temperature_),
                    "cross_features_selected": chimera_model.estimator.cross_features_selected_,
                }
                probabilities = chimera_model.predict_proba(valid_task)
            else:
                tabm_model = fit_tabm(
                    train_task,
                    train_labels,
                    task=task,
                    params=model_params[family],
                    validation_frame=valid_task,
                    validation_labels=valid_labels,
                )
                detail = {
                    "best_iteration": tabm_model.best_epoch,
                    "fit_device": tabm_model.fit_device,
                    "training_history": tabm_model.training_history,
                    "feature_profile": frontier_model_features(
                        tabm_model.preprocessor.categorical_features
                    ),
                }
                probabilities = tabm_model.predict_proba(valid_task)
            candidates.append(
                {
                    "task": task,
                    "candidate": family,
                    "family": family,
                    "fit_seconds": time.perf_counter() - family_started,
                    "parameters": model_params[family],
                    **detail,
                    "metrics": _metric_payload(valid_labels, probabilities, task),
                }
            )

        if "lightgbm_temporal_4y" in families:
            weights = temporal_sample_weights(
                train_task["Year"],
                prediction_year=validation_year,
                half_life_years=4.0,
            )
            family_started = time.perf_counter()
            model = fit_lightgbm(
                train_task,
                train_labels,
                task=task,
                params=model_params["lightgbm"],
                validation_frame=valid_task,
                validation_labels=valid_labels,
                sample_weight=weights,
            )
            probabilities = model.predict_proba(valid_task)
            candidates.append(
                {
                    "task": task,
                    "candidate": "lightgbm_temporal_4y",
                    "family": "lightgbm",
                    "fit_seconds": time.perf_counter() - family_started,
                    "parameters": {**model_params["lightgbm"], "sample_weight_half_life": 4.0},
                    "best_iteration": model.best_iteration,
                    "metrics": _metric_payload(valid_labels, probabilities, task),
                }
            )

    manifest_payload: dict[str, Any] = json.loads(feature_manifest.read_text(encoding="utf-8"))
    package_versions = {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scikit_learn": version("scikit-learn"),
        "lightgbm": version("lightgbm"),
        "catboost": version("catboost"),
    }
    if any(name.startswith("chimeraboost") for name in families):
        package_versions["chimeraboost"] = version("chimeraboost")
    if any(name.startswith("tabm") for name in families):
        package_versions["tabm"] = version("tabm")
        package_versions["torch"] = version("torch")
        package_versions["rtdl_num_embeddings"] = version("rtdl_num_embeddings")

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": (
            "COMPLETE_ORACLE_DIAGNOSTIC_NOT_DEPLOYABLE_NOT_CONFIRMATORY"
            if "catboost_oracle" in families
            else "COMPLETE_SCREENING_NOT_CONFIRMATORY"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "feature_manifest": feature_manifest.as_posix(),
        "feature_manifest_sha256": sha256_file(feature_manifest),
        "feature_variant": manifest_payload["feature_variant"],
        "train_years": list(train_years),
        "validation_year": validation_year,
        "rows_per_train_year": rows_per_train_year,
        "validation_limit": validation_limit,
        "loaded_train_rows": len(train),
        "loaded_validation_rows": len(validation),
        "seed": seed,
        "tasks": list(tasks),
        "families": list(families),
        "versions": package_versions,
        "candidates": candidates,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result["result_sha256"] = canonical_json_sha256(result)
    write_canonical_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-start", type=int, default=2011)
    parser.add_argument("--train-end", type=int, default=2017)
    parser.add_argument("--validation-year", type=int, default=2018)
    parser.add_argument("--rows-per-train-year", type=int, default=75_000)
    parser.add_argument("--validation-limit", type=int, default=200_000)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["delay", "cancellation", "joint"],
        default=["delay", "cancellation"],
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=[
            "lightgbm",
            "lightgbm_temporal_4y",
            "catboost",
            "catboost_oracle",
            "chimeraboost",
            "chimeraboost_ensemble",
            "tabm",
            "tabm_periodic",
            "tabm_route",
        ],
        default=["lightgbm", "lightgbm_temporal_4y", "catboost"],
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_benchmark(
        feature_dir=args.feature_dir,
        feature_manifest=args.feature_manifest,
        output_path=args.output,
        train_years=tuple(range(args.train_start, args.train_end + 1)),
        validation_year=args.validation_year,
        rows_per_train_year=args.rows_per_train_year,
        validation_limit=args.validation_limit,
        tasks=tuple(args.tasks),
        families=tuple(args.families),
    )
    print(
        json.dumps({"output": args.output.as_posix(), "candidates": result["candidates"]}, indent=2)
    )


if __name__ == "__main__":
    main()
