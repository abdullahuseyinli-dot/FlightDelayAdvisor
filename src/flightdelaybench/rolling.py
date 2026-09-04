"""Run fixed-method rolling-origin model evaluations through the selection year."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from numpy.typing import NDArray

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

CATBOOST_PARAMS: dict[TaskName, dict[str, Any]] = {
    "delay": {
        "iterations": 144,
        "learning_rate": 0.04819163887123088,
        "depth": 7,
        "l2_leaf_reg": 5.125915831634838,
        "random_strength": 0.27575628939933405,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.3984891589789224,
        "border_count": 128,
    },
    "cancellation": {
        "iterations": 117,
        "learning_rate": 0.04266275558194561,
        "depth": 10,
        "l2_leaf_reg": 1.8439374710684058,
        "random_strength": 1.8211594669189084,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.1653117632683447,
        "border_count": 64,
    },
    "joint": {
        "iterations": 180,
        "learning_rate": 0.06,
        "depth": 8,
        "l2_leaf_reg": 6.0,
    },
}

LIGHTGBM_PARAMS: dict[TaskName, dict[str, Any]] = {
    "delay": {
        "n_estimators": 55,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 250,
        "max_bin": 127,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
    },
    "cancellation": {
        "n_estimators": 40,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 250,
        "max_bin": 127,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
    },
    "joint": {
        "n_estimators": 75,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 250,
        "max_bin": 127,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
    },
}


def _atomic_joblib(payload: Any, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite model evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated partial model exists: {partial}")
    joblib.dump(payload, partial, compress=3)
    partial.replace(path)
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prediction evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated partial predictions exist: {partial}")
    frame.to_parquet(partial, index=False, compression="zstd", row_group_size=100_000)
    partial.replace(path)
    return {
        "path": path.as_posix(),
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _prior_probability(frame: pd.DataFrame, task: TaskName) -> NDArray[np.float64]:
    outcome = "delay" if task == "delay" else "cancel"
    return np.asarray(frame[f"prior_route_{outcome}_rate"], dtype=np.float64)


def _training_years(
    target_year: int, *, minimum_year: int, recent_window_years: int | None
) -> tuple[int, ...]:
    start = minimum_year
    if recent_window_years is not None:
        if recent_window_years < 2:
            raise ValueError("recent training windows must span at least two years")
        start = max(start, target_year - recent_window_years)
    years = tuple(range(start, target_year))
    if not years:
        raise ValueError(f"no training years precede {target_year}")
    return years


def _prediction_frame(
    evaluation: pd.DataFrame,
    probabilities: NDArray[np.float64],
    *,
    task: TaskName,
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
    output = evaluation.loc[:, columns].copy()
    if task == "joint":
        if probabilities.ndim != 2 or probabilities.shape[1] != 3:
            raise ValueError("joint model must return three probability columns")
        output[["prob_on_time", "prob_delayed", "prob_cancelled"]] = probabilities.astype("float32")
    else:
        binary = probabilities[:, 1] if probabilities.ndim == 2 else probabilities
        output["probability"] = binary.astype("float32")
        outcome = "delay" if task == "delay" else "cancel"
        for level in ("global", "route", "slot"):
            output[f"baseline_{level}"] = evaluation[f"prior_{level}_{outcome}_rate"].astype(
                "float32"
            )
    return output


def _score_task(
    evaluation: pd.DataFrame,
    probabilities: NDArray[np.float64],
    task: TaskName,
) -> dict[str, Any]:
    eligible, labels = task_view(evaluation, task)
    if task == "delay":
        mask = evaluation["Cancelled"].eq(0) & evaluation["delay_label_observed"].eq(1)
    elif task == "cancellation":
        mask = evaluation["Cancelled"].isin([0, 1])
    else:
        mask = evaluation["joint_label_observed"].eq(1)
    selected_probabilities = probabilities[np.asarray(mask)]
    if len(eligible) != len(selected_probabilities):
        raise AssertionError("probabilities and task population are misaligned")
    if task == "joint":
        return {
            "n": len(labels),
            "log_loss": float(
                -np.mean(
                    np.log(
                        np.clip(selected_probabilities[np.arange(len(labels)), labels], 1e-6, 1.0)
                    )
                )
            ),
            "multiclass_brier": multiclass_brier(labels, selected_probabilities),
            "class_prevalence": {str(label): float(np.mean(labels == label)) for label in range(3)},
        }
    binary = (
        selected_probabilities[:, 1] if selected_probabilities.ndim == 2 else selected_probabilities
    )
    metrics = binary_metrics(labels, binary).as_dict()
    metrics["prior_route_log_loss"] = binary_metrics(
        labels, _prior_probability(eligible, task)
    ).log_loss
    return metrics


def run_rolling_candidate(
    *,
    candidate_name: str,
    family: str,
    feature_dir: Path,
    feature_manifest: Path,
    run_dir: Path,
    report_path: Path,
    target_years: tuple[int, ...],
    tasks: tuple[TaskName, ...],
    rows_per_train_year: int,
    evaluation_limit: int | None = None,
    recent_window_years: int | None = None,
    temporal_weight_half_life: float | None = None,
    seed: int = 20260903,
) -> dict[str, Any]:
    """Evaluate one frozen candidate on successive unseen calendar years."""

    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse rolling run directory: {run_dir}")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite rolling report: {report_path}")
    if not target_years or min(target_years) < 2018 or max(target_years) > 2024:
        raise ValueError("development/selection rolling targets must be within 2018-2024")
    if family not in {"catboost", "lightgbm"}:
        raise ValueError(f"unsupported family: {family}")
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
        train = load_feature_years(
            feature_dir,
            years,
            rows_per_year=rows_per_train_year,
            seed=seed,
        )
        evaluation = load_feature_year(
            feature_dir,
            target_year,
            limit=evaluation_limit,
            seed=seed,
        )
        for task in tasks:
            task_started = time.perf_counter()
            train_task, train_labels = task_view(train, task)
            weights = temporal_sample_weights(
                train_task["Year"],
                prediction_year=target_year,
                half_life_years=temporal_weight_half_life,
            )
            parameters = (
                CATBOOST_PARAMS[task].copy()
                if family == "catboost"
                else LIGHTGBM_PARAMS[task].copy()
            )
            if family == "catboost":
                catboost_model = fit_catboost(
                    train_task,
                    train_labels,
                    task=task,
                    params=parameters,
                    sample_weight=weights,
                )
                probabilities = catboost_model.predict_proba(evaluation)
                model_record = _atomic_joblib(
                    catboost_model,
                    run_dir / "models" / f"{task}_{target_year}.joblib",
                )
            else:
                lightgbm_model = fit_lightgbm(
                    train_task,
                    train_labels,
                    task=task,
                    params=parameters,
                    sample_weight=weights,
                )
                probabilities = lightgbm_model.predict_proba(evaluation)
                model_record = _atomic_joblib(
                    lightgbm_model,
                    run_dir / "models" / f"{task}_{target_year}.joblib",
                )
            prediction_record = _atomic_parquet(
                _prediction_frame(evaluation, probabilities, task=task),
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
                    "parameters": parameters,
                    "metrics": _score_task(evaluation, probabilities, task),
                    "elapsed_seconds": time.perf_counter() - task_started,
                }
            )
            if family == "catboost":
                del catboost_model
            else:
                del lightgbm_model
            del probabilities, train_task, train_labels, weights
            gc.collect()
        del train, evaluation
        gc.collect()

    feature_payload: dict[str, Any] = json.loads(feature_manifest.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_DEVELOPMENT_AND_SELECTION_NOT_CONFIRMATORY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "candidate_name": candidate_name,
        "family": family,
        "feature_variant": feature_payload["feature_variant"],
        "feature_manifest": feature_manifest.as_posix(),
        "feature_manifest_sha256": sha256_file(feature_manifest),
        "target_years": list(target_years),
        "tasks": list(tasks),
        "rows_per_train_year": rows_per_train_year,
        "evaluation_limit": evaluation_limit,
        "recent_window_years": recent_window_years,
        "temporal_weight_half_life": temporal_weight_half_life,
        "seed": seed,
        "versions": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
            family: version(family),
        },
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
    parser.add_argument("--family", choices=["catboost", "lightgbm"], required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-years", type=int, nargs="+", default=list(range(2018, 2025)))
    parser.add_argument(
        "--tasks",
        choices=["delay", "cancellation", "joint"],
        nargs="+",
        default=["delay", "cancellation"],
    )
    parser.add_argument("--rows-per-train-year", type=int, default=100_000)
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--recent-window-years", type=int)
    parser.add_argument("--temporal-weight-half-life", type=float)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_rolling_candidate(
        candidate_name=args.candidate_name,
        family=args.family,
        feature_dir=args.feature_dir,
        feature_manifest=args.feature_manifest,
        run_dir=args.run_dir,
        report_path=args.report,
        target_years=tuple(args.target_years),
        tasks=tuple(args.tasks),
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
