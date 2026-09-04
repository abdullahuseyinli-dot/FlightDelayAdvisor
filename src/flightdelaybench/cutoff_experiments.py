"""Matched chronological model experiments for timestamp-validated feature tables.

This runner deliberately does not load the old daily-history feature directories.
It requires a new, hash-bound dataset with availability evidence and never reads
2026. A successful software test is not a completed research run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .bootstrap import paired_cluster_mean_difference
from .contracts import FEATURE_BY_NAME, AvailabilityHorizon, validate_predictors
from .cutoff_history import _dates, utc_timestamps
from .flare_boundary_contracts import BOUNDARY_ALL_FEATURES
from .flare_capacity_contracts import CAPACITY_ALL_FEATURES
from .flare_evaluation import joint_loss_rows
from .flare_reconciliation import hurdle_joint_probabilities
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json


@dataclass(frozen=True)
class ForwardFold:
    name: str
    train_end: str
    stopping_start: str
    stopping_end: str
    score_start: str
    score_end: str


FOLDS = (
    ForwardFold("spring", "2024-02-27", "2024-03-01", "2024-03-29", "2024-04-03", "2024-06-29"),
    ForwardFold("summer", "2024-05-29", "2024-06-01", "2024-06-28", "2024-07-03", "2024-09-29"),
    ForwardFold("autumn", "2024-08-29", "2024-09-01", "2024-09-28", "2024-10-03", "2024-12-29"),
)


@dataclass(frozen=True)
class Trial:
    name: str
    family: str
    formulation: str
    context: str
    monthly_cap: int | None
    seed: int = 20260904


def matched_trials() -> tuple[Trial, ...]:
    """Nine scaling/context fits plus model controls, without a sprawling search."""
    trials = [
        Trial(f"catboost_hurdle_{context}_{cap or 'full'}", "catboost", "hurdle", context, cap)
        for cap in (125_000, 250_000, None)
        for context in ("baseline", "induced", "boundary")
    ]
    trials.extend(
        Trial(f"{family}_{formulation}_boundary_125000", family, formulation, "boundary", 125_000)
        for family in ("catboost", "lightgbm", "tabm")
        for formulation in ("direct", "hurdle")
        if (family, formulation) != ("catboost", "hurdle")
    )
    return tuple(trials)


def nested_monthly_sample(frame: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    """One stable seeded rank per ID gives true nested samples across all caps."""
    if frame["sample_id"].duplicated().any() or frame["sample_id"].isna().any():
        raise ValueError("sampling requires unique nonmissing sample IDs")
    if cap is not None and cap < 1:
        raise ValueError("monthly cap must be positive or None")
    ranks = frame["sample_id"].astype(str).map(
        lambda value: canonical_json_sha256([seed, value])
    )
    ordered = frame.assign(_rank=ranks).sort_values(["_rank", "sample_id"])
    if cap is not None:
        periods = pd.to_datetime(ordered["FlightDate"]).dt.to_period("M")
        ordered = ordered.groupby(periods, sort=False).head(cap)
    return ordered.drop(columns="_rank").reset_index(drop=True)


def validate_new_features(features: tuple[str, ...]) -> None:
    if not features or len(features) != len(set(features)):
        raise ValueError("features must be nonempty and unique")
    legacy = [name for name in features if name.startswith(("recent_", "graph_"))
              or "rotation" in name]
    if legacy:
        raise ValueError(f"legacy untimed history/risk cannot enter corrected trials: {legacy}")
    supplemental = set(CAPACITY_ALL_FEATURES) | set(BOUNDARY_ALL_FEATURES)
    validate_predictors([name for name in features if name not in supplemental], AvailabilityHorizon.FORECAST_24H)


def validate_dataset(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    validate_new_features(features)
    required = {
        "sample_id", "FlightDate", "cutoff_time_utc", "departure_time_utc",
        "features_available_at_utc", "cancel_label_available_at_utc",
        "delay_label_available_at_utc", "Cancelled", "ArrDel15",
        "joint_label_observed", "delay_label_observed", "disruption_state", *features,
    }
    if required - set(frame):
        raise ValueError(f"new dataset missing columns: {sorted(required - set(frame))}")
    if frame.empty or frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("new dataset requires nonempty unique sample IDs")
    result = frame.copy()
    dates = _dates(result["FlightDate"])
    result["FlightDate"] = dates
    if not dates.dt.year.eq(2024).all():
        raise ValueError("development runner only accepts 2024; 2025 audit and 2026 confirmation are separate")
    for column in ("cutoff_time_utc", "departure_time_utc", "features_available_at_utc", "cancel_label_available_at_utc"):
        result[column] = utc_timestamps(result[column], name=column)
    if not result[["joint_label_observed", "delay_label_observed"]].isin([0, 1]).all().all():
        raise ValueError("label-observation flags must be binary")
    observed = result["delay_label_observed"].eq(1)
    if (observed & result["Cancelled"].eq(1)).any():
        raise ValueError("cancelled flights cannot have observed conditional-delay labels")
    if observed.any():
        # Delay evidence can be missing on cancelled/diverted flights only.
        utc_timestamps(result.loc[observed, "delay_label_available_at_utc"], name="delay label availability")
    result["delay_label_available_at_utc"] = pd.to_datetime(result["delay_label_available_at_utc"], utc=True, errors="raise")
    if not (result["departure_time_utc"] - result["cutoff_time_utc"]).eq(pd.Timedelta(hours=24)).all():
        raise ValueError("incorrect T-24 target cutoff")
    if (result["features_available_at_utc"] > result["cutoff_time_utc"]).any():
        raise ValueError("post-cutoff input in corrected dataset")
    if not result["Cancelled"].isin([0, 1]).all() or not result.loc[observed, "ArrDel15"].isin([0, 1]).all():
        raise ValueError("invalid observed binary labels")
    joint = result["joint_label_observed"].eq(1)
    if not joint.eq(result["Cancelled"].eq(1) | observed).all():
        raise ValueError("joint-observation flag disagrees with observed tasks")
    expected = np.where(result["Cancelled"].eq(1), 2, result["ArrDel15"])
    if not np.array_equal(result.loc[joint, "disruption_state"], expected[joint]):
        raise ValueError("joint labels disagree with cancellation/conditional delay semantics")
    if not result.loc[joint, "disruption_state"].isin([0, 1, 2]).all():
        raise ValueError("invalid joint class")
    return result


def split_forward(frame: pd.DataFrame, fold: ForwardFold) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = pd.to_datetime(frame["FlightDate"])
    train = frame.loc[dates.le(fold.train_end)]
    stopping = frame.loc[dates.between(fold.stopping_start, fold.stopping_end)]
    score = frame.loc[dates.between(fold.score_start, fold.score_end)]
    if train.empty or stopping.empty or score.empty:
        raise ValueError(f"fold {fold.name} requires nonempty training, stopping and scoring periods")
    fit_at = utc_timestamps(score["cutoff_time_utc"], name="scoring cutoff").min()
    if not pd.Timestamp(fold.train_end) < pd.Timestamp(fold.stopping_start) <= pd.Timestamp(fold.stopping_end) < pd.Timestamp(fold.score_start) <= pd.Timestamp(fold.score_end):
        raise ValueError("forward fold periods overlap")
    return train, stopping, score, fit_at


def _task_rows(frame: pd.DataFrame, task: str, fit_at: pd.Timestamp) -> tuple[pd.DataFrame, NDArray[np.int64]]:
    known_cancel = frame["cancel_label_available_at_utc"].le(fit_at)
    if task == "cancellation":
        mask = known_cancel
        label = frame["Cancelled"]
    elif task == "delay":
        mask = known_cancel & frame["Cancelled"].eq(0) & frame["delay_label_observed"].eq(1) & frame["delay_label_available_at_utc"].le(fit_at)
        label = frame["ArrDel15"]
    elif task == "joint":
        mask = known_cancel & frame["joint_label_observed"].eq(1) & (
            frame["Cancelled"].eq(1) | frame["delay_label_available_at_utc"].le(fit_at)
        )
        label = frame["disruption_state"]
    else:
        raise ValueError(f"unknown task {task}")
    selected = frame.loc[mask]
    y = label.loc[mask].to_numpy(dtype=np.int64)
    expected = [0, 1, 2] if task == "joint" else [0, 1]
    if sorted(np.unique(y).tolist()) != expected:
        raise ValueError(f"{task} lacks all classes among labels available before fit")
    return selected, y


@dataclass
class FittedClassifier:
    family: str
    estimator: Any
    features: tuple[str, ...]
    categorical: tuple[str, ...]
    levels: dict[str, list[str]]

    def matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = frame.loc[:, list(self.features)].copy()
        for name in self.features:
            if name in self.categorical:
                values = matrix[name].astype("string").fillna("__MISSING__")
                matrix[name] = pd.Categorical(values, categories=self.levels[name]) if self.family == "lightgbm" else values.astype(str)
            else:
                matrix[name] = pd.to_numeric(matrix[name], errors="raise").astype("float32")
                if np.isinf(matrix[name].to_numpy()).any():
                    raise ValueError("model input contains infinity")
        return matrix

    def predict(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        data = frame if self.family == "tabm" else self.matrix(frame)
        probability = np.asarray(self.estimator.predict_proba(data), dtype=np.float64)
        if probability.ndim != 2 or len(probability) != len(frame) or not np.isfinite(probability).all():
            raise ValueError("invalid model probability shape/values")
        if (probability < 0).any() or not np.allclose(probability.sum(axis=1), 1, atol=1e-6, rtol=0):
            raise ValueError("model probabilities violate simplex")
        if hasattr(self.estimator, "classes_") and not np.array_equal(self.estimator.classes_, np.arange(probability.shape[1])):
            raise ValueError("unexpected probability class order")
        return probability


def fit_classifier(
    train: pd.DataFrame, y: NDArray[np.int64], stopping: pd.DataFrame,
    stopping_y: NDArray[np.int64], *, family: str, features: tuple[str, ...],
    categorical: tuple[str, ...], task: str, seed: int, iterations: int = 600,
) -> FittedClassifier:
    validate_new_features(features)
    levels = {name: sorted(train[name].astype("string").fillna("__MISSING__").unique().tolist()) for name in categorical}
    fitted = FittedClassifier(family, None, features, categorical, levels)
    if family == "catboost":
        from catboost import CatBoostClassifier  # type: ignore[import-untyped]
        fitted.estimator = CatBoostClassifier(
            iterations=iterations, depth=8, learning_rate=0.04, l2_leaf_reg=8.0,
            max_ctr_complexity=1, boosting_type="Plain", random_seed=seed,
            loss_function="MultiClass" if task == "joint" else "Logloss",
            task_type="CPU", thread_count=8, allow_writing_files=False, verbose=False,
        )
        fitted.estimator.fit(fitted.matrix(train), y, cat_features=list(categorical),
                             eval_set=(fitted.matrix(stopping), stopping_y), early_stopping_rounds=75)
    elif family == "lightgbm":
        import lightgbm as lgb
        fitted.estimator = lgb.LGBMClassifier(
            n_estimators=iterations, num_leaves=63, learning_rate=0.04,
            reg_lambda=8.0, random_state=seed, n_jobs=8, verbosity=-1,
        )
        fitted.estimator.fit(fitted.matrix(train), y, eval_set=[(fitted.matrix(stopping), stopping_y)],
                             callbacks=[lgb.early_stopping(75, verbose=False)])
    elif family == "tabm":
        from .frontier import fit_tabm
        from .modeling import TaskName
        fitted.estimator = fit_tabm(
            train, y, task=cast(TaskName, task),
            params={"seed": seed, "feature_columns": features, "categorical_features": categorical,
                    "max_epochs": min(40, iterations), "patience": 7, "embedding": "none", "k": 8},
            validation_frame=stopping, validation_labels=stopping_y,
        )
    else:
        raise ValueError(f"unknown model family {family}")
    return fitted


def decision_scores(labels: NDArray[np.int64], probabilities: NDArray[np.float64]) -> dict[str, Any]:
    log_rows, brier = joint_loss_rows(labels, probabilities)
    predicted = probabilities.argmax(axis=1)
    recalls = [float((predicted[labels == k] == k).mean()) if np.any(labels == k) else None for k in range(3)]
    return {
        "rows": len(labels), "accuracy": float((predicted == labels).mean()),
        "joint_log_loss": float(log_rows.mean()), "multiclass_brier": float(brier.mean()),
        "class_recall": recalls,
        "balanced_accuracy": float(np.mean([v for v in recalls if v is not None])),
        "class_prevalence": [float((labels == k).mean()) for k in range(3)],
    }


def select_incumbent_blend(labels: NDArray[np.int64], incumbent: NDArray[np.float64], candidate: NDArray[np.float64]) -> dict[str, float]:
    """Call on forward development predictions only; weight zero retains incumbent."""
    if incumbent.shape != candidate.shape:
        raise ValueError("blend members must align")
    evaluated = [
        (float(joint_loss_rows(labels, (1 - weight) * incumbent + weight * candidate)[0].mean()), float(weight))
        for weight in np.linspace(0, 1, 21)
    ]
    minimum = min(loss for loss, _ in evaluated)
    # Floating point interpolation of identical members need not be bit-identical.
    # Keep the least change within numerical tolerance, not a spurious nonzero fit.
    loss, weight = min((item for item in evaluated if item[0] <= minimum + 1e-12), key=lambda item: item[1])
    return {"candidate_weight": weight, "joint_log_loss": loss}


def paired_diagnostics(frame: pd.DataFrame, candidate: NDArray[np.float64], reference: NDArray[np.float64], repetitions: int = 2000) -> dict[str, Any]:
    y = frame["disruption_state"].to_numpy(dtype=np.int64)
    candidate_log, candidate_brier = joint_loss_rows(y, candidate)
    reference_log, reference_brier = joint_loss_rows(y, reference)
    dates = pd.to_datetime(frame["FlightDate"])
    result: dict[str, Any] = {}
    # Fixed nonoverlapping week blocks are a serial-dependence sensitivity, not
    # an independence claim about airports/days within a block.
    for name, clusters in (("date", dates), ("seven_day_block", dates.dt.to_period("W-SUN").astype(str))):
        if pd.Series(clusters).nunique() < 2:
            result[name] = {"status": "NOT_ESTIMABLE_INSUFFICIENT_CLUSTERS"}
            continue
        result[name] = {
            metric: asdict(paired_cluster_mean_difference(a, b, clusters, repetitions=repetitions))
            for metric, a, b in (
                ("joint_log_loss", candidate_log, reference_log),
                ("multiclass_brier", candidate_brier, reference_brier),
                ("accuracy", (candidate.argmax(axis=1) == y).astype(float), (reference.argmax(axis=1) == y).astype(float)),
            )
        }
    return result


def run_trial(frame: pd.DataFrame, *, trial: Trial, fold: ForwardFold, features: tuple[str, ...], output_dir: Path, iterations: int = 600) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite trial {output_dir}")
    try:
        return _run_trial(frame, trial=trial, fold=fold, features=features, output_dir=output_dir, iterations=iterations)
    except Exception as error:
        if output_dir.exists() and not (output_dir / "failure.json").exists():
            write_canonical_json(output_dir / "failure.json", {
                "status": "FAILED_DEVELOPMENT_TRIAL", "exception_type": type(error).__name__,
                "message": str(error), "confirmation_outcomes_accessed": False,
            })
        raise


def _run_trial(frame: pd.DataFrame, *, trial: Trial, fold: ForwardFold, features: tuple[str, ...], output_dir: Path, iterations: int = 600) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite trial {output_dir}")
    validated = validate_dataset(frame, features)
    train, stopping, score, fit_at = split_forward(validated, fold)
    # Training must also be honest for the stopping period, not just the later
    # score period. Otherwise late-published training labels contaminate tuning.
    training_fit_at = utc_timestamps(stopping["cutoff_time_utc"], name="stopping cutoff").min()
    train = nested_monthly_sample(train, trial.monthly_cap, trial.seed)
    stopping = nested_monthly_sample(stopping, 250_000, trial.seed)
    categorical = tuple(name for name in features if name in FEATURE_BY_NAME and FEATURE_BY_NAME[name].categorical)
    tasks = ("joint",) if trial.formulation == "direct" else ("cancellation", "delay")
    if trial.formulation not in {"direct", "hurdle"}:
        raise ValueError("formulation must be direct or hurdle")
    output_dir.mkdir(parents=True)
    write_canonical_json(output_dir / "intent.json", {
        "status": "STARTED_DEVELOPMENT_TRIAL", "trial": asdict(trial), "fold": asdict(fold),
        "created_at_utc": datetime.now(UTC).isoformat(), "features": list(features),
        "iterations": iterations, "training_fit_information_cutoff": training_fit_at.isoformat(),
        "selection_information_cutoff": fit_at.isoformat(),
    })
    predictions: dict[str, NDArray[np.float64]] = {}
    models: dict[str, FittedClassifier] = {}
    sizes = {}
    try:
        for task in tasks:
            train_task, y = _task_rows(train, task, training_fit_at)
            stop_task, stop_y = _task_rows(stopping, task, fit_at)
            model = fit_classifier(train_task, y, stop_task, stop_y, family=trial.family,
                                   features=features, categorical=categorical, task=task, seed=trial.seed, iterations=iterations)
            predictions[task] = model.predict(score)
            models[task] = model
            sizes[task] = {
                "training": len(y), "stopping": len(stop_y),
                "training_ids_sha256": canonical_json_sha256(sorted(train_task["sample_id"].astype(str))),
                "stopping_ids_sha256": canonical_json_sha256(sorted(stop_task["sample_id"].astype(str))),
            }
    except Exception as error:
        write_canonical_json(output_dir / "failure.json", {
            "status": "FAILED_DEVELOPMENT_TRIAL", "exception_type": type(error).__name__,
            "message": str(error), "completed_tasks": list(predictions),
            "confirmation_outcomes_accessed": False,
        })
        raise
    joint = predictions["joint"] if trial.formulation == "direct" else hurdle_joint_probabilities(
        cancel_probability=predictions["cancellation"][:, 1],
        delay_given_operated_probability=predictions["delay"][:, 1],
    )
    observed = score["joint_label_observed"].eq(1).to_numpy()
    metrics = decision_scores(score.loc[observed, "disruption_state"].to_numpy(dtype=np.int64), joint[observed])
    artifact = score[["sample_id", "FlightDate", "disruption_state", "joint_label_observed"]].reset_index(drop=True)
    artifact[["on_time", "delayed", "cancelled"]] = joint
    prediction_path = output_dir / "predictions.parquet"
    artifact.to_parquet(prediction_path, index=False)
    model_path = output_dir / "models.joblib"
    joblib.dump(models, model_path, compress=3)
    report = {
        "schema_version": 1, "status": "COMPLETED_DEVELOPMENT_TRIAL",
        "trial": asdict(trial), "fold": asdict(fold),
        "training_fit_information_cutoff": training_fit_at.isoformat(),
        "selection_information_cutoff": fit_at.isoformat(),
        "task_rows": sizes, "features": list(features), "scores": metrics,
        "training_ids_sha256": canonical_json_sha256(sorted(train["sample_id"].astype(str))),
        "scored_ids_sha256": canonical_json_sha256(sorted(score.loc[observed, "sample_id"].astype(str))),
        "predictions": {"path": prediction_path.as_posix(), "sha256": sha256_file(prediction_path)},
        "models": {"path": model_path.as_posix(), "sha256": sha256_file(model_path)},
        "confirmation_outcomes_accessed": False,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(output_dir / "report.json", report)
    return report


def compare_prediction_tables(candidate: pd.DataFrame, reference: pd.DataFrame, *, repetitions: int = 2000) -> dict[str, Any]:
    """Pair by exact identity, not row order or independently selected cohorts."""
    for frame in (candidate, reference):
        if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ValueError("prediction comparison requires unique nonmissing IDs")
    if set(candidate["sample_id"]) != set(reference["sample_id"]):
        raise ValueError("prediction cohorts differ")
    left = candidate.set_index("sample_id").sort_index()
    right = reference.set_index("sample_id").reindex(left.index)
    observed = left["joint_label_observed"].eq(1)
    for column in ("FlightDate", "joint_label_observed"):
        if not left[column].eq(right[column]).all():
            raise ValueError(f"prediction provenance differs: {column}")
    if not left.loc[observed, "disruption_state"].eq(right.loc[observed, "disruption_state"]).all():
        raise ValueError("prediction labels differ")
    columns = ["on_time", "delayed", "cancelled"]
    return paired_diagnostics(left.loc[observed], left.loc[observed, columns].to_numpy(dtype=float),
                              right.loc[observed, columns].to_numpy(dtype=float), repetitions=repetitions)


def preflight(data_root: Path, corrected_manifest: Path | None) -> dict[str, Any]:
    disk = shutil.disk_usage(data_root)
    dependencies = {name: importlib.util.find_spec(name) is not None for name in ("catboost", "lightgbm", "torch", "tabm")}
    blockers = []
    if disk.free < 20 * 1024**3:
        blockers.append("Fewer than 20 GiB available for corrected data and preserved experiment artifacts")
    if corrected_manifest is None or not corrected_manifest.is_file():
        blockers.append("No hash-bound dataset with observation/source/label availability timestamps")
    else:
        from .cutoff_dataset import validate_dataset_manifest
        try:
            validate_dataset_manifest(corrected_manifest)
        except (ValueError, OSError, KeyError, TypeError) as error:
            blockers.append(f"Corrected dataset manifest rejected: {error}")
    if not all(dependencies.values()):
        blockers.append("Required optional model dependencies missing; installation needs storage")
    return {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "BLOCKED_RESEARCH_PREREQUISITES" if blockers else "READY_FOR_MATCHED_DEVELOPMENT",
        "data_root": data_root.as_posix(), "free_bytes": disk.free,
        "required_free_bytes_planning_reserve": 20 * 1024**3,
        "reserve_is_measured_output_requirement": False,
        "dependencies": dependencies, "blockers": blockers,
        "trials_per_fold": [asdict(trial) for trial in matched_trials()],
        "folds": [asdict(fold) for fold in FOLDS],
        "research_trials_executed": 0,
        "conditional_steps": ["retune finalists", "repeat finalists at seeds 20260905 and 20260906", "new-information pilot", "capacity model only if pilot passes"],
        "confirmation_outcomes_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--corrected-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, help="Execute the registered grid into this new directory after a passing preflight")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite preflight {args.output}")
    report = preflight(args.data_root, args.corrected_manifest)
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(args.output, report)
    print(json.dumps(report, indent=2))
    if report["blockers"]:
        raise SystemExit(2)
    if args.run_dir is not None:
        if args.corrected_manifest is None:
            raise ValueError("execution requires a corrected dataset manifest")
        run_grid(args.corrected_manifest, args.run_dir)


def run_grid(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    """Execute registered trials sequentially, retaining every trial and failure.

    The grid is development evidence. It does not tune on scoring outcomes, select
    confirmation models, or imply that subsequent pilot/architecture gates passed.
    """
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite matched grid {output_dir}")
    from .cutoff_dataset import load_dataset
    frame, manifest = load_dataset(manifest_path)
    output_dir.mkdir(parents=True)
    write_canonical_json(output_dir / "lock.json", {
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "folds": [asdict(fold) for fold in FOLDS],
        "trials": [asdict(trial) for trial in matched_trials()],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "confirmation_outcomes_accessed": False,
    })
    reports = []
    failures = []
    comparisons = []
    for fold in FOLDS:
        for trial in matched_trials():
            trial_dir = output_dir / fold.name / trial.name
            try:
                report = run_trial(frame, trial=trial, fold=fold,
                                   features=tuple(manifest["features_by_context"][trial.context]), output_dir=trial_dir)
                reports.append({"fold": fold.name, "trial": trial.name, "report_sha256": report["report_sha256"], "scores": report["scores"]})
                reference_name = f"catboost_hurdle_baseline_{trial.monthly_cap or 'full'}"
                reference_path = output_dir / fold.name / reference_name / "predictions.parquet"
                if trial.name != reference_name:
                    comparison: dict[str, Any] = {"fold": fold.name, "candidate": trial.name, "reference": reference_name}
                    if reference_path.is_file():
                        comparison["status"] = "PAIRED_DEVELOPMENT_COMPARISON"
                        comparison["candidate_minus_reference"] = compare_prediction_tables(pd.read_parquet(trial_dir / "predictions.parquet"), pd.read_parquet(reference_path))
                        comparison["accuracy_unit"] = "absolute fraction; +0.05 means five percentage points"
                    else:
                        comparison["status"] = "NOT_ESTIMABLE_REFERENCE_TRIAL_MISSING"
                    comparisons.append(comparison)
                    write_canonical_json(trial_dir / "paired_comparison.json", comparison)
            except Exception as error:
                failure = {"fold": fold.name, "trial": trial.name, "exception_type": type(error).__name__, "message": str(error)}
                failures.append(failure)
                trial_dir.mkdir(parents=True, exist_ok=True)
                if not (trial_dir / "failure.json").exists():
                    write_canonical_json(trial_dir / "failure.json", failure)
                # Resource exhaustion is not a reason to launch more large fits.
                if isinstance(error, (MemoryError, OSError)):
                    break
            print(json.dumps({"fold": fold.name, "trial": trial.name, "completed": len(reports), "failed": len(failures)}), flush=True)
        if failures and failures[-1]["exception_type"] in {"MemoryError", "OSError"}:
            break
    result: dict[str, Any] = {
        "status": "COMPLETED_MATCHED_DEVELOPMENT_GRID" if len(reports) == len(FOLDS) * len(matched_trials()) and not failures else "INCOMPLETE_MATCHED_DEVELOPMENT_GRID",
        "reports": reports, "failures": failures, "comparisons": comparisons,
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "retuning_and_finalist_repeats_completed": False,
        "new_information_pilot_gate_passed": False, "publication_ready": False,
        "confirmation_outcomes_accessed": False,
    }
    result["report_sha256"] = canonical_json_sha256(result)
    write_canonical_json(output_dir / "summary.json", result)
    if failures:
        raise RuntimeError("matched grid incomplete; failures preserved in summary.json")
    return result


if __name__ == "__main__":
    main()
