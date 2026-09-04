"""Post-hoc task-factorized extension of the CC-RTH flight forecast.

The extension learns cancellation and delay-given-operation mixtures separately
from the existing Q4 2024 forward predictions, then recombines the components as
a coherent hurdle distribution.  It also performs a transparent 5x5 component
swap audit on the already-opened 2025 retrospective period.  That grid is
hypothesis-generating only; its winner is frozen for a future, unopened 2026
confirmation rather than described as independently validated.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
import tomllib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .flare_capacity_study import (
    BLEND_CANDIDATES,
    GATING_FEATURE,
    GATING_LABELS,
    PRIMARY_AUDIT_END,
    PRIMARY_AUDIT_START,
    _airport_scores,
    _atomic_parquet,
    _get_joint_columns,
    _joint_columns,
    _joint_point_scores,
    _monthly_scores,
    _read_self_hashed,
    _regime_scores,
    _set_joint_columns,
    _verified_artifact,
    capacity_regimes,
)
from .flare_evaluation import evaluate_joint_probabilities, joint_loss_rows
from .flare_reconciliation import (
    hurdle_joint_probabilities,
    joint_binary_marginals,
)
from .flare_study import PREDICTION_ID_COLUMNS
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

FACTORIZED_METHODS = (
    "task_factorized_global_q4",
    "task_factorized_capacity_gated_q4",
)
Q4_SELECTION_METHODS = (
    "flare24",
    "capacity_gated_simplex",
    *FACTORIZED_METHODS,
)
COMPONENT_TASKS = ("cancellation", "delay_given_operated")
PRIMARY_YEAR = 2025
CONFIRMATION_YEAR = 2026
EXPECTED_PARENT_STATUS = (
    "COMPLETE_CCRTH_2024_SELECTION_2025_RETROSPECTIVE_EVALUATION"
)
EXPECTED_PARENT_VALIDATION_STATUS = "PASS_CCRTH_STUDY_VALIDATION"
EXPECTED_PARENT_LOCK_STATUS = (
    "LOCKED_CCRTH_V1_BEFORE_THIS_RUN_OPENED_2025_OUTCOMES_NOT_BLIND_CONFIRMATION"
)


def _frozen_factorized_implementation_files() -> tuple[Path, ...]:
    directory = Path(__file__).parent
    return tuple(
        directory / name
        for name in (
            "flare_capacity_factorized.py",
            "flare_capacity_study.py",
            "flare_evaluation.py",
            "flare_reconciliation.py",
            "flare_study.py",
            "bootstrap.py",
            "metrics.py",
            "hashing.py",
            "provenance.py",
        )
    )


def _verify_implementation_unchanged(
    started: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    current = capture_provenance(_frozen_factorized_implementation_files())
    if (
        current.get("git_head") != started.get("git_head")
        or current.get("source_files") != started.get("source_files")
    ):
        raise RuntimeError(
            "TF-CC-RTH implementation provenance changed before "
            f"{phase}; retain the partial run and restart with a new version"
        )
    return current


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite TF-CC-RTH artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated TF-CC-RTH partial exists: {partial}")
    write_canonical_json(partial, payload)
    partial.replace(path)


def _load_protocol(path: Path, *, repetitions: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        protocol: dict[str, Any] = tomllib.load(handle)
    identity = protocol.get("identity", {})
    boundary = protocol.get("information_boundary", {})
    factorization = protocol.get("factorization", {})
    gating = protocol.get("stress_gating", {})
    evaluation = protocol.get("evaluation", {})
    grid = protocol.get("post_hoc_grid", {})
    if (
        identity.get("method") != "TF-CC-RTH-v1"
        or identity.get("status") != "POST_HOC_EXPLORATORY_2025_REANALYSIS"
        or int(boundary.get("selection_year", -1)) != 2024
        or int(boundary.get("exploratory_evaluation_year", -1)) != PRIMARY_YEAR
        or int(boundary.get("confirmation_year", -1)) != CONFIRMATION_YEAR
        or boundary.get("confirmation_outcomes_accessed") is not False
        or boundary.get("prior_2025_aggregate_results_informed_hypothesis") is not True
        or boundary.get("row_level_2025_predictions_loaded_before_q4_lock") is not False
        or tuple(factorization.get("components", ())) != COMPONENT_TASKS
        or tuple(factorization.get("base_candidates", ())) != BLEND_CANDIDATES
        or gating.get("feature") != GATING_FEATURE
        or tuple(gating.get("regimes", ())) != GATING_LABELS
        or int(gating.get("minimum_component_rows", -1)) != 10_000
        or tuple(evaluation.get("primary_dates", ()))
        != (PRIMARY_AUDIT_START, PRIMARY_AUDIT_END)
        or int(evaluation.get("bootstrap_repetitions", -1)) != repetitions
        or int(evaluation.get("bootstrap_seed", -1)) != seed
        or grid.get("enabled") is not True
        or int(grid.get("component_pairs", -1)) != len(BLEND_CANDIDATES) ** 2
    ):
        raise ValueError("TF-CC-RTH executable settings differ from the frozen protocol")
    return protocol, {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "status": identity["status"],
    }


def _verify_parent_inputs(
    report_path: Path,
    validation_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    report, report_hash_key = _read_self_hashed(report_path)
    if report.get("status") != EXPECTED_PARENT_STATUS:
        raise ValueError("TF-CC-RTH requires the completed CC-RTH-v1 study")
    if report.get("outcomes_accessed", {}).get("2026_accessed") is not False:
        raise ValueError("parent CC-RTH report crossed the 2026 confirmation boundary")
    validation, validation_hash_key = _read_self_hashed(validation_path)
    if validation.get("status") != EXPECTED_PARENT_VALIDATION_STATUS:
        raise ValueError("TF-CC-RTH requires a passing CC-RTH study validation")
    bound_report = validation.get("report", {})
    if (
        Path(str(bound_report.get("path", ""))).resolve() != report_path.resolve()
        or bound_report.get("sha256") != sha256_file(report_path)
        or bound_report.get("self_hash") != report[report_hash_key]
    ):
        raise ValueError("CC-RTH validation is not bound to the supplied parent report")
    lock_record = dict(report.get("method_lock", {}))
    lock_path = _verified_artifact(lock_record, role="parent CC-RTH method lock")
    parent_lock, parent_lock_hash_key = _read_self_hashed(lock_path)
    if (
        parent_lock.get("status") != EXPECTED_PARENT_LOCK_STATUS
        or parent_lock.get("outcomes_accessed_by_this_extension_before_lock", {}).get(
            "2025_loaded"
        )
        is not False
        or parent_lock.get("outcomes_accessed_by_this_extension_before_lock", {}).get(
            "2026_loaded"
        )
        is not False
    ):
        raise ValueError("parent CC-RTH method lock has an invalid boundary status")
    return report, parent_lock, {
        "path": report_path.as_posix(),
        "bytes": report_path.stat().st_size,
        "sha256": sha256_file(report_path),
        "self_hash_key": report_hash_key,
        "self_hash": report[report_hash_key],
    }, {
        "path": validation_path.as_posix(),
        "bytes": validation_path.stat().st_size,
        "sha256": sha256_file(validation_path),
        "self_hash_key": validation_hash_key,
        "self_hash": validation[validation_hash_key],
        "status": validation["status"],
        "parent_method_lock": {
            "path": lock_path.as_posix(),
            "sha256": sha256_file(lock_path),
            "self_hash_key": parent_lock_hash_key,
            "self_hash": parent_lock[parent_lock_hash_key],
        },
    }


def select_binary_probability_simplex(
    labels: ArrayLike,
    probabilities: Mapping[str, ArrayLike],
) -> dict[str, Any]:
    """Fit a non-negative probability mixture for one binary hurdle task."""

    if tuple(probabilities) != BLEND_CANDIDATES:
        raise ValueError(f"binary blend candidates must be ordered as {BLEND_CANDIDATES}")
    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1 or y.size == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("binary blend labels must be a non-empty binary vector")
    vectors = [np.asarray(probabilities[name], dtype=np.float64) for name in BLEND_CANDIDATES]
    if any(vector.shape != y.shape for vector in vectors):
        raise ValueError("binary probability vectors must align with labels")
    if any(
        not np.isfinite(vector).all() or (vector < 0.0).any() or (vector > 1.0).any()
        for vector in vectors
    ):
        raise ValueError("binary probabilities must be finite and in [0, 1]")
    true_class = np.column_stack(
        [np.where(y == 1, vector, 1.0 - vector) for vector in vectors]
    )

    def objective(weights: NDArray[np.float64]) -> float:
        mixture = np.clip(true_class @ weights, 1e-12, 1.0)
        return float(-np.log(mixture).mean())

    def gradient(weights: NDArray[np.float64]) -> NDArray[np.float64]:
        mixture = np.clip(true_class @ weights, 1e-12, 1.0)
        return np.asarray(-(true_class / mixture[:, None]).mean(axis=0), dtype=np.float64)

    initial = np.full(len(BLEND_CANDIDATES), 1.0 / len(BLEND_CANDIDATES))
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(initial),
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    if not result.success:
        raise RuntimeError(f"TF-CC-RTH binary simplex failed: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    weights /= weights.sum()
    blended = sum(weight * vector for weight, vector in zip(weights, vectors, strict=True))
    clipped = np.clip(blended, 1e-12, 1.0 - 1e-12)
    return {
        "selection_metric": "binary_log_loss",
        "optimizer": "convex SLSQP with analytic gradient",
        "success": True,
        "iterations": int(result.nit),
        "rows": len(y),
        "prevalence": float(y.mean()),
        "weights": {
            name: float(weight)
            for name, weight in zip(BLEND_CANDIDATES, weights, strict=True)
        },
        "binary_log_loss": float(
            -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean()
        ),
        "binary_brier": float(np.square(blended - y).mean()),
    }


def blend_binary_probabilities(
    probabilities: Mapping[str, ArrayLike],
    weights: dict[str, float],
) -> NDArray[np.float64]:
    if tuple(probabilities) != BLEND_CANDIDATES or set(weights) != set(BLEND_CANDIDATES):
        raise ValueError("TF-CC-RTH binary blend has an unexpected candidate set")
    vectors = {
        name: np.asarray(probabilities[name], dtype=np.float64)
        for name in BLEND_CANDIDATES
    }
    shapes = {vector.shape for vector in vectors.values()}
    weight_values = np.asarray([weights[name] for name in BLEND_CANDIDATES])
    if (
        len(shapes) != 1
        or len(next(iter(shapes))) != 1
        or not np.isfinite(weight_values).all()
        or (weight_values < 0.0).any()
        or not np.isclose(weight_values.sum(), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise ValueError("TF-CC-RTH binary blend inputs are invalid")
    result = sum(weights[name] * vectors[name] for name in BLEND_CANDIDATES)
    output = np.asarray(result, dtype=np.float64)
    if not np.isfinite(output).all() or (output < -1e-12).any() or (output > 1.0 + 1e-12).any():
        raise RuntimeError("TF-CC-RTH binary blend produced invalid probabilities")
    return np.clip(output, 0.0, 1.0)


def _components(
    probabilities: Mapping[str, NDArray[np.float64]],
) -> dict[str, dict[str, NDArray[np.float64]]]:
    result: dict[str, dict[str, NDArray[np.float64]]] = {
        "cancellation": {},
        "delay_given_operated": {},
    }
    for name in BLEND_CANDIDATES:
        views = joint_binary_marginals(probabilities[name])
        result["cancellation"][name] = views["cancel_probability"]
        result["delay_given_operated"][name] = views[
            "delay_given_operated_probability"
        ]
    return result


def _task_mask_labels(
    frame: pd.DataFrame,
    task: str,
) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
    if task == "cancellation":
        mask = frame["Cancelled"].isin([0, 1]).to_numpy(dtype=np.bool_)
        labels = frame["Cancelled"].fillna(-1).to_numpy(dtype=np.int64)
    elif task == "delay_given_operated":
        mask = (
            frame["Cancelled"].eq(0) & frame["delay_label_observed"].eq(1)
        ).to_numpy(dtype=np.bool_)
        labels = frame["ArrDel15"].fillna(-1).to_numpy(dtype=np.int64)
    else:
        raise ValueError(f"unknown TF-CC-RTH task: {task}")
    if not mask.any() or not np.isin(labels[mask], [0, 1]).all():
        raise ValueError(f"TF-CC-RTH {task} has no valid binary outcomes")
    return mask, labels


def select_task_factorized_global(
    frame: pd.DataFrame,
    probabilities: Mapping[str, NDArray[np.float64]],
) -> dict[str, Any]:
    components = _components(probabilities)
    selections: dict[str, Any] = {}
    for task in COMPONENT_TASKS:
        mask, labels = _task_mask_labels(frame, task)
        selections[task] = select_binary_probability_simplex(
            labels[mask],
            {name: components[task][name][mask] for name in BLEND_CANDIDATES},
        )
    return {
        "selection_scope": "Q4 2024 forward predictions",
        "components": selections,
    }


def apply_task_factorized_global(
    probabilities: Mapping[str, NDArray[np.float64]],
    selection: dict[str, Any],
) -> NDArray[np.float64]:
    components = _components(probabilities)
    cancel = blend_binary_probabilities(
        components["cancellation"],
        cast(dict[str, float], selection["components"]["cancellation"]["weights"]),
    )
    delay = blend_binary_probabilities(
        components["delay_given_operated"],
        cast(
            dict[str, float],
            selection["components"]["delay_given_operated"]["weights"],
        ),
    )
    return hurdle_joint_probabilities(cancel, delay)


def select_task_factorized_capacity_gated(
    frame: pd.DataFrame,
    probabilities: Mapping[str, NDArray[np.float64]],
    *,
    cutpoints: ArrayLike,
    minimum_rows: int,
    global_selection: dict[str, Any],
) -> dict[str, Any]:
    cuts = np.asarray(cutpoints, dtype=np.float64)
    if cuts.shape != (2,) or not np.isfinite(cuts).all() or cuts[0] > cuts[1]:
        raise ValueError("TF-CC-RTH capacity gating requires two ordered cutpoints")
    regimes = capacity_regimes(frame[GATING_FEATURE], cuts)
    components = _components(probabilities)
    selections: dict[str, Any] = {}
    for regime in GATING_LABELS:
        regime_mask = regimes == regime
        task_selections: dict[str, Any] = {}
        for task in COMPONENT_TASKS:
            eligible, labels = _task_mask_labels(frame, task)
            mask = regime_mask & eligible
            if int(mask.sum()) < minimum_rows or np.unique(labels[mask]).size < 2:
                task_selections[task] = {
                    **global_selection["components"][task],
                    "rows": int(mask.sum()),
                    "fallback_to_global": True,
                }
                continue
            selected = select_binary_probability_simplex(
                labels[mask],
                {name: components[task][name][mask] for name in BLEND_CANDIDATES},
            )
            task_selections[task] = {
                **selected,
                "fallback_to_global": False,
            }
        selections[regime] = {
            "rows_all_flights": int(regime_mask.sum()),
            "components": task_selections,
        }
    return {
        "gating_feature": GATING_FEATURE,
        "cutpoints": cuts.tolist(),
        "cutpoints_source": "frozen parent CC-RTH-v1 Q4 method lock",
        "minimum_component_rows": minimum_rows,
        "global": global_selection,
        "regimes": selections,
    }


def apply_task_factorized_capacity_gated(
    probabilities: Mapping[str, NDArray[np.float64]],
    gating_values: ArrayLike,
    selection: dict[str, Any],
) -> NDArray[np.float64]:
    components = _components(probabilities)
    regimes = capacity_regimes(gating_values, selection["cutpoints"])
    n_rows = len(next(iter(probabilities.values())))
    if regimes.shape != (n_rows,):
        raise ValueError("TF-CC-RTH gating values do not align")
    cancel = np.empty(n_rows, dtype=np.float64)
    delay = np.empty(n_rows, dtype=np.float64)
    for regime in GATING_LABELS:
        mask = regimes == regime
        if not mask.any():
            continue
        regime_selection = selection["regimes"][regime]["components"]
        cancel[mask] = blend_binary_probabilities(
            {name: components["cancellation"][name][mask] for name in BLEND_CANDIDATES},
            cast(dict[str, float], regime_selection["cancellation"]["weights"]),
        )
        delay[mask] = blend_binary_probabilities(
            {
                name: components["delay_given_operated"][name][mask]
                for name in BLEND_CANDIDATES
            },
            cast(
                dict[str, float],
                regime_selection["delay_given_operated"]["weights"],
            ),
        )
    return hurdle_joint_probabilities(cancel, delay)


def component_pair_grid(
    frame: pd.DataFrame,
    probabilities: Mapping[str, NDArray[np.float64]],
) -> list[dict[str, Any]]:
    """Evaluate every one-hot cancellation/delay component pairing."""

    components = _components(probabilities)
    observed = frame["joint_label_observed"].eq(1).to_numpy(dtype=np.bool_)
    labels = frame.loc[observed, "disruption_state"].to_numpy(dtype=np.int64)
    records: list[dict[str, Any]] = []
    for cancel_source in BLEND_CANDIDATES:
        for delay_source in BLEND_CANDIDATES:
            joint = hurdle_joint_probabilities(
                components["cancellation"][cancel_source],
                components["delay_given_operated"][delay_source],
            )
            log_rows, brier_rows = joint_loss_rows(labels, joint[observed])
            records.append(
                {
                    "method": _pair_name(cancel_source, delay_source),
                    "cancellation_source": cancel_source,
                    "delay_given_operated_source": delay_source,
                    "n": len(labels),
                    "joint_log_loss": float(log_rows.mean()),
                    "multiclass_brier": float(brier_rows.mean()),
                }
            )
    return records


def _pair_name(cancel_source: str, delay_source: str) -> str:
    return f"cancel_{cancel_source}__delay_{delay_source}"


def _pair_probabilities(
    probabilities: Mapping[str, NDArray[np.float64]],
    *,
    cancel_source: str,
    delay_source: str,
) -> NDArray[np.float64]:
    if cancel_source not in BLEND_CANDIDATES or delay_source not in BLEND_CANDIDATES:
        raise ValueError("TF-CC-RTH component-pair source is not registered")
    components = _components(probabilities)
    return hurdle_joint_probabilities(
        components["cancellation"][cancel_source],
        components["delay_given_operated"][delay_source],
    )


def _load_prediction_frame(
    record: dict[str, Any],
    *,
    role: str,
    expected_year: int,
    expected_month: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _verified_artifact(record, role=role)
    methods = (*BLEND_CANDIDATES, "capacity_gated_simplex")
    columns = [
        *PREDICTION_ID_COLUMNS,
        GATING_FEATURE,
        *(column for name in methods for column in _joint_columns(name)),
    ]
    frame = pd.read_parquet(path, columns=columns)
    years = set(pd.to_numeric(frame["Year"], errors="raise").astype(int).unique())
    dates = pd.to_datetime(frame["FlightDate"], errors="raise")
    if years != {expected_year} or set(dates.dt.year.unique()) != {expected_year}:
        raise ValueError(f"{role} crossed its declared year boundary")
    if expected_year == CONFIRMATION_YEAR or (dates.dt.year >= CONFIRMATION_YEAR).any():
        raise ValueError(f"{role} attempted to open confirmation outcomes")
    if expected_month is not None:
        months = set(pd.to_numeric(frame["Month"], errors="raise").astype(int).unique())
        if months != {expected_month} or set(dates.dt.month.unique()) != {expected_month}:
            raise ValueError(f"{role} has the wrong month")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError(f"{role} contains invalid sample ids")
    return frame, {
        "path": path.as_posix(),
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "year": expected_year,
        **({} if expected_month is None else {"month": expected_month}),
    }


def _probability_methods(
    frame: pd.DataFrame,
    *,
    normalization_audits: list[dict[str, Any]],
    role: str,
) -> dict[str, NDArray[np.float64]]:
    return {
        name: _get_joint_columns(
            frame,
            name,
            audit_records=normalization_audits,
            role=role,
        )
        for name in (*BLEND_CANDIDATES, "capacity_gated_simplex")
    }


def _selection_scores(
    frame: pd.DataFrame,
    methods: dict[str, NDArray[np.float64]],
) -> dict[str, Any]:
    return _joint_point_scores(
        frame,
        {name: methods[name] for name in Q4_SELECTION_METHODS},
    )


def run_task_factorized_extension(
    *,
    protocol_path: Path,
    parent_report_path: Path,
    parent_validation_path: Path,
    run_dir: Path,
    q4_lock_path: Path,
    confirmation_lock_path: Path,
    output_path: Path,
    bootstrap_repetitions: int = 2_000,
    seed: int = 20260904,
) -> dict[str, Any]:
    """Run Q4 selection, post-hoc 2025 analysis, and unopened-2026 lock."""

    started = time.perf_counter()
    destinations = (run_dir, q4_lock_path, confirmation_lock_path, output_path)
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite TF-CC-RTH outputs: {existing}")
    protocol, protocol_record = _load_protocol(
        protocol_path,
        repetitions=bootstrap_repetitions,
        seed=seed,
    )
    parent_report, parent_lock, parent_record, validation_record = _verify_parent_inputs(
        parent_report_path,
        parent_validation_path,
    )
    implementation_at_start = capture_provenance(
        _frozen_factorized_implementation_files()
    )
    run_dir.mkdir(parents=True)

    selection_record = dict(parent_report["selection_crossfit_prediction_artifact"])
    selection_frame, verified_selection_record = _load_prediction_frame(
        selection_record,
        role="CC-RTH Q4 2024 cross-fit predictions",
        expected_year=2024,
        expected_month=None,
    )
    selection_normalization: list[dict[str, Any]] = []
    selection_base = _probability_methods(
        selection_frame,
        normalization_audits=selection_normalization,
        role="checksum-loaded Q4 2024 persisted joint probabilities",
    )
    base_only: dict[str, NDArray[np.float64]] = {
        name: selection_base[name] for name in BLEND_CANDIDATES
    }
    global_selection = select_task_factorized_global(selection_frame, base_only)
    cutpoints = parent_lock["capacity_gated_simplex"]["cutpoints"]
    minimum_rows = int(parent_lock["capacity_gated_simplex"]["minimum_rows"])
    gated_selection = select_task_factorized_capacity_gated(
        selection_frame,
        base_only,
        cutpoints=cutpoints,
        minimum_rows=minimum_rows,
        global_selection=global_selection,
    )
    selection_methods = {
        "flare24": selection_base["flare24"],
        "capacity_gated_simplex": selection_base["capacity_gated_simplex"],
        "task_factorized_global_q4": apply_task_factorized_global(
            base_only, global_selection
        ),
        "task_factorized_capacity_gated_q4": apply_task_factorized_capacity_gated(
            base_only,
            selection_frame[GATING_FEATURE].to_numpy(dtype=np.float64),
            gated_selection,
        ),
    }
    selection_scores = _selection_scores(selection_frame, selection_methods)
    selected_q4 = min(
        Q4_SELECTION_METHODS,
        key=lambda name: (
            float(selection_scores[name]["joint_log_loss"]),
            Q4_SELECTION_METHODS.index(name),
        ),
    )
    selection_grid = component_pair_grid(selection_frame, base_only)
    implementation_at_q4_lock = _verify_implementation_unchanged(
        implementation_at_start,
        phase="Q4 factorized lock",
    )
    q4_lock: dict[str, Any] = {
        "schema_version": 1,
        "status": (
            "LOCKED_TF_CCRTH_Q4_WEIGHTS_AFTER_2025_AGGREGATE_RESULTS_WERE_KNOWN_"
            "BEFORE_THIS_RUN_LOADED_ROW_LEVEL_2025_PREDICTIONS"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "TF-CC-RTH-v1",
        "protocol": protocol_record,
        "parent_report": parent_record,
        "parent_validation": validation_record,
        "selection_prediction_artifact": verified_selection_record,
        "global_component_simplexes": global_selection,
        "capacity_gated_component_simplexes": gated_selection,
        "selection_scores": selection_scores,
        "selected_method_by_q4_forward_joint_log_loss": selected_q4,
        "information_boundary": {
            "2024_row_level_forward_predictions_loaded": True,
            "prior_2025_aggregate_results_known_during_hypothesis_formation": True,
            "2025_row_level_predictions_loaded_by_this_run_before_lock": False,
            "2026_outcomes_accessed": False,
        },
        "epistemic_status": (
            "This lock prevents row-level 2025 optimization of the Q4-fitted weights, "
            "but the factorization hypothesis arose after aggregate 2025 CC-RTH results "
            "were inspected. Its 2025 result is retrospective, not blind confirmation."
        ),
        "provenance": implementation_at_q4_lock,
    }
    q4_lock["manifest_sha256"] = canonical_json_sha256(q4_lock)
    _atomic_json(q4_lock_path, q4_lock)
    print(
        "locked TF-CC-RTH Q4 component weights; "
        f"selected={selected_q4}; no 2026 outcomes accessed",
        flush=True,
    )

    retrospective_records = sorted(
        (dict(record) for record in parent_report["retrospective_prediction_artifacts"]),
        key=lambda record: int(record["month"]),
    )
    if [int(record["month"]) for record in retrospective_records] != list(range(1, 13)):
        raise ValueError("parent CC-RTH retrospective artifacts do not cover 2025 months")
    audit_parts: list[pd.DataFrame] = []
    input_audit_records: list[dict[str, Any]] = []
    for record in retrospective_records:
        month = int(record["month"])
        print(f"loading TF-CC-RTH retrospective source 2025-{month:02d}", flush=True)
        frame, verified = _load_prediction_frame(
            record,
            role=f"CC-RTH 2025-{month:02d} predictions",
            expected_year=PRIMARY_YEAR,
            expected_month=month,
        )
        normalization: list[dict[str, Any]] = []
        month_base = _probability_methods(
            frame,
            normalization_audits=normalization,
            role=f"checksum-loaded 2025-{month:02d} persisted joint probabilities",
        )
        base_month_only: dict[str, NDArray[np.float64]] = {
            name: month_base[name] for name in BLEND_CANDIDATES
        }
        _set_joint_columns(
            frame,
            "task_factorized_global_q4",
            apply_task_factorized_global(base_month_only, global_selection),
        )
        _set_joint_columns(
            frame,
            "task_factorized_capacity_gated_q4",
            apply_task_factorized_capacity_gated(
                base_month_only,
                frame[GATING_FEATURE].to_numpy(dtype=np.float64),
                gated_selection,
            ),
        )
        audit_parts.append(frame)
        input_audit_records.append({**verified, "probability_normalization": normalization})
        del month_base, base_month_only, frame
        gc.collect()
    audit_full = pd.concat(audit_parts, ignore_index=True)
    del audit_parts
    if audit_full["sample_id"].duplicated().any():
        raise ValueError("TF-CC-RTH 2025 source repeats sample ids across months")

    assembled_normalization: list[dict[str, Any]] = []
    audit_base = _probability_methods(
        audit_full,
        normalization_audits=assembled_normalization,
        role="assembled persisted 2025 base joint probabilities",
    )
    for name in FACTORIZED_METHODS:
        audit_base[name] = _get_joint_columns(
            audit_full,
            name,
            audit_records=assembled_normalization,
            role="assembled persisted 2025 TF-CC-RTH joint probabilities",
        )
    audit_dates = pd.to_datetime(audit_full["FlightDate"], errors="raise")
    primary_mask = audit_dates.between(
        pd.Timestamp(PRIMARY_AUDIT_START), pd.Timestamp(PRIMARY_AUDIT_END)
    ).to_numpy(dtype=np.bool_)
    audit = audit_full.loc[primary_mask].reset_index(drop=True)
    primary_base: dict[str, NDArray[np.float64]] = {
        name: values[primary_mask]
        for name, values in audit_base.items()
        if name in BLEND_CANDIDATES
    }
    posthoc_grid = component_pair_grid(audit, primary_base)
    posthoc_winner = min(
        posthoc_grid,
        key=lambda record: (
            float(record["joint_log_loss"]),
            float(record["multiclass_brier"]),
            BLEND_CANDIDATES.index(str(record["cancellation_source"])),
            BLEND_CANDIDATES.index(str(record["delay_given_operated_source"])),
        ),
    )
    winner_name = str(posthoc_winner["method"])
    winner_full = _pair_probabilities(
        {name: audit_base[name] for name in BLEND_CANDIDATES},
        cancel_source=str(posthoc_winner["cancellation_source"]),
        delay_source=str(posthoc_winner["delay_given_operated_source"]),
    )
    audit_base[winner_name] = winner_full
    _set_joint_columns(audit_full, winner_name, winner_full)

    evaluation_names = (
        "flare24",
        "capacity_gated_simplex",
        *FACTORIZED_METHODS,
        winner_name,
    )
    full_year_methods = {name: audit_base[name] for name in evaluation_names}
    primary_methods = {
        name: probabilities[primary_mask]
        for name, probabilities in full_year_methods.items()
    }
    retrospective_evaluation = evaluate_joint_probabilities(
        audit,
        cast(dict[str, ArrayLike], primary_methods),
        reference_method="flare24",
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=seed,
    )
    full_year_scores = _joint_point_scores(audit_full, full_year_methods)
    monthly_scores = _monthly_scores(audit, primary_methods)
    regime_scores = _regime_scores(audit, primary_methods, gated_selection)
    airport_scores = _airport_scores(audit, primary_methods)

    output_prediction_records: list[dict[str, Any]] = []
    prediction_columns = [
        "sample_id",
        "FlightDate",
        "Year",
        "Month",
        *(
            column
            for name in (*FACTORIZED_METHODS, winner_name)
            for column in _joint_columns(name)
        ),
    ]
    for month in range(1, 13):
        month_mask = pd.to_numeric(audit_full["Month"], errors="raise").eq(month)
        artifact = _atomic_parquet(
            audit_full.loc[month_mask, prediction_columns].reset_index(drop=True),
            run_dir / "predictions" / f"retrospective_2025_{month:02d}.parquet",
        )
        output_prediction_records.append({"year": PRIMARY_YEAR, "month": month, **artifact})

    confirmation_lock: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED_TF_CCRTH_V1_FOR_UNOPENED_2026_CONFIRMATION",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "TF-CC-RTH-v1",
        "locked_candidate": {
            "name": winner_name,
            "cancellation_source": posthoc_winner["cancellation_source"],
            "delay_given_operated_source": posthoc_winner[
                "delay_given_operated_source"
            ],
            "recombination": protocol["factorization"]["recombination"],
        },
        "selection_disclosure": {
            "selected_from_pairs": len(posthoc_grid),
            "selection_period": [PRIMARY_AUDIT_START, PRIMARY_AUDIT_END],
            "selection_metric": "2025 primary-period joint log loss",
            "2025_result_is_post_hoc": True,
            "reported_intervals_are_not_selection_adjusted": True,
        },
        "confirmation_protocol": {
            "year": CONFIRMATION_YEAR,
            "primary_dates": ["2026-01-03", "2026-12-29"],
            "reference": "exact FLARE-24 pipeline probabilities under the same T-24 contract",
            "primary_metric": "paired date-cluster joint log-loss difference",
            "co_primary_guardrail": "paired date-cluster multiclass Brier difference",
            "success_rule": (
                "both 95% paired date-cluster upper bounds are below zero on the "
                "predeclared primary period"
            ),
            "bootstrap_repetitions": bootstrap_repetitions,
            "bootstrap_seed": seed + 20_000,
            "no_method_or_threshold_refit_on_2026": True,
        },
        "information_boundary": {
            "2024_outcomes_accessed": True,
            "2025_outcomes_accessed_for_exploratory_selection": True,
            "2026_outcomes_accessed": False,
        },
        "parent_report": parent_record,
        "q4_weight_lock": {
            "path": q4_lock_path.as_posix(),
            "sha256": sha256_file(q4_lock_path),
            "self_hash": q4_lock["manifest_sha256"],
        },
        "provenance": _verify_implementation_unchanged(
            implementation_at_start,
            phase="2026 confirmation lock",
        ),
    }
    confirmation_lock["manifest_sha256"] = canonical_json_sha256(confirmation_lock)
    _atomic_json(confirmation_lock_path, confirmation_lock)

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_TF_CCRTH_POST_HOC_2025_ANALYSIS_2026_UNOPENED",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "Task-Factorized CC-RTH (TF-CC-RTH-v1)",
        "protocol": protocol_record,
        "parent_report": parent_record,
        "parent_validation": validation_record,
        "selection_prediction_artifact": verified_selection_record,
        "selection_probability_normalization": selection_normalization,
        "global_component_simplexes": global_selection,
        "capacity_gated_component_simplexes": gated_selection,
        "selection_component_pair_grid": selection_grid,
        "selection_scores": selection_scores,
        "selected_method_by_q4_forward_joint_log_loss": selected_q4,
        "q4_method_lock": {
            "path": q4_lock_path.as_posix(),
            "bytes": q4_lock_path.stat().st_size,
            "sha256": sha256_file(q4_lock_path),
            "self_hash": q4_lock["manifest_sha256"],
        },
        "retrospective_source_artifacts": input_audit_records,
        "assembled_probability_normalization": assembled_normalization,
        "retrospective_prediction_artifacts": output_prediction_records,
        "primary_evaluation_period": [PRIMARY_AUDIT_START, PRIMARY_AUDIT_END],
        "full_year_descriptive_period": ["2025-01-01", "2025-12-31"],
        "retrospective_evaluation": retrospective_evaluation,
        "retrospective_full_year_descriptive_joint_scores": full_year_scores,
        "retrospective_monthly_scores": monthly_scores,
        "retrospective_capacity_regime_scores": regime_scores,
        "retrospective_airport_scores": airport_scores,
        "post_hoc_component_pair_grid": posthoc_grid,
        "post_hoc_component_pair_winner": posthoc_winner,
        "post_hoc_multiplicity_disclosure": (
            "The 2025 winner was selected after evaluating all 25 registered component "
            "pairs. Its paired interval is descriptive and not selection-adjusted; only "
            "the locked 2026 evaluation can provide an independent confirmation."
        ),
        "confirmation_lock": {
            "path": confirmation_lock_path.as_posix(),
            "bytes": confirmation_lock_path.stat().st_size,
            "sha256": sha256_file(confirmation_lock_path),
            "self_hash": confirmation_lock["manifest_sha256"],
            "confirmation_outcomes_accessed": False,
        },
        "outcomes_accessed": {
            "selection_years": [2024],
            "exploratory_evaluation_years": [2025],
            "2026_accessed": False,
        },
        "claim_limits": [
            "The factorization hypothesis was formed after aggregate 2025 CC-RTH results were known.",
            "The 2025 component grid is post-hoc and hypothesis-generating.",
            "Intervals for the 2025 grid winner are not adjusted for selecting among 25 pairs.",
            "No blind-confirmation, causal, physical-capacity, or state-of-the-art claim is made.",
            "The Q4-fitted factorized mixtures did not load row-level 2025 predictions before their lock.",
            "No 2026 outcome-bearing artifact was opened.",
        ],
        "versions": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
        },
        "provenance": _verify_implementation_unchanged(
            implementation_at_start,
            phase="final report",
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    _atomic_json(output_path, report)
    _atomic_json(run_dir / "run_manifest.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--parent-validation", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--q4-lock", type=Path, required=True)
    parser.add_argument("--confirmation-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_task_factorized_extension(
        protocol_path=args.protocol,
        parent_report_path=args.parent_report,
        parent_validation_path=args.parent_validation,
        run_dir=args.run_dir,
        q4_lock_path=args.q4_lock,
        confirmation_lock_path=args.confirmation_lock,
        output_path=args.output,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "q4_selected_method": result[
                    "selected_method_by_q4_forward_joint_log_loss"
                ],
                "post_hoc_2025_winner": result[
                    "post_hoc_component_pair_winner"
                ]["method"],
                "2026_accessed": result["outcomes_accessed"]["2026_accessed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
