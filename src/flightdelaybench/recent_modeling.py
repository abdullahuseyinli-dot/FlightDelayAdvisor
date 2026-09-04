"""Flight-level models using strict prior-day multi-timescale operating history."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .contracts import (
    RECENT_OPERATIONAL_FEATURES,
    RECENT_OPERATIONAL_VIEWS,
    RECENT_WINDOWS_DAYS,
    SCHEDULE_CONTEXT_FEATURES,
    AvailabilityHorizon,
    validate_predictors,
)
from .modeling import (
    CATEGORICAL_FEATURES,
    TaskName,
    engineer_model_features,
    load_feature_year,
)
from .recent import attach_recent_features

DIRECT_OPERATIONAL_VIEWS = (
    "global",
    "route",
    "airline",
    "origin_outbound",
    "dest_inbound",
)


def _selected_recent_inputs(include_cross_direction: bool) -> tuple[str, ...]:
    views = RECENT_OPERATIONAL_VIEWS if include_cross_direction else DIRECT_OPERATIONAL_VIEWS
    return tuple(
        name
        for name in RECENT_OPERATIONAL_FEATURES
        if any(name.startswith(f"recent_{view}_") for view in views)
    )


def engineer_recent_model_features(
    frame: pd.DataFrame,
    *,
    include_cross_direction: bool,
    include_schedule_context: bool = False,
) -> pd.DataFrame:
    """Add multi-scale momentum and network-flow contrasts to base features."""

    recent_inputs = _selected_recent_inputs(include_cross_direction)
    validate_predictors(recent_inputs, AvailabilityHorizon.FORECAST_24H)
    missing = sorted(set(recent_inputs) - set(frame.columns))
    if missing:
        raise ValueError(f"prior-day model is missing registered inputs: {missing}")
    base = engineer_model_features(frame)
    recent = frame.loc[:, list(recent_inputs)].apply(pd.to_numeric, errors="raise").astype("float32")
    if not np.isfinite(recent.to_numpy()).all():
        raise ValueError("prior-day model inputs contain non-finite values")

    derived: dict[str, pd.Series | NDArray[np.float32]] = {}
    active_views = (
        RECENT_OPERATIONAL_VIEWS if include_cross_direction else DIRECT_OPERATIONAL_VIEWS
    )
    for view in active_views:
        for outcome in ("delay", "cancel"):
            rate_7 = recent[f"recent_{view}_{outcome}_rate_7d"]
            rate_28 = recent[f"recent_{view}_{outcome}_rate_28d"]
            rate_90 = recent[f"recent_{view}_{outcome}_rate_90d"]
            derived[f"recent_{view}_{outcome}_momentum_7v90"] = (rate_7 - rate_90).astype(
                "float32"
            )
            derived[f"recent_{view}_{outcome}_momentum_28v90"] = (
                rate_28 - rate_90
            ).astype("float32")
            derived[f"recent_{view}_{outcome}_acceleration"] = (
                rate_7 - 2.0 * rate_28 + rate_90
            ).astype("float32")

    for level in ("route", "airline"):
        for outcome in ("delay", "cancel"):
            for window in RECENT_WINDOWS_DAYS:
                derived[f"recent_{level}_{outcome}_excess_global_{window}d"] = (
                    recent[f"recent_{level}_{outcome}_rate_{window}d"]
                    - recent[f"recent_global_{outcome}_rate_{window}d"]
                ).astype("float32")

    for outcome in ("delay", "cancel"):
        for window in RECENT_WINDOWS_DAYS:
            origin_outbound = recent[
                f"recent_origin_outbound_{outcome}_rate_{window}d"
            ]
            dest_inbound = recent[f"recent_dest_inbound_{outcome}_rate_{window}d"]
            derived[f"recent_endpoints_{outcome}_mean_{window}d"] = (
                (origin_outbound + dest_inbound) / 2.0
            ).astype("float32")
            derived[f"recent_endpoints_{outcome}_max_{window}d"] = np.maximum(
                origin_outbound, dest_inbound
            ).astype("float32")
            derived[f"recent_endpoints_{outcome}_gap_{window}d"] = (
                origin_outbound - dest_inbound
            ).astype("float32")
            if include_cross_direction:
                origin_inbound = recent[
                    f"recent_origin_inbound_{outcome}_rate_{window}d"
                ]
                dest_outbound = recent[
                    f"recent_dest_outbound_{outcome}_rate_{window}d"
                ]
                derived[f"recent_origin_flow_pressure_{outcome}_{window}d"] = (
                    origin_inbound - origin_outbound
                ).astype("float32")
                derived[f"recent_dest_flow_pressure_{outcome}_{window}d"] = (
                    dest_inbound - dest_outbound
                ).astype("float32")
                derived[f"recent_network_inbound_max_{outcome}_{window}d"] = np.maximum(
                    origin_inbound, dest_inbound
                ).astype("float32")

    schedule: pd.DataFrame | None = None
    if include_schedule_context:
        validate_predictors(SCHEDULE_CONTEXT_FEATURES, AvailabilityHorizon.FORECAST_24H)
        missing_schedule = sorted(set(SCHEDULE_CONTEXT_FEATURES) - set(frame.columns))
        if missing_schedule:
            raise ValueError(f"schedule-context model is missing inputs: {missing_schedule}")
        schedule = (
            frame.loc[:, list(SCHEDULE_CONTEXT_FEATURES)]
            .apply(pd.to_numeric, errors="raise")
            .astype("float32")
        )
        schedule_recent_pairs = {
            "global": "schedule_global_day_log1p",
            "airline": "schedule_airline_day_log1p",
            "route": "schedule_route_day_log1p",
            "origin_outbound": "schedule_origin_outbound_day_log1p",
            "dest_inbound": "schedule_dest_inbound_day_log1p",
        }
        for view, schedule_column in schedule_recent_pairs.items():
            for window in RECENT_WINDOWS_DAYS:
                recent_count = np.expm1(recent[f"recent_{view}_count_log1p_{window}d"])
                expected_daily_log = np.log1p(recent_count / float(window))
                derived[f"schedule_{view}_surge_vs_{window}d"] = (
                    schedule[schedule_column] - expected_daily_log
                ).astype("float32")

    components = [base, recent]
    if schedule is not None:
        components.append(schedule)
    components.append(pd.DataFrame(derived, index=base.index))
    output = pd.concat(components, axis=1, copy=False)
    numeric = output.drop(columns=list(CATEGORICAL_FEATURES)).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("engineered prior-day model features contain non-finite values")
    return output


def load_recent_feature_year(
    feature_dir: Path,
    recent_dir: Path,
    year: int,
    *,
    limit: int | None = None,
    seed: int = 20260903,
) -> pd.DataFrame:
    frame = load_feature_year(feature_dir, year, limit=limit, seed=seed)
    return attach_recent_features(frame, recent_dir=recent_dir)


def load_recent_feature_years(
    feature_dir: Path,
    recent_dir: Path,
    years: tuple[int, ...],
    *,
    rows_per_year: int | None = None,
    seed: int = 20260903,
) -> pd.DataFrame:
    if not years:
        raise ValueError("at least one recent-feature year is required")
    return pd.concat(
        [
            load_recent_feature_year(
                feature_dir,
                recent_dir,
                year,
                limit=rows_per_year,
                seed=seed,
            )
            for year in years
        ],
        ignore_index=True,
    )


@dataclass(slots=True)
class RecentCatBoostModel:
    estimator: Any
    task: TaskName
    include_cross_direction: bool
    include_schedule_context: bool = False

    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        matrix = engineer_recent_model_features(
            frame,
            include_cross_direction=self.include_cross_direction,
            include_schedule_context=self.include_schedule_context,
        )
        for column in CATEGORICAL_FEATURES:
            matrix[column] = matrix[column].astype(str)
        return np.asarray(self.estimator.predict_proba(matrix), dtype=np.float64)


def fit_recent_catboost(
    train_frame: pd.DataFrame,
    train_labels: NDArray[np.int64],
    *,
    task: TaskName,
    params: dict[str, Any],
    include_cross_direction: bool,
    include_schedule_context: bool = False,
    validation_frame: pd.DataFrame | None = None,
    validation_labels: NDArray[np.int64] | None = None,
    sample_weight: NDArray[np.float64] | None = None,
) -> RecentCatBoostModel:
    from catboost import CatBoostClassifier  # type: ignore[import-untyped]

    train_matrix = engineer_recent_model_features(
        train_frame,
        include_cross_direction=include_cross_direction,
        include_schedule_context=include_schedule_context,
    )
    for column in CATEGORICAL_FEATURES:
        train_matrix[column] = train_matrix[column].astype(str)
    defaults: dict[str, Any] = {
        "loss_function": "MultiClass" if task == "joint" else "Logloss",
        "eval_metric": "MultiClass" if task == "joint" else "Logloss",
        "iterations": 1_200,
        "learning_rate": 0.04,
        "depth": 8,
        "l2_leaf_reg": 6.0,
        "random_seed": 20260903,
        "task_type": "GPU",
        "devices": "0",
        "verbose": False,
        "allow_writing_files": False,
    }
    defaults.update(params)
    estimator = CatBoostClassifier(**defaults)
    fit_kwargs: dict[str, Any] = {
        "X": train_matrix,
        "y": train_labels,
        "cat_features": list(CATEGORICAL_FEATURES),
        "sample_weight": sample_weight,
    }
    if validation_frame is not None and validation_labels is not None:
        valid_matrix = engineer_recent_model_features(
            validation_frame,
            include_cross_direction=include_cross_direction,
            include_schedule_context=include_schedule_context,
        )
        for column in CATEGORICAL_FEATURES:
            valid_matrix[column] = valid_matrix[column].astype(str)
        fit_kwargs["eval_set"] = (valid_matrix, validation_labels)
        fit_kwargs["early_stopping_rounds"] = 100
        fit_kwargs["use_best_model"] = True
    estimator.fit(**fit_kwargs)
    return RecentCatBoostModel(
        estimator=estimator,
        task=task,
        include_cross_direction=include_cross_direction,
        include_schedule_context=include_schedule_context,
    )


def recent_model_profile(
    include_cross_direction: bool,
    include_schedule_context: bool = False,
) -> dict[str, Any]:
    return {
        "availability_horizon": AvailabilityHorizon.FORECAST_24H.name,
        "weather_forecast_included": False,
        "historical_feed_proxy": "BTS outcomes through the previous operating day",
        "cutoff": "target day excluded",
        "windows_days": list(RECENT_WINDOWS_DAYS),
        "views": list(
            RECENT_OPERATIONAL_VIEWS
            if include_cross_direction
            else DIRECT_OPERATIONAL_VIEWS
        ),
        "cross_direction_network_features": include_cross_direction,
        "schedule_context_features": include_schedule_context,
        "claim_limit": "requires a live-feed reporting-lag validation before deployment",
    }
