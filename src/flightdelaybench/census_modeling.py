"""Models for the official BTS top-airport census and rich schedule track."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .census_graph import attach_census_graph_features
from .census_recent import attach_census_recent_features
from .contracts import (
    CENSUS_FLIGHT_RECENT_FEATURES,
    CENSUS_GRAPH_MESSAGE_FEATURES,
    CENSUS_RICH_SCHEDULE_FEATURES,
    RECENT_OPERATIONAL_FEATURES,
    RECENT_OPERATIONAL_VIEWS,
    RECENT_WINDOWS_DAYS,
    SCHEDULE_CONTEXT_FEATURES,
    AvailabilityHorizon,
    validate_predictors,
)
from .modeling import CATEGORICAL_FEATURES, TaskName
from .recent_modeling import DIRECT_OPERATIONAL_VIEWS
from .schedule_context import attach_schedule_context

CENSUS_LABEL_COLUMNS = (
    "sample_id",
    "FlightDate",
    "ArrDel15",
    "Cancelled",
    "Diverted",
    "delay_label_observed",
    "joint_label_observed",
    "disruption_state",
)
CENSUS_BASE_SCHEDULE_FEATURES = (
    "Year",
    "Month",
    "DayOfMonth",
    "DayOfWeek",
    "DayOfYear",
    "DepHour",
    "DepHour_sin",
    "DepHour_cos",
    "Month_sin",
    "Month_cos",
    "IsWeekend",
    "IsHolidaySeason",
    "Distance",
)
CENSUS_BASE_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES
CENSUS_FLIGHT_CATEGORICAL_FEATURES = ("ScheduledFlightId",)
CENSUS_RICH_NUMERIC_FEATURES = tuple(
    name for name in CENSUS_RICH_SCHEDULE_FEATURES if name != "ScheduledFlightId"
)
CENSUS_LOAD_COLUMNS = tuple(
    dict.fromkeys(
        [
            *CENSUS_LABEL_COLUMNS,
            *CENSUS_BASE_SCHEDULE_FEATURES,
            *CENSUS_BASE_CATEGORICAL_FEATURES,
            *CENSUS_RICH_SCHEDULE_FEATURES,
        ]
    )
)


def _census_year_files(census_dir: Path, year: int) -> tuple[Path, ...]:
    files = tuple(sorted((census_dir / f"year={year}").glob("month=*.parquet")))
    if len(files) != 12:
        raise FileNotFoundError(f"expected 12 census partitions for {year}, found {len(files)}")
    return files


def load_census_year(
    census_dir: Path,
    recent_dir: Path,
    flight_recent_dir: Path,
    year: int,
    *,
    limit: int | None,
    graph_dir: Path | None = None,
    seed: int = 20260903,
) -> pd.DataFrame:
    """Compute full monthly schedule context before reproducible month-stratified sampling."""

    files = _census_year_files(census_dir, year)
    allocations: list[int | None]
    if limit is None:
        allocations = [None] * 12
    else:
        if limit < 12:
            raise ValueError("census sampling limit must be at least 12")
        quotient, remainder = divmod(limit, 12)
        allocations = [quotient + int(index < remainder) for index in range(12)]
    parts: list[pd.DataFrame] = []
    for month_index, (path, allocation) in enumerate(zip(files, allocations, strict=True), start=1):
        frame = pd.read_parquet(path, columns=list(CENSUS_LOAD_COLUMNS))
        if not frame["Year"].eq(year).all() or not frame["Month"].eq(month_index).all():
            raise ValueError(f"census partition period mismatch: {path}")
        contextual = attach_schedule_context(frame)
        if allocation is not None and len(contextual) > allocation:
            contextual = contextual.sample(
                n=allocation,
                replace=False,
                random_state=seed + year * 100 + month_index,
            )
        parts.append(contextual.reset_index(drop=True))
    sampled = pd.concat(parts, ignore_index=True)
    if limit is not None and len(sampled) != limit:
        raise ValueError(f"census year {year} could not satisfy exact sampling limit {limit}")
    for column in CENSUS_BASE_CATEGORICAL_FEATURES + CENSUS_FLIGHT_CATEGORICAL_FEATURES:
        sampled[column] = sampled[column].astype("string").fillna("__MISSING__")
    attached = attach_census_recent_features(
        sampled,
        recent_dir=recent_dir,
        flight_recent_dir=flight_recent_dir,
    )
    if graph_dir is not None:
        attached = attach_census_graph_features(attached, graph_dir=graph_dir)
    return attached


def load_census_years(
    census_dir: Path,
    recent_dir: Path,
    flight_recent_dir: Path,
    years: tuple[int, ...],
    *,
    rows_per_year: int | None,
    graph_dir: Path | None = None,
    seed: int = 20260903,
) -> pd.DataFrame:
    if not years:
        raise ValueError("at least one census year is required")
    return pd.concat(
        [
            load_census_year(
                census_dir,
                recent_dir,
                flight_recent_dir,
                year,
                limit=rows_per_year,
                graph_dir=graph_dir,
                seed=seed,
            )
            for year in years
        ],
        ignore_index=True,
    )


def _recent_inputs(include_cross_direction: bool) -> tuple[str, ...]:
    views = RECENT_OPERATIONAL_VIEWS if include_cross_direction else DIRECT_OPERATIONAL_VIEWS
    return tuple(
        name
        for name in RECENT_OPERATIONAL_FEATURES
        if any(name.startswith(f"recent_{view}_") for view in views)
    )


def census_model_input_columns(
    *,
    include_cross_direction: bool,
    include_rich_schedule: bool,
    include_flight_history: bool,
    include_schedule_context: bool,
    include_graph_pressure: bool = False,
) -> tuple[str, ...]:
    """Return the raw columns needed to engineer a census model matrix.

    Keeping this projection explicit lets large retrospective studies discard joined
    columns that a particular estimator never consumes before applying row masks.
    """

    registered: list[str] = [
        *CENSUS_BASE_SCHEDULE_FEATURES,
        *CENSUS_BASE_CATEGORICAL_FEATURES,
    ]
    if include_rich_schedule:
        registered.extend(CENSUS_RICH_NUMERIC_FEATURES)
    registered.extend(_recent_inputs(include_cross_direction))
    if include_flight_history:
        registered.extend(CENSUS_FLIGHT_RECENT_FEATURES)
        registered.append("ScheduledFlightId")
    if include_schedule_context:
        registered.extend(SCHEDULE_CONTEXT_FEATURES)
    if include_graph_pressure:
        registered.extend(CENSUS_GRAPH_MESSAGE_FEATURES)
    columns = tuple(dict.fromkeys(registered))
    validate_predictors(columns, AvailabilityHorizon.FORECAST_24H)
    return columns


def engineer_census_features(
    frame: pd.DataFrame,
    *,
    include_cross_direction: bool,
    include_rich_schedule: bool,
    include_flight_history: bool,
    include_schedule_context: bool,
    include_graph_pressure: bool = False,
) -> pd.DataFrame:
    """Engineer only registered schedule or closed-left operating-history inputs."""

    categorical = list(CENSUS_BASE_CATEGORICAL_FEATURES)
    if include_flight_history:
        categorical.append("ScheduledFlightId")
    registered = census_model_input_columns(
        include_cross_direction=include_cross_direction,
        include_rich_schedule=include_rich_schedule,
        include_flight_history=include_flight_history,
        include_schedule_context=include_schedule_context,
        include_graph_pressure=include_graph_pressure,
    )
    missing = sorted(set(registered) - set(frame.columns))
    if missing:
        raise ValueError(f"census model is missing registered features: {missing}")
    matrix = frame.loc[:, list(registered)].copy()
    for column in categorical:
        matrix[column] = matrix[column].astype("string").fillna("__MISSING__")
    for column in set(matrix) - set(categorical):
        matrix[column] = pd.to_numeric(matrix[column], errors="raise").astype("float32")

    derived: dict[str, pd.Series | NDArray[np.float32]] = {}
    annual_phase = 2.0 * np.pi * (matrix["DayOfYear"] - 1.0) / 365.25
    for harmonic in (1, 2, 3, 4, 6, 12):
        derived[f"annual_fourier_sin_{harmonic}"] = np.sin(harmonic * annual_phase).astype(
            "float32"
        )
        derived[f"annual_fourier_cos_{harmonic}"] = np.cos(harmonic * annual_phase).astype(
            "float32"
        )
    derived["distance_log1p"] = np.log1p(matrix["Distance"].clip(lower=0)).astype("float32")
    if include_rich_schedule:
        elapsed_missing = matrix["CRSElapsedTime"].isna()
        elapsed = matrix["CRSElapsedTime"].fillna(
            30.0 + matrix["Distance"].clip(lower=0) / 8.0
        ).clip(lower=1)
        matrix["CRSElapsedTime"] = elapsed.astype("float32")
        derived["scheduled_block_missing"] = elapsed_missing.astype("float32")
        derived["scheduled_block_per_100mi"] = (
            elapsed / matrix["Distance"].clip(lower=50) * 100.0
        ).astype("float32")
        derived["scheduled_block_physics_residual"] = (
            elapsed - (30.0 + matrix["Distance"] / 8.0)
        ).astype("float32")
        derived["scheduled_crosses_midnight_clock"] = (
            matrix["CRSArrMinutes"] < matrix["CRSDepMinutes"]
        ).astype("float32")
        derived["scheduled_red_eye"] = (
            matrix["DepHour"].ge(21) | matrix["DepHour"].le(4)
        ).astype("float32")

    active_views = (
        RECENT_OPERATIONAL_VIEWS if include_cross_direction else DIRECT_OPERATIONAL_VIEWS
    )
    for view in active_views:
        for outcome in ("delay", "cancel"):
            rate_7 = matrix[f"recent_{view}_{outcome}_rate_7d"]
            rate_28 = matrix[f"recent_{view}_{outcome}_rate_28d"]
            rate_90 = matrix[f"recent_{view}_{outcome}_rate_90d"]
            derived[f"recent_{view}_{outcome}_momentum_7v90"] = (rate_7 - rate_90).astype(
                "float32"
            )
            derived[f"recent_{view}_{outcome}_acceleration"] = (
                rate_7 - 2.0 * rate_28 + rate_90
            ).astype("float32")
    if include_flight_history:
        for outcome in ("delay", "cancel"):
            for window in RECENT_WINDOWS_DAYS:
                derived[f"recent_flight_{outcome}_excess_global_{window}d"] = (
                    matrix[f"recent_flight_{outcome}_rate_{window}d"]
                    - matrix[f"recent_global_{outcome}_rate_{window}d"]
                ).astype("float32")
            derived[f"recent_flight_{outcome}_momentum_7v90"] = (
                matrix[f"recent_flight_{outcome}_rate_7d"]
                - matrix[f"recent_flight_{outcome}_rate_90d"]
            ).astype("float32")
    if include_schedule_context:
        schedule_recent_pairs = {
            "global": "schedule_global_day_log1p",
            "airline": "schedule_airline_day_log1p",
            "route": "schedule_route_day_log1p",
            "origin_outbound": "schedule_origin_outbound_day_log1p",
            "dest_inbound": "schedule_dest_inbound_day_log1p",
        }
        for view, schedule_column in schedule_recent_pairs.items():
            for window in RECENT_WINDOWS_DAYS:
                expected = np.log1p(
                    np.expm1(matrix[f"recent_{view}_count_log1p_{window}d"])
                    / float(window)
                )
                derived[f"census_schedule_{view}_surge_vs_{window}d"] = (
                    matrix[schedule_column] - expected
                ).astype("float32")
    if include_graph_pressure:
        own_views = {"origin": "origin_outbound", "dest": "dest_inbound"}
        for side, own_view in own_views.items():
            for outcome in ("delay", "cancel"):
                for window in RECENT_WINDOWS_DAYS:
                    own = matrix[f"recent_{own_view}_{outcome}_rate_{window}d"]
                    mean = matrix[f"graph_{side}_partner_{outcome}_mean_{window}d"]
                    maximum = matrix[f"graph_{side}_partner_{outcome}_max_{window}d"]
                    derived[f"graph_{side}_{outcome}_mean_minus_local_{window}d"] = (
                        mean - own
                    ).astype("float32")
                    derived[f"graph_{side}_{outcome}_max_minus_local_{window}d"] = (
                        maximum - own
                    ).astype("float32")
                    derived[f"graph_{side}_{outcome}_tail_spread_{window}d"] = (
                        maximum - mean
                    ).astype("float32")
    output = pd.concat([matrix, pd.DataFrame(derived, index=matrix.index)], axis=1, copy=False)
    numeric = output.drop(columns=categorical).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("census model features contain non-finite values")
    return output


@dataclass(slots=True)
class CensusCatBoostModel:
    estimator: Any
    task: TaskName
    include_cross_direction: bool
    include_rich_schedule: bool
    include_flight_history: bool
    include_schedule_context: bool
    include_graph_pressure: bool = False

    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        matrix = engineer_census_features(
            frame,
            include_cross_direction=self.include_cross_direction,
            include_rich_schedule=self.include_rich_schedule,
            include_flight_history=self.include_flight_history,
            include_schedule_context=self.include_schedule_context,
            include_graph_pressure=self.include_graph_pressure,
        )
        categorical = list(CENSUS_BASE_CATEGORICAL_FEATURES)
        if self.include_flight_history:
            categorical.extend(CENSUS_FLIGHT_CATEGORICAL_FEATURES)
        for column in categorical:
            matrix[column] = matrix[column].astype(str)
        return np.asarray(self.estimator.predict_proba(matrix), dtype=np.float64)


def fit_census_catboost(
    train_frame: pd.DataFrame,
    train_labels: NDArray[np.int64],
    *,
    task: TaskName,
    params: dict[str, Any],
    include_cross_direction: bool,
    include_rich_schedule: bool,
    include_flight_history: bool,
    include_schedule_context: bool,
    include_graph_pressure: bool = False,
    validation_frame: pd.DataFrame | None = None,
    validation_labels: NDArray[np.int64] | None = None,
) -> CensusCatBoostModel:
    from catboost import CatBoostClassifier  # type: ignore[import-untyped]

    train_matrix = engineer_census_features(
        train_frame,
        include_cross_direction=include_cross_direction,
        include_rich_schedule=include_rich_schedule,
        include_flight_history=include_flight_history,
        include_schedule_context=include_schedule_context,
        include_graph_pressure=include_graph_pressure,
    )
    categorical = list(CENSUS_BASE_CATEGORICAL_FEATURES)
    if include_flight_history:
        categorical.extend(CENSUS_FLIGHT_CATEGORICAL_FEATURES)
    for column in categorical:
        train_matrix[column] = train_matrix[column].astype(str)
    defaults: dict[str, Any] = {
        "loss_function": "MultiClass" if task == "joint" else "Logloss",
        "eval_metric": "MultiClass" if task == "joint" else "Logloss",
        "iterations": 1_200,
        "learning_rate": 0.04,
        "depth": 8,
        "l2_leaf_reg": 8.0,
        "random_seed": 20260903,
        "task_type": "GPU",
        "devices": "0",
        "verbose": False,
        "allow_writing_files": False,
    }
    defaults.update(params)
    if defaults.get("task_type") == "CPU":
        defaults.pop("devices", None)
    estimator = CatBoostClassifier(**defaults)
    fit_kwargs: dict[str, Any] = {
        "X": train_matrix,
        "y": train_labels,
        "cat_features": categorical,
    }
    if validation_frame is not None or validation_labels is not None:
        if validation_frame is None or validation_labels is None:
            raise ValueError("census validation frame and labels must be supplied together")
        validation_matrix = engineer_census_features(
            validation_frame,
            include_cross_direction=include_cross_direction,
            include_rich_schedule=include_rich_schedule,
            include_flight_history=include_flight_history,
            include_schedule_context=include_schedule_context,
            include_graph_pressure=include_graph_pressure,
        )
        for column in categorical:
            validation_matrix[column] = validation_matrix[column].astype(str)
        fit_kwargs.update(
            {
                "eval_set": (validation_matrix, validation_labels),
                "early_stopping_rounds": 100,
                "use_best_model": True,
            }
        )
    estimator.fit(**fit_kwargs)
    return CensusCatBoostModel(
        estimator=estimator,
        task=task,
        include_cross_direction=include_cross_direction,
        include_rich_schedule=include_rich_schedule,
        include_flight_history=include_flight_history,
        include_schedule_context=include_schedule_context,
        include_graph_pressure=include_graph_pressure,
    )


def census_model_profile(
    *,
    include_cross_direction: bool,
    include_rich_schedule: bool,
    include_flight_history: bool,
    include_schedule_context: bool,
    include_graph_pressure: bool = False,
) -> dict[str, Any]:
    return {
        "availability_horizon": AvailabilityHorizon.FORECAST_24H.name,
        "source": "official monthly BTS top-100 census track",
        "cross_direction_network_history": include_cross_direction,
        "rich_schedule": include_rich_schedule,
        "flight_number_identity_and_history": include_flight_history,
        "full_cohort_schedule_context": include_schedule_context,
        "closed_left_schedule_graph_messages": include_graph_pressure,
        "annual_fourier_harmonics": [1, 2, 3, 4, 6, 12],
        "schedule_context_claim_limit": (
            "All retrospective cohort rows, including diversions, contribute to counts; "
            "equivalence to an actual advance schedule snapshot is not claimed."
        ),
        "flight_number_claim_limit": (
            "Published flight number is used; tail assignment and realised rotation state are not."
        ),
    }
