"""Model adapters for leakage-controlled FLARE-24 hurdle prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .census_modeling import (
    CENSUS_BASE_CATEGORICAL_FEATURES,
    CENSUS_FLIGHT_CATEGORICAL_FEATURES,
    engineer_census_features,
)
from .contracts import (
    FLARE24_AVIATION_WEATHER_FEATURES,
    FLARE24_CORRIDOR_FEATURES,
    FLARE24_ROTATION_FEATURES,
    FLARE24_WEATHER_FEATURES,
    AvailabilityHorizon,
    validate_predictors,
)
from .flare_reconciliation import hurdle_joint_probabilities
from .modeling import TaskName

FlareTask = Literal["delay", "cancellation"]
FLARE24_BASE_EXTRA_FEATURES = (
    *FLARE24_WEATHER_FEATURES,
    *FLARE24_AVIATION_WEATHER_FEATURES,
    *FLARE24_CORRIDOR_FEATURES,
)
FLARE24_ALL_EXTRA_FEATURES = (
    *FLARE24_BASE_EXTRA_FEATURES,
    *FLARE24_ROTATION_FEATURES,
)


def attach_flare_feature_partitions(
    frame: pd.DataFrame,
    *,
    feature_dir: Path,
    feature_columns: tuple[str, ...] = FLARE24_BASE_EXTRA_FEATURES,
) -> pd.DataFrame:
    """Attach sample-id keyed FLARE feature partitions without changing row order."""

    required = {"sample_id", "Year", "Month"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"FLARE feature join input is missing columns: {missing}")
    validate_predictors(feature_columns, AvailabilityHorizon.FORECAST_24H)
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("FLARE feature join requires unique non-missing sample_id values")
    if set(feature_columns) & set(frame.columns):
        overlap = sorted(set(feature_columns) & set(frame.columns))
        raise ValueError(f"FLARE features are already present: {overlap}")
    requested_ids = frame.loc[:, ["sample_id", "Year", "Month"]].copy()
    requested_ids["_row_order"] = np.arange(len(frame), dtype=np.int64)
    attached_parts: list[pd.DataFrame] = []
    for (year_value, month_value), rows in requested_ids.groupby(
        ["Year", "Month"], sort=True, observed=True
    ):
        year = int(year_value)
        month = int(month_value)
        path = feature_dir / f"year={year}" / f"month={month:02d}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing FLARE feature partition: {path}")
        partition = pd.read_parquet(path, columns=["sample_id", *feature_columns])
        if partition["sample_id"].duplicated().any():
            raise ValueError(f"duplicate sample_id values in FLARE partition: {path}")
        selected = rows.loc[:, ["sample_id", "_row_order"]].merge(
            partition,
            how="left",
            on="sample_id",
            sort=False,
            validate="one_to_one",
            indicator=True,
        )
        if not selected["_merge"].eq("both").all():
            missing_count = int(selected["_merge"].ne("both").sum())
            raise ValueError(f"{missing_count} sample ids are absent from {path}")
        attached_parts.append(selected.drop(columns="_merge"))
    attached = (
        pd.concat(attached_parts, ignore_index=True)
        .sort_values("_row_order", kind="mergesort")
        .drop(columns=["sample_id", "_row_order"])
        .reset_index(drop=True)
    )
    result = frame.reset_index(drop=True).copy()
    return pd.concat([result, attached], axis=1, copy=False)


def select_usable_extra_features(
    training_frame: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    minimum_nonmissing_fraction: float = 0.01,
    minimum_unique_values: int = 2,
) -> tuple[str, ...]:
    """Select availability, never target association, using training covariates only."""

    if not 0.0 <= minimum_nonmissing_fraction <= 1.0:
        raise ValueError("minimum_nonmissing_fraction must be in [0, 1]")
    if minimum_unique_values < 1:
        raise ValueError("minimum_unique_values must be positive")
    validate_predictors(candidates, AvailabilityHorizon.FORECAST_24H)
    missing = sorted(set(candidates) - set(training_frame.columns))
    if missing:
        raise ValueError(f"training frame is missing FLARE candidates: {missing}")
    selected: list[str] = []
    for feature in candidates:
        values = pd.to_numeric(training_frame[feature], errors="raise")
        fraction = float(values.notna().mean())
        unique = int(values.dropna().nunique())
        if fraction >= minimum_nonmissing_fraction and unique >= minimum_unique_values:
            selected.append(feature)
    if not selected:
        raise ValueError("no usable FLARE extra features remain on training covariates")
    return tuple(selected)


def engineer_flare_matrix(
    frame: pd.DataFrame,
    *,
    extra_features: tuple[str, ...],
    include_cross_direction: bool = True,
    include_rich_schedule: bool = True,
    include_flight_history: bool = True,
    include_schedule_context: bool = True,
    include_graph_pressure: bool = True,
) -> pd.DataFrame:
    """Combine the strongest census baseline with registered FLARE covariates."""

    validate_predictors(extra_features, AvailabilityHorizon.FORECAST_24H)
    missing = sorted(set(extra_features) - set(frame.columns))
    if missing:
        raise ValueError(f"FLARE model input is missing extra features: {missing}")
    base = engineer_census_features(
        frame,
        include_cross_direction=include_cross_direction,
        include_rich_schedule=include_rich_schedule,
        include_flight_history=include_flight_history,
        include_schedule_context=include_schedule_context,
        include_graph_pressure=include_graph_pressure,
    )
    extra = frame.loc[:, list(extra_features)].copy()
    for feature in extra_features:
        extra[feature] = pd.to_numeric(extra[feature], errors="raise").astype("float32")
    numeric = extra.to_numpy(dtype=np.float64)
    if np.isinf(numeric).any():
        raise ValueError("FLARE extra features contain infinity")
    return pd.concat([base.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)


@dataclass(slots=True)
class FlareCatBoostModel:
    estimator: Any
    task: FlareTask
    extra_features: tuple[str, ...]
    include_cross_direction: bool = True
    include_rich_schedule: bool = True
    include_flight_history: bool = True
    include_schedule_context: bool = True
    include_graph_pressure: bool = True

    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        matrix = engineer_flare_matrix(
            frame,
            extra_features=self.extra_features,
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
        probabilities = np.asarray(self.estimator.predict_proba(matrix), dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise RuntimeError("FLARE binary estimator returned invalid probabilities")
        return probabilities[:, 1]


def fit_flare_catboost(
    train_frame: pd.DataFrame,
    train_labels: ArrayLike,
    *,
    task: FlareTask,
    extra_features: tuple[str, ...],
    params: dict[str, Any],
    validation_frame: pd.DataFrame | None = None,
    validation_labels: ArrayLike | None = None,
    include_cross_direction: bool = True,
    include_rich_schedule: bool = True,
    include_flight_history: bool = True,
    include_schedule_context: bool = True,
    include_graph_pressure: bool = True,
) -> FlareCatBoostModel:
    """Fit a binary FLARE component with optional chronological early stopping."""

    from catboost import CatBoostClassifier  # type: ignore[import-untyped]

    if task not in {"delay", "cancellation"}:
        raise ValueError(f"FLARE hurdle task must be binary: {task}")
    labels = np.asarray(train_labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) != len(train_frame) or not np.isin(labels, [0, 1]).all():
        raise ValueError("FLARE training labels must be aligned and binary")
    matrix = engineer_flare_matrix(
        train_frame,
        extra_features=extra_features,
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
        matrix[column] = matrix[column].astype(str)
    defaults: dict[str, Any] = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 1_500,
        "learning_rate": 0.035,
        "depth": 9,
        "l2_leaf_reg": 10.0,
        "random_strength": 0.2,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "border_count": 128,
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
        "X": matrix,
        "y": labels,
        "cat_features": categorical,
    }
    if validation_frame is not None or validation_labels is not None:
        if validation_frame is None or validation_labels is None:
            raise ValueError("FLARE validation frame and labels must be supplied together")
        valid_y = np.asarray(validation_labels, dtype=np.int64)
        if (
            valid_y.ndim != 1
            or len(valid_y) != len(validation_frame)
            or not np.isin(valid_y, [0, 1]).all()
        ):
            raise ValueError("FLARE validation labels must be aligned and binary")
        valid_matrix = engineer_flare_matrix(
            validation_frame,
            extra_features=extra_features,
            include_cross_direction=include_cross_direction,
            include_rich_schedule=include_rich_schedule,
            include_flight_history=include_flight_history,
            include_schedule_context=include_schedule_context,
            include_graph_pressure=include_graph_pressure,
        )
        for column in categorical:
            valid_matrix[column] = valid_matrix[column].astype(str)
        fit_kwargs.update(
            {
                "eval_set": (valid_matrix, valid_y),
                "early_stopping_rounds": 150,
                "use_best_model": True,
            }
        )
    estimator.fit(**fit_kwargs)
    return FlareCatBoostModel(
        estimator=estimator,
        task=task,
        extra_features=extra_features,
        include_cross_direction=include_cross_direction,
        include_rich_schedule=include_rich_schedule,
        include_flight_history=include_flight_history,
        include_schedule_context=include_schedule_context,
        include_graph_pressure=include_graph_pressure,
    )


@dataclass(slots=True)
class FlareHurdleModel:
    cancellation_model: FlareCatBoostModel
    conditional_delay_model: FlareCatBoostModel

    def predict_joint(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        cancellation = self.cancellation_model.predict_proba(frame)
        conditional_delay = self.conditional_delay_model.predict_proba(frame)
        return hurdle_joint_probabilities(cancellation, conditional_delay)


def task_frame(
    frame: pd.DataFrame,
    task: TaskName,
) -> tuple[pd.DataFrame, NDArray[np.int64]]:
    """Create the hurdle task view while retaining all attached feature columns."""

    if task == "cancellation":
        mask = frame["Cancelled"].isin([0, 1])
        labels = frame.loc[mask, "Cancelled"]
    elif task == "delay":
        mask = frame["Cancelled"].eq(0) & frame["delay_label_observed"].eq(1)
        labels = frame.loc[mask, "ArrDel15"]
    else:
        raise ValueError(f"FLARE hurdle does not fit direct task: {task}")
    numeric = pd.to_numeric(labels, errors="raise").astype("int64")
    if not numeric.isin([0, 1]).all():
        raise ValueError(f"invalid labels for FLARE {task}")
    return frame.loc[mask].reset_index(drop=True), numeric.to_numpy(dtype=np.int64)
