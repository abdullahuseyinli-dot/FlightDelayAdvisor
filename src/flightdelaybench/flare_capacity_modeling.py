"""Model adapters and nested ablations for the CC-RTH extension."""

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
)
from .contracts import FLARE24_AVIATION_WEATHER_FEATURES, AvailabilityHorizon
from .flare_capacity_contracts import (
    CAPACITY_ALL_FEATURES,
    CAPACITY_DEMAND_FEATURES,
    CAPACITY_STATE_FEATURES,
    CAPACITY_STATIC_FEATURES,
    validate_capacity_predictors,
)
from .flare_modeling import (
    FLARE24_BASE_EXTRA_FEATURES,
    engineer_flare_matrix,
)

CapacityTask = Literal["delay", "cancellation"]
CapacityCandidate = Literal[
    "flare24",
    "raw_demand",
    "normalized_capacity",
    "queue_shadow",
    "hypergraph",
]

QUEUE_SHADOW_TOKENS = (
    "expected_queue",
    "queue_p90",
    "recovery_minutes",
    "shadow_price",
    "marginal_overload",
)

CAPACITY_QUEUE_SHADOW_FEATURES = tuple(
    feature
    for feature in CAPACITY_STATE_FEATURES
    if any(token in feature for token in QUEUE_SHADOW_TOKENS)
)
CAPACITY_NORMALIZED_STATE_FEATURES = tuple(
    feature
    for feature in CAPACITY_STATE_FEATURES
    if feature not in CAPACITY_QUEUE_SHADOW_FEATURES
)

CAPACITY_CANDIDATE_FEATURES: dict[CapacityCandidate, tuple[str, ...]] = {
    "flare24": (),
    "raw_demand": (*CAPACITY_STATIC_FEATURES, *CAPACITY_DEMAND_FEATURES),
    "normalized_capacity": (
        *CAPACITY_STATIC_FEATURES,
        *CAPACITY_DEMAND_FEATURES,
        *CAPACITY_NORMALIZED_STATE_FEATURES,
    ),
    "queue_shadow": (
        *CAPACITY_STATIC_FEATURES,
        *CAPACITY_DEMAND_FEATURES,
        *CAPACITY_STATE_FEATURES,
    ),
    "hypergraph": CAPACITY_ALL_FEATURES,
}


def attach_capacity_feature_partitions(
    frame: pd.DataFrame,
    *,
    feature_dir: Path,
    feature_columns: tuple[str, ...] = CAPACITY_ALL_FEATURES,
) -> pd.DataFrame:
    """Attach sample-keyed CC-RTH features without changing row order."""

    required = {"sample_id", "Year", "Month"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"capacity feature join is missing columns: {missing}")
    validate_capacity_predictors(list(feature_columns), AvailabilityHorizon.FORECAST_24H)
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("capacity feature join requires unique non-missing sample ids")
    overlap = sorted(set(feature_columns) & set(frame.columns))
    if overlap:
        raise ValueError(f"capacity features are already attached: {overlap}")
    requested = frame.loc[:, ["sample_id", "Year", "Month"]].copy()
    requested["_row_order"] = np.arange(len(frame), dtype=np.int64)
    parts: list[pd.DataFrame] = []
    for (year_value, month_value), rows in requested.groupby(
        ["Year", "Month"], observed=True, sort=True
    ):
        year = int(year_value)
        month = int(month_value)
        path = feature_dir / f"year={year}" / f"month={month:02d}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing capacity feature partition: {path}")
        partition = pd.read_parquet(path, columns=["sample_id", *feature_columns])
        if partition["sample_id"].isna().any() or partition["sample_id"].duplicated().any():
            raise ValueError(f"invalid sample ids in capacity partition: {path}")
        selected = rows.loc[:, ["sample_id", "_row_order"]].merge(
            partition,
            on="sample_id",
            how="left",
            sort=False,
            validate="one_to_one",
            indicator=True,
        )
        if not selected["_merge"].eq("both").all():
            raise ValueError(f"capacity partition omits requested flights: {path}")
        parts.append(selected.drop(columns="_merge"))
    attached = (
        pd.concat(parts, ignore_index=True)
        .sort_values("_row_order", kind="mergesort")
        .drop(columns=["sample_id", "_row_order"])
        .reset_index(drop=True)
    )
    return pd.concat([frame.reset_index(drop=True), attached], axis=1, copy=False)


def select_usable_capacity_features(
    training_frame: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    minimum_nonmissing_fraction: float = 0.01,
    minimum_unique_values: int = 2,
) -> tuple[str, ...]:
    """Select only by training-covariate availability and variation."""

    validate_capacity_predictors(list(candidates))
    missing = sorted(set(candidates) - set(training_frame.columns))
    if missing:
        raise ValueError(f"training frame omits capacity candidates: {missing}")
    selected: list[str] = []
    for feature in candidates:
        values = pd.to_numeric(training_frame[feature], errors="raise")
        if (
            float(values.notna().mean()) >= minimum_nonmissing_fraction
            and int(values.dropna().nunique()) >= minimum_unique_values
        ):
            selected.append(feature)
    return tuple(selected)


def engineer_capacity_matrix(
    frame: pd.DataFrame,
    *,
    flare_features: tuple[str, ...],
    capacity_features: tuple[str, ...],
) -> pd.DataFrame:
    """Combine frozen FLARE covariates with registered CC-RTH messages."""

    validate_capacity_predictors(list(capacity_features))
    missing = sorted(set(capacity_features) - set(frame.columns))
    if missing:
        raise ValueError(f"CC-RTH model input is missing capacity features: {missing}")
    base = engineer_flare_matrix(frame, extra_features=flare_features)
    capacity = frame.loc[:, list(capacity_features)].copy()
    for feature in capacity_features:
        capacity[feature] = pd.to_numeric(capacity[feature], errors="raise").astype("float32")
    if np.isinf(capacity.to_numpy(dtype=np.float64)).any():
        raise ValueError("CC-RTH model features contain infinity")
    return pd.concat([base.reset_index(drop=True), capacity.reset_index(drop=True)], axis=1)


@dataclass(slots=True)
class CapacityCatBoostModel:
    estimator: Any
    task: CapacityTask
    flare_features: tuple[str, ...]
    capacity_features: tuple[str, ...]

    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        matrix = engineer_capacity_matrix(
            frame,
            flare_features=self.flare_features,
            capacity_features=self.capacity_features,
        )
        categorical = [
            *CENSUS_BASE_CATEGORICAL_FEATURES,
            *CENSUS_FLIGHT_CATEGORICAL_FEATURES,
        ]
        for column in categorical:
            matrix[column] = matrix[column].astype(str)
        probabilities = np.asarray(self.estimator.predict_proba(matrix), dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise RuntimeError("CC-RTH estimator returned invalid probabilities")
        return probabilities[:, 1]


def fit_capacity_catboost(
    train_frame: pd.DataFrame,
    train_labels: ArrayLike,
    *,
    task: CapacityTask,
    capacity_features: tuple[str, ...],
    params: dict[str, Any],
    flare_features: tuple[str, ...] | None = None,
    validation_frame: pd.DataFrame | None = None,
    validation_labels: ArrayLike | None = None,
) -> CapacityCatBoostModel:
    """Fit one binary hurdle component with chronological early stopping."""

    from catboost import CatBoostClassifier  # type: ignore[import-untyped]

    if task not in {"delay", "cancellation"}:
        raise ValueError(f"invalid capacity hurdle task: {task}")
    selected_flare = flare_features or FLARE24_BASE_EXTRA_FEATURES
    labels = np.asarray(train_labels, dtype=np.int64)
    if labels.shape != (len(train_frame),) or not np.isin(labels, [0, 1]).all():
        raise ValueError("CC-RTH training labels must be aligned and binary")
    matrix = engineer_capacity_matrix(
        train_frame,
        flare_features=selected_flare,
        capacity_features=capacity_features,
    )
    categorical = [
        *CENSUS_BASE_CATEGORICAL_FEATURES,
        *CENSUS_FLIGHT_CATEGORICAL_FEATURES,
    ]
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
        "random_seed": 20260904,
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
            raise ValueError("CC-RTH validation frame and labels must be supplied together")
        validation_y = np.asarray(validation_labels, dtype=np.int64)
        if validation_y.shape != (len(validation_frame),) or not np.isin(
            validation_y, [0, 1]
        ).all():
            raise ValueError("CC-RTH validation labels must be aligned and binary")
        validation_matrix = engineer_capacity_matrix(
            validation_frame,
            flare_features=selected_flare,
            capacity_features=capacity_features,
        )
        for column in categorical:
            validation_matrix[column] = validation_matrix[column].astype(str)
        fit_kwargs.update(
            {
                "eval_set": (validation_matrix, validation_y),
                "early_stopping_rounds": 150,
                "use_best_model": True,
            }
        )
    estimator.fit(**fit_kwargs)
    return CapacityCatBoostModel(
        estimator=estimator,
        task=task,
        flare_features=selected_flare,
        capacity_features=capacity_features,
    )


def capacity_candidate_profile() -> dict[str, Any]:
    return {
        "candidate_order": list(CAPACITY_CANDIDATE_FEATURES),
        "nested": all(
            set(CAPACITY_CANDIDATE_FEATURES[left]).issubset(
                CAPACITY_CANDIDATE_FEATURES[right]
            )
            for left, right in zip(
                tuple(CAPACITY_CANDIDATE_FEATURES)[:-1],
                tuple(CAPACITY_CANDIDATE_FEATURES)[1:],
                strict=True,
            )
        ),
        "existing_weather_features": len(FLARE24_BASE_EXTRA_FEATURES),
        "existing_aviation_weather_features": len(FLARE24_AVIATION_WEATHER_FEATURES),
        "capacity_features": {
            name: len(features) for name, features in CAPACITY_CANDIDATE_FEATURES.items()
        },
        "architecture": (
            "CatBoost on explicit one-hop resource-time hypergraph messages; the sparse "
            "node/edge artifacts remain available for later neural hypergraph ablation."
        ),
    }
