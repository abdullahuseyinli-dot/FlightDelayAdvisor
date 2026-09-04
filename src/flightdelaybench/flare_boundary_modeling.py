"""Two-view CatBoost models for boundary-complete airport-network residuals."""

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
    census_model_input_columns,
)
from .flare_boundary_contracts import (
    BOUNDARY_ALL_FEATURES,
    BOUNDARY_DYNAMIC_SOURCE_FEATURES,
    BOUNDARY_FULL_BY_SOURCE,
    BOUNDARY_FULL_FEATURES,
    BOUNDARY_OBSERVATION_FEATURES,
    BOUNDARY_RESIDUAL_BY_SOURCE,
    validate_boundary_predictors,
)
from .flare_capacity_contracts import (
    CAPACITY_ALL_FEATURES,
    CAPACITY_STATIC_FEATURES,
    validate_capacity_predictors,
)
from .flare_modeling import FLARE24_BASE_EXTRA_FEATURES, engineer_flare_matrix

BoundaryTask = Literal["delay", "cancellation"]
BoundaryCandidate = Literal["boundary_only", "counterfactual_residual"]
BOUNDARY_GPU_TRAINING_PARAMETERS: dict[str, Any] = {
    "boosting_type": "Plain",
    "max_ctr_complexity": 1,
    "gpu_cat_features_storage": "CpuPinnedMemory",
    "gpu_ram_part": 0.75,
    "pinned_memory_size": "2gb",
}


def boundary_model_input_columns(
    *,
    flare_features: tuple[str, ...],
    capacity_features: tuple[str, ...],
    boundary_features: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the exact raw-column projection consumed by a boundary model."""

    validate_capacity_predictors(list(capacity_features))
    validate_boundary_predictors(list(boundary_features))
    base = census_model_input_columns(
        include_cross_direction=True,
        include_rich_schedule=True,
        include_flight_history=True,
        include_schedule_context=True,
        include_graph_pressure=True,
    )
    return tuple(
        dict.fromkeys(
            (
                *base,
                *flare_features,
                *capacity_features,
                *boundary_features,
            )
        )
    )


def attach_boundary_views(
    frame: pd.DataFrame,
    *,
    boundary_feature_dir: Path,
) -> pd.DataFrame:
    """Attach the boundary-complete view and signed induced-graph residual."""

    required = {"sample_id", "Year", "Month", *CAPACITY_ALL_FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"boundary feature join is missing columns: {missing}")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("boundary feature join requires unique non-missing sample ids")
    overlap = sorted(set(BOUNDARY_ALL_FEATURES) & set(frame.columns))
    if overlap:
        raise ValueError(f"boundary features are already attached: {overlap}")

    requested = frame.loc[:, ["sample_id", "Year", "Month"]].copy()
    requested["_row_order"] = np.arange(len(frame), dtype=np.int64)
    parts: list[pd.DataFrame] = []
    for (year_value, month_value), rows in requested.groupby(
        ["Year", "Month"], observed=True, sort=True
    ):
        year, month = int(year_value), int(month_value)
        path = boundary_feature_dir / f"year={year}" / f"month={month:02d}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing boundary feature partition: {path}")
        partition = pd.read_parquet(path, columns=["sample_id", *CAPACITY_ALL_FEATURES])
        if partition["sample_id"].isna().any() or partition["sample_id"].duplicated().any():
            raise ValueError(f"invalid sample ids in boundary partition: {path}")
        selected = rows.loc[:, ["sample_id", "_row_order"]].merge(
            partition,
            on="sample_id",
            how="left",
            sort=False,
            validate="one_to_one",
            indicator=True,
        )
        if not selected["_merge"].eq("both").all():
            raise ValueError(f"boundary partition omits requested flights: {path}")
        parts.append(selected.drop(columns="_merge"))
    boundary = (
        pd.concat(parts, ignore_index=True)
        .sort_values("_row_order", kind="mergesort")
        .drop(columns=["sample_id", "_row_order"])
        .reset_index(drop=True)
    )

    for feature in CAPACITY_STATIC_FEATURES:
        induced = pd.to_numeric(frame[feature], errors="raise").to_numpy(dtype=np.float64)
        complete = pd.to_numeric(boundary[feature], errors="raise").to_numpy(dtype=np.float64)
        if not np.allclose(induced, complete, rtol=0.0, atol=1e-6, equal_nan=True):
            raise ValueError(f"boundary context unexpectedly changed static feature: {feature}")

    output: dict[str, pd.Series[Any] | NDArray[np.float32]] = {}
    for source in BOUNDARY_DYNAMIC_SOURCE_FEATURES:
        complete_series = pd.to_numeric(boundary[source], errors="raise").astype("float32")
        induced_series = pd.to_numeric(frame[source], errors="raise").astype("float32")
        output[BOUNDARY_FULL_BY_SOURCE[source]] = complete_series
        output[BOUNDARY_RESIDUAL_BY_SOURCE[source]] = (
            complete_series.fillna(0.0) - induced_series.fillna(0.0)
        ).astype("float32")
    source = "ccrth_rotation_predecessor_capacity_shadow_price"
    induced_observed = pd.to_numeric(frame[source], errors="raise").notna()
    complete_observed = pd.to_numeric(boundary[source], errors="raise").notna()
    output[BOUNDARY_OBSERVATION_FEATURES[0]] = (complete_observed & ~induced_observed).to_numpy(
        dtype=np.float32
    )
    output[BOUNDARY_OBSERVATION_FEATURES[1]] = (induced_observed & ~complete_observed).to_numpy(
        dtype=np.float32
    )
    attached = pd.DataFrame(output)
    if np.isinf(attached.to_numpy(dtype=np.float64)).any():
        raise ValueError("boundary views contain infinity")
    return pd.concat([frame.reset_index(drop=True), attached], axis=1, copy=False)


def select_usable_boundary_features(
    training_frame: pd.DataFrame,
    candidates: tuple[str, ...] = BOUNDARY_ALL_FEATURES,
    *,
    minimum_nonmissing_fraction: float = 0.01,
    minimum_unique_values: int = 2,
) -> tuple[str, ...]:
    """Select features using training covariates only, never target association."""

    validate_boundary_predictors(list(candidates))
    missing = sorted(set(candidates) - set(training_frame.columns))
    if missing:
        raise ValueError(f"training frame omits boundary candidates: {missing}")
    return tuple(
        feature
        for feature in candidates
        if float(training_frame[feature].notna().mean()) >= minimum_nonmissing_fraction
        and int(training_frame[feature].dropna().nunique()) >= minimum_unique_values
    )


def engineer_boundary_matrix(
    frame: pd.DataFrame,
    *,
    candidate: BoundaryCandidate,
    flare_features: tuple[str, ...],
    capacity_features: tuple[str, ...],
    boundary_features: tuple[str, ...],
) -> pd.DataFrame:
    validate_capacity_predictors(list(capacity_features))
    validate_boundary_predictors(list(boundary_features))
    missing = sorted((set(capacity_features) | set(boundary_features)) - set(frame.columns))
    if missing:
        raise ValueError(f"boundary model input is missing features: {missing}")
    base = engineer_flare_matrix(frame, extra_features=flare_features)
    capacity = frame.loc[:, list(capacity_features)].copy()
    for feature in capacity_features:
        capacity[feature] = pd.to_numeric(capacity[feature], errors="raise").astype("float32")
    boundary = frame.loc[:, list(boundary_features)].copy()
    for feature in boundary_features:
        boundary[feature] = pd.to_numeric(boundary[feature], errors="raise").astype("float32")
    if candidate == "boundary_only":
        unexpected = sorted(set(boundary_features) - set(BOUNDARY_FULL_FEATURES))
        if unexpected:
            raise ValueError(f"boundary-only matrix received residual features: {unexpected}")
    elif candidate != "counterfactual_residual":
        raise ValueError(f"unknown boundary candidate: {candidate}")
    numeric = pd.concat([capacity, boundary], axis=1)
    if np.isinf(numeric.to_numpy(dtype=np.float64)).any():
        raise ValueError("boundary model matrix contains infinity")
    return pd.concat([base.reset_index(drop=True), numeric.reset_index(drop=True)], axis=1)


@dataclass(slots=True)
class BoundaryCatBoostModel:
    estimator: Any
    task: BoundaryTask
    candidate: BoundaryCandidate
    flare_features: tuple[str, ...]
    capacity_features: tuple[str, ...]
    boundary_features: tuple[str, ...]

    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        matrix = engineer_boundary_matrix(
            frame,
            candidate=self.candidate,
            flare_features=self.flare_features,
            capacity_features=self.capacity_features,
            boundary_features=self.boundary_features,
        )
        categorical = [
            *CENSUS_BASE_CATEGORICAL_FEATURES,
            *CENSUS_FLIGHT_CATEGORICAL_FEATURES,
        ]
        for column in categorical:
            matrix[column] = matrix[column].astype(str)
        probabilities = np.asarray(self.estimator.predict_proba(matrix), dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise RuntimeError("boundary estimator returned invalid probabilities")
        return probabilities[:, 1]


def fit_boundary_catboost(
    train_frame: pd.DataFrame,
    train_labels: ArrayLike,
    *,
    task: BoundaryTask,
    candidate: BoundaryCandidate,
    capacity_features: tuple[str, ...],
    boundary_features: tuple[str, ...],
    params: dict[str, Any],
    flare_features: tuple[str, ...] | None = None,
    validation_frame: pd.DataFrame | None = None,
    validation_labels: ArrayLike | None = None,
) -> BoundaryCatBoostModel:
    from catboost import CatBoostClassifier  # type: ignore[import-untyped]

    selected_flare = flare_features or FLARE24_BASE_EXTRA_FEATURES
    labels = np.asarray(train_labels, dtype=np.int64)
    if labels.shape != (len(train_frame),) or not np.isin(labels, [0, 1]).all():
        raise ValueError("boundary training labels must be aligned and binary")
    matrix = engineer_boundary_matrix(
        train_frame,
        candidate=candidate,
        flare_features=selected_flare,
        capacity_features=capacity_features,
        boundary_features=boundary_features,
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
        **BOUNDARY_GPU_TRAINING_PARAMETERS,
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
            raise ValueError("boundary validation frame and labels must be supplied together")
        validation_y = np.asarray(validation_labels, dtype=np.int64)
        if (
            validation_y.shape != (len(validation_frame),)
            or not np.isin(validation_y, [0, 1]).all()
        ):
            raise ValueError("boundary validation labels must be aligned and binary")
        validation_matrix = engineer_boundary_matrix(
            validation_frame,
            candidate=candidate,
            flare_features=selected_flare,
            capacity_features=capacity_features,
            boundary_features=boundary_features,
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
    return BoundaryCatBoostModel(
        estimator=estimator,
        task=task,
        candidate=candidate,
        flare_features=selected_flare,
        capacity_features=capacity_features,
        boundary_features=boundary_features,
    )
