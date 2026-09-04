"""Memory-bounded data access and model wrappers for FlightDelayBench."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .contracts import AvailabilityHorizon, validate_predictors

TaskName = Literal["delay", "cancellation", "joint"]

CATEGORICAL_FEATURES = (
    "Reporting_Airline",
    "Origin",
    "Dest",
    "Route",
    "DistanceBand",
)

SCHEDULE_NUMERIC_FEATURES = (
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

PRIOR_RATE_FEATURES = tuple(
    f"prior_{level}_{outcome}_rate"
    for level in ("global", "route", "airline", "origin", "dest", "slot")
    for outcome in ("delay", "cancel")
)
PRIOR_SUPPORT_FEATURES = (
    "prior_global_count",
    "prior_route_count",
    "prior_route_delay_support",
    "prior_airline_count",
    "prior_airline_delay_support",
    "prior_origin_count",
    "prior_origin_delay_support",
    "prior_dest_count",
    "prior_dest_delay_support",
    "prior_slot_count",
    "prior_slot_delay_support",
)
CLIMATOLOGY_FEATURES = tuple(
    item
    for side in ("origin", "dest")
    for variable in ("tavg", "prcp", "snow", "wspd")
    for item in (f"clim_{side}_{variable}", f"clim_{side}_{variable}_missing")
)

REGISTERED_MODEL_INPUTS = (
    *CATEGORICAL_FEATURES,
    *SCHEDULE_NUMERIC_FEATURES,
    *PRIOR_RATE_FEATURES,
    *PRIOR_SUPPORT_FEATURES,
    *CLIMATOLOGY_FEATURES,
)

DERIVED_NUMERIC_FEATURES = (
    "log_distance",
    *tuple(f"log1p_{name}" for name in PRIOR_SUPPORT_FEATURES),
    *tuple(f"logit_{name}" for name in PRIOR_RATE_FEATURES),
    "origin_dest_delay_mean",
    "origin_dest_delay_max",
    "origin_dest_delay_gap",
    "origin_dest_cancel_mean",
    "origin_dest_cancel_max",
    "origin_dest_cancel_gap",
    "route_delay_excess",
    "airline_delay_excess",
    "slot_delay_excess",
    "route_cancel_excess",
    "airline_cancel_excess",
    "slot_cancel_excess",
    "clim_temperature_gap",
    "clim_precipitation_sum",
    "clim_snow_sum",
    "clim_wind_max",
)

NUMERIC_MODEL_FEATURES = (
    *SCHEDULE_NUMERIC_FEATURES,
    *PRIOR_RATE_FEATURES,
    *PRIOR_SUPPORT_FEATURES,
    *CLIMATOLOGY_FEATURES,
    *DERIVED_NUMERIC_FEATURES,
)
MODEL_FEATURES = (*CATEGORICAL_FEATURES, *NUMERIC_MODEL_FEATURES)

LOAD_COLUMNS = tuple(
    dict.fromkeys(
        [
            "sample_id",
            "FlightDate",
            "ArrDel15",
            "Cancelled",
            "delay_label_observed",
            "joint_label_observed",
            "disruption_state",
            *REGISTERED_MODEL_INPUTS,
        ]
    )
)


def _variant_files(feature_dir: Path, year: int) -> list[Path]:
    annual = feature_dir / f"year={year}.parquet"
    if annual.is_file():
        return [annual]
    monthly = sorted((feature_dir / f"year={year}").glob("month=*.parquet"))
    if monthly:
        return monthly
    raise FileNotFoundError(f"no point-in-time feature partition for {year} in {feature_dir}")


def _stable_sample(frame: pd.DataFrame, limit: int | None, *, seed: int) -> pd.DataFrame:
    if limit is None or len(frame) <= limit:
        return frame.reset_index(drop=True)
    return frame.sample(n=limit, random_state=seed, replace=False).reset_index(drop=True)


def load_feature_year(
    feature_dir: Path,
    year: int,
    *,
    limit: int | None = None,
    seed: int = 20260903,
    extra_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Load one target year and optionally take an exact reproducible sample."""

    columns = list(dict.fromkeys((*LOAD_COLUMNS, *extra_columns)))
    frames = [pd.read_parquet(path, columns=columns) for path in _variant_files(feature_dir, year)]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if not frame["Year"].eq(year).all():
        raise ValueError(f"partition contains rows outside declared year {year}")
    frame = _stable_sample(frame, limit, seed=seed + year)
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("string").fillna("__MISSING__")
    return frame


def load_feature_years(
    feature_dir: Path,
    years: tuple[int, ...],
    *,
    rows_per_year: int | None = None,
    seed: int = 20260903,
    extra_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    if not years:
        raise ValueError("at least one year is required")
    return pd.concat(
        [
            load_feature_year(
                feature_dir,
                year,
                limit=rows_per_year,
                seed=seed,
                extra_columns=extra_columns,
            )
            for year in years
        ],
        ignore_index=True,
    )


def _safe_logit(values: pd.Series, epsilon: float = 1e-5) -> pd.Series:
    clipped = values.astype("float64").clip(epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped)).astype("float32")


def engineer_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic algebra to registered schedule-horizon inputs."""

    validate_predictors(REGISTERED_MODEL_INPUTS, AvailabilityHorizon.SCHEDULE_CLIMATOLOGY)
    missing = sorted(set(REGISTERED_MODEL_INPUTS) - set(frame.columns))
    if missing:
        raise ValueError(f"model input is missing registered features: {missing}")
    result = frame.loc[:, list(REGISTERED_MODEL_INPUTS)].copy()
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].astype("string").fillna("__MISSING__")
    for column in set(
        SCHEDULE_NUMERIC_FEATURES
        + PRIOR_RATE_FEATURES
        + PRIOR_SUPPORT_FEATURES
        + CLIMATOLOGY_FEATURES
    ):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float32")

    result["log_distance"] = np.log1p(result["Distance"].clip(lower=0)).astype("float32")
    for column in PRIOR_SUPPORT_FEATURES:
        result[f"log1p_{column}"] = np.log1p(result[column].clip(lower=0)).astype("float32")
    for column in PRIOR_RATE_FEATURES:
        result[f"logit_{column}"] = _safe_logit(result[column])

    for outcome in ("delay", "cancel"):
        origin = result[f"prior_origin_{outcome}_rate"]
        dest = result[f"prior_dest_{outcome}_rate"]
        result[f"origin_dest_{outcome}_mean"] = ((origin + dest) / 2).astype("float32")
        result[f"origin_dest_{outcome}_max"] = np.maximum(origin, dest).astype("float32")
        result[f"origin_dest_{outcome}_gap"] = (origin - dest).astype("float32")
        global_rate = result[f"prior_global_{outcome}_rate"]
        for level in ("route", "airline", "slot"):
            result[f"{level}_{outcome}_excess"] = (
                result[f"prior_{level}_{outcome}_rate"] - global_rate
            ).astype("float32")

    result["clim_temperature_gap"] = (result["clim_dest_tavg"] - result["clim_origin_tavg"]).astype(
        "float32"
    )
    result["clim_precipitation_sum"] = (
        result["clim_origin_prcp"] + result["clim_dest_prcp"]
    ).astype("float32")
    result["clim_snow_sum"] = (result["clim_origin_snow"] + result["clim_dest_snow"]).astype(
        "float32"
    )
    result["clim_wind_max"] = np.maximum(
        result["clim_origin_wspd"], result["clim_dest_wspd"]
    ).astype("float32")
    return result.loc[:, list(MODEL_FEATURES)]


def task_view(frame: pd.DataFrame, task: TaskName) -> tuple[pd.DataFrame, NDArray[np.int64]]:
    if task == "delay":
        mask = frame["Cancelled"].eq(0) & frame["delay_label_observed"].eq(1)
        labels = frame.loc[mask, "ArrDel15"].to_numpy(dtype=np.int64)
    elif task == "cancellation":
        mask = frame["Cancelled"].isin([0, 1])
        labels = frame.loc[mask, "Cancelled"].to_numpy(dtype=np.int64)
    elif task == "joint":
        mask = frame["joint_label_observed"].eq(1)
        labels = frame.loc[mask, "disruption_state"].to_numpy(dtype=np.int64)
    else:
        raise ValueError(f"unknown task: {task}")
    selected = frame if bool(mask.all()) else frame.loc[mask].reset_index(drop=True)
    if len(selected) == 0 or len(selected) != len(labels):
        raise ValueError(f"task {task} has no valid or aligned observations")
    return selected, labels


def temporal_sample_weights(
    years: pd.Series,
    *,
    prediction_year: int,
    half_life_years: float | None,
) -> NDArray[np.float64] | None:
    if half_life_years is None:
        return None
    if half_life_years <= 0:
        raise ValueError("half_life_years must be positive")
    age = prediction_year - pd.to_numeric(years, errors="raise").to_numpy(dtype="float64")
    if (age <= 0).any():
        raise ValueError("training years must strictly precede prediction year")
    weights = np.asarray(np.exp(-math.log(2.0) * age / half_life_years), dtype=np.float64)
    mean_weight = float(np.mean(weights))
    return np.asarray(weights / mean_weight, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class CategoryCodec:
    levels: dict[str, tuple[str, ...]]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> CategoryCodec:
        return cls(
            levels={
                column: tuple(sorted(frame[column].astype("string").fillna("__MISSING__").unique()))
                for column in CATEGORICAL_FEATURES
            }
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = engineer_model_features(frame)
        for column, levels in self.levels.items():
            matrix[column] = pd.Categorical(
                matrix[column].astype("string").fillna("__MISSING__"),
                categories=list(levels),
            ).codes.astype("int32")
        return matrix


@dataclass(slots=True)
class LightGBMModel:
    estimator: Any
    codec: CategoryCodec
    task: TaskName
    best_iteration: int

    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        matrix = self.codec.transform(frame)
        probabilities = self.estimator.predict_proba(matrix, num_iteration=self.best_iteration)
        return np.asarray(probabilities, dtype=np.float64)


def fit_lightgbm(
    train_frame: pd.DataFrame,
    train_labels: NDArray[np.int64],
    *,
    task: TaskName,
    params: dict[str, Any],
    validation_frame: pd.DataFrame | None = None,
    validation_labels: NDArray[np.int64] | None = None,
    sample_weight: NDArray[np.float64] | None = None,
) -> LightGBMModel:
    """Fit a native-categorical LightGBM model with a serializable codec."""

    import lightgbm as lgb

    codec = CategoryCodec.fit(train_frame)
    train_matrix = codec.transform(train_frame)
    objective = "multiclass" if task == "joint" else "binary"
    defaults: dict[str, Any] = {
        "objective": objective,
        "n_estimators": 700,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 200,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 20260903,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if task == "joint":
        defaults["num_class"] = 3
    defaults.update(params)
    estimator = lgb.LGBMClassifier(**defaults)
    fit_kwargs: dict[str, Any] = {
        "categorical_feature": list(CATEGORICAL_FEATURES),
        "sample_weight": sample_weight,
        "callbacks": [lgb.log_evaluation(period=0)],
    }
    if validation_frame is not None and validation_labels is not None:
        fit_kwargs["eval_set"] = [(codec.transform(validation_frame), validation_labels)]
        fit_kwargs["callbacks"].append(lgb.early_stopping(75, verbose=False))
    estimator.fit(train_matrix, train_labels, **fit_kwargs)
    best_iteration = int(estimator.best_iteration_ or defaults["n_estimators"])
    return LightGBMModel(
        estimator=estimator,
        codec=codec,
        task=task,
        best_iteration=best_iteration,
    )


@dataclass(slots=True)
class CatBoostModel:
    estimator: Any
    task: TaskName

    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        matrix = engineer_model_features(frame)
        for column in CATEGORICAL_FEATURES:
            matrix[column] = matrix[column].astype(str)
        return np.asarray(self.estimator.predict_proba(matrix), dtype=np.float64)


def fit_catboost(
    train_frame: pd.DataFrame,
    train_labels: NDArray[np.int64],
    *,
    task: TaskName,
    params: dict[str, Any],
    validation_frame: pd.DataFrame | None = None,
    validation_labels: NDArray[np.int64] | None = None,
    sample_weight: NDArray[np.float64] | None = None,
) -> CatBoostModel:
    """Fit CatBoost while preserving native string categorical variables."""

    from catboost import CatBoostClassifier  # type: ignore[import-untyped]

    train_matrix = engineer_model_features(train_frame)
    for column in CATEGORICAL_FEATURES:
        train_matrix[column] = train_matrix[column].astype(str)
    defaults: dict[str, Any] = {
        "loss_function": "MultiClass" if task == "joint" else "Logloss",
        "eval_metric": "MultiClass" if task == "joint" else "Logloss",
        "iterations": 800,
        "learning_rate": 0.06,
        "depth": 8,
        "l2_leaf_reg": 5.0,
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
        valid_matrix = engineer_model_features(validation_frame)
        for column in CATEGORICAL_FEATURES:
            valid_matrix[column] = valid_matrix[column].astype(str)
        fit_kwargs["eval_set"] = (valid_matrix, validation_labels)
        fit_kwargs["early_stopping_rounds"] = 75
        fit_kwargs["use_best_model"] = True
    estimator.fit(**fit_kwargs)
    return CatBoostModel(estimator=estimator, task=task)
