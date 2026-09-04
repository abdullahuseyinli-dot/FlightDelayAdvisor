"""Explicit realised-weather oracle diagnostic models.

These features are intentionally isolated from deployable model code.  They estimate
the value of perfect same-day weather information and provide a fairer comparison to
the legacy repository, but must never be described as schedule-time predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .contracts import AvailabilityHorizon, validate_predictors
from .modeling import CATEGORICAL_FEATURES, TaskName, engineer_model_features

ORACLE_WEATHER_FEATURES = tuple(
    f"oracle_{side}_{variable}"
    for side in ("origin", "dest")
    for variable in ("tavg", "prcp", "snow", "wspd")
)

ORACLE_DERIVED_FEATURES = (
    "oracle_temperature_gap",
    "oracle_precipitation_sum",
    "oracle_snow_sum",
    "oracle_wind_max",
    "oracle_origin_temperature_anomaly",
    "oracle_dest_temperature_anomaly",
    "oracle_origin_precipitation_excess",
    "oracle_dest_precipitation_excess",
    "oracle_origin_wind_excess",
    "oracle_dest_wind_excess",
)


def engineer_oracle_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add explicitly labelled realised-weather variables to schedule features."""

    validate_predictors(
        ORACLE_WEATHER_FEATURES,
        AvailabilityHorizon.ORACLE_REALISED,
        allow_oracle=True,
    )
    missing = sorted(set(ORACLE_WEATHER_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"oracle diagnostic is missing realised weather: {missing}")
    result = engineer_model_features(frame)
    weather: dict[str, pd.Series | NDArray[np.float32]] = {}
    for name in ORACLE_WEATHER_FEATURES:
        values = pd.to_numeric(frame[name], errors="raise").astype("float32")
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"oracle weather feature contains non-finite values: {name}")
        weather[name] = values

    weather["oracle_temperature_gap"] = (
        weather["oracle_dest_tavg"] - weather["oracle_origin_tavg"]
    ).astype("float32")
    weather["oracle_precipitation_sum"] = (
        weather["oracle_origin_prcp"] + weather["oracle_dest_prcp"]
    ).astype("float32")
    weather["oracle_snow_sum"] = (
        weather["oracle_origin_snow"] + weather["oracle_dest_snow"]
    ).astype("float32")
    weather["oracle_wind_max"] = np.maximum(
        weather["oracle_origin_wspd"], weather["oracle_dest_wspd"]
    ).astype("float32")
    for side in ("origin", "dest"):
        weather[f"oracle_{side}_temperature_anomaly"] = (
            weather[f"oracle_{side}_tavg"] - result[f"clim_{side}_tavg"]
        ).astype("float32")
        weather[f"oracle_{side}_precipitation_excess"] = (
            weather[f"oracle_{side}_prcp"] - result[f"clim_{side}_prcp"]
        ).astype("float32")
        weather[f"oracle_{side}_wind_excess"] = (
            weather[f"oracle_{side}_wspd"] - result[f"clim_{side}_wspd"]
        ).astype("float32")
    return pd.concat([result, pd.DataFrame(weather, index=result.index)], axis=1, copy=False)


@dataclass(slots=True)
class OracleCatBoostModel:
    """Native-categorical CatBoost model for the non-deployable oracle track."""

    estimator: Any
    task: TaskName

    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        matrix = engineer_oracle_features(frame)
        for column in CATEGORICAL_FEATURES:
            matrix[column] = matrix[column].astype(str)
        return np.asarray(self.estimator.predict_proba(matrix), dtype=np.float64)


def fit_oracle_catboost(
    train_frame: pd.DataFrame,
    train_labels: NDArray[np.int64],
    *,
    task: TaskName,
    params: dict[str, Any],
    validation_frame: pd.DataFrame | None = None,
    validation_labels: NDArray[np.int64] | None = None,
    sample_weight: NDArray[np.float64] | None = None,
) -> OracleCatBoostModel:
    """Fit the explicit oracle model; no caller can silently enable oracle inputs."""

    from catboost import CatBoostClassifier  # type: ignore[import-untyped]

    train_matrix = engineer_oracle_features(train_frame)
    for column in CATEGORICAL_FEATURES:
        train_matrix[column] = train_matrix[column].astype(str)
    defaults: dict[str, Any] = {
        "loss_function": "MultiClass" if task == "joint" else "Logloss",
        "eval_metric": "MultiClass" if task == "joint" else "Logloss",
        "iterations": 900,
        "learning_rate": 0.06,
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
        valid_matrix = engineer_oracle_features(validation_frame)
        for column in CATEGORICAL_FEATURES:
            valid_matrix[column] = valid_matrix[column].astype(str)
        fit_kwargs["eval_set"] = (valid_matrix, validation_labels)
        fit_kwargs["early_stopping_rounds"] = 75
        fit_kwargs["use_best_model"] = True
    estimator.fit(**fit_kwargs)
    return OracleCatBoostModel(estimator=estimator, task=task)


def oracle_feature_profile() -> dict[str, Any]:
    return {
        "availability_horizon": AvailabilityHorizon.ORACLE_REALISED.name,
        "deployable": False,
        "raw_realised_weather": list(ORACLE_WEATHER_FEATURES),
        "derived_weather": list(ORACLE_DERIVED_FEATURES),
    }
