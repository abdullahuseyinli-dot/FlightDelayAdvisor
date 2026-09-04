"""Past-observation to fixed-lead-forecast weather transfer models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .contracts import (
    FORECAST24_DAILY_FEATURES,
    AvailabilityHorizon,
    validate_predictors,
)
from .forecast_features import DAILY_FORECAST_COLUMNS
from .modeling import CATEGORICAL_FEATURES, TaskName, load_feature_year
from .oracle import ORACLE_WEATHER_FEATURES
from .recent import attach_recent_features
from .recent_modeling import engineer_recent_model_features

WeatherSource = Literal["observed_training", "forecast24_inference"]

RESIDUAL_CONTEXT_FEATURES = (
    *CATEGORICAL_FEATURES,
    "Month",
    "DayOfWeek",
    "DayOfYear",
    "DepHour",
    "DepHour_sin",
    "DepHour_cos",
    "IsWeekend",
    "IsHolidaySeason",
    "Distance",
)


def attach_daily_forecasts(frame: pd.DataFrame, forecast_dir: Path) -> pd.DataFrame:
    """Join airport-local daily forecasts while preserving exact flight row order."""

    output = frame.copy()
    output["FlightDate"] = pd.to_datetime(output["FlightDate"])
    output["__forecast_row_order"] = np.arange(len(output), dtype=np.int64)
    years = sorted(int(value) for value in output["Year"].unique())
    tables: list[pd.DataFrame] = []
    for year in years:
        path = forecast_dir / f"year={year}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing fixed-lead daily forecast table: {path}")
        tables.append(pd.read_parquet(path))
    lookup = pd.concat(tables, ignore_index=True)
    lookup["FlightDate"] = pd.to_datetime(lookup["FlightDate"])
    if lookup.duplicated(["Airport", "FlightDate"]).any():
        raise ValueError("daily forecast lookup contains duplicate airport dates")

    for side, airport_column in (("origin", "Origin"), ("dest", "Dest")):
        rename = {
            name: name.replace("forecast24_", f"forecast24_{side}_", 1)
            for name in DAILY_FORECAST_COLUMNS
        }
        side_lookup = lookup.rename(columns={"Airport": airport_column, **rename})
        output = output.merge(
            side_lookup,
            on=[airport_column, "FlightDate"],
            how="left",
            sort=False,
            validate="many_to_one",
        )
    output = output.sort_values("__forecast_row_order", kind="stable").drop(
        columns="__forecast_row_order"
    )
    output = output.reset_index(drop=True)
    if len(output) != len(frame):
        raise AssertionError("forecast join changed the flight row count")
    validate_predictors(FORECAST24_DAILY_FEATURES, AvailabilityHorizon.FORECAST_24H)
    return output


def load_observed_recent_year(
    feature_dir: Path,
    recent_dir: Path,
    year: int,
    *,
    limit: int | None,
    seed: int = 20260903,
) -> pd.DataFrame:
    frame = load_feature_year(
        feature_dir,
        year,
        limit=limit,
        seed=seed,
        extra_columns=ORACLE_WEATHER_FEATURES,
    )
    return attach_recent_features(frame, recent_dir=recent_dir)


def load_observed_recent_years(
    feature_dir: Path,
    recent_dir: Path,
    years: tuple[int, ...],
    *,
    rows_per_year: int | None,
    seed: int = 20260903,
) -> pd.DataFrame:
    if not years:
        raise ValueError("at least one weather-transfer training year is required")
    return pd.concat(
        [
            load_observed_recent_year(
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


def load_forecast_recent_year(
    feature_dir: Path,
    recent_dir: Path,
    forecast_dir: Path,
    year: int,
    *,
    limit: int | None,
    seed: int = 20260903,
) -> pd.DataFrame:
    frame = load_feature_year(feature_dir, year, limit=limit, seed=seed)
    recent = attach_recent_features(frame, recent_dir=recent_dir)
    return attach_daily_forecasts(recent, forecast_dir)


def engineer_weather_transfer_features(
    frame: pd.DataFrame,
    *,
    source: WeatherSource,
    include_cross_direction: bool,
) -> pd.DataFrame:
    """Map past observations and fixed-lead forecasts into shared weather semantics."""

    base = engineer_recent_model_features(
        frame,
        include_cross_direction=include_cross_direction,
    )
    weather: dict[str, pd.Series | NDArray[np.float32]] = {}
    if source == "observed_training":
        validate_predictors(
            ORACLE_WEATHER_FEATURES,
            AvailabilityHorizon.ORACLE_REALISED,
            allow_oracle=True,
        )
        for side in ("origin", "dest"):
            for statistic in ("tavg", "prcp", "wspd"):
                column = f"oracle_{side}_{statistic}"
                values = pd.to_numeric(frame[column], errors="raise").astype("float32")
                if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
                    raise ValueError(f"observed training weather is non-finite: {column}")
                weather[f"transfer_{side}_{statistic}"] = values
            weather[f"transfer_{side}_forecast_missing"] = pd.Series(
                np.zeros(len(frame), dtype=np.float32), index=frame.index
            )
            weather[f"transfer_{side}_forecast_coverage"] = pd.Series(
                np.ones(len(frame), dtype=np.float32), index=frame.index
            )
    elif source == "forecast24_inference":
        validate_predictors(FORECAST24_DAILY_FEATURES, AvailabilityHorizon.FORECAST_24H)
        missing = sorted(set(FORECAST24_DAILY_FEATURES) - set(frame.columns))
        if missing:
            raise ValueError(f"fixed-lead forecast inference is missing features: {missing}")
        for side in ("origin", "dest"):
            mappings = {
                "tavg": "tavg",
                "prcp": "prcp_sum",
                "wspd": "wspd_mean",
            }
            for statistic, forecast_statistic in mappings.items():
                raw = pd.to_numeric(
                    frame[f"forecast24_{side}_{forecast_statistic}"], errors="coerce"
                ).astype("float32")
                fallback = pd.to_numeric(base[f"clim_{side}_{statistic}"], errors="raise")
                weather[f"transfer_{side}_{statistic}"] = raw.fillna(fallback).astype("float32")
            weather[f"transfer_{side}_forecast_missing"] = (
                pd.to_numeric(frame[f"forecast24_{side}_missing"], errors="coerce")
                .fillna(1)
                .astype("float32")
            )
            weather[f"transfer_{side}_forecast_coverage"] = (
                pd.to_numeric(
                    frame[f"forecast24_{side}_min_variable_coverage"], errors="coerce"
                )
                .fillna(0)
                .clip(0, 1)
                .astype("float32")
            )
    else:
        raise ValueError(f"unknown weather source: {source}")

    weather["transfer_temperature_gap"] = (
        weather["transfer_dest_tavg"] - weather["transfer_origin_tavg"]
    ).astype("float32")
    weather["transfer_precipitation_sum"] = (
        weather["transfer_origin_prcp"] + weather["transfer_dest_prcp"]
    ).astype("float32")
    weather["transfer_wind_max"] = np.maximum(
        weather["transfer_origin_wspd"], weather["transfer_dest_wspd"]
    ).astype("float32")
    for side in ("origin", "dest"):
        weather[f"transfer_{side}_temperature_anomaly"] = (
            weather[f"transfer_{side}_tavg"] - base[f"clim_{side}_tavg"]
        ).astype("float32")
        weather[f"transfer_{side}_precipitation_excess"] = (
            weather[f"transfer_{side}_prcp"] - base[f"clim_{side}_prcp"]
        ).astype("float32")
        weather[f"transfer_{side}_wind_excess"] = (
            weather[f"transfer_{side}_wspd"] - base[f"clim_{side}_wspd"]
        ).astype("float32")
    matrix = pd.concat([base, pd.DataFrame(weather, index=base.index)], axis=1, copy=False)
    numeric = matrix.drop(columns=list(CATEGORICAL_FEATURES)).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("weather-transfer model matrix contains non-finite values")
    return matrix


@dataclass(slots=True)
class WeatherTransferCatBoostModel:
    estimator: Any
    task: TaskName
    include_cross_direction: bool

    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        matrix = engineer_weather_transfer_features(
            frame,
            source="forecast24_inference",
            include_cross_direction=self.include_cross_direction,
        )
        for column in CATEGORICAL_FEATURES:
            matrix[column] = matrix[column].astype(str)
        return np.asarray(self.estimator.predict_proba(matrix), dtype=np.float64)


def fit_weather_transfer_catboost(
    train_frame: pd.DataFrame,
    train_labels: NDArray[np.int64],
    *,
    task: TaskName,
    params: dict[str, Any],
    include_cross_direction: bool,
    sample_weight: NDArray[np.float64] | None = None,
) -> WeatherTransferCatBoostModel:
    """Fit on past observed weather; inference is hard-wired to fixed-lead forecasts."""

    from catboost import CatBoostClassifier  # type: ignore[import-untyped]

    matrix = engineer_weather_transfer_features(
        train_frame,
        source="observed_training",
        include_cross_direction=include_cross_direction,
    )
    for column in CATEGORICAL_FEATURES:
        matrix[column] = matrix[column].astype(str)
    defaults: dict[str, Any] = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 600,
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
    estimator.fit(
        matrix,
        train_labels,
        cat_features=list(CATEGORICAL_FEATURES),
        sample_weight=sample_weight,
    )
    return WeatherTransferCatBoostModel(
        estimator=estimator,
        task=task,
        include_cross_direction=include_cross_direction,
    )


def weather_transfer_profile() -> dict[str, Any]:
    return {
        "availability_horizon_at_inference": AvailabilityHorizon.FORECAST_24H.name,
        "training_weather": "past realised daily airport weather",
        "inference_weather": "GFS fixed previous_day1 daily forecast",
        "shared_semantics": ["daily mean temperature", "daily precipitation", "daily mean wind"],
        "missing_forecast_fallback": "strict earlier-year airport-month climatology",
        "claim_limit": (
            "Observed-to-forecast substitution is a covariate-transfer assumption and is tested "
            "out of time; it is not evidence of causal weather effects or live-feed reliability."
        ),
    }


def engineer_forecast_residual_features(
    frame: pd.DataFrame,
    *,
    include_cross_direction: bool,
    include_weather: bool,
) -> pd.DataFrame:
    """Build outcome-blind covariates for a baseline-anchored residual model.

    The base risk itself is supplied to CatBoost through ``Pool.baseline`` rather
    than as an ordinary feature.  This makes every learned tree an additive
    correction to the frozen earlier-year model logit.
    """

    recent = engineer_recent_model_features(
        frame,
        include_cross_direction=include_cross_direction,
    )
    dynamic = [name for name in recent if name.startswith("recent_")]
    matrix = recent.loc[:, [*RESIDUAL_CONTEXT_FEATURES, *dynamic]].copy()
    annual_phase = (
        2.0 * np.pi * (pd.to_numeric(frame["DayOfYear"], errors="raise") - 1.0) / 365.25
    )
    for harmonic in (1, 2, 3, 4, 6, 12):
        matrix[f"annual_fourier_sin_{harmonic}"] = np.sin(harmonic * annual_phase).astype(
            "float32"
        )
        matrix[f"annual_fourier_cos_{harmonic}"] = np.cos(harmonic * annual_phase).astype(
            "float32"
        )
    if not include_weather:
        return matrix

    validate_predictors(FORECAST24_DAILY_FEATURES, AvailabilityHorizon.FORECAST_24H)
    missing = sorted(set(FORECAST24_DAILY_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"forecast residual model is missing features: {missing}")
    forecast = frame.loc[:, list(FORECAST24_DAILY_FEATURES)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    derived: dict[str, pd.Series | NDArray[np.float64]] = {}
    for side in ("origin", "dest"):
        prefix = f"forecast24_{side}"
        derived[f"{prefix}_temperature_range"] = (
            forecast[f"{prefix}_tmax"] - forecast[f"{prefix}_tmin"]
        )
        derived[f"{prefix}_temperature_anomaly"] = (
            forecast[f"{prefix}_tavg"] - recent[f"clim_{side}_tavg"]
        )
        derived[f"{prefix}_precipitation_anomaly"] = (
            forecast[f"{prefix}_prcp_sum"] - recent[f"clim_{side}_prcp"]
        )
        derived[f"{prefix}_wind_anomaly"] = (
            forecast[f"{prefix}_wspd_mean"] - recent[f"clim_{side}_wspd"]
        )
        derived[f"{prefix}_gust_excess"] = (
            forecast[f"{prefix}_gust_max"] - forecast[f"{prefix}_wspd_mean"]
        )
        derived[f"{prefix}_cape_log1p"] = np.log1p(
            forecast[f"{prefix}_cape_max"].clip(lower=0)
        )
        derived[f"{prefix}_precipitation_log1p"] = np.log1p(
            forecast[f"{prefix}_prcp_sum"].clip(lower=0)
        )

    for statistic in (
        "prcp_sum",
        "prcp_max",
        "rh_max",
        "cloud_max",
        "wspd_max",
        "gust_max",
        "cape_max",
    ):
        origin = forecast[f"forecast24_origin_{statistic}"]
        dest = forecast[f"forecast24_dest_{statistic}"]
        derived[f"forecast24_endpoints_{statistic}_max"] = np.maximum(origin, dest)
        derived[f"forecast24_endpoints_{statistic}_sum"] = origin + dest
    derived["forecast24_pressure_gap"] = (
        forecast["forecast24_origin_pressure_mean"]
        - forecast["forecast24_dest_pressure_mean"]
    )
    derived["forecast24_temperature_gap"] = (
        forecast["forecast24_dest_tavg"] - forecast["forecast24_origin_tavg"]
    )
    derived["forecast24_any_missing"] = np.maximum(
        forecast["forecast24_origin_missing"],
        forecast["forecast24_dest_missing"],
    )
    output = pd.concat(
        [matrix, forecast.astype("float32"), pd.DataFrame(derived, index=matrix.index)],
        axis=1,
        copy=False,
    )
    numeric = output.drop(columns=list(CATEGORICAL_FEATURES)).to_numpy(dtype=np.float64)
    if np.isinf(numeric).any():
        raise ValueError("forecast residual features contain infinite values")
    return output


def _binary_logit(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float64)


@dataclass(slots=True)
class ForecastResidualCatBoostModel:
    """Additive operational/weather correction anchored to an earlier-year model."""

    baseline_model: Any
    estimator: Any
    task: TaskName
    include_cross_direction: bool
    include_weather: bool

    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        from catboost import Pool

        baseline = np.asarray(self.baseline_model.predict_proba(frame), dtype=np.float64)[:, 1]
        matrix = engineer_forecast_residual_features(
            frame,
            include_cross_direction=self.include_cross_direction,
            include_weather=self.include_weather,
        )
        for column in CATEGORICAL_FEATURES:
            matrix[column] = matrix[column].astype(str)
        pool = Pool(
            matrix,
            cat_features=list(CATEGORICAL_FEATURES),
            baseline=_binary_logit(baseline),
        )
        return np.asarray(self.estimator.predict_proba(pool), dtype=np.float64)


def fit_forecast_residual_catboost(
    train_frame: pd.DataFrame,
    train_labels: NDArray[np.int64],
    *,
    baseline_model: Any,
    task: TaskName,
    include_cross_direction: bool,
    include_weather: bool,
    params: dict[str, Any] | None = None,
    validation_frame: pd.DataFrame | None = None,
    validation_labels: NDArray[np.int64] | None = None,
) -> ForecastResidualCatBoostModel:
    """Fit additive corrections while retaining the frozen base-model logit."""

    from catboost import CatBoostClassifier, Pool

    if task == "joint":
        raise ValueError("forecast residual adaptation currently supports binary tasks only")
    train_probabilities = np.asarray(
        baseline_model.predict_proba(train_frame),
        dtype=np.float64,
    )[:, 1]
    train_matrix = engineer_forecast_residual_features(
        train_frame,
        include_cross_direction=include_cross_direction,
        include_weather=include_weather,
    )
    for column in CATEGORICAL_FEATURES:
        train_matrix[column] = train_matrix[column].astype(str)
    train_pool = Pool(
        train_matrix,
        label=train_labels,
        cat_features=list(CATEGORICAL_FEATURES),
        baseline=_binary_logit(train_probabilities),
    )
    defaults: dict[str, Any] = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 700,
        "learning_rate": 0.035,
        "depth": 7,
        "l2_leaf_reg": 12.0,
        "random_strength": 0.25,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.4,
        "random_seed": 20260903,
        "task_type": "GPU",
        "devices": "0",
        "verbose": False,
        "allow_writing_files": False,
    }
    if params:
        defaults.update(params)
    if defaults.get("task_type") == "CPU":
        defaults.pop("devices", None)
    estimator = CatBoostClassifier(**defaults)
    fit_kwargs: dict[str, Any] = {"X": train_pool}
    if validation_frame is not None or validation_labels is not None:
        if validation_frame is None or validation_labels is None:
            raise ValueError("forecast residual validation frame and labels must be paired")
        valid_probabilities = np.asarray(
            baseline_model.predict_proba(validation_frame),
            dtype=np.float64,
        )[:, 1]
        valid_matrix = engineer_forecast_residual_features(
            validation_frame,
            include_cross_direction=include_cross_direction,
            include_weather=include_weather,
        )
        for column in CATEGORICAL_FEATURES:
            valid_matrix[column] = valid_matrix[column].astype(str)
        fit_kwargs.update(
            {
                "eval_set": Pool(
                    valid_matrix,
                    label=validation_labels,
                    cat_features=list(CATEGORICAL_FEATURES),
                    baseline=_binary_logit(valid_probabilities),
                ),
                "early_stopping_rounds": 75,
                "use_best_model": True,
            }
        )
    estimator.fit(**fit_kwargs)
    return ForecastResidualCatBoostModel(
        baseline_model=baseline_model,
        estimator=estimator,
        task=task,
        include_cross_direction=include_cross_direction,
        include_weather=include_weather,
    )


def forecast_residual_profile(*, include_weather: bool) -> dict[str, Any]:
    return {
        "working_name": "prior_anchored_forecast_residual_adaptation",
        "availability_horizon": AvailabilityHorizon.FORECAST_24H.name,
        "base_risk": "frozen earlier-year HMOP CatBoost logit",
        "optimization": "CatBoost additive correction initialized at the base-risk logit",
        "correction_inputs": (
            "closed-left operational shifts, Fourier calendar embedding, and fixed-lead "
            "forecast weather"
            if include_weather
            else "closed-left operational shifts and Fourier calendar embedding only"
        ),
        "forecast_weather_included": include_weather,
        "claim_limit": (
            "The weather contribution requires a forward-year comparison against the "
            "operational-only residual ablation; no causal effect is claimed."
        ),
    }
