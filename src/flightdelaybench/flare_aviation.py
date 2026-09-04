"""Aviation-aware transformations of cutoff-coherent FLARE-24 weather."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .contracts import FLARE24_AVIATION_WEATHER_FEATURES

_KMH_PER_KNOT = 1.852


def runway_heading_from_identifier(identifier: str) -> float:
    """Convert an FAA-style runway-end identifier (for example 09L) to degrees."""

    text = str(identifier).strip().upper()
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) not in {1, 2}:
        raise ValueError(f"invalid runway-end identifier: {identifier}")
    number = int(digits)
    if number < 1 or number > 36:
        raise ValueError(f"invalid runway-end number: {identifier}")
    return float(360 if number == 36 else number * 10)


def _wind_optimal_components(
    airports: pd.Series,
    direction_degrees: pd.Series,
    speed_kmh: pd.Series,
    gust_kmh: pd.Series,
    runway_headings_by_airport: dict[str, tuple[float, ...]],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int8],
]:
    count = len(airports)
    headwind = np.full(count, np.nan, dtype=np.float64)
    crosswind = np.full(count, np.nan, dtype=np.float64)
    gust_crosswind = np.full(count, np.nan, dtype=np.float64)
    available = np.zeros(count, dtype=np.int8)
    airport_values = airports.astype("string").to_numpy()
    direction = pd.to_numeric(direction_degrees, errors="coerce").to_numpy(
        dtype=np.float64
    )
    speed = pd.to_numeric(speed_kmh, errors="coerce").to_numpy(dtype=np.float64)
    gust = pd.to_numeric(gust_kmh, errors="coerce").to_numpy(dtype=np.float64)
    for airport in pd.unique(airport_values):
        positions = np.flatnonzero(airport_values == airport).astype(np.int64, copy=False)
        headings = np.asarray(
            runway_headings_by_airport.get(str(airport), ()), dtype=np.float64
        )
        headings = np.unique(headings[np.isfinite(headings)])
        if headings.size == 0:
            continue
        if ((headings <= 0.0) | (headings > 360.0)).any():
            raise ValueError(f"runway headings for {airport} must be in (0, 360]")
        valid = np.isfinite(direction[positions]) & np.isfinite(speed[positions])
        if not valid.any():
            available[positions] = 1
            continue
        valid_positions = positions[valid]
        angle = np.deg2rad(
            direction[valid_positions, None] - headings[None, :]
        )
        speed_knots = speed[valid_positions, None] / _KMH_PER_KNOT
        headwind_options = speed_knots * np.cos(angle)
        chosen = np.argmax(headwind_options, axis=1)
        row = np.arange(len(valid_positions))
        headwind[valid_positions] = headwind_options[row, chosen]
        crosswind_factor = np.abs(np.sin(angle[row, chosen]))
        crosswind[valid_positions] = speed[valid_positions] / _KMH_PER_KNOT * crosswind_factor
        valid_gust = np.isfinite(gust[valid_positions])
        gust_crosswind[valid_positions[valid_gust]] = (
            gust[valid_positions[valid_gust]]
            / _KMH_PER_KNOT
            * crosswind_factor[valid_gust]
        )
        available[positions] = 1
    return headwind, crosswind, gust_crosswind, available


def _visibility_hazard(visibility_metres: pd.Series) -> NDArray[np.float64]:
    visibility = pd.to_numeric(visibility_metres, errors="coerce").to_numpy(
        dtype=np.float64
    )
    hazard = np.full(len(visibility), np.nan, dtype=np.float64)
    valid = np.isfinite(visibility)
    hazard[valid & (visibility >= 8_000.0)] = 0.0
    hazard[valid & (visibility < 8_000.0) & (visibility >= 4_828.0)] = 1.0
    hazard[valid & (visibility < 4_828.0) & (visibility >= 1_609.0)] = 2.0
    hazard[valid & (visibility < 1_609.0)] = 3.0
    return hazard


def _convective_index(
    cape_values: pd.Series,
    precipitation_values: pd.Series,
    gust_values: pd.Series,
) -> NDArray[np.float64]:
    cape = pd.to_numeric(cape_values, errors="coerce").to_numpy(dtype=np.float64)
    precipitation = pd.to_numeric(
        precipitation_values, errors="coerce"
    ).to_numpy(dtype=np.float64)
    gust = pd.to_numeric(gust_values, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(cape) & np.isfinite(precipitation) & np.isfinite(gust)
    result = np.full(len(cape), np.nan, dtype=np.float64)
    result[valid] = (
        np.log1p(np.maximum(cape[valid], 0.0))
        * np.log1p(np.maximum(precipitation[valid], 0.0))
        * (1.0 + np.maximum(gust[valid], 0.0) / 100.0)
    )
    return result


def _icing_index(
    temperature_values: pd.Series,
    precipitation_values: pd.Series,
    snowfall_values: pd.Series,
    freezing_level_values: pd.Series,
) -> NDArray[np.float64]:
    temperature = pd.to_numeric(temperature_values, errors="coerce").to_numpy(
        dtype=np.float64
    )
    precipitation = pd.to_numeric(
        precipitation_values, errors="coerce"
    ).to_numpy(dtype=np.float64)
    snowfall = pd.to_numeric(snowfall_values, errors="coerce").to_numpy(
        dtype=np.float64
    )
    freezing_level = pd.to_numeric(
        freezing_level_values, errors="coerce"
    ).to_numpy(dtype=np.float64)
    valid = np.isfinite(temperature) & (
        np.isfinite(precipitation) | np.isfinite(snowfall)
    )
    moisture = np.nan_to_num(np.maximum(precipitation, 0.0), nan=0.0) + 10.0 * np.nan_to_num(
        np.maximum(snowfall, 0.0), nan=0.0
    )
    cold_factor = np.clip((3.0 - temperature) / 8.0, 0.0, 1.0)
    freezing_factor = np.where(
        np.isfinite(freezing_level),
        np.clip((3_000.0 - freezing_level) / 2_500.0, 0.0, 1.0),
        0.5,
    )
    result = np.full(len(temperature), np.nan, dtype=np.float64)
    result[valid] = (
        cold_factor[valid]
        * (1.0 - np.exp(-moisture[valid]))
        * freezing_factor[valid]
    )
    return result


def _side_features(
    frame: pd.DataFrame,
    *,
    side: str,
    airport_column: str,
    runway_headings_by_airport: dict[str, tuple[float, ...]],
) -> dict[str, NDArray[np.float64] | NDArray[np.int8]]:
    prefix = f"flare24_{side}"
    source = f"flare24_{side}_{'departure' if side == 'origin' else 'arrival'}"
    headwind, crosswind, gust_crosswind, runway_available = _wind_optimal_components(
        frame[airport_column],
        frame[f"{source}_wind_direction"],
        frame[f"{source}_wind_speed"],
        frame[f"{source}_wind_gust"],
        runway_headings_by_airport,
    )
    visibility = pd.to_numeric(
        frame[f"{source}_visibility"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    visibility_hazard = _visibility_hazard(frame[f"{source}_visibility"])
    convective = _convective_index(
        frame[f"{source}_cape"],
        frame[f"{source}_precipitation"],
        frame[f"{source}_wind_gust"],
    )
    icing = _icing_index(
        frame[f"{source}_temperature"],
        frame[f"{source}_precipitation"],
        frame[f"{source}_snowfall"],
        frame[f"{source}_freezing_level"],
    )
    snowfall = pd.to_numeric(
        frame[f"{source}_snowfall"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    gust = pd.to_numeric(frame[f"{source}_wind_gust"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    speed = pd.to_numeric(frame[f"{source}_wind_speed"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    return {
        f"{prefix}_wind_optimal_headwind_knots": headwind,
        f"{prefix}_wind_optimal_crosswind_knots": crosswind,
        f"{prefix}_wind_optimal_gust_crosswind_knots": gust_crosswind,
        f"{prefix}_runway_heading_available": runway_available,
        f"{prefix}_visibility_hazard": visibility_hazard,
        f"{prefix}_low_visibility": np.where(
            np.isfinite(visibility), (visibility < 4_828.0).astype(np.float64), np.nan
        ),
        f"{prefix}_severe_visibility": np.where(
            np.isfinite(visibility), (visibility < 1_609.0).astype(np.float64), np.nan
        ),
        f"{prefix}_convective_index": convective,
        f"{prefix}_icing_environment_index": icing,
        f"{prefix}_snow_intensity": np.where(
            np.isfinite(snowfall), np.log1p(np.maximum(snowfall, 0.0)), np.nan
        ),
        f"{prefix}_gust_excess_knots": np.where(
            np.isfinite(gust) & np.isfinite(speed),
            np.maximum(gust - speed, 0.0) / _KMH_PER_KNOT,
            np.nan,
        ),
    }


def attach_aviation_weather_features(
    frame: pd.DataFrame,
    *,
    runway_headings_by_airport: dict[str, tuple[float, ...]],
) -> pd.DataFrame:
    """Attach deterministic aviation transformations without reading outcomes."""

    required = {"Origin", "Dest"}
    for side, event in (("origin", "departure"), ("dest", "arrival")):
        required.update(
            f"flare24_{side}_{event}_{variable}"
            for variable in (
                "temperature",
                "precipitation",
                "wind_speed",
                "wind_gust",
                "wind_direction",
                "cape",
                "visibility",
                "snowfall",
                "freezing_level",
            )
        )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"aviation weather input is missing columns: {missing}")
    features: dict[str, NDArray[np.float64] | NDArray[np.int8]] = {}
    features.update(
        _side_features(
            frame,
            side="origin",
            airport_column="Origin",
            runway_headings_by_airport=runway_headings_by_airport,
        )
    )
    features.update(
        _side_features(
            frame,
            side="dest",
            airport_column="Dest",
            runway_headings_by_airport=runway_headings_by_airport,
        )
    )

    def endpoint_max(origin_name: str, dest_name: str) -> NDArray[np.float64]:
        pair = np.column_stack(
            (
                np.asarray(features[origin_name], dtype=np.float64),
                np.asarray(features[dest_name], dtype=np.float64),
            )
        )
        valid = np.isfinite(pair).all(axis=1)
        result = np.full(len(frame), np.nan, dtype=np.float64)
        result[valid] = pair[valid].max(axis=1)
        return result

    features["flare24_route_endpoint_convective_max"] = endpoint_max(
        "flare24_origin_convective_index", "flare24_dest_convective_index"
    )
    features["flare24_route_endpoint_icing_max"] = endpoint_max(
        "flare24_origin_icing_environment_index",
        "flare24_dest_icing_environment_index",
    )
    features["flare24_route_endpoint_crosswind_max"] = endpoint_max(
        "flare24_origin_wind_optimal_gust_crosswind_knots",
        "flare24_dest_wind_optimal_gust_crosswind_knots",
    )
    features["flare24_route_endpoint_visibility_hazard_max"] = endpoint_max(
        "flare24_origin_visibility_hazard", "flare24_dest_visibility_hazard"
    )

    schedule_defaults = pd.Series(0.0, index=frame.index, dtype="float64")
    origin_bank = pd.to_numeric(
        frame.get("schedule_origin_departure_bank_log1p", schedule_defaults),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    dest_inbound = pd.to_numeric(
        frame.get("schedule_dest_inbound_day_log1p", schedule_defaults),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    features["flare24_origin_convective_bank_load"] = (
        np.asarray(features["flare24_origin_convective_index"], dtype=np.float64)
        * origin_bank
    )
    features["flare24_origin_crosswind_bank_load"] = (
        np.asarray(
            features["flare24_origin_wind_optimal_gust_crosswind_knots"],
            dtype=np.float64,
        )
        * origin_bank
    )
    features["flare24_dest_visibility_inbound_load"] = (
        np.asarray(features["flare24_dest_visibility_hazard"], dtype=np.float64)
        * dest_inbound
    )
    features["flare24_dest_icing_inbound_load"] = (
        np.asarray(features["flare24_dest_icing_environment_index"], dtype=np.float64)
        * dest_inbound
    )
    feature_frame = pd.DataFrame(features, index=frame.index)
    if tuple(feature_frame.columns) != FLARE24_AVIATION_WEATHER_FEATURES:
        missing_features = sorted(
            set(FLARE24_AVIATION_WEATHER_FEATURES) - set(feature_frame.columns)
        )
        extras = sorted(set(feature_frame.columns) - set(FLARE24_AVIATION_WEATHER_FEATURES))
        if missing_features or extras:
            raise AssertionError(
                f"aviation feature contract mismatch; missing={missing_features}, extra={extras}"
            )
        feature_frame = feature_frame.loc[:, list(FLARE24_AVIATION_WEATHER_FEATURES)]
    return pd.concat([frame.copy(), feature_frame], axis=1, copy=False)
