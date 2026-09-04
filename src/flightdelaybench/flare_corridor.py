"""Cutoff-coherent great-circle corridor proxy features for FLARE-24."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .contracts import FLARE24_CORRIDOR_FEATURES
from .flare_weather import (
    scheduled_flight_times,
    select_latest_safe_forecast,
    validate_weather_cube,
)

_EARTH_RADIUS_KM = 6_371.0088


def coordinate_catalog(path: Path) -> dict[str, tuple[float, float]]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("airports")
    if not isinstance(records, list) or not records:
        raise ValueError("airport coordinate catalog contains no records")
    result = {
        str(record["iata"]): (float(record["latitude"]), float(record["longitude"]))
        for record in records
    }
    if len(result) != len(records):
        raise ValueError("airport coordinate catalog identifiers must be unique")
    coordinates = np.asarray(list(result.values()), dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError("airport coordinate catalog contains non-finite values")
    return result


def _unit_vector(latitude: float, longitude: float) -> NDArray[np.float64]:
    latitude_radians = np.deg2rad(latitude)
    longitude_radians = np.deg2rad(longitude)
    return np.asarray(
        [
            np.cos(latitude_radians) * np.cos(longitude_radians),
            np.cos(latitude_radians) * np.sin(longitude_radians),
            np.sin(latitude_radians),
        ],
        dtype=np.float64,
    )


def _great_circle_point(
    origin: tuple[float, float],
    destination: tuple[float, float],
    fraction: float,
) -> tuple[float, float]:
    start = _unit_vector(*origin)
    end = _unit_vector(*destination)
    omega = float(np.arccos(np.clip(np.dot(start, end), -1.0, 1.0)))
    if omega < 1e-12:
        vector = start
    else:
        vector = (
            np.sin((1.0 - fraction) * omega) / np.sin(omega) * start
            + np.sin(fraction * omega) / np.sin(omega) * end
        )
    vector /= np.linalg.norm(vector)
    latitude = float(np.rad2deg(np.arcsin(vector[2])))
    longitude = float(np.rad2deg(np.arctan2(vector[1], vector[0])))
    return latitude, longitude


def _nearest_airport(
    point: tuple[float, float],
    airport_names: NDArray[np.str_],
    airport_vectors: NDArray[np.float64],
) -> tuple[str, float]:
    vector = _unit_vector(*point)
    angles = np.arccos(np.clip(airport_vectors @ vector, -1.0, 1.0))
    nearest = int(np.argmin(angles))
    return str(airport_names[nearest]), float(angles[nearest] * _EARTH_RADIUS_KM)


def _route_proxy_map(
    flights: pd.DataFrame,
    *,
    airport_coordinates: dict[str, tuple[float, float]],
    fractions: tuple[float, ...],
) -> dict[tuple[str, str], tuple[tuple[str, float], ...]]:
    if not fractions or any(fraction <= 0.0 or fraction >= 1.0 for fraction in fractions):
        raise ValueError("corridor fractions must lie strictly between zero and one")
    airport_names = np.asarray(sorted(airport_coordinates), dtype=np.str_)
    airport_vectors = np.vstack(
        [_unit_vector(*airport_coordinates[name]) for name in airport_names]
    )
    routes = flights.loc[:, ["Origin", "Dest"]].drop_duplicates()
    result: dict[tuple[str, str], tuple[tuple[str, float], ...]] = {}
    for row in routes.itertuples(index=False):
        origin = str(row.Origin)
        destination = str(row.Dest)
        if origin not in airport_coordinates or destination not in airport_coordinates:
            raise ValueError(f"corridor coordinates are missing for {origin}-{destination}")
        proxies = tuple(
            _nearest_airport(
                _great_circle_point(
                    airport_coordinates[origin], airport_coordinates[destination], fraction
                ),
                airport_names,
                airport_vectors,
            )
            for fraction in fractions
        )
        result[(origin, destination)] = proxies
    return result


def _convective(cape: NDArray[np.float64], precipitation: NDArray[np.float64], gust: NDArray[np.float64]) -> NDArray[np.float64]:
    valid = np.isfinite(cape) & np.isfinite(precipitation) & np.isfinite(gust)
    result = np.full(len(cape), np.nan, dtype=np.float64)
    result[valid] = (
        np.log1p(np.maximum(cape[valid], 0.0))
        * np.log1p(np.maximum(precipitation[valid], 0.0))
        * (1.0 + np.maximum(gust[valid], 0.0) / 100.0)
    )
    return result


def _icing(
    temperature: NDArray[np.float64],
    precipitation: NDArray[np.float64],
    snowfall: NDArray[np.float64],
    freezing_level: NDArray[np.float64],
) -> NDArray[np.float64]:
    valid = np.isfinite(temperature) & (np.isfinite(precipitation) | np.isfinite(snowfall))
    moisture = np.nan_to_num(np.maximum(precipitation, 0.0), nan=0.0) + 10.0 * np.nan_to_num(
        np.maximum(snowfall, 0.0), nan=0.0
    )
    cold = np.clip((3.0 - temperature) / 8.0, 0.0, 1.0)
    level = np.where(
        np.isfinite(freezing_level),
        np.clip((3_000.0 - freezing_level) / 2_500.0, 0.0, 1.0),
        0.5,
    )
    result = np.full(len(temperature), np.nan, dtype=np.float64)
    result[valid] = cold[valid] * (1.0 - np.exp(-moisture[valid])) * level[valid]
    return result


def attach_corridor_weather_features(
    flights: pd.DataFrame,
    weather_cube: pd.DataFrame,
    *,
    timezone_by_airport: dict[str, str],
    airport_coordinates: dict[str, tuple[float, float]],
    fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> pd.DataFrame:
    """Attach spatially explicit nearest-airport proxies along great-circle routes."""

    required = {"FlightDate", "Origin", "Dest", "CRSDepMinutes", "CRSElapsedTime"}
    missing = sorted(required - set(flights.columns))
    if missing:
        raise ValueError(f"corridor schedule is missing columns: {missing}")
    validate_weather_cube(weather_cube)
    route_map = _route_proxy_map(
        flights,
        airport_coordinates=airport_coordinates,
        fractions=fractions,
    )
    schedule = scheduled_flight_times(
        flights,
        timezone_by_airport=timezone_by_airport,
    )
    origins = flights["Origin"].astype(str).to_numpy()
    destinations = flights["Dest"].astype(str).to_numpy()
    row_id = np.arange(len(flights), dtype=np.int64)
    snapshots: list[pd.DataFrame] = []
    distances: list[NDArray[np.float64]] = []
    proxy_airports: list[NDArray[np.str_]] = []
    for fraction_index, fraction in enumerate(fractions):
        proxy = np.asarray(
            [route_map[(origin, dest)][fraction_index][0] for origin, dest in zip(origins, destinations, strict=True)],
            dtype=np.str_,
        )
        distance = np.asarray(
            [route_map[(origin, dest)][fraction_index][1] for origin, dest in zip(origins, destinations, strict=True)],
            dtype=np.float64,
        )
        target_time = schedule["departure_time_utc"] + (
            schedule["arrival_time_utc"] - schedule["departure_time_utc"]
        ) * fraction
        targets = pd.DataFrame(
            {
                "row_id": row_id,
                "Airport": proxy,
                "target_time_utc": target_time,
                "cutoff_time_utc": schedule["cutoff_time_utc"],
            }
        )
        snapshots.append(
            select_latest_safe_forecast(
                targets,
                weather_cube,
                _cube_validated=True,
            )
        )
        distances.append(distance)
        proxy_airports.append(proxy)

    def matrix(variable: str) -> NDArray[np.float64]:
        return np.column_stack(
            [
                pd.to_numeric(snapshot[variable], errors="coerce").to_numpy(dtype=np.float64)
                for snapshot in snapshots
            ]
        )

    cape = matrix("cape")
    precipitation = matrix("precipitation")
    gust = matrix("wind_gust")
    visibility = matrix("visibility")
    freezing = matrix("freezing_level")
    temperature = matrix("temperature")
    snowfall = matrix("snowfall")
    convective = np.column_stack(
        [
            _convective(cape[:, index], precipitation[:, index], gust[:, index])
            for index in range(len(fractions))
        ]
    )
    icing = np.column_stack(
        [
            _icing(
                temperature[:, index],
                precipitation[:, index],
                snowfall[:, index],
                freezing[:, index],
            )
            for index in range(len(fractions))
        ]
    )

    def complete_reduction(values: NDArray[np.float64], reduction: str) -> NDArray[np.float64]:
        available = np.isfinite(values)
        result = np.full(len(values), np.nan, dtype=np.float64)
        rows = available.any(axis=1)
        if reduction == "max":
            result[rows] = np.nanmax(values[rows], axis=1)
        elif reduction == "min":
            result[rows] = np.nanmin(values[rows], axis=1)
        else:
            raise ValueError(f"unknown corridor reduction: {reduction}")
        return result

    distance_matrix = np.column_stack(distances)
    proxy_matrix = np.column_stack(proxy_airports)
    coverage = np.column_stack(
        [snapshot["issue_time_utc"].notna().to_numpy(dtype=np.float64) for snapshot in snapshots]
    ).mean(axis=1)
    proxy_counts = np.asarray(
        [len(set(row)) for row in proxy_matrix], dtype=np.float64
    )
    features = pd.DataFrame(
        {
            "flare24_route_corridor_proxy_cape_max": complete_reduction(cape, "max"),
            "flare24_route_corridor_proxy_precipitation_max": complete_reduction(
                precipitation, "max"
            ),
            "flare24_route_corridor_proxy_wind_gust_max": complete_reduction(gust, "max"),
            "flare24_route_corridor_proxy_visibility_min": complete_reduction(
                visibility, "min"
            ),
            "flare24_route_corridor_proxy_freezing_level_min": complete_reduction(
                freezing, "min"
            ),
            "flare24_route_corridor_proxy_convective_index_max": complete_reduction(
                convective, "max"
            ),
            "flare24_route_corridor_proxy_icing_index_max": complete_reduction(
                icing, "max"
            ),
            "flare24_route_corridor_proxy_coverage": coverage,
            "flare24_route_corridor_proxy_distance_mean_km": distance_matrix.mean(axis=1),
            "flare24_route_corridor_proxy_distance_max_km": distance_matrix.max(axis=1),
            "flare24_route_corridor_proxy_airport_count": proxy_counts,
        },
        index=flights.index,
    )
    if tuple(features.columns) != FLARE24_CORRIDOR_FEATURES:
        raise AssertionError("corridor feature contract mismatch")
    return pd.concat([flights.copy(), features], axis=1, copy=False)
