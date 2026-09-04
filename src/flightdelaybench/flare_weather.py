"""Issuance-aware, schedule-aligned weather features for FLARE-24.

The legacy weather track summarizes forecasts by local calendar day.  This
module keeps valid time and issue time separate and selects a weather record
only when ``issue_time_utc <= flight_cutoff_utc``.  It therefore supports a
single flight-specific information set for departure, arrival, and route
targets even when those targets require different forecast lead times.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    FLARE24_FORECAST_REVISION_VARIABLES,
    FLARE24_WEATHER_FEATURES,
    FLARE24_WEATHER_VARIABLES,
)
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

OPEN_METEO_VARIABLES: dict[str, str] = {
    "temperature": "temperature_2m",
    "precipitation": "precipitation",
    "humidity": "relative_humidity_2m",
    "cloud": "cloud_cover",
    "pressure": "surface_pressure",
    "wind_speed": "wind_speed_10m",
    "wind_gust": "wind_gusts_10m",
    "cape": "cape",
    "wind_direction": "wind_direction_10m",
    "visibility": "visibility",
    "ceiling": "cloud_ceiling",
    "snowfall": "snowfall",
    "freezing_level": "freezing_level_height",
    "reflectivity": "composite_reflectivity",
}
OPEN_METEO_UNITS: dict[str, str] = {
    "temperature": "°C",
    "precipitation": "mm",
    "humidity": "%",
    "cloud": "%",
    "pressure": "hPa",
    "wind_speed": "km/h",
    "wind_gust": "km/h",
    "cape": "J/kg",
    "wind_direction": "°",
    "visibility": "m",
    "ceiling": "m",
    "snowfall": "cm",
    "freezing_level": "m",
    "reflectivity": "dBZ",
}

WEATHER_CUBE_REQUIRED_COLUMNS = (
    "Airport",
    "valid_time_utc",
    "issue_time_utc",
    "lead_hours",
    *FLARE24_WEATHER_VARIABLES,
)


def _verify_self_hashed_manifest(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("manifest_sha256")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"manifest self-hash failed: {path}")
    return payload


def _local_wall_times_to_utc(
    values: Sequence[str],
    timezone: str,
) -> tuple[pd.Series, int]:
    """Resolve provider-local wall times conservatively.

    Open-Meteo returns a regular local-hour grid.  Spring-forward wall times
    that do not exist are converted to missing rather than silently shifted.
    For a fall-back ambiguous hour, the first occurrence is selected and the
    policy is recorded in the manifest.
    """

    local = pd.DatetimeIndex(pd.to_datetime(pd.Series(values, dtype="string"), errors="raise"))
    aware = local.tz_localize(timezone, ambiguous=True, nonexistent="NaT")
    invalid = int(aware.isna().sum())
    utc = aware.tz_convert("UTC").tz_localize(None)
    return pd.Series(utc, dtype="datetime64[ns]"), invalid


def _numeric_hourly(
    hourly: dict[str, list[Any]],
    *,
    stem: str,
    lead_suffix: str,
    length: int,
) -> pd.Series:
    key = f"{stem}_{lead_suffix}"
    if key not in hourly:
        return pd.Series(np.full(length, np.nan), dtype="float64")
    if len(hourly[key]) != length:
        raise ValueError(f"hourly variable length mismatch: {key}")
    return pd.to_numeric(pd.Series(hourly[key], dtype="object"), errors="coerce")


def normalize_open_meteo_request(
    path: Path,
    *,
    airport: str,
    lead_suffix: str,
    lead_hours: int,
) -> tuple[pd.DataFrame, int]:
    """Normalize one preserved Open-Meteo response into the weather cube."""

    if lead_hours <= 0:
        raise ValueError("lead_hours must be positive")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    hourly = payload.get("hourly")
    hourly_units = payload.get("hourly_units")
    timezone = payload.get("timezone")
    if (
        not isinstance(hourly, dict)
        or not isinstance(hourly_units, dict)
        or not isinstance(timezone, str)
    ):
        raise ValueError(f"invalid Open-Meteo response: {path}")
    raw_times = hourly.get("time")
    if not isinstance(raw_times, list) or not raw_times:
        raise ValueError(f"Open-Meteo response has no hourly timestamps: {path}")
    valid_time, invalid_local_times = _local_wall_times_to_utc(raw_times, timezone)
    count = len(valid_time)
    data: dict[str, Any] = {
        "Airport": pd.Series([airport] * count, dtype="string"),
        "valid_time_utc": valid_time,
        "issue_time_utc": valid_time - pd.to_timedelta(lead_hours, unit="h"),
        "lead_hours": np.full(count, float(lead_hours), dtype=np.float32),
    }
    for output_name, source_stem in OPEN_METEO_VARIABLES.items():
        source_name = f"{source_stem}_{lead_suffix}"
        if source_name in hourly and hourly_units.get(source_name) != OPEN_METEO_UNITS[output_name]:
            raise ValueError(
                f"unexpected Open-Meteo unit for {source_name}: {hourly_units.get(source_name)}"
            )
        data[output_name] = _numeric_hourly(
            hourly,
            stem=source_stem,
            lead_suffix=lead_suffix,
            length=count,
        ).astype("float32")
    frame = pd.DataFrame(data)
    frame = frame.loc[frame["valid_time_utc"].notna()].reset_index(drop=True)
    validate_weather_cube(frame)
    return frame, invalid_local_times


def validate_weather_cube(frame: pd.DataFrame) -> dict[str, int | float]:
    """Validate temporal semantics and uniqueness of an issuance-aware cube."""

    missing = sorted(set(WEATHER_CUBE_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"weather cube is missing columns: {missing}")
    if frame.empty:
        raise ValueError("weather cube is empty")
    valid = pd.to_datetime(frame["valid_time_utc"], errors="raise")
    issue = pd.to_datetime(frame["issue_time_utc"], errors="raise")
    lead = pd.to_numeric(frame["lead_hours"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(lead).all() or (lead <= 0).any():
        raise ValueError("weather lead_hours must be finite and positive")
    observed_lead = (valid - issue).dt.total_seconds().to_numpy(dtype=np.float64) / 3600.0
    if not np.allclose(observed_lead, lead, rtol=0.0, atol=1e-6):
        raise ValueError("weather issue, valid, and lead timestamps are inconsistent")
    if (issue > valid).any():
        raise ValueError("weather issue time occurs after valid time")
    duplicate = frame.duplicated(["Airport", "valid_time_utc", "issue_time_utc"])
    if duplicate.any():
        raise ValueError("weather cube contains duplicate airport-valid-issue rows")
    available_values = int(frame.loc[:, list(FLARE24_WEATHER_VARIABLES)].notna().sum().sum())
    if available_values == 0:
        raise ValueError("weather cube has no numeric weather values")
    return {
        "rows": len(frame),
        "airports": int(frame["Airport"].nunique()),
        "minimum_lead_hours": float(lead.min()),
        "maximum_lead_hours": float(lead.max()),
        "available_weather_values": available_values,
    }


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite FLARE-24 weather cube: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated partial weather cube exists: {partial}")
    frame.to_parquet(partial, index=False, compression="zstd")
    partial.replace(path)
    return {
        "path": path.as_posix(),
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_open_meteo_weather_cube(
    *,
    acquisition_manifests: tuple[Path, ...],
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Build a versioned weather cube from one or more fixed-lead archives."""

    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite FLARE-24 manifest: {manifest_path}")
    if not acquisition_manifests:
        raise ValueError("at least one acquisition manifest is required")
    started = time.perf_counter()
    parts_by_year: dict[int, list[pd.DataFrame]] = {}
    sources: list[dict[str, Any]] = []
    invalid_local_times = 0
    for manifest_path_in in acquisition_manifests:
        acquisition = _verify_self_hashed_manifest(manifest_path_in)
        lead_hours = int(acquisition.get("fixed_lead_hours", -1))
        lead_suffix = str(acquisition.get("lead_suffix", ""))
        if lead_hours <= 0 or not lead_suffix:
            raise ValueError(f"acquisition lacks a fixed-lead contract: {manifest_path_in}")
        source_records: list[dict[str, Any]] = []
        for record in acquisition.get("requests", []):
            raw_path = Path(record["path"])
            actual_hash = sha256_file(raw_path)
            if actual_hash != record["sha256"]:
                raise ValueError(f"raw weather checksum failed: {raw_path}")
            year = int(record["year"])
            normalized, invalid = normalize_open_meteo_request(
                raw_path,
                airport=str(record["airport"]),
                lead_suffix=lead_suffix,
                lead_hours=lead_hours,
            )
            parts_by_year.setdefault(year, []).append(normalized)
            invalid_local_times += invalid
            source_records.append(
                {
                    "airport": str(record["airport"]),
                    "year": year,
                    "path": raw_path.as_posix(),
                    "sha256": actual_hash,
                }
            )
        sources.append(
            {
                "manifest": manifest_path_in.as_posix(),
                "sha256": sha256_file(manifest_path_in),
                "self_hash": acquisition["manifest_sha256"],
                "lead_hours": lead_hours,
                "lead_suffix": lead_suffix,
                "records": source_records,
            }
        )

    outputs: list[dict[str, Any]] = []
    validation: dict[str, Any] = {}
    for year, parts in sorted(parts_by_year.items()):
        frame = pd.concat(parts, ignore_index=True)
        frame = frame.sort_values(
            ["Airport", "valid_time_utc", "issue_time_utc"], kind="mergesort"
        ).reset_index(drop=True)
        validation[str(year)] = validate_weather_cube(frame)
        outputs.append(_atomic_parquet(frame, output_dir / f"year={year}.parquet"))

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "method": "FLARE-24_CUTOFF_COHERENT_WEATHER_CUBE",
        "status": "COMPLETE_COVARIATE_ONLY_NO_FLIGHT_OUTCOMES_ACCESSED",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "sources": sources,
        "outputs": outputs,
        "validation": validation,
        "invalid_nonexistent_local_hours_excluded": invalid_local_times,
        "ambiguous_local_hour_policy": "first occurrence; provider grid contains one wall-clock row",
        "variable_selection_policy": (
            "freshest non-missing value for each variable subject to issue_time_utc <= cutoff; "
            "the reported selected lead is the minimum lead among contributing variables"
        ),
        "feature_contract": list(FLARE24_WEATHER_FEATURES),
        "unit_contract": OPEN_METEO_UNITS,
        "environment": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "pyarrow": version("pyarrow"),
        },
        "provenance": capture_provenance((Path(__file__), Path(__file__).with_name("contracts.py"))),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "The cube records archived forecast issue/valid semantics. A flight feature is "
            "deployable only after the flight-specific selector proves issue_time_utc <= cutoff."
        ),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(manifest_path, manifest)
    return manifest


def load_weather_cube(weather_dir: Path, years: tuple[int, ...]) -> pd.DataFrame:
    if not years:
        raise ValueError("at least one weather year is required")
    frames: list[pd.DataFrame] = []
    for year in years:
        path = weather_dir / f"year={year}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing weather cube partition: {path}")
        frames.append(pd.read_parquet(path))
    frame = pd.concat(frames, ignore_index=True)
    validate_weather_cube(frame)
    return frame


def _localized_utc(
    local_times: pd.Series,
    airports: pd.Series,
    timezone_by_airport: dict[str, str],
) -> pd.Series:
    result = pd.Series(pd.NaT, index=local_times.index, dtype="datetime64[ns]")
    airport_values = airports.astype("string")
    missing_airports = sorted(set(airport_values.dropna().astype(str)) - set(timezone_by_airport))
    if missing_airports:
        raise ValueError(f"airport timezones are missing: {missing_airports}")
    for airport, indexes in airport_values.groupby(airport_values, sort=False).groups.items():
        timezone = timezone_by_airport[str(airport)]
        values = pd.DatetimeIndex(pd.to_datetime(local_times.loc[indexes], errors="raise"))
        aware = values.tz_localize(timezone, ambiguous=True, nonexistent="NaT")
        result.loc[indexes] = aware.tz_convert("UTC").tz_localize(None).to_numpy()
    return result


def scheduled_flight_times(
    flights: pd.DataFrame,
    *,
    timezone_by_airport: dict[str, str],
    cutoff_hours: int = 24,
) -> pd.DataFrame:
    """Construct departure, arrival, and cutoff timestamps without outcomes."""

    required = {"FlightDate", "Origin", "Dest", "CRSDepMinutes", "CRSElapsedTime"}
    missing = sorted(required - set(flights.columns))
    if missing:
        raise ValueError(f"flight schedule is missing columns: {missing}")
    if cutoff_hours <= 0:
        raise ValueError("cutoff_hours must be positive")
    dates = pd.to_datetime(flights["FlightDate"], errors="raise").dt.normalize()
    departure_minutes = pd.to_numeric(flights["CRSDepMinutes"], errors="raise")
    elapsed_minutes = pd.to_numeric(flights["CRSElapsedTime"], errors="coerce")
    local_departure = dates + pd.to_timedelta(departure_minutes, unit="m")
    departure_utc = _localized_utc(
        local_departure,
        flights["Origin"],
        timezone_by_airport,
    )
    arrival_utc = departure_utc + pd.to_timedelta(elapsed_minutes, unit="m")
    return pd.DataFrame(
        {
            "departure_time_utc": departure_utc,
            "arrival_time_utc": arrival_utc,
            "cutoff_time_utc": departure_utc - pd.to_timedelta(cutoff_hours, unit="h"),
        },
        index=flights.index,
    )


def select_latest_safe_forecast(
    targets: pd.DataFrame,
    weather_cube: pd.DataFrame,
    *,
    _cube_validated: bool = False,
) -> pd.DataFrame:
    """Select the freshest exact-valid-hour forecast available at each cutoff."""

    required = {"row_id", "Airport", "target_time_utc", "cutoff_time_utc"}
    missing = sorted(required - set(targets.columns))
    if missing:
        raise ValueError(f"weather targets are missing columns: {missing}")
    if targets["row_id"].duplicated().any():
        raise ValueError("weather target row_id values must be unique")
    if not _cube_validated:
        validate_weather_cube(weather_cube)
    target = targets.copy()
    target["target_time_utc"] = pd.to_datetime(target["target_time_utc"], errors="coerce")
    target["cutoff_time_utc"] = pd.to_datetime(target["cutoff_time_utc"], errors="coerce")
    target["target_valid_time_utc"] = target["target_time_utc"].dt.floor("h")
    weather_columns = [
        "Airport",
        "valid_time_utc",
        "issue_time_utc",
        "lead_hours",
        *FLARE24_WEATHER_VARIABLES,
    ]
    candidates = target.merge(
        weather_cube.loc[:, weather_columns],
        how="left",
        left_on=["Airport", "target_valid_time_utc"],
        right_on=["Airport", "valid_time_utc"],
        sort=False,
    )
    eligible = candidates.loc[
        candidates["issue_time_utc"].notna()
        & candidates["cutoff_time_utc"].notna()
        & candidates["issue_time_utc"].le(candidates["cutoff_time_utc"])
    ].copy()
    if eligible.empty:
        selected = pd.DataFrame(
            columns=(
                "row_id",
                "valid_time_utc",
                "issue_time_utc",
                "lead_hours",
                *FLARE24_WEATHER_VARIABLES,
            )
        )
    else:
        ordered = eligible.sort_values(
            ["row_id", "issue_time_utc"], kind="mergesort"
        )
        grouped = ordered.groupby("row_id", sort=False, observed=True)
        selected_values = grouped[list(FLARE24_WEATHER_VARIABLES)].last()
        issue_by_variable = pd.DataFrame(
            {
                variable: ordered["issue_time_utc"].where(ordered[variable].notna())
                for variable in FLARE24_WEATHER_VARIABLES
            }
        ).assign(row_id=ordered["row_id"].to_numpy())
        freshest_issue = issue_by_variable.groupby("row_id", sort=False).max().max(axis=1)
        selected = selected_values.reset_index()
        selected["issue_time_utc"] = selected["row_id"].map(freshest_issue)
        selected["valid_time_utc"] = selected["row_id"].map(
            grouped["valid_time_utc"].last()
        )
        selected["lead_hours"] = (
            selected["valid_time_utc"] - selected["issue_time_utc"]
        ).dt.total_seconds() / 3600.0
    result = target.loc[:, ["row_id", "target_time_utc", "cutoff_time_utc"]].merge(
        selected,
        how="left",
        on="row_id",
        sort=False,
        validate="one_to_one",
    )
    unsafe = result["issue_time_utc"].notna() & result["issue_time_utc"].gt(
        result["cutoff_time_utc"]
    )
    if unsafe.any():
        raise AssertionError("selector admitted weather issued after the flight cutoff")
    return result.sort_values("row_id", kind="mergesort").reset_index(drop=True)


def _paired_complete(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.notna() & right.notna()


def attach_flare24_weather(
    flights: pd.DataFrame,
    weather_cube: pd.DataFrame,
    *,
    timezone_by_airport: dict[str, str],
    origin_window_hours: tuple[int, ...] = (-6, -3, 0),
) -> pd.DataFrame:
    """Attach FLARE-24 weather while enforcing one flight-specific cutoff."""

    if not origin_window_hours or 0 not in origin_window_hours:
        raise ValueError("origin_window_hours must include departure offset 0")
    if any(offset > 0 for offset in origin_window_hours):
        raise ValueError("predeparture window offsets cannot be positive")
    validate_weather_cube(weather_cube)
    schedule = scheduled_flight_times(flights, timezone_by_airport=timezone_by_airport)
    row_id = np.arange(len(flights), dtype=np.int64)
    snapshots: dict[int, pd.DataFrame] = {}
    origin_targets: dict[int, pd.DataFrame] = {}
    for offset in origin_window_hours:
        targets = pd.DataFrame(
            {
                "row_id": row_id,
                "Airport": flights["Origin"].astype("string").to_numpy(),
                "target_time_utc": schedule["departure_time_utc"]
                + pd.to_timedelta(offset, unit="h"),
                "cutoff_time_utc": schedule["cutoff_time_utc"],
            }
        )
        origin_targets[offset] = targets
        snapshots[offset] = select_latest_safe_forecast(
            targets, weather_cube, _cube_validated=True
        )
    destination = select_latest_safe_forecast(
        pd.DataFrame(
            {
                "row_id": row_id,
                "Airport": flights["Dest"].astype("string").to_numpy(),
                "target_time_utc": schedule["arrival_time_utc"],
                "cutoff_time_utc": schedule["cutoff_time_utc"],
            }
        ),
        weather_cube,
        _cube_validated=True,
    )
    current = snapshots[0]
    features: dict[str, Any] = {}
    for variable in FLARE24_WEATHER_VARIABLES:
        features[f"flare24_origin_departure_{variable}"] = current[variable].to_numpy()
        features[f"flare24_dest_arrival_{variable}"] = destination[variable].to_numpy()

    def window_values(variable: str) -> pd.DataFrame:
        return pd.DataFrame(
            {str(offset): snapshots[offset][variable].to_numpy() for offset in origin_window_hours}
        )

    temperature = window_values("temperature")
    precipitation = window_values("precipitation")
    humidity = window_values("humidity")
    cloud = window_values("cloud")
    pressure = window_values("pressure")
    wind = window_values("wind_speed")
    gust = window_values("wind_gust")
    cape = window_values("cape")
    visibility = window_values("visibility")
    ceiling = window_values("ceiling")
    snowfall = window_values("snowfall")
    reflectivity = window_values("reflectivity")
    first_offset = min(origin_window_hours)
    features.update(
        {
            "flare24_origin_window_temperature_mean": temperature.mean(axis=1),
            "flare24_origin_window_temperature_change": temperature["0"]
            - temperature[str(first_offset)],
            "flare24_origin_window_precipitation_sum": precipitation.sum(axis=1, min_count=1),
            "flare24_origin_window_precipitation_max": precipitation.max(axis=1),
            "flare24_origin_window_humidity_max": humidity.max(axis=1),
            "flare24_origin_window_cloud_max": cloud.max(axis=1),
            "flare24_origin_window_pressure_change": pressure["0"]
            - pressure[str(first_offset)],
            "flare24_origin_window_wind_speed_max": wind.max(axis=1),
            "flare24_origin_window_wind_gust_max": gust.max(axis=1),
            "flare24_origin_window_cape_max": cape.max(axis=1),
            "flare24_origin_window_visibility_min": visibility.min(axis=1),
            "flare24_origin_window_ceiling_min": ceiling.min(axis=1),
            "flare24_origin_window_snowfall_sum": snowfall.sum(axis=1, min_count=1),
            "flare24_origin_window_reflectivity_max": reflectivity.max(axis=1),
        }
    )
    origin_present = current["issue_time_utc"].notna()
    dest_present = destination["issue_time_utc"].notna()
    features["flare24_origin_selected_lead_hours"] = current["lead_hours"].to_numpy()
    features["flare24_dest_selected_lead_hours"] = destination["lead_hours"].to_numpy()
    features["flare24_origin_weather_age_hours"] = (
        (current["target_time_utc"] - current["valid_time_utc"]).dt.total_seconds() / 3600.0
    ).to_numpy()
    features["flare24_dest_weather_age_hours"] = (
        (destination["target_time_utc"] - destination["valid_time_utc"]).dt.total_seconds()
        / 3600.0
    ).to_numpy()
    features["flare24_origin_weather_missing"] = (~origin_present).astype("int8").to_numpy()
    features["flare24_dest_weather_missing"] = (~dest_present).astype("int8").to_numpy()
    features["flare24_origin_variable_coverage"] = (
        current.loc[:, list(FLARE24_WEATHER_VARIABLES)].notna().mean(axis=1).to_numpy()
    )
    features["flare24_dest_variable_coverage"] = (
        destination.loc[:, list(FLARE24_WEATHER_VARIABLES)].notna().mean(axis=1).to_numpy()
    )
    window_present = pd.DataFrame(
        {str(offset): snapshots[offset]["issue_time_utc"].notna() for offset in origin_window_hours}
    )
    features["flare24_origin_window_coverage"] = window_present.mean(axis=1).to_numpy()
    all_selected = [current, destination, *snapshots.values()]
    temporal_valid = np.ones(len(flights), dtype=np.int8)
    for selected in all_selected:
        unsafe = selected["issue_time_utc"].notna() & selected["issue_time_utc"].gt(
            selected["cutoff_time_utc"]
        )
        temporal_valid[unsafe.to_numpy()] = 0
    features["flare24_cutoff_coherent_valid"] = temporal_valid

    for variable, reduction in (
        ("wind_speed", "max"),
        ("wind_gust", "max"),
        ("cape", "max"),
        ("visibility", "min"),
        ("ceiling", "min"),
    ):
        origin = pd.Series(features[f"flare24_origin_departure_{variable}"])
        dest = pd.Series(features[f"flare24_dest_arrival_{variable}"])
        pair = pd.concat([origin, dest], axis=1)
        reduced = pair.max(axis=1) if reduction == "max" else pair.min(axis=1)
        features[f"flare24_od_{variable}_{reduction}"] = reduced.where(
            _paired_complete(origin, dest)
        )
    origin_temperature = pd.Series(features["flare24_origin_departure_temperature"])
    dest_temperature = pd.Series(features["flare24_dest_arrival_temperature"])
    features["flare24_od_temperature_gap"] = (
        dest_temperature - origin_temperature
    ).where(_paired_complete(origin_temperature, dest_temperature))

    day2_or_older = weather_cube.loc[
        pd.to_numeric(weather_cube["lead_hours"], errors="raise").ge(48.0)
    ]
    if day2_or_older.empty:
        day2_reference = current.copy()
        day2_reference["valid_time_utc"] = pd.NaT
        day2_reference["issue_time_utc"] = pd.NaT
        day2_reference["lead_hours"] = np.nan
        day2_reference.loc[:, list(FLARE24_WEATHER_VARIABLES)] = np.nan
    else:
        day2_reference = select_latest_safe_forecast(
            origin_targets[0], day2_or_older, _cube_validated=True
        )
    revision_available = (
        current["lead_hours"].notna()
        & day2_reference["lead_hours"].notna()
        & current["lead_hours"].lt(day2_reference["lead_hours"])
    )
    for variable in FLARE24_FORECAST_REVISION_VARIABLES:
        difference = current[variable] - day2_reference[variable]
        features[f"flare24_origin_revision_{variable}_fresh_minus_day2"] = (
            difference.where(revision_available).to_numpy()
        )
    features["flare24_origin_revision_available"] = (
        revision_available.astype("int8").to_numpy()
    )

    feature_frame = pd.DataFrame(features, index=flights.index)
    missing_features = sorted(set(FLARE24_WEATHER_FEATURES) - set(feature_frame.columns))
    if missing_features:
        raise AssertionError(f"FLARE-24 weather implementation omitted features: {missing_features}")
    result = pd.concat([flights.copy(), feature_frame.loc[:, list(FLARE24_WEATHER_FEATURES)]], axis=1)
    if not result["flare24_cutoff_coherent_valid"].eq(1).all():
        raise AssertionError("FLARE-24 weather cutoff invariant failed")
    return result


def timezone_catalog(path: Path) -> dict[str, str]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    airports = payload.get("airports")
    if not isinstance(airports, list):
        raise ValueError(f"airport catalog has no airports list: {path}")
    result = {str(item["iata"]): str(item["timezone"]) for item in airports}
    if not result:
        raise ValueError("airport timezone catalog is empty")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = build_open_meteo_weather_cube(
        acquisition_manifests=tuple(args.acquisition_manifest),
        output_dir=args.output_dir,
        manifest_path=args.manifest,
    )
    print(
        json.dumps(
            {
                "manifest": args.manifest.as_posix(),
                "outputs": result["outputs"],
                "validation": result["validation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
