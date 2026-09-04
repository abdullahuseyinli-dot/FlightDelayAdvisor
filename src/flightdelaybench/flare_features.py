"""Materialize cutoff-coherent FLARE-24 weather and aviation features."""

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
    FLARE24_AVIATION_WEATHER_FEATURES,
    FLARE24_CORRIDOR_FEATURES,
    FLARE24_WEATHER_FEATURES,
)
from .flare_aviation import attach_aviation_weather_features
from .flare_corridor import attach_corridor_weather_features, coordinate_catalog
from .flare_runways import load_runway_headings
from .flare_weather import attach_flare24_weather, timezone_catalog, validate_weather_cube
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

FLARE24_MATERIALIZED_FEATURES = (
    *FLARE24_WEATHER_FEATURES,
    *FLARE24_AVIATION_WEATHER_FEATURES,
    *FLARE24_CORRIDOR_FEATURES,
)
SCHEDULE_INPUT_COLUMNS = (
    "sample_id",
    "FlightDate",
    "Origin",
    "Dest",
    "CRSDepMinutes",
    "CRSElapsedTime",
    "DepHour",
)


def _verified_manifest(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("manifest_sha256")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"manifest self-hash failed: {path}")
    return payload


def _load_verified_weather(manifest_path: Path) -> pd.DataFrame:
    manifest = _verified_manifest(manifest_path)
    frames: list[pd.DataFrame] = []
    for output in manifest.get("outputs", []):
        path = Path(output["path"])
        if sha256_file(path) != output["sha256"]:
            raise ValueError(f"weather cube checksum failed: {path}")
        frame = pd.read_parquet(path)
        if len(frame) != int(output["rows"]):
            raise ValueError(f"weather cube row count failed: {path}")
        frames.append(frame)
    if not frames:
        raise ValueError("weather cube manifest contains no outputs")
    weather = pd.concat(frames, ignore_index=True)
    validate_weather_cube(weather)
    return weather


def _schedule_files(census_dir: Path, years: tuple[int, ...]) -> tuple[Path, ...]:
    files: list[Path] = []
    for year in years:
        year_files = sorted((census_dir / f"year={year}").glob("month=*.parquet"))
        if len(year_files) != 12:
            raise FileNotFoundError(
                f"expected 12 census schedule partitions for {year}, found {len(year_files)}"
            )
        files.extend(year_files)
    return tuple(files)


def _atomic_feature_partition(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite FLARE-24 feature partition: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated FLARE-24 feature partial exists: {partial}")
    frame.to_parquet(partial, index=False, compression="zstd")
    partial.replace(path)
    return {
        "path": path.as_posix(),
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def materialize_flare24_features(
    *,
    census_dir: Path,
    weather_manifest_path: Path,
    airport_catalog_path: Path,
    runway_catalog_path: Path,
    output_dir: Path,
    manifest_path: Path,
    years: tuple[int, ...],
) -> dict[str, Any]:
    """Build sample-id keyed features while reading schedule columns only."""

    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite FLARE-24 feature manifest: {manifest_path}")
    if not years or any(year not in {2024, 2025} for year in years):
        raise ValueError("FLARE-24 weather feature years must be a subset of 2024-2025")
    started = time.perf_counter()
    weather = _load_verified_weather(weather_manifest_path)
    timezones = timezone_catalog(airport_catalog_path)
    coordinates = coordinate_catalog(airport_catalog_path)
    runway_headings = load_runway_headings(runway_catalog_path)
    schedule_files = _schedule_files(census_dir, years)
    outputs: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    coverage_sums = {feature: 0 for feature in FLARE24_MATERIALIZED_FEATURES}
    total_rows = 0
    for partition_index, schedule_path in enumerate(schedule_files, start=1):
        schedule = pd.read_parquet(schedule_path, columns=list(SCHEDULE_INPUT_COLUMNS))
        if schedule.empty:
            raise ValueError(f"census schedule partition is empty: {schedule_path}")
        if schedule["sample_id"].isna().any() or schedule["sample_id"].duplicated().any():
            raise ValueError(f"invalid sample_id values in schedule partition: {schedule_path}")
        schedule = schedule.reset_index(drop=True)
        schedule["schedule_origin_departure_bank_log1p"] = np.log1p(
            schedule.groupby(
                ["FlightDate", "Origin", "DepHour"], observed=True, sort=False
            )["sample_id"].transform("size")
        )
        schedule["schedule_dest_inbound_day_log1p"] = np.log1p(
            schedule.groupby(["FlightDate", "Dest"], observed=True, sort=False)[
                "sample_id"
            ].transform("size")
        )
        weathered = attach_flare24_weather(
            schedule,
            weather,
            timezone_by_airport=timezones,
        )
        weathered = attach_corridor_weather_features(
            weathered,
            weather,
            timezone_by_airport=timezones,
            airport_coordinates=coordinates,
        )
        enriched = attach_aviation_weather_features(
            weathered,
            runway_headings_by_airport=runway_headings,
        )
        feature_output = enriched.loc[
            :, ["sample_id", *FLARE24_MATERIALIZED_FEATURES]
        ].copy()
        if not feature_output["flare24_cutoff_coherent_valid"].eq(1).all():
            raise AssertionError(f"weather cutoff invariant failed: {schedule_path}")
        numeric = feature_output.loc[:, list(FLARE24_MATERIALIZED_FEATURES)].to_numpy(
            dtype=np.float64
        )
        if np.isinf(numeric).any():
            raise ValueError(f"FLARE-24 feature partition contains infinity: {schedule_path}")
        for feature in FLARE24_MATERIALIZED_FEATURES:
            coverage_sums[feature] += int(feature_output[feature].notna().sum())
            feature_output[feature] = pd.to_numeric(
                feature_output[feature], errors="raise"
            ).astype("float32")
        year = int(pd.to_datetime(schedule["FlightDate"], errors="raise").dt.year.iloc[0])
        month = int(pd.to_datetime(schedule["FlightDate"], errors="raise").dt.month.iloc[0])
        if not pd.to_datetime(schedule["FlightDate"]).dt.year.eq(year).all() or not pd.to_datetime(
            schedule["FlightDate"]
        ).dt.month.eq(month).all():
            raise ValueError(f"census schedule period is not homogeneous: {schedule_path}")
        output_path = output_dir / f"year={year}" / f"month={month:02d}.parquet"
        output_record = _atomic_feature_partition(feature_output, output_path)
        outputs.append(
            {
                "year": year,
                "month": month,
                **output_record,
            }
        )
        inputs.append(
            {
                "path": schedule_path.as_posix(),
                "rows": len(schedule),
                "bytes": schedule_path.stat().st_size,
                "sha256": sha256_file(schedule_path),
                "columns_read": list(SCHEDULE_INPUT_COLUMNS),
            }
        )
        total_rows += len(feature_output)
        print(
            f"materialized FLARE partition {partition_index}/{len(schedule_files)}: "
            f"{year}-{month:02d} rows={len(feature_output)}",
            flush=True,
        )

    coverage = {
        feature: coverage_sums[feature] / total_rows
        for feature in FLARE24_MATERIALIZED_FEATURES
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "method": "FLARE-24-CUTOFF-COHERENT-AVIATION-FEATURES",
        "status": "COMPLETE_COVARIATE_ONLY_NO_FLIGHT_OUTCOMES_ACCESSED",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "years": list(years),
        "rows": total_rows,
        "features": list(FLARE24_MATERIALIZED_FEATURES),
        "feature_nonmissing_fraction": coverage,
        "schedule_inputs": inputs,
        "weather_manifest": {
            "path": weather_manifest_path.as_posix(),
            "sha256": sha256_file(weather_manifest_path),
        },
        "airport_catalog": {
            "path": airport_catalog_path.as_posix(),
            "sha256": sha256_file(airport_catalog_path),
        },
        "runway_catalog": {
            "path": runway_catalog_path.as_posix(),
            "sha256": sha256_file(runway_catalog_path),
        },
        "outputs": outputs,
        "outcome_columns_read": [],
        "confirmation_outcomes_accessed": False,
        "runway_interpretation": "wind-optimal eligible-runway envelope; active runway unknown",
        "environment": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "pyarrow": version("pyarrow"),
        },
        "provenance": capture_provenance(
            (
                Path(__file__),
                Path(__file__).with_name("flare_weather.py"),
                Path(__file__).with_name("flare_aviation.py"),
                Path(__file__).with_name("flare_runways.py"),
                Path(__file__).with_name("contracts.py"),
            )
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "BTS retrospective rows proxy the advance schedule; operational schedule snapshot "
            "equivalence, active runway, and production weather latency are not claimed."
        ),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-dir", type=Path, required=True)
    parser.add_argument("--weather-manifest", type=Path, required=True)
    parser.add_argument("--airport-catalog", type=Path, required=True)
    parser.add_argument("--runway-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = materialize_flare24_features(
        census_dir=args.census_dir,
        weather_manifest_path=args.weather_manifest,
        airport_catalog_path=args.airport_catalog,
        runway_catalog_path=args.runway_catalog,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        years=tuple(args.years),
    )
    print(
        json.dumps(
            {
                "manifest": args.manifest.as_posix(),
                "rows": result["rows"],
                "partitions": len(result["outputs"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
