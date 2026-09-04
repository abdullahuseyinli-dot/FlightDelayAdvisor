"""Acquire fixed-lead archived GFS forecasts without accessing flight outcomes."""

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

import pandas as pd
import requests

from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .modeling import _variant_files

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "gfs_seamless"
LEAD_SUFFIX = "previous_day1"
HOURLY_VARIABLES = (
    f"temperature_2m_{LEAD_SUFFIX}",
    f"precipitation_{LEAD_SUFFIX}",
    f"relative_humidity_2m_{LEAD_SUFFIX}",
    f"cloud_cover_{LEAD_SUFFIX}",
    f"surface_pressure_{LEAD_SUFFIX}",
    f"wind_speed_10m_{LEAD_SUFFIX}",
    f"wind_gusts_10m_{LEAD_SUFFIX}",
    f"cape_{LEAD_SUFFIX}",
)


def discover_airports(feature_dir: Path, reference_year: int = 2024) -> tuple[str, ...]:
    frames = [
        pd.read_parquet(path, columns=["Origin", "Dest"])
        for path in _variant_files(feature_dir, reference_year)
    ]
    codes: set[str] = set()
    for frame in frames:
        codes.update(frame["Origin"].dropna().astype(str))
        codes.update(frame["Dest"].dropna().astype(str))
    if not codes:
        raise ValueError("no airports found in the reference feature cohort")
    return tuple(sorted(codes))


def airport_catalog(codes: tuple[str, ...]) -> list[dict[str, Any]]:
    try:
        import airportsdata
    except ImportError as error:
        raise RuntimeError("install the 'weather' extra to resolve airport coordinates") from error
    database = airportsdata.load("IATA")
    missing = sorted(set(codes) - set(database))
    if missing:
        raise ValueError(f"airport metadata missing for cohort codes: {missing}")
    return [
        {
            "iata": code,
            "icao": str(database[code]["icao"]),
            "name": str(database[code]["name"]),
            "latitude": float(database[code]["lat"]),
            "longitude": float(database[code]["lon"]),
            "timezone": str(database[code]["tz"]),
        }
        for code in codes
    ]


def _coverage(hourly: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
    times = hourly["time"]
    result: dict[str, dict[str, Any]] = {}
    for variable in HOURLY_VARIABLES:
        values = hourly[variable]
        valid = [index for index, value in enumerate(values) if value is not None]
        result[variable] = {
            "rows": len(values),
            "valid_rows": len(valid),
            "fraction": len(valid) / len(values) if values else 0.0,
            "first_valid_time": times[valid[0]] if valid else None,
            "last_valid_time": times[valid[-1]] if valid else None,
        }
    return result


def _validate_response(
    payload: dict[str, Any],
    *,
    airport: dict[str, Any],
    year: int,
) -> dict[str, dict[str, Any]]:
    if payload.get("error"):
        raise ValueError(f"forecast API error: {payload}")
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError("forecast response is missing the hourly object")
    expected = {"time", *HOURLY_VARIABLES}
    missing = sorted(expected - set(hourly))
    if missing:
        raise ValueError(f"forecast response is missing variables: {missing}")
    lengths = {len(hourly[name]) for name in expected}
    expected_hours = 8_784 if year % 4 == 0 else 8_760
    if lengths != {expected_hours}:
        raise ValueError(
            f"unexpected hourly response lengths for {airport['iata']} {year}: {sorted(lengths)}"
        )
    if payload.get("timezone") != airport["timezone"]:
        raise ValueError(
            f"timezone mismatch for {airport['iata']}: "
            f"{payload.get('timezone')} != {airport['timezone']}"
        )
    return _coverage(hourly)


def _download_json(
    *,
    session: requests.Session,
    airport: dict[str, Any],
    year: int,
    path: Path,
    retries: int = 5,
) -> dict[str, Any]:
    parameters = {
        "latitude": airport["latitude"],
        "longitude": airport["longitude"],
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "timezone": airport["timezone"],
        "models": MODEL,
        "hourly": ",".join(HOURLY_VARIABLES),
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(API_URL, params=parameters, timeout=(15, 180))
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            coverage = _validate_response(payload, airport=airport, year=year)
            if path.exists():
                raise FileExistsError(f"refusing to overwrite raw forecast response: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(path.suffix + ".part")
            if partial.exists():
                raise FileExistsError(f"unadjudicated partial forecast response exists: {partial}")
            partial.write_bytes(response.content)
            partial.replace(path)
            return {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "coverage": coverage,
            }
        except (requests.RequestException, ValueError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 == retries:
                break
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(
        f"forecast acquisition failed for {airport['iata']} {year} after {retries} attempts"
    ) from last_error


def acquire_forecasts(
    *,
    feature_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    start_year: int = 2024,
    end_year: int = 2025,
    reference_year: int = 2024,
    resume: bool = False,
) -> dict[str, Any]:
    """Download fixed 24-hour-lead covariates for the declared airport cohort."""

    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite acquisition manifest: {manifest_path}")
    if start_year < 2024 or end_year > 2025 or start_year > end_year:
        raise ValueError("this acquisition is intentionally bounded to 2024-2025")
    codes = discover_airports(feature_dir, reference_year)
    catalog = airport_catalog(codes)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = output_dir / "airport_catalog.json"
    catalog_payload = {
        "source_package": "airportsdata",
        "source_package_version": version("airportsdata"),
        "reference_feature_year": reference_year,
        "airports": catalog,
    }
    if catalog_path.exists():
        existing = json.loads(catalog_path.read_text(encoding="utf-8"))
        if existing != catalog_payload:
            raise ValueError("existing airport catalog differs from the current resolved metadata")
    else:
        write_canonical_json(catalog_path, catalog_payload)

    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers["User-Agent"] = "FlightDelayBench/0.1 research acquisition"
    try:
        for airport in catalog:
            for year in range(start_year, end_year + 1):
                path = output_dir / "raw" / f"airport={airport['iata']}" / f"year={year}.json"
                if path.exists():
                    if not resume:
                        raise FileExistsError(f"raw response already exists; use --resume: {path}")
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    coverage = _validate_response(payload, airport=airport, year=year)
                    record = {
                        "path": path.as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "coverage": coverage,
                    }
                else:
                    record = _download_json(
                        session=session,
                        airport=airport,
                        year=year,
                        path=path,
                    )
                records.append(
                    {
                        "airport": airport["iata"],
                        "year": year,
                        **record,
                    }
                )
    finally:
        session.close()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_COVARIATE_ONLY_NO_FLIGHT_OUTCOMES_ACCESSED",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": "Open-Meteo Previous Model Runs API",
        "source_url": API_URL,
        "model": MODEL,
        "fixed_lead_hours": 24,
        "lead_suffix": LEAD_SUFFIX,
        "hourly_variables": list(HOURLY_VARIABLES),
        "license_scope": "Open-Meteo non-commercial research use; attribution required",
        "start_year": start_year,
        "end_year": end_year,
        "reference_feature_year": reference_year,
        "airport_count": len(catalog),
        "airport_catalog": {
            "path": catalog_path.as_posix(),
            "bytes": catalog_path.stat().st_size,
            "sha256": sha256_file(catalog_path),
        },
        "requests": records,
        "request_count": len(records),
        "environment": {
            "python": platform.python_version(),
            "requests": version("requests"),
            "airportsdata": version("airportsdata"),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "Fixed-lead archived numerical forecasts are covariates only. Acquisition does not "
            "establish operational latency, uptime, or production feed equivalence."
        ),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--reference-year", type=int, default=2024)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = acquire_forecasts(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        start_year=args.start_year,
        end_year=args.end_year,
        reference_year=args.reference_year,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "manifest": args.manifest.as_posix(),
                "airports": manifest["airport_count"],
                "requests": manifest["request_count"],
                "elapsed_seconds": manifest["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
