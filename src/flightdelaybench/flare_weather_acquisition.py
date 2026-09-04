"""Acquire additional fixed-lead weather runs for FLARE-24 without outcomes."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import requests

from .flare_weather import OPEN_METEO_UNITS, OPEN_METEO_VARIABLES
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "gfs_seamless"
SUPPORTED_STEMS = tuple(
    stem
    for name, stem in OPEN_METEO_VARIABLES.items()
    if name not in {"ceiling", "reflectivity"}
)


def _catalog(path: Path) -> list[dict[str, Any]]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    airports = payload.get("airports")
    if not isinstance(airports, list) or not airports:
        raise ValueError(f"airport catalog has no airport records: {path}")
    required = {"iata", "latitude", "longitude", "timezone"}
    for airport in airports:
        if not isinstance(airport, dict) or not required.issubset(airport):
            raise ValueError(f"invalid airport catalog record: {airport}")
    return airports


def _hourly_variables(lead_days: int) -> tuple[str, ...]:
    return tuple(f"{stem}_previous_day{lead_days}" for stem in SUPPORTED_STEMS)


def _coverage(hourly: dict[str, list[Any]], variables: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    times = hourly["time"]
    for variable in variables:
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


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    """Return a bounded provider-aware retry delay in seconds."""

    ordinary = float(min(2**attempt, 20))
    if response is None or response.status_code != 429:
        return ordinary
    provider_delay = 0.0
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            provider_delay = float(retry_after)
        except ValueError:
            try:
                provider_time = parsedate_to_datetime(retry_after)
                provider_delay = max(
                    0.0, (provider_time - datetime.now(provider_time.tzinfo)).total_seconds()
                )
            except (TypeError, ValueError):
                provider_delay = 0.0
    exponential_rate_delay = float(15 * 2**attempt)
    return min(max(provider_delay, exponential_rate_delay), 300.0)


def validate_response(
    payload: dict[str, Any],
    *,
    airport: dict[str, Any],
    year: int,
    variables: tuple[str, ...],
) -> dict[str, Any]:
    if payload.get("error"):
        raise ValueError(f"forecast API error: {payload}")
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError("forecast response is missing the hourly object")
    expected = {"time", *variables}
    missing = sorted(expected - set(hourly))
    if missing:
        raise ValueError(f"forecast response is missing variables: {missing}")
    expected_hours = 8_784 if year % 4 == 0 else 8_760
    lengths = {len(hourly[name]) for name in expected}
    if lengths != {expected_hours}:
        raise ValueError(
            f"unexpected hourly lengths for {airport['iata']} {year}: {sorted(lengths)}"
        )
    if payload.get("timezone") != airport["timezone"]:
        raise ValueError(
            f"timezone mismatch for {airport['iata']}: "
            f"{payload.get('timezone')} != {airport['timezone']}"
        )
    units = payload.get("hourly_units")
    if not isinstance(units, dict):
        raise ValueError("forecast response is missing hourly_units")
    expected_units = {
        variable: OPEN_METEO_UNITS[name]
        for name, stem in OPEN_METEO_VARIABLES.items()
        for variable in variables
        if variable.startswith(f"{stem}_previous_day")
    }
    wrong_units = {
        variable: (units.get(variable), expected)
        for variable, expected in expected_units.items()
        if units.get(variable) != expected
    }
    if wrong_units:
        raise ValueError(f"forecast response has unexpected units: {wrong_units}")
    return _coverage(hourly, variables)


def _download(
    *,
    airport: dict[str, Any],
    year: int,
    lead_days: int,
    variables: tuple[str, ...],
    path: Path,
    retries: int,
) -> dict[str, Any]:
    parameters: dict[str, str | float] = {
        "latitude": float(airport["latitude"]),
        "longitude": float(airport["longitude"]),
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "timezone": str(airport["timezone"]),
        "models": MODEL,
        "hourly": ",".join(variables),
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        response: requests.Response | None = None
        try:
            response = requests.get(
                API_URL,
                params=parameters,
                headers={"User-Agent": "FlightDelayBench-FLARE24/0.1 research acquisition"},
                timeout=(15, 240),
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            coverage = validate_response(
                payload,
                airport=airport,
                year=year,
                variables=variables,
            )
            if path.exists():
                raise FileExistsError(f"refusing to overwrite raw FLARE weather: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(path.suffix + ".part")
            if partial.exists():
                raise FileExistsError(f"unadjudicated partial FLARE weather exists: {partial}")
            partial.write_bytes(response.content)
            partial.replace(path)
            return {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "coverage": coverage,
            }
        except (requests.RequestException, json.JSONDecodeError, ValueError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(_retry_delay(response, attempt))
    raise RuntimeError(
        f"FLARE weather acquisition failed for {airport['iata']} {year}"
    ) from last_error


def _existing_record(
    path: Path,
    *,
    airport: dict[str, Any],
    year: int,
    variables: tuple[str, ...],
) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    coverage = validate_response(payload, airport=airport, year=year, variables=variables)
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "coverage": coverage,
    }


def acquire_fixed_lead_weather(
    *,
    airport_catalog_path: Path,
    output_dir: Path,
    manifest_path: Path,
    lead_days: int,
    start_year: int,
    end_year: int,
    workers: int = 4,
    retries: int = 5,
    resume: bool = False,
) -> dict[str, Any]:
    """Acquire one preserved 2-7 day lead archive for cutoff-coherent selection."""

    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite acquisition manifest: {manifest_path}")
    if lead_days not in range(2, 8):
        raise ValueError("lead_days must be between 2 and 7")
    if start_year < 2024 or end_year > 2025 or start_year > end_year:
        raise ValueError("Open-Meteo FLARE acquisition is bounded to 2024-2025")
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    airports = _catalog(airport_catalog_path)
    variables = _hourly_variables(lead_days)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tasks: list[tuple[dict[str, Any], int, Path]] = []
    records: list[dict[str, Any]] = []
    for airport in airports:
        for year in range(start_year, end_year + 1):
            path = output_dir / "raw" / f"airport={airport['iata']}" / f"year={year}.json"
            if path.exists():
                if not resume:
                    raise FileExistsError(f"raw response exists; use --resume: {path}")
                record = _existing_record(
                    path,
                    airport=airport,
                    year=year,
                    variables=variables,
                )
                records.append({"airport": airport["iata"], "year": year, **record})
            else:
                tasks.append((airport, year, path))

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="flare-weather") as executor:
        future_map = {
            executor.submit(
                _download,
                airport=airport,
                year=year,
                lead_days=lead_days,
                variables=variables,
                path=path,
                retries=retries,
            ): (airport, year)
            for airport, year, path in tasks
        }
        for future in as_completed(future_map):
            airport, year = future_map[future]
            try:
                record = future.result()
                records.append({"airport": airport["iata"], "year": year, **record})
                if len(records) % 10 == 0:
                    print(f"verified FLARE weather responses: {len(records)}", flush=True)
            except Exception as error:
                failures.append(f"{airport['iata']} {year}: {error}")
    if failures:
        summary = "; ".join(failures[:10])
        raise RuntimeError(
            f"{len(failures)} FLARE weather requests failed; completion manifest withheld: {summary}"
        )
    records.sort(key=lambda item: (str(item["airport"]), int(item["year"])))

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_COVARIATE_ONLY_NO_FLIGHT_OUTCOMES_ACCESSED",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": "Open-Meteo Previous Model Runs API",
        "source_url": API_URL,
        "model": MODEL,
        "fixed_lead_hours": lead_days * 24,
        "lead_suffix": f"previous_day{lead_days}",
        "hourly_variables": list(variables),
        "unit_contract": {
            variable: OPEN_METEO_UNITS[name]
            for name, stem in OPEN_METEO_VARIABLES.items()
            for variable in (f"{stem}_previous_day{lead_days}",)
            if variable in variables
        },
        "airport_catalog": {
            "path": airport_catalog_path.as_posix(),
            "sha256": sha256_file(airport_catalog_path),
        },
        "start_year": start_year,
        "end_year": end_year,
        "airport_count": len(airports),
        "request_count": len(records),
        "workers": workers,
        "requests": records,
        "license_scope": "Open-Meteo non-commercial research use; attribution required",
        "environment": {
            "python": platform.python_version(),
            "requests": version("requests"),
        },
        "provenance": capture_provenance(
            (Path(__file__), Path(__file__).with_name("flare_weather.py"))
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "These fixed-lead forecasts contain no flight outcomes. They permit an older safe "
            "forecast when a 24-hour-valid-time record would be issued after a flight cutoff; "
            "they do not establish production latency or same-cycle equivalence."
        ),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lead-days", type=int, default=2)
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = acquire_fixed_lead_weather(
        airport_catalog_path=args.airport_catalog,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        lead_days=args.lead_days,
        start_year=args.start_year,
        end_year=args.end_year,
        workers=args.workers,
        retries=args.retries,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "manifest": args.manifest.as_posix(),
                "requests": result["request_count"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
