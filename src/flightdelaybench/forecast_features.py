"""Build daily fixed-24-hour forecast covariates from preserved API responses."""

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

from .forecast_acquisition import HOURLY_VARIABLES, LEAD_SUFFIX
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

DAILY_FORECAST_COLUMNS = (
    "forecast24_tavg",
    "forecast24_tmin",
    "forecast24_tmax",
    "forecast24_prcp_sum",
    "forecast24_prcp_max",
    "forecast24_rh_mean",
    "forecast24_rh_max",
    "forecast24_cloud_mean",
    "forecast24_cloud_max",
    "forecast24_pressure_mean",
    "forecast24_pressure_range",
    "forecast24_wspd_mean",
    "forecast24_wspd_max",
    "forecast24_gust_max",
    "forecast24_cape_max",
    "forecast24_min_variable_coverage",
    "forecast24_missing",
)


def _verify_manifest(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("manifest_sha256")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"forecast acquisition manifest self-hash failed: {path}")
    if int(payload.get("fixed_lead_hours", -1)) != 24:
        raise ValueError("feature builder requires the fixed 24-hour acquisition")
    if tuple(payload.get("hourly_variables", [])) != HOURLY_VARIABLES:
        raise ValueError("acquisition variables differ from the feature-builder contract")
    return payload


def _series(hourly: dict[str, list[Any]], stem: str) -> pd.Series:
    name = f"{stem}_{LEAD_SUFFIX}"
    return pd.to_numeric(pd.Series(hourly[name], dtype="float64"), errors="coerce")


def _aggregate_airport_year(path: Path, airport: str, year: int) -> pd.DataFrame:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    hourly: dict[str, list[Any]] = payload["hourly"]
    dates = pd.to_datetime(pd.Series(hourly["time"], dtype="string")).dt.normalize()
    values = pd.DataFrame(
        {
            "date": dates,
            "temperature": _series(hourly, "temperature_2m"),
            "precipitation": _series(hourly, "precipitation"),
            "humidity": _series(hourly, "relative_humidity_2m"),
            "cloud": _series(hourly, "cloud_cover"),
            "pressure": _series(hourly, "surface_pressure"),
            "wind": _series(hourly, "wind_speed_10m"),
            "gust": _series(hourly, "wind_gusts_10m"),
            "cape": _series(hourly, "cape"),
        }
    )
    grouped = values.groupby("date", sort=True, observed=True)
    daily = pd.DataFrame(
        {
            "forecast24_tavg": grouped["temperature"].mean(),
            "forecast24_tmin": grouped["temperature"].min(),
            "forecast24_tmax": grouped["temperature"].max(),
            "forecast24_prcp_sum": grouped["precipitation"].sum(min_count=1),
            "forecast24_prcp_max": grouped["precipitation"].max(),
            "forecast24_rh_mean": grouped["humidity"].mean(),
            "forecast24_rh_max": grouped["humidity"].max(),
            "forecast24_cloud_mean": grouped["cloud"].mean(),
            "forecast24_cloud_max": grouped["cloud"].max(),
            "forecast24_pressure_mean": grouped["pressure"].mean(),
            "forecast24_pressure_range": grouped["pressure"].max()
            - grouped["pressure"].min(),
            "forecast24_wspd_mean": grouped["wind"].mean(),
            "forecast24_wspd_max": grouped["wind"].max(),
            "forecast24_gust_max": grouped["gust"].max(),
            "forecast24_cape_max": grouped["cape"].max(),
        }
    )
    coverage = grouped[
        ["temperature", "precipitation", "humidity", "cloud", "pressure", "wind", "gust", "cape"]
    ].count() / 24.0
    daily["forecast24_min_variable_coverage"] = coverage.min(axis=1)
    expected_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    daily = daily.reindex(expected_dates)
    daily.index.name = "FlightDate"
    daily["forecast24_missing"] = daily["forecast24_min_variable_coverage"].fillna(0).lt(1.0)
    daily["forecast24_min_variable_coverage"] = daily[
        "forecast24_min_variable_coverage"
    ].fillna(0)
    daily.insert(0, "Airport", airport)
    result = daily.reset_index()
    for column in DAILY_FORECAST_COLUMNS:
        if column == "forecast24_missing":
            result[column] = result[column].astype("int8")
        else:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("float32")
    return result


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite forecast features: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated partial forecast features exist: {partial}")
    frame.to_parquet(partial, index=False, compression="zstd")
    partial.replace(path)
    return {
        "path": path.as_posix(),
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_daily_forecasts(
    *,
    acquisition_manifest_path: Path,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite forecast feature manifest: {manifest_path}")
    acquisition = _verify_manifest(acquisition_manifest_path)
    started = time.perf_counter()
    by_year: dict[int, list[pd.DataFrame]] = {}
    input_records: list[dict[str, Any]] = []
    for record in acquisition["requests"]:
        path = Path(record["path"])
        actual_hash = sha256_file(path)
        if actual_hash != record["sha256"]:
            raise ValueError(f"raw forecast checksum failed: {path}")
        airport = str(record["airport"])
        year = int(record["year"])
        by_year.setdefault(year, []).append(_aggregate_airport_year(path, airport, year))
        input_records.append(
            {
                "airport": airport,
                "year": year,
                "path": path.as_posix(),
                "sha256": actual_hash,
            }
        )

    outputs: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    for year, parts in sorted(by_year.items()):
        frame = pd.concat(parts, ignore_index=True)
        expected_rows = int(acquisition["airport_count"]) * (366 if year % 4 == 0 else 365)
        if len(frame) != expected_rows:
            raise ValueError(f"daily forecast row count failed for {year}")
        if frame.duplicated(["Airport", "FlightDate"]).any():
            raise ValueError(f"duplicate airport-date forecast rows for {year}")
        output = _atomic_parquet(frame, output_dir / f"year={year}.parquet")
        outputs.append({"year": year, **output})
        coverage[str(year)] = {
            "rows": len(frame),
            "complete_days": int(frame["forecast24_missing"].eq(0).sum()),
            "missing_or_partial_days": int(frame["forecast24_missing"].eq(1).sum()),
            "mean_min_variable_coverage": float(
                frame["forecast24_min_variable_coverage"].mean()
            ),
        }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_FIXED_24H_FORECAST_COVARIATES_NO_OUTCOMES",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "acquisition_manifest": acquisition_manifest_path.as_posix(),
        "acquisition_manifest_sha256": sha256_file(acquisition_manifest_path),
        "acquisition_manifest_self_hash": acquisition["manifest_sha256"],
        "fixed_lead_hours": 24,
        "hourly_inputs": list(HOURLY_VARIABLES),
        "daily_features": list(DAILY_FORECAST_COLUMNS),
        "input_records": input_records,
        "outputs": outputs,
        "coverage": coverage,
        "environment": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "pyarrow": version("pyarrow"),
        },
        "provenance": capture_provenance(
            (
                Path(__file__),
                Path(__file__).with_name("forecast_acquisition.py"),
            )
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "Daily summaries preserve a fixed 24-hour lead but do not reproduce a production "
            "schedule join or establish live-feed availability."
        ),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = build_daily_forecasts(
        acquisition_manifest_path=args.acquisition_manifest,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
    )
    print(json.dumps({"manifest": args.manifest.as_posix(), "coverage": result["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
