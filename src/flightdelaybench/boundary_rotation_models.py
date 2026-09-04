"""Fit prior-year latent-rotation kernels on all top-100-touching flights.

Historical tail identifiers supervise only year Y-1. No target-year rows, outcomes, or
actual operations are read, and no historical tail-bearing table is persisted.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import tomllib
from collections.abc import Collection, Iterator, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import joblib  # type: ignore[import-untyped]
import pandas as pd

from .acquisition import validate_bts_zip
from .bts import normalize_crs_minutes
from .flare_rotation import LatentRotationGraph
from .flare_weather import timezone_catalog
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

RAW_HISTORY_COLUMNS = (
    "Year",
    "Month",
    "FlightDate",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSElapsedTime",
    "Distance",
    "Tail_Number",
)
MODEL_HISTORY_COLUMNS = (
    "sample_id",
    "FlightDate",
    "Origin",
    "Dest",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "CRSDepMinutes",
    "CRSElapsedTime",
    "Distance",
    "Tail_Number",
)
FORBIDDEN_HISTORY_COLUMNS = frozenset(
    {
        "ArrDel15",
        "Cancelled",
        "Diverted",
        "DepDelay",
        "ArrDelay",
        "ActualElapsedTime",
        "AirTime",
        "TaxiIn",
        "TaxiOut",
    }
)


def _verified_manifest(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("manifest_sha256")
    body = {name: value for name, value in payload.items() if name != "manifest_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"manifest self-hash failed: {path}")
    return payload


def _verified_protocol(
    path: Path,
    *,
    history_years: tuple[int, ...],
    minimum_turn_minutes: float,
    maximum_layover_minutes: float,
    maximum_candidates: int,
) -> dict[str, Any]:
    protocol: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    information = protocol.get("information_boundary", {})
    model = protocol.get("model", {})
    if (
        protocol.get("identity", {}).get("method") != "BC-POT-Rotation-v1"
        or tuple(int(value) for value in information.get("history_years", ()))
        != history_years
        or information.get("historical_outcomes_allowed") is not False
        or information.get("target_year_rows_allowed_during_fit") is not False
        or information.get("target_year_tail_number_allowed") is not False
        or information.get("confirmation_gate_opened") is not False
        or float(model.get("minimum_turn_minutes", -1.0)) != minimum_turn_minutes
        or float(model.get("maximum_layover_minutes", -1.0))
        != maximum_layover_minutes
        or int(model.get("maximum_candidates", -1)) != maximum_candidates
        or int(model.get("history_end_month", -1)) != 9
        or int(model.get("minimum_lag_to_earliest_target_cutoff_days", -1)) != 91
    ):
        raise ValueError("boundary rotation settings differ from the frozen protocol")
    return protocol


def normalize_boundary_rotation_history_chunk(
    frame: pd.DataFrame,
    *,
    source_year: int,
    source_month: int,
    source_offset: int,
    target_airports: Collection[str],
) -> pd.DataFrame:
    """Project a raw chunk to prior-year schedule and tail supervision only."""

    result = frame.copy().reset_index(drop=True)
    result.columns = [column.strip() for column in result.columns]
    missing = sorted(set(RAW_HISTORY_COLUMNS) - set(result.columns))
    if missing:
        raise ValueError(f"boundary rotation history is missing columns: {missing}")
    if set(result.columns) & FORBIDDEN_HISTORY_COLUMNS:
        raise ValueError("boundary rotation normalization received outcome columns")
    years = pd.to_numeric(result["Year"], errors="raise")
    months = pd.to_numeric(result["Month"], errors="raise")
    if not years.eq(source_year).all() or not months.eq(source_month).all():
        raise ValueError("boundary rotation rows differ from the source period")
    positions = pd.Series(range(source_offset, source_offset + len(result)), index=result.index)
    airports = {str(airport) for airport in target_airports}
    touching = result["Origin"].astype("string").isin(airports) | result["Dest"].astype(
        "string"
    ).isin(airports)
    result = result.loc[touching].copy()
    selected_positions = positions.loc[touching]
    result["sample_id"] = [
        f"bts-{source_year}-{source_month:02d}-{int(position):07d}"
        for position in selected_positions
    ]
    result["FlightDate"] = pd.to_datetime(result["FlightDate"], errors="raise")
    result["Flight_Number_Reporting_Airline"] = (
        pd.to_numeric(result["Flight_Number_Reporting_Airline"], errors="coerce")
        .fillna(-1)
        .astype("int64")
    )
    result["CRSDepMinutes"] = normalize_crs_minutes(result["CRSDepTime"])
    elapsed = pd.to_numeric(result["CRSElapsedTime"], errors="coerce")
    distance = pd.to_numeric(result["Distance"], errors="coerce")
    invalid = elapsed.isna() | elapsed.le(0.0)
    elapsed = elapsed.where(~invalid, 30.0 + distance.clip(lower=0.0) / 8.0)
    if elapsed.isna().any() or elapsed.le(0.0).any():
        raise ValueError("boundary rotation scheduled elapsed time cannot be resolved")
    result["CRSElapsedTime"] = elapsed.astype("float32")
    result["Distance"] = distance.astype("float32")
    result["Tail_Number"] = result["Tail_Number"].astype("string")
    result = result.loc[:, list(MODEL_HISTORY_COLUMNS)].reset_index(drop=True)
    if result["sample_id"].isna().any() or result["sample_id"].duplicated().any():
        raise ValueError("boundary rotation history has invalid sample IDs")
    return result


def iter_boundary_rotation_history(
    archive_path: Path,
    *,
    year: int,
    month: int,
    target_airports: Collection[str],
    chunksize: int = 200_000,
) -> Iterator[pd.DataFrame]:
    member = validate_bts_zip(archive_path)
    offset = 0
    with ZipFile(archive_path) as archive, archive.open(member) as source:
        reader = pd.read_csv(
            source,
            usecols=lambda column: column.strip() in RAW_HISTORY_COLUMNS,
            chunksize=chunksize,
            low_memory=False,
        )
        for raw in reader:
            normalized = normalize_boundary_rotation_history_chunk(
                raw,
                source_year=year,
                source_month=month,
                source_offset=offset,
                target_airports=target_airports,
            )
            offset += len(raw)
            if not normalized.empty:
                yield normalized


def _records_by_year(
    manifest_paths: tuple[Path, ...],
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_year: dict[int, list[dict[str, Any]]] = {}
    manifest_records: list[dict[str, Any]] = []
    for path in manifest_paths:
        payload = _verified_manifest(path)
        for record_value in payload.get("records", []):
            record = dict(record_value)
            year = int(record["year"])
            archive = Path(str(record["local_path"]))
            if (
                record.get("status") != "VERIFIED"
                or not archive.is_file()
                or sha256_file(archive) != record.get("sha256")
            ):
                raise ValueError(f"raw BTS evidence failed: {archive}")
            by_year.setdefault(year, []).append(record)
        manifest_records.append(
            {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "self_hash": payload["manifest_sha256"],
            }
        )
    for year, records in by_year.items():
        records.sort(key=lambda record: int(record["month"]))
        if [int(record["month"]) for record in records] != list(range(1, 13)):
            raise ValueError(f"raw BTS evidence does not cover every month of {year}")
    return by_year, manifest_records


def _atomic_joblib(value: Any, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite boundary rotation model: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated boundary rotation partial exists: {partial}")
    joblib.dump(value, partial)
    partial.replace(path)
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def fit_boundary_rotation_models(
    *,
    protocol_path: Path,
    raw_manifest_paths: tuple[Path, ...],
    airport_config_path: Path,
    airport_catalog_path: Path,
    model_dir: Path,
    manifest_path: Path,
    history_years: tuple[int, ...] = (2023, 2024),
    minimum_turn_minutes: float = 20.0,
    maximum_layover_minutes: float = 720.0,
    maximum_candidates: int = 12,
) -> dict[str, Any]:
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite boundary rotation manifest: {manifest_path}")
    if not history_years or any(year not in {2023, 2024} for year in history_years):
        raise ValueError("boundary rotation history years must be 2023 and/or 2024")
    protocol = _verified_protocol(
        protocol_path,
        history_years=history_years,
        minimum_turn_minutes=minimum_turn_minutes,
        maximum_layover_minutes=maximum_layover_minutes,
        maximum_candidates=maximum_candidates,
    )
    history_end_month = int(protocol["model"]["history_end_month"])
    airport_config = json.loads(airport_config_path.read_text(encoding="utf-8"))
    target_airports = {str(value) for value in airport_config["airports"]}
    if len(target_airports) != 100:
        raise ValueError("boundary rotation requires the frozen 100-airport config")
    timezones = timezone_catalog(airport_catalog_path)
    by_year, raw_manifest_records = _records_by_year(raw_manifest_paths)
    started = time.perf_counter()
    models: list[dict[str, Any]] = []
    raw_inputs: list[dict[str, Any]] = []
    for history_year in history_years:
        if history_year not in by_year:
            raise ValueError(f"raw manifests omit history year {history_year}")
        parts: list[pd.DataFrame] = []
        for record in by_year[history_year]:
            if int(record["month"]) > history_end_month:
                continue
            archive = Path(str(record["local_path"]))
            parts.extend(
                iter_boundary_rotation_history(
                    archive,
                    year=history_year,
                    month=int(record["month"]),
                    target_airports=target_airports,
                )
            )
            raw_inputs.append(
                {
                    "path": archive.as_posix(),
                    "bytes": archive.stat().st_size,
                    "sha256": record["sha256"],
                    "year": history_year,
                    "month": int(record["month"]),
                    "columns_read": list(RAW_HISTORY_COLUMNS),
                    "role": (f"strictly-prior-year tail supervision for target {history_year + 1}"),
                }
            )
        history = pd.concat(parts, ignore_index=True)
        del parts
        if history["sample_id"].duplicated().any():
            raise ValueError(f"boundary rotation history IDs repeat in {history_year}")
        missing_timezones = sorted(
            (set(history["Origin"].astype(str)) | set(history["Dest"].astype(str))) - set(timezones)
        )
        if missing_timezones:
            raise ValueError(f"boundary rotation timezones are missing: {missing_timezones}")
        model = LatentRotationGraph(
            minimum_turn_minutes=minimum_turn_minutes,
            maximum_layover_minutes=maximum_layover_minutes,
            maximum_candidates=maximum_candidates,
        ).fit(history, timezone_by_airport=timezones)
        artifact = _atomic_joblib(
            model,
            model_dir / f"boundary_rotation_history_{history_year}_for_{history_year + 1}.joblib",
        )
        models.append(
            {
                "history_year": history_year,
                "target_year": history_year + 1,
                "history_rows": len(history),
                "history_tail_nonmissing_rows": int(history["Tail_Number"].notna().sum()),
                "model_card": model.model_card().as_dict(),
                **artifact,
            }
        )
        print(
            f"fit boundary rotation history {history_year}: rows={len(history)}",
            flush=True,
        )
        del history, model

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_PRIOR_YEAR_BOUNDARY_ROTATION_MODELS_NO_TARGET_TAILS",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "BC-POT-STRICTLY-PRIOR-YEAR-BOUNDARY-TAIL-SUPERVISION",
        "protocol": {
            "path": protocol_path.as_posix(),
            "bytes": protocol_path.stat().st_size,
            "sha256": sha256_file(protocol_path),
            "identity": protocol["identity"],
        },
        "history_years": list(history_years),
        "history_months": list(range(1, history_end_month + 1)),
        "history_tail_supervision_latest_month": history_end_month,
        "minimum_lag_to_earliest_target_cutoff_days": int(
            protocol["model"]["minimum_lag_to_earliest_target_cutoff_days"]
        ),
        "target_years": [year + 1 for year in history_years],
        "target_airports": sorted(target_airports),
        "models": models,
        "raw_manifests": raw_manifest_records,
        "raw_inputs": raw_inputs,
        "historical_columns_read": list(RAW_HISTORY_COLUMNS),
        "historical_tail_number_role": "supervision in target year minus one only",
        "historical_outcome_columns_read": [],
        "target_year_rows_read": False,
        "target_tail_number_read": False,
        "target_outcome_columns_read": [],
        "confirmation_outcomes_accessed": False,
        "airport_config": {
            "path": airport_config_path.as_posix(),
            "sha256": sha256_file(airport_config_path),
        },
        "airport_catalog": {
            "path": airport_catalog_path.as_posix(),
            "sha256": sha256_file(airport_catalog_path),
        },
        "configuration": {
            "minimum_turn_minutes": minimum_turn_minutes,
            "maximum_layover_minutes": maximum_layover_minutes,
            "maximum_candidates": maximum_candidates,
            "history_end_month": history_end_month,
        },
        "environment": {
            "python": platform.python_version(),
            "joblib": version("joblib"),
            "pandas": version("pandas"),
        },
        "provenance": capture_provenance(
            (
                Path(__file__),
                Path(__file__).with_name("flare_rotation.py"),
                Path(__file__).with_name("boundary_context.py"),
            )
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "Historical tails supervise only latent schedule transition likelihoods. "
            "Target-year tail assignments remain unavailable and are not inferred as fact."
        ),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--raw-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--airport-config", type=Path, required=True)
    parser.add_argument("--airport-catalog", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--history-years", type=int, nargs="+", default=[2023, 2024])
    parser.add_argument("--minimum-turn-minutes", type=float, default=20.0)
    parser.add_argument("--maximum-layover-minutes", type=float, default=720.0)
    parser.add_argument("--maximum-candidates", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = fit_boundary_rotation_models(
        protocol_path=args.protocol,
        raw_manifest_paths=tuple(args.raw_manifests),
        airport_config_path=args.airport_config,
        airport_catalog_path=args.airport_catalog,
        model_dir=args.model_dir,
        manifest_path=args.manifest,
        history_years=tuple(args.history_years),
        minimum_turn_minutes=args.minimum_turn_minutes,
        maximum_layover_minutes=args.maximum_layover_minutes,
        maximum_candidates=args.maximum_candidates,
    )
    print(
        json.dumps(
            {"status": result["status"], "models": len(result["models"])},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
