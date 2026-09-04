"""Build a schedule-only boundary-complete context track for FLARE-24.

The scored population remains the frozen top-100 induced network.  This module
retains every BTS row with at least one top-100 endpoint and projects the raw
archive to schedule fields before writing.  Outcome, tail, and realised-time
columns are never read by this pipeline.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Collection, Iterator, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .acquisition import validate_bts_zip
from .bts import normalize_crs_minutes
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

RAW_SCHEDULE_COLUMNS = (
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
)

CONTEXT_COLUMNS = (
    "sample_id",
    "FlightDate",
    "Origin",
    "Dest",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "CRSDepMinutes",
    "CRSElapsedTime",
    "Distance",
    "top100_origin",
    "top100_dest",
    "context_class",
)

FORBIDDEN_CONTEXT_COLUMNS = frozenset(
    {
        "Tail_Number",
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
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"manifest self-hash failed: {path}")
    return payload


def _resolve_evidence_path(value: str, repository_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def normalize_boundary_schedule_chunk(
    frame: pd.DataFrame,
    *,
    source_year: int,
    source_month: int,
    source_offset: int,
    target_airports: Collection[str],
) -> pd.DataFrame:
    """Normalize only cutoff-safe schedule fields while preserving raw-row ids."""

    result = frame.copy().reset_index(drop=True)
    result.columns = [column.strip() for column in result.columns]
    missing = sorted(set(RAW_SCHEDULE_COLUMNS) - set(result.columns))
    if missing:
        raise ValueError(f"BTS boundary input is missing schedule columns: {missing}")
    if set(result.columns) & FORBIDDEN_CONTEXT_COLUMNS:
        raise ValueError("boundary normalization received forbidden operational columns")
    years = pd.to_numeric(result["Year"], errors="raise")
    months = pd.to_numeric(result["Month"], errors="raise")
    if not years.eq(source_year).all() or not months.eq(source_month).all():
        raise ValueError("boundary rows do not match the declared year and month")

    raw_positions = pd.Series(range(source_offset, source_offset + len(result)), index=result.index)
    airports = set(str(value) for value in target_airports)
    origin_inside = result["Origin"].astype("string").isin(airports)
    dest_inside = result["Dest"].astype("string").isin(airports)
    touching = origin_inside | dest_inside
    result = result.loc[touching].copy()
    origin_inside = origin_inside.loc[touching]
    dest_inside = dest_inside.loc[touching]
    selected_positions = raw_positions.loc[touching]

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
    result["CRSElapsedTime"] = pd.to_numeric(
        result["CRSElapsedTime"], errors="coerce"
    ).astype("float32")
    result["Distance"] = pd.to_numeric(result["Distance"], errors="coerce").astype(
        "float32"
    )
    result["top100_origin"] = origin_inside.astype("int8").to_numpy()
    result["top100_dest"] = dest_inside.astype("int8").to_numpy()
    result["context_class"] = pd.Series(
        ["induced" if left and right else "boundary" for left, right in zip(
            origin_inside, dest_inside, strict=True
        )],
        index=result.index,
        dtype="string",
    )
    result = result.loc[:, list(CONTEXT_COLUMNS)].reset_index(drop=True)
    if result["sample_id"].isna().any() or result["sample_id"].duplicated().any():
        raise ValueError("boundary schedule has invalid sample ids")
    if not result["context_class"].isin(["induced", "boundary"]).all():
        raise RuntimeError("boundary classification failed")
    return result


def iter_boundary_schedule_archive(
    archive_path: Path,
    *,
    year: int,
    month: int,
    target_airports: Collection[str],
    chunksize: int = 200_000,
) -> Iterator[pd.DataFrame]:
    """Stream a raw archive without loading outcomes or target-year tails."""

    member = validate_bts_zip(archive_path)
    offset = 0
    with ZipFile(archive_path) as archive, archive.open(member) as source:
        reader = pd.read_csv(
            source,
            usecols=lambda column: column.strip() in RAW_SCHEDULE_COLUMNS,
            chunksize=chunksize,
            low_memory=False,
        )
        for raw in reader:
            normalized = normalize_boundary_schedule_chunk(
                raw,
                source_year=year,
                source_month=month,
                source_offset=offset,
                target_airports=target_airports,
            )
            offset += len(raw)
            if not normalized.empty:
                yield normalized


def _materialize_month(
    archive_path: Path,
    output_path: Path,
    *,
    year: int,
    month: int,
    target_airports: Collection[str],
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite boundary context: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated boundary partial exists: {partial}")
    writer: pq.ParquetWriter | None = None
    rows = 0
    induced_rows = 0
    boundary_rows = 0
    airports: set[str] = set()
    try:
        for frame in iter_boundary_schedule_archive(
            archive_path,
            year=year,
            month=month,
            target_airports=target_airports,
        ):
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(partial, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(frame)
            induced_rows += int(frame["context_class"].eq("induced").sum())
            boundary_rows += int(frame["context_class"].eq("boundary").sum())
            airports.update(frame["Origin"].astype(str))
            airports.update(frame["Dest"].astype(str))
    finally:
        if writer is not None:
            writer.close()
    if rows == 0 or not partial.is_file():
        raise ValueError(f"boundary context produced no rows: {archive_path}")
    partial.replace(output_path)
    return {
        "year": year,
        "month": month,
        "path": output_path.resolve().as_posix(),
        "rows": rows,
        "induced_rows": induced_rows,
        "boundary_rows": boundary_rows,
        "airports": sorted(airports),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }


def _verify_frozen_targets(context_path: Path, target_path: Path) -> dict[str, Any]:
    context = pd.read_parquet(
        context_path, columns=["sample_id", "context_class"]
    )
    target = pd.read_parquet(target_path, columns=["sample_id"])
    context_ids = context.loc[context["context_class"].eq("induced"), "sample_id"].astype(
        str
    )
    target_ids = target["sample_id"].astype(str)
    if len(context_ids) != len(target_ids) or set(context_ids) != set(target_ids):
        raise ValueError(f"boundary context does not preserve frozen targets: {context_path}")
    return {
        "target_path": target_path.resolve().as_posix(),
        "target_rows": len(target_ids),
        "target_sha256": sha256_file(target_path),
        "sample_id_set_match": True,
    }


def _write_timezone_catalog(airports: set[str], path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite boundary timezone catalog: {path}")
    try:
        import airportsdata
    except ImportError as error:  # pragma: no cover - optional dependency guard
        raise RuntimeError("boundary context requires the weather extra (airportsdata)") from error
    catalog = airportsdata.load("IATA")
    missing = sorted(
        airport
        for airport in airports
        if airport not in catalog or not catalog[airport].get("tz")
    )
    if missing:
        raise ValueError(f"airportsdata has no timezone for boundary airports: {missing}")
    records = [
        {
            "iata": airport,
            "icao": str(catalog[airport].get("icao", "")),
            "latitude": float(catalog[airport]["lat"]),
            "longitude": float(catalog[airport]["lon"]),
            "name": str(catalog[airport]["name"]),
            "timezone": str(catalog[airport]["tz"]),
        }
        for airport in sorted(airports)
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "method": "BOUNDARY-COMPLETE-IATA-TIMEZONE-CATALOG",
        "source_package": "airportsdata",
        "source_package_version": version("airportsdata"),
        "airports": records,
    }
    payload["catalog_sha256"] = canonical_json_sha256(payload)
    write_canonical_json(path, payload)
    return {
        "path": path.resolve().as_posix(),
        "airports": len(records),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "self_hash": payload["catalog_sha256"],
    }


def materialize_boundary_context(
    *,
    raw_manifest_paths: tuple[Path, ...],
    airport_config_path: Path,
    target_census_dir: Path,
    output_dir: Path,
    timezone_catalog_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Materialize immutable schedule context and prove target-set equivalence."""

    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite boundary manifest: {manifest_path}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse boundary output directory: {output_dir}")
    if not raw_manifest_paths:
        raise ValueError("at least one raw manifest is required")
    airport_payload: dict[str, Any] = json.loads(
        airport_config_path.read_text(encoding="utf-8")
    )
    target_airports = tuple(str(value) for value in airport_payload.get("airports", []))
    if len(target_airports) != 100 or len(set(target_airports)) != 100:
        raise ValueError("boundary context requires the frozen 100-airport cohort")

    repository_root = Path(__file__).resolve().parents[2]
    started = time.perf_counter()
    sources: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    all_airports: set[str] = set()
    seen_periods: set[tuple[int, int]] = set()
    output_dir.mkdir(parents=True, exist_ok=False)
    for manifest_path_value in raw_manifest_paths:
        raw = _verified_manifest(manifest_path_value)
        records = list(raw.get("records", []))
        for value in sorted(records, key=lambda item: (int(item["year"]), int(item["month"]))):
            year = int(value["year"])
            month = int(value["month"])
            period = (year, month)
            if period in seen_periods:
                raise ValueError(f"duplicate raw boundary period: {period}")
            if year >= 2026:
                raise ValueError("boundary context refuses 2026 or later inputs")
            seen_periods.add(period)
            raw_path = _resolve_evidence_path(str(value["local_path"]), repository_root)
            if not raw_path.is_file() or sha256_file(raw_path) != value["sha256"]:
                raise ValueError(f"raw boundary source checksum failed: {raw_path}")
            output_path = output_dir / f"year={year}" / f"month={month:02d}.parquet"
            record = _materialize_month(
                raw_path,
                output_path,
                year=year,
                month=month,
                target_airports=target_airports,
            )
            target_path = target_census_dir / f"year={year}" / f"month={month:02d}.parquet"
            if not target_path.is_file():
                raise FileNotFoundError(f"missing frozen target partition: {target_path}")
            record["frozen_target_validation"] = _verify_frozen_targets(
                output_path, target_path
            )
            all_airports.update(record.pop("airports"))
            outputs.append(record)
            print(
                f"boundary context {year}-{month:02d}: rows={record['rows']} "
                f"induced={record['induced_rows']} boundary={record['boundary_rows']}",
                flush=True,
            )
        sources.append(
            {
                "path": manifest_path_value.as_posix(),
                "sha256": sha256_file(manifest_path_value),
                "self_hash": raw["manifest_sha256"],
            }
        )

    years = sorted({year for year, _ in seen_periods})
    for year in years:
        months = sorted(month for candidate_year, month in seen_periods if candidate_year == year)
        if months != list(range(1, 13)):
            raise ValueError(f"boundary context year is incomplete: {year} -> {months}")
    timezone_record = _write_timezone_catalog(all_airports, timezone_catalog_path)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "method": "FLARE24-BOUNDARY-COMPLETE-SCHEDULE-CONTEXT",
        "short_name": "BCSC-v1",
        "status": "COMPLETE_SCHEDULE_ONLY_TOP100_TOUCHING_CONTEXT_TARGETS_FROZEN",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "years": years,
        "target_airports": list(target_airports),
        "airport_config": {
            "path": airport_config_path.as_posix(),
            "sha256": sha256_file(airport_config_path),
        },
        "raw_manifests": sources,
        "outputs": outputs,
        "timezone_catalog": timezone_record,
        "rows": sum(int(record["rows"]) for record in outputs),
        "induced_rows": sum(int(record["induced_rows"]) for record in outputs),
        "boundary_rows": sum(int(record["boundary_rows"]) for record in outputs),
        "context_airports": len(all_airports),
        "columns": list(CONTEXT_COLUMNS),
        "raw_columns_read": list(RAW_SCHEDULE_COLUMNS),
        "outcome_columns_read": [],
        "tail_number_read": False,
        "confirmation_outcomes_accessed": False,
        "information_boundary": {
            "scored_population": "unchanged frozen top100-to-top100 sample ids",
            "context_population": "all BTS rows with at least one frozen top-100 endpoint",
            "target_year_context": "schedule columns only",
            "maximum_year": max(years),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": version("pandas"),
            "pyarrow": version("pyarrow"),
        },
        "provenance": capture_provenance((Path(__file__),)),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "BTS final schedule remains a retrospective proxy for an authentic T-24 "
            "schedule snapshot. Boundary rows provide network context only and do not "
            "establish predictive benefit or physical capacity."
        ),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--airport-config", type=Path, required=True)
    parser.add_argument("--target-census-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timezone-catalog", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = materialize_boundary_context(
        raw_manifest_paths=tuple(args.raw_manifests),
        airport_config_path=args.airport_config,
        target_census_dir=args.target_census_dir,
        output_dir=args.output_dir,
        timezone_catalog_path=args.timezone_catalog,
        manifest_path=args.manifest,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": result["rows"],
                "induced_rows": result["induced_rows"],
                "boundary_rows": result["boundary_rows"],
                "context_airports": result["context_airports"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
