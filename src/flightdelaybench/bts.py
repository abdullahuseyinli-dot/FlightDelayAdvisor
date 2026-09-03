"""Memory-bounded normalization of BTS monthly on-time archives."""

from __future__ import annotations

import argparse
import json
from collections.abc import Collection, Iterator
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .acquisition import validate_bts_zip
from .hashing import sha256_file, write_canonical_json

RAW_COLUMNS = (
    "Year",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "FlightDate",
    "Reporting_Airline",
    "DOT_ID_Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "Tail_Number",
    "OriginAirportID",
    "Origin",
    "OriginState",
    "DestAirportID",
    "Dest",
    "DestState",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "ArrDel15",
    "Cancelled",
    "Diverted",
    "Distance",
)


def normalize_crs_hour(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0).astype("int32")
    return ((numeric % 2400) // 100).clip(0, 23).astype("int8")


def normalize_bts_chunk(
    frame: pd.DataFrame,
    *,
    source_year: int,
    source_month: int,
    source_offset: int,
    allowed_airports: Collection[str] | None = None,
) -> pd.DataFrame:
    """Normalize one raw chunk and retain explicit source-row lineage."""

    frame = frame.copy()
    frame.columns = [column.strip() for column in frame.columns]
    required = set(RAW_COLUMNS) - {"Tail_Number"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"BTS input is missing columns: {missing}")
    if "Tail_Number" not in frame:
        frame["Tail_Number"] = ""

    frame["Year"] = pd.to_numeric(frame["Year"], errors="coerce")
    frame["Month"] = pd.to_numeric(frame["Month"], errors="coerce")
    if not frame["Year"].eq(source_year).all() or not frame["Month"].eq(source_month).all():
        raise ValueError("archive rows do not match the declared year and month")

    frame["Diverted"] = pd.to_numeric(frame["Diverted"], errors="coerce").fillna(0)
    frame = frame.loc[frame["Diverted"].eq(0)].copy()
    if allowed_airports is not None:
        allowed = set(allowed_airports)
        frame = frame.loc[frame["Origin"].isin(allowed) & frame["Dest"].isin(allowed)].copy()

    original_positions = frame.index.to_numpy(dtype=np.int64) + source_offset
    frame["sample_id"] = [
        f"bts-{source_year}-{source_month:02d}-{position:07d}" for position in original_positions
    ]
    frame["FlightDate"] = pd.to_datetime(frame["FlightDate"], errors="raise")
    frame["DayOfMonth"] = frame["FlightDate"].dt.day.astype("int8")
    frame["DayOfYear"] = frame["FlightDate"].dt.dayofyear.astype("int16")
    frame["DepHour"] = normalize_crs_hour(frame["CRSDepTime"])
    frame["ArrHour"] = normalize_crs_hour(frame["CRSArrTime"])
    frame["DepHour_sin"] = np.sin(2 * np.pi * frame["DepHour"] / 24).astype("float32")
    frame["DepHour_cos"] = np.cos(2 * np.pi * frame["DepHour"] / 24).astype("float32")
    frame["Month_sin"] = np.sin(2 * np.pi * frame["Month"] / 12).astype("float32")
    frame["Month_cos"] = np.cos(2 * np.pi * frame["Month"] / 12).astype("float32")
    frame["IsWeekend"] = frame["DayOfWeek"].isin([6, 7]).astype("int8")
    frame["IsHolidaySeason"] = frame["Month"].isin([11, 12, 1]).astype("int8")
    frame["Route"] = frame["Origin"].astype(str) + "_" + frame["Dest"].astype(str)
    distance = pd.to_numeric(frame["Distance"], errors="coerce")
    frame["DistanceBand"] = pd.cut(
        distance,
        bins=[-np.inf, 500, 1000, 2000, np.inf],
        labels=["short", "medium", "long", "very_long"],
    ).astype("string")
    frame["Cancelled"] = pd.to_numeric(frame["Cancelled"], errors="coerce").astype("Int8")
    frame["ArrDel15"] = pd.to_numeric(frame["ArrDel15"], errors="coerce").astype("Float32")
    frame.loc[frame["Cancelled"].eq(1), "ArrDel15"] = pd.NA

    columns = [
        "sample_id",
        "Year",
        "Month",
        "DayOfMonth",
        "DayOfWeek",
        "DayOfYear",
        "FlightDate",
        "Reporting_Airline",
        "DOT_ID_Reporting_Airline",
        "Flight_Number_Reporting_Airline",
        "Tail_Number",
        "OriginAirportID",
        "Origin",
        "OriginState",
        "DestAirportID",
        "Dest",
        "DestState",
        "CRSDepTime",
        "CRSArrTime",
        "CRSElapsedTime",
        "DepHour",
        "ArrHour",
        "DepHour_sin",
        "DepHour_cos",
        "Month_sin",
        "Month_cos",
        "IsWeekend",
        "IsHolidaySeason",
        "Distance",
        "DistanceBand",
        "Route",
        "ArrDel15",
        "Cancelled",
    ]
    return frame.loc[:, columns].reset_index(drop=True)


def iter_normalized_archive(
    archive_path: Path,
    *,
    year: int,
    month: int,
    chunksize: int = 200_000,
    allowed_airports: Collection[str] | None = None,
) -> Iterator[pd.DataFrame]:
    member = validate_bts_zip(archive_path)
    offset = 0
    with ZipFile(archive_path) as archive, archive.open(member) as source:
        reader = pd.read_csv(
            source,
            usecols=lambda column: column.strip() in RAW_COLUMNS,
            chunksize=chunksize,
            low_memory=False,
        )
        for raw in reader:
            normalized = normalize_bts_chunk(
                raw,
                source_year=year,
                source_month=month,
                source_offset=offset,
                allowed_airports=allowed_airports,
            )
            offset += len(raw)
            if not normalized.empty:
                yield normalized


def materialize_archive(
    archive_path: Path,
    output_path: Path,
    *,
    year: int,
    month: int,
    chunksize: int = 200_000,
    allowed_airports: Collection[str] | None = None,
) -> dict[str, int | str | float]:
    """Create one normalized Parquet file; never replace an existing output."""

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite normalized evidence: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated partial normalization exists: {partial}")

    writer: pq.ParquetWriter | None = None
    rows = 0
    delays = 0
    cancellations = 0
    try:
        for frame in iter_normalized_archive(
            archive_path,
            year=year,
            month=month,
            chunksize=chunksize,
            allowed_airports=allowed_airports,
        ):
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(partial, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(frame)
            delays += int(frame["ArrDel15"].fillna(0).sum())
            cancellations += int(frame["Cancelled"].fillna(0).sum())
    finally:
        if writer is not None:
            writer.close()
    if rows == 0 or not partial.exists():
        raise ValueError(f"normalization produced no rows for {archive_path}")
    partial.replace(output_path)
    return {
        "year": year,
        "month": month,
        "rows": rows,
        "delays": delays,
        "cancellations": cancellations,
        "delay_rate_non_cancelled": delays / max(1, rows - cancellations),
        "cancellation_rate": cancellations / rows,
        "raw_sha256": sha256_file(archive_path),
        "output_sha256": sha256_file(output_path),
        "output_path": output_path.as_posix(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--month", required=True, type=int)
    parser.add_argument("--airports-json", type=Path)
    parser.add_argument("--summary", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    airports: Collection[str] | None = None
    if args.airports_json:
        airports_payload = json.loads(args.airports_json.read_text(encoding="utf-8"))
        airports = airports_payload["airports"]
    summary = materialize_archive(
        args.archive,
        args.output,
        year=args.year,
        month=args.month,
        allowed_airports=airports,
    )
    if args.summary:
        write_canonical_json(args.summary, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
