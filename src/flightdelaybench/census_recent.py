"""Flight-number priors and attachment helpers for the official BTS census track."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .contracts import CENSUS_FLIGHT_RECENT_FEATURES, RECENT_WINDOWS_DAYS
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance
from .recent import (
    _VALUE_COLUMNS,
    _daily_aggregate,
    _global_features,
    _group_features,
    _load_level_year,
    _manifest_outputs,
    _merge_lookup,
    _write_level_partitions,
    attach_recent_features,
)

FLIGHT_KEYS = ("ScheduledFlightId",)
FLIGHT_SMOOTHING_STRENGTH = 75.0
PANDAS_MAX_SOURCE_ROWS = 5_000_000


def _pandas_flight_recent_records(
    source_paths: list[Path],
    *,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Reference implementation for small datasets and unit tests."""

    flight_parts: list[pd.DataFrame] = []
    global_parts: list[pd.DataFrame] = []
    for source_path in source_paths:
        frame = pd.read_parquet(
            source_path,
            columns=[
                "FlightDate",
                "ScheduledFlightId",
                "ArrDel15",
                "Cancelled",
                "delay_label_observed",
            ],
        )
        flight_parts.append(_daily_aggregate(frame, FLIGHT_KEYS))
        global_parts.append(_daily_aggregate(frame, ()))
    flight_daily = (
        pd.concat(flight_parts, ignore_index=True)
        .groupby([*FLIGHT_KEYS, "FlightDate"], observed=True, sort=False)[list(_VALUE_COLUMNS)]
        .sum()
        .reset_index()
    )
    global_daily = (
        pd.concat(global_parts, ignore_index=True)
        .groupby("FlightDate", observed=True, sort=False)[list(_VALUE_COLUMNS)]
        .sum()
        .reset_index()
    )
    _, global_raw = _global_features(global_daily)
    table = _group_features(
        flight_daily,
        level="flight",
        keys=FLIGHT_KEYS,
        global_raw=global_raw,
        smoothing_strength=FLIGHT_SMOOTHING_STRENGTH,
    )
    return _write_level_partitions(table, level="flight", output_dir=output_dir)


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _rolling_expressions(prefix: str) -> str:
    expressions: list[str] = []
    for window in RECENT_WINDOWS_DAYS:
        for value in _VALUE_COLUMNS:
            expressions.append(
                f"SUM({value}) OVER ("
                f"{prefix}ORDER BY FlightDate RANGE BETWEEN "
                f"INTERVAL '{window} days' PRECEDING AND INTERVAL '1 day' PRECEDING"
                f") AS {value}_{window}d"
            )
    return ",\n".join(expressions)


def _duckdb_feature_columns() -> str:
    columns: list[str] = []
    alpha = FLIGHT_SMOOTHING_STRENGTH
    for window in RECENT_WINDOWS_DAYS:
        global_delay = (
            f"g.delay_events_{window}d / NULLIF(g.delay_support_{window}d, 0.0)"
        )
        global_cancel = (
            f"g.cancel_events_{window}d / NULLIF(g.flight_count_{window}d, 0.0)"
        )
        columns.extend(
            [
                f"CAST((COALESCE(f.delay_events_{window}d, 0.0) + {alpha} * "
                f"({global_delay})) / (COALESCE(f.delay_support_{window}d, 0.0) + "
                f"{alpha}) AS FLOAT) AS recent_flight_delay_rate_{window}d",
                f"CAST((COALESCE(f.cancel_events_{window}d, 0.0) + {alpha} * "
                f"({global_cancel})) / (COALESCE(f.flight_count_{window}d, 0.0) + "
                f"{alpha}) AS FLOAT) AS recent_flight_cancel_rate_{window}d",
                f"CAST(LN(1.0 + COALESCE(f.flight_count_{window}d, 0.0)) AS FLOAT) "
                f"AS recent_flight_count_log1p_{window}d",
                f"CAST(LN(1.0 + COALESCE(f.delay_support_{window}d, 0.0)) AS FLOAT) "
                f"AS recent_flight_delay_support_log1p_{window}d",
            ]
        )
    return ",\n".join(columns)


def _duckdb_flight_recent_records(
    source_paths: list[Path],
    *,
    output_dir: Path,
    years: list[int],
    memory_limit: str,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Out-of-core exact RANGE windows, exported atomically one year at a time."""

    import duckdb

    spill_dir = output_dir / "_duckdb_spill"
    spill_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(f"SET memory_limit = {_sql_literal(memory_limit)}")
        connection.execute("SET threads = 4")
        connection.execute(f"SET temp_directory = {_sql_literal(spill_dir.as_posix())}")
        connection.read_parquet(
            [path.as_posix() for path in source_paths],
            union_by_name=True,
        ).create_view("census_source")
        connection.execute(
            """
            CREATE TEMP TABLE flight_daily AS
            SELECT
                CAST(ScheduledFlightId AS VARCHAR) AS ScheduledFlightId,
                CAST(FlightDate AS DATE) AS FlightDate,
                CAST(COUNT(*) AS DOUBLE) AS flight_count,
                CAST(SUM(CAST(Cancelled AS DOUBLE)) AS DOUBLE) AS cancel_events,
                CAST(SUM(CAST(delay_label_observed AS DOUBLE)) AS DOUBLE) AS delay_support,
                CAST(SUM(COALESCE(CAST(ArrDel15 AS DOUBLE), 0.0)
                         * CAST(delay_label_observed AS DOUBLE)) AS DOUBLE) AS delay_events
            FROM census_source
            GROUP BY ScheduledFlightId, FlightDate
            """
        )
        connection.execute(
            """
            CREATE TEMP TABLE global_daily AS
            SELECT
                FlightDate,
                SUM(flight_count) AS flight_count,
                SUM(cancel_events) AS cancel_events,
                SUM(delay_support) AS delay_support,
                SUM(delay_events) AS delay_events
            FROM flight_daily
            GROUP BY FlightDate
            """
        )
        records: list[dict[str, Any]] = []
        flight_windows = _rolling_expressions("PARTITION BY ScheduledFlightId ")
        global_windows = _rolling_expressions("")
        feature_columns = _duckdb_feature_columns()
        for year in years:
            year_start = f"{year:04d}-01-01"
            year_end = f"{year:04d}-12-31"
            lookback_start = (pd.Timestamp(year_start) - pd.Timedelta(days=90)).date().isoformat()
            query = f"""
                WITH flight_roll AS (
                    SELECT ScheduledFlightId, FlightDate, {flight_windows}
                    FROM flight_daily
                    WHERE FlightDate BETWEEN DATE '{lookback_start}' AND DATE '{year_end}'
                ),
                global_roll AS (
                    SELECT FlightDate, {global_windows}
                    FROM global_daily
                    WHERE FlightDate BETWEEN DATE '{lookback_start}' AND DATE '{year_end}'
                )
                SELECT
                    f.ScheduledFlightId,
                    CAST(f.FlightDate AS TIMESTAMP) AS FlightDate,
                    {feature_columns}
                FROM flight_roll AS f
                INNER JOIN global_roll AS g USING (FlightDate)
                WHERE f.FlightDate BETWEEN DATE '{year_start}' AND DATE '{year_end}'
                ORDER BY f.FlightDate, f.ScheduledFlightId
            """
            output_path = output_dir / "level=flight" / f"year={year}.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            partial = output_path.with_suffix(output_path.suffix + ".part")
            if output_path.exists() or partial.exists():
                raise FileExistsError(f"refusing to replace flight-history evidence: {output_path}")
            connection.execute(
                f"COPY ({query}) TO {_sql_literal(partial.as_posix())} "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
            )
            parquet = pq.ParquetFile(partial)
            statistics = connection.execute(
                "SELECT MIN(FlightDate), MAX(FlightDate) FROM read_parquet(?)",
                [partial.as_posix()],
            ).fetchone()
            if statistics is None or statistics[0] is None or statistics[1] is None:
                raise ValueError(f"DuckDB produced an empty flight-history year: {year}")
            parquet_rows = int(parquet.metadata.num_rows)
            parquet.close()
            partial.replace(output_path)
            record = {
                "level": "flight",
                "year": year,
                "path": output_path.as_posix(),
                "rows": parquet_rows,
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "first_date": pd.Timestamp(statistics[0]).date().isoformat(),
                "last_date": pd.Timestamp(statistics[1]).date().isoformat(),
            }
            records.append(record)
            if verbose:
                print(
                    f"flight-history DuckDB year={year}: {record['rows']} rows",
                    flush=True,
                )
        return records
    finally:
        connection.close()


def build_flight_recent_tables(
    *,
    feature_manifest: Path,
    output_dir: Path,
    output_manifest: Path,
    verbose: bool = False,
    backend: str = "auto",
    duckdb_memory_limit: str = "8GB",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse flight-history directory: {output_dir}")
    if output_manifest.exists():
        raise FileExistsError(f"refusing to overwrite flight-history manifest: {output_manifest}")
    repository_root, source_manifest, source_records = _manifest_outputs(feature_manifest)
    output_dir.mkdir(parents=True)
    if backend not in {"auto", "pandas", "duckdb"}:
        raise ValueError(f"unsupported flight-history backend: {backend}")
    total_source_rows = sum(int(record["rows"]) for record in source_records)
    selected_backend = (
        "duckdb" if backend == "auto" and total_source_rows > PANDAS_MAX_SOURCE_ROWS else backend
    )
    if selected_backend == "auto":
        selected_backend = "pandas"
    source_paths: list[Path] = []
    verified_sources: list[dict[str, Any]] = []
    for index, record in enumerate(source_records, start=1):
        source_path = Path(str(record["path"]))
        if not source_path.is_absolute():
            source_path = repository_root / source_path
        actual_hash = sha256_file(source_path)
        if actual_hash != record["sha256"]:
            raise ValueError(f"census source hash mismatch: {source_path}")
        source_paths.append(source_path)
        verified_sources.append(
            {"path": str(record["path"]), "rows": int(record["rows"]), "sha256": actual_hash}
        )
        if verbose:
            print(f"flight-history source={index}/{len(source_records)}", flush=True)
    years: set[int] = set()
    for record, source_path in zip(source_records, source_paths, strict=True):
        if "year" in record:
            years.add(int(record["year"]))
        else:
            dates = pd.read_parquet(source_path, columns=["FlightDate"])["FlightDate"]
            years.update(pd.to_datetime(dates, errors="raise").dt.year.unique().tolist())
    if selected_backend == "duckdb":
        records = _duckdb_flight_recent_records(
            source_paths,
            output_dir=output_dir,
            years=sorted(years),
            memory_limit=duckdb_memory_limit,
            verbose=verbose,
        )
    else:
        records = _pandas_flight_recent_records(source_paths, output_dir=output_dir)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "DERIVED_STRICT_PRIOR_DAY_FLIGHT_NUMBER_HISTORY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_feature_manifest": feature_manifest.as_posix(),
        "source_feature_manifest_sha256": sha256_file(feature_manifest),
        "source_feature_variant": source_manifest["feature_variant"],
        "source_partitions_verified": verified_sources,
        "source_rows": total_source_rows,
        "builder_backend": selected_backend,
        "duckdb_memory_limit": duckdb_memory_limit if selected_backend == "duckdb" else None,
        "cutoff_rule": "closed-left calendar windows [target_date-window, target_date)",
        "target_day_outcomes_excluded": True,
        "key": list(FLIGHT_KEYS),
        "windows_days": list(RECENT_WINDOWS_DAYS),
        "smoothing_strength": FLIGHT_SMOOTHING_STRENGTH,
        "features": list(CENSUS_FLIGHT_RECENT_FEATURES),
        "outputs": records,
        "total_lookup_rows": sum(int(record["rows"]) for record in records),
        "provenance": capture_provenance(
            (Path(__file__), Path(__file__).with_name("recent.py"))
        ),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(output_manifest, manifest)
    return manifest


def attach_census_recent_features(
    frame: pd.DataFrame,
    *,
    recent_dir: Path,
    flight_recent_dir: Path,
) -> pd.DataFrame:
    """Attach cohort and flight-number histories with target-day exclusion."""

    common = attach_recent_features(
        frame,
        recent_dir=recent_dir,
        fallback_strategy="recent_global",
    )
    common["__flight_recent_order"] = np.arange(len(common), dtype=np.int64)
    pieces: list[pd.DataFrame] = []
    for year, year_frame in common.groupby("Year", sort=False, observed=True):
        lookup = _load_level_year(flight_recent_dir, "flight", int(year))
        current = _merge_lookup(
            year_frame,
            lookup,
            left_key="ScheduledFlightId",
            right_key="ScheduledFlightId",
        )
        for window in RECENT_WINDOWS_DAYS:
            for outcome in ("delay", "cancel"):
                name = f"recent_flight_{outcome}_rate_{window}d"
                current[name] = pd.to_numeric(current[name], errors="coerce").fillna(
                    current[f"recent_global_{outcome}_rate_{window}d"]
                )
            for statistic in ("count_log1p", "delay_support_log1p"):
                name = f"recent_flight_{statistic}_{window}d"
                current[name] = pd.to_numeric(current[name], errors="coerce").fillna(0.0)
        pieces.append(current)
    attached = pd.concat(pieces).sort_values("__flight_recent_order", kind="mergesort")
    values = attached.loc[:, list(CENSUS_FLIGHT_RECENT_FEATURES)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("flight-number prior attachment produced non-finite values")
    if len(attached) != len(frame):
        raise AssertionError("flight-number prior attachment changed row count")
    return attached.drop(columns="__flight_recent_order").reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", choices=["auto", "pandas", "duckdb"], default="auto")
    parser.add_argument("--duckdb-memory-limit", default="8GB")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = build_flight_recent_tables(
        feature_manifest=args.feature_manifest,
        output_dir=args.output_dir,
        output_manifest=args.manifest,
        verbose=True,
        backend=args.backend,
        duckdb_memory_limit=args.duckdb_memory_limit,
    )
    print(
        json.dumps(
            {
                "manifest": args.manifest.as_posix(),
                "status": result["status"],
                "outputs": len(result["outputs"]),
                "total_lookup_rows": result["total_lookup_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
