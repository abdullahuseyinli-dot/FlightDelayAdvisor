"""Audit daily census continuity and the boundary of the top-airport network."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import duckdb
import pandas as pd

from .acquisition import validate_bts_zip
from .census_normalization import _resolve_evidence_path, _verify_json_self_hash
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance


def classify_network_scope(frame: pd.DataFrame, airports: set[str]) -> pd.DataFrame:
    """Return mutually exclusive induced, boundary, and outside masks."""

    origin_inside = frame["Origin"].astype(str).isin(airports)
    dest_inside = frame["Dest"].astype(str).isin(airports)
    return pd.DataFrame(
        {
            "induced": origin_inside & dest_inside,
            "boundary": origin_inside ^ dest_inside,
            "outside": ~origin_inside & ~dest_inside,
        },
        index=frame.index,
    )


def _expected_dates(years: Sequence[int]) -> pd.DatetimeIndex:
    pieces = [
        pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        for year in sorted(set(years))
    ]
    if not pieces:
        return pd.DatetimeIndex([])
    return pieces[0].append(pieces[1:])


def _distribution(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="raise")
    return {
        "min": int(numeric.min()),
        "p05": float(numeric.quantile(0.05)),
        "median": float(numeric.median()),
        "mean": float(numeric.mean()),
        "p95": float(numeric.quantile(0.95)),
        "max": int(numeric.max()),
    }


def _read_raw_schedule_chunks(path: Path, *, chunksize: int) -> Iterator[pd.DataFrame]:
    member = validate_bts_zip(path)
    archive = ZipFile(path)
    source = archive.open(member)
    reader = pd.read_csv(
        source,
        usecols=["FlightDate", "Origin", "Dest"],
        chunksize=chunksize,
        low_memory=False,
    )

    def iterator() -> Iterator[pd.DataFrame]:
        try:
            for frame in reader:
                frame.columns = [str(column).strip() for column in frame.columns]
                yield frame
        finally:
            source.close()
            archive.close()

    return iterator()


def _raw_boundary_period(
    *,
    raw_path: Path,
    expected_sha256: str,
    year: int,
    month: int,
    airports: set[str],
    chunksize: int,
) -> tuple[dict[str, Any], Counter[str], Counter[str]]:
    if sha256_file(raw_path) != expected_sha256:
        raise ValueError(f"raw archive hash mismatch: {raw_path}")
    counts: Counter[str] = Counter()
    incident_by_airport: Counter[str] = Counter()
    boundary_by_airport: Counter[str] = Counter()
    observed_dates: set[str] = set()
    for frame in _read_raw_schedule_chunks(raw_path, chunksize=chunksize):
        dates = pd.to_datetime(frame["FlightDate"], errors="raise")
        if not dates.dt.year.eq(year).all() or not dates.dt.month.eq(month).all():
            raise ValueError(f"raw archive period mismatch: {raw_path}")
        observed_dates.update(dates.dt.date.astype(str).unique().tolist())
        scope = classify_network_scope(frame, airports)
        counts["raw_rows"] += len(frame)
        for name in ("induced", "boundary", "outside"):
            counts[f"{name}_rows"] += int(scope[name].sum())

        origin = frame["Origin"].astype(str)
        dest = frame["Dest"].astype(str)
        origin_inside = origin.isin(airports)
        dest_inside = dest.isin(airports)
        incident_by_airport.update(origin.loc[origin_inside].value_counts().to_dict())
        incident_by_airport.update(dest.loc[dest_inside].value_counts().to_dict())
        boundary_by_airport.update(
            origin.loc[origin_inside & ~dest_inside].value_counts().to_dict()
        )
        boundary_by_airport.update(dest.loc[dest_inside & ~origin_inside].value_counts().to_dict())

    expected = pd.date_range(
        pd.Timestamp(year=year, month=month, day=1),
        pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0),
        freq="D",
    )
    missing_dates = sorted(set(expected.date.astype(str)) - observed_dates)
    raw_rows = counts["raw_rows"]
    incident_rows = counts["induced_rows"] + counts["boundary_rows"]
    record: dict[str, Any] = {
        "year": year,
        "month": month,
        **dict(counts),
        "top100_touching_rows": incident_rows,
        "induced_fraction_of_all_raw": counts["induced_rows"] / raw_rows,
        "boundary_fraction_of_top100_touching": counts["boundary_rows"] / incident_rows,
        "observed_dates": len(observed_dates),
        "missing_dates": missing_dates,
        "status": "PASS" if not missing_dates else "FAIL",
    }
    return record, incident_by_airport, boundary_by_airport


def audit_census_context(
    *,
    census_manifest_path: Path,
    raw_manifest_paths: tuple[Path, ...],
    output_path: Path,
    boundary_years: tuple[int, ...] = (2024, 2025),
    chunksize: int = 500_000,
) -> dict[str, Any]:
    """Create a checksummed audit without interpreting zero-service days as data loss."""

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite context audit: {output_path}")
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    started = time.perf_counter()
    repository_root = Path(__file__).resolve().parents[2]
    census = _verify_json_self_hash(census_manifest_path)
    airports = tuple(str(code) for code in census["airports"])
    if len(airports) != 100 or len(set(airports)) != 100:
        raise ValueError("census must bind exactly 100 unique airports")
    airport_set = set(airports)
    outputs = list(census["outputs"])
    years = tuple(sorted(int(year) for year in census["years"]))
    expected_periods = {(year, month) for year in years for month in range(1, 13)}
    recorded_periods = {(int(row["year"]), int(row["month"])) for row in outputs}
    if recorded_periods != expected_periods:
        raise ValueError("census does not cover every expected year-month")

    parquet_paths: list[str] = []
    census_rows_by_period: dict[tuple[int, int], int] = {}
    for record in outputs:
        path = _resolve_evidence_path(str(record["path"]), repository_root)
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"census partition hash mismatch: {path}")
        parquet_paths.append(path.as_posix())
        census_rows_by_period[(int(record["year"]), int(record["month"]))] = int(
            record["rows"]
        )

    connection = duckdb.connect(database=":memory:")
    try:
        relation = connection.read_parquet(parquet_paths)
        relation.create_view("census")
        daily = connection.execute(
            """
            SELECT CAST(FlightDate AS DATE) AS flight_date, COUNT(*)::BIGINT AS flights
            FROM census
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchdf()
        airport_daily = connection.execute(
            """
            SELECT flight_date, airport,
                   SUM(outbound)::BIGINT AS outbound,
                   SUM(inbound)::BIGINT AS inbound
            FROM (
                SELECT CAST(FlightDate AS DATE) AS flight_date, Origin AS airport,
                       1::BIGINT AS outbound, 0::BIGINT AS inbound
                FROM census
                UNION ALL
                SELECT CAST(FlightDate AS DATE) AS flight_date, Dest AS airport,
                       0::BIGINT AS outbound, 1::BIGINT AS inbound
                FROM census
            )
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        ).fetchdf()
    finally:
        connection.close()

    daily["flight_date"] = pd.to_datetime(daily["flight_date"])
    expected_dates = _expected_dates(years)
    observed_date_set = set(daily["flight_date"])
    missing_dates = [date.date().isoformat() for date in expected_dates if date not in observed_date_set]
    extra_dates = sorted(
        date.date().isoformat() for date in observed_date_set if date not in set(expected_dates)
    )

    complete_index = pd.MultiIndex.from_product(
        [expected_dates, airports], names=["flight_date", "airport"]
    )
    airport_daily["flight_date"] = pd.to_datetime(airport_daily["flight_date"])
    complete_airport_daily = (
        airport_daily.set_index(["flight_date", "airport"])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )
    complete_airport_daily["year"] = complete_airport_daily["flight_date"].dt.year
    complete_airport_daily["any_activity"] = (
        complete_airport_daily["outbound"] + complete_airport_daily["inbound"]
    ).gt(0)

    per_year: list[dict[str, Any]] = []
    for year in years:
        year_daily = daily.loc[daily["flight_date"].dt.year.eq(year), "flights"]
        year_airport = complete_airport_daily.loc[complete_airport_daily["year"].eq(year)]
        expected_year_days = len(pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"))
        per_year.append(
            {
                "year": year,
                "rows": int(year_daily.sum()),
                "expected_dates": expected_year_days,
                "observed_dates": int(year_daily.size),
                "missing_dates": [
                    date for date in missing_dates if date.startswith(f"{year}-")
                ],
                "daily_flights": _distribution(year_daily),
                "airport_days": len(year_airport),
                "zero_any_activity_airport_days": int((~year_airport["any_activity"]).sum()),
                "zero_outbound_airport_days": int(year_airport["outbound"].eq(0).sum()),
                "zero_inbound_airport_days": int(year_airport["inbound"].eq(0).sum()),
            }
        )

    per_airport: list[dict[str, Any]] = []
    for airport, group in complete_airport_daily.groupby("airport", sort=True):
        per_airport.append(
            {
                "airport": str(airport),
                "airport_days": len(group),
                "zero_any_activity_days": int((~group["any_activity"]).sum()),
                "zero_outbound_days": int(group["outbound"].eq(0).sum()),
                "zero_inbound_days": int(group["inbound"].eq(0).sum()),
                "mean_daily_incident_flights": float(
                    (group["outbound"] + group["inbound"]).mean()
                ),
            }
        )

    raw_records: dict[tuple[int, int], dict[str, Any]] = {}
    raw_sources: list[dict[str, Any]] = []
    for manifest_path in raw_manifest_paths:
        raw_manifest = _verify_json_self_hash(manifest_path)
        raw_sources.append(
            {
                "path": manifest_path.as_posix(),
                "sha256": sha256_file(manifest_path),
                "self_hash": raw_manifest["manifest_sha256"],
            }
        )
        for record in raw_manifest["records"]:
            key = (int(record["year"]), int(record["month"]))
            if key in raw_records:
                raise ValueError(f"duplicate raw period supplied: {key}")
            raw_records[key] = record

    boundary_periods: list[dict[str, Any]] = []
    incident_total: Counter[str] = Counter()
    boundary_total: Counter[str] = Counter()
    for year in sorted(set(boundary_years)):
        for month in range(1, 13):
            key = (year, month)
            if key not in raw_records:
                raise ValueError(f"missing raw manifest period for boundary audit: {key}")
            raw = raw_records[key]
            raw_path = _resolve_evidence_path(str(raw["local_path"]), repository_root)
            period, incident, boundary = _raw_boundary_period(
                raw_path=raw_path,
                expected_sha256=str(raw["sha256"]),
                year=year,
                month=month,
                airports=airport_set,
                chunksize=chunksize,
            )
            expected_rows = census_rows_by_period.get(key)
            period["census_rows"] = expected_rows
            period["induced_rows_match_census"] = period["induced_rows"] == expected_rows
            if not period["induced_rows_match_census"]:
                period["status"] = "FAIL"
            boundary_periods.append(period)
            incident_total.update(incident)
            boundary_total.update(boundary)

    totals: Counter[str] = Counter()
    for period in boundary_periods:
        for metric in ("raw_rows", "induced_rows", "boundary_rows", "outside_rows"):
            totals[metric] += int(period[metric])
    touching = totals["induced_rows"] + totals["boundary_rows"]
    boundary_airports = [
        {
            "airport": airport,
            "incident_endpoint_count": int(incident_total[airport]),
            "boundary_endpoint_count": int(boundary_total[airport]),
            "boundary_fraction": boundary_total[airport] / incident_total[airport],
        }
        for airport in sorted(airports)
    ]
    errors: list[str] = []
    if missing_dates:
        errors.append("one or more expected census dates have no retained flights")
    if extra_dates:
        errors.append("census contains dates outside its declared years")
    failed_boundary = [
        f"{row['year']}-{row['month']:02d}"
        for row in boundary_periods
        if row["status"] != "PASS"
    ]
    if failed_boundary:
        errors.append(f"raw/census boundary reconstruction failed: {failed_boundary}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": (
            "PASS_TEMPORAL_CENSUS_BOUNDARY_QUANTIFIED" if not errors else "FAIL"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "census_manifest": {
            "path": census_manifest_path.as_posix(),
            "sha256": sha256_file(census_manifest_path),
            "self_hash": census["manifest_sha256"],
        },
        "raw_manifests": raw_sources,
        "cohort_definition": {
            "airports": list(airports),
            "retention_rule": "Origin and Dest must both be in the frozen top-100 set",
            "network_interpretation": "complete induced top-100 schedule subgraph, not a nationwide network",
        },
        "temporal_completeness": {
            "years": list(years),
            "partitions": len(outputs),
            "rows": int(daily["flights"].sum()),
            "expected_dates": len(expected_dates),
            "observed_dates": len(observed_date_set),
            "missing_dates": missing_dates,
            "extra_dates": extra_dates,
            "daily_flights": _distribution(daily["flights"]),
            "per_year": per_year,
        },
        "airport_day_continuity": {
            "interpretation": (
                "Zero-service airport-days are disclosed observations and are not, by themselves, "
                "evidence of source-data loss."
            ),
            "expected_airport_days": len(complete_airport_daily),
            "zero_any_activity_airport_days": int(
                (~complete_airport_daily["any_activity"]).sum()
            ),
            "zero_outbound_airport_days": int(complete_airport_daily["outbound"].eq(0).sum()),
            "zero_inbound_airport_days": int(complete_airport_daily["inbound"].eq(0).sum()),
            "per_airport": per_airport,
        },
        "network_boundary": {
            "audited_years": sorted(set(boundary_years)),
            "raw_rows": totals["raw_rows"],
            "induced_rows": totals["induced_rows"],
            "boundary_rows": totals["boundary_rows"],
            "outside_rows": totals["outside_rows"],
            "top100_touching_rows": touching,
            "induced_fraction_of_all_raw": totals["induced_rows"] / totals["raw_rows"],
            "boundary_fraction_of_top100_touching": totals["boundary_rows"] / touching,
            "periods": boundary_periods,
            "per_airport": boundary_airports,
        },
        "errors": errors,
        "claim_limit": (
            "This audit proves date continuity and exact raw-to-induced row counts for the "
            "audited boundary years. It does not prove that a zero-service airport-day is a "
            "missing-data event, recover flights outside the induced cohort, or turn the BTS "
            "final schedule into an issue-time T-24 schedule."
        ),
        "environment": {
            "python": platform.python_version(),
            "duckdb": version("duckdb"),
            "pandas": version("pandas"),
        },
        "provenance": capture_provenance((Path(__file__),)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-manifest", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundary-years", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--chunksize", type=int, default=500_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_census_context(
        census_manifest_path=args.census_manifest,
        raw_manifest_paths=tuple(args.raw_manifest),
        output_path=args.output,
        boundary_years=tuple(args.boundary_years),
        chunksize=args.chunksize,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": args.output.as_posix(),
                "dates": report["temporal_completeness"]["observed_dates"],
                "boundary_fraction": report["network_boundary"][
                    "boundary_fraction_of_top100_touching"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
