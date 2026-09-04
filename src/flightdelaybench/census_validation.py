"""Validate official BTS census lineage, hashes, labels, and reconstruction."""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .bts import iter_normalized_archive
from .census_normalization import _resolve_evidence_path, _verify_json_self_hash
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

REQUIRED_COLUMNS = {
    "sample_id",
    "Year",
    "Month",
    "FlightDate",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "ScheduledFlightId",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "CRSDepMinutes",
    "CRSArrMinutes",
    "ArrDel15",
    "Cancelled",
    "Diverted",
    "delay_label_observed",
    "joint_label_observed",
    "disruption_state",
}


def _raw_period_map(
    census: dict[str, Any],
    *,
    repository_root: Path,
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]]]:
    periods: dict[tuple[int, int], dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []
    for record in census["raw_manifests"]:
        manifest_path = _resolve_evidence_path(str(record["path"]), repository_root)
        payload = _verify_json_self_hash(manifest_path)
        actual_hash = sha256_file(manifest_path)
        if actual_hash != record["sha256"] or payload["manifest_sha256"] != record["self_hash"]:
            raise ValueError(f"raw-manifest provenance mismatch: {manifest_path}")
        checks.append(
            {
                "path": manifest_path.as_posix(),
                "sha256": actual_hash,
                "self_hash": payload["manifest_sha256"],
                "records": len(payload["records"]),
            }
        )
        for raw in payload["records"]:
            key = (int(raw["year"]), int(raw["month"]))
            if key in periods:
                raise ValueError(f"duplicate raw period in census provenance: {key}")
            raw_path = _resolve_evidence_path(str(raw["local_path"]), repository_root)
            periods[key] = {**raw, "resolved_path": raw_path}
    return periods, checks


def _calendar_periods(start: str, end: str) -> list[tuple[int, int]]:
    """Expand inclusive YYYY-MM lineage bounds without accepting gaps."""

    start_year, start_month = (int(value) for value in start.split("-"))
    end_year, end_month = (int(value) for value in end.split("-"))
    if not 1 <= start_month <= 12 or not 1 <= end_month <= 12:
        raise ValueError("lineage segment month must be within 1..12")
    first = start_year * 12 + start_month - 1
    last = end_year * 12 + end_month - 1
    if first > last:
        raise ValueError("lineage segment starts after it ends")
    return [(value // 12, value % 12 + 1) for value in range(first, last + 1)]


def _verify_lineage_addendum(
    path: Path,
    *,
    census_manifest: Path,
    census: dict[str, Any],
) -> dict[str, Any]:
    """Bind an adjudicated mixed-revision history to every census partition."""

    payload = _verify_json_self_hash(path)
    census_file_hash = sha256_file(census_manifest)
    if payload.get("census_manifest_sha256") != census_file_hash:
        raise ValueError("lineage addendum does not bind the census manifest file")
    if payload.get("census_manifest_self_hash") != census.get("manifest_sha256"):
        raise ValueError("lineage addendum does not bind the census manifest self-hash")
    expected = sorted(
        (int(record["year"]), int(record["month"])) for record in census["outputs"]
    )
    covered: list[tuple[int, int]] = []
    for segment in payload.get("partition_segments", []):
        periods = _calendar_periods(str(segment["start"]), str(segment["end"]))
        if int(segment.get("partitions", -1)) != len(periods):
            raise ValueError("lineage segment partition count disagrees with its bounds")
        if not str(segment.get("bts_source_sha256", "")):
            raise ValueError("lineage segment lacks its normalizer source hash")
        covered.extend(periods)
    if covered != expected:
        raise ValueError("lineage segments do not cover each census partition exactly once")
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "self_hash": payload["manifest_sha256"],
        "status": payload["status"],
        "segments": len(payload["partition_segments"]),
        "partitions_covered": len(covered),
    }


def _partition_invariants(
    path: Path,
    *,
    year: int,
    month: int,
    airports: set[str],
) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    missing = sorted(REQUIRED_COLUMNS - set(parquet.schema.names))
    if missing:
        raise ValueError(f"partition lacks required census columns: {missing}")
    rows = 0
    cancellations = 0
    diversions = 0
    delay_support = 0
    joint_ineligible = 0
    seen_ids: set[str] = set()
    first_date: pd.Timestamp | None = None
    last_date: pd.Timestamp | None = None
    columns = sorted(REQUIRED_COLUMNS)
    for batch in parquet.iter_batches(batch_size=150_000, columns=columns):
        frame = batch.to_pandas()
        rows += len(frame)
        ids = frame["sample_id"].astype(str)
        if ids.duplicated().any() or any(value in seen_ids for value in ids):
            raise ValueError(f"duplicate sample_id in census partition: {path}")
        seen_ids.update(ids)
        if not frame["Year"].eq(year).all() or not frame["Month"].eq(month).all():
            raise ValueError(f"period values disagree with census partition: {path}")
        if not frame["Origin"].isin(airports).all() or not frame["Dest"].isin(airports).all():
            raise ValueError(f"airport outside frozen top-100 cohort: {path}")
        cancelled = frame["Cancelled"].eq(1)
        diverted = frame["Diverted"].eq(1)
        observed = frame["delay_label_observed"].eq(1)
        joint = frame["joint_label_observed"].eq(1)
        state = pd.to_numeric(frame["disruption_state"], errors="raise")
        delay = pd.to_numeric(frame["ArrDel15"], errors="coerce")
        if frame.loc[cancelled | diverted, "ArrDel15"].notna().any():
            raise ValueError(f"cancelled/diverted row retained an arrival label: {path}")
        if (observed != (~cancelled & ~diverted & delay.notna())).any():
            raise ValueError(f"delay eligibility invariant failed: {path}")
        if (joint != (cancelled | observed)).any():
            raise ValueError(f"joint eligibility invariant failed: {path}")
        expected_state = np.select(
            [cancelled, observed & delay.eq(1), observed & delay.eq(0)],
            [2, 1, 0],
            default=-1,
        )
        if not np.array_equal(state.to_numpy(), expected_state):
            raise ValueError(f"disruption-state invariant failed: {path}")
        dates = pd.to_datetime(frame["FlightDate"], errors="raise")
        batch_first = dates.min()
        batch_last = dates.max()
        first_date = batch_first if first_date is None else min(first_date, batch_first)
        last_date = batch_last if last_date is None else max(last_date, batch_last)
        cancellations += int(cancelled.sum())
        diversions += int(diverted.sum())
        delay_support += int(observed.sum())
        joint_ineligible += int((~joint).sum())
    if rows != parquet.metadata.num_rows or first_date is None or last_date is None:
        raise ValueError(f"census partition scan was incomplete: {path}")
    return {
        "rows": rows,
        "first_date": first_date.date().isoformat(),
        "last_date": last_date.date().isoformat(),
        "cancellations": cancellations,
        "diversions": diversions,
        "delay_support": delay_support,
        "joint_label_ineligible": joint_ineligible,
    }


def _reconstruction_check(
    *,
    raw_path: Path,
    normalized_path: Path,
    year: int,
    month: int,
    airports: tuple[str, ...],
    rows: int = 256,
) -> dict[str, Any]:
    reconstructed = next(
        iter_normalized_archive(
            raw_path,
            year=year,
            month=month,
            allowed_airports=airports,
            retain_diverted=True,
        )
    ).head(rows)
    stored = pd.read_parquet(normalized_path).head(len(reconstructed))
    if list(stored.columns) != list(reconstructed.columns):
        raise ValueError("independent census reconstruction column order differs")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mismatched null-like values.*")
        pd.testing.assert_frame_equal(
            stored.reset_index(drop=True),
            reconstructed.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-6,
            atol=1e-6,
        )
    return {
        "year": year,
        "month": month,
        "rows_compared": len(reconstructed),
        "raw_path": raw_path.as_posix(),
        "normalized_path": normalized_path.as_posix(),
        "status": "PASS",
    }


def validate_census(
    *,
    census_manifest: Path,
    output_path: Path,
    reconstruction_partitions: int = 12,
    lineage_addendum: Path | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite census validation: {output_path}")
    repository_root = Path(__file__).resolve().parents[2]
    census = _verify_json_self_hash(census_manifest)
    lineage_check = (
        _verify_lineage_addendum(
            lineage_addendum,
            census_manifest=census_manifest,
            census=census,
        )
        if lineage_addendum is not None
        else None
    )
    periods, raw_manifest_checks = _raw_period_map(census, repository_root=repository_root)
    airports = tuple(str(code) for code in census["airports"])
    airport_set = set(airports)
    errors: list[str] = []
    partitions: list[dict[str, Any]] = []
    output_records = list(census["outputs"])
    for record in output_records:
        year = int(record["year"])
        month = int(record["month"])
        path = _resolve_evidence_path(str(record["path"]), repository_root)
        try:
            if sha256_file(path) != record["sha256"]:
                raise ValueError("normalized SHA-256 mismatch")
            invariants = _partition_invariants(
                path,
                year=year,
                month=month,
                airports=airport_set,
            )
            for name in (
                "rows",
                "first_date",
                "last_date",
                "cancellations",
                "diversions",
                "delay_support",
                "joint_label_ineligible",
            ):
                if invariants[name] != record[name]:
                    raise ValueError(
                        f"recorded {name}={record[name]!r}, reconstructed {invariants[name]!r}"
                    )
            partitions.append({"year": year, "month": month, "status": "PASS", **invariants})
        except Exception as error:  # validation must retain all observed failures
            errors.append(f"{year}-{month:02d}: {type(error).__name__}: {error}")
            partitions.append({"year": year, "month": month, "status": "FAIL"})

    reconstructions: list[dict[str, Any]] = []
    if output_records and reconstruction_partitions > 0:
        indices = np.linspace(
            0,
            len(output_records) - 1,
            min(reconstruction_partitions, len(output_records)),
            dtype=int,
        )
        for index in sorted(set(indices.tolist())):
            record = output_records[index]
            year, month = int(record["year"]), int(record["month"])
            raw = periods[(year, month)]
            normalized_path = _resolve_evidence_path(str(record["path"]), repository_root)
            try:
                if sha256_file(raw["resolved_path"]) != raw["sha256"]:
                    raise ValueError("raw archive SHA-256 mismatch")
                reconstructions.append(
                    _reconstruction_check(
                        raw_path=raw["resolved_path"],
                        normalized_path=normalized_path,
                        year=year,
                        month=month,
                        airports=airports,
                    )
                )
            except Exception as error:  # validation must retain all observed failures
                errors.append(
                    f"reconstruction {year}-{month:02d}: {type(error).__name__}: {error}"
                )
                reconstructions.append({"year": year, "month": month, "status": "FAIL"})

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "census_manifest": census_manifest.as_posix(),
        "census_manifest_sha256": sha256_file(census_manifest),
        "census_manifest_self_hash": census["manifest_sha256"],
        "lineage_addendum": lineage_check,
        "raw_manifest_checks": raw_manifest_checks,
        "partitions": partitions,
        "independent_reconstructions": reconstructions,
        "errors": errors,
        "checks": {
            "all_declared_checks_passed": not errors,
            "required_schema": "evaluated",
            "full_partition_hashes": "evaluated",
            "unique_sample_id_within_partition": "evaluated",
            "frozen_airport_domain": "evaluated",
            "period_alignment": "evaluated",
            "target_eligibility_and_state": "evaluated",
            "raw_to_normalized_reconstruction": "evaluated",
            "mixed_revision_lineage": "evaluated" if lineage_check else "not_supplied",
        },
        "provenance": capture_provenance(
            (
                Path(__file__),
                Path(__file__).with_name("census_normalization.py"),
                Path(__file__).with_name("bts.py"),
            )
        ),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reconstruction-partitions", type=int, default=12)
    parser.add_argument("--lineage-addendum", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = validate_census(
        census_manifest=args.census_manifest,
        output_path=args.output,
        reconstruction_partitions=args.reconstruction_partitions,
        lineage_addendum=args.lineage_addendum,
    )
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "status": report["status"],
                "partitions": len(report["partitions"]),
                "reconstructions": len(report["independent_reconstructions"]),
                "errors": report["errors"],
            },
            indent=2,
        )
    )
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
