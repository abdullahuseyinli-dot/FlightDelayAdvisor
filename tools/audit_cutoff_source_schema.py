"""Audit only 2024 source schemas, without reading flight outcomes or changing data."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from flightdelaybench.cutoff_dataset import guard_parquet_years
from flightdelaybench.hashing import canonical_json_sha256, write_canonical_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--normalized-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("schema evidence is create-only")
    required = ("event_time_utc", "source_published_at_utc", "available_at_utc", "cancel_label_available_at_utc", "delay_label_available_at_utc")
    event_fields = ("DepTime", "ArrTime", "DepDelay", "ArrDelay", "ActualElapsedTime")
    records = []
    for month in range(1, 13):
        raw = args.raw_dir / "2024" / f"on_time_2024_{month:02}.zip"
        normalized = args.normalized_dir / "year=2024" / f"month={month:02}.parquet"
        with zipfile.ZipFile(raw) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(members) != 1:
                raise ValueError("ambiguous raw CSV source")
            with archive.open(members[0]) as handle:
                columns = next(csv.reader(io.TextIOWrapper(handle, encoding="utf-8-sig")))
        guard_parquet_years(normalized, {2024})
        metadata = pq.read_metadata(normalized)
        normalized_columns = metadata.schema.names
        records.append({
            "month": month, "raw_archive": raw.as_posix(), "raw_bytes": raw.stat().st_size,
            "raw_header_sha256": canonical_json_sha256(columns), "raw_columns": columns,
            "normalized_partition": normalized.as_posix(), "normalized_rows": metadata.num_rows,
            "normalized_columns": normalized_columns,
            "raw_event_fields_present": [name for name in event_fields if name in columns],
            "normalized_event_fields_present": [name for name in event_fields if name in normalized_columns],
            "raw_evidence_fields_missing": [name for name in required if name not in columns],
            "normalized_evidence_fields_missing": [name for name in required if name not in normalized_columns],
        })
    report = {
        "schema_version": 1, "status": "SCHEMA_AUDIT_MISSING_AVAILABILITY_EVIDENCE",
        "created_at_utc": datetime.now(UTC).isoformat(), "operating_year": 2024,
        "months_checked": len(records), "normalized_rows_in_metadata": sum(record["normalized_rows"] for record in records),
        "records": records, "flight_outcome_values_read": False,
        "archive_payload_sha256_reverified": False,
        "claim_limit": "Header/schema inspection only. Event-clock columns exist in raw archives; value completeness, date rollover, source publication, revisions and consumer receipt are not established.",
        "confirmation_outcomes_accessed": False,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(args.output, report)
    print(json.dumps({"status": report["status"], "months_checked": len(records), "normalized_rows": report["normalized_rows_in_metadata"]}))


if __name__ == "__main__":
    main()
