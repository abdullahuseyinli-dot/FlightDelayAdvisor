#!/usr/bin/env python3
"""Describe the immutable historical cohort without loading it fully into memory."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from flightdelaybench.hashing import canonical_json_sha256, sha256_file, write_canonical_json


def build_manifest(dataset_path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(dataset_path)
    airports: set[str] = set()
    year_counts: Counter[int] = Counter()
    for batch in parquet.iter_batches(batch_size=500_000, columns=["Year", "Origin", "Dest"]):
        frame = batch.to_pandas()
        airports.update(frame["Origin"].dropna().astype(str).unique())
        airports.update(frame["Dest"].dropna().astype(str).unique())
        year_counts.update(
            {int(year): int(count) for year, count in frame["Year"].value_counts().items()}
        )
    body: dict[str, object] = {
        "schema_version": 1,
        "cohort_id": "legacy-top100-weather-v1",
        "source_path": dataset_path.as_posix(),
        "source_sha256": sha256_file(dataset_path),
        "rows": parquet.metadata.num_rows,
        "columns": parquet.schema.names,
        "year_counts": {str(year): year_counts[year] for year in sorted(year_counts)},
        "airports": sorted(airports),
        "airport_count": len(airports),
        "sampling_note": (
            "Historical rows were sampled approximately uniformly by month within year, "
            "then restricted to flights whose origin and destination were in the selected "
            "100-airport weather cohort. The sample is not a census of US flights."
        ),
        "evidence_status": "LEGACY_INPUT_DESCRIBED_NOT_REGENERATED",
    }
    body["manifest_sha256"] = canonical_json_sha256(body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/bts_delay_2010_2024_balanced_research_weather.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manifests/cohort_airports_v1.json"),
    )
    args = parser.parse_args()
    manifest = build_manifest(args.dataset)
    write_canonical_json(args.output, manifest)
    print(f"wrote {args.output} ({manifest['rows']} rows, {manifest['airport_count']} airports)")


if __name__ == "__main__":
    main()

