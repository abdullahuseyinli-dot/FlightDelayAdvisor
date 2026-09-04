"""Validate fixed-lead forecast evidence and independently reconstruct sampled tables."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .forecast_features import DAILY_FORECAST_COLUMNS, _aggregate_airport_year
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json


def _self_hash(payload: dict[str, Any], key: str) -> bool:
    recorded = payload.get(key)
    body = {name: value for name, value in payload.items() if name != key}
    return recorded == canonical_json_sha256(body)


def validate_forecast_features(
    *,
    acquisition_manifest_path: Path,
    feature_manifest_path: Path,
    output_path: Path,
    sampled_inputs: int = 12,
    minimum_mean_coverage: float = 0.94,
    seed: int = 20260903,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite forecast validation: {output_path}")
    if sampled_inputs < 2:
        raise ValueError("at least two raw inputs must be independently reconstructed")
    acquisition: dict[str, Any] = json.loads(
        acquisition_manifest_path.read_text(encoding="utf-8")
    )
    features: dict[str, Any] = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    acquisition_hash_ok = _self_hash(acquisition, "manifest_sha256")
    feature_hash_ok = _self_hash(features, "manifest_sha256")
    checks.extend(
        [
            {"check": "acquisition_manifest_self_hash", "passed": acquisition_hash_ok},
            {"check": "feature_manifest_self_hash", "passed": feature_hash_ok},
        ]
    )
    if not acquisition_hash_ok:
        errors.append("acquisition manifest self-hash failed")
    if not feature_hash_ok:
        errors.append("feature manifest self-hash failed")
    if sha256_file(acquisition_manifest_path) != features.get("acquisition_manifest_sha256"):
        errors.append("feature manifest does not bind the acquisition manifest checksum")

    raw_hash_failures: list[str] = []
    for record in acquisition.get("requests", []):
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raw_hash_failures.append(path.as_posix())
    checks.append(
        {
            "check": "all_raw_response_hashes",
            "passed": not raw_hash_failures,
            "failures": raw_hash_failures,
        }
    )
    if raw_hash_failures:
        errors.append(f"{len(raw_hash_failures)} raw response checksums failed")

    output_tables: dict[int, pd.DataFrame] = {}
    output_hash_failures: list[str] = []
    for record in features.get("outputs", []):
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            output_hash_failures.append(path.as_posix())
            continue
        frame = pd.read_parquet(path)
        year = int(record["year"])
        output_tables[year] = frame
        expected_rows = int(acquisition["airport_count"]) * (366 if year % 4 == 0 else 365)
        if len(frame) != expected_rows:
            errors.append(f"daily forecast row count failed for {year}")
        if frame.duplicated(["Airport", "FlightDate"]).any():
            errors.append(f"daily forecast key uniqueness failed for {year}")
        if not frame["FlightDate"].dt.year.eq(year).all():
            errors.append(f"daily forecast year domain failed for {year}")
        coverage = float(frame["forecast24_min_variable_coverage"].mean())
        if coverage < minimum_mean_coverage:
            errors.append(
                f"mean minimum-variable coverage {coverage:.6f} below gate "
                f"{minimum_mean_coverage:.6f} for {year}"
            )
    checks.append(
        {
            "check": "output_hashes_and_domains",
            "passed": not output_hash_failures,
            "hash_failures": output_hash_failures,
        }
    )

    records = list(acquisition.get("requests", []))
    generator = np.random.default_rng(seed)
    count = min(sampled_inputs, len(records))
    indices: NDArray[np.int64] = (
        generator.choice(len(records), size=count, replace=False).astype(np.int64)
        if count
        else np.array([], dtype=np.int64)
    )
    reconstruction_records: list[dict[str, Any]] = []
    for index in indices:
        record = records[int(index)]
        airport = str(record["airport"])
        year = int(record["year"])
        reconstructed = _aggregate_airport_year(Path(record["path"]), airport, year)
        stored = output_tables[year]
        stored_airport = stored.loc[stored["Airport"].eq(airport)].sort_values("FlightDate")
        reconstructed = reconstructed.sort_values("FlightDate")
        keys_equal = np.array_equal(
            stored_airport[["Airport", "FlightDate"]].to_numpy(),
            reconstructed[["Airport", "FlightDate"]].to_numpy(),
        )
        values_equal = True
        for column in DAILY_FORECAST_COLUMNS:
            left = pd.to_numeric(stored_airport[column], errors="coerce").to_numpy(dtype=float)
            right = pd.to_numeric(reconstructed[column], errors="coerce").to_numpy(dtype=float)
            if not np.allclose(left, right, equal_nan=True, atol=1e-6, rtol=1e-6):
                values_equal = False
                break
        passed = bool(keys_equal and values_equal)
        reconstruction_records.append(
            {"airport": airport, "year": year, "passed": passed}
        )
        if not passed:
            errors.append(f"independent reconstruction failed for {airport} {year}")
    checks.append(
        {
            "check": "independent_raw_to_daily_reconstruction",
            "passed": all(record["passed"] for record in reconstruction_records),
            "records": reconstruction_records,
        }
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "acquisition_manifest": acquisition_manifest_path.as_posix(),
        "acquisition_manifest_sha256": sha256_file(acquisition_manifest_path),
        "feature_manifest": feature_manifest_path.as_posix(),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "minimum_mean_coverage": minimum_mean_coverage,
        "sampled_inputs": count,
        "seed": seed,
        "checks": checks,
        "errors": errors,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampled-inputs", type=int, default=12)
    parser.add_argument("--minimum-mean-coverage", type=float, default=0.94)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = validate_forecast_features(
        acquisition_manifest_path=args.acquisition_manifest,
        feature_manifest_path=args.feature_manifest,
        output_path=args.output,
        sampled_inputs=args.sampled_inputs,
        minimum_mean_coverage=args.minimum_mean_coverage,
        seed=args.seed,
    )
    print(json.dumps({"output": args.output.as_posix(), "status": report["status"]}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
