"""Independently validate point-in-time feature evidence and its manifest."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .hashing import canonical_json_sha256, sha256_file, write_canonical_json


def _verify_self_hash(payload: dict[str, Any]) -> None:
    recorded = payload.get("manifest_sha256")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError("feature manifest self-hash is invalid")


def _resolve(repo_root: Path, recorded_path: str) -> Path:
    path = Path(recorded_path)
    return path if path.is_absolute() else repo_root / path


def _verify_recorded_file(repo_root: Path, record: dict[str, Any]) -> Path:
    path = _resolve(repo_root, str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"manifest file is missing: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"byte-size mismatch: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return path


def _batch_min_max(array: pa.Array) -> tuple[float, float] | None:
    values = pc.min_max(array)
    if not values.is_valid:
        return None
    mapping = values.as_py()
    if mapping["min"] is None or mapping["max"] is None:
        return None
    return float(mapping["min"]), float(mapping["max"])


def _validate_partition(
    path: Path,
    record: dict[str, Any],
    deployable: list[str],
) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != int(record["rows"]):
        raise ValueError(f"row-count mismatch: {path}")
    required = {
        *deployable,
        "sample_id",
        "FlightDate",
        "ArrDel15",
        "Cancelled",
        "delay_label_observed",
        "joint_label_observed",
        "disruption_state",
    }
    missing_columns = sorted(required - set(parquet.schema_arrow.names))
    if missing_columns:
        raise ValueError(f"{path} lacks required columns: {missing_columns}")

    ids = parquet.read(columns=["sample_id"])["sample_id"].to_pandas()
    if ids.duplicated().any():
        raise ValueError(f"duplicate sample_id within {path}")

    rates = [column for column in deployable if column.endswith("_rate")]
    counts = [
        column for column in deployable if column.endswith("_count") or column.endswith("_support")
    ]
    observed_rows = 0
    cancellations = 0
    delays = 0
    missing_delay = 0
    joint_ineligible = 0
    columns = [
        *deployable,
        "ArrDel15",
        "Cancelled",
        "joint_label_observed",
        "disruption_state",
    ]
    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        observed_rows += batch.num_rows
        for column in deployable:
            array = batch[column]
            if array.null_count:
                raise ValueError(f"deployable feature {column} has nulls in {path}")
        for column in rates:
            bounds = _batch_min_max(batch[column])
            if bounds is not None and (bounds[0] < -1e-7 or bounds[1] > 1.0000001):
                raise ValueError(f"probability feature {column} is out of range in {path}")
        for column in counts:
            bounds = _batch_min_max(batch[column])
            if bounds is not None and bounds[0] < -1e-7:
                raise ValueError(f"support feature {column} is negative in {path}")

        cancel = batch["Cancelled"].to_numpy(zero_copy_only=False)
        delay = batch["ArrDel15"].to_numpy(zero_copy_only=False)
        joint = batch["joint_label_observed"].to_numpy(zero_copy_only=False)
        state = batch["disruption_state"].to_numpy(zero_copy_only=False)
        cancellations += int(np.sum(cancel == 1))
        delays += int(np.nansum(delay))
        missing_delay += int(np.sum((cancel == 0) & np.isnan(delay)))
        joint_ineligible += int(np.sum(joint == 0))
        expected_state = np.select([cancel == 1, delay == 1, delay == 0], [2, 1, 0], default=-1)
        if not np.array_equal(state, expected_state):
            raise ValueError(f"joint outcome encoding is inconsistent in {path}")

    observed = {
        "rows": observed_rows,
        "delays": delays,
        "cancellations": cancellations,
        "noncancelled_missing_delay_label": missing_delay,
        "joint_label_ineligible": joint_ineligible,
    }
    for key, value in observed.items():
        if value != int(record[key]):
            raise ValueError(f"{key} mismatch for {path}: {value} != {record[key]}")
    return observed


def _legacy_global_year(path: Path, year: int) -> dict[str, float]:
    frame = pq.read_table(
        path,
        columns=["ArrDel15", "Cancelled"],
        filters=[("Year", "=", year)],
    ).to_pandas()
    cancel = pd.to_numeric(frame["Cancelled"], errors="raise")
    delay = pd.to_numeric(frame["ArrDel15"], errors="coerce")
    eligible = cancel.eq(0) & delay.notna()
    return {
        "scheduled_count": float(len(frame)),
        "delay_support": float(eligible.sum()),
        "delay_sum": float(delay.where(eligible, 0.0).sum()),
        "cancel_sum": float(cancel.sum()),
    }


def _first_global_priors(path: Path) -> dict[str, float]:
    columns = [
        "prior_global_delay_rate",
        "prior_global_cancel_rate",
        "prior_global_count",
    ]
    table = pq.ParquetFile(path).read_row_group(0, columns=columns).slice(0, 1)
    return {column: float(table[column][0].as_py()) for column in columns}


def _validate_global_prior_chronology(
    *,
    repo_root: Path,
    manifest: dict[str, Any],
    output_paths: list[tuple[dict[str, Any], Path]],
) -> list[dict[str, float | int]]:
    legacy_records = [record for record in manifest["sources"] if "years" in record]
    if len(legacy_records) != 1:
        raise ValueError("manifest must identify exactly one historical source")
    legacy_path = _resolve(repo_root, str(legacy_records[0]["path"]))
    half_life_raw = manifest["half_life_years"]
    half_life = None if half_life_raw is None else float(half_life_raw)
    state = _legacy_global_year(legacy_path, 2010)
    previous_year = 2010
    evidence: list[dict[str, float | int]] = []

    paths_by_year: dict[int, list[Path]] = {}
    for record, path in output_paths:
        paths_by_year.setdefault(int(record["year"]), []).append(path)
    for year in sorted(paths_by_year):
        elapsed = year - previous_year
        if elapsed <= 0:
            raise ValueError("output years are not chronological")
        if half_life is not None:
            factor = math.exp(-math.log(2.0) * elapsed / half_life)
            state = {key: value * factor for key, value in state.items()}
        expected = {
            "prior_global_delay_rate": state["delay_sum"] / state["delay_support"],
            "prior_global_cancel_rate": state["cancel_sum"] / state["scheduled_count"],
            "prior_global_count": state["scheduled_count"],
        }
        for path in paths_by_year[year]:
            actual = _first_global_priors(path)
            for name, expected_value in expected.items():
                if not np.isclose(actual[name], expected_value, rtol=2e-6, atol=1e-5):
                    raise ValueError(
                        f"strict prior mismatch for {name}, year {year}: "
                        f"{actual[name]} != {expected_value}"
                    )
        evidence.append({"year": year, **expected})
        if year <= 2024:
            addition = _legacy_global_year(legacy_path, year)
            state = {key: state[key] + addition[key] for key in state}
        previous_year = year
    return evidence


def validate_feature_dataset(
    manifest_path: Path,
    *,
    report_path: Path,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Validate checksums, schema, labels, and strict temporal global priors."""

    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite validation evidence: {report_path}")
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_self_hash(payload)
    deployable = [str(column) for column in payload["deployable_features"]]

    source_paths = [_verify_recorded_file(repo_root, record) for record in payload["sources"]]
    output_paths: list[tuple[dict[str, Any], Path]] = []
    partition_evidence: list[dict[str, Any]] = []
    for record in payload["outputs"]:
        path = _verify_recorded_file(repo_root, record)
        output_paths.append((record, path))
        partition_evidence.append(
            {
                "path": record["path"],
                **_validate_partition(path, record, deployable),
            }
        )
    state_path = _verify_recorded_file(repo_root, payload["state"])
    chronology = _validate_global_prior_chronology(
        repo_root=repo_root,
        manifest=payload,
        output_paths=output_paths,
    )
    total_rows = sum(int(item["rows"]) for item in partition_evidence)
    if total_rows != int(payload["total_output_rows"]):
        raise ValueError("manifest total_output_rows does not equal partition total")

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "feature_variant": payload["feature_variant"],
        "builder_version": payload["builder_version"],
        "checks": [
            "manifest self-hash",
            "all recorded source, output, and state SHA-256 hashes and byte sizes",
            "partition row totals and required schema",
            "no missing deployable feature values",
            "probability and support feature domains",
            "unique sample identifiers within each immutable partition",
            "delay, cancellation, missing-label, and joint-state reconciliation",
            "strict earlier-year global prior reconstruction from raw outcomes",
        ],
        "source_files_verified": len(source_paths),
        "state_path": state_path.relative_to(repo_root).as_posix(),
        "partitions_verified": len(partition_evidence),
        "rows_verified": total_rows,
        "partition_evidence": partition_evidence,
        "prior_chronology": chronology,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = validate_feature_dataset(
        args.manifest,
        report_path=args.report,
        repo_root=args.repo_root.resolve(),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "rows": report["rows_verified"],
                "partitions": report["partitions_verified"],
                "report": args.report.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
