"""Validate boundary-complete probabilistic-operations-twin feature artifacts."""

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

from .flare_capacity_contracts import CAPACITY_ALL_FEATURES
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

BOUNDARY_METHOD = "FLARE-24-BOUNDARY-COMPLETE-PROBABILISTIC-OPERATIONS-TWIN"
COMPLETE_STATUS = "COMPLETE_COVARIATE_FEATURES_NO_TARGET_OUTCOMES_ACCESSED"
PARTIAL_STATUS = "PARTIAL_SMOKE_COVARIATE_FEATURES_NO_TARGET_OUTCOMES_ACCESSED"
FORBIDDEN_COLUMNS = frozenset(
    {
        "ActualElapsedTime",
        "AirTime",
        "ArrDelay",
        "ArrDel15",
        "Cancelled",
        "CancellationCode",
        "DepDelay",
        "DepDel15",
        "Diverted",
        "Tail_Number",
        "TaxiIn",
        "TaxiOut",
        "WheelsOff",
        "WheelsOn",
    }
)


def _self_hashed(path: Path) -> tuple[dict[str, Any], str]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    keys = [key for key in ("manifest_sha256", "validation_sha256") if key in payload]
    if len(keys) != 1:
        raise ValueError(f"invalid self-hash keys in {path}")
    key = keys[0]
    recorded = str(payload[key])
    body = {name: value for name, value in payload.items() if name != key}
    if canonical_json_sha256(body) != recorded:
        raise ValueError(f"self-hash validation failed: {path}")
    return payload, key


def _verified_file(record: dict[str, Any], *, role: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{role} checksum failed: {path}")
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{role} size failed: {path}")
    return path


def _numeric_audit(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path, columns=list(CAPACITY_ALL_FEATURES))
    numeric = frame.to_numpy(dtype=np.float64)
    if np.isinf(numeric).any():
        raise ValueError(f"boundary feature partition contains infinity: {path}")
    probability_columns = [
        column
        for column in CAPACITY_ALL_FEATURES
        if "probability" in column or column.endswith("_fraction")
    ]
    for column in probability_columns:
        values = pd.to_numeric(frame[column], errors="raise").dropna()
        if values.lt(-1e-7).any() or values.gt(1.0 + 1e-7).any():
            raise ValueError(f"bounded boundary feature leaves [0,1]: {path}: {column}")
    simplex_error = 0.0
    for side in ("origin", "dest"):
        columns = [
            f"ccrth_{side}_capacity_scenario_{scenario}_probability"
            for scenario in ("constrained", "marginal", "good")
        ]
        sums = frame.loc[:, columns].sum(axis=1).to_numpy(dtype=np.float64)
        error = float(np.max(np.abs(sums - 1.0), initial=0.0))
        simplex_error = max(simplex_error, error)
        if error > 2e-6:
            raise ValueError(f"capacity scenario simplex failed: {path}: {side}")
    return {
        "rows": len(frame),
        "cells": int(frame.shape[0] * frame.shape[1]),
        "missing_cells": int(frame.isna().sum().sum()),
        "maximum_scenario_simplex_error": simplex_error,
        "rotation_message_coverage_mean": float(
            frame["ccrth_rotation_resource_message_coverage"].mean()
        ),
        "route_shadow_price_p90": float(frame["ccrth_route_sum_shadow_price"].quantile(0.90)),
    }


def validate_boundary_feature_manifest(
    manifest_path: Path,
    *,
    target_census_dir: Path,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Validate provenance, information boundary, schema, IDs, and numerics."""

    manifest, hash_key = _self_hashed(manifest_path)
    expected_status = COMPLETE_STATUS if require_complete else None
    if expected_status is not None and manifest.get("status") != expected_status:
        raise ValueError("boundary operations-twin feature manifest is not complete")
    if not require_complete and manifest.get("status") not in {
        COMPLETE_STATUS,
        PARTIAL_STATUS,
    }:
        raise ValueError("boundary operations-twin manifest has an invalid status")
    if manifest.get("method") != BOUNDARY_METHOD:
        raise ValueError("manifest is not the boundary-complete operations twin")
    if manifest.get("artifact_mode") != "features-only":
        raise ValueError("boundary feature validator requires features-only artifacts")
    if tuple(manifest.get("features", ())) != CAPACITY_ALL_FEATURES:
        raise ValueError("boundary feature contract differs from the registry")
    if manifest.get("outcome_columns_read") != []:
        raise ValueError("boundary feature construction read outcomes")
    if manifest.get("target_tail_number_read") is not False:
        raise ValueError("boundary feature construction read target tails")
    if manifest.get("confirmation_outcomes_accessed") is not False:
        raise ValueError("boundary feature construction crossed the 2026 gate")
    if manifest.get("scored_population") != ("unchanged frozen top100-to-top100 sample ids"):
        raise ValueError("boundary feature construction changed the scored population")

    context_record = manifest.get("context_schedule_manifest")
    if not isinstance(context_record, dict):
        raise ValueError("boundary context manifest binding is missing")
    context_path = _verified_file(context_record, role="boundary context manifest")
    context, context_hash_key = _self_hashed(context_path)
    if context[context_hash_key] != context_record.get("self_hash"):
        raise ValueError("boundary context manifest self-hash binding differs")
    if (
        context.get("outcome_columns_read") != []
        or context.get("tail_number_read") is not False
        or context.get("confirmation_outcomes_accessed") is not False
    ):
        raise ValueError("bound boundary context violates the information boundary")
    _verified_file(manifest["context_airport_catalog"], role="context airport catalog")

    for record in manifest.get("schedule_inputs", []):
        overlap = sorted(FORBIDDEN_COLUMNS & set(record.get("columns_read", ())))
        if overlap:
            raise ValueError(f"schedule input includes forbidden columns: {overlap}")
        _verified_file(record, role="boundary operations-twin schedule input")
    for role in (
        "weather_feature_manifest",
        "resource_catalog_manifest",
        "rotation_manifest",
    ):
        _verified_file(manifest[role], role=role)

    periods: set[tuple[int, int]] = set()
    rows = 0
    numeric_cells = 0
    missing_cells = 0
    maximum_simplex_error = 0.0
    context_only_edges = 0
    context_only_states = 0
    period_records: list[dict[str, Any]] = []
    diagnostics_by_period = {
        (int(record["year"]), int(record["month"])): record
        for record in manifest.get("diagnostics", [])
    }
    for output in manifest.get("outputs", []):
        year, month = int(output["year"]), int(output["month"])
        period = (year, month)
        if period in periods:
            raise ValueError(f"duplicate boundary feature period: {period}")
        periods.add(period)
        if any(
            output.get(role) is not None
            for role in (
                "flight_nodes",
                "resource_nodes",
                "incidence_edges",
                "rotation_edges",
            )
        ):
            raise ValueError("features-only manifest unexpectedly binds graph tables")
        feature_path = _verified_file(output["features"], role=f"{period} features")
        schema = tuple(pq.read_schema(feature_path).names)
        if schema != ("sample_id", *CAPACITY_ALL_FEATURES):
            raise ValueError(f"boundary feature schema mismatch: {feature_path}")
        identifiers = pd.read_parquet(feature_path, columns=["sample_id"])
        if identifiers["sample_id"].isna().any() or identifiers["sample_id"].duplicated().any():
            raise ValueError(f"boundary features have invalid sample ids: {period}")
        target_path = target_census_dir / f"year={year}" / f"month={month:02d}.parquet"
        target_ids = pd.read_parquet(target_path, columns=["sample_id"])
        if set(identifiers["sample_id"].astype(str)) != set(target_ids["sample_id"].astype(str)):
            raise ValueError(f"boundary features changed target IDs: {period}")
        numeric = _numeric_audit(feature_path)
        rows += len(identifiers)
        numeric_cells += int(numeric["cells"])
        missing_cells += int(numeric["missing_cells"])
        maximum_simplex_error = max(
            maximum_simplex_error,
            float(numeric["maximum_scenario_simplex_error"]),
        )
        diagnostic = diagnostics_by_period.get(period)
        if diagnostic is None:
            raise ValueError(f"boundary diagnostics omit period: {period}")
        if (
            int(diagnostic.get("resource_airport_count", -1)) != 100
            or int(diagnostic.get("context_events_after_resource_filter", 0))
            < int(diagnostic.get("target_events", 0))
            or int(diagnostic.get("context_events_after_resource_filter", 0))
            > int(diagnostic.get("context_events_before_resource_filter", 0))
        ):
            raise ValueError(f"boundary resource filtering is incoherent: {period}")
        context_only_edges += int(diagnostic["rotation_context_only_edges"])
        context_only_states += int(diagnostic["rotation_context_only_predecessor_states"])
        period_records.append(
            {
                "year": year,
                "month": month,
                "rows": len(identifiers),
                **numeric,
                "rotation_context_only_edges": int(diagnostic["rotation_context_only_edges"]),
                "rotation_context_only_predecessor_states": int(
                    diagnostic["rotation_context_only_predecessor_states"]
                ),
            }
        )

    declared_periods = {
        (int(year), int(month))
        for year in manifest.get("target_years", ())
        for month in manifest.get("target_months", ())
    }
    if periods != declared_periods:
        raise ValueError("boundary feature outputs differ from declared periods")
    expected_periods = {(year, month) for year in (2024, 2025) for month in range(1, 13)}
    if require_complete and periods != expected_periods:
        raise ValueError("boundary feature outputs omit required periods")
    if rows != int(manifest.get("rows", -1)):
        raise ValueError("boundary feature aggregate row count differs")
    if context_only_edges <= 0 or context_only_states <= 0:
        raise ValueError("boundary context did not contribute any rotation state")
    for record in manifest.get("frontier_outputs", []):
        _verified_file(record, role="boundary prior-year frontier")
        if int(record["history_year"]) != int(record["target_year"]) - 1:
            raise ValueError("boundary frontier is not strictly prior-year")
    for record in manifest.get("provenance", {}).get("source_files", []):
        _verified_file(record, role="boundary source provenance")

    return {
        "status": (
            "PASS_BOUNDARY_OPERATIONS_TWIN_FEATURE_VALIDATION"
            if require_complete
            else "PASS_BOUNDARY_OPERATIONS_TWIN_PARTIAL_FEATURE_VALIDATION"
        ),
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "manifest": {
            "path": manifest_path.as_posix(),
            "sha256": sha256_file(manifest_path),
            "self_hash_key": hash_key,
            "self_hash": manifest[hash_key],
        },
        "periods": len(periods),
        "rows": rows,
        "numeric_cells": numeric_cells,
        "missing_feature_fraction": missing_cells / numeric_cells,
        "maximum_scenario_simplex_error": maximum_simplex_error,
        "context_only_rotation_edges": context_only_edges,
        "context_only_predecessor_states": context_only_states,
        "period_records": period_records,
        "target_id_set_match": True,
        "outcome_columns_read": [],
        "target_tail_number_read": False,
        "confirmation_outcomes_accessed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-census-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = validate_boundary_feature_manifest(
        args.manifest,
        target_census_dir=args.target_census_dir,
        require_complete=not args.allow_partial,
    )
    report["validator_provenance"] = capture_provenance(
        (
            Path(__file__),
            Path(__file__).with_name("flare_capacity_contracts.py"),
            Path(__file__).with_name("hashing.py"),
        )
    )
    report["validation_sha256"] = canonical_json_sha256(report)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite boundary validation: {args.output}")
    write_canonical_json(args.output, report)
    print(json.dumps({"status": report["status"], "rows": report["rows"]}, indent=2))


if __name__ == "__main__":
    main()
