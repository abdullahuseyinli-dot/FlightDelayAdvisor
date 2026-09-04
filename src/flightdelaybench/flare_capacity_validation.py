"""Validate CC-RTH graph, feature, experiment, and evidence artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .flare_capacity_contracts import CAPACITY_ALL_FEATURES
from .flare_capacity_modeling import CAPACITY_CANDIDATE_FEATURES
from .flare_capacity_study import (
    AUGMENTED_CANDIDATES,
    CAPACITY_CANDIDATES,
    FINAL_CALIBRATION_END,
    PERSISTED_FLOAT32_SIMPLEX_ACCEPTANCE,
    PRIMARY_AUDIT_END,
    PRIMARY_AUDIT_START,
    PROBABILITY_STATES,
    PURGE_DAYS,
    RECOVERY_ARTIFACT_ORIGIN,
    _load_recovery_model_record,
)
from .flare_study import TASKS
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance


def _self_hashed(path: Path) -> tuple[dict[str, Any], str]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    keys = [key for key in ("manifest_sha256", "report_sha256") if key in payload]
    if len(keys) != 1:
        raise ValueError(f"invalid self-hash keys in {path}")
    key = keys[0]
    recorded = str(payload[key])
    if canonical_json_sha256({k: v for k, v in payload.items() if k != key}) != recorded:
        raise ValueError(f"self-hash validation failed: {path}")
    return payload, key


def _verify_file(record: dict[str, Any], *, role: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{role} checksum failed: {path}")
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{role} size failed: {path}")
    return path


def _numeric_audit(path: Path, columns: tuple[str, ...]) -> dict[str, Any]:
    frame = pd.read_parquet(path, columns=list(columns))
    numeric = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if np.isinf(numeric).any():
        raise ValueError(f"feature partition contains infinity: {path}")
    probability_columns = tuple(
        column
        for column in columns
        if "probability" in column or column.endswith("_fraction")
    )
    for column in probability_columns:
        values = pd.to_numeric(frame[column], errors="raise").dropna()
        if values.lt(-1e-7).any() or values.gt(1.0 + 1e-7).any():
            raise ValueError(f"bounded feature leaves [0,1]: {path}: {column}")
    nonnegative_tokens = ("demand_", "queue", "recovery_minutes", "shadow_price")
    for column in columns:
        if any(token in column for token in nonnegative_tokens):
            values = pd.to_numeric(frame[column], errors="raise").dropna()
            if values.lt(-1e-7).any():
                raise ValueError(f"non-negative feature is negative: {path}: {column}")
    diagnostic_columns = (
        "ccrth_origin_expected_queue",
        "ccrth_origin_queue_p90",
        "ccrth_origin_overload_probability",
        "ccrth_origin_direction_utilization",
        "ccrth_route_sum_shadow_price",
        "ccrth_rotation_resource_message_coverage",
    )
    diagnostics: dict[str, Any] = {}
    for column in diagnostic_columns:
        values = pd.to_numeric(frame[column], errors="raise").dropna()
        diagnostics[column] = {
            "nonmissing_fraction": float(frame[column].notna().mean()),
            "p50": float(values.quantile(0.50)),
            "p90": float(values.quantile(0.90)),
            "p99": float(values.quantile(0.99)),
            "maximum": float(values.max()),
        }
    return {
        "rows": len(frame),
        "columns": len(columns),
        "cells": int(frame.shape[0] * frame.shape[1]),
        "missing_cells": int(frame.isna().sum().sum()),
        "diagnostic_quantiles": diagnostics,
    }


def validate_capacity_graph_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest, hash_key = _self_hashed(manifest_path)
    if manifest.get("status") != "COMPLETE_COVARIATE_GRAPH_NO_TARGET_OUTCOMES_ACCESSED":
        raise ValueError("CC-RTH graph manifest is not complete")
    if tuple(manifest.get("features", ())) != CAPACITY_ALL_FEATURES:
        raise ValueError("CC-RTH graph manifest differs from the registered feature contract")
    if set(manifest.get("target_years", ())) != {2024, 2025} or set(
        manifest.get("target_months", ())
    ) != set(range(1, 13)):
        raise ValueError("CC-RTH graph manifest does not cover all required periods")
    if manifest.get("outcome_columns_read") != []:
        raise ValueError("CC-RTH graph construction read outcome columns")
    if manifest.get("target_tail_number_read") is not False:
        raise ValueError("CC-RTH graph construction read target-year tail identity")
    if manifest.get("confirmation_outcomes_accessed") is not False:
        raise ValueError("CC-RTH graph construction crossed the confirmation gate")
    forbidden = {
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
    for record in manifest.get("schedule_inputs", []):
        overlap = sorted(forbidden & set(record.get("columns_read", [])))
        if overlap:
            raise ValueError(f"schedule-only CC-RTH input includes forbidden columns: {overlap}")
        _verify_file(record, role="CC-RTH schedule input")
    for role in ("weather_feature_manifest", "resource_catalog_manifest", "rotation_manifest"):
        _verify_file(manifest[role], role=role)
    if manifest.get("operational_constraints") is not None:
        _verify_file(manifest["operational_constraints"], role="operational constraints")
    periods: set[tuple[int, int]] = set()
    totals = {"rows": 0, "resource_nodes": 0, "incidence_edges": 0, "rotation_edges": 0}
    numeric_totals = {"cells": 0, "missing_cells": 0}
    period_records: list[dict[str, Any]] = []
    for output in manifest.get("outputs", []):
        year = int(output["year"])
        month = int(output["month"])
        period = (year, month)
        if period in periods:
            raise ValueError(f"duplicate CC-RTH period: {period}")
        periods.add(period)
        paths = {
            role: _verify_file(dict(output[role]), role=f"{period} {role}")
            for role in (
                "features",
                "flight_nodes",
                "resource_nodes",
                "incidence_edges",
                "rotation_edges",
            )
        }
        schema = pq.read_schema(paths["features"]).names
        if tuple(schema) != ("sample_id", *CAPACITY_ALL_FEATURES):
            raise ValueError(f"CC-RTH feature schema mismatch: {paths['features']}")
        identifiers = pd.read_parquet(paths["features"], columns=["sample_id"])
        if identifiers["sample_id"].isna().any() or identifiers["sample_id"].duplicated().any():
            raise ValueError(f"CC-RTH features have invalid sample ids: {period}")
        flight_nodes = pd.read_parquet(paths["flight_nodes"])
        if flight_nodes["sample_id"].isna().any() or flight_nodes["sample_id"].duplicated().any():
            raise ValueError(f"CC-RTH flight nodes have invalid ids: {period}")
        if set(flight_nodes["sample_id"].astype(str)) != set(
            identifiers["sample_id"].astype(str)
        ):
            raise ValueError(f"CC-RTH flight and feature nodes differ: {period}")
        resources = pd.read_parquet(paths["resource_nodes"], columns=["resource_node_id"])
        if resources["resource_node_id"].isna().any() or resources[
            "resource_node_id"
        ].duplicated().any():
            raise ValueError(f"CC-RTH resource nodes have invalid ids: {period}")
        incidence = pd.read_parquet(
            paths["incidence_edges"], columns=["sample_id", "resource_node_id"]
        )
        if not set(incidence["sample_id"].astype(str)).issubset(
            set(identifiers["sample_id"].astype(str))
        ) or not set(incidence["resource_node_id"].astype(str)).issubset(
            set(resources["resource_node_id"].astype(str))
        ):
            raise ValueError(f"CC-RTH incidence has dangling endpoints: {period}")
        minimum_incidence = incidence.groupby("sample_id", observed=True).size().min()
        if pd.isna(minimum_incidence) or int(minimum_incidence) < 2:
            raise ValueError(f"CC-RTH flight lacks origin/destination incidence: {period}")
        rotation = pd.read_parquet(
            paths["rotation_edges"],
            columns=["predecessor_sample_id", "successor_sample_id", "probability"],
        )
        target_ids = set(identifiers["sample_id"].astype(str))
        if not set(rotation["successor_sample_id"].astype(str)).issubset(target_ids):
            raise ValueError(f"CC-RTH rotation successor escapes target month: {period}")
        probability = pd.to_numeric(rotation["probability"], errors="raise")
        if probability.lt(-1e-10).any() or probability.gt(1.0 + 1e-8).any():
            raise ValueError(f"CC-RTH rotation probability is invalid: {period}")
        incoming = rotation.groupby("successor_sample_id", observed=True)["probability"].sum()
        if incoming.gt(1.0 + 1e-6).any():
            raise ValueError(f"CC-RTH rotation incoming probability exceeds one: {period}")
        numeric = _numeric_audit(paths["features"], CAPACITY_ALL_FEATURES)
        totals["rows"] += len(identifiers)
        totals["resource_nodes"] += len(resources)
        totals["incidence_edges"] += len(incidence)
        totals["rotation_edges"] += len(rotation)
        numeric_totals["cells"] += numeric["cells"]
        numeric_totals["missing_cells"] += numeric["missing_cells"]
        period_records.append(
            {
                "year": year,
                "month": month,
                "rows": len(identifiers),
                "resource_nodes": len(resources),
                "incidence_edges": len(incidence),
                "rotation_edges": len(rotation),
                "minimum_incidence_per_flight": int(minimum_incidence),
                "maximum_incoming_rotation_probability": (
                    0.0 if incoming.empty else float(incoming.max())
                ),
                "missing_feature_fraction": numeric["missing_cells"] / numeric["cells"],
                "diagnostic_quantiles": numeric["diagnostic_quantiles"],
            }
        )
    expected_periods = {(year, month) for year in (2024, 2025) for month in range(1, 13)}
    if periods != expected_periods:
        raise ValueError("CC-RTH graph outputs omit required periods")
    for key, measured in totals.items():
        manifest_key = "rows" if key == "rows" else key
        if int(manifest[manifest_key]) != measured:
            raise ValueError(f"CC-RTH aggregate count mismatch: {key}")
    for record in manifest.get("frontier_outputs", []):
        _verify_file(record, role="CC-RTH prior-year frontier")
        if int(record["history_year"]) != int(record["target_year"]) - 1:
            raise ValueError("CC-RTH frontier is not strictly prior-year")
    for record in manifest.get("provenance", {}).get("source_files", []):
        _verify_file(record, role="CC-RTH source provenance")
    feature_coverage = manifest.get("feature_nonmissing_fraction", {})
    if set(feature_coverage) != set(CAPACITY_ALL_FEATURES) or any(
        not np.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        for value in feature_coverage.values()
    ):
        raise ValueError("CC-RTH aggregate feature coverage is invalid")
    zero_coverage_features = [
        feature for feature, value in feature_coverage.items() if float(value) == 0.0
    ]
    incomplete_coverage_features = [
        feature for feature, value in feature_coverage.items() if float(value) < 1.0
    ]
    saturated_overload_medians = [
        f"{record['year']}-{int(record['month']):02d}"
        for record in period_records
        if float(
            record.get("diagnostic_quantiles", {})
            .get("ccrth_origin_overload_probability", {})
            .get("p50", -1.0)
        )
        >= 1.0
    ]
    advisories: list[str] = []
    if zero_coverage_features:
        advisories.append(
            "Zero-coverage registered features remain explicit and are excluded by the "
            "training-covariate usability rule."
        )
    if saturated_overload_medians:
        advisories.append(
            "The origin overload-probability proxy has a median of one in the listed "
            "partitions; interpret it with continuous utilization, queue, slack, and "
            "shadow-price features, and retain the normalized-capacity ablation."
        )
    return {
        "status": "PASS_CCRTH_GRAPH_VALIDATION",
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "manifest": {
            "path": manifest_path.as_posix(),
            "sha256": sha256_file(manifest_path),
            "self_hash_key": hash_key,
            "self_hash": manifest[hash_key],
        },
        "totals": totals,
        "numeric_cells": numeric_totals,
        "periods": period_records,
        "feature_coverage": {
            "zero_coverage_features": zero_coverage_features,
            "incomplete_coverage_features": incomplete_coverage_features,
        },
        "proxy_diagnostics": {
            "origin_overload_probability_p50_at_one_periods": saturated_overload_medians,
        },
        "advisories": advisories,
        "information_boundary": {
            "outcome_columns_read": [],
            "target_tail_number_read": False,
            "confirmation_outcomes_accessed": False,
            "primary_complete_context_period": [
                PRIMARY_AUDIT_START,
                PRIMARY_AUDIT_END,
            ],
            "descriptive_only_boundary_dates": [
                "2025-01-01",
                "2025-01-02",
                "2025-12-30",
                "2025-12-31",
            ],
            "right_boundary_reason": (
                "December 30-31 require unopened 2026 schedules for the declared "
                "+2-day graph context and are excluded from primary inference."
            ),
        },
    }


def _validate_prediction_probabilities(path: Path) -> dict[str, Any]:
    names = pq.read_schema(path).names
    methods = sorted(
        {
            column[len("prob_") : -len("_on_time")]
            for column in names
            if column.startswith("prob_") and column.endswith("_on_time")
        }
    )
    if not methods:
        raise ValueError(f"prediction artifact contains no joint probabilities: {path}")
    columns = [
        f"prob_{method}_{state}" for method in methods for state in PROBABILITY_STATES
    ]
    frame = pd.read_parquet(path, columns=columns)
    for method in methods:
        probability = frame.loc[
            :, [f"prob_{method}_{state}" for state in PROBABILITY_STATES]
        ].to_numpy(dtype=np.float64)
        if not np.isfinite(probability).all() or (probability < 0.0).any():
            raise ValueError(f"prediction probability is invalid: {path}: {method}")
        if not np.allclose(probability.sum(axis=1), 1.0, atol=2e-6):
            raise ValueError(f"prediction simplex failed: {path}: {method}")
    return {"rows": len(frame), "methods": methods}


def _validate_model_recovery(
    report: dict[str, Any],
    lock: dict[str, Any],
) -> dict[str, Any] | None:
    """Revalidate any cross-run model recovery and its provenance symmetry."""

    recovery = report.get("model_recovery")
    if lock.get("model_recovery") != recovery:
        raise ValueError("CC-RTH model-recovery provenance differs across report and lock")
    model_records = report.get("model_artifacts", [])
    if recovery is None:
        if any(
            record.get("artifact_origin") == RECOVERY_ARTIFACT_ORIGIN
            for record in model_records
        ):
            raise ValueError("CC-RTH recovered model lacks a bound recovery record")
        return None

    recovery_path = _verify_file(recovery, role="CC-RTH recovery record")
    _, verified = _load_recovery_model_record(
        recovery_path,
        protocol_sha256=str(report.get("protocol", {}).get("sha256", "")),
        capacity_manifest_self_hash=str(
            report.get("capacity_manifest", {}).get("self_hash", "")
        ),
    )
    if verified != recovery:
        raise ValueError("CC-RTH recovery record summary differs from fresh verification")
    expected_keys = {
        (candidate, task)
        for candidate in AUGMENTED_CANDIDATES
        for task in TASKS
    }
    observed_keys = {
        (str(record.get("candidate", "")), str(record.get("task", "")))
        for record in model_records
    }
    if observed_keys != expected_keys or any(
        record.get("artifact_origin") != RECOVERY_ARTIFACT_ORIGIN
        for record in model_records
    ):
        raise ValueError("CC-RTH recovery does not account for every final model")
    return verified


def _validate_nested_feature_sets(value: Any) -> dict[str, list[str]]:
    """Validate candidate membership and nesting without relying on JSON key order."""

    if not isinstance(value, dict) or set(value) != set(CAPACITY_CANDIDATES):
        raise ValueError("CC-RTH study candidate set is invalid")
    feature_sets: dict[str, list[str]] = {}
    for candidate in CAPACITY_CANDIDATES:
        features = value[candidate]
        if not isinstance(features, list) or any(
            not isinstance(feature, str) for feature in features
        ):
            raise ValueError(f"CC-RTH feature set is invalid: {candidate}")
        if len(features) != len(set(features)):
            raise ValueError(f"CC-RTH feature set contains duplicates: {candidate}")
        if not set(features).issubset(CAPACITY_CANDIDATE_FEATURES[candidate]):
            raise ValueError(f"CC-RTH feature set exceeds its candidate: {candidate}")
        feature_sets[candidate] = features
    for left, right in pairwise(CAPACITY_CANDIDATES):
        if not set(feature_sets[left]).issubset(feature_sets[right]):
            raise ValueError("CC-RTH study candidates are not nested")
    return feature_sets


def _validate_probability_normalization_audit(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Verify bounded persisted-probability normalization and provenance symmetry."""

    audit = report.get("probability_normalization_audit")
    if not isinstance(audit, dict):
        raise ValueError("CC-RTH probability-normalization audit is missing")
    selection = audit.get("selection_reference")
    references = audit.get("retrospective_reference_partitions")
    assembled = audit.get("retrospective_assembled_methods")
    if (
        selection
        != report.get("baseline_selection", {}).get("probability_normalization_audit")
        or references
        != report.get("baseline_audit", {}).get("probability_normalization_audits")
    ):
        raise ValueError("CC-RTH probability-normalization provenance differs")
    if (
        not isinstance(selection, dict)
        or not isinstance(references, list)
        or not isinstance(assembled, list)
    ):
        raise ValueError("CC-RTH probability-normalization audit has invalid structure")
    required_methods = {
        *CAPACITY_CANDIDATES,
        "global_simplex",
        "capacity_gated_simplex",
    }
    assembled_methods = {
        str(record.get("method", ""))
        for record in assembled
        if isinstance(record, dict)
    }
    if len(references) != 12 or assembled_methods != required_methods:
        raise ValueError("CC-RTH probability-normalization audit has incomplete coverage")
    records = [selection, *references, *assembled]
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("CC-RTH probability-normalization record is invalid")
        maximum_error = float(
            record.get("maximum_absolute_row_sum_error_before_normalization", np.inf)
        )
        maximum_adjustment = float(
            record.get("maximum_absolute_probability_adjustment", np.inf)
        )
        if (
            int(record.get("rows", 0)) <= 0
            or float(record.get("acceptance_bound", np.nan))
            != PERSISTED_FLOAT32_SIMPLEX_ACCEPTANCE
            or not np.isfinite(maximum_error)
            or not 0.0 <= maximum_error <= PERSISTED_FLOAT32_SIMPLEX_ACCEPTANCE
            or not np.isfinite(maximum_adjustment)
            or not 0.0 <= maximum_adjustment <= PERSISTED_FLOAT32_SIMPLEX_ACCEPTANCE
            or int(record.get("rows_over_acceptance_bound", -1)) != 0
            or record.get("normalization")
            != "divide each accepted three-state row by its float64 row sum"
        ):
            raise ValueError("CC-RTH persisted-probability normalization exceeds its bound")
    reference_rows = sum(int(record["rows"]) for record in references)
    assembled_rows = {int(record["rows"]) for record in assembled}
    if assembled_rows != {reference_rows}:
        raise ValueError("CC-RTH probability-normalization row counts differ")
    return {
        "acceptance_bound": PERSISTED_FLOAT32_SIMPLEX_ACCEPTANCE,
        "selection_reference_rows": int(selection["rows"]),
        "retrospective_reference_partitions": len(references),
        "retrospective_rows": reference_rows,
        "assembled_methods": len(assembled),
        "maximum_absolute_row_sum_error_before_normalization": max(
            float(record["maximum_absolute_row_sum_error_before_normalization"])
            for record in records
        ),
        "maximum_absolute_probability_adjustment": max(
            float(record["maximum_absolute_probability_adjustment"])
            for record in records
        ),
    }


def validate_capacity_study_report(report_path: Path) -> dict[str, Any]:
    report, hash_key = _self_hashed(report_path)
    if report.get("status") != "COMPLETE_CCRTH_2024_SELECTION_2025_RETROSPECTIVE_EVALUATION":
        raise ValueError("CC-RTH study report is not complete")
    if report.get("outcomes_accessed", {}).get("2026_accessed") is not False or report.get(
        "confirmation_gate", {}
    ).get("opened") is not False:
        raise ValueError("CC-RTH study crossed the confirmation boundary")
    feature_sets = _validate_nested_feature_sets(report.get("feature_sets", {}))
    feature_coverage = report.get("capacity_feature_nonmissing_fraction", {})
    if set(feature_coverage) != set(CAPACITY_ALL_FEATURES) or any(
        not np.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        for value in feature_coverage.values()
    ):
        raise ValueError("CC-RTH study feature coverage is invalid")
    zero_coverage = {
        feature for feature, value in feature_coverage.items() if float(value) == 0.0
    }
    if any(zero_coverage & set(features) for features in feature_sets.values()):
        raise ValueError("CC-RTH study selected a zero-coverage feature")
    probability_normalization = _validate_probability_normalization_audit(report)
    if report.get("baseline_reproduction", {}).get("passed") is not True:
        raise ValueError("CC-RTH study did not reproduce its FLARE-24 reference")
    if report.get("primary_evaluation_period") != [PRIMARY_AUDIT_START, PRIMARY_AUDIT_END]:
        raise ValueError("CC-RTH primary evaluation does not preserve both graph boundaries")
    lock_path = _verify_file(report["method_lock"], role="CC-RTH method lock")
    lock, lock_hash_key = _self_hashed(lock_path)
    if lock[lock_hash_key] != report["method_lock"]["self_hash"]:
        raise ValueError("CC-RTH method lock self-hash differs from the report")
    if lock.get("outcomes_accessed_by_this_extension_before_lock", {}).get(
        "2025_loaded"
    ) is not False:
        raise ValueError("CC-RTH method lock was not written before 2025 access")
    if lock.get("graph_boundary_purge_days") != PURGE_DAYS or lock.get("periods", {}).get(
        "final_calibration_fit_through"
    ) != FINAL_CALIBRATION_END:
        raise ValueError("CC-RTH method lock does not preserve the temporal purge")
    recovery_validation = _validate_model_recovery(report, lock)
    artifacts = [
        *report.get("model_artifacts", []),
        *report.get("calibration_artifacts", []),
        *report.get("raw_selection_prediction_artifacts", []),
        report["selection_crossfit_prediction_artifact"],
        *report.get("retrospective_prediction_artifacts", []),
    ]
    prediction_records: list[dict[str, Any]] = []
    for record in artifacts:
        path = _verify_file(record, role="CC-RTH study artifact")
        if path.suffix == ".parquet":
            audit = _validate_prediction_probabilities(path) if "joint" in path.name or "retrospective" in path.name else {"rows": int(record["rows"])}
            prediction_records.append({"path": path.as_posix(), **audit})
    for record in report.get("provenance", {}).get("source_files", []):
        _verify_file(record, role="CC-RTH study source provenance")
    evaluation = report.get("retrospective_evaluation", {})
    methods = evaluation.get("methods", {})
    required_methods = {
        *CAPACITY_CANDIDATES,
        "global_simplex",
        "capacity_gated_simplex",
    }
    if set(methods) != required_methods:
        raise ValueError("CC-RTH study evaluation has an unexpected method set")
    n_values = {int(metrics["joint"]["n"]) for metrics in methods.values()}
    if len(n_values) != 1 or next(iter(n_values)) <= 0:
        raise ValueError("CC-RTH evaluation row counts are inconsistent")
    comparisons = evaluation.get("paired_date_cluster_comparisons", {})
    expected_clusters = (
        pd.Timestamp(PRIMARY_AUDIT_END) - pd.Timestamp(PRIMARY_AUDIT_START)
    ).days + 1
    if not comparisons or any(
        int(interval[metric]["clusters"]) != expected_clusters
        for interval in comparisons.values()
        for metric in ("joint_log_loss", "multiclass_brier")
    ):
        raise ValueError(
            "CC-RTH primary inference does not use every complete-context date"
        )
    return {
        "status": "PASS_CCRTH_STUDY_VALIDATION",
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "report": {
            "path": report_path.as_posix(),
            "sha256": sha256_file(report_path),
            "self_hash_key": hash_key,
            "self_hash": report[hash_key],
        },
        "method_lock": {
            "path": lock_path.as_posix(),
            "sha256": sha256_file(lock_path),
            "self_hash": lock[lock_hash_key],
        },
        "prediction_artifacts": prediction_records,
        "evaluated_rows": next(iter(n_values)),
        "methods": sorted(methods),
        "baseline_reproduction": report["baseline_reproduction"],
        "model_recovery": recovery_validation,
        "probability_normalization": probability_normalization,
        "confirmation_outcomes_accessed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-manifest", type=Path)
    parser.add_argument("--study-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if (args.capacity_manifest is None) == (args.study_report is None):
        raise SystemExit("supply exactly one of --capacity-manifest or --study-report")
    report = (
        validate_capacity_graph_manifest(args.capacity_manifest)
        if args.capacity_manifest is not None
        else validate_capacity_study_report(args.study_report)
    )
    report["validator_provenance"] = capture_provenance(
        (
            Path(__file__),
            Path(__file__).with_name("flare_capacity_contracts.py"),
            Path(__file__).with_name("flare_capacity_study.py"),
            Path(__file__).with_name("hashing.py"),
            Path(__file__).with_name("provenance.py"),
        )
    )
    report["validation_sha256"] = canonical_json_sha256(report)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite CC-RTH validation: {args.output}")
    write_canonical_json(args.output, report)
    print(json.dumps({"status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
