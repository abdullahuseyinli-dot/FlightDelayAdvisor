"""Freeze the FLARE-24 confirmation analysis without opening 2026 outcomes."""

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance


def _self_hashed_payload(path: Path) -> tuple[dict[str, Any], str]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    keys = [key for key in ("report_sha256", "manifest_sha256") if key in payload]
    if len(keys) != 1:
        raise ValueError(f"confirmation input has no unique self-hash: {path}")
    key = keys[0]
    body = {name: value for name, value in payload.items() if name != key}
    if payload[key] != canonical_json_sha256(body):
        raise ValueError(f"confirmation input self-hash failed: {path}")
    return payload, key


def _reference(path: Path, *, role: str, expected_status: str) -> dict[str, Any]:
    payload, key = _self_hashed_payload(path)
    if payload.get("status") != expected_status:
        raise ValueError(f"{role} has invalid status")
    return {
        "role": role,
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "self_hash_key": key,
        "self_hash": payload[key],
        "status": payload["status"],
    }


def write_flare24_confirmation_lock(
    *,
    protocol_path: Path,
    selection_report_path: Path,
    method_lock_path: Path,
    audit_report_path: Path,
    publication_bundle_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write the create-only analysis lock that must precede any 2026 scoring."""

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite FLARE-24 confirmation lock: {output_path}")
    protocol = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("information_boundary", {}).get("confirmation_gate_opened") is not False:
        raise ValueError("FLARE-24 protocol confirmation gate is not closed")
    if protocol.get("information_boundary", {}).get("confirmation_year") != 2026:
        raise ValueError("FLARE-24 protocol has an unexpected confirmation year")
    selection = _reference(
        selection_report_path,
        role="2024_selection_report",
        expected_status="COMPLETE_2024_FLARE24_SELECTION_NOT_CONFIRMATORY",
    )
    method = _reference(
        method_lock_path,
        role="pre_2025_method_lock",
        expected_status="LOCKED_FLARE24_METHOD_BEFORE_2025_RETROSPECTIVE_AUDIT",
    )
    audit = _reference(
        audit_report_path,
        role="2025_retrospective_audit",
        expected_status=(
            "COMPLETE_2025_FLARE24_RETROSPECTIVE_AUDIT_NOT_BLIND_CONFIRMATION"
        ),
    )
    publication = _reference(
        publication_bundle_path,
        role="retrospective_publication_bundle",
        expected_status="COMPLETE_FLARE24_PUBLICATION_BUNDLE",
    )
    method_payload, _ = _self_hashed_payload(method_lock_path)
    audit_payload, _ = _self_hashed_payload(audit_report_path)
    if audit_payload.get("outcomes_accessed", {}).get("2026_accessed") is not False:
        raise ValueError("2026 outcomes were accessed before the confirmation lock")
    if audit_payload.get("method_lock", {}).get("sha256") != method["sha256"]:
        raise ValueError("2025 audit is not bound to the supplied pre-audit method lock")
    if method_payload.get("selection_report", {}).get("sha256") != selection["sha256"]:
        raise ValueError("pre-audit method lock is not bound to the supplied selection")
    evaluation = protocol.get("evaluation", {})
    if evaluation.get("primary_metrics") != ["joint_log_loss", "multiclass_brier"]:
        raise ValueError("FLARE-24 confirmation metrics differ from the frozen protocol")
    lock: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED_2026_FLARE24_CONFIRMATION_PROTOCOL_OUTCOMES_UNOPENED",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "FLARE-24",
        "protocol": {
            "role": "frozen_protocol",
            "path": protocol_path.as_posix(),
            "bytes": protocol_path.stat().st_size,
            "sha256": sha256_file(protocol_path),
        },
        "locked_artifacts": [selection, method, audit, publication],
        "frozen_method": method_payload["frozen_choices"],
        "confirmation_population": {
            "calendar_window": ["2026-01-01", "2026-06-30"],
            "airport_scope": (
                "same frozen top-100-airport union and official-census eligibility rules"
            ),
            "joint_state_population": (
                "rows with observed on-time, delayed, or cancelled joint state"
            ),
            "cancellation_population": "all scheduled cohort rows with binary Cancelled",
            "delay_population": (
                "operated non-diverted flights with observed ArrDel15"
            ),
        },
        "analysis": {
            "reference_method": "baseline",
            "candidate_method": "reconciled",
            "candidate_interpretation": (
                "frozen structural-rotation output; convex ensemble selected it at weight "
                "one and aggregate alignment was locked off"
            ),
            "co_primary_metrics": ["joint_log_loss", "multiclass_brier"],
            "difference_direction": "candidate_minus_baseline; negative favours candidate",
            "cluster_key": "FlightDate",
            "bootstrap_repetitions": int(evaluation["bootstrap_repetitions"]),
            "bootstrap_seed": int(evaluation["bootstrap_seed"]),
            "confidence": 0.95,
            "confirmation_success_rule": (
                "both co-primary paired date-cluster interval upper bounds are below zero"
            ),
            "monthly_binary_calibration_and_prespecified_strata": "secondary_descriptive",
            "missing_outcome_rule": "never impute a missing delay or joint label as on time",
        },
        "prohibited_after_lock": [
            "feature changes",
            "model or calibrator refit",
            "ensemble or reconciliation reselection",
            "cohort exclusions based on 2026 outcomes",
            "metric or multiplicity-rule changes",
        ],
        "outcomes_accessed_at_lock": {
            "years": [2024, 2025],
            "maximum_calendar_date": "2025-12-31",
            "2026": False,
        },
        "confirmation_gate": {
            "year": 2026,
            "opened": False,
            "authorization": "not granted by writing this lock",
        },
        "provenance": capture_provenance(
            (
                Path(__file__),
                Path(__file__).with_name("flare_evaluation.py"),
                Path(__file__).with_name("bootstrap.py"),
                Path(__file__).with_name("metrics.py"),
            )
        ),
        "claim_limit": (
            "This lock freezes a future confirmation analysis. It contains no 2026 outcome "
            "and does not itself upgrade retrospective evidence to confirmation."
        ),
    }
    lock["manifest_sha256"] = canonical_json_sha256(lock)
    write_canonical_json(output_path, lock)
    return lock


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--method-lock", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--publication-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    lock = write_flare24_confirmation_lock(
        protocol_path=args.protocol,
        selection_report_path=args.selection_report,
        method_lock_path=args.method_lock,
        audit_report_path=args.audit_report,
        publication_bundle_path=args.publication_bundle,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": lock["status"],
                "output": args.output.as_posix(),
                "manifest_sha256": lock["manifest_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
