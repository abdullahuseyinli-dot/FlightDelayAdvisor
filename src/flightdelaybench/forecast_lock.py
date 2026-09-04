"""Freeze the 2024-selected forecast method before the 2025 retrospective audit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

LOCK_STATUS = "LOCKED_FOR_2025_RETROSPECTIVE_AUDIT"


def _verify_report(path: Path, *, status: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("report_sha256")
    body = {key: value for key, value in payload.items() if key != "report_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"report self-hash failed: {path}")
    if payload.get("status") != status:
        raise ValueError(f"report has unexpected status: {path}")
    return payload


def _verified_artifacts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for record in records:
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"artifact checksum failed before method lock: {path}")
        verified.append(dict(record))
    return verified


def validate_forecast_method_lock(path: Path) -> dict[str, Any]:
    """Validate the self-hash, audit boundary, confirmation gate, and artifacts."""

    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("lock_sha256")
    body = {key: value for key, value in payload.items() if key != "lock_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"forecast method lock self-hash failed: {path}")
    if payload.get("status") != LOCK_STATUS:
        raise PermissionError("forecast method is not locked for retrospective audit")
    if payload.get("audit_plan", {}).get("evaluation_year") != 2025:
        raise PermissionError("forecast method lock does not authorize this audit year")
    if payload.get("confirmation_gate", {}).get("authorized") is not False:
        raise PermissionError("2026 confirmation must remain unauthorized")
    _verified_artifacts(payload.get("frozen_artifacts", []))
    return payload


def create_forecast_method_lock(
    selection_report_path: Path,
    calibration_report_path: Path,
    *,
    output_path: Path,
) -> dict[str, Any]:
    """Create one immutable, create-only lock from validated 2024 evidence."""

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite forecast method lock: {output_path}")
    selection = _verify_report(
        selection_report_path,
        status="COMPLETE_2024_SELECTION_NOT_CONFIRMATORY",
    )
    calibration = _verify_report(
        calibration_report_path,
        status="COMPLETE_2024_CALIBRATION_SELECTION_NOT_CONFIRMATORY",
    )
    if calibration.get("selection_report_sha256") != sha256_file(selection_report_path):
        raise ValueError("calibration report is not bound to the supplied selection report")
    if any(
        item.get("selected_by_log_loss") != "forecast_residual"
        for item in selection["task_results"]
    ):
        raise ValueError("the forecast residual was not selected for every binary task")
    if selection.get("outcomes_accessed", {}).get("maximum_calendar_date") != "2024-12-31":
        raise ValueError("selection outcome boundary is not lock-compatible")
    if calibration.get("outcomes_accessed", {}).get("maximum_calendar_date") != "2024-12-31":
        raise ValueError("calibration outcome boundary is not lock-compatible")

    models = [record for record in selection["artifacts"] if record.get("kind") == "model"]
    calibrators = [
        record for record in calibration["artifacts"] if record.get("kind") == "calibrator"
    ]
    frozen_artifacts = _verified_artifacts([*models, *calibrators])
    selected_calibration = {
        item["task"]: item["selected_method"] for item in calibration["task_results"]
    }
    selection_evidence = {
        item["task"]: {
            "metrics": item["metrics"],
            "paired_forecast_minus_baseline": item["paired_forecast_minus_baseline"],
            "paired_forecast_minus_operational_residual": item[
                "paired_forecast_minus_operational_residual"
            ],
        }
        for item in selection["task_results"]
    }
    lock: dict[str, Any] = {
        "schema_version": 1,
        "status": LOCK_STATUS,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "prior_anchored_forecast_residual_adaptation",
        "working_acronym": "PAFRA",
        "primary_method": {
            "delay": "forecast_residual_raw_probability",
            "cancellation": "forecast_residual_raw_probability",
            "rationale": (
                "The forecast residual won Q4 2024 log loss against both frozen-baseline and "
                "operational-only controls. Raw probability is primary because the additional "
                "delay intercept calibration had a paired interval spanning zero."
            ),
        },
        "secondary_sensitivity": {
            "calibration_by_task": selected_calibration,
            "role": "prelocked secondary analysis; cannot replace the primary after viewing 2025",
        },
        "selection_report": {
            "path": selection_report_path.as_posix(),
            "sha256": sha256_file(selection_report_path),
            "self_hash": selection["report_sha256"],
        },
        "calibration_report": {
            "path": calibration_report_path.as_posix(),
            "sha256": sha256_file(calibration_report_path),
            "self_hash": calibration["report_sha256"],
        },
        "selection_evidence": selection_evidence,
        "frozen_artifacts": frozen_artifacts,
        "audit_plan": {
            "evaluation_year": 2025,
            "evaluation_dates": ["2025-01-01", "2025-12-31"],
            "model_refit": False,
            "calibrator_refit": False,
            "comparators": [
                "frozen_baseline",
                "operational_residual",
                "forecast_residual_raw",
                "forecast_residual_prelocked_calibration",
            ],
            "primary_scores": ["log_loss", "brier", "brier_skill"],
            "secondary_scores": [
                "roc_auc",
                "average_precision",
                "ece_equal_mass",
                "calibration_intercept",
                "calibration_slope",
                "top_decile_lift",
            ],
            "paired_uncertainty": "flight-date cluster bootstrap, 2000 repetitions",
            "method_changes_after_audit": False,
        },
        "outcome_boundary_at_lock": "2024-12-31",
        "blinding_status": (
            "NOT_BLIND_RETROSPECTIVE_AUDIT: 2025 outcomes and legacy scores existed in the "
            "repository before this method was developed. The lock prevents new-method tuning "
            "on the audit but does not turn 2025 into a never-seen confirmation."
        ),
        "confirmation_gate": {
            "year": 2026,
            "authorized": False,
            "rule": "No 2026 outcome may be acquired, opened, summarized, or scored by this lock.",
        },
        "claim_limits": {
            "same_cohort_state_of_the_art": False,
            "causal_weather_effect": False,
            "live_feed_equivalence": False,
            "never_seen_2025_confirmation": False,
        },
        "provenance": capture_provenance(
            (
                Path(__file__),
                Path(__file__).with_name("forecast_adaptation.py"),
                Path(__file__).with_name("forecast_calibration.py"),
                Path(__file__).with_name("forecast_modeling.py"),
            )
        ),
    }
    lock["lock_sha256"] = canonical_json_sha256(lock)
    write_canonical_json(output_path, lock)
    validate_forecast_method_lock(output_path)
    return lock


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection_report", type=Path)
    parser.add_argument("calibration_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    lock = create_forecast_method_lock(
        args.selection_report,
        args.calibration_report,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "status": lock["status"],
                "lock_sha256": lock["lock_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
