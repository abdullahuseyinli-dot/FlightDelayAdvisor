"""Validate strictly-prior-year boundary-trained latent-rotation models."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]

from .boundary_rotation_models import RAW_HISTORY_COLUMNS
from .flare_rotation import LatentRotationGraph
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance


def validate_boundary_rotation_manifest(manifest_path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = payload.get("manifest_sha256")
    body = {name: value for name, value in payload.items() if name != "manifest_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError("boundary rotation manifest self-hash failed")
    if (
        payload.get("status") != "COMPLETE_PRIOR_YEAR_BOUNDARY_ROTATION_MODELS_NO_TARGET_TAILS"
        or set(payload.get("history_years", ())) != {2023, 2024}
        or set(payload.get("target_years", ())) != {2024, 2025}
        or tuple(payload.get("history_months", ())) != tuple(range(1, 10))
        or int(payload.get("history_tail_supervision_latest_month", -1)) != 9
        or int(payload.get("minimum_lag_to_earliest_target_cutoff_days", -1)) != 91
        or tuple(payload.get("historical_columns_read", ())) != RAW_HISTORY_COLUMNS
        or payload.get("historical_outcome_columns_read") != []
        or payload.get("target_year_rows_read") is not False
        or payload.get("target_tail_number_read") is not False
        or payload.get("target_outcome_columns_read") != []
        or payload.get("confirmation_outcomes_accessed") is not False
    ):
        raise ValueError("boundary rotation information boundary is invalid")
    protocol = payload.get("protocol", {})
    protocol_path = Path(str(protocol.get("path", "")))
    if (
        not protocol_path.is_file()
        or protocol_path.stat().st_size != int(protocol.get("bytes", -1))
        or sha256_file(protocol_path) != protocol.get("sha256")
        or protocol.get("identity", {}).get("method") != "BC-POT-Rotation-v1"
    ):
        raise ValueError("boundary rotation protocol binding failed")
    models = payload.get("models", [])
    if len(models) != 2:
        raise ValueError("boundary rotation manifest does not contain two models")
    model_records: list[dict[str, Any]] = []
    for record in models:
        history_year = int(record["history_year"])
        target_year = int(record["target_year"])
        if target_year != history_year + 1 or int(record.get("history_rows", 0)) <= 0:
            raise ValueError("boundary rotation model violates prior-year expansion")
        path = Path(str(record["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"boundary rotation model checksum failed: {path}")
        model = joblib.load(path)
        if not isinstance(model, LatentRotationGraph):
            raise TypeError(f"unexpected boundary rotation model type: {type(model)!r}")
        if model.model_card().as_dict() != record["model_card"]:
            raise ValueError("boundary rotation model card does not reproduce")
        model_records.append(
            {
                "history_year": history_year,
                "target_year": target_year,
                "history_rows": int(record["history_rows"]),
                "history_tail_nonmissing_rows": int(record["history_tail_nonmissing_rows"]),
                "path": path.as_posix(),
                "sha256": record["sha256"],
            }
        )
    raw_periods: set[tuple[int, int]] = set()
    forbidden = {
        "ArrDel15",
        "Cancelled",
        "Diverted",
        "DepDelay",
        "ArrDelay",
        "ActualElapsedTime",
        "AirTime",
        "TaxiIn",
        "TaxiOut",
    }
    for record in payload.get("raw_inputs", []):
        period = (int(record["year"]), int(record["month"]))
        if period in raw_periods:
            raise ValueError(f"duplicate boundary rotation raw period: {period}")
        raw_periods.add(period)
        if forbidden & set(record.get("columns_read", ())):
            raise ValueError("boundary rotation raw projection contains outcomes")
        path = Path(str(record["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"boundary rotation raw input checksum failed: {path}")
    expected = {(year, month) for year in (2023, 2024) for month in range(1, 10)}
    if raw_periods != expected:
        raise ValueError("boundary rotation raw inputs omit required periods")
    for record in payload.get("provenance", {}).get("source_files", []):
        path = Path(str(record["path"]))
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"boundary rotation source provenance failed: {path}")
    return {
        "status": "PASS_BOUNDARY_ROTATION_MODEL_VALIDATION",
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "manifest": {
            "path": manifest_path.as_posix(),
            "sha256": sha256_file(manifest_path),
            "self_hash": recorded,
        },
        "models": model_records,
        "raw_periods": len(raw_periods),
        "history_tail_supervision_latest_month": 9,
        "minimum_lag_to_earliest_target_cutoff_days": 91,
        "historical_tail_supervision": True,
        "historical_outcomes_accessed": False,
        "target_year_rows_read": False,
        "target_tail_number_read": False,
        "confirmation_outcomes_accessed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = validate_boundary_rotation_manifest(args.manifest)
    report["validator_provenance"] = capture_provenance(
        (
            Path(__file__),
            Path(__file__).with_name("boundary_rotation_models.py"),
            Path(__file__).with_name("flare_rotation.py"),
        )
    )
    report["validation_sha256"] = canonical_json_sha256(report)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite boundary rotation validation: {args.output}")
    write_canonical_json(args.output, report)
    print(json.dumps({"status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
