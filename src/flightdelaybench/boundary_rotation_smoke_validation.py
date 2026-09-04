"""Validate the outcome-blind BC-POT boundary-rotation materiality smoke gate."""

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .flare_capacity_contracts import CAPACITY_ALL_FEATURES
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

ROTATION_MESSAGE_FEATURES = (
    "ccrth_rotation_connected_probability",
    "ccrth_rotation_resource_message_coverage",
    "ccrth_rotation_predecessor_capacity_overload",
    "ccrth_rotation_predecessor_capacity_queue",
    "ccrth_rotation_predecessor_capacity_shadow_price",
)
NON_ROTATION_FEATURES = tuple(
    feature for feature in CAPACITY_ALL_FEATURES if feature not in ROTATION_MESSAGE_FEATURES
)


def _self_hashed(path: Path) -> tuple[dict[str, Any], str]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    keys = [key for key in ("manifest_sha256", "validation_sha256") if key in payload]
    if len(keys) != 1:
        raise ValueError(f"invalid self-hash keys: {path}")
    key = keys[0]
    body = {name: value for name, value in payload.items() if name != key}
    if canonical_json_sha256(body) != payload[key]:
        raise ValueError(f"self-hash failed: {path}")
    return payload, key


def _verified_file(record: dict[str, Any], *, role: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{role} checksum failed: {path}")
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{role} byte count failed: {path}")
    return path


def _feature_path(manifest: dict[str, Any], *, year: int, month: int) -> Path:
    records = [
        record
        for record in manifest.get("outputs", [])
        if int(record.get("year", -1)) == year and int(record.get("month", -1)) == month
    ]
    if len(records) != 1:
        raise ValueError(f"feature manifest has {len(records)} outputs for {year}-{month:02d}")
    return _verified_file(records[0]["features"], role=f"{year}-{month:02d} feature")


def _changed(left: np.ndarray, right: np.ndarray, *, atol: float) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError("comparison arrays do not align")
    both_missing = np.isnan(left) & np.isnan(right)
    close = np.isclose(left, right, rtol=0.0, atol=atol, equal_nan=False)
    return np.asarray(~(both_missing | close), dtype=np.bool_)


def compare_rotation_feature_frames(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Verify isolation and measure rotation-message changes without labels."""

    expected = {"sample_id", *CAPACITY_ALL_FEATURES}
    for role, frame in (("reference", reference), ("candidate", candidate)):
        missing = sorted(expected - set(frame.columns))
        if missing:
            raise ValueError(f"{role} rotation smoke frame omits columns: {missing}")
        if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ValueError(f"{role} rotation smoke frame has invalid IDs")
    if set(reference["sample_id"].astype(str)) != set(candidate["sample_id"].astype(str)):
        raise ValueError("boundary rotation smoke changed the target ID set")
    joined = reference.merge(
        candidate,
        on="sample_id",
        how="inner",
        sort=False,
        validate="one_to_one",
        suffixes=("__reference", "__candidate"),
    )
    non_rotation_drift: dict[str, float] = {}
    for feature in NON_ROTATION_FEATURES:
        left = pd.to_numeric(joined[f"{feature}__reference"], errors="raise").to_numpy(
            dtype=np.float64
        )
        right = pd.to_numeric(joined[f"{feature}__candidate"], errors="raise").to_numpy(
            dtype=np.float64
        )
        changed = _changed(left, right, atol=tolerance)
        if changed.any():
            difference = np.abs(left[changed] - right[changed])
            finite = difference[np.isfinite(difference)]
            non_rotation_drift[feature] = (
                float(finite.max()) if finite.size else float("inf")
            )
    if non_rotation_drift:
        raise ValueError(
            "boundary rotation smoke changed non-rotation features: "
            f"{sorted(non_rotation_drift)}"
        )
    per_feature: dict[str, Any] = {}
    any_changed = np.zeros(len(joined), dtype=np.bool_)
    for feature in ROTATION_MESSAGE_FEATURES:
        left = pd.to_numeric(joined[f"{feature}__reference"], errors="raise").to_numpy(
            dtype=np.float64
        )
        right = pd.to_numeric(joined[f"{feature}__candidate"], errors="raise").to_numpy(
            dtype=np.float64
        )
        changed = _changed(left, right, atol=tolerance)
        any_changed |= changed
        observed_pair = np.isfinite(left) & np.isfinite(right)
        absolute = np.abs(right[observed_pair] - left[observed_pair])
        per_feature[feature] = {
            "changed_rows": int(changed.sum()),
            "changed_row_fraction": float(changed.mean()),
            "mean_absolute_change_on_jointly_observed_rows": (
                float(absolute.mean()) if absolute.size else None
            ),
            "maximum_absolute_change_on_jointly_observed_rows": (
                float(absolute.max()) if absolute.size else None
            ),
            "newly_observed_rows": int((np.isfinite(right) & ~np.isfinite(left)).sum()),
            "lost_observed_rows": int((np.isfinite(left) & ~np.isfinite(right)).sum()),
        }
    return {
        "rows": len(joined),
        "target_id_set_match": True,
        "static_feature_equality": True,
        "non_rotation_feature_equality": True,
        "any_rotation_feature_changed_rows": int(any_changed.sum()),
        "any_rotation_feature_changed_row_fraction": float(any_changed.mean()),
        "per_rotation_feature": per_feature,
    }


def validate_boundary_rotation_smoke(
    *,
    protocol_path: Path,
    rotation_validation_path: Path,
    reference_manifest_path: Path,
    candidate_manifest_path: Path,
    year: int = 2024,
    month: int = 1,
) -> dict[str, Any]:
    protocol: dict[str, Any] = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    gate = protocol.get("outcome_blind_smoke_gate", {})
    if (
        protocol.get("identity", {}).get("method") != "BC-POT-Rotation-v1"
        or gate.get("period") != f"{year}-{month:02d}"
        or gate.get("required_target_id_set_match") is not True
        or gate.get("required_static_feature_equality") is not True
    ):
        raise ValueError("rotation smoke invocation differs from the frozen protocol")
    minimum_fraction = float(gate.get("minimum_changed_rotation_feature_row_fraction", -1.0))
    if not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("rotation smoke materiality threshold is invalid")

    rotation_validation, rotation_validation_key = _self_hashed(rotation_validation_path)
    if rotation_validation.get("status") != "PASS_BOUNDARY_ROTATION_MODEL_VALIDATION":
        raise ValueError("boundary rotation models do not have a passing validation")
    rotation_manifest_record = rotation_validation.get("manifest", {})
    rotation_manifest_path = _verified_file(
        rotation_manifest_record,
        role="boundary rotation manifest",
    )
    rotation_manifest, rotation_manifest_key = _self_hashed(rotation_manifest_path)
    if rotation_manifest[rotation_manifest_key] != rotation_manifest_record.get("self_hash"):
        raise ValueError("boundary rotation manifest binding differs")

    reference_manifest, _ = _self_hashed(reference_manifest_path)
    candidate_manifest, _ = _self_hashed(candidate_manifest_path)
    if reference_manifest.get("method") != candidate_manifest.get("method"):
        raise ValueError("rotation smoke feature methods differ")
    candidate_rotation = candidate_manifest.get("rotation_manifest", {})
    if (
        Path(str(candidate_rotation.get("path", ""))).resolve()
        != rotation_manifest_path.resolve()
        or candidate_rotation.get("sha256") != sha256_file(rotation_manifest_path)
    ):
        raise ValueError("candidate features are not bound to the validated boundary rotation")
    reference_path = _feature_path(reference_manifest, year=year, month=month)
    candidate_path = _feature_path(candidate_manifest, year=year, month=month)
    columns = ["sample_id", *CAPACITY_ALL_FEATURES]
    comparison = compare_rotation_feature_frames(
        pd.read_parquet(reference_path, columns=columns),
        pd.read_parquet(candidate_path, columns=columns),
    )
    fraction = float(comparison["any_rotation_feature_changed_row_fraction"])
    passed = fraction >= minimum_fraction
    return {
        "status": (
            "PASS_BOUNDARY_ROTATION_OUTCOME_BLIND_MATERIALITY_GATE"
            if passed
            else "FAIL_BOUNDARY_ROTATION_OUTCOME_BLIND_MATERIALITY_GATE"
        ),
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "path": protocol_path.as_posix(),
            "bytes": protocol_path.stat().st_size,
            "sha256": sha256_file(protocol_path),
        },
        "rotation_validation": {
            "path": rotation_validation_path.as_posix(),
            "bytes": rotation_validation_path.stat().st_size,
            "sha256": sha256_file(rotation_validation_path),
            "self_hash": rotation_validation[rotation_validation_key],
        },
        "reference_manifest": {
            "path": reference_manifest_path.as_posix(),
            "sha256": sha256_file(reference_manifest_path),
        },
        "candidate_manifest": {
            "path": candidate_manifest_path.as_posix(),
            "sha256": sha256_file(candidate_manifest_path),
        },
        "period": {"year": year, "month": month},
        "minimum_changed_rotation_feature_row_fraction": minimum_fraction,
        "comparison": comparison,
        "gate_passed": passed,
        "outcomes_read": False,
        "target_tail_number_read": False,
        "confirmation_outcomes_accessed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--rotation-validation", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = validate_boundary_rotation_smoke(
        protocol_path=args.protocol,
        rotation_validation_path=args.rotation_validation,
        reference_manifest_path=args.reference_manifest,
        candidate_manifest_path=args.candidate_manifest,
        year=args.year,
        month=args.month,
    )
    report["validator_provenance"] = capture_provenance((Path(__file__),))
    report["validation_sha256"] = canonical_json_sha256(report)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite rotation smoke validation: {args.output}")
    write_canonical_json(args.output, report)
    print(json.dumps({"status": report["status"], "gate_passed": report["gate_passed"]}, indent=2))


if __name__ == "__main__":
    main()
