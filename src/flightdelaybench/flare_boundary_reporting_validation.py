"""Validate BC-POT-R publication tables, figures, and arithmetic."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .flare_boundary_study_validation import METHODS
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance


def _self_hashed(path: Path) -> tuple[dict[str, Any], str]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    keys = [
        key
        for key in ("manifest_sha256", "report_sha256", "validation_sha256")
        if key in payload
    ]
    if len(keys) != 1:
        raise ValueError(f"invalid self-hash keys: {path}")
    key = keys[0]
    body = {name: value for name, value in payload.items() if name != key}
    if canonical_json_sha256(body) != payload[key]:
        raise ValueError(f"self-hash failed: {path}")
    return payload, key


def _verified(record: dict[str, Any], *, role: str) -> Path:
    path = Path(str(record.get("path", "")))
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or sha256_file(path) != record.get("sha256")
    ):
        raise ValueError(f"{role} checksum or byte count failed: {path}")
    return path


def validate_boundary_publication_assets(manifest_path: Path) -> dict[str, Any]:
    from PIL import Image

    manifest, manifest_key = _self_hashed(manifest_path)
    if manifest.get("status") != "COMPLETE_BCPOTR_PUBLICATION_ASSETS":
        raise ValueError("BC-POT-R publication manifest is not complete")
    report_path = _verified(manifest["source_report"], role="source report")
    validation_path = _verified(manifest["source_validation"], role="source validation")
    report, report_key = _self_hashed(report_path)
    validation, validation_key = _self_hashed(validation_path)
    if (
        report[report_key] != manifest["source_report"].get("self_hash")
        or validation[validation_key] != manifest["source_validation"].get("self_hash")
        or validation.get("status") != "PASS_BCPOTR_STUDY_VALIDATION"
        or validation.get("report", {}).get("self_hash") != report[report_key]
    ):
        raise ValueError("BC-POT-R publication source binding failed")

    tables: dict[str, pd.DataFrame] = {}
    table_records: list[dict[str, Any]] = []
    for record in manifest.get("tables", []):
        name = str(record["name"])
        if name in tables:
            raise ValueError(f"duplicate publication table: {name}")
        path = _verified(record, role=f"publication table {name}")
        frame = pd.read_csv(path)
        if len(frame) != int(record["rows"]):
            raise ValueError(f"publication table row count failed: {name}")
        tables[name] = frame
        table_records.append({"name": name, "rows": len(frame), "path": path.as_posix()})
    required_tables = {
        "primary_metrics",
        "monthly_metrics",
        "boundary_residual_regimes",
        "boundary_signal_outcome_associations",
        "feature_importance",
        "airport_scores",
    }
    if set(tables) != required_tables:
        raise ValueError("BC-POT-R publication table set is incomplete")

    primary = tables["primary_metrics"].set_index("method")
    if set(primary.index) != set(METHODS):
        raise ValueError("primary publication table method set differs")
    schedule_accuracy = float(primary.loc["schedule_baseline", "argmax_accuracy"])
    meta_accuracy = float(primary.loc["meta_current", "argmax_accuracy"])
    for method in METHODS:
        accuracy = float(primary.loc[method, "argmax_accuracy"])
        if (
            not np.isclose(
                float(primary.loc[method, "absolute_accuracy_gain_vs_schedule"]),
                accuracy - schedule_accuracy,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.isclose(
                float(primary.loc[method, "absolute_accuracy_gain_vs_previous_meta"]),
                accuracy - meta_accuracy,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.isclose(
                float(primary.loc[method, "accuracy_percentage_point_gain_vs_schedule"]),
                100.0 * (accuracy - schedule_accuracy),
                rtol=0.0,
                atol=1e-10,
            )
        ):
            raise ValueError(f"publication accuracy arithmetic failed: {method}")
    selected = str(manifest["selected_method"])
    if selected != report.get("selected_method_by_2024_forward_score"):
        raise ValueError("publication selected method differs from study")
    reported_gate = report["breakthrough_gate"]
    published_gate = manifest["breakthrough_gate"]
    if published_gate != reported_gate:
        raise ValueError("publication breakthrough gate differs from study")
    selected_gain = float(primary.loc[selected, "absolute_accuracy_gain_vs_previous_meta"])
    if not np.isclose(selected_gain, float(reported_gate["absolute_gain"]), atol=1e-12):
        raise ValueError("publication selected accuracy gain differs from study")

    figure_records: list[dict[str, Any]] = []
    for record in manifest.get("figures", []):
        path = _verified(record, role=f"publication figure {record.get('name')}")
        with Image.open(path) as image:
            image.verify()
            width, height = image.size
        if width < 800 or height < 500:
            raise ValueError(f"publication figure is too small: {path}")
        figure_records.append(
            {"name": record["name"], "path": path.as_posix(), "width": width, "height": height}
        )
    if {record["name"] for record in figure_records} != {
        "primary_comparison",
        "monthly_stability",
    }:
        raise ValueError("BC-POT-R publication figure set is incomplete")
    summary_path = _verified(manifest["summary"], role="publication summary")
    summary = summary_path.read_text(encoding="utf-8")
    if (
        "Absolute accuracy gain vs previous meta-stack" not in summary
        or "Requested +0.05 gate passed" not in summary
        or "2026 confirmation gate remains unopened" not in summary
    ):
        raise ValueError("publication summary omits required claim boundaries")
    return {
        "status": "PASS_BCPOTR_PUBLICATION_ASSET_VALIDATION",
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "manifest": {
            "path": manifest_path.as_posix(),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "self_hash": manifest[manifest_key],
        },
        "source_report_self_hash": report[report_key],
        "tables": table_records,
        "figures": figure_records,
        "selected_method": selected,
        "absolute_accuracy_gain_vs_previous_meta": selected_gain,
        "breakthrough_gate_reproduced": True,
        "summary_path": summary_path.as_posix(),
        "confirmation_outcomes_accessed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = validate_boundary_publication_assets(args.manifest)
    result["validator_provenance"] = capture_provenance((Path(__file__),))
    result["validation_sha256"] = canonical_json_sha256(result)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite publication validation: {args.output}")
    write_canonical_json(args.output, result)
    print(json.dumps({"status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
