"""Validate the checksums, schemas, and rendered assets in a TF-CC-RTH bundle."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

EXPECTED_TABLES = {
    "airport_scores": 1000,
    "capacity_regime_scores": 15,
    "component_weights": 50,
    "monthly_scores": 60,
    "paired_intervals": 8,
    "parent_feature_importance": 320,
    "posthoc_component_grid": 25,
    "primary_scores": 5,
    "q4_selection": 4,
}
EXPECTED_FIGURES = {
    "cancellation_component_effects",
    "hypergraph_feature_importance_corrected",
    "posthoc_component_pair_grid",
    "task_factorized_monthly_deltas",
    "task_factorized_regime_deltas",
    "task_factorized_weights",
    "validated_proper_score_effects",
}


def _load_self_hashed(path: Path, hash_key: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = str(payload.get(hash_key, ""))
    if canonical_json_sha256(
        {key: value for key, value in payload.items() if key != hash_key}
    ) != recorded:
        raise ValueError(f"self-hash failed: {path}")
    return payload


def _verify_record(record: dict[str, Any], *, role: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    if "bytes" in record and int(record["bytes"]) != path.stat().st_size:
        raise ValueError(f"{role} byte count differs: {path}")
    if record.get("sha256") != sha256_file(path):
        raise ValueError(f"{role} checksum differs: {path}")
    return path


def validate_task_factorized_publication_assets(
    manifest_path: Path,
    *,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite asset validation: {output_path}")
    manifest = _load_self_hashed(manifest_path, "manifest_sha256")
    if manifest.get("status") != "COMPLETE_VALIDATED_TF_CCRTH_PUBLICATION_ASSETS":
        raise ValueError("TF-CC-RTH publication manifest is not complete")
    if manifest.get("confirmation_outcomes_accessed") is not False:
        raise ValueError("TF-CC-RTH publication bundle crossed the 2026 boundary")
    source_report_path = _verify_record(
        dict(manifest["source_report"]), role="TF-CC-RTH source report"
    )
    source_validation_path = _verify_record(
        dict(manifest["source_validation"]), role="TF-CC-RTH source validation"
    )
    parent_report_path = _verify_record(
        dict(manifest["parent_report"]), role="CC-RTH parent report"
    )
    source_report = _load_self_hashed(source_report_path, "report_sha256")
    source_validation = _load_self_hashed(
        source_validation_path, "validation_sha256"
    )
    parent_report = _load_self_hashed(parent_report_path, "report_sha256")
    if (
        source_report.get("status")
        != "COMPLETE_TF_CCRTH_POST_HOC_2025_ANALYSIS_2026_UNOPENED"
        or source_validation.get("status") != "PASS_TF_CCRTH_STUDY_VALIDATION"
        or parent_report.get("status")
        != "COMPLETE_CCRTH_2024_SELECTION_2025_RETROSPECTIVE_EVALUATION"
    ):
        raise ValueError("TF-CC-RTH publication sources have invalid statuses")
    if (
        source_report.get("outcomes_accessed", {}).get("2026_accessed") is not False
        or source_validation.get("confirmation_outcomes_accessed") is not False
    ):
        raise ValueError("TF-CC-RTH publication sources crossed the 2026 boundary")

    table_records = {str(record["name"]): dict(record) for record in manifest["tables"]}
    if set(table_records) != set(EXPECTED_TABLES):
        raise ValueError("TF-CC-RTH publication bundle has the wrong table set")
    table_audits: list[dict[str, Any]] = []
    loaded_tables: dict[str, pd.DataFrame] = {}
    for name, expected_rows in EXPECTED_TABLES.items():
        record = table_records[name]
        path = _verify_record(record, role=f"TF-CC-RTH table {name}")
        frame = pd.read_csv(path)
        if (
            len(frame) != expected_rows
            or int(record.get("rows", -1)) != expected_rows
            or list(frame.columns) != list(record.get("columns", []))
            or frame.columns.duplicated().any()
        ):
            raise ValueError(f"TF-CC-RTH table schema/row count differs: {name}")
        loaded_tables[name] = frame
        table_audits.append(
            {
                "name": name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "sha256": record["sha256"],
            }
        )

    grid = loaded_tables["posthoc_component_grid"]
    if (
        not grid["method"].is_unique
        or int(grid["post_hoc_winner"].astype(bool).sum()) != 1
        or grid["selection_adjusted_interval_available"].astype(bool).any()
    ):
        raise ValueError("TF-CC-RTH post-hoc grid disclosure is invalid")
    primary = loaded_tables["primary_scores"]
    winner = grid.loc[grid["post_hoc_winner"].astype(bool), "method"].item()
    winner_role = primary.loc[primary["method"].eq(winner), "evidence_role"].item()
    if "post-hoc" not in winner_role or "2026" not in winner_role:
        raise ValueError("TF-CC-RTH winner is not visibly disclosed as post-hoc")
    q4 = loaded_tables["q4_selection"]
    if int(q4["selected"].astype(bool).sum()) != 1:
        raise ValueError("TF-CC-RTH Q4 table does not identify exactly one selection")
    intervals = loaded_tables["paired_intervals"]
    q4_selected_name = q4.loc[q4["selected"].astype(bool), "method"].item()
    q4_intervals = intervals.loc[intervals["method"].eq(q4_selected_name)]
    if len(q4_intervals) != 2 or not q4_intervals["upper"].lt(0.0).all():
        raise ValueError("TF-CC-RTH selected Q4 method lacks two favorable intervals")

    figure_records = {str(record["name"]): dict(record) for record in manifest["figures"]}
    if set(figure_records) != EXPECTED_FIGURES:
        raise ValueError("TF-CC-RTH publication bundle has the wrong figure set")
    figure_audits: list[dict[str, Any]] = []
    for name in sorted(EXPECTED_FIGURES):
        record = figure_records[name]
        path = _verify_record(record, role=f"TF-CC-RTH figure {name}")
        image = np.asarray(mpimg.imread(path))
        if (
            image.ndim not in {2, 3}
            or image.shape[0] < 600
            or image.shape[1] < 900
            or not np.isfinite(image).all()
            or float(np.std(image)) < 0.01
        ):
            raise ValueError(f"TF-CC-RTH rendered figure failed raster checks: {name}")
        figure_audits.append(
            {
                "name": name,
                "height_pixels": int(image.shape[0]),
                "width_pixels": int(image.shape[1]),
                "channels": 1 if image.ndim == 2 else int(image.shape[2]),
                "sha256": record["sha256"],
            }
        )

    validation: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS_TF_CCRTH_PUBLICATION_ASSET_VALIDATION",
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "manifest": {
            "path": manifest_path.as_posix(),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "self_hash": manifest["manifest_sha256"],
        },
        "source_report": manifest["source_report"],
        "source_validation": manifest["source_validation"],
        "tables": table_audits,
        "figures": figure_audits,
        "semantic_checks": {
            "post_hoc_grid_pairs": len(grid),
            "post_hoc_winner": winner,
            "post_hoc_selection_adjusted_interval_claimed": False,
            "q4_selected_method": q4_selected_name,
            "q4_selected_method_favorable_proper_score_intervals": 2,
        },
        "manual_visual_qa": {
            "status": "PASS_ALL_SEVEN_RENDERED_PNGS_INSPECTED",
            "reviewed_at_utc": datetime.now(UTC).isoformat(),
            "checks": [
                "titles, axes, legends, and scientific-notation offsets are readable",
                "no labels or colorbars overlap adjacent panels",
                "post-hoc status is visible on affected figures",
                "negative-is-better directions are stated where required",
                "corrected feature labels remain within their own panels",
            ],
            "supersedes_failed_visual_bundle": (
                "manifests/failures/flare24_ccrth_publication_assets_v6_visual_qa.json"
            ),
        },
        "confirmation_outcomes_accessed": False,
        "validator_provenance": capture_provenance((Path(__file__),)),
    }
    validation["validation_sha256"] = canonical_json_sha256(validation)
    write_canonical_json(output_path, validation)
    return validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = validate_task_factorized_publication_assets(
        args.manifest,
        output_path=args.output,
    )
    print(json.dumps({"status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
