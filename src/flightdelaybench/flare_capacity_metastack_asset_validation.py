"""Validate the self-contained CC-RTH v9 publication tables and figures."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .flare_capacity_factorized_reporting import _load_self_hashed
from .flare_capacity_metastack_reporting import (
    EXPECTED_PARENT_ASSET_STATUS,
    EXPECTED_REPORT_STATUS,
    EXPECTED_VALIDATION_STATUS,
    coefficient_table,
    confirmation_table,
    monthly_score_table,
    paired_interval_table,
    primary_score_table,
    regime_score_table,
    regularization_table,
)
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

EXPECTED_MANIFEST_STATUS = "COMPLETE_VALIDATED_TF_CCRTH_METASTACK_PUBLICATION_ASSETS"
EXPECTED_PARENT_VALIDATION_STATUS = "PASS_TF_CCRTH_PUBLICATION_ASSET_VALIDATION"
EXPECTED_TABLES = {
    "primary_scores": 5,
    "paired_intervals": 8,
    "q4_selection": 4,
    "component_weights": 50,
    "posthoc_component_grid": 25,
    "monthly_scores": 60,
    "capacity_regime_scores": 15,
    "airport_scores": 1_000,
    "parent_feature_importance": 320,
    "metastack_primary_scores": 4,
    "metastack_paired_intervals": 4,
    "metastack_regularization_selection": 7,
    "metastack_coefficients": 6,
    "metastack_monthly_scores": 24,
    "metastack_capacity_regime_scores": 6,
    "metastack_airport_scores": 400,
    "metastack_confirmation_protocol": 1,
}
EXPECTED_FIGURES = {
    "validated_proper_score_effects",
    "cancellation_component_effects",
    "task_factorized_weights",
    "task_factorized_monthly_deltas",
    "posthoc_component_pair_grid",
    "task_factorized_regime_deltas",
    "hypergraph_feature_importance_corrected",
    "metastack_proper_score_intervals",
    "metastack_regularization_curve",
    "metastack_coefficients",
    "metastack_monthly_deltas",
    "metastack_regime_deltas",
    "metastack_cancellation_effects",
}


def _verify_record(record: Mapping[str, Any], *, role: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"{role} byte count differs")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{role} checksum differs")
    return path


def _records_by_name(
    records: Sequence[Mapping[str, Any]], *, role: str
) -> dict[str, Mapping[str, Any]]:
    result = {str(record.get("name", "")): record for record in records}
    if len(result) != len(records) or "" in result:
        raise ValueError(f"{role} records repeat or omit names")
    return result


def _assert_frame_equal(actual_path: Path, expected: pd.DataFrame, *, role: str) -> None:
    actual = pd.read_csv(actual_path)
    if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
        raise ValueError(f"{role} schema or row count differs")
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
    except AssertionError as error:
        raise ValueError(f"{role} values differ: {error}") from error


def validate_metastack_publication_assets(
    manifest_path: Path,
    *,
    report_path: Path,
    validation_path: Path,
    parent_manifest_path: Path,
    parent_validation_path: Path,
    output_path: Path,
    visual_qa_completed: bool,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite publication validation: {output_path}")
    if not visual_qa_completed:
        raise ValueError("manual visual QA must be completed before asset validation")
    manifest = _load_self_hashed(manifest_path, "manifest_sha256")
    report = _load_self_hashed(report_path, "report_sha256")
    validation = _load_self_hashed(validation_path, "validation_sha256")
    parent_manifest = _load_self_hashed(parent_manifest_path, "manifest_sha256")
    parent_validation = _load_self_hashed(parent_validation_path, "validation_sha256")
    if (
        manifest.get("status") != EXPECTED_MANIFEST_STATUS
        or report.get("status") != EXPECTED_REPORT_STATUS
        or validation.get("status") != EXPECTED_VALIDATION_STATUS
        or parent_manifest.get("status") != EXPECTED_PARENT_ASSET_STATUS
        or parent_validation.get("status") != EXPECTED_PARENT_VALIDATION_STATUS
    ):
        raise ValueError("publication input status differs")
    if (
        any(
            payload.get("confirmation_outcomes_accessed") is not False
            for payload in (manifest, validation, parent_manifest, parent_validation)
        )
        or report.get("outcomes_accessed", {}).get("2026_accessed") is not False
    ):
        raise ValueError("publication inputs crossed the unopened-2026 boundary")
    bindings = (
        (manifest["source_report"], report_path, report["report_sha256"], "report"),
        (
            manifest["source_validation"],
            validation_path,
            validation["validation_sha256"],
            "validation",
        ),
        (
            manifest["parent_asset_manifest"],
            parent_manifest_path,
            parent_manifest["manifest_sha256"],
            "parent asset manifest",
        ),
        (
            parent_validation["manifest"],
            parent_manifest_path,
            parent_manifest["manifest_sha256"],
            "parent asset validation",
        ),
    )
    for record, path, self_hash, role in bindings:
        if (
            Path(str(record.get("path", ""))).resolve() != path.resolve()
            or record.get("sha256") != sha256_file(path)
            or record.get("self_hash") != self_hash
        ):
            raise ValueError(f"publication {role} binding differs")

    table_records = _records_by_name(manifest["tables"], role="table")
    figure_records = _records_by_name(manifest["figures"], role="figure")
    if set(table_records) != set(EXPECTED_TABLES):
        raise ValueError("publication table set differs")
    if set(figure_records) != EXPECTED_FIGURES:
        raise ValueError("publication figure set differs")
    if int(manifest["inherited_table_count"]) != len(parent_manifest["tables"]) or int(
        manifest["inherited_figure_count"]
    ) != len(parent_manifest["figures"]):
        raise ValueError("publication inherited-asset counts differ")

    parent_tables = _records_by_name(parent_manifest["tables"], role="parent table")
    parent_figures = _records_by_name(parent_manifest["figures"], role="parent figure")
    for name, parent_record in {**parent_tables, **parent_figures}.items():
        record = table_records.get(name, figure_records.get(name))
        if record is None:
            raise ValueError(f"publication omitted inherited asset {name}")
        inherited_path = Path(str(record.get("inherited_from", "")))
        if (
            inherited_path.resolve() != Path(str(parent_record.get("path", ""))).resolve()
            or record.get("sha256") != parent_record.get("sha256")
            or int(record.get("bytes", -1)) != int(parent_record.get("bytes", -2))
        ):
            raise ValueError(f"publication inherited asset {name} differs")

    generated_tables = {
        "metastack_primary_scores": primary_score_table(report),
        "metastack_paired_intervals": paired_interval_table(report),
        "metastack_regularization_selection": regularization_table(report),
        "metastack_coefficients": coefficient_table(report),
        "metastack_monthly_scores": monthly_score_table(report),
        "metastack_capacity_regime_scores": regime_score_table(report),
        "metastack_airport_scores": pd.DataFrame.from_records(report["descriptive_airport_scores"]),
        "metastack_confirmation_protocol": confirmation_table(report),
    }
    table_results: list[dict[str, Any]] = []
    asset_paths: set[Path] = set()
    for name, expected_rows in EXPECTED_TABLES.items():
        record = table_records[name]
        path = _verify_record(record, role=f"publication table {name}")
        asset_paths.add(path.resolve())
        frame = pd.read_csv(path)
        if (
            len(frame) != expected_rows
            or int(record.get("rows", -1)) != expected_rows
            or list(frame.columns) != list(record.get("columns", ()))
        ):
            raise ValueError(f"publication table {name} contract differs")
        if name in generated_tables:
            _assert_frame_equal(path, generated_tables[name], role=name)
        table_results.append(
            {
                "name": name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "sha256": sha256_file(path),
                "inherited": name in parent_tables,
            }
        )

    figure_results: list[dict[str, Any]] = []
    for name in sorted(EXPECTED_FIGURES):
        record = figure_records[name]
        path = _verify_record(record, role=f"publication figure {name}")
        asset_paths.add(path.resolve())
        with Image.open(path) as figure:
            width, height = figure.size
            channels = len(figure.getbands())
            extrema = np.asarray(figure.convert("RGB").getextrema(), dtype=np.int64)
        if (
            width < 900
            or height < 600
            or channels not in (3, 4)
            or not (extrema[:, 0] < extrema[:, 1]).all()
        ):
            raise ValueError(f"publication figure {name} is undersized or blank")
        figure_results.append(
            {
                "name": name,
                "width_pixels": width,
                "height_pixels": height,
                "channels": channels,
                "sha256": sha256_file(path),
                "inherited": name in parent_figures,
            }
        )

    output_directories = {path.parent for path in asset_paths}
    if len(output_directories) != 1:
        raise ValueError("publication assets are not self-contained in one directory")
    listed_names = {path.name for path in asset_paths}
    disk_names = {path.name for path in next(iter(output_directories)).iterdir() if path.is_file()}
    if listed_names != disk_names:
        raise ValueError("publication directory contains unlisted or missing files")

    interval_table = generated_tables["metastack_paired_intervals"]
    if (
        interval_table["independent_confirmation"].any()
        or interval_table["selection_adjusted"].any()
        or not (interval_table["upper"] < 0.0).all()
    ):
        raise ValueError("publication interval disclosure or direction differs")
    monthly = generated_tables["metastack_monthly_scores"]
    stack_monthly = monthly.loc[monthly["method"].ne("flare24")]
    favorable_log_months = int((stack_monthly["joint_log_loss_delta_vs_flare24"] < 0.0).sum())
    favorable_brier_months = int((stack_monthly["multiclass_brier_delta_vs_flare24"] < 0.0).sum())
    if favorable_log_months != 12 or favorable_brier_months != 11:
        raise ValueError("publication temporal-stability summary differs")

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS_TF_CCRTH_METASTACK_PUBLICATION_ASSET_VALIDATION",
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "manifest": {
            "path": manifest_path.as_posix(),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "self_hash": manifest["manifest_sha256"],
        },
        "source_report": manifest["source_report"],
        "source_validation": manifest["source_validation"],
        "parent_asset_manifest": manifest["parent_asset_manifest"],
        "parent_asset_validation": {
            "path": parent_validation_path.as_posix(),
            "sha256": sha256_file(parent_validation_path),
            "self_hash": parent_validation["validation_sha256"],
        },
        "tables": table_results,
        "figures": figure_results,
        "semantic_checks": {
            "selected_regularization_c": float(report["selected_regularization_c"]),
            "held_forward_proper_score_upper_bounds_below_zero": 2,
            "full_primary_proper_score_upper_bounds_below_zero": 2,
            "favorable_joint_log_loss_months": favorable_log_months,
            "favorable_multiclass_brier_months": favorable_brier_months,
            "independent_2025_confirmation_claimed": False,
            "selection_adjusted_2025_interval_claimed": False,
        },
        "manual_visual_qa": {
            "status": "PASS_SIX_NEW_PNGS_INSPECTED_AT_ORIGINAL_RESOLUTION",
            "reviewed_at_utc": datetime.now(UTC).isoformat(),
            "inherited_figures": (
                "Seven parent figures are byte-identical to the independently validated v7 bundle."
            ),
            "checks": [
                "titles, axes, tick offsets, and period labels are readable",
                "no labels, annotations, legends, or panels overlap",
                "negative-is-better directions are explicit",
                "retrospective and non-blind status is visible on affected figures",
            ],
            "supersedes_failed_visual_bundle": (
                "manifests/failures/flare24_ccrth_publication_assets_v8_visual_qa.json"
            ),
        },
        "confirmation_outcomes_accessed": False,
        "validator_provenance": capture_provenance((Path(__file__),)),
    }
    result["validation_sha256"] = canonical_json_sha256(result)
    write_canonical_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--parent-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-qa-completed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = validate_metastack_publication_assets(
        args.manifest,
        report_path=args.report,
        validation_path=args.validation,
        parent_manifest_path=args.parent_manifest,
        parent_validation_path=args.parent_validation,
        output_path=args.output,
        visual_qa_completed=args.visual_qa_completed,
    )
    print(json.dumps({"status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
