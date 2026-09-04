from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from flightdelaybench.contracts import FLARE24_ROTATION_FEATURES
from flightdelaybench.flare_features import FLARE24_MATERIALIZED_FEATURES
from flightdelaybench.flare_validation import (
    validate_feature_artifact,
    validate_method_lock_artifact,
    validate_publication_bundle_artifact,
    validate_rotation_artifact,
    validate_study_artifact,
    validate_weather_artifact,
)
from flightdelaybench.flare_weather import OPEN_METEO_VARIABLES
from flightdelaybench.hashing import canonical_json_sha256, sha256_file


def test_weather_artifact_recomputes_hashes_and_invariants(tmp_path: Path) -> None:
    row: dict[str, object] = {
        "Airport": "AAA",
        "valid_time_utc": pd.Timestamp("2024-01-02 00:00"),
        "issue_time_utc": pd.Timestamp("2024-01-01 00:00"),
        "lead_hours": 24.0,
    }
    row.update({name: 1.0 for name in OPEN_METEO_VARIABLES})
    weather_path = tmp_path / "year=2024.parquet"
    pd.DataFrame([row]).to_parquet(weather_path, index=False)
    manifest: dict[str, object] = {
        "outputs": [
            {
                "path": weather_path.as_posix(),
                "rows": 1,
                "bytes": weather_path.stat().st_size,
                "sha256": sha256_file(weather_path),
            }
        ]
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_weather_artifact(manifest_path)
    assert result["outputs"][0]["checks"]["rows"] == 1


def test_weather_artifact_allows_local_year_utc_rollover(tmp_path: Path) -> None:
    row: dict[str, object] = {
        "Airport": "HNL",
        "valid_time_utc": pd.Timestamp("2026-01-01 09:00"),
        "issue_time_utc": pd.Timestamp("2025-12-31 09:00"),
        "lead_hours": 24.0,
    }
    row.update({name: 1.0 for name in OPEN_METEO_VARIABLES})
    weather_path = tmp_path / "year=2025.parquet"
    pd.DataFrame([row]).to_parquet(weather_path, index=False)
    manifest: dict[str, object] = {
        "outputs": [
            {
                "path": weather_path.as_posix(),
                "rows": 1,
                "bytes": weather_path.stat().st_size,
                "sha256": sha256_file(weather_path),
            }
        ]
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_weather_artifact(manifest_path)
    assert result["outputs"][0]["partition_local_year"] == 2025
    assert result["outputs"][0]["utc_years"] == [2026]


def _write_manifest(path: Path, payload: dict[str, object]) -> Path:
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_feature_artifact_recomputes_schema_cutoff_and_coverage(tmp_path: Path) -> None:
    values = {feature: 1.0 for feature in FLARE24_MATERIALIZED_FEATURES}
    feature_path = tmp_path / "features.parquet"
    pd.DataFrame([{"sample_id": "a", **values}]).to_parquet(feature_path, index=False)
    manifest = {
        "rows": 1,
        "features": list(FLARE24_MATERIALIZED_FEATURES),
        "feature_nonmissing_fraction": {
            feature: 1.0 for feature in FLARE24_MATERIALIZED_FEATURES
        },
        "outputs": [
            {
                "path": feature_path.as_posix(),
                "rows": 1,
                "bytes": feature_path.stat().st_size,
                "sha256": sha256_file(feature_path),
            }
        ],
        "schedule_inputs": [],
        "weather_manifest": {
            "path": feature_path.as_posix(),
            "sha256": sha256_file(feature_path),
        },
        "airport_catalog": {
            "path": feature_path.as_posix(),
            "sha256": sha256_file(feature_path),
        },
        "runway_catalog": {
            "path": feature_path.as_posix(),
            "sha256": sha256_file(feature_path),
        },
        "outcome_columns_read": [],
        "confirmation_outcomes_accessed": False,
    }
    result = validate_feature_artifact(
        _write_manifest(tmp_path / "feature-manifest.json", manifest)
    )
    assert result["cutoff_coherent_rows"] == 1


def test_rotation_artifact_rejects_capacity_violation(tmp_path: Path) -> None:
    values = {feature: 0.0 for feature in FLARE24_ROTATION_FEATURES}
    values["flare24_rotation_inbound_disruption_risk"] = float("nan")
    rotation_path = tmp_path / "rotation.parquet"
    pd.DataFrame([{"sample_id": "a", **values}]).to_parquet(rotation_path, index=False)
    record = {
        "path": rotation_path.as_posix(),
        "rows": 1,
        "bytes": rotation_path.stat().st_size,
        "sha256": sha256_file(rotation_path),
        "diagnostics": [
            {
                "target_outcomes_accessed": False,
                "maximum_inbound_mass": 1.1,
                "capacity_acceptance_tolerance": 1e-8,
                "emitted_rows": 1,
            }
        ],
    }
    manifest = {
        "rows": 1,
        "features": list(FLARE24_ROTATION_FEATURES),
        "outputs": [record],
        "models": [],
        "source_files": [],
        "airport_catalog": {
            "path": rotation_path.as_posix(),
            "sha256": sha256_file(rotation_path),
        },
        "target_tail_number_read": False,
        "target_outcome_columns_read": [],
    }
    manifest_path = _write_manifest(tmp_path / "rotation-manifest.json", manifest)
    with pytest.raises(ValueError, match="inbound capacity"):
        validate_rotation_artifact(manifest_path)


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _selection_prediction(
    *,
    year: int,
    month: int,
    joint: bool,
) -> pd.DataFrame:
    row: dict[str, object] = {
        "sample_id": f"{year}-{month:02d}-sample",
        "FlightDate": pd.Timestamp(year=year, month=month, day=20),
        "Year": year,
        "Month": month,
    }
    for candidate in ("baseline", "weather", "rotation_structural", "rotation_risk"):
        row[f"raw_{candidate}_delay"] = 0.2
        row[f"raw_{candidate}_cancellation"] = 0.1
    if joint:
        for method in (
            "baseline",
            "weather",
            "rotation_structural",
            "rotation_risk",
            "ensemble",
            "reconciled",
        ):
            row[f"prob_{method}_on_time"] = 0.72
            row[f"prob_{method}_delayed"] = 0.18
            row[f"prob_{method}_cancelled"] = 0.10
    return pd.DataFrame([row])


def test_selection_and_method_lock_validation_reopen_artifacts(tmp_path: Path) -> None:
    input_manifest = _write_manifest(tmp_path / "input.json", {"status": "COMPLETE"})
    input_record = {
        **_file_record(input_manifest),
        "self_hash_key": "manifest_sha256",
        "self_hash": json.loads(input_manifest.read_text(encoding="utf-8"))[
            "manifest_sha256"
        ],
    }
    model_records: list[dict[str, object]] = []
    calibrator_records: list[dict[str, object]] = []
    for task in ("delay", "cancellation"):
        for candidate in (
            "baseline",
            "weather",
            "rotation_structural",
            "rotation_risk",
        ):
            model_path = tmp_path / f"{candidate}-{task}-model.bin"
            model_path.write_bytes(b"model")
            model_records.append(
                {"candidate": candidate, "task": task, **_file_record(model_path)}
            )
            calibrator_path = tmp_path / f"{candidate}-{task}-calibrator.bin"
            calibrator_path.write_bytes(b"calibrator")
            calibrator_records.append(
                {"candidate": candidate, "task": task, **_file_record(calibrator_path)}
            )

    raw_records: list[dict[str, object]] = []
    for month in (10, 11, 12):
        path = tmp_path / f"raw-{month}.parquet"
        frame = _selection_prediction(year=2024, month=month, joint=False)
        frame.to_parquet(path, index=False)
        raw_records.append(
            {"year": 2024, "month": month, "rows": len(frame), **_file_record(path)}
        )
    crossfit_path = tmp_path / "crossfit.parquet"
    crossfit = pd.concat(
        [_selection_prediction(year=2024, month=month, joint=True) for month in (10, 11, 12)],
        ignore_index=True,
    )
    crossfit.to_parquet(crossfit_path, index=False)

    aggregate_model_path = tmp_path / "aggregate.bin"
    aggregate_model_path.write_bytes(b"aggregate")
    aggregate_path = tmp_path / "aggregate.parquet"
    pd.DataFrame(
        [
            {
                "FlightDate": pd.Timestamp("2024-10-20"),
                "group_type": "carrier_day",
                "scheduled_count": 1,
                "mean_on_time": 0.7,
                "mean_delayed": 0.2,
                "mean_cancelled": 0.1,
                "variance_on_time": 0.2,
                "variance_delayed": 0.2,
                "variance_cancelled": 0.1,
            }
        ]
    ).to_parquet(aggregate_path, index=False)
    weights = {
        "baseline": 1.0,
        "weather": 0.0,
        "rotation_structural": 0.0,
        "rotation_risk": 0.0,
    }
    report: dict[str, object] = {
        "status": "COMPLETE_2024_FLARE24_SELECTION_NOT_CONFIRMATORY",
        "input_artifacts": [input_record],
        "feature_sets": {candidate: [] for candidate in weights},
        "model_artifacts": model_records,
        "calibration": {"artifacts": calibrator_records},
        "raw_prediction_artifacts": raw_records,
        "crossfit_prediction_artifact": {
            "rows": len(crossfit),
            **_file_record(crossfit_path),
        },
        "aggregate_model": _file_record(aggregate_model_path),
        "aggregate_forecasts": {
            "rows": 1,
            **_file_record(aggregate_path),
        },
        "ensemble_selection": {"selected": {"weights": weights}},
        "reconciliation_selection": {
            "identity_candidate_included": True,
            "selected_mode": "soft_marginal_alignment",
            "candidates": [
                {
                    "mode": "identity_no_alignment",
                    "variance_multiplier": None,
                    "joint_log_loss": 0.5,
                },
                {
                    "mode": "soft_marginal_alignment",
                    "variance_multiplier": 1.0,
                    "joint_log_loss": 0.4,
                },
            ],
        },
        "selected_variance_multiplier": 1.0,
        "reconciliation_diagnostics": [
            {
                "FlightDate": f"2024-{month:02d}-20",
                "converged": True,
                "objective_before": 1.0,
                "objective_after": 0.9,
                "maximum_simplex_error": 1e-16,
                "maximum_absolute_dual_gradient": 1e-6,
                "rms_dual_gradient": 1e-7,
                "dual_preconditioner_minimum": 0.1,
                "dual_preconditioner_maximum": 10.0,
            }
            for month in (10, 11, 12)
        ],
        "diagnostic_weather_severity": {
            "selection_quartile_cutpoints": [0.1, 0.2, 0.3]
        },
        "outcomes_accessed": {
            "years": [2024],
            "maximum_calendar_date": "2024-12-31",
            "2025_accessed_by_this_run": False,
            "2026_accessed": False,
        },
        "confirmation_gate": {"year": 2026, "opened": False},
    }
    report["report_sha256"] = canonical_json_sha256(report)
    report_path = tmp_path / "selection.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = validate_study_artifact(report_path)
    assert len(result["raw_prediction_artifacts"]) == 3

    from flightdelaybench.flare_study import write_flare24_method_lock

    lock_path = tmp_path / "lock.json"
    write_flare24_method_lock(report_path, output_path=lock_path)
    lock_result = validate_method_lock_artifact(lock_path)
    assert lock_result["model_artifacts"] == 8
    assert lock_result["confirmation_gate_opened"] is False


def test_publication_bundle_validation_reopens_tables_and_figures(tmp_path: Path) -> None:
    selection: dict[str, object] = {
        "status": "COMPLETE_2024_FLARE24_SELECTION_NOT_CONFIRMATORY"
    }
    selection["report_sha256"] = canonical_json_sha256(selection)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    audit: dict[str, object] = {
        "status": "COMPLETE_2025_FLARE24_RETROSPECTIVE_AUDIT_NOT_BLIND_CONFIRMATION"
    }
    audit["report_sha256"] = canonical_json_sha256(audit)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    artifacts: list[dict[str, object]] = []
    for index in range(6):
        table_path = tmp_path / f"table-{index}.csv"
        pd.DataFrame({"value": [index]}).to_csv(table_path, index=False)
        artifacts.append(
            {
                "kind": "table",
                "rows": 1,
                **_file_record(table_path),
            }
        )
    for stem in range(3):
        for file_format, content in (("png", b"\x89PNG\r\n\x1a\n"), ("pdf", b"%PDF-1.7")):
            figure_path = tmp_path / f"figure-{stem}.{file_format}"
            figure_path.write_bytes(content)
            artifacts.append(
                {
                    "kind": "figure",
                    "stem": f"figure-{stem}",
                    "format": file_format,
                    **_file_record(figure_path),
                }
            )
    manifest: dict[str, object] = {
        "status": "COMPLETE_FLARE24_PUBLICATION_BUNDLE",
        "confirmation_outcomes_accessed": False,
        "source_reports": [_file_record(selection_path), _file_record(audit_path)],
        "artifacts": artifacts,
    }
    manifest_path = _write_manifest(tmp_path / "publication.json", manifest)

    result = validate_publication_bundle_artifact(manifest_path)

    assert result["source_reports"] == 2
    assert len(result["artifacts"]) == 12
