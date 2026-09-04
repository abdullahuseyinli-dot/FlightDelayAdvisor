"""Independent artifact validation for FLARE-24 research assets."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import FLARE24_ROTATION_FEATURES
from .flare_aggregate import STATE_NAMES
from .flare_features import FLARE24_MATERIALIZED_FEATURES
from .flare_study import CANDIDATES, TASKS, _verify_manifest_output_root
from .flare_weather import validate_weather_cube
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance


def _verified_payload(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("manifest_sha256")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"manifest self-hash failed: {path}")
    return payload


def _verified_study_payload(path: Path) -> tuple[dict[str, Any], str]:
    """Verify either a report or manifest self-hash without trusting its label."""

    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    keys = [key for key in ("report_sha256", "manifest_sha256") if key in payload]
    if len(keys) != 1:
        raise ValueError(f"study artifact must contain exactly one self-hash: {path}")
    key = keys[0]
    recorded = payload[key]
    body = {name: value for name, value in payload.items() if name != key}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"study artifact self-hash failed: {path}")
    return payload, key


def validate_weather_artifact(manifest_path: Path) -> dict[str, Any]:
    """Recompute hashes and temporal invariants for a weather-cube manifest."""

    manifest = _verified_payload(manifest_path)
    records: list[dict[str, Any]] = []
    for output in manifest.get("outputs", []):
        path = Path(output["path"])
        if not path.is_file():
            raise FileNotFoundError(f"weather output is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != output["sha256"]:
            raise ValueError(f"weather output checksum failed: {path}")
        frame = pd.read_parquet(path)
        checks = validate_weather_cube(frame)
        if int(output["rows"]) != len(frame):
            raise ValueError(f"weather output row count failed: {path}")
        valid_times = pd.to_datetime(frame["valid_time_utc"], errors="raise")
        years = valid_times.dt.year.unique()
        partition_match = re.search(r"year=(\d{4})", path.as_posix())
        if partition_match:
            partition_year = int(partition_match.group(1))
            earliest = pd.Timestamp(f"{partition_year}-01-01") - pd.Timedelta(hours=15)
            latest = pd.Timestamp(f"{partition_year + 1}-01-01") + pd.Timedelta(hours=15)
            if valid_times.min() < earliest or valid_times.max() > latest:
                raise ValueError(f"weather UTC timestamps escape local-year tolerance: {path}")
        else:
            partition_year = None
        records.append(
            {
                "path": path.as_posix(),
                "sha256": actual_hash,
                "bytes": path.stat().st_size,
                "checks": checks,
                "partition_local_year": partition_year,
                "utc_years": sorted(int(year) for year in years),
            }
        )
    if not records:
        raise ValueError("weather artifact manifest contains no outputs")
    return {
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_self_hash": manifest["manifest_sha256"],
        "outputs": records,
    }


def _verified_file(record: dict[str, Any], *, role: str) -> Path:
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{role} is missing: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != record["sha256"]:
        raise ValueError(f"{role} checksum failed: {path}")
    if "bytes" in record and int(record["bytes"]) != path.stat().st_size:
        raise ValueError(f"{role} byte count failed: {path}")
    return path


def _verify_study_reference(record: dict[str, Any], *, role: str) -> Path:
    path = _verified_file(record, role=role)
    payload, key = _verified_study_payload(path)
    recorded_key = record.get("self_hash_key")
    recorded_hash = record.get("self_hash")
    if recorded_key is not None and recorded_key != key:
        raise ValueError(f"{role} self-hash key failed: {path}")
    if recorded_hash is not None and recorded_hash != payload[key]:
        raise ValueError(f"{role} self-hash value failed: {path}")
    return path


def _required_candidate_tasks(records: list[dict[str, Any]], *, role: str) -> None:
    observed = [(str(record.get("candidate")), str(record.get("task"))) for record in records]
    expected = [(candidate, task) for task in TASKS for candidate in CANDIDATES]
    if sorted(observed) != sorted(expected):
        raise ValueError(f"{role} does not contain exactly one artifact per candidate/task")


def _validate_probability_matrix(
    frame: pd.DataFrame,
    *,
    method: str,
    prefix: str = "prob",
) -> None:
    columns = [f"{prefix}_{method}_{state}" for state in STATE_NAMES]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"FLARE-24 {method} probability columns are missing: {missing}")
    values = frame.loc[:, columns].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(values).all()
        or (values < 0.0).any()
        or (values > 1.0).any()
        or not np.allclose(values.sum(axis=1), 1.0, atol=2e-6, rtol=0.0)
    ):
        raise ValueError(f"FLARE-24 {method} probabilities leave the simplex")


def _validate_binary_probabilities(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"FLARE-24 binary probability columns are missing: {missing}")
    values = frame.loc[:, columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("FLARE-24 binary probabilities fall outside [0, 1]")


def _validate_prediction_partition(
    record: dict[str, Any],
    *,
    role: str,
    expected_year: int,
    expected_month: int | None,
    joint_methods: tuple[str, ...],
    binary_prefixes: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _verified_file(record, role=role)
    frame = pd.read_parquet(path)
    if len(frame) != int(record.get("rows", -1)):
        raise ValueError(f"{role} row count failed: {path}")
    required = {"sample_id", "FlightDate", "Year", "Month"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{role} identifier schema failed: {path}")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError(f"{role} sample_id invariant failed: {path}")
    dates = pd.to_datetime(frame["FlightDate"], errors="raise")
    if not frame["Year"].eq(expected_year).all() or not dates.dt.year.eq(expected_year).all():
        raise ValueError(f"{role} year boundary failed: {path}")
    if expected_month is not None and (
        not frame["Month"].eq(expected_month).all()
        or not dates.dt.month.eq(expected_month).all()
    ):
        raise ValueError(f"{role} month boundary failed: {path}")
    for method in joint_methods:
        _validate_probability_matrix(frame, method=method)
    binary_columns = [
        f"{prefix}_{candidate}_{task}"
        for prefix in binary_prefixes
        for candidate in CANDIDATES
        for task in TASKS
    ]
    if binary_columns:
        _validate_binary_probabilities(frame, binary_columns)
    return frame, {
        "path": path.as_posix(),
        "sha256": record["sha256"],
        "bytes": path.stat().st_size,
        "rows": len(frame),
        "minimum_date": dates.min().date().isoformat(),
        "maximum_date": dates.max().date().isoformat(),
    }


def _validate_aggregate_forecast(record: dict[str, Any], *, role: str) -> dict[str, Any]:
    path = _verified_file(record, role=role)
    frame = pd.read_parquet(path)
    if "rows" in record and len(frame) != int(record["rows"]):
        raise ValueError(f"{role} row count failed: {path}")
    required = {
        "FlightDate",
        "group_type",
        "scheduled_count",
        *(f"mean_{state}" for state in STATE_NAMES),
        *(f"variance_{state}" for state in STATE_NAMES),
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"{role} schema failed: {path}; missing={missing}")
    numeric_columns = [
        "scheduled_count",
        *(f"mean_{state}" for state in STATE_NAMES),
        *(f"variance_{state}" for state in STATE_NAMES),
    ]
    numeric = frame.loc[:, numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or (numeric < 0.0).any():
        raise ValueError(f"{role} numeric invariant failed: {path}")
    means = frame.loc[:, [f"mean_{state}" for state in STATE_NAMES]].sum(axis=1)
    if not np.allclose(
        means.to_numpy(dtype=np.float64),
        frame["scheduled_count"].to_numpy(dtype=np.float64),
        atol=1e-7,
        rtol=0.0,
    ):
        raise ValueError(f"{role} state means do not sum to scheduled counts: {path}")
    return {
        "path": path.as_posix(),
        "sha256": record["sha256"],
        "bytes": path.stat().st_size,
        "rows": len(frame),
    }


def _validate_reconciliation_diagnostics(
    records: list[dict[str, Any]],
    *,
    enabled: bool,
    expected_dates: int,
) -> None:
    if not enabled:
        if records:
            raise ValueError("disabled FLARE-24 reconciliation has solver diagnostics")
        return
    if len(records) != expected_dates:
        raise ValueError("FLARE-24 reconciliation diagnostic date coverage failed")
    dates = [str(record.get("FlightDate")) for record in records]
    if len(set(dates)) != expected_dates:
        raise ValueError("FLARE-24 reconciliation diagnostics repeat a date")
    for record in records:
        numeric = np.asarray(
            [
                record.get("objective_before", np.nan),
                record.get("objective_after", np.nan),
                record.get("maximum_simplex_error", np.nan),
                record.get("maximum_absolute_dual_gradient", np.nan),
                record.get("rms_dual_gradient", np.nan),
                record.get("dual_preconditioner_minimum", np.nan),
                record.get("dual_preconditioner_maximum", np.nan),
            ],
            dtype=np.float64,
        )
        if not np.isfinite(numeric).all() or record.get("converged") is not True:
            raise ValueError("FLARE-24 reconciliation contains a failed solver date")
        if numeric[1] > numeric[0] + 1e-8 or numeric[2] > 1e-8:
            raise ValueError("FLARE-24 reconciliation objective or simplex check failed")
        if numeric[5] <= 0.0 or numeric[6] < numeric[5]:
            raise ValueError("FLARE-24 reconciliation preconditioner diagnostic failed")


def validate_feature_artifact(manifest_path: Path) -> dict[str, Any]:
    """Recompute integrity, schema, cutoff, and coverage checks for FLARE features."""

    manifest = _verified_payload(manifest_path)
    expected_features = list(FLARE24_MATERIALIZED_FEATURES)
    if manifest.get("features") != expected_features:
        raise ValueError("FLARE-24 feature manifest does not match the registered schema")
    if manifest.get("outcome_columns_read") != []:
        raise ValueError("FLARE-24 feature manifest reports outcome-column access")
    if manifest.get("confirmation_outcomes_accessed") is not False:
        raise ValueError("FLARE-24 feature manifest lacks a negative outcome-access declaration")

    records: list[dict[str, Any]] = []
    nonmissing = {feature: 0 for feature in expected_features}
    total_rows = 0
    for output in manifest.get("outputs", []):
        path = _verified_file(output, role="FLARE-24 feature output")
        frame = pd.read_parquet(path)
        expected_columns = ["sample_id", *expected_features]
        if frame.columns.tolist() != expected_columns:
            raise ValueError(f"FLARE-24 feature schema failed: {path}")
        if len(frame) != int(output["rows"]):
            raise ValueError(f"FLARE-24 feature row count failed: {path}")
        if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ValueError(f"FLARE-24 feature sample_id invariant failed: {path}")
        numeric = frame.loc[:, expected_features].to_numpy(dtype=np.float64)
        if np.isinf(numeric).any():
            raise ValueError(f"FLARE-24 feature infinity check failed: {path}")
        if not frame["flare24_cutoff_coherent_valid"].eq(1.0).all():
            raise ValueError(f"FLARE-24 feature cutoff invariant failed: {path}")
        for feature in expected_features:
            nonmissing[feature] += int(frame[feature].notna().sum())
        total_rows += len(frame)
        records.append(
            {
                "path": path.as_posix(),
                "sha256": output["sha256"],
                "bytes": path.stat().st_size,
                "rows": len(frame),
            }
        )
    if not records:
        raise ValueError("FLARE-24 feature manifest contains no outputs")
    if total_rows != int(manifest.get("rows", -1)):
        raise ValueError("FLARE-24 feature manifest total row count failed")
    recorded_coverage = manifest.get("feature_nonmissing_fraction", {})
    for feature, count in nonmissing.items():
        actual = count / total_rows
        if not np.isclose(actual, float(recorded_coverage.get(feature, -1.0)), atol=1e-12):
            raise ValueError(f"FLARE-24 feature coverage failed: {feature}")

    for record in manifest.get("schedule_inputs", []):
        _verified_file(record, role="FLARE-24 schedule input")
    for key in ("weather_manifest", "airport_catalog", "runway_catalog"):
        _verified_file(manifest[key], role=f"FLARE-24 {key}")
    return {
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_self_hash": manifest["manifest_sha256"],
        "rows": total_rows,
        "outputs": records,
        "cutoff_coherent_rows": total_rows,
        "confirmation_outcomes_accessed": False,
    }


def validate_rotation_artifact(manifest_path: Path) -> dict[str, Any]:
    """Recheck schedule-only latent-rotation integrity and capacity invariants."""

    manifest = _verified_payload(manifest_path)
    expected_features = list(FLARE24_ROTATION_FEATURES)
    if manifest.get("features") != expected_features:
        raise ValueError("FLARE-24 rotation manifest does not match the registered schema")
    if manifest.get("target_tail_number_read") is not False:
        raise ValueError("FLARE-24 rotation manifest reports target-tail access")
    if manifest.get("target_outcome_columns_read") != []:
        raise ValueError("FLARE-24 rotation manifest reports target-outcome access")

    records: list[dict[str, Any]] = []
    total_rows = 0
    maximum_inbound_mass = 0.0
    risk_source = str(manifest.get("inbound_risk_source", ""))
    risk_aware = risk_source.startswith("closed-left HMOP")
    for output in manifest.get("outputs", []):
        path = _verified_file(output, role="FLARE-24 rotation output")
        frame = pd.read_parquet(path)
        expected_columns = ["sample_id", *expected_features]
        if frame.columns.tolist() != expected_columns:
            raise ValueError(f"FLARE-24 rotation schema failed: {path}")
        if len(frame) != int(output["rows"]):
            raise ValueError(f"FLARE-24 rotation row count failed: {path}")
        if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ValueError(f"FLARE-24 rotation sample_id invariant failed: {path}")
        numeric = frame.loc[:, expected_features].to_numpy(dtype=np.float64)
        if np.isinf(numeric).any():
            raise ValueError(f"FLARE-24 rotation infinity check failed: {path}")
        propagated_risk = frame["flare24_rotation_inbound_disruption_risk"]
        if risk_aware:
            connected = frame["flare24_rotation_predecessor_probability"].gt(0.0)
            if not connected.any() or propagated_risk.loc[connected].isna().any():
                raise ValueError(f"risk-aware rotation output omitted propagated risk: {path}")
            observed_risk = propagated_risk.dropna().to_numpy(dtype=np.float64)
            if ((observed_risk < 0.0) | (observed_risk > 1.0)).any():
                raise ValueError(f"risk-aware rotation output has risk outside [0, 1]: {path}")
        elif propagated_risk.notna().any():
            raise ValueError(f"structural rotation output unexpectedly contains risk: {path}")
        emitted_rows = 0
        for diagnostic in output.get("diagnostics", []):
            if diagnostic.get("target_outcomes_accessed") is not False:
                raise ValueError(f"rotation diagnostic reports target-outcome access: {path}")
            tolerance = float(diagnostic.get("capacity_acceptance_tolerance", -1.0))
            mass = float(diagnostic.get("maximum_inbound_mass", np.inf))
            if tolerance < 0.0 or mass > 1.0 + tolerance:
                raise ValueError(f"rotation inbound capacity failed: {path}")
            maximum_inbound_mass = max(maximum_inbound_mass, mass)
            emitted_rows += int(diagnostic.get("emitted_rows", -1))
        if emitted_rows != len(frame):
            raise ValueError(f"rotation diagnostic emitted-row total failed: {path}")
        total_rows += len(frame)
        records.append(
            {
                "path": path.as_posix(),
                "sha256": output["sha256"],
                "bytes": path.stat().st_size,
                "rows": len(frame),
            }
        )
    if not records:
        raise ValueError("FLARE-24 rotation manifest contains no outputs")
    if total_rows != int(manifest.get("rows", -1)):
        raise ValueError("FLARE-24 rotation manifest total row count failed")
    for record in manifest.get("models", []):
        _verified_file(record, role="FLARE-24 rotation model")
    for record in manifest.get("source_files", []):
        _verified_file(record, role="FLARE-24 rotation source")
    for record in manifest.get("recent_lookup_files", []):
        _verified_file(record, role="FLARE-24 rotation recent lookup")
    recent_manifest = manifest.get("recent_feature_manifest")
    if risk_aware:
        if not isinstance(recent_manifest, dict):
            raise ValueError("risk-aware rotation manifest omits recent-feature lineage")
        _verified_file(recent_manifest, role="FLARE-24 rotation recent manifest")
    _verified_file(manifest["airport_catalog"], role="FLARE-24 airport catalog")
    return {
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_self_hash": manifest["manifest_sha256"],
        "rows": total_rows,
        "outputs": records,
        "maximum_inbound_mass": maximum_inbound_mass,
        "target_tail_number_read": False,
        "target_outcome_columns_read": [],
        "inbound_risk_source": risk_source,
    }


def validate_method_lock_artifact(manifest_path: Path) -> dict[str, Any]:
    """Verify the immutable 2024 selection lock and every frozen artifact."""

    manifest, key = _verified_study_payload(manifest_path)
    if key != "manifest_sha256":
        raise ValueError("FLARE-24 method lock must use a manifest self-hash")
    if manifest.get("status") != "LOCKED_FLARE24_METHOD_BEFORE_2025_RETROSPECTIVE_AUDIT":
        raise ValueError("FLARE-24 method lock status is invalid")
    outcomes = manifest.get("outcomes_accessed_by_lock", {})
    if (
        outcomes.get("years") != [2024]
        or outcomes.get("maximum_calendar_date") != "2024-12-31"
        or outcomes.get("2025") is not False
        or outcomes.get("2026") is not False
    ):
        raise ValueError("FLARE-24 method lock outcome boundary is invalid")
    gate = manifest.get("confirmation_gate", {})
    if gate.get("year") != 2026 or gate.get("opened") is not False:
        raise ValueError("FLARE-24 method lock confirmation gate is invalid")
    selection_path = _verify_study_reference(
        manifest["selection_report"], role="FLARE-24 locked selection report"
    )
    selection, _ = _verified_study_payload(selection_path)
    if selection.get("status") != "COMPLETE_2024_FLARE24_SELECTION_NOT_CONFIRMATORY":
        raise ValueError("FLARE-24 method lock references an invalid selection stage")
    choices = manifest.get("frozen_choices", {})
    feature_sets = choices.get("feature_sets", {})
    if set(feature_sets) != set(CANDIDATES):
        raise ValueError("FLARE-24 method lock has invalid candidate feature sets")
    model_records = list(choices.get("model_artifacts", []))
    calibrator_records = list(choices.get("calibrator_artifacts", []))
    _required_candidate_tasks(model_records, role="FLARE-24 frozen models")
    _required_candidate_tasks(calibrator_records, role="FLARE-24 frozen calibrators")
    for record in model_records:
        _verified_file(record, role="FLARE-24 frozen model")
    for record in calibrator_records:
        _verified_file(record, role="FLARE-24 frozen calibrator")
    weights = choices.get("ensemble_weights", {})
    if set(weights) != set(CANDIDATES):
        raise ValueError("FLARE-24 method lock ensemble candidates are invalid")
    weight_values = np.asarray(list(weights.values()), dtype=np.float64)
    if (
        not np.isfinite(weight_values).all()
        or (weight_values < 0.0).any()
        or not np.isclose(weight_values.sum(), 1.0, atol=1e-12)
    ):
        raise ValueError("FLARE-24 method lock ensemble weights are invalid")
    reconciliation_enabled = choices.get("reconciliation_enabled")
    multiplier_value = choices.get("reconciliation_variance_multiplier")
    if reconciliation_enabled is True:
        multiplier = float(multiplier_value)
        if not np.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("FLARE-24 method lock reconciliation multiplier is invalid")
    elif reconciliation_enabled is False:
        if multiplier_value is not None:
            raise ValueError("disabled FLARE-24 reconciliation must not retain a multiplier")
    else:
        raise ValueError("FLARE-24 method lock reconciliation decision is missing")
    cutpoints = np.asarray(choices.get("weather_severity_cutpoints", []), dtype=np.float64)
    if (
        cutpoints.shape != (3,)
        or not np.isfinite(cutpoints).all()
        or (np.diff(cutpoints) < 0.0).any()
    ):
        raise ValueError("FLARE-24 method lock weather cutpoints are invalid")
    expected_choices = {
        "feature_sets": selection.get("feature_sets"),
        "model_artifacts": selection.get("model_artifacts"),
        "calibrator_artifacts": selection.get("calibration", {}).get("artifacts"),
        "ensemble_weights": selection.get("ensemble_selection", {})
        .get("selected", {})
        .get("weights"),
        "reconciliation_enabled": selection.get("selected_variance_multiplier") is not None,
        "reconciliation_variance_multiplier": selection.get(
            "selected_variance_multiplier"
        ),
        "weather_severity_cutpoints": selection.get(
            "diagnostic_weather_severity", {}
        ).get("selection_quartile_cutpoints"),
    }
    if choices != expected_choices:
        raise ValueError("FLARE-24 method lock differs from the selected choices")
    return {
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_self_hash": manifest[key],
        "selection_report": selection_path.as_posix(),
        "model_artifacts": len(model_records),
        "calibrator_artifacts": len(calibrator_records),
        "confirmation_gate_opened": False,
    }


def _validate_selection_report(
    report_path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    outcomes = report.get("outcomes_accessed", {})
    if (
        outcomes.get("years") != [2024]
        or outcomes.get("maximum_calendar_date") != "2024-12-31"
        or outcomes.get("2025_accessed_by_this_run") is not False
        or outcomes.get("2026_accessed") is not False
    ):
        raise ValueError("FLARE-24 selection outcome boundary is invalid")
    gate = report.get("confirmation_gate", {})
    if gate.get("year") != 2026 or gate.get("opened") is not False:
        raise ValueError("FLARE-24 selection confirmation gate is invalid")
    reconciliation = report.get("reconciliation_selection", {})
    candidates = list(reconciliation.get("candidates", []))
    identity_candidates = [
        record for record in candidates if record.get("mode") == "identity_no_alignment"
    ]
    if len(identity_candidates) != 1 or reconciliation.get("identity_candidate_included") is not True:
        raise ValueError("FLARE-24 selection omitted the no-alignment reconciliation control")
    selected_multiplier = report.get("selected_variance_multiplier")
    selected_mode = reconciliation.get("selected_mode")
    if (selected_mode == "identity_no_alignment") != (selected_multiplier is None):
        raise ValueError("FLARE-24 selected reconciliation mode and multiplier disagree")
    for record in report.get("input_artifacts", []):
        if "self_hash_key" in record or "self_hash" in record:
            path = _verify_study_reference(record, role="FLARE-24 selection input")
            if "output_root" in record:
                manifest, _ = _verified_study_payload(path)
                count = _verify_manifest_output_root(
                    manifest,
                    root=Path(str(record["output_root"])),
                    years={int(year) for year in record.get("verified_output_years", [])},
                    role=f"validated {record.get('role', 'selection input')}",
                )
                if count != int(record.get("verified_output_files", -1)):
                    raise ValueError("FLARE-24 selection input verified-file count failed")
        else:
            _verified_file(record, role="FLARE-24 selection input")
    models = list(report.get("model_artifacts", []))
    calibrators = list(report.get("calibration", {}).get("artifacts", []))
    _required_candidate_tasks(models, role="FLARE-24 selected models")
    _required_candidate_tasks(calibrators, role="FLARE-24 selected calibrators")
    for record in models:
        _verified_file(record, role="FLARE-24 selected model")
    for record in calibrators:
        _verified_file(record, role="FLARE-24 selected calibrator")

    raw_records = list(report.get("raw_prediction_artifacts", []))
    observed_months = {(int(record["year"]), int(record["month"])) for record in raw_records}
    if observed_months != {(2024, 10), (2024, 11), (2024, 12)}:
        raise ValueError("FLARE-24 raw selection partitions are incomplete")
    raw_outputs: list[dict[str, Any]] = []
    for record in raw_records:
        frame, output = _validate_prediction_partition(
            record,
            role="FLARE-24 raw selection prediction",
            expected_year=2024,
            expected_month=int(record["month"]),
            joint_methods=(),
            binary_prefixes=("raw",),
        )
        raw_outputs.append(output)
        del frame

    crossfit_record = report["crossfit_prediction_artifact"]
    crossfit, crossfit_output = _validate_prediction_partition(
        crossfit_record,
        role="FLARE-24 cross-fitted selection prediction",
        expected_year=2024,
        expected_month=None,
        joint_methods=(*CANDIDATES, "ensemble", "reconciled"),
    )
    dates = pd.to_datetime(crossfit["FlightDate"], errors="raise")
    if dates.min() < pd.Timestamp("2024-10-16") or dates.max() > pd.Timestamp("2024-12-31"):
        raise ValueError("FLARE-24 cross-fitted selection dates escape the frozen window")
    reconciliation_enabled = selected_multiplier is not None
    _validate_reconciliation_diagnostics(
        list(report.get("reconciliation_diagnostics", [])),
        enabled=reconciliation_enabled,
        expected_dates=int(dates.dt.normalize().nunique()),
    )
    del crossfit
    _verified_file(report["aggregate_model"], role="FLARE-24 selection aggregate model")
    aggregate_output = _validate_aggregate_forecast(
        report["aggregate_forecasts"], role="FLARE-24 selection aggregate forecast"
    )
    return {
        "report": report_path.as_posix(),
        "report_sha256": sha256_file(report_path),
        "status": report["status"],
        "raw_prediction_artifacts": raw_outputs,
        "crossfit_prediction_artifact": crossfit_output,
        "aggregate_forecast_artifact": aggregate_output,
        "model_artifacts": len(models),
        "calibrator_artifacts": len(calibrators),
        "confirmation_gate_opened": False,
    }


def _validate_audit_report(
    report_path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    outcomes = report.get("outcomes_accessed", {})
    if (
        outcomes.get("historical_aggregate_fit_year") != 2024
        or outcomes.get("retrospective_audit_year") != 2025
        or outcomes.get("maximum_calendar_date") != "2025-12-31"
        or outcomes.get("2026_accessed") is not False
    ):
        raise ValueError("FLARE-24 audit outcome boundary is invalid")
    gate = report.get("confirmation_gate", {})
    if gate.get("year") != 2026 or gate.get("opened") is not False:
        raise ValueError("FLARE-24 audit confirmation gate is invalid")
    _verify_study_reference(report["selection_report"], role="FLARE-24 audit selection")
    method_lock_path = _verify_study_reference(
        report["method_lock"], role="FLARE-24 audit method lock"
    )
    validate_method_lock_artifact(method_lock_path)
    method_lock, _ = _verified_study_payload(method_lock_path)

    frozen = report.get("frozen_method", {})
    if frozen.get("model_refit") is not False or frozen.get("calibrator_refit") is not False:
        raise ValueError("FLARE-24 audit reports a forbidden flight-model refit")
    models = list(frozen.get("model_artifacts", []))
    calibrators = list(frozen.get("calibrator_artifacts", []))
    _required_candidate_tasks(models, role="FLARE-24 audited frozen models")
    _required_candidate_tasks(calibrators, role="FLARE-24 audited frozen calibrators")
    for record in models:
        _verified_file(record, role="FLARE-24 audited frozen model")
    for record in calibrators:
        _verified_file(record, role="FLARE-24 audited frozen calibrator")
    locked_choices = method_lock["frozen_choices"]
    if (
        frozen.get("feature_sets") != locked_choices.get("feature_sets")
        or frozen.get("model_artifacts") != locked_choices.get("model_artifacts")
        or frozen.get("calibrator_artifacts")
        != locked_choices.get("calibrator_artifacts")
        or frozen.get("ensemble_weights") != locked_choices.get("ensemble_weights")
        or frozen.get("reconciliation_enabled")
        != locked_choices.get("reconciliation_enabled")
        or frozen.get("reconciliation_variance_multiplier")
        != locked_choices.get("reconciliation_variance_multiplier")
    ):
        raise ValueError("FLARE-24 audit frozen choices differ from the method lock")

    prediction_records = list(report.get("prediction_artifacts", []))
    observed_months = {
        (int(record["year"]), int(record["month"])) for record in prediction_records
    }
    if observed_months != {(2025, month) for month in range(1, 13)}:
        raise ValueError("FLARE-24 audit prediction partitions are incomplete")
    prediction_outputs: list[dict[str, Any]] = []
    total_rows = 0
    for record in prediction_records:
        frame, output = _validate_prediction_partition(
            record,
            role="FLARE-24 audit prediction",
            expected_year=2025,
            expected_month=int(record["month"]),
            joint_methods=(*CANDIDATES, "ensemble", "reconciled"),
            binary_prefixes=("raw", "calibrated"),
        )
        total_rows += len(frame)
        prediction_outputs.append(output)
        del frame
    if total_rows != int(report.get("evaluation_rows", -1)):
        raise ValueError("FLARE-24 audit evaluation row count failed")
    _validate_reconciliation_diagnostics(
        list(report.get("reconciliation_diagnostics", [])),
        enabled=frozen.get("reconciliation_enabled") is True,
        expected_dates=365,
    )

    aggregate_refit = report["aggregate_refit"]
    _verified_file(aggregate_refit, role="FLARE-24 audit aggregate model")
    if aggregate_refit.get("model_card", {}).get("maximum_history_date") != "2024-12-31":
        raise ValueError("FLARE-24 audit aggregate history boundary is invalid")
    aggregate_records = list(report.get("aggregate_forecast_artifacts", []))
    aggregate_months = {
        (int(record["year"]), int(record["month"])) for record in aggregate_records
    }
    if aggregate_months != {(2025, month) for month in range(1, 13)}:
        raise ValueError("FLARE-24 audit aggregate forecast partitions are incomplete")
    aggregate_outputs = [
        _validate_aggregate_forecast(record, role="FLARE-24 audit aggregate forecast")
        for record in aggregate_records
    ]
    return {
        "report": report_path.as_posix(),
        "report_sha256": sha256_file(report_path),
        "status": report["status"],
        "prediction_artifacts": prediction_outputs,
        "aggregate_forecast_artifacts": aggregate_outputs,
        "evaluation_rows": total_rows,
        "confirmation_gate_opened": False,
    }


def validate_study_artifact(report_path: Path) -> dict[str, Any]:
    """Independently validate a completed FLARE-24 selection or audit report."""

    report, key = _verified_study_payload(report_path)
    if key != "report_sha256":
        raise ValueError("FLARE-24 study report must use a report self-hash")
    status = report.get("status")
    if status == "COMPLETE_2024_FLARE24_SELECTION_NOT_CONFIRMATORY":
        result = _validate_selection_report(report_path, report)
    elif status == "COMPLETE_2025_FLARE24_RETROSPECTIVE_AUDIT_NOT_BLIND_CONFIRMATION":
        result = _validate_audit_report(report_path, report)
    else:
        raise ValueError(f"unsupported FLARE-24 study status: {status}")
    result["report_self_hash"] = report[key]
    return result


def validate_publication_bundle_artifact(manifest_path: Path) -> dict[str, Any]:
    """Reopen every table and figure in a FLARE-24 publication bundle."""

    manifest, key = _verified_study_payload(manifest_path)
    if key != "manifest_sha256":
        raise ValueError("FLARE-24 publication bundle must use a manifest self-hash")
    if manifest.get("status") != "COMPLETE_FLARE24_PUBLICATION_BUNDLE":
        raise ValueError("FLARE-24 publication bundle status is invalid")
    if manifest.get("confirmation_outcomes_accessed") is not False:
        raise ValueError("FLARE-24 publication bundle opened confirmation outcomes")
    source_records = list(manifest.get("source_reports", []))
    if len(source_records) != 2:
        raise ValueError("FLARE-24 publication bundle must bind selection and audit reports")
    source_statuses: set[str] = set()
    for record in source_records:
        path = _verified_file(record, role="FLARE-24 publication source report")
        payload, source_key = _verified_study_payload(path)
        if source_key != "report_sha256":
            raise ValueError("FLARE-24 publication source is not a report")
        source_statuses.add(str(payload.get("status")))
    if source_statuses != {
        "COMPLETE_2024_FLARE24_SELECTION_NOT_CONFIRMATORY",
        "COMPLETE_2025_FLARE24_RETROSPECTIVE_AUDIT_NOT_BLIND_CONFIRMATION",
    }:
        raise ValueError("FLARE-24 publication bundle source stages are incomplete")

    artifacts = list(manifest.get("artifacts", []))
    table_records = [record for record in artifacts if record.get("kind") == "table"]
    figure_records = [record for record in artifacts if record.get("kind") == "figure"]
    if len(table_records) != 6 or len(figure_records) != 6 or len(artifacts) != 12:
        raise ValueError("FLARE-24 publication bundle artifact inventory is incomplete")
    paths: set[str] = set()
    validated: list[dict[str, Any]] = []
    for record in artifacts:
        path = _verified_file(record, role="FLARE-24 publication artifact")
        normalized_path = path.resolve().as_posix()
        if normalized_path in paths:
            raise ValueError("FLARE-24 publication bundle repeats an artifact path")
        paths.add(normalized_path)
        if int(record.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"FLARE-24 publication artifact byte count failed: {path}")
        if record.get("kind") == "table":
            frame = pd.read_csv(path)
            if len(frame) != int(record.get("rows", -1)) or frame.empty:
                raise ValueError(f"FLARE-24 publication table row count failed: {path}")
            details: dict[str, Any] = {"rows": len(frame), "columns": list(frame.columns)}
        else:
            file_format = str(record.get("format"))
            prefix = path.read_bytes()[:8]
            if file_format == "png" and prefix != b"\x89PNG\r\n\x1a\n":
                raise ValueError(f"FLARE-24 PNG signature failed: {path}")
            if file_format == "pdf" and not prefix.startswith(b"%PDF-"):
                raise ValueError(f"FLARE-24 PDF signature failed: {path}")
            if file_format not in {"png", "pdf"}:
                raise ValueError(f"unsupported FLARE-24 figure format: {file_format}")
            details = {"format": file_format, "stem": str(record.get("stem"))}
        validated.append(
            {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                **details,
            }
        )
    return {
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_self_hash": manifest[key],
        "source_reports": len(source_records),
        "artifacts": validated,
        "confirmation_outcomes_accessed": False,
    }


def validate_confirmation_lock_artifact(manifest_path: Path) -> dict[str, Any]:
    """Verify the analysis lock that precedes any FLARE-24 2026 outcome access."""

    manifest, key = _verified_study_payload(manifest_path)
    if key != "manifest_sha256":
        raise ValueError("FLARE-24 confirmation lock must use a manifest self-hash")
    if manifest.get("status") != (
        "LOCKED_2026_FLARE24_CONFIRMATION_PROTOCOL_OUTCOMES_UNOPENED"
    ):
        raise ValueError("FLARE-24 confirmation lock status is invalid")
    outcomes = manifest.get("outcomes_accessed_at_lock", {})
    if (
        outcomes.get("years") != [2024, 2025]
        or outcomes.get("maximum_calendar_date") != "2025-12-31"
        or outcomes.get("2026") is not False
    ):
        raise ValueError("FLARE-24 confirmation lock outcome boundary is invalid")
    gate = manifest.get("confirmation_gate", {})
    if gate.get("year") != 2026 or gate.get("opened") is not False:
        raise ValueError("FLARE-24 confirmation gate was opened by its lock")
    population = manifest.get("confirmation_population", {})
    if population.get("calendar_window") != ["2026-01-01", "2026-06-30"]:
        raise ValueError("FLARE-24 confirmation window is invalid")
    analysis = manifest.get("analysis", {})
    if (
        analysis.get("reference_method") != "baseline"
        or analysis.get("candidate_method") != "reconciled"
        or analysis.get("co_primary_metrics")
        != ["joint_log_loss", "multiclass_brier"]
        or analysis.get("cluster_key") != "FlightDate"
        or int(analysis.get("bootstrap_repetitions", 0)) != 2000
        or int(analysis.get("bootstrap_seed", -1)) != 20260903
        or float(analysis.get("confidence", 0.0)) != 0.95
    ):
        raise ValueError("FLARE-24 confirmation analysis differs from the frozen protocol")
    protocol_path = _verified_file(manifest["protocol"], role="FLARE-24 frozen protocol")
    if protocol_path.as_posix() != "configs/flare24_v1.toml":
        raise ValueError("FLARE-24 confirmation lock references an unexpected protocol")

    artifact_records = list(manifest.get("locked_artifacts", []))
    by_role = {str(record.get("role")): record for record in artifact_records}
    expected_roles = {
        "2024_selection_report",
        "pre_2025_method_lock",
        "2025_retrospective_audit",
        "retrospective_publication_bundle",
    }
    if set(by_role) != expected_roles or len(artifact_records) != len(expected_roles):
        raise ValueError("FLARE-24 confirmation lock artifact set is incomplete")
    paths = {
        role: _verify_study_reference(record, role=f"FLARE-24 confirmation {role}")
        for role, record in by_role.items()
    }
    selection_result = validate_study_artifact(paths["2024_selection_report"])
    method_result = validate_method_lock_artifact(paths["pre_2025_method_lock"])
    audit_result = validate_study_artifact(paths["2025_retrospective_audit"])
    publication_result = validate_publication_bundle_artifact(
        paths["retrospective_publication_bundle"]
    )
    method_payload, _ = _verified_study_payload(paths["pre_2025_method_lock"])
    audit_payload, _ = _verified_study_payload(paths["2025_retrospective_audit"])
    if manifest.get("frozen_method") != method_payload.get("frozen_choices"):
        raise ValueError("FLARE-24 confirmation method differs from the pre-audit lock")
    if audit_payload.get("method_lock", {}).get("sha256") != sha256_file(
        paths["pre_2025_method_lock"]
    ):
        raise ValueError("FLARE-24 confirmation audit and method lock are not linked")
    source_records = list(manifest.get("provenance", {}).get("source_files", []))
    if not source_records:
        raise ValueError("FLARE-24 confirmation lock has no source provenance")
    for record in source_records:
        _verified_file(record, role="FLARE-24 confirmation source")
    return {
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_self_hash": manifest[key],
        "selection_status": selection_result["status"],
        "method_lock_status": method_payload["status"],
        "audit_status": audit_result["status"],
        "publication_artifacts": len(publication_result["artifacts"]),
        "frozen_model_artifacts": method_result["model_artifacts"],
        "confirmation_window": population["calendar_window"],
        "confirmation_gate_opened": False,
    }


def validate_nested_ablation_artifact(report_path: Path) -> dict[str, Any]:
    """Verify the paired incremental FLARE-24 ablation supplement."""

    report, key = _verified_study_payload(report_path)
    if key != "report_sha256":
        raise ValueError("FLARE-24 nested ablation must use a report self-hash")
    if report.get("status") != "COMPLETE_2025_FLARE24_NESTED_ABLATION_ANALYSIS":
        raise ValueError("FLARE-24 nested ablation status is invalid")
    if report.get("confirmation_outcomes_accessed") is not False:
        raise ValueError("FLARE-24 nested ablation opened confirmation outcomes")
    audit_path = _verify_study_reference(
        report["source_audit"], role="FLARE-24 nested-ablation audit"
    )
    audit_result = validate_study_artifact(audit_path)
    prediction_records = list(report.get("source_predictions", []))
    months = {(int(record["year"]), int(record["month"])) for record in prediction_records}
    if months != {(2025, month) for month in range(1, 13)}:
        raise ValueError("FLARE-24 nested-ablation predictions are incomplete")
    total_rows = 0
    for record in prediction_records:
        path = _verified_file(record, role="FLARE-24 nested-ablation prediction")
        dates = pd.to_datetime(pd.read_parquet(path, columns=["FlightDate"])["FlightDate"])
        if len(dates) != int(record["rows"]):
            raise ValueError(f"FLARE-24 nested-ablation row count failed: {path}")
        total_rows += len(dates)
    table_record = report["table_artifact"]
    table_path = _verified_file(table_record, role="FLARE-24 nested-ablation table")
    table = pd.read_csv(table_path)
    if len(table) != int(table_record["rows"]) or len(table) != 30:
        raise ValueError("FLARE-24 nested-ablation table is incomplete")
    expected_comparisons = [
        ("weather", "baseline"),
        ("rotation_structural", "weather"),
        ("rotation_risk", "rotation_structural"),
        ("ensemble", "rotation_structural"),
        ("reconciled", "ensemble"),
    ]
    observed_comparisons = [
        (str(record.get("candidate")), str(record.get("reference")))
        for record in report.get("comparisons", [])
    ]
    if observed_comparisons != expected_comparisons:
        raise ValueError("FLARE-24 nested comparison order changed")
    numeric = table.loc[:, ["estimate", "lower", "upper", "confidence"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all() or not (table["clusters"] == 365).all():
        raise ValueError("FLARE-24 nested-ablation intervals are invalid")
    if not (table["repetitions"] == 2000).all() or not (table["seed"] == 20270903).all():
        raise ValueError("FLARE-24 nested-ablation bootstrap protocol changed")
    normalization = report.get("probability_normalization_audit", {})
    if set(normalization) != {*CANDIDATES, "ensemble", "reconciled"}:
        raise ValueError("FLARE-24 nested-ablation normalization audit is incomplete")
    if any(
        int(record.get("rows_over_recovery_1e_6_absolute_tolerance", -1)) != 0
        for record in normalization.values()
    ):
        raise ValueError("FLARE-24 nested-ablation normalization exceeded its bound")
    return {
        "report": report_path.as_posix(),
        "report_sha256": sha256_file(report_path),
        "report_self_hash": report[key],
        "audit_status": audit_result["status"],
        "prediction_partitions": len(prediction_records),
        "prediction_rows": total_rows,
        "comparison_rows": len(table),
        "confirmation_outcomes_accessed": False,
    }


def write_weather_validation_report(
    *,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite FLARE validation report: {output_path}")
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS_FLARE24_WEATHER_CUBE_VALIDATION",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "weather_artifact": validate_weather_artifact(manifest_path),
        "confirmation_outcomes_accessed": False,
        "provenance": capture_provenance(
            (
                Path(__file__),
                Path(__file__).with_name("flare_weather.py"),
                Path(__file__).with_name("contracts.py"),
            )
        ),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(output_path, report)
    return report


def write_asset_validation_report(
    *,
    output_path: Path,
    weather_manifest_path: Path | None = None,
    feature_manifest_path: Path | None = None,
    rotation_manifest_path: Path | None = None,
    method_lock_path: Path | None = None,
    study_report_path: Path | None = None,
    publication_bundle_path: Path | None = None,
    confirmation_lock_path: Path | None = None,
    nested_ablation_path: Path | None = None,
) -> dict[str, Any]:
    """Validate one or more FLARE-24 assets into one immutable acceptance report."""

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite FLARE validation report: {output_path}")
    if all(
        path is None
        for path in (
            weather_manifest_path,
            feature_manifest_path,
            rotation_manifest_path,
            method_lock_path,
            study_report_path,
            publication_bundle_path,
            confirmation_lock_path,
            nested_ablation_path,
        )
    ):
        raise ValueError("at least one FLARE-24 artifact manifest is required")
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS_FLARE24_ASSET_VALIDATION",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "confirmation_outcomes_accessed": False,
    }
    if weather_manifest_path is not None:
        report["weather_artifact"] = validate_weather_artifact(weather_manifest_path)
    if feature_manifest_path is not None:
        report["feature_artifact"] = validate_feature_artifact(feature_manifest_path)
    if rotation_manifest_path is not None:
        report["rotation_artifact"] = validate_rotation_artifact(rotation_manifest_path)
    if method_lock_path is not None:
        report["method_lock_artifact"] = validate_method_lock_artifact(method_lock_path)
    if study_report_path is not None:
        report["study_artifact"] = validate_study_artifact(study_report_path)
    if publication_bundle_path is not None:
        report["publication_bundle_artifact"] = validate_publication_bundle_artifact(
            publication_bundle_path
        )
    if confirmation_lock_path is not None:
        report["confirmation_lock_artifact"] = validate_confirmation_lock_artifact(
            confirmation_lock_path
        )
    if nested_ablation_path is not None:
        report["nested_ablation_artifact"] = validate_nested_ablation_artifact(
            nested_ablation_path
        )
    report["provenance"] = capture_provenance(
        (
            Path(__file__),
            Path(__file__).with_name("flare_weather.py"),
            Path(__file__).with_name("flare_features.py"),
            Path(__file__).with_name("flare_rotation.py"),
            Path(__file__).with_name("flare_study.py"),
            Path(__file__).with_name("contracts.py"),
        )
    )
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-manifest", type=Path)
    parser.add_argument("--feature-manifest", type=Path)
    parser.add_argument("--rotation-manifest", type=Path)
    parser.add_argument("--method-lock", type=Path)
    parser.add_argument("--study-report", type=Path)
    parser.add_argument("--publication-bundle", type=Path)
    parser.add_argument("--confirmation-lock", type=Path)
    parser.add_argument("--nested-ablation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = write_asset_validation_report(
        output_path=args.output,
        weather_manifest_path=args.weather_manifest,
        feature_manifest_path=args.feature_manifest,
        rotation_manifest_path=args.rotation_manifest,
        method_lock_path=args.method_lock,
        study_report_path=args.study_report,
        publication_bundle_path=args.publication_bundle,
        confirmation_lock_path=args.confirmation_lock,
        nested_ablation_path=args.nested_ablation,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
