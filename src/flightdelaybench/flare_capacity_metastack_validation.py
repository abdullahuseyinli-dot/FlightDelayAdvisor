"""Independent artifact and numerical validation for TF-CC-RTH-LogitStack-v1."""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression

from .flare_capacity_factorized import (
    _load_prediction_frame,
    _probability_methods,
    _verify_parent_inputs,
)
from .flare_capacity_metastack import (
    CONFIRMATION_YEAR,
    EXPECTED_FACTORIZED_STATUS,
    EXPECTED_FACTORIZED_VALIDATION_STATUS,
    H2_EMBARGO,
    HELD_FORWARD_END,
    HELD_FORWARD_START,
    META_METHOD,
    META_PROBABILITY_COLUMN,
    SELECTION_END,
    SELECTION_START,
    _c_key,
    _frozen_metastack_implementation_files,
    _grid_point_scores,
    _load_protocol,
    _metastack_joint,
    fit_cancellation_metastack,
)
from .flare_capacity_study import (
    BLEND_CANDIDATES,
    GATING_FEATURE,
    PRIMARY_AUDIT_END,
    PRIMARY_AUDIT_START,
    _airport_scores,
    _get_joint_columns,
    _joint_columns,
    _joint_point_scores,
    _monthly_scores,
    _read_self_hashed,
    _regime_scores,
    _verified_artifact,
)
from .flare_evaluation import evaluate_joint_probabilities
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

EXPECTED_STATUS = "COMPLETE_TF_CCRTH_METASTACK_POST_HOC_2025_2026_UNOPENED"
EXPECTED_GRID_LOCK_STATUS = "LOCKED_METASTACK_CANDIDATE_GRID_BEFORE_FORMAL_2025_REANALYSIS"
EXPECTED_H2_LOCK_STATUS = "LOCKED_METASTACK_AFTER_H1_BEFORE_FORMAL_H2_PARTITION_LOAD"
EXPECTED_CONFIRMATION_LOCK_STATUS = "LOCKED_TF_CCRTH_LOGIT_STACK_FOR_UNOPENED_2026_CONFIRMATION"
POINT_TOLERANCE = 2e-11
COEFFICIENT_TOLERANCE = 2e-10
PREDICTION_TOLERANCE = 2e-7


def _assert_close(actual: float, expected: float, *, role: str) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=POINT_TOLERANCE):
        raise ValueError(f"{role} differs: observed={actual}, expected={expected}")


def _assert_nested_close(actual: Any, expected: Any, *, role: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"{role} mapping fields differ")
        for key, value in expected.items():
            _assert_nested_close(actual[key], value, role=f"{role}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{role} list shape differs")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _assert_nested_close(left, right, role=f"{role}[{index}]")
        return
    if isinstance(expected, float):
        _assert_close(actual, expected, role=role)
        return
    if actual != expected:
        raise ValueError(f"{role} differs: observed={actual!r}, expected={expected!r}")


def _verify_factorized_context(
    report_path: Path,
    validation_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report, report_key = _read_self_hashed(report_path)
    validation, validation_key = _read_self_hashed(validation_path)
    if (
        report.get("status") != EXPECTED_FACTORIZED_STATUS
        or validation.get("status") != EXPECTED_FACTORIZED_VALIDATION_STATUS
        or report.get("outcomes_accessed", {}).get("2026_accessed") is not False
        or validation.get("confirmation_outcomes_accessed") is not False
    ):
        raise ValueError("meta-stack validator requires validated TF-CC-RTH context")
    bound = validation.get("report", {})
    if (
        Path(str(bound.get("path", ""))).resolve() != report_path.resolve()
        or bound.get("sha256") != sha256_file(report_path)
        or bound.get("self_hash") != report[report_key]
    ):
        raise ValueError("TF-CC-RTH validation does not bind its report")
    return {
        "path": report_path.as_posix(),
        "bytes": report_path.stat().st_size,
        "sha256": sha256_file(report_path),
        "self_hash_key": report_key,
        "self_hash": report[report_key],
    }, {
        "path": validation_path.as_posix(),
        "bytes": validation_path.stat().st_size,
        "sha256": sha256_file(validation_path),
        "self_hash_key": validation_key,
        "self_hash": validation[validation_key],
        "status": validation["status"],
    }


def _verify_lock(
    record: Mapping[str, Any],
    *,
    role: str,
    expected_status: str,
) -> tuple[Path, dict[str, Any], str]:
    path = _verified_artifact(dict(record), role=role)
    payload, hash_key = _read_self_hashed(path)
    if payload.get("status") != expected_status or record.get("self_hash") != payload[hash_key]:
        raise ValueError(f"{role} status or self-hash differs")
    return path, payload, hash_key


def _models_by_c(
    records: Sequence[Mapping[str, Any]],
) -> dict[float, Mapping[str, Any]]:
    result = {float(record["regularization_c"]): record for record in records}
    if len(result) != len(records):
        raise ValueError("meta-stack model records repeat regularization C")
    return result


def _assert_model_reproduced(
    recorded: Mapping[str, Any],
    model: LogisticRegression,
    *,
    regularization_c: float,
) -> Path:
    if (
        float(recorded["regularization_c"]) != regularization_c
        or tuple(recorded["feature_order"]) != BLEND_CANDIDATES
        or int(recorded["iterations"]) != int(model.n_iter_[0])
    ):
        raise ValueError(f"meta-stack C={regularization_c} model metadata differs")
    if not np.isclose(
        float(recorded["intercept"]),
        float(model.intercept_[0]),
        rtol=0.0,
        atol=COEFFICIENT_TOLERANCE,
    ):
        raise ValueError(f"meta-stack C={regularization_c} intercept differs")
    for index, name in enumerate(BLEND_CANDIDATES):
        if not np.isclose(
            float(recorded["coefficients"][name]),
            float(model.coef_[0, index]),
            rtol=0.0,
            atol=COEFFICIENT_TOLERANCE,
        ):
            raise ValueError(f"meta-stack C={regularization_c} coefficient {name} differs")
    artifact_path = _verified_artifact(
        dict(recorded), role=f"meta-stack C={regularization_c} serialized model"
    )
    serialized = joblib.load(artifact_path)
    if not isinstance(serialized, LogisticRegression):
        raise TypeError(f"unexpected serialized meta-stack type: {type(serialized)!r}")
    if not np.allclose(
        serialized.coef_, model.coef_, rtol=0.0, atol=COEFFICIENT_TOLERANCE
    ) or not np.allclose(
        serialized.intercept_, model.intercept_, rtol=0.0, atol=COEFFICIENT_TOLERANCE
    ):
        raise ValueError(f"meta-stack C={regularization_c} serialized model differs")
    return artifact_path


def validate_cancellation_metastack_report(
    report_path: Path,
    *,
    protocol_path: Path,
    parent_report_path: Path,
    parent_validation_path: Path,
    factorized_report_path: Path,
    factorized_validation_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite meta-stack validation: {output_path}")
    report, report_hash_key = _read_self_hashed(report_path)
    if (
        report.get("status") != EXPECTED_STATUS
        or report.get("outcomes_accessed", {}).get("2026_accessed") is not False
    ):
        raise ValueError("meta-stack report is incomplete or crossed the 2026 boundary")
    repetitions = int(report["held_forward_evaluation"]["bootstrap_repetitions"])
    seed = int(report["held_forward_evaluation"]["bootstrap_seed"])
    protocol, protocol_record = _load_protocol(protocol_path, repetitions=repetitions, seed=seed)
    parent_report, parent_lock, parent_record, parent_validation_record = _verify_parent_inputs(
        parent_report_path, parent_validation_path
    )
    factorized_record, factorized_validation_record = _verify_factorized_context(
        factorized_report_path, factorized_validation_path
    )
    for recorded, expected, role in (
        (report["protocol"], protocol_record, "protocol"),
        (report["parent_report"], parent_record, "parent report"),
        (report["parent_validation"], parent_validation_record, "parent validation"),
        (report["factorized_context"], factorized_record, "factorized report"),
        (
            report["factorized_validation"],
            factorized_validation_record,
            "factorized validation",
        ),
    ):
        if recorded != expected:
            raise ValueError(f"meta-stack {role} binding differs")

    current_implementation = capture_provenance(_frozen_metastack_implementation_files())
    for role, provenance in (("report", report["provenance"]),):
        if provenance.get("git_head") != current_implementation.get("git_head") or provenance.get(
            "source_files"
        ) != current_implementation.get("source_files"):
            raise ValueError(f"meta-stack {role} implementation provenance drifted")

    grid_path, grid_lock, grid_hash_key = _verify_lock(
        report["grid_lock"],
        role="meta-stack candidate-grid lock",
        expected_status=EXPECTED_GRID_LOCK_STATUS,
    )
    h2_path, h2_lock, h2_hash_key = _verify_lock(
        report["held_forward_lock"],
        role="meta-stack H2 lock",
        expected_status=EXPECTED_H2_LOCK_STATUS,
    )
    confirmation_path, confirmation_lock, confirmation_hash_key = _verify_lock(
        report["confirmation_lock"],
        role="meta-stack 2026 confirmation lock",
        expected_status=EXPECTED_CONFIRMATION_LOCK_STATUS,
    )
    if (
        grid_lock.get("information_boundary", {}).get("2026_outcomes_accessed") is not False
        or grid_lock.get("information_boundary", {}).get(
            "formal_2025_partitions_loaded_by_this_run"
        )
        is not False
        or h2_lock.get("information_boundary", {}).get("2026_outcomes_accessed") is not False
        or h2_lock.get("information_boundary", {}).get(
            "formal_h2_2025_partitions_loaded_before_lock"
        )
        is not False
        or confirmation_lock.get("information_boundary", {}).get("2026_outcomes_accessed")
        is not False
        or int(confirmation_lock["confirmation_protocol"]["year"]) != CONFIRMATION_YEAR
    ):
        raise ValueError("meta-stack lock information boundary is invalid")
    for role, lock in (("grid", grid_lock), ("H2", h2_lock), ("confirmation", confirmation_lock)):
        provenance = lock.get("provenance", {})
        if provenance.get("git_head") != current_implementation.get("git_head") or provenance.get(
            "source_files"
        ) != current_implementation.get("source_files"):
            raise ValueError(f"meta-stack {role} lock implementation provenance drifted")
    if (
        h2_lock["grid_lock"]["sha256"] != sha256_file(grid_path)
        or h2_lock["grid_lock"]["self_hash"] != grid_lock[grid_hash_key]
        or confirmation_lock["grid_lock"]["sha256"] != sha256_file(grid_path)
        or confirmation_lock["held_forward_lock"]["sha256"] != sha256_file(h2_path)
        or confirmation_lock["held_forward_lock"]["self_hash"] != h2_lock[h2_hash_key]
    ):
        raise ValueError("meta-stack chained lock binding differs")

    q4_frame, q4_input_record = _load_prediction_frame(
        dict(parent_report["selection_crossfit_prediction_artifact"]),
        role="meta-stack validation Q4 source",
        expected_year=2024,
        expected_month=None,
    )
    if report["q4_input"] != q4_input_record:
        raise ValueError("meta-stack Q4 source binding differs")
    q4_dates = pd.to_datetime(q4_frame["FlightDate"], errors="raise")
    q4_mask = q4_dates.between("2024-10-03", "2024-12-31").to_numpy()
    q4_normalization: list[dict[str, Any]] = []
    q4_all = _probability_methods(
        q4_frame,
        normalization_audits=q4_normalization,
        role="meta-stack validation Q4 probabilities",
    )
    q4_base: dict[str, NDArray[np.float64]] = {
        name: q4_all[name][q4_mask] for name in BLEND_CANDIDATES
    }
    q4_fit = q4_frame.loc[q4_mask].reset_index(drop=True)
    q4_labels = q4_fit["Cancelled"].to_numpy(dtype=np.int64)
    design_columns: list[NDArray[np.float64]] = []
    for candidate_name in BLEND_CANDIDATES:
        # Independently reproduce the registered marginal extractor: protect
        # the joint simplex first, then obtain and clip cancellation marginal.
        joint = np.clip(q4_base[candidate_name], 1e-12, 1.0)
        joint /= joint.sum(axis=1, keepdims=True)
        cancellation = np.clip(joint[:, 2], 1e-6, 1.0 - 1e-6)
        design_columns.append(np.log(cancellation / (1.0 - cancellation)))
    q4_design = np.column_stack(design_columns)
    c_grid = tuple(float(value) for value in protocol["meta_model"]["regularization_c_grid"])
    recorded_models = _models_by_c(report["model_artifacts"])
    if set(recorded_models) != set(c_grid):
        raise ValueError("meta-stack model artifact grid differs")
    if grid_lock["model_artifacts"] != report["model_artifacts"]:
        raise ValueError("meta-stack grid-lock model artifact binding differs")
    models: dict[str, LogisticRegression] = {}
    model_paths: list[Path] = []
    q4_predictions: dict[str, NDArray[np.float64]] = {}
    for regularization_c in c_grid:
        model = fit_cancellation_metastack(
            q4_design,
            q4_labels,
            regularization_c=regularization_c,
            maximum_iterations=int(protocol["meta_model"]["maximum_iterations"]),
            tolerance=float(protocol["meta_model"]["tolerance"]),
        )
        model_name = _c_key(regularization_c)
        models[model_name] = model
        model_paths.append(
            _assert_model_reproduced(
                recorded_models[regularization_c],
                model,
                regularization_c=regularization_c,
            )
        )
        _, q4_predictions[model_name] = _metastack_joint(model, q4_base)
    q4_scores = _grid_point_scores(q4_fit, q4_predictions, q4_base["flare24"])
    _assert_nested_close(report["q4_fit_scores"], q4_scores, role="Q4 fit scores")
    _assert_nested_close(grid_lock["q4_fit_scores"], q4_scores, role="grid-lock Q4 scores")
    del q4_frame, q4_all, q4_base, q4_fit, q4_design, q4_predictions
    gc.collect()

    selected_name = str(report["selected_model"])
    selected_c = float(report["selected_regularization_c"])
    if (
        selected_name != _c_key(selected_c)
        or selected_name != h2_lock["selected_model"]
        or selected_c != float(h2_lock["selected_regularization_c"])
    ):
        raise ValueError("meta-stack selected model binding differs")
    selected_model = models[selected_name]
    if confirmation_lock["locked_candidate"]["name"] != META_METHOD:
        raise ValueError("meta-stack confirmation lock has the wrong method")
    selected_record = recorded_models[selected_c]
    if (
        h2_lock["selected_model_artifact"] != selected_record
        or confirmation_lock["locked_candidate"]["intercept"] != selected_record["intercept"]
        or confirmation_lock["locked_candidate"]["coefficients"] != selected_record["coefficients"]
    ):
        raise ValueError("meta-stack selected coefficient lock binding differs")

    parent_sources = sorted(
        (dict(record) for record in parent_report["retrospective_prediction_artifacts"]),
        key=lambda record: int(record["month"]),
    )
    recorded_sources = sorted(
        (dict(record) for record in report["retrospective_source_artifacts"]),
        key=lambda record: int(record["month"]),
    )
    output_records = sorted(
        (dict(record) for record in report["retrospective_prediction_artifacts"]),
        key=lambda record: int(record["month"]),
    )
    if any(
        [int(record["month"]) for record in records] != list(range(1, 13))
        for records in (parent_sources, recorded_sources, output_records)
    ):
        raise ValueError("meta-stack validation requires exactly 12 ordered months")

    h1_sums = {_c_key(value): {"n": 0, "log": 0.0, "brier": 0.0} for value in c_grid}
    h1_sums["flare24"] = {"n": 0, "log": 0.0, "brier": 0.0}
    evaluation_frames: list[pd.DataFrame] = []
    flare_parts: list[NDArray[np.float64]] = []
    meta_parts: list[NDArray[np.float64]] = []
    validated_outputs: list[dict[str, Any]] = []
    maximum_prediction_difference = 0.0
    for parent_source, recorded_source, output_record in zip(
        parent_sources, recorded_sources, output_records, strict=True
    ):
        month = int(parent_source["month"])
        for field in ("path", "bytes", "sha256", "year", "month"):
            if recorded_source.get(field) != parent_source.get(field):
                raise ValueError(f"meta-stack 2025-{month:02d} source binding differs")
        source, _ = _load_prediction_frame(
            parent_source,
            role=f"meta-stack validation 2025-{month:02d} source",
            expected_year=2025,
            expected_month=month,
        )
        normalization: list[dict[str, Any]] = []
        source_all = _probability_methods(
            source,
            normalization_audits=normalization,
            role=f"meta-stack validation 2025-{month:02d} probabilities",
        )
        base: dict[str, NDArray[np.float64]] = {name: source_all[name] for name in BLEND_CANDIDATES}
        expected_cancel, expected_meta = _metastack_joint(selected_model, base)
        output_path_for_month = _verified_artifact(
            output_record, role=f"meta-stack validation 2025-{month:02d} output"
        )
        output = pd.read_parquet(
            output_path_for_month,
            columns=[
                "sample_id",
                META_PROBABILITY_COLUMN,
                *_joint_columns(META_METHOD),
            ],
        )
        if len(output) != len(source) or not output["sample_id"].equals(source["sample_id"]):
            raise ValueError(f"meta-stack 2025-{month:02d} output ids do not align")
        stored_cancel = output[META_PROBABILITY_COLUMN].to_numpy(dtype=np.float64)
        stored_meta = _get_joint_columns(output, META_METHOD)
        difference = max(
            float(np.max(np.abs(stored_cancel - expected_cancel))),
            float(np.max(np.abs(stored_meta - expected_meta))),
        )
        maximum_prediction_difference = max(maximum_prediction_difference, difference)
        if difference > PREDICTION_TOLERANCE:
            raise ValueError(f"meta-stack 2025-{month:02d} predictions differ")

        dates = pd.to_datetime(source["FlightDate"], errors="raise")
        if month <= 6:
            selection_mask = dates.between(SELECTION_START, SELECTION_END).to_numpy()
            selection_frame = source.loc[selection_mask].reset_index(drop=True)
            candidates = {
                name: _metastack_joint(model, base)[1][selection_mask]
                for name, model in models.items()
            }
            selection_scores = _grid_point_scores(
                selection_frame,
                candidates,
                base["flare24"][selection_mask],
            )
            for score_name, values in selection_scores.items():
                n = int(values["n"])
                h1_sums[score_name]["n"] += n
                h1_sums[score_name]["log"] += float(values["joint_log_loss"]) * n
                h1_sums[score_name]["brier"] += float(values["multiclass_brier"]) * n
        primary_mask = dates.between(PRIMARY_AUDIT_START, PRIMARY_AUDIT_END).to_numpy()
        frame_columns = [
            "FlightDate",
            "Month",
            "Origin",
            "Dest",
            "Cancelled",
            "ArrDel15",
            "delay_label_observed",
            "joint_label_observed",
            "disruption_state",
            GATING_FEATURE,
        ]
        evaluation_frames.append(source.loc[primary_mask, frame_columns].copy())
        persisted_flare = base["flare24"].astype(np.float32).astype(np.float64)
        persisted_flare /= persisted_flare.sum(axis=1, keepdims=True)
        flare_parts.append(persisted_flare[primary_mask])
        meta_parts.append(stored_meta[primary_mask])
        validated_outputs.append(
            {
                "year": 2025,
                "month": month,
                "path": output_path_for_month.as_posix(),
                "rows": len(output),
                "bytes": output_path_for_month.stat().st_size,
                "sha256": sha256_file(output_path_for_month),
            }
        )
        del source, source_all, base, output, expected_cancel, expected_meta
        gc.collect()

    h1_scores = {
        name: {
            "n": int(values["n"]),
            "joint_log_loss": float(values["log"]) / int(values["n"]),
            "multiclass_brier": float(values["brier"]) / int(values["n"]),
        }
        for name, values in h1_sums.items()
    }
    _assert_nested_close(report["h1_candidate_scores"], h1_scores, role="H1 scores")
    _assert_nested_close(h2_lock["h1_candidate_scores"], h1_scores, role="H2-lock H1 scores")
    reproduced_selected = min(
        (_c_key(value) for value in c_grid),
        key=lambda name: (
            float(h1_scores[name]["joint_log_loss"]),
            tuple(_c_key(value) for value in c_grid).index(name),
        ),
    )
    if reproduced_selected != selected_name:
        raise ValueError("meta-stack H1 regularization selection does not reproduce")

    evaluation_frame = pd.concat(evaluation_frames, ignore_index=True)
    flare = np.concatenate(flare_parts)
    meta = np.concatenate(meta_parts)
    methods: dict[str, NDArray[np.float64]] = {"flare24": flare, META_METHOD: meta}
    held_mask = (
        pd.to_datetime(evaluation_frame["FlightDate"], errors="raise")
        .between(HELD_FORWARD_START, HELD_FORWARD_END)
        .to_numpy()
    )
    held_evaluation = evaluate_joint_probabilities(
        evaluation_frame.loc[held_mask].reset_index(drop=True),
        cast(dict[str, ArrayLike], {name: values[held_mask] for name, values in methods.items()}),
        reference_method="flare24",
        bootstrap_repetitions=repetitions,
        bootstrap_seed=seed,
    )
    full_evaluation = evaluate_joint_probabilities(
        evaluation_frame,
        cast(dict[str, ArrayLike], methods),
        reference_method="flare24",
        bootstrap_repetitions=repetitions,
        bootstrap_seed=seed + 10_000,
    )
    full_points = _joint_point_scores(evaluation_frame, methods)
    monthly = _monthly_scores(evaluation_frame, methods)
    regimes = _regime_scores(
        evaluation_frame,
        methods,
        {"cutpoints": parent_lock["capacity_gated_simplex"]["cutpoints"]},
    )
    airports = _airport_scores(evaluation_frame, methods)
    _assert_nested_close(
        report["held_forward_evaluation"], held_evaluation, role="held-forward evaluation"
    )
    _assert_nested_close(
        report["descriptive_full_primary_evaluation"],
        full_evaluation,
        role="full-primary evaluation",
    )
    _assert_nested_close(
        report["descriptive_full_primary_joint_scores"],
        full_points,
        role="full-primary point scores",
    )
    _assert_nested_close(report["descriptive_monthly_scores"], monthly, role="monthly scores")
    _assert_nested_close(
        report["descriptive_capacity_regime_scores"], regimes, role="regime scores"
    )
    _assert_nested_close(report["descriptive_airport_scores"], airports, role="airport scores")

    if (
        list(report["h2_embargo_dates"]) != list(H2_EMBARGO)
        or report["held_forward_evaluation_period"] != [HELD_FORWARD_START, HELD_FORWARD_END]
        or report["descriptive_full_primary_period"] != [PRIMARY_AUDIT_START, PRIMARY_AUDIT_END]
        or protocol["claim_limits"].get("blind_2025_confirmation_claimed") is not False
    ):
        raise ValueError("meta-stack periods or claim limits differ")

    validation: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS_TF_CCRTH_METASTACK_STUDY_VALIDATION",
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "report": {
            "path": report_path.as_posix(),
            "bytes": report_path.stat().st_size,
            "sha256": sha256_file(report_path),
            "self_hash_key": report_hash_key,
            "self_hash": report[report_hash_key],
        },
        "protocol": protocol_record,
        "parent_report": parent_record,
        "parent_validation": parent_validation_record,
        "factorized_report": factorized_record,
        "factorized_validation": factorized_validation_record,
        "locks": {
            "grid": {
                "path": grid_path.as_posix(),
                "sha256": sha256_file(grid_path),
                "self_hash": grid_lock[grid_hash_key],
            },
            "held_forward": {
                "path": h2_path.as_posix(),
                "sha256": sha256_file(h2_path),
                "self_hash": h2_lock[h2_hash_key],
            },
            "confirmation": {
                "path": confirmation_path.as_posix(),
                "sha256": sha256_file(confirmation_path),
                "self_hash": confirmation_lock[confirmation_hash_key],
                "2026_outcomes_accessed": False,
            },
        },
        "selected_model": selected_name,
        "selected_regularization_c": selected_c,
        "coefficient_fit_rows": len(q4_labels),
        "h1_selection_rows": int(h1_scores["flare24"]["n"]),
        "held_forward_rows": int(
            report["held_forward_evaluation"]["methods"]["flare24"]["joint"]["n"]
        ),
        "full_primary_rows": len(evaluation_frame),
        "full_primary_date_clusters": int(pd.Series(evaluation_frame["FlightDate"]).nunique()),
        "model_artifacts": [
            {
                "regularization_c": value,
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for value, path in zip(c_grid, model_paths, strict=True)
        ],
        "retrospective_artifacts": validated_outputs,
        "validated_secondary_tables": ["monthly", "capacity_regime", "airport"],
        "maximum_absolute_stored_prediction_difference": maximum_prediction_difference,
        "prediction_tolerance": PREDICTION_TOLERANCE,
        "coefficient_tolerance": COEFFICIENT_TOLERANCE,
        "proper_score_tolerance": POINT_TOLERANCE,
        "confirmation_outcomes_accessed": False,
        "validator_provenance": capture_provenance(
            (Path(__file__), *_frozen_metastack_implementation_files())
        ),
        "advisories": [
            "The architecture and grid were informed by exploratory 2025 inspection.",
            "The execution-held-forward H2 period is not epistemically blind.",
            "Its intervals are not adjusted for prior architecture exploration.",
            "Independent confirmation remains reserved for the unopened 2026 lock.",
        ],
    }
    validation["validation_sha256"] = canonical_json_sha256(validation)
    write_canonical_json(output_path, validation)
    return validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--parent-validation", type=Path, required=True)
    parser.add_argument("--factorized-report", type=Path, required=True)
    parser.add_argument("--factorized-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = validate_cancellation_metastack_report(
        args.report,
        protocol_path=args.protocol,
        parent_report_path=args.parent_report,
        parent_validation_path=args.parent_validation,
        factorized_report_path=args.factorized_report,
        factorized_validation_path=args.factorized_validation,
        output_path=args.output,
    )
    print(json.dumps({"status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
