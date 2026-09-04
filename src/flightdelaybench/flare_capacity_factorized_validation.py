"""Independent artifact and numerical validation for TF-CC-RTH-v1."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .bootstrap import paired_cluster_mean_difference
from .flare_capacity_factorized import (
    CONFIRMATION_YEAR,
    EXPECTED_PARENT_STATUS,
    FACTORIZED_METHODS,
    PRIMARY_YEAR,
    Q4_SELECTION_METHODS,
    _frozen_factorized_implementation_files,
    _load_prediction_frame,
    _load_protocol,
    _pair_name,
    _pair_probabilities,
    _probability_methods,
    _selection_scores,
    _verify_parent_inputs,
    apply_task_factorized_capacity_gated,
    apply_task_factorized_global,
    component_pair_grid,
    select_task_factorized_capacity_gated,
    select_task_factorized_global,
)
from .flare_capacity_study import (
    BLEND_CANDIDATES,
    GATING_FEATURE,
    GATING_LABELS,
    PRIMARY_AUDIT_END,
    PRIMARY_AUDIT_START,
    _get_joint_columns,
    _joint_columns,
    _read_self_hashed,
    _verified_artifact,
)
from .flare_evaluation import joint_loss_rows
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

EXPECTED_STATUS = "COMPLETE_TF_CCRTH_POST_HOC_2025_ANALYSIS_2026_UNOPENED"
EXPECTED_Q4_LOCK_STATUS = (
    "LOCKED_TF_CCRTH_Q4_WEIGHTS_AFTER_2025_AGGREGATE_RESULTS_WERE_KNOWN_"
    "BEFORE_THIS_RUN_LOADED_ROW_LEVEL_2025_PREDICTIONS"
)
EXPECTED_CONFIRMATION_LOCK_STATUS = (
    "LOCKED_TF_CCRTH_V1_FOR_UNOPENED_2026_CONFIRMATION"
)
POINT_TOLERANCE = 2e-11
WEIGHT_TOLERANCE = 2e-9
PREDICTION_TOLERANCE = 2e-7


def _assert_simplex_weights(selection: Mapping[str, Any], *, role: str) -> None:
    weights = selection.get("weights", {})
    if set(weights) != set(BLEND_CANDIDATES):
        raise ValueError(f"{role} weights have the wrong candidate set")
    values = np.asarray([weights[name] for name in BLEND_CANDIDATES], dtype=np.float64)
    if (
        not np.isfinite(values).all()
        or (values < -1e-12).any()
        or (values > 1.0 + 1e-12).any()
        or not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise ValueError(f"{role} weights are not a probability simplex")


def _assert_factorized_selections(selection: Mapping[str, Any], *, gated: bool) -> None:
    if gated:
        if (
            selection.get("gating_feature") != GATING_FEATURE
            or set(selection.get("regimes", {})) != set(GATING_LABELS)
        ):
            raise ValueError("capacity-gated factorized selection has invalid regimes")
        _assert_factorized_selections(selection["global"], gated=False)
        for regime in GATING_LABELS:
            components = selection["regimes"][regime].get("components", {})
            if set(components) != {"cancellation", "delay_given_operated"}:
                raise ValueError(f"factorized {regime} selection has invalid tasks")
            for task, task_selection in components.items():
                _assert_simplex_weights(task_selection, role=f"{regime} {task}")
        return
    components = selection.get("components", {})
    if set(components) != {"cancellation", "delay_given_operated"}:
        raise ValueError("global factorized selection has invalid tasks")
    for task, task_selection in components.items():
        _assert_simplex_weights(task_selection, role=f"global {task}")


def _assert_close(actual: float, expected: float, *, role: str, tolerance: float) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=tolerance):
        raise ValueError(f"{role} differs: observed={actual}, expected={expected}")


def _assert_selection_reproduced(
    recorded: Mapping[str, Any],
    reproduced: Mapping[str, Any],
    *,
    gated: bool,
) -> None:
    if gated:
        if not np.allclose(
            np.asarray(recorded["cutpoints"], dtype=np.float64),
            np.asarray(reproduced["cutpoints"], dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("TF-CC-RTH gated cutpoints do not reproduce")
        _assert_selection_reproduced(
            recorded["global"], reproduced["global"], gated=False
        )
        for regime in GATING_LABELS:
            for task in ("cancellation", "delay_given_operated"):
                left = recorded["regimes"][regime]["components"][task]
                right = reproduced["regimes"][regime]["components"][task]
                if bool(left.get("fallback_to_global")) != bool(
                    right.get("fallback_to_global")
                ) or int(left["rows"]) != int(right["rows"]):
                    raise ValueError(f"TF-CC-RTH {regime} {task} selection metadata differs")
                for name in BLEND_CANDIDATES:
                    _assert_close(
                        left["weights"][name],
                        right["weights"][name],
                        role=f"{regime} {task} weight {name}",
                        tolerance=WEIGHT_TOLERANCE,
                    )
        return
    for task in ("cancellation", "delay_given_operated"):
        left = recorded["components"][task]
        right = reproduced["components"][task]
        if int(left["rows"]) != int(right["rows"]):
            raise ValueError(f"TF-CC-RTH global {task} row count differs")
        for name in BLEND_CANDIDATES:
            _assert_close(
                left["weights"][name],
                right["weights"][name],
                role=f"global {task} weight {name}",
                tolerance=WEIGHT_TOLERANCE,
            )


def _records_by_name(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {str(record["method"]): record for record in records}
    if len(result) != len(records):
        raise ValueError("TF-CC-RTH component grid repeats a method")
    return result


def _validate_grid(
    recorded: Sequence[Mapping[str, Any]],
    reproduced: Sequence[Mapping[str, Any]],
    *,
    role: str,
) -> None:
    expected_names = {
        _pair_name(cancel, delay)
        for cancel in BLEND_CANDIDATES
        for delay in BLEND_CANDIDATES
    }
    left = _records_by_name(recorded)
    right = _records_by_name(reproduced)
    if set(left) != expected_names or set(right) != expected_names:
        raise ValueError(f"{role} does not contain the complete 5x5 grid")
    for name in expected_names:
        for field in ("joint_log_loss", "multiclass_brier"):
            _assert_close(
                left[name][field],
                right[name][field],
                role=f"{role} {name} {field}",
                tolerance=POINT_TOLERANCE,
            )
        if int(left[name]["n"]) != int(right[name]["n"]):
            raise ValueError(f"{role} {name} row count differs")


def _method_point_accumulator(methods: Sequence[str]) -> dict[str, dict[str, float | int]]:
    return {
        name: {"full_n": 0, "full_log": 0.0, "full_brier": 0.0, "primary_n": 0, "primary_log": 0.0, "primary_brier": 0.0}
        for name in methods
    }


def _accumulate_scores(
    accumulator: dict[str, dict[str, float | int]],
    *,
    labels: NDArray[np.int64],
    observed: NDArray[np.bool_],
    primary: NDArray[np.bool_],
    methods: Mapping[str, NDArray[np.float64]],
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    losses: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for name, probabilities in methods.items():
        full_log, full_brier = joint_loss_rows(labels[observed], probabilities[observed])
        primary_mask = observed & primary
        primary_log, primary_brier = joint_loss_rows(
            labels[primary_mask], probabilities[primary_mask]
        )
        values = accumulator[name]
        values["full_n"] = int(values["full_n"]) + len(full_log)
        values["full_log"] = float(values["full_log"]) + float(full_log.sum())
        values["full_brier"] = float(values["full_brier"]) + float(full_brier.sum())
        values["primary_n"] = int(values["primary_n"]) + len(primary_log)
        values["primary_log"] = float(values["primary_log"]) + float(primary_log.sum())
        values["primary_brier"] = float(values["primary_brier"]) + float(
            primary_brier.sum()
        )
        losses[name] = (primary_log, primary_brier)
    return losses


def _validate_point_scores(
    report: Mapping[str, Any],
    accumulator: Mapping[str, Mapping[str, float | int]],
) -> None:
    primary_report = report["retrospective_evaluation"]["methods"]
    full_report = report["retrospective_full_year_descriptive_joint_scores"]
    for name, values in accumulator.items():
        primary_n = int(values["primary_n"])
        full_n = int(values["full_n"])
        if primary_n != int(primary_report[name]["joint"]["n"]):
            raise ValueError(f"TF-CC-RTH primary {name} row count differs")
        if full_n != int(full_report[name]["n"]):
            raise ValueError(f"TF-CC-RTH full-year {name} row count differs")
        for field, sum_field, report_field in (
            ("primary log loss", "primary_log", "log_loss"),
            ("primary Brier", "primary_brier", "multiclass_brier"),
        ):
            _assert_close(
                float(values[sum_field]) / primary_n,
                primary_report[name]["joint"][report_field],
                role=f"{name} {field}",
                tolerance=POINT_TOLERANCE,
            )
        for field, sum_field, report_field in (
            ("full-year log loss", "full_log", "joint_log_loss"),
            ("full-year Brier", "full_brier", "multiclass_brier"),
        ):
            _assert_close(
                float(values[sum_field]) / full_n,
                full_report[name][report_field],
                role=f"{name} {field}",
                tolerance=POINT_TOLERANCE,
            )


def _validate_interval(
    recorded: Mapping[str, Any],
    candidate_log: NDArray[np.float64],
    reference_log: NDArray[np.float64],
    candidate_brier: NDArray[np.float64],
    reference_brier: NDArray[np.float64],
    dates: NDArray[Any],
    *,
    repetitions: int,
    seed: int,
    role: str,
) -> None:
    expected = {
        "joint_log_loss": asdict(
            paired_cluster_mean_difference(
                candidate_log,
                reference_log,
                dates,
                repetitions=repetitions,
                seed=seed,
            )
        ),
        "multiclass_brier": asdict(
            paired_cluster_mean_difference(
                candidate_brier,
                reference_brier,
                dates,
                repetitions=repetitions,
                seed=seed,
            )
        ),
    }
    for metric in ("joint_log_loss", "multiclass_brier"):
        for field in ("estimate", "lower", "upper"):
            _assert_close(
                recorded[metric][field],
                expected[metric][field],
                role=f"{role} {metric} {field}",
                tolerance=POINT_TOLERANCE,
            )
        for field in ("clusters", "repetitions", "seed"):
            if int(recorded[metric][field]) != int(expected[metric][field]):
                raise ValueError(f"{role} {metric} {field} differs")


def validate_task_factorized_report(
    report_path: Path,
    *,
    protocol_path: Path,
    parent_report_path: Path,
    parent_validation_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite TF-CC-RTH validation: {output_path}")
    report, report_hash_key = _read_self_hashed(report_path)
    if report.get("status") != EXPECTED_STATUS:
        raise ValueError("TF-CC-RTH report is not complete")
    if report.get("outcomes_accessed", {}).get("2026_accessed") is not False:
        raise ValueError("TF-CC-RTH report crossed the 2026 confirmation boundary")
    repetitions = int(report["retrospective_evaluation"]["bootstrap_repetitions"])
    seed = int(report["retrospective_evaluation"]["bootstrap_seed"])
    protocol, protocol_record = _load_protocol(
        protocol_path,
        repetitions=repetitions,
        seed=seed,
    )
    parent_report, parent_lock, parent_record, parent_validation_record = (
        _verify_parent_inputs(parent_report_path, parent_validation_path)
    )
    if parent_report.get("status") != EXPECTED_PARENT_STATUS:
        raise ValueError("TF-CC-RTH validator received the wrong parent study")
    for recorded, expected, role in (
        (report["protocol"], protocol_record, "protocol"),
        (report["parent_report"], parent_record, "parent report"),
        (report["parent_validation"], parent_validation_record, "parent validation"),
    ):
        if recorded != expected:
            raise ValueError(f"TF-CC-RTH {role} binding differs")

    q4_record = dict(report["q4_method_lock"])
    q4_path = _verified_artifact(q4_record, role="TF-CC-RTH Q4 method lock")
    q4_lock, q4_hash_key = _read_self_hashed(q4_path)
    if (
        q4_lock.get("status") != EXPECTED_Q4_LOCK_STATUS
        or q4_lock.get("information_boundary", {}).get("2026_outcomes_accessed") is not False
        or q4_lock.get("information_boundary", {}).get(
            "2025_row_level_predictions_loaded_by_this_run_before_lock"
        )
        is not False
    ):
        raise ValueError("TF-CC-RTH Q4 method lock has an invalid boundary status")
    if q4_record.get("self_hash") != q4_lock[q4_hash_key]:
        raise ValueError("TF-CC-RTH Q4 method-lock self-hash binding differs")
    _assert_factorized_selections(q4_lock["global_component_simplexes"], gated=False)
    _assert_factorized_selections(
        q4_lock["capacity_gated_component_simplexes"], gated=True
    )

    selection_frame, _ = _load_prediction_frame(
        dict(parent_report["selection_crossfit_prediction_artifact"]),
        role="TF-CC-RTH validation Q4 source",
        expected_year=2024,
        expected_month=None,
    )
    selection_normalization: list[dict[str, Any]] = []
    selection_all = _probability_methods(
        selection_frame,
        normalization_audits=selection_normalization,
        role="TF-CC-RTH validation Q4 probabilities",
    )
    selection_base: dict[str, NDArray[np.float64]] = {
        name: selection_all[name] for name in BLEND_CANDIDATES
    }
    reproduced_global = select_task_factorized_global(selection_frame, selection_base)
    reproduced_gated = select_task_factorized_capacity_gated(
        selection_frame,
        selection_base,
        cutpoints=parent_lock["capacity_gated_simplex"]["cutpoints"],
        minimum_rows=int(parent_lock["capacity_gated_simplex"]["minimum_rows"]),
        global_selection=reproduced_global,
    )
    _assert_selection_reproduced(
        q4_lock["global_component_simplexes"], reproduced_global, gated=False
    )
    _assert_selection_reproduced(
        q4_lock["capacity_gated_component_simplexes"], reproduced_gated, gated=True
    )
    selection_methods = {
        "flare24": selection_all["flare24"],
        "capacity_gated_simplex": selection_all["capacity_gated_simplex"],
        "task_factorized_global_q4": apply_task_factorized_global(
            selection_base, reproduced_global
        ),
        "task_factorized_capacity_gated_q4": apply_task_factorized_capacity_gated(
            selection_base,
            selection_frame[GATING_FEATURE].to_numpy(dtype=np.float64),
            reproduced_gated,
        ),
    }
    selection_scores = _selection_scores(selection_frame, selection_methods)
    for name in Q4_SELECTION_METHODS:
        for metric in ("joint_log_loss", "multiclass_brier"):
            _assert_close(
                q4_lock["selection_scores"][name][metric],
                selection_scores[name][metric],
                role=f"Q4 {name} {metric}",
                tolerance=POINT_TOLERANCE,
            )
    selected_q4 = min(
        Q4_SELECTION_METHODS,
        key=lambda name: (
            float(selection_scores[name]["joint_log_loss"]),
            Q4_SELECTION_METHODS.index(name),
        ),
    )
    if (
        selected_q4 != q4_lock["selected_method_by_q4_forward_joint_log_loss"]
        or selected_q4 != report["selected_method_by_q4_forward_joint_log_loss"]
    ):
        raise ValueError("TF-CC-RTH Q4 method selection does not reproduce")
    selection_grid = component_pair_grid(selection_frame, selection_base)
    _validate_grid(
        report["selection_component_pair_grid"],
        selection_grid,
        role="Q4 component grid",
    )
    del selection_frame, selection_all, selection_base, selection_methods

    confirmation_record = dict(report["confirmation_lock"])
    confirmation_path = _verified_artifact(
        confirmation_record, role="TF-CC-RTH 2026 confirmation lock"
    )
    confirmation_lock, confirmation_hash_key = _read_self_hashed(confirmation_path)
    if (
        confirmation_lock.get("status") != EXPECTED_CONFIRMATION_LOCK_STATUS
        or confirmation_lock.get("information_boundary", {}).get(
            "2026_outcomes_accessed"
        )
        is not False
        or confirmation_record.get("confirmation_outcomes_accessed") is not False
        or confirmation_record.get("self_hash")
        != confirmation_lock[confirmation_hash_key]
    ):
        raise ValueError("TF-CC-RTH confirmation lock has an invalid boundary status")
    if int(confirmation_lock["confirmation_protocol"]["year"]) != CONFIRMATION_YEAR:
        raise ValueError("TF-CC-RTH confirmation lock has the wrong confirmation year")

    winner = dict(report["post_hoc_component_pair_winner"])
    winner_name = str(winner["method"])
    if winner_name != confirmation_lock["locked_candidate"]["name"]:
        raise ValueError("TF-CC-RTH confirmation lock does not bind the 2025 grid winner")
    methods = (
        "flare24",
        "capacity_gated_simplex",
        *FACTORIZED_METHODS,
        winner_name,
    )
    accumulator = _method_point_accumulator(methods)
    paired_values: dict[str, dict[str, list[NDArray[np.float64]]]] = {
        name: {"log": [], "brier": []} for name in methods
    }
    paired_dates: list[NDArray[Any]] = []
    grid_sums = {
        _pair_name(cancel, delay): {"n": 0, "log": 0.0, "brier": 0.0}
        for cancel in BLEND_CANDIDATES
        for delay in BLEND_CANDIDATES
    }
    maximum_prediction_difference = 0.0
    source_records = sorted(
        (dict(record) for record in report["retrospective_source_artifacts"]),
        key=lambda record: int(record["month"]),
    )
    output_records = sorted(
        (dict(record) for record in report["retrospective_prediction_artifacts"]),
        key=lambda record: int(record["month"]),
    )
    if (
        [int(record["month"]) for record in source_records] != list(range(1, 13))
        or [int(record["month"]) for record in output_records] != list(range(1, 13))
    ):
        raise ValueError("TF-CC-RTH validation requires exactly 12 source/output months")
    validated_output_records: list[dict[str, Any]] = []
    for source_record, output_record in zip(source_records, output_records, strict=True):
        month = int(source_record["month"])
        if month != int(output_record["month"]):
            raise ValueError("TF-CC-RTH source/output month alignment differs")
        source, _ = _load_prediction_frame(
            source_record,
            role=f"TF-CC-RTH validation 2025-{month:02d} source",
            expected_year=PRIMARY_YEAR,
            expected_month=month,
        )
        normalization: list[dict[str, Any]] = []
        source_all = _probability_methods(
            source,
            normalization_audits=normalization,
            role=f"TF-CC-RTH validation 2025-{month:02d} source probabilities",
        )
        source_base: dict[str, NDArray[np.float64]] = {
            name: source_all[name] for name in BLEND_CANDIDATES
        }
        expected_factorized = {
            "task_factorized_global_q4": apply_task_factorized_global(
                source_base, q4_lock["global_component_simplexes"]
            ),
            "task_factorized_capacity_gated_q4": (
                apply_task_factorized_capacity_gated(
                    source_base,
                    source[GATING_FEATURE].to_numpy(dtype=np.float64),
                    q4_lock["capacity_gated_component_simplexes"],
                )
            ),
        }
        expected_winner = _pair_probabilities(
            source_base,
            cancel_source=str(winner["cancellation_source"]),
            delay_source=str(winner["delay_given_operated_source"]),
        )
        output_path_for_month = _verified_artifact(
            output_record,
            role=f"TF-CC-RTH validation 2025-{month:02d} output",
        )
        output_columns = [
            "sample_id",
            *(
                column
                for name in (*FACTORIZED_METHODS, winner_name)
                for column in _joint_columns(name)
            ),
        ]
        output = pd.read_parquet(output_path_for_month, columns=output_columns)
        if len(output) != len(source) or not output["sample_id"].equals(source["sample_id"]):
            raise ValueError(f"TF-CC-RTH 2025-{month:02d} output ids do not align")
        stored_factorized = {
            name: _get_joint_columns(output, name) for name in FACTORIZED_METHODS
        }
        stored_winner = _get_joint_columns(output, winner_name)
        for name in FACTORIZED_METHODS:
            difference = float(np.max(np.abs(stored_factorized[name] - expected_factorized[name])))
            maximum_prediction_difference = max(maximum_prediction_difference, difference)
            if difference > PREDICTION_TOLERANCE:
                raise ValueError(f"TF-CC-RTH 2025-{month:02d} {name} predictions differ")
        winner_difference = float(np.max(np.abs(stored_winner - expected_winner)))
        maximum_prediction_difference = max(maximum_prediction_difference, winner_difference)
        if winner_difference > PREDICTION_TOLERANCE:
            raise ValueError(f"TF-CC-RTH 2025-{month:02d} grid-winner predictions differ")
        current_methods = {
            "flare24": source_all["flare24"],
            "capacity_gated_simplex": source_all["capacity_gated_simplex"],
            **stored_factorized,
            winner_name: expected_winner,
        }
        labels = source["disruption_state"].to_numpy(dtype=np.int64)
        observed = source["joint_label_observed"].eq(1).to_numpy(dtype=np.bool_)
        dates = pd.to_datetime(source["FlightDate"], errors="raise")
        primary = dates.between(
            pd.Timestamp(PRIMARY_AUDIT_START), pd.Timestamp(PRIMARY_AUDIT_END)
        ).to_numpy(dtype=np.bool_)
        losses = _accumulate_scores(
            accumulator,
            labels=labels,
            observed=observed,
            primary=primary,
            methods=current_methods,
        )
        paired_dates.append(dates.to_numpy()[observed & primary])
        for name, (log_rows, brier_rows) in losses.items():
            paired_values[name]["log"].append(log_rows)
            paired_values[name]["brier"].append(brier_rows)
        primary_frame = source.loc[observed & primary].reset_index(drop=True)
        primary_source = {
            name: values[observed & primary] for name, values in source_base.items()
        }
        for record in component_pair_grid(primary_frame, primary_source):
            values = grid_sums[str(record["method"])]
            values["n"] += int(record["n"])
            values["log"] += float(record["joint_log_loss"]) * int(record["n"])
            values["brier"] += float(record["multiclass_brier"]) * int(record["n"])
        validated_output_records.append(
            {
                "year": PRIMARY_YEAR,
                "month": month,
                "path": output_path_for_month.as_posix(),
                "rows": len(output),
                "bytes": output_path_for_month.stat().st_size,
                "sha256": sha256_file(output_path_for_month),
            }
        )

    _validate_point_scores(report, accumulator)
    reproduced_grid: list[dict[str, Any]] = []
    for cancel_source in BLEND_CANDIDATES:
        for delay_source in BLEND_CANDIDATES:
            name = _pair_name(cancel_source, delay_source)
            values = grid_sums[name]
            n = int(values["n"])
            reproduced_grid.append(
                {
                    "method": name,
                    "cancellation_source": cancel_source,
                    "delay_given_operated_source": delay_source,
                    "n": n,
                    "joint_log_loss": float(values["log"]) / n,
                    "multiclass_brier": float(values["brier"]) / n,
                }
            )
    _validate_grid(
        report["post_hoc_component_pair_grid"],
        reproduced_grid,
        role="2025 post-hoc component grid",
    )
    reproduced_winner = min(
        reproduced_grid,
        key=lambda record: (
            float(record["joint_log_loss"]),
            float(record["multiclass_brier"]),
            BLEND_CANDIDATES.index(str(record["cancellation_source"])),
            BLEND_CANDIDATES.index(str(record["delay_given_operated_source"])),
        ),
    )
    if reproduced_winner["method"] != winner_name:
        raise ValueError("TF-CC-RTH 2025 component-grid winner does not reproduce")

    dates_all = np.concatenate(paired_dates)
    comparisons = report["retrospective_evaluation"][
        "paired_date_cluster_comparisons"
    ]
    reference_log = np.concatenate(paired_values["flare24"]["log"])
    reference_brier = np.concatenate(paired_values["flare24"]["brier"])
    for name in methods:
        if name == "flare24":
            continue
        key = f"{name}_minus_flare24"
        _validate_interval(
            comparisons[key],
            np.concatenate(paired_values[name]["log"]),
            reference_log,
            np.concatenate(paired_values[name]["brier"]),
            reference_brier,
            dates_all,
            repetitions=repetitions,
            seed=seed,
            role=key,
        )

    airport_records = report.get("retrospective_airport_scores", [])
    if (
        not airport_records
        or {str(record["method"]) for record in airport_records} != set(methods)
        or any(int(record["n"]) <= 0 for record in airport_records)
    ):
        raise ValueError("TF-CC-RTH airport heterogeneity records are incomplete")
    if protocol["claim_limits"].get("blind_2025_confirmation_claimed") is not False:
        raise ValueError("TF-CC-RTH protocol incorrectly claims blind 2025 confirmation")

    validation: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS_TF_CCRTH_STUDY_VALIDATION",
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "report": {
            "path": report_path.as_posix(),
            "bytes": report_path.stat().st_size,
            "sha256": sha256_file(report_path),
            "self_hash_key": report_hash_key,
            "self_hash": report[report_hash_key],
        },
        "parent_report": parent_record,
        "parent_validation": parent_validation_record,
        "protocol": protocol_record,
        "q4_lock": {
            "path": q4_path.as_posix(),
            "sha256": sha256_file(q4_path),
            "self_hash": q4_lock[q4_hash_key],
            "selected_method": selected_q4,
            "selection_rows": selection_scores["flare24"]["n"],
        },
        "confirmation_lock": {
            "path": confirmation_path.as_posix(),
            "sha256": sha256_file(confirmation_path),
            "self_hash": confirmation_lock[confirmation_hash_key],
            "locked_candidate": winner_name,
            "2026_outcomes_accessed": False,
        },
        "retrospective_artifacts": validated_output_records,
        "evaluated_primary_rows": int(accumulator["flare24"]["primary_n"]),
        "evaluated_full_year_rows": int(accumulator["flare24"]["full_n"]),
        "primary_date_clusters": int(pd.Series(dates_all).nunique()),
        "methods": list(methods),
        "component_pair_grid_size": len(reproduced_grid),
        "post_hoc_winner": winner_name,
        "maximum_absolute_stored_prediction_difference": maximum_prediction_difference,
        "prediction_tolerance": PREDICTION_TOLERANCE,
        "proper_score_tolerance": POINT_TOLERANCE,
        "confirmation_outcomes_accessed": False,
        "validator_provenance": capture_provenance(
            (
                Path(__file__),
                *_frozen_factorized_implementation_files(),
            )
        ),
        "advisories": [
            "The 2025 component-pair winner was selected post hoc from 25 pairs.",
            "Its 2025 interval is descriptive and not selection-adjusted.",
            "The Q4-locked factorized methods are retrospective evidence, not blind confirmation.",
            "Independent evaluation remains reserved for the unopened 2026 lock.",
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
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = validate_task_factorized_report(
        args.report,
        protocol_path=args.protocol,
        parent_report_path=args.parent_report,
        parent_validation_path=args.parent_validation,
        output_path=args.output,
    )
    print(json.dumps({"status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
