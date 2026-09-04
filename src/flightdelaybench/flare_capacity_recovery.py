"""Create an audited recovery record for a failed 2024-only CC-RTH study run."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]

from .flare_capacity_contracts import validate_capacity_predictors
from .flare_capacity_modeling import (
    CAPACITY_CANDIDATE_FEATURES,
    CapacityCatBoostModel,
)
from .flare_capacity_study import (
    AUGMENTED_CANDIDATES,
    RECOVERY_DISPOSITION,
    _read_self_hashed,
)
from .flare_study import TASKS
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _assert_closed_outcome_boundary(
    failed_run_dir: Path,
    expected_study_report_path: Path,
) -> None:
    forbidden = [
        failed_run_dir / "method_lock.json",
        failed_run_dir / "run_manifest.json",
        expected_study_report_path,
    ]
    present = [path for path in forbidden if path.exists()]
    confirmation_year_pattern = re.compile(r"(?<!\d)202[56](?!\d)")
    present.extend(
        path
        for path in failed_run_dir.rglob("*")
        if path.is_file()
        and confirmation_year_pattern.search(
            path.relative_to(failed_run_dir).as_posix()
        )
    )
    calibrators = failed_run_dir / "calibrators"
    if calibrators.is_dir():
        present.extend(path for path in calibrators.rglob("*") if path.is_file())
    if present:
        raise ValueError(
            "failed CC-RTH run is not demonstrably pre-lock and 2024-only: "
            + ", ".join(path.as_posix() for path in sorted(set(present)))
        )


def _validate_estimator_parameters(
    model: CapacityCatBoostModel,
    *,
    task: str,
    protocol: dict[str, Any],
    tree_count: int,
) -> dict[str, Any]:
    parameters = dict(model.estimator.get_params())
    expected = dict(protocol["models"][f"{task}_parameters"])
    expected["iterations"] = tree_count
    expected["random_seed"] = int(protocol["models"]["sampling_seed"])
    for name, value in expected.items():
        if parameters.get(name) != value:
            raise ValueError(f"recovery model parameter differs for {task}: {name}")
    fixed = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "bootstrap_type": "Bayesian",
        "task_type": "GPU",
        "devices": "0",
        "verbose": False,
        "allow_writing_files": False,
    }
    for name, value in fixed.items():
        if parameters.get(name) != value:
            raise ValueError(f"recovery model execution parameter differs: {task}/{name}")
    return parameters


def create_capacity_model_recovery_record(
    *,
    failed_run_dir: Path,
    protocol_path: Path,
    capacity_manifest_path: Path,
    expected_study_report_path: Path,
    output_path: Path,
    failure_summary: str,
    failure_observed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Inspect all eight models and write an immutable, self-hashed recovery record."""

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite CC-RTH recovery record: {output_path}")
    if not failed_run_dir.is_dir():
        raise FileNotFoundError(failed_run_dir)
    if not failure_summary.strip():
        raise ValueError("CC-RTH recovery requires a non-empty failure summary")
    _assert_closed_outcome_boundary(failed_run_dir, expected_study_report_path)

    protocol: dict[str, Any] = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("identity", {}).get("method") != "CC-RTH-v1":
        raise ValueError("recovery protocol is not CC-RTH-v1")
    capacity_manifest, manifest_hash_key = _read_self_hashed(capacity_manifest_path)
    if (
        capacity_manifest.get("status")
        != "COMPLETE_COVARIATE_GRAPH_NO_TARGET_OUTCOMES_ACCESSED"
    ):
        raise ValueError("recovery requires the complete covariate-only graph manifest")

    model_dir = failed_run_dir / "models"
    expected_paths = {
        model_dir / f"{candidate}_{task}.joblib"
        for candidate in AUGMENTED_CANDIDATES
        for task in TASKS
    }
    observed_paths = set(model_dir.glob("*.joblib")) if model_dir.is_dir() else set()
    if observed_paths != expected_paths:
        missing = sorted(path.as_posix() for path in expected_paths - observed_paths)
        unexpected = sorted(path.as_posix() for path in observed_paths - expected_paths)
        raise ValueError(
            f"CC-RTH recovery model set differs; missing={missing}, unexpected={unexpected}"
        )

    records: list[dict[str, Any]] = []
    feature_sets: dict[tuple[str, str], tuple[str, ...]] = {}
    flare_feature_sets: set[tuple[str, ...]] = set()
    for task in TASKS:
        for candidate in AUGMENTED_CANDIDATES:
            path = model_dir / f"{candidate}_{task}.joblib"
            model = joblib.load(path)
            if not isinstance(model, CapacityCatBoostModel) or model.task != task:
                raise TypeError(f"unexpected CC-RTH recovery model contract: {path}")
            validate_capacity_predictors(list(model.capacity_features))
            if not set(model.capacity_features).issubset(
                CAPACITY_CANDIDATE_FEATURES[candidate]
            ):
                raise ValueError(f"recovery model exceeds frozen candidate: {candidate}/{task}")
            tree_count = int(model.estimator.tree_count_)
            if tree_count <= 0:
                raise ValueError(f"recovery model has no fitted trees: {candidate}/{task}")
            parameters = _validate_estimator_parameters(
                model,
                task=task,
                protocol=protocol,
                tree_count=tree_count,
            )
            feature_sets[(candidate, task)] = model.capacity_features
            flare_feature_sets.add(model.flare_features)
            records.append(
                {
                    "candidate": candidate,
                    "task": task,
                    "early_stopping_best_iteration": tree_count - 1,
                    "refit_iterations": tree_count,
                    "flare_feature_count": len(model.flare_features),
                    "capacity_feature_count": len(model.capacity_features),
                    "flare_features_sha256": canonical_json_sha256(
                        list(model.flare_features)
                    ),
                    "capacity_features_sha256": canonical_json_sha256(
                        list(model.capacity_features)
                    ),
                    "estimator_parameters_sha256": canonical_json_sha256(parameters),
                    **_artifact(path),
                }
            )
    if len(flare_feature_sets) != 1:
        raise ValueError("recovery models do not share one frozen FLARE feature contract")
    for candidate in AUGMENTED_CANDIDATES:
        if feature_sets[(candidate, TASKS[0])] != feature_sets[(candidate, TASKS[1])]:
            raise ValueError(f"recovery feature set differs by task: {candidate}")
    for task in TASKS:
        for left, right in pairwise(AUGMENTED_CANDIDATES):
            if not set(feature_sets[(left, task)]).issubset(feature_sets[(right, task)]):
                raise ValueError(f"recovery feature sets are not nested: {left}/{right}/{task}")

    partial_records = [
        _artifact(path)
        for path in sorted(failed_run_dir.rglob("*.part"))
        if path.is_file()
    ]
    completed_2024_predictions = [
        _artifact(path)
        for path in sorted((failed_run_dir / "predictions").glob("raw_selection_2024_*.parquet"))
        if path.is_file()
    ]
    record: dict[str, Any] = {
        "schema_version": 1,
        "failure_recorded_at_utc": datetime.now(UTC).isoformat(),
        "failure_observed_at_utc": failure_observed_at_utc,
        "disposition": RECOVERY_DISPOSITION,
        "run_directory": failed_run_dir.as_posix(),
        "failure": failure_summary.strip(),
        "boundary_status": {
            "2024_outcomes_accessed": True,
            "2025_outcomes_accessed": False,
            "2026_outcomes_accessed": False,
            "method_lock_written": False,
            "study_report_written": False,
        },
        "protocol": _artifact(protocol_path),
        "capacity_manifest": {
            **_artifact(capacity_manifest_path),
            "self_hash_key": manifest_hash_key,
            "self_hash": capacity_manifest[manifest_hash_key],
        },
        "completed_model_artifacts": records,
        "completed_2024_prediction_artifacts": completed_2024_predictions,
        "retained_partial_artifacts": partial_records,
        "recovery_contract": {
            "all_eight_models_present": True,
            "all_model_checksums_verified": True,
            "model_tasks_features_trees_and_parameters_verified": True,
            "candidate_feature_sets_nested": True,
            "no_method_lock_present": True,
            "no_study_report_present": True,
            "no_2025_artifact_present": True,
            "no_2026_outcome_accessed": True,
        },
        "impact": (
            "The failed run produced no method lock, no 2025 prediction or score, and no "
            "study report. Its complete 2024-only final-model set may be checksum-reused "
            "by a fresh run; every later selection and evaluation step must be rerun."
        ),
        "results_claimed": False,
        "provenance": capture_provenance((Path(__file__),)),
    }
    record["record_sha256"] = canonical_json_sha256(record)
    write_canonical_json(output_path, record)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--capacity-manifest", type=Path, required=True)
    parser.add_argument("--expected-study-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-summary", required=True)
    parser.add_argument("--failure-observed-at-utc")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = create_capacity_model_recovery_record(
        failed_run_dir=args.failed_run_dir,
        protocol_path=args.protocol,
        capacity_manifest_path=args.capacity_manifest,
        expected_study_report_path=args.expected_study_report,
        output_path=args.output,
        failure_summary=args.failure_summary,
        failure_observed_at_utc=args.failure_observed_at_utc,
    )
    print(
        json.dumps(
            {
                "disposition": result["disposition"],
                "model_count": len(result["completed_model_artifacts"]),
                "record_sha256": result["record_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
