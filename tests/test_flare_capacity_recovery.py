from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

from flightdelaybench.flare_capacity_modeling import (
    CAPACITY_CANDIDATE_FEATURES,
    CapacityCatBoostModel,
)
from flightdelaybench.flare_capacity_recovery import (
    create_capacity_model_recovery_record,
)
from flightdelaybench.flare_capacity_study import (
    AUGMENTED_CANDIDATES,
    _atomic_joblib,
    _load_recovery_model_record,
)
from flightdelaybench.flare_study import TASKS
from flightdelaybench.hashing import canonical_json_sha256, sha256_file, write_canonical_json


class _FakeCatBoostEstimator:
    def __init__(self, parameters: dict[str, Any], tree_count: int) -> None:
        self._parameters = parameters
        self.tree_count_ = tree_count

    def get_params(self) -> dict[str, Any]:
        return self._parameters


def _parameters(protocol: dict[str, Any], task: str, tree_count: int) -> dict[str, Any]:
    return {
        **protocol["models"][f"{task}_parameters"],
        "iterations": tree_count,
        "random_seed": protocol["models"]["sampling_seed"],
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "bootstrap_type": "Bayesian",
        "task_type": "GPU",
        "devices": "0",
        "verbose": False,
        "allow_writing_files": False,
    }


def test_create_recovery_record_reinspects_all_models(tmp_path: Path) -> None:
    protocol_path = Path("configs/flare24_ccrth_v1.toml")
    protocol = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    failed_run = tmp_path / "failed_run"
    for task in TASKS:
        for candidate in AUGMENTED_CANDIDATES:
            model = CapacityCatBoostModel(
                estimator=_FakeCatBoostEstimator(_parameters(protocol, task, 7), 7),
                task=task,
                flare_features=("frozen_flare_feature",),
                capacity_features=CAPACITY_CANDIDATE_FEATURES[candidate],
            )
            _atomic_joblib(
                model,
                failed_run / "models" / f"{candidate}_{task}.joblib",
            )

    manifest = {
        "schema_version": 1,
        "status": "COMPLETE_COVARIATE_GRAPH_NO_TARGET_OUTCOMES_ACCESSED",
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path = tmp_path / "capacity_manifest.json"
    write_canonical_json(manifest_path, manifest)
    output = tmp_path / "recovery_record.json"

    record = create_capacity_model_recovery_record(
        failed_run_dir=failed_run,
        protocol_path=protocol_path,
        capacity_manifest_path=manifest_path,
        expected_study_report_path=tmp_path / "absent_report.json",
        output_path=output,
        failure_summary="Synthetic pre-lock storage interruption.",
    )

    assert len(record["completed_model_artifacts"]) == 8
    assert record["boundary_status"]["2025_outcomes_accessed"] is False
    recovered, summary = _load_recovery_model_record(
        output,
        protocol_sha256=sha256_file(protocol_path),
        capacity_manifest_self_hash=manifest["manifest_sha256"],
    )
    assert len(recovered) == 8
    assert summary is not None and summary["model_count"] == 8


@pytest.mark.parametrize("year", [2025, 2026])
def test_recovery_rejects_outcome_year_in_nested_path(
    tmp_path: Path,
    year: int,
) -> None:
    failed_run = tmp_path / "failed_run"
    outcome_path = failed_run / "predictions" / f"year={year}" / "part.parquet"
    outcome_path.parent.mkdir(parents=True)
    outcome_path.write_bytes(b"not-opened-by-test")

    with pytest.raises(ValueError, match="not demonstrably pre-lock and 2024-only"):
        create_capacity_model_recovery_record(
            failed_run_dir=failed_run,
            protocol_path=Path("configs/flare24_ccrth_v1.toml"),
            capacity_manifest_path=tmp_path / "unused_manifest.json",
            expected_study_report_path=tmp_path / "absent_report.json",
            output_path=tmp_path / "recovery.json",
            failure_summary="Synthetic boundary violation.",
        )
