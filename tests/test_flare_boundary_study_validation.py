from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flightdelaybench.flare_boundary_study import _joint_columns
from flightdelaybench.flare_boundary_study_validation import (
    METHODS,
    RECOVERY_STATUS,
    SELECTION_METHODS,
    _audit_probability_frame,
    _validate_recovery_contract,
)
from flightdelaybench.hashing import sha256_file


def _prediction_frame() -> pd.DataFrame:
    payload: dict[str, object] = {
        "sample_id": ["a", "b"],
        "FlightDate": pd.to_datetime(["2025-01-03", "2025-01-03"]),
        "joint_label_observed": [1, 1],
        "disruption_state": [0, 1],
    }
    for method in METHODS:
        for column, values in zip(
            _joint_columns(method),
            ([0.7, 0.2], [0.2, 0.7], [0.1, 0.1]),
            strict=True,
        ):
            payload[column] = values
    return pd.DataFrame(payload)


def test_probability_artifact_audit_accepts_simplex(tmp_path: Path) -> None:
    path = tmp_path / "prediction.parquet"
    _prediction_frame().to_parquet(path, index=False)
    frame, audit = _audit_probability_frame(path)
    assert len(frame) == 2
    assert np.isclose(audit["maximum_absolute_probability_sum_error"], 0.0)


def test_probability_artifact_audit_rejects_drift(tmp_path: Path) -> None:
    frame = _prediction_frame()
    frame.loc[0, _joint_columns(METHODS[0])[0]] = 0.8
    path = tmp_path / "prediction.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="drift"):
        _audit_probability_frame(path)


def test_selection_probability_audit_does_not_require_test_only_references(
    tmp_path: Path,
) -> None:
    frame = _prediction_frame().drop(
        columns=[
            column
            for method in set(METHODS) - set(SELECTION_METHODS)
            for column in _joint_columns(method)
        ]
    )
    path = tmp_path / "selection.parquet"
    frame.to_parquet(path, index=False)
    selected, _ = _audit_probability_frame(path, methods=SELECTION_METHODS)
    assert len(selected) == 2


def test_recovery_contract_verifies_exact_duplicate_and_prediction_errors(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    recovery_run = tmp_path / "recovery"
    source.mkdir()
    recovery_run.mkdir()
    lock = source / "method_lock.json"
    lock.write_text("lock", encoding="utf-8")
    report_path = tmp_path / "report.json"
    report_path.write_text("report", encoding="utf-8")
    (recovery_run / "run_manifest.json").write_text("report", encoding="utf-8")
    failure = tmp_path / "failure.json"
    failure.write_text("failure", encoding="utf-8")
    report = {
        "recovery": {
            "status": RECOVERY_STATUS,
            "models_refit": False,
            "calibrators_refit_for_prediction": False,
            "predictions_regenerated": False,
            "selection_reconstructed_from_locked_q4_predictions": True,
            "metrics_recomputed_from_locked_2025_predictions": True,
            "failure_manifest": {
                "path": failure.as_posix(),
                "bytes": failure.stat().st_size,
                "sha256": sha256_file(failure),
            },
            "source_run_directory": source.as_posix(),
            "recovery_run_directory": recovery_run.as_posix(),
            "selection_artifact_verification": {
                "exact_reconstructed_sample_id_order": True,
                "rows": 7,
                "probability_reproduction_atol": 2e-6,
                "maximum_absolute_prediction_errors": {"model": 2e-8},
            },
            "retrospective_artifact_verification": {
                "months": 12,
                "rows": 11,
                "all_predictions_immutable_and_reproduced": True,
                "probability_reproduction_atol": 2e-6,
                "global_ensemble_maximum_absolute_error": 3e-8,
                "gated_ensemble_maximum_absolute_error": 4e-8,
            },
        }
    }
    audit = _validate_recovery_contract(
        report,
        report_path=report_path,
        lock_path=lock,
        selection_rows=7,
        audit_rows=11,
    )
    assert audit is not None
    assert audit["maximum_retrospective_prediction_error"] == pytest.approx(4e-8)


def test_recovery_contract_rejects_regenerated_predictions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="changed a predictive artifact"):
        _validate_recovery_contract(
            {
                "recovery": {
                    "status": RECOVERY_STATUS,
                    "models_refit": False,
                    "calibrators_refit_for_prediction": False,
                    "predictions_regenerated": True,
                    "selection_reconstructed_from_locked_q4_predictions": True,
                    "metrics_recomputed_from_locked_2025_predictions": True,
                }
            },
            report_path=tmp_path / "report.json",
            lock_path=tmp_path / "lock.json",
            selection_rows=7,
            audit_rows=11,
        )
