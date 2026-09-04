from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from flightdelaybench.flare_study import (
    CANDIDATES,
    _reconciliation_scale_selection,
    select_forward_calibration,
    select_simplex_ensemble,
    write_flare24_method_lock,
)
from flightdelaybench.hashing import canonical_json_sha256, sha256_file


def _selection_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-10-01", "2024-12-31", freq="D").repeat(4)
    labels = np.tile([0, 1, 0, 1], len(dates) // 4)
    return pd.DataFrame(
        {
            "FlightDate": dates,
            "Cancelled": np.zeros(len(dates), dtype=np.int8),
            "delay_label_observed": np.ones(len(dates), dtype=np.int8),
            "ArrDel15": labels,
        }
    )


def test_forward_calibration_never_scores_before_training_window() -> None:
    frame = _selection_frame()
    raw = np.where(frame["ArrDel15"].eq(1), 0.7, 0.3)
    selection = select_forward_calibration(
        frame,
        raw,
        task="delay",
        methods=("identity", "intercept", "platt"),
    )
    dates = pd.to_datetime(frame["FlightDate"])
    assert not selection.crossfit_mask[dates.lt("2024-10-16")].any()
    assert selection.crossfit_mask[dates.ge("2024-10-16")].all()
    assert np.isfinite(
        selection.crossfit_probabilities[selection.crossfit_mask]
    ).all()
    assert selection.method in {"identity", "intercept", "platt"}


def test_simplex_ensemble_selects_dominant_well_scored_candidate() -> None:
    labels = np.tile(np.array([0, 1, 2]), 20)
    correct = np.eye(3)[labels] * 0.88 + 0.04
    medium = np.eye(3)[labels] * 0.55 + 0.15
    poor = np.full((len(labels), 3), 1.0 / 3.0)
    result = select_simplex_ensemble(
        labels,
        dict(
            zip(
                CANDIDATES,
                (poor, medium, medium, correct),
                strict=True,
            )
        ),
        step=0.1,
    )
    assert result["selected"]["weights"] == {
        "baseline": 0.0,
        "weather": 0.0,
        "rotation_structural": 0.0,
        "rotation_risk": 1.0,
    }


def test_method_lock_copies_frozen_choices_without_opening_audit(tmp_path: Path) -> None:
    model_artifacts: list[dict[str, object]] = []
    calibrator_artifacts: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        for task in ("delay", "cancellation"):
            model = tmp_path / f"{candidate}-{task}-model.bin"
            model.write_bytes(b"model")
            model_artifacts.append(
                {
                    "candidate": candidate,
                    "task": task,
                    "path": model.as_posix(),
                    "sha256": sha256_file(model),
                }
            )
            calibrator = tmp_path / f"{candidate}-{task}-calibrator.bin"
            calibrator.write_bytes(b"calibrator")
            calibrator_artifacts.append(
                {
                    "candidate": candidate,
                    "task": task,
                    "path": calibrator.as_posix(),
                    "sha256": sha256_file(calibrator),
                }
            )
    weights = {candidate: float(candidate == "baseline") for candidate in CANDIDATES}
    report: dict[str, object] = {
        "status": "COMPLETE_2024_FLARE24_SELECTION_NOT_CONFIRMATORY",
        "feature_sets": {candidate: [] for candidate in CANDIDATES},
        "model_artifacts": model_artifacts,
        "calibration": {"artifacts": calibrator_artifacts},
        "ensemble_selection": {"selected": {"weights": weights}},
        "reconciliation_selection": {
            "identity_candidate_included": True,
            "selected_mode": "soft_marginal_alignment",
        },
        "selected_variance_multiplier": 1.0,
        "diagnostic_weather_severity": {"selection_quartile_cutpoints": [0.1, 0.2, 0.3]},
        "outcomes_accessed": {
            "maximum_calendar_date": "2024-12-31",
            "2025_accessed_by_this_run": False,
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    report_path = tmp_path / "selection.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    lock = write_flare24_method_lock(
        report_path,
        output_path=tmp_path / "lock.json",
    )
    assert lock["outcomes_accessed_by_lock"]["2025"] is False
    assert lock["confirmation_gate"]["opened"] is False


def test_reconciliation_selection_retains_identity_control() -> None:
    flights = pd.DataFrame(
        {"joint_label_observed": [1, 1], "disruption_state": [0, 1]}
    )
    probabilities = np.asarray([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    result = _reconciliation_scale_selection(
        flights,
        probabilities,
        pd.DataFrame(),
        candidates=(),
    )
    assert result["identity_candidate_included"] is True
    assert result["selected_mode"] == "identity_no_alignment"
    assert result["selected_variance_multiplier"] is None
