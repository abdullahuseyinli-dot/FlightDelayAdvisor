from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flightdelaybench.cutoff_experiments import (
    FOLDS,
    Trial,
    _task_rows,
    compare_prediction_tables,
    matched_trials,
    nested_monthly_sample,
    paired_diagnostics,
    run_trial,
    select_incumbent_blend,
    split_forward,
    validate_dataset,
    validate_new_features,
)

FEATURES = ("Origin", "Distance")


def synthetic_dataset() -> pd.DataFrame:
    dates = pd.to_datetime(np.repeat(["2024-02-20", "2024-03-10", "2024-04-04", "2024-04-15"], 36))
    labels = np.tile([0, 1, 2], len(dates) // 3)
    departure = dates.tz_localize("UTC") + pd.Timedelta(hours=15)
    result = pd.DataFrame({
        "sample_id": [f"synthetic-{i}" for i in range(len(dates))],
        "FlightDate": dates, "departure_time_utc": departure,
        "cutoff_time_utc": departure - pd.Timedelta(hours=24),
        "features_available_at_utc": departure - pd.Timedelta(hours=25),
        "cancel_label_available_at_utc": departure + pd.Timedelta(hours=3),
        "delay_label_available_at_utc": departure + pd.Timedelta(hours=3),
        "Cancelled": (labels == 2).astype(int), "ArrDel15": np.where(labels == 2, np.nan, labels),
        "joint_label_observed": 1, "delay_label_observed": (labels != 2).astype(int),
        "disruption_state": labels, "Origin": np.tile(["AAA", "BBB"], len(dates) // 2),
        "Distance": np.arange(len(dates)) % 17 + 100,
    })
    result.loc[result["Cancelled"].eq(1), "delay_label_available_at_utc"] = pd.NaT
    return result


def test_nested_monthly_caps_and_contexts_share_ids() -> None:
    frame = synthetic_dataset()
    small = nested_monthly_sample(frame, 5, 7)
    larger = nested_monthly_sample(frame.sample(frac=1, random_state=4), 12, 7)
    assert set(small.sample_id) < set(larger.sample_id)
    assert small.groupby(small.FlightDate.dt.to_period("M")).size().eq(5).all()
    changed_features = frame.assign(Distance=-999)
    assert nested_monthly_sample(changed_features, 5, 7).sample_id.tolist() == small.sample_id.tolist()
    assert len(nested_monthly_sample(frame, None, 7)) == len(frame)


def test_trial_grid_is_matched_not_repeated_or_confounded() -> None:
    trials = matched_trials()
    assert len(trials) == len({trial.name for trial in trials}) == 14
    assert {trial.context for trial in trials[:9]} == {"baseline", "induced", "boundary"}
    assert len({trial.seed for trial in trials}) == 1


@pytest.mark.parametrize("feature", ["recent_route_delay_rate_7d", "graph_origin_partner_count_log1p", "flare24_rotation_inbound_disruption_risk", "bcpot_rotation_predecessor_state_lost", "Cancelled"])
def test_legacy_or_label_predictors_rejected(feature: str) -> None:
    with pytest.raises(ValueError):
        validate_new_features((feature,))


@pytest.mark.parametrize("column", ["features_available_at_utc", "cancel_label_available_at_utc"])
def test_missing_timestamp_rejected(column: str) -> None:
    frame = synthetic_dataset()
    frame.loc[0, column] = pd.NaT
    with pytest.raises(ValueError, match="complete timezone-aware"):
        validate_dataset(frame, FEATURES)


def test_future_feature_and_invalid_joint_observation_rejected() -> None:
    frame = synthetic_dataset()
    frame.loc[0, "features_available_at_utc"] = frame.loc[0, "departure_time_utc"]
    with pytest.raises(ValueError, match="post-cutoff"):
        validate_dataset(frame, FEATURES)
    frame = synthetic_dataset()
    frame.loc[0, "delay_label_observed"] = 0
    with pytest.raises(ValueError, match="joint-observation"):
        validate_dataset(frame, FEATURES)


@pytest.mark.parametrize("label", [0, 1])
def test_observed_arrival_cannot_be_silently_removed_by_both_endpoint_masks(label: int) -> None:
    frame = synthetic_dataset()
    frame.loc[72, ["ArrDel15", "disruption_state"]] = label
    frame.loc[72, ["delay_label_observed", "joint_label_observed"]] = 0
    with pytest.raises(ValueError, match="delay-observation flag"):
        validate_dataset(frame, FEATURES)


def test_missing_arrival_remains_unobserved_without_removing_the_flight() -> None:
    frame = synthetic_dataset()
    frame.loc[72, "ArrDel15"] = np.nan
    frame.loc[72, "delay_label_available_at_utc"] = pd.NaT
    frame.loc[72, ["delay_label_observed", "joint_label_observed"]] = 0
    frame.loc[72, "disruption_state"] = -1
    result = validate_dataset(frame, FEATURES)
    pd.testing.assert_frame_equal(result, frame)


def test_missing_arrival_cannot_be_marked_observed() -> None:
    frame = synthetic_dataset()
    frame.loc[72, "ArrDel15"] = np.nan
    with pytest.raises(ValueError, match="delay-observation flag"):
        validate_dataset(frame, FEATURES)


def test_cancelled_flight_cannot_retain_an_arrival_label_behind_an_unobserved_mask() -> None:
    frame = synthetic_dataset()
    frame.loc[74, "ArrDel15"] = 0
    with pytest.raises(ValueError, match="conditional-delay labels must remain missing"):
        validate_dataset(frame, FEATURES)


@pytest.mark.parametrize("diverted", [1, None])
def test_represented_diversion_status_cannot_authorize_an_arrival_label(diverted: int | None) -> None:
    frame = synthetic_dataset()
    frame["Diverted"] = pd.Series(0, index=frame.index, dtype="Int8")
    frame.loc[72, "Diverted"] = diverted
    with pytest.raises(ValueError, match="conditional-delay labels must remain missing"):
        validate_dataset(frame, FEATURES)


def test_represented_diversion_keeps_schedule_and_cancellation_endpoint() -> None:
    frame = synthetic_dataset()
    frame["Diverted"] = pd.Series(0, index=frame.index, dtype="Int8")
    frame.loc[72, "Diverted"] = 1
    frame.loc[72, "ArrDel15"] = np.nan
    frame.loc[72, "delay_label_available_at_utc"] = pd.NaT
    frame.loc[72, ["delay_label_observed", "joint_label_observed"]] = 0
    frame.loc[72, "disruption_state"] = -1
    # Missing diversion status does not suppress an observed cancellation.
    frame.loc[74, "Diverted"] = pd.NA
    result = validate_dataset(frame, FEATURES)
    pd.testing.assert_frame_equal(result, frame)
    assert result.loc[74, "joint_label_observed"] == 1


def test_training_labels_must_be_available_before_stopping_predictions() -> None:
    frame = validate_dataset(synthetic_dataset(), FEATURES)
    train, stop, _, fit_at = split_forward(frame, FOLDS[0])
    train = train.copy()
    train.loc[train.index[0], "cancel_label_available_at_utc"] = pd.Timestamp("2024-03-20T00:00Z")
    honest, _ = _task_rows(train, "cancellation", stop.cutoff_time_utc.min())
    too_late, _ = _task_rows(train, "cancellation", fit_at)
    assert len(honest) + 1 == len(too_late)
    assert train.iloc[0].sample_id not in set(honest.sample_id)


def test_zero_weight_can_preserve_stronger_incumbent() -> None:
    labels = np.array([0, 1, 2])
    good = np.eye(3) * 0.7 + 0.1
    bad = np.full((3, 3), 1 / 3)
    assert select_incumbent_blend(labels, good, bad)["candidate_weight"] == 0
    assert select_incumbent_blend(labels, good, good)["candidate_weight"] == 0


def test_paired_metrics_use_dates_and_multiday_blocks() -> None:
    frame = synthetic_dataset().iloc[72:]
    probability = np.full((len(frame), 3), 1 / 3)
    result = paired_diagnostics(frame, probability, probability, repetitions=100)
    for cluster in ("date", "seven_day_block"):
        assert result[cluster]["accuracy"]["estimate"] == 0
        assert result[cluster]["joint_log_loss"]["clusters"] == 2


@pytest.mark.parametrize("family", ["catboost", "lightgbm", "tabm"])
@pytest.mark.parametrize("formulation", ["direct", "hurdle"])
def test_real_estimator_synthetic_end_to_end(tmp_path: Path, family: str, formulation: str) -> None:
    pytest.importorskip(family)
    trial = Trial("software-smoke", family, formulation, "baseline", 100, seed=7)
    output = tmp_path / "trial"
    if family == "tabm":
        torch = pytest.importorskip("torch")
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
    try:
        report = run_trial(synthetic_dataset(), trial=trial, fold=FOLDS[0], features=FEATURES, output_dir=output, iterations=3)
    finally:
        if family == "tabm":
            torch.set_num_threads(previous_threads)
    assert report["scores"]["rows"] == 72
    assert report["confirmation_outcomes_accessed"] is False
    probabilities = pd.read_parquet(output / "predictions.parquet")[["on_time", "delayed", "cancelled"]]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert (output / "models.joblib").is_file()
    with pytest.raises(FileExistsError):
        run_trial(synthetic_dataset(), trial=trial, fold=FOLDS[0], features=FEATURES, output_dir=output)


def test_failed_trial_preserves_failure_and_intent(tmp_path: Path) -> None:
    trial = Trial("bad-family-test", "unknown", "direct", "baseline", 100)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="unknown model family"):
        run_trial(synthetic_dataset(), trial=trial, fold=FOLDS[0], features=FEATURES, output_dir=output, iterations=1)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["status"] == "FAILED_DEVELOPMENT_TRIAL"
    assert (output / "intent.json").is_file()
    assert not (output / "report.json").exists()


def test_prediction_comparison_aligns_ids_and_rejects_cohort_changes() -> None:
    frame = synthetic_dataset().iloc[72:].copy()
    frame[["on_time", "delayed", "cancelled"]] = np.full((len(frame), 3), 1 / 3)
    paired = compare_prediction_tables(frame, frame.sample(frac=1, random_state=2), repetitions=100)
    assert paired["date"]["accuracy"]["estimate"] == 0
    with pytest.raises(ValueError, match="cohorts differ"):
        compare_prediction_tables(frame, frame.iloc[1:], repetitions=100)
    changed = frame.copy()
    changed.iloc[0, changed.columns.get_loc("disruption_state")] = 2
    with pytest.raises(ValueError, match="labels differ"):
        compare_prediction_tables(frame, changed, repetitions=100)
