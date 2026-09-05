from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flightdelaybench.cutoff_history import (
    assert_safe_history_attachment,
    build_cutoff_history,
)


def _targets() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": ["target"], "FlightDate": pd.to_datetime(["2024-01-02"]),
        "cutoff_time_utc": pd.to_datetime(["2024-01-01T13:00:00Z"]),
        "departure_time_utc": pd.to_datetime(["2024-01-02T13:00:00Z"]),
        "Route": ["JFK-LAX"], "Origin": ["JFK"], "Dest": ["LAX"],
        "Reporting_Airline": ["AA"], "ScheduledFlightId": ["AA-1"],
    })


def _observations() -> pd.DataFrame:
    frame = pd.DataFrame({
        "sample_id": ["early", "late", "unpublished"],
        "FlightDate": pd.to_datetime(["2024-01-01"] * 3),
        "event_time_utc": pd.to_datetime(["2024-01-01T10:00Z", "2024-01-02T04:00Z", "2024-01-01T11:00Z"]),
        "source_published_at_utc": pd.to_datetime(["2024-01-01T10:01Z", "2024-01-02T04:01Z", "2024-01-01T14:00Z"]),
        "available_at_utc": pd.to_datetime(["2024-01-01T10:02Z", "2024-01-02T04:02Z", "2024-01-01T14:01Z"]),
        "source_id": ["test-feed"] * 3, "outcome": ["delay"] * 3, "value": [0, 1, 1],
        "Route": ["JFK-LAX"] * 3, "Origin": ["JFK"] * 3, "Dest": ["LAX"] * 3,
        "Reporting_Airline": ["AA"] * 3, "ScheduledFlightId": ["AA-1"] * 3,
    })
    return frame


def test_prior_date_does_not_admit_late_or_unpublished_outcome() -> None:
    history, audit = build_cutoff_history(_targets(), _observations())
    assert history.loc[0, "asof_route_delay_rate_7d"] == 0
    assert history.loc[0, "asof_route_delay_support_log1p_7d"] == pytest.approx(np.log(2))
    assert history.loc[0, "history_max_available_at_utc"] == pd.Timestamp("2024-01-01T10:02Z")
    assert audit["global_90d_observation_target_pairs"] == 1
    assert np.isnan(history.loc[0, "asof_global_cancel_rate_7d"])
    assert history.loc[0, "asof_global_cancel_support_log1p_7d"] == 0


def test_future_outcome_mutation_does_not_change_features() -> None:
    events = _observations()
    first, _ = build_cutoff_history(_targets(), events)
    events.loc[[1, 2], "value"] = 0
    second, _ = build_cutoff_history(_targets(), events)
    pd.testing.assert_frame_equal(first, second)


def test_exact_cutoff_allowed_but_later_ingestion_not_allowed() -> None:
    events = _observations().iloc[:1].copy()
    events.loc[0, "available_at_utc"] = pd.Timestamp("2024-01-01T13:00Z")
    first, _ = build_cutoff_history(_targets(), events)
    assert first.loc[0, "asof_global_delay_support_log1p_7d"] > 0
    events.loc[0, "available_at_utc"] += pd.Timedelta(microseconds=1)
    second, audit = build_cutoff_history(_targets(), events)
    assert second.loc[0, "asof_global_delay_support_log1p_7d"] == 0
    assert audit["rows_without_history"] == 1


def test_same_operating_date_excluded_even_if_cancellation_announced_early() -> None:
    events = _observations().iloc[:1].copy()
    events.loc[0, "FlightDate"] = pd.Timestamp("2024-01-02")
    events.loc[0, "outcome"] = "cancel"
    result, _ = build_cutoff_history(_targets(), events)
    assert result.loc[0, "asof_global_cancel_support_log1p_7d"] == 0


@pytest.mark.parametrize("column", ["event_time_utc", "source_published_at_utc", "available_at_utc", "source_id"])
def test_timestamp_evidence_must_not_be_invented(column: str) -> None:
    with pytest.raises(ValueError, match="missing required"):
        build_cutoff_history(_targets(), _observations().drop(columns=column))


def test_naive_timestamp_rejected() -> None:
    events = _observations()
    events["available_at_utc"] = events["available_at_utc"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        build_cutoff_history(_targets(), events)


def test_publication_cannot_precede_event() -> None:
    events = _observations()
    events.loc[0, "source_published_at_utc"] = pd.Timestamp("2023-12-31T00:00Z")
    with pytest.raises(ValueError, match="event <= publication"):
        build_cutoff_history(_targets(), events)


def test_confirmation_year_rejected() -> None:
    targets = _targets()
    targets["FlightDate"] = pd.Timestamp("2026-01-02")
    with pytest.raises(ValueError, match="2026"):
        build_cutoff_history(targets, _observations())


def test_duplicate_observations_rejected() -> None:
    events = _observations()
    with pytest.raises(ValueError, match="duplicate observation"):
        build_cutoff_history(_targets(), pd.concat([events, events]))


@pytest.mark.parametrize("delay", [0, 1])
def test_cancelled_flight_cannot_contribute_a_conditional_delay(delay: int) -> None:
    events = _observations().iloc[:1].copy()
    events["value"] = delay
    cancellation = events.assign(outcome="cancel", value=1)
    with pytest.raises(ValueError, match="cancelled flights cannot contribute"):
        build_cutoff_history(_targets(), pd.concat([events, cancellation], ignore_index=True))


def test_operated_flight_can_contribute_both_tasks() -> None:
    events = _observations().iloc[:1].copy()
    cancellation = events.assign(outcome="cancel", value=0)
    result, _ = build_cutoff_history(_targets(), pd.concat([events, cancellation], ignore_index=True))
    assert result.loc[0, "asof_global_delay_rate_7d"] == 0
    assert result.loc[0, "asof_global_cancel_rate_7d"] == 0


@pytest.mark.parametrize(("column", "value"), [
    ("FlightDate", pd.Timestamp("2023-12-31")),
    ("Route", "SFO-LAX"), ("Origin", "SFO"), ("Dest", "SFO"),
    ("Reporting_Airline", "UA"), ("ScheduledFlightId", "AA-2"),
])
def test_cross_task_identity_cannot_move_one_flight_between_history_groups(column: str, value: object) -> None:
    events = _observations().iloc[:1].copy()
    cancellation = events.assign(outcome="cancel", value=0)
    cancellation[column] = value
    with pytest.raises(ValueError, match="observation sample ID has conflicting"):
        build_cutoff_history(_targets(), pd.concat([events, cancellation], ignore_index=True))


def test_target_metadata_conflict_cannot_contaminate_a_later_target_history() -> None:
    earlier = _targets()
    earlier["sample_id"] = "early"
    earlier["FlightDate"] -= pd.Timedelta(days=1)
    earlier["departure_time_utc"] -= pd.Timedelta(days=1)
    earlier["cutoff_time_utc"] -= pd.Timedelta(days=1)
    targets = pd.concat([earlier, _targets()], ignore_index=True)
    events = _observations().iloc[:1].copy()
    events["Origin"] = "SFO"
    with pytest.raises(ValueError, match="target/observation sample ID has conflicting group keys"):
        build_cutoff_history(targets, events)


@pytest.mark.parametrize("diverted", [1, None])
def test_represented_diversion_cannot_enter_conditional_delay_history(diverted: int | None) -> None:
    events = _observations().iloc[:1].copy()
    events["Diverted"] = pd.Series([diverted], dtype="Int8")
    with pytest.raises(ValueError, match="require known nondiverted status"):
        build_cutoff_history(_targets(), events)


def test_diversion_does_not_remove_cancellation_history() -> None:
    events = _observations().iloc[:1].copy().assign(outcome="cancel", value=0, Diverted=1)
    result, _ = build_cutoff_history(_targets(), events)
    assert result.loc[0, "asof_global_cancel_rate_7d"] == 0
    assert np.isnan(result.loc[0, "asof_global_delay_rate_7d"])


def test_cross_task_diversion_conflict_is_rejected() -> None:
    events = _observations().iloc[:1].copy().assign(Diverted=0)
    cancellation = events.assign(outcome="cancel", value=0, Diverted=1)
    with pytest.raises(ValueError, match="conflicting diversion status"):
        build_cutoff_history(_targets(), pd.concat([events, cancellation], ignore_index=True))


def test_join_requires_matching_cutoff_and_preserves_target_order() -> None:
    targets = _targets()
    history, _ = build_cutoff_history(targets, _observations())
    attached = assert_safe_history_attachment(targets, history)
    assert attached["sample_id"].tolist() == ["target"]
    history["cutoff_time_utc"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="different prediction cutoff"):
        assert_safe_history_attachment(targets, history)


def test_window_boundaries_use_availability_not_flight_date() -> None:
    events = _observations().iloc[:1].copy()
    events["FlightDate"] = pd.Timestamp("2023-12-20")
    for column in ("event_time_utc", "source_published_at_utc", "available_at_utc"):
        events[column] = pd.Timestamp("2023-12-25T13:00Z")
    inside, _ = build_cutoff_history(_targets(), events)
    assert inside.loc[0, "asof_global_delay_support_log1p_7d"] > 0
    for column in ("event_time_utc", "source_published_at_utc", "available_at_utc"):
        events[column] -= pd.Timedelta(seconds=1)
    outside, _ = build_cutoff_history(_targets(), events)
    assert outside.loc[0, "asof_global_delay_support_log1p_7d"] == 0
    assert outside.loc[0, "asof_global_delay_support_log1p_28d"] > 0


def test_conflicting_target_identity_rejected() -> None:
    events = _observations()
    events.loc[0, "sample_id"] = "target"
    with pytest.raises(ValueError, match="conflicting operating dates"):
        build_cutoff_history(_targets(), events)


def test_attachment_rejects_silent_replacement_and_missing_provenance() -> None:
    targets = _targets()
    history, _ = build_cutoff_history(targets, _observations())
    attached = assert_safe_history_attachment(targets, history)
    with pytest.raises(ValueError, match="already contains history"):
        assert_safe_history_attachment(attached, history)
    history["history_max_available_at_utc"] = pd.NaT
    with pytest.raises(ValueError, match="matching availability evidence"):
        assert_safe_history_attachment(targets, history)
