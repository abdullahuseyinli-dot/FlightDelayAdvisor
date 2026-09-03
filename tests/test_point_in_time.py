from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flightdelaybench.contracts import AvailabilityHorizon, features_available_at
from flightdelaybench.feature_validation import validate_feature_dataset
from flightdelaybench.point_in_time import (
    ClimatologyState,
    OutcomePriorState,
    build_dataset,
    prepare_base,
    transform_point_in_time,
)


def _rows(year: int, outcomes: list[tuple[float, int]]) -> pd.DataFrame:
    count = len(outcomes)
    dates = pd.date_range(f"{year}-01-01", periods=count, freq="D")
    frame = pd.DataFrame(
        {
            "Year": year,
            "Month": 1,
            "DayOfMonth": dates.day,
            "DayOfWeek": dates.dayofweek + 1,
            "DayOfYear": dates.dayofyear,
            "FlightDate": dates,
            "Reporting_Airline": ["AA"] * count,
            "Origin": ["AAA"] * count,
            "Dest": ["BBB"] * count,
            "Distance": [500.0] * count,
            "DepHour": [8] * count,
            "IsWeekend": [0] * count,
            "IsHolidaySeason": [1] * count,
            "DepHour_sin": [np.sin(2 * np.pi * 8 / 24)] * count,
            "DepHour_cos": [np.cos(2 * np.pi * 8 / 24)] * count,
            "Route": ["AAA_BBB"] * count,
            "DistanceBand": ["short"] * count,
            "ArrDel15": [outcome[0] for outcome in outcomes],
            "Cancelled": [outcome[1] for outcome in outcomes],
            "Origin_tavg": [10.0 + index for index in range(count)],
            "Origin_prcp": [1.0] * count,
            "Origin_snow": [0.0] * count,
            "Origin_wspd": [5.0] * count,
            "Dest_tavg": [20.0 + index for index in range(count)],
            "Dest_prcp": [2.0] * count,
            "Dest_snow": [0.0] * count,
            "Dest_wspd": [7.0] * count,
        }
    )
    frame.loc[frame["Cancelled"].eq(1), "ArrDel15"] = np.nan
    return frame


def _states(history: pd.DataFrame) -> tuple[OutcomePriorState, ClimatologyState]:
    outcome = OutcomePriorState()
    climate = ClimatologyState()
    outcome.update(history, year=int(history["Year"].iloc[0]))
    climate.update(history, year=int(history["Year"].iloc[0]))
    return outcome, climate


def test_current_year_outcomes_and_weather_cannot_change_deployable_features() -> None:
    history = _rows(2010, [(0.0, 0), (1.0, 0), (np.nan, 1)])
    target = _rows(2011, [(0.0, 0), (1.0, 0)])
    outcome, climate = _states(history)
    outcome.advance_to(2011)
    climate.advance_to(2011)

    original = transform_point_in_time(
        target,
        source_cohort="test",
        outcome_state=outcome,
        climatology_state=climate,
    )
    mutated = target.copy()
    mutated["ArrDel15"] = [1.0, 0.0]
    mutated["Cancelled"] = [0, 1]
    mutated.loc[1, "ArrDel15"] = np.nan
    mutated["Origin_tavg"] = 999.0
    changed = transform_point_in_time(
        mutated,
        source_cohort="test",
        outcome_state=outcome,
        climatology_state=climate,
    )

    columns = list(features_available_at(AvailabilityHorizon.SCHEDULE_CLIMATOLOGY))
    pd.testing.assert_frame_equal(original[columns], changed[columns])
    assert not original["oracle_origin_tavg"].equals(changed["oracle_origin_tavg"])


def test_prior_rates_use_only_history_and_cancelled_rows_have_no_delay_label() -> None:
    history = _rows(2010, [(0.0, 0), (1.0, 0), (np.nan, 1), (np.nan, 1)])
    outcome, _ = _states(history)
    target = prepare_base(_rows(2011, [(1.0, 0)]), source_cohort="test")
    prior = outcome.transform(target)

    assert prior.loc[0, "prior_global_delay_rate"] == pytest.approx(0.5)
    assert prior.loc[0, "prior_global_cancel_rate"] == pytest.approx(0.5)
    assert prior.loc[0, "prior_global_count"] == pytest.approx(4.0)
    assert prior.loc[0, "prior_route_delay_support"] == pytest.approx(2.0)


def test_climatology_counts_unique_airport_days_not_flights() -> None:
    history = pd.concat(
        [
            _rows(2010, [(0.0, 0)]),
            _rows(2010, [(0.0, 0)]),
            _rows(2010, [(0.0, 0)]).assign(
                FlightDate=pd.Timestamp("2010-01-02"),
                DayOfMonth=2,
                DayOfYear=2,
                Origin_tavg=20.0,
            ),
        ],
        ignore_index=True,
    )
    history.loc[:1, "Origin_tavg"] = 10.0
    climate = ClimatologyState()
    climate.update(history, year=2010)
    target = prepare_base(_rows(2011, [(0.0, 0)]), source_cohort="test")
    result = climate.transform(target)

    assert result.loc[0, "clim_origin_tavg"] == pytest.approx(15.0)
    assert result.loc[0, "clim_origin_tavg_missing"] == 0


def test_exponential_decay_ages_support_before_prediction_year() -> None:
    outcome = OutcomePriorState(half_life_years=2.0)
    outcome.update(_rows(2010, [(1.0, 0), (0.0, 0)]), year=2010)
    outcome.advance_to(2012)
    target = prepare_base(_rows(2012, [(0.0, 0)]), source_cohort="test")
    result = outcome.transform(target)
    assert result.loc[0, "prior_global_count"] == pytest.approx(1.0)
    assert result.loc[0, "prior_global_delay_rate"] == pytest.approx(0.5)


def test_unknown_legacy_arrival_outcome_is_retained_but_label_ineligible() -> None:
    frame = _rows(2018, [(np.nan, 0)])
    result = prepare_base(frame, source_cohort="legacy")
    assert result.loc[0, "delay_label_observed"] == 0
    assert result.loc[0, "joint_label_observed"] == 0
    assert result.loc[0, "disruption_state"] == -1


def test_small_build_is_create_only_and_self_hashed(tmp_path: Path) -> None:
    legacy = pd.concat(
        [_rows(2010, [(0.0, 0), (1.0, 0)]), _rows(2011, [(1.0, 0)])],
        ignore_index=True,
    )
    legacy_path = tmp_path / "legacy.parquet"
    legacy.to_parquet(legacy_path, index=False)
    output_dir = tmp_path / "features"
    manifest_path = tmp_path / "manifest.json"
    manifest = build_dataset(
        legacy_path=legacy_path,
        normalized_2025_dir=tmp_path / "unused",
        output_dir=output_dir,
        manifest_path=manifest_path,
        start_year=2011,
        end_year=2011,
    )

    assert manifest["total_output_rows"] == 1
    assert manifest["manifest_sha256"]
    built = pd.read_parquet(output_dir / "year=2011.parquet")
    assert built.loc[0, "prior_global_delay_rate"] == pytest.approx(0.5)
    report = validate_feature_dataset(
        manifest_path,
        report_path=tmp_path / "validation.json",
        repo_root=tmp_path,
    )
    assert report["status"] == "PASS"
    assert report["rows_verified"] == 1
    with pytest.raises(FileExistsError, match="overwrite feature manifest"):
        build_dataset(
            legacy_path=legacy_path,
            normalized_2025_dir=tmp_path / "unused",
            output_dir=output_dir,
            manifest_path=manifest_path,
            start_year=2011,
            end_year=2011,
        )
