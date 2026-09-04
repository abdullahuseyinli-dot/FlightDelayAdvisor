from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from flightdelaybench.contracts import FLARE24_WEATHER_FEATURES
from flightdelaybench.flare_weather import (
    OPEN_METEO_UNITS,
    OPEN_METEO_VARIABLES,
    attach_flare24_weather,
    normalize_open_meteo_request,
    select_latest_safe_forecast,
    validate_weather_cube,
)


def _weather_row(
    airport: str,
    valid: str,
    issue: str,
    lead: float,
    value: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "Airport": airport,
        "valid_time_utc": pd.Timestamp(valid),
        "issue_time_utc": pd.Timestamp(issue),
        "lead_hours": lead,
    }
    row.update({name: value for name in OPEN_METEO_VARIABLES})
    return row


def test_selector_rejects_late_issue_and_uses_older_safe_run() -> None:
    cube = pd.DataFrame(
        [
            _weather_row("AAA", "2024-01-02 12:00", "2024-01-01 12:00", 24, 1.0),
            _weather_row("BBB", "2024-01-02 14:00", "2024-01-01 14:00", 24, 9.0),
            _weather_row("BBB", "2024-01-02 14:00", "2023-12-31 14:00", 48, 2.0),
        ]
    )
    targets = pd.DataFrame(
        {
            "row_id": [0],
            "Airport": ["BBB"],
            "target_time_utc": [pd.Timestamp("2024-01-02 14:30")],
            "cutoff_time_utc": [pd.Timestamp("2024-01-01 12:30")],
        }
    )
    selected = select_latest_safe_forecast(targets, cube)
    assert selected.loc[0, "temperature"] == 2.0
    assert selected.loc[0, "lead_hours"] == 48.0
    assert selected.loc[0, "issue_time_utc"] <= selected.loc[0, "cutoff_time_utc"]


def test_selector_uses_freshest_safe_value_per_variable() -> None:
    older = _weather_row("AAA", "2025-01-02 12:00", "2024-12-31 12:00", 48, 10.0)
    fresher = _weather_row("AAA", "2025-01-02 12:00", "2025-01-01 12:00", 24, 20.0)
    older["visibility"] = 8_000.0
    fresher["visibility"] = np.nan
    selected = select_latest_safe_forecast(
        pd.DataFrame(
            {
                "row_id": [0],
                "Airport": ["AAA"],
                "target_time_utc": [pd.Timestamp("2025-01-02 12:10")],
                "cutoff_time_utc": [pd.Timestamp("2025-01-01 12:00")],
            }
        ),
        pd.DataFrame([older, fresher]),
    )
    assert selected.loc[0, "temperature"] == 20.0
    assert selected.loc[0, "visibility"] == 8_000.0
    assert selected.loc[0, "issue_time_utc"] == pd.Timestamp("2025-01-01 12:00")
    assert selected.loc[0, "lead_hours"] == 24.0


def test_attach_uses_one_cutoff_for_origin_and_destination() -> None:
    rows = []
    for hour, value in ((6, 1.0), (9, 2.0), (12, 3.0)):
        rows.append(
            _weather_row(
                "AAA",
                f"2024-01-02 {hour:02d}:00",
                f"2024-01-01 {hour:02d}:00",
                24,
                value,
            )
        )
    rows.extend(
        [
            _weather_row("BBB", "2024-01-02 14:00", "2024-01-01 14:00", 24, 99.0),
            _weather_row("BBB", "2024-01-02 14:00", "2023-12-31 14:00", 48, 5.0),
        ]
    )
    flights = pd.DataFrame(
        {
            "sample_id": ["one"],
            "FlightDate": [pd.Timestamp("2024-01-02")],
            "Origin": ["AAA"],
            "Dest": ["BBB"],
            "CRSDepMinutes": [12 * 60 + 30],
            "CRSElapsedTime": [120.0],
        }
    )
    result = attach_flare24_weather(
        flights,
        pd.DataFrame(rows),
        timezone_by_airport={"AAA": "UTC", "BBB": "UTC"},
    )
    assert set(FLARE24_WEATHER_FEATURES).issubset(result.columns)
    assert result.loc[0, "flare24_origin_departure_temperature"] == 3.0
    assert result.loc[0, "flare24_dest_arrival_temperature"] == 5.0
    assert result.loc[0, "flare24_dest_selected_lead_hours"] == 48.0
    assert result.loc[0, "flare24_origin_window_temperature_mean"] == 2.0
    assert result.loc[0, "flare24_origin_window_temperature_change"] == 2.0
    assert result.loc[0, "flare24_cutoff_coherent_valid"] == 1
    assert result.loc[0, "flare24_origin_revision_available"] == 0


def test_attach_exposes_only_cutoff_safe_origin_revision() -> None:
    rows = []
    for hour in (6, 9, 12):
        rows.extend(
            [
                _weather_row(
                    "AAA",
                    f"2024-01-02 {hour:02d}:00",
                    f"2024-01-01 {hour:02d}:00",
                    24,
                    10.0,
                ),
                _weather_row(
                    "AAA",
                    f"2024-01-02 {hour:02d}:00",
                    f"2023-12-31 {hour:02d}:00",
                    48,
                    4.0,
                ),
            ]
        )
    rows.append(
        _weather_row("BBB", "2024-01-02 14:00", "2023-12-31 14:00", 48, 5.0)
    )
    flights = pd.DataFrame(
        {
            "FlightDate": [pd.Timestamp("2024-01-02")],
            "Origin": ["AAA"],
            "Dest": ["BBB"],
            "CRSDepMinutes": [12 * 60 + 30],
            "CRSElapsedTime": [120.0],
        }
    )
    result = attach_flare24_weather(
        flights,
        pd.DataFrame(rows),
        timezone_by_airport={"AAA": "UTC", "BBB": "UTC"},
    )
    assert result.loc[0, "flare24_origin_revision_available"] == 1
    assert (
        result.loc[0, "flare24_origin_revision_temperature_fresh_minus_day2"]
        == 6.0
    )


def test_normalize_open_meteo_response_records_implied_issue_time(tmp_path: Path) -> None:
    hourly: dict[str, list[object]] = {"time": ["2024-01-01T00:00", "2024-01-01T01:00"]}
    hourly_units: dict[str, str] = {"time": "iso8601"}
    for stem in OPEN_METEO_VARIABLES.values():
        hourly[f"{stem}_previous_day1"] = [1.0, 2.0]
    for name, stem in OPEN_METEO_VARIABLES.items():
        hourly_units[f"{stem}_previous_day1"] = OPEN_METEO_UNITS[name]
    path = tmp_path / "raw.json"
    path.write_text(
        json.dumps({"timezone": "UTC", "hourly": hourly, "hourly_units": hourly_units}),
        encoding="utf-8",
    )
    frame, invalid = normalize_open_meteo_request(
        path,
        airport="AAA",
        lead_suffix="previous_day1",
        lead_hours=24,
    )
    assert invalid == 0
    assert frame.loc[0, "issue_time_utc"] == pd.Timestamp("2023-12-31 00:00")
    assert np.isclose(frame.loc[1, "temperature"], 2.0)
    assert validate_weather_cube(frame)["rows"] == 2
