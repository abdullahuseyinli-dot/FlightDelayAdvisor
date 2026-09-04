from __future__ import annotations

import pandas as pd

from flightdelaybench.contracts import FLARE24_CORRIDOR_FEATURES
from flightdelaybench.flare_corridor import attach_corridor_weather_features
from flightdelaybench.flare_weather import OPEN_METEO_VARIABLES


def _row(airport: str, valid: str, issue: str, value: float) -> dict[str, object]:
    row: dict[str, object] = {
        "Airport": airport,
        "valid_time_utc": pd.Timestamp(valid),
        "issue_time_utc": pd.Timestamp(issue),
        "lead_hours": 48.0,
    }
    row.update({name: value for name in OPEN_METEO_VARIABLES})
    return row


def test_corridor_selects_midroute_proxy_under_flight_cutoff() -> None:
    flights = pd.DataFrame(
        {
            "FlightDate": [pd.Timestamp("2024-01-03")],
            "Origin": ["AAA"],
            "Dest": ["CCC"],
            "CRSDepMinutes": [12 * 60],
            "CRSElapsedTime": [240.0],
        }
    )
    weather = pd.DataFrame(
        [
            _row("AAA", "2024-01-03 13:00", "2024-01-01 13:00", 1.0),
            _row("BBB", "2024-01-03 14:00", "2024-01-01 14:00", 5.0),
            _row("BBB", "2024-01-03 15:00", "2024-01-01 15:00", 4.0),
            _row("CCC", "2024-01-03 15:00", "2024-01-01 15:00", 2.0),
        ]
    )
    result = attach_corridor_weather_features(
        flights,
        weather,
        timezone_by_airport={"AAA": "UTC", "BBB": "UTC", "CCC": "UTC"},
        airport_coordinates={
            "AAA": (0.0, 0.0),
            "BBB": (0.0, 10.0),
            "CCC": (0.0, 20.0),
        },
        fractions=(0.25, 0.5, 0.75),
    )
    assert tuple(result.loc[:, list(FLARE24_CORRIDOR_FEATURES)].columns) == FLARE24_CORRIDOR_FEATURES
    assert result.loc[0, "flare24_route_corridor_proxy_cape_max"] == 5.0
    assert result.loc[0, "flare24_route_corridor_proxy_coverage"] == 1.0
    assert result.loc[0, "flare24_route_corridor_proxy_distance_max_km"] > 0.0
