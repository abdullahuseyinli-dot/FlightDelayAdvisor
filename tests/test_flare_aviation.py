from __future__ import annotations

import numpy as np
import pandas as pd

from flightdelaybench.contracts import FLARE24_AVIATION_WEATHER_FEATURES
from flightdelaybench.flare_aviation import (
    attach_aviation_weather_features,
    runway_heading_from_identifier,
)


def _weather_frame() -> pd.DataFrame:
    values: dict[str, object] = {
        "Origin": ["AAA"],
        "Dest": ["BBB"],
        "schedule_origin_departure_bank_log1p": [2.0],
        "schedule_dest_inbound_day_log1p": [3.0],
    }
    for side, event in (("origin", "departure"), ("dest", "arrival")):
        prefix = f"flare24_{side}_{event}"
        values.update(
            {
                f"{prefix}_temperature": [0.0],
                f"{prefix}_precipitation": [1.0],
                f"{prefix}_wind_speed": [18.52],
                f"{prefix}_wind_gust": [37.04],
                f"{prefix}_wind_direction": [90.0],
                f"{prefix}_cape": [1_000.0],
                f"{prefix}_visibility": [1_000.0 if side == "dest" else 10_000.0],
                f"{prefix}_snowfall": [0.2],
                f"{prefix}_freezing_level": [1_000.0],
            }
        )
    return pd.DataFrame(values)


def test_runway_identifier_heading() -> None:
    assert runway_heading_from_identifier("09L") == 90.0
    assert runway_heading_from_identifier("36") == 360.0


def test_aviation_weather_components_and_load_interactions() -> None:
    result = attach_aviation_weather_features(
        _weather_frame(),
        runway_headings_by_airport={"AAA": (90.0, 270.0), "BBB": (90.0, 270.0)},
    )
    assert set(FLARE24_AVIATION_WEATHER_FEATURES).issubset(result.columns)
    assert np.isclose(result.loc[0, "flare24_origin_wind_optimal_headwind_knots"], 10.0)
    assert np.isclose(result.loc[0, "flare24_origin_wind_optimal_crosswind_knots"], 0.0)
    assert result.loc[0, "flare24_dest_visibility_hazard"] == 3.0
    assert result.loc[0, "flare24_origin_convective_bank_load"] > 0.0
    assert result.loc[0, "flare24_dest_icing_inbound_load"] > 0.0


def test_outcomes_do_not_change_aviation_transform() -> None:
    base = _weather_frame()
    first = attach_aviation_weather_features(
        base.assign(ArrDel15=0, Cancelled=0),
        runway_headings_by_airport={"AAA": (90.0,), "BBB": (90.0,)},
    )
    second = attach_aviation_weather_features(
        base.assign(ArrDel15=1, Cancelled=1),
        runway_headings_by_airport={"AAA": (90.0,), "BBB": (90.0,)},
    )
    pd.testing.assert_frame_equal(
        first.loc[:, list(FLARE24_AVIATION_WEATHER_FEATURES)],
        second.loc[:, list(FLARE24_AVIATION_WEATHER_FEATURES)],
    )
