from __future__ import annotations

import pytest
import requests

from flightdelaybench.flare_weather_acquisition import (
    SUPPORTED_STEMS,
    _hourly_variables,
    _retry_delay,
    validate_response,
)


def test_day2_response_contract_includes_aviation_variables() -> None:
    variables = _hourly_variables(2)
    assert "wind_direction_10m_previous_day2" in variables
    assert "visibility_previous_day2" in variables
    assert "snowfall_previous_day2" in variables
    assert len(variables) == len(SUPPORTED_STEMS)


def test_response_validation_rejects_wrong_hour_count() -> None:
    variables = _hourly_variables(2)
    hourly = {"time": ["2024-01-01T00:00"]}
    hourly.update({name: [1.0] for name in variables})
    with pytest.raises(ValueError, match="unexpected hourly lengths"):
        validate_response(
            {"timezone": "UTC", "hourly": hourly},
            airport={"iata": "AAA", "timezone": "UTC"},
            year=2024,
            variables=variables,
        )


def test_rate_limit_retry_is_longer_and_bounded() -> None:
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = "45"
    assert _retry_delay(response, 0) == 45.0
    assert _retry_delay(response, 8) == 300.0
    assert _retry_delay(None, 8) == 20.0
