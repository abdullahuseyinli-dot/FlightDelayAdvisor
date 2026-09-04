from __future__ import annotations

import numpy as np
import pandas as pd

from flightdelaybench.contracts import SCHEDULE_CONTEXT_FEATURES
from flightdelaybench.schedule_context import attach_schedule_context


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "FlightDate": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-02"]
            ),
            "Reporting_Airline": ["AA", "AA", "DL", "AA"],
            "Origin": ["JFK", "JFK", "LAX", "JFK"],
            "Dest": ["LAX", "SFO", "JFK", "LAX"],
            "Route": ["JFK_LAX", "JFK_SFO", "LAX_JFK", "JFK_LAX"],
            "DepHour": [8, 8, 9, 8],
            "ArrDel15": [0, 1, 1, 0],
            "Cancelled": [0, 0, 1, 0],
        }
    )


def test_schedule_context_counts_full_target_day_before_sampling() -> None:
    result = attach_schedule_context(_schedule())
    first = result.iloc[0]
    assert np.isclose(first["schedule_global_day_log1p"], np.log1p(3))
    assert np.isclose(first["schedule_airline_day_log1p"], np.log1p(2))
    assert np.isclose(first["schedule_origin_outbound_day_log1p"], np.log1p(2))
    assert np.isclose(first["schedule_origin_inbound_day_log1p"], np.log1p(1))
    assert np.isclose(first["schedule_origin_departure_bank_log1p"], np.log1p(2))
    assert np.isclose(first["schedule_origin_bank_share"], 1.0)
    assert np.isclose(first["schedule_airline_origin_share"], 1.0)
    assert np.isclose(first["schedule_route_origin_share"], 0.5)


def test_schedule_context_is_invariant_to_outcome_values() -> None:
    original = _schedule()
    perturbed = original.copy()
    perturbed["ArrDel15"] = 1 - perturbed["ArrDel15"]
    perturbed["Cancelled"] = 1 - perturbed["Cancelled"]
    left = attach_schedule_context(original).loc[:, list(SCHEDULE_CONTEXT_FEATURES)]
    right = attach_schedule_context(perturbed).loc[:, list(SCHEDULE_CONTEXT_FEATURES)]
    pd.testing.assert_frame_equal(left, right)
