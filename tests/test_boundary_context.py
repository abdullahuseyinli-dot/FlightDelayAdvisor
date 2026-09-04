from __future__ import annotations

import pandas as pd
import pytest

from flightdelaybench.boundary_context import (
    CONTEXT_COLUMNS,
    FORBIDDEN_CONTEXT_COLUMNS,
    normalize_boundary_schedule_chunk,
)


def _raw_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [2024, 2024, 2024],
            "Month": [1, 1, 1],
            "FlightDate": ["2024-01-02"] * 3,
            "Reporting_Airline": ["AA", "DL", "UA"],
            "Flight_Number_Reporting_Airline": [1, 2, 3],
            "Origin": ["AAA", "CCC", "DDD"],
            "Dest": ["BBB", "AAA", "EEE"],
            "CRSDepTime": [5, 1230, 2400],
            "CRSElapsedTime": [60, 90, 100],
            "Distance": [200, 400, 500],
        }
    )


def test_boundary_normalization_keeps_induced_and_boundary_with_raw_ids() -> None:
    result = normalize_boundary_schedule_chunk(
        _raw_schedule(),
        source_year=2024,
        source_month=1,
        source_offset=100,
        target_airports={"AAA", "BBB"},
    )

    assert tuple(result.columns) == CONTEXT_COLUMNS
    assert result["sample_id"].tolist() == [
        "bts-2024-01-0000100",
        "bts-2024-01-0000101",
    ]
    assert result["context_class"].tolist() == ["induced", "boundary"]
    assert result["CRSDepMinutes"].tolist() == [5, 750]
    assert not (set(result.columns) & FORBIDDEN_CONTEXT_COLUMNS)


def test_boundary_normalization_rejects_operational_columns() -> None:
    raw = _raw_schedule()
    raw["Cancelled"] = 0
    with pytest.raises(ValueError, match="forbidden operational columns"):
        normalize_boundary_schedule_chunk(
            raw,
            source_year=2024,
            source_month=1,
            source_offset=0,
            target_airports={"AAA", "BBB"},
        )
