from __future__ import annotations

import pandas as pd

from flightdelaybench.bts import normalize_bts_chunk, normalize_crs_hour


def _raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Year": 2025,
                "Month": 1,
                "DayofMonth": 1,
                "DayOfWeek": 3,
                "FlightDate": "2025-01-01",
                "Reporting_Airline": "AA",
                "DOT_ID_Reporting_Airline": 1,
                "Flight_Number_Reporting_Airline": 10,
                "Tail_Number": "N1",
                "OriginAirportID": 100,
                "Origin": "JFK",
                "OriginState": "NY",
                "DestAirportID": 200,
                "Dest": "LAX",
                "DestState": "CA",
                "CRSDepTime": 2400,
                "CRSArrTime": 300,
                "CRSElapsedTime": 360,
                "ArrDel15": 1,
                "Cancelled": 0,
                "Diverted": 0,
                "Distance": 2475,
            },
            {
                "Year": 2025,
                "Month": 1,
                "DayofMonth": 2,
                "DayOfWeek": 4,
                "FlightDate": "2025-01-02",
                "Reporting_Airline": "DL",
                "DOT_ID_Reporting_Airline": 2,
                "Flight_Number_Reporting_Airline": 20,
                "Tail_Number": "N2",
                "OriginAirportID": 300,
                "Origin": "ATL",
                "OriginState": "GA",
                "DestAirportID": 400,
                "Dest": "BOS",
                "DestState": "MA",
                "CRSDepTime": 930,
                "CRSArrTime": 1200,
                "CRSElapsedTime": 150,
                "ArrDel15": 1,
                "Cancelled": 1,
                "Diverted": 0,
                "Distance": 946,
            },
        ]
    )


def test_normalize_crs_hour_handles_2400() -> None:
    assert normalize_crs_hour(pd.Series([0, 59, 930, 2359, 2400])).tolist() == [0, 0, 9, 23, 0]


def test_normalization_has_stable_ids_and_no_cancelled_delay_label() -> None:
    result = normalize_bts_chunk(
        _raw_rows(),
        source_year=2025,
        source_month=1,
        source_offset=100,
    )
    assert result["sample_id"].tolist() == ["bts-2025-01-0000100", "bts-2025-01-0000101"]
    assert result["DepHour"].tolist() == [0, 9]
    assert result["Route"].tolist() == ["JFK_LAX", "ATL_BOS"]
    assert pd.isna(result.loc[1, "ArrDel15"])


def test_airport_cohort_filter_preserves_source_row_identity() -> None:
    result = normalize_bts_chunk(
        _raw_rows(),
        source_year=2025,
        source_month=1,
        source_offset=20,
        allowed_airports={"JFK", "LAX"},
    )
    assert result["sample_id"].tolist() == ["bts-2025-01-0000020"]
