"""Fast tests for the 2025 BTS schema-normalization stage."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
SRC = REPOSITORY / "src"
sys.path.insert(0, str(SRC))

from prepare_bts_2025_for_backtest import clean_2025_file


def test_clean_2025_file_normalizes_schema(tmp_path: Path) -> None:
    source = tmp_path / "bts_sample.csv"
    pd.DataFrame(
        [
            {
                "Year": 2025,
                "Month": 1,
                "DayOfWeek": 3,
                "FlightDate": "2025-01-15",
                "Reporting_Airline": "AA",
                "Origin": "JFK",
                "Dest": "LAX",
                "CRSDepTime": 930,
                "Distance": 2475.0,
                "ArrDel15": 1.0,
                "Cancelled": 0,
                "Diverted": 0,
                "OriginState": "NY",
                "DestState": "CA",
            },
            {
                "Year": 2025,
                "Month": 1,
                "DayOfWeek": 4,
                "FlightDate": "2025-01-16",
                "Reporting_Airline": "DL",
                "Origin": "ATL",
                "Dest": "BOS",
                "CRSDepTime": 2400,
                "Distance": 946.0,
                "ArrDel15": None,
                "Cancelled": 1,
                "Diverted": 0,
                "OriginState": "GA",
                "DestState": "MA",
            },
            {
                "Year": 2025,
                "Month": 1,
                "DayOfWeek": 5,
                "FlightDate": "2025-01-17",
                "Reporting_Airline": "UA",
                "Origin": "ORD",
                "Dest": "SFO",
                "CRSDepTime": 1430,
                "Distance": 1846.0,
                "ArrDel15": 0.0,
                "Cancelled": 0,
                "Diverted": 1,
                "OriginState": "IL",
                "DestState": "CA",
            },
        ]
    ).to_csv(source, index=False)

    result = clean_2025_file(source)

    assert len(result) == 2
    assert result["DepHour"].tolist() == [9, 0]
    assert result["Route"].tolist() == ["JFK_LAX", "ATL_BOS"]
    assert result["Cancelled"].tolist() == [0, 1]
    assert result["OriginState"].tolist() == ["NY", "GA"]
    assert {"DayOfMonth", "DayOfYear", "Season", "DistanceBand"}.issubset(result)


def test_clean_2025_file_rejects_diverted_rows(tmp_path: Path) -> None:
    source = tmp_path / "diverted.csv"
    pd.DataFrame(
        [
            {
                "Year": 2025,
                "Month": 2,
                "DayOfWeek": 1,
                "FlightDate": "2025-02-03",
                "Reporting_Airline": "WN",
                "Origin": "DAL",
                "Dest": "HOU",
                "CRSDepTime": 800,
                "Distance": 239.0,
                "ArrDel15": 0.0,
                "Cancelled": 0,
                "Diverted": 1,
            }
        ]
    ).to_csv(source, index=False)

    assert clean_2025_file(source).empty
