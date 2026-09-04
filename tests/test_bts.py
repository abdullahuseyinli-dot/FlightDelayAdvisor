from __future__ import annotations

from zipfile import ZipFile

import pandas as pd

from flightdelaybench.bts import (
    iter_normalized_archive,
    materialize_archive,
    normalize_bts_chunk,
    normalize_crs_hour,
    normalize_crs_minutes,
)


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
    assert normalize_crs_minutes(pd.Series([0, 59, 930, 2359, 2400])).tolist() == [
        0,
        59,
        570,
        1439,
        0,
    ]


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


def test_census_mode_retains_diversion_for_schedule_but_masks_outcome() -> None:
    rows = _raw_rows()
    rows.loc[0, "Diverted"] = 1
    result = normalize_bts_chunk(
        rows,
        source_year=2025,
        source_month=1,
        source_offset=0,
        retain_diverted=True,
    )

    assert len(result) == 2
    assert result.loc[0, "Diverted"] == 1
    assert result.loc[0, "joint_label_observed"] == 0
    assert result.loc[0, "disruption_state"] == -1
    assert pd.isna(result.loc[0, "ArrDel15"])
    assert result.loc[0, "ScheduledFlightId"] == "AA_10"


def test_archive_source_ids_advance_once_across_chunks(tmp_path) -> None:
    archive_path = tmp_path / "month.zip"
    raw = pd.concat([_raw_rows(), _raw_rows()], ignore_index=True)
    csv_path = tmp_path / "month.csv"
    raw.to_csv(csv_path, index=False)
    with ZipFile(archive_path, "w") as archive:
        archive.write(csv_path, arcname="month.csv")

    chunks = list(
        iter_normalized_archive(
            archive_path,
            year=2025,
            month=1,
            chunksize=2,
            retain_diverted=True,
        )
    )

    assert [value for chunk in chunks for value in chunk["sample_id"]] == [
        "bts-2025-01-0000000",
        "bts-2025-01-0000001",
        "bts-2025-01-0000002",
        "bts-2025-01-0000003",
    ]


def test_archive_materialization_stabilizes_flight_number_schema(tmp_path) -> None:
    archive_path = tmp_path / "month.zip"
    raw = pd.concat([_raw_rows(), _raw_rows()], ignore_index=True)
    # The first CSV chunk is integral while the second contains the first
    # missing identifier, reproducing the BTS 2024-08 schema transition.
    raw.loc[2, "Flight_Number_Reporting_Airline"] = pd.NA
    csv_path = tmp_path / "month.csv"
    raw.to_csv(csv_path, index=False)
    with ZipFile(archive_path, "w") as archive:
        archive.write(csv_path, arcname="month.csv")

    output_path = tmp_path / "month.parquet"
    summary = materialize_archive(
        archive_path,
        output_path,
        year=2025,
        month=1,
        chunksize=2,
        retain_diverted=True,
    )
    materialized = pd.read_parquet(output_path)

    assert summary["rows"] == 4
    assert materialized["Flight_Number_Reporting_Airline"].dtype == "int64"
    assert materialized["Flight_Number_Reporting_Airline"].tolist() == [10, 20, -1, 20]
    assert materialized["ScheduledFlightId"].tolist() == ["AA_10", "DL_20", "AA_-1", "DL_20"]
