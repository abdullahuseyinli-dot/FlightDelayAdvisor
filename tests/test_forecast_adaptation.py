from __future__ import annotations

import pandas as pd

from flightdelaybench.forecast_adaptation import _split_selection_year


def test_adaptation_split_is_chronological_and_date_disjoint() -> None:
    dates = pd.to_datetime(
        [
            "2024-01-01",
            "2024-08-31",
            "2024-09-01",
            "2024-09-30",
            "2024-10-01",
            "2024-12-31",
        ]
    )
    frame = pd.DataFrame({"FlightDate": dates, "sample_id": range(len(dates))})

    train, validation, selection = _split_selection_year(frame)

    assert train["sample_id"].tolist() == [0, 1]
    assert validation["sample_id"].tolist() == [2, 3]
    assert selection["sample_id"].tolist() == [4, 5]
    assert pd.to_datetime(train["FlightDate"]).max() < pd.to_datetime(validation["FlightDate"]).min()
    assert pd.to_datetime(validation["FlightDate"]).max() < pd.to_datetime(
        selection["FlightDate"]
    ).min()
