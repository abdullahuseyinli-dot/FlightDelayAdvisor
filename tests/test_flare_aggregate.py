from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flightdelaybench.flare_aggregate import HierarchicalAggregateForecaster


def _history(repeats: int = 1) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        for day, state in enumerate((0, 0, 1, 2, 1, 0, 0), start=1):
            rows.append(
                {
                    "FlightDate": pd.Timestamp("2023-01-01")
                    + pd.Timedelta(days=day + repeat * 7 - 1),
                    "Origin": "AAA",
                    "Dest": "BBB",
                    "DepHour": 8,
                    "ArrHour": 10,
                    "Reporting_Airline": "XX",
                    "Route": "AAA-BBB",
                    "disruption_state": state,
                }
            )
    return pd.DataFrame(rows)


def _schedule(date: str = "2023-02-01", flights: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "FlightDate": [date] * flights,
            "Origin": ["AAA"] * flights,
            "Dest": ["BBB"] * flights,
            "DepHour": [8] * flights,
            "ArrHour": [10] * flights,
            "Reporting_Airline": ["XX"] * flights,
            "Route": ["AAA-BBB"] * flights,
        }
    )


def test_aggregate_forecasts_are_coherent_and_independent() -> None:
    model = HierarchicalAggregateForecaster().fit(_history())
    forecast = model.predict(_schedule())
    assert set(forecast["group_type"]) == {
        "origin_hour",
        "destination_hour",
        "carrier_day",
        "route_day",
    }
    means = forecast[["mean_on_time", "mean_delayed", "mean_cancelled"]].sum(axis=1)
    np.testing.assert_allclose(means, forecast["scheduled_count"])
    assert (forecast.filter(like="variance_") > 0.0).all().all()
    assert model.model_card().method.startswith("FLARE24-")


def test_target_outcomes_are_rejected_not_silently_consumed() -> None:
    model = HierarchicalAggregateForecaster().fit(_history())
    leaked = _schedule().assign(disruption_state=[2, 2, 2])
    with pytest.raises(ValueError, match="outcome columns"):
        model.predict(leaked)


def test_target_must_be_strictly_after_history() -> None:
    model = HierarchicalAggregateForecaster().fit(_history())
    with pytest.raises(ValueError, match="strictly after"):
        model.predict(_schedule("2023-01-07"))


def test_more_historical_support_reduces_predictive_variance() -> None:
    sparse_model = HierarchicalAggregateForecaster(half_life_days=10_000).fit(_history(1))
    dense_model = HierarchicalAggregateForecaster(half_life_days=10_000).fit(_history(5))
    target = _schedule("2023-03-01", flights=20)
    sparse = sparse_model.predict(target)
    dense = dense_model.predict(target)
    sparse_variance = sparse.filter(like="variance_").to_numpy().mean()
    dense_variance = dense.filter(like="variance_").to_numpy().mean()
    assert dense_variance < sparse_variance


def test_schedule_derived_route_and_hours_match_explicit_columns() -> None:
    model = HierarchicalAggregateForecaster().fit(_history())
    explicit = model.predict(_schedule())
    derived_schedule = _schedule().drop(columns=["Route", "DepHour", "ArrHour"]).assign(
        CRSDepMinutes=480,
        CRSArrMinutes=600,
    )
    derived = model.predict(derived_schedule)
    columns = [
        "group_type",
        "scheduled_count",
        "mean_on_time",
        "mean_delayed",
        "mean_cancelled",
    ]
    pd.testing.assert_frame_equal(explicit[columns], derived[columns])
