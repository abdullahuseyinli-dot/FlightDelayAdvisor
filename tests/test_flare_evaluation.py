from __future__ import annotations

import numpy as np
import pandas as pd

from flightdelaybench.flare_evaluation import (
    evaluate_joint_probabilities,
    joint_loss_rows,
    reconcile_joint_by_date,
)


def _flights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "FlightDate": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3),
            "Origin": ["A"] * 6,
            "Dest": ["B"] * 6,
            "DepHour": [8] * 6,
            "ArrHour": [10] * 6,
            "Reporting_Airline": ["X"] * 6,
            "Route": ["A-B"] * 6,
            "disruption_state": [0, 1, 2, 0, 1, 0],
            "joint_label_observed": [1] * 6,
            "Cancelled": [0, 0, 1, 0, 0, 0],
            "delay_label_observed": [1, 1, 0, 1, 1, 1],
            "ArrDel15": [0, 1, np.nan, 0, 1, 0],
        }
    )


def _aggregates() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date in pd.to_datetime(["2024-01-01", "2024-01-02"]):
        for group_type, keys in (
            ("origin_hour", {"Origin": "A", "DepHour": 8}),
            ("destination_hour", {"Dest": "B", "ArrHour": 10}),
            ("carrier_day", {"Reporting_Airline": "X"}),
            ("route_day", {"Route": "A-B"}),
        ):
            rows.append(
                {
                    "FlightDate": date,
                    "group_type": group_type,
                    **keys,
                    "mean_on_time": 1.2,
                    "mean_delayed": 1.2,
                    "mean_cancelled": 0.6,
                    "variance_on_time": 0.5,
                    "variance_delayed": 0.5,
                    "variance_cancelled": 0.3,
                }
            )
    return pd.DataFrame(rows)


def test_reconcile_by_date_preserves_simplex() -> None:
    flights = _flights()
    base = np.tile(np.array([0.8, 0.15, 0.05]), (len(flights), 1))
    result = reconcile_joint_by_date(flights, base, _aggregates())
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0)
    assert len(result.date_diagnostics) == 2
    assert not np.allclose(result.probabilities, base)


def test_joint_evaluation_uses_paired_date_clusters() -> None:
    flights = _flights()
    weak = np.tile(np.array([0.7, 0.2, 0.1]), (len(flights), 1))
    good = np.eye(3)[flights["disruption_state"].to_numpy(dtype=np.int64)] * 0.8 + 0.2 / 3
    result = evaluate_joint_probabilities(
        flights,
        {"weak": weak, "good": good},
        reference_method="weak",
        bootstrap_repetitions=100,
    )
    assert result["methods"]["good"]["joint"]["log_loss"] < result["methods"]["weak"]["joint"]["log_loss"]
    interval = result["paired_date_cluster_comparisons"]["good_minus_weak"]["joint_log_loss"]
    assert interval["estimate"] < 0.0
    assert interval["clusters"] == 2


def test_joint_loss_rows_is_strictly_proper_for_example() -> None:
    labels = np.array([0, 1, 2])
    perfect = np.eye(3) * 0.98 + 0.02 / 3
    diffuse = np.full((3, 3), 1.0 / 3.0)
    assert joint_loss_rows(labels, perfect)[0].mean() < joint_loss_rows(labels, diffuse)[0].mean()
