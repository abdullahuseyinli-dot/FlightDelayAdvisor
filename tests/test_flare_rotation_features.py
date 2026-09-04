from __future__ import annotations

import numpy as np
import pandas as pd

from flightdelaybench.contracts import FLARE24_ROTATION_FEATURES
from flightdelaybench.flare_rotation import LatentRotationGraph
from flightdelaybench.flare_rotation_features import infer_rotation_period


def _scheduled(
    sample_id: str,
    date: str,
    origin: str,
    dest: str,
    departure_minutes: int,
    number: int,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "FlightDate": date,
        "Origin": origin,
        "Dest": dest,
        "Reporting_Airline": "XX",
        "Flight_Number_Reporting_Airline": number,
        "CRSDepMinutes": departure_minutes,
        "CRSElapsedTime": 60.0,
        "Distance": 400.0,
    }


def _history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day in range(1, 4):
        date = f"2023-01-{day:02d}"
        first = _scheduled(f"h{day}a", date, "AAA", "HUB", 8 * 60, 1)
        first["Tail_Number"] = f"N{day}"
        second = _scheduled(f"h{day}b", date, "HUB", "BBB", 10 * 60, 2)
        second["Tail_Number"] = f"N{day}"
        rows.extend([first, second])
    return pd.DataFrame(rows)


def test_period_materialization_is_one_to_one_and_structural_only() -> None:
    model = LatentRotationGraph().fit(_history(), timezone_by_airport={"AAA": "UTC", "HUB": "UTC", "BBB": "UTC"})
    target = pd.DataFrame(
        [
            _scheduled("t-in", "2024-01-02", "AAA", "HUB", 8 * 60, 1),
            _scheduled("t-out", "2024-01-02", "HUB", "BBB", 10 * 60, 2),
        ]
    )
    context = pd.concat(
        [
            pd.DataFrame(
                [_scheduled("previous", "2024-01-01", "BBB", "AAA", 12 * 60, 3)]
            ),
            target,
            pd.DataFrame(
                [_scheduled("next", "2024-01-03", "BBB", "AAA", 12 * 60, 3)]
            ),
        ],
        ignore_index=True,
    )
    features, diagnostics = infer_rotation_period(
        model,
        target,
        context,
        timezone_by_airport={"AAA": "UTC", "HUB": "UTC", "BBB": "UTC"},
    )
    assert features["sample_id"].tolist() == ["t-in", "t-out"]
    assert tuple(features.columns[1:]) == FLARE24_ROTATION_FEATURES
    assert features["flare24_rotation_inbound_disruption_risk"].isna().all()
    assert features.loc[1, "flare24_rotation_predecessor_probability"] > 0.0
    assert diagnostics[0]["target_outcomes_accessed"] is False


def test_period_materialization_propagates_only_supplied_prior_risk() -> None:
    model = LatentRotationGraph().fit(
        _history(),
        timezone_by_airport={"AAA": "UTC", "HUB": "UTC", "BBB": "UTC"},
    )
    target = pd.DataFrame(
        [
            _scheduled("t-in", "2024-01-02", "AAA", "HUB", 8 * 60, 1),
            _scheduled("t-out", "2024-01-02", "HUB", "BBB", 10 * 60, 2),
        ]
    )
    context = target.copy()
    features, diagnostics = infer_rotation_period(
        model,
        target,
        context,
        timezone_by_airport={"AAA": "UTC", "HUB": "UTC", "BBB": "UTC"},
        context_disruption_risk=np.array([0.75, 0.10]),
    )
    assert np.isclose(
        features.loc[1, "flare24_rotation_inbound_disruption_risk"],
        0.75,
    )
    assert diagnostics[0]["inbound_risk_source"] == "supplied model prediction"
