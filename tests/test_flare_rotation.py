from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flightdelaybench.flare_rotation import ROTATION_FEATURES, LatentRotationGraph


def _flight(
    *,
    date: str,
    origin: str,
    dest: str,
    departure: str,
    arrival: str,
    number: int,
    tail: str,
) -> dict[str, object]:
    return {
        "FlightDate": date,
        "Origin": origin,
        "Dest": dest,
        "Reporting_Airline": "XX",
        "Flight_Number_Reporting_Airline": number,
        "departure_time_utc": pd.Timestamp(departure),
        "arrival_time_utc": pd.Timestamp(arrival),
        "Tail_Number": tail,
    }


def _history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day in ("01", "02", "03"):
        rows.extend(
            [
                _flight(
                    date=f"2023-01-{day}",
                    origin="AAA",
                    dest="HUB",
                    departure=f"2023-01-{day} 08:00",
                    arrival=f"2023-01-{day} 09:00",
                    number=1,
                    tail=f"N{day}",
                ),
                _flight(
                    date=f"2023-01-{day}",
                    origin="HUB",
                    dest="BBB",
                    departure=f"2023-01-{day} 10:00",
                    arrival=f"2023-01-{day} 11:00",
                    number=2,
                    tail=f"N{day}",
                ),
            ]
        )
    return pd.DataFrame(rows)


def _target() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _flight(
                date="2024-01-01",
                origin="AAA",
                dest="HUB",
                departure="2024-01-01 08:00",
                arrival="2024-01-01 09:00",
                number=1,
                tail="SECRET-A",
            ),
            _flight(
                date="2024-01-01",
                origin="CCC",
                dest="HUB",
                departure="2024-01-01 08:10",
                arrival="2024-01-01 09:10",
                number=3,
                tail="SECRET-B",
            ),
            _flight(
                date="2024-01-01",
                origin="HUB",
                dest="BBB",
                departure="2024-01-01 10:00",
                arrival="2024-01-01 11:00",
                number=2,
                tail="SECRET-A",
            ),
            _flight(
                date="2024-01-01",
                origin="HUB",
                dest="DDD",
                departure="2024-01-01 10:05",
                arrival="2024-01-01 11:05",
                number=4,
                tail="SECRET-B",
            ),
        ]
    )


def test_inference_ignores_target_tail_numbers_and_is_coherent() -> None:
    model = LatentRotationGraph().fit(_history())
    target = _target()
    first = model.infer(target, inbound_disruption_risk=np.array([0.8, 0.2, 0.1, 0.1]))
    permuted = target.copy()
    permuted["Tail_Number"] = list(reversed(permuted["Tail_Number"].tolist()))
    second = model.infer(
        permuted,
        inbound_disruption_risk=np.array([0.8, 0.2, 0.1, 0.1]),
    )
    pd.testing.assert_frame_equal(first.features, second.features)
    assert tuple(first.features.columns) == ROTATION_FEATURES
    outbound_mass = first.edges.groupby("successor_position")["probability"].sum()
    inbound_mass = first.edges.groupby("predecessor_position")["probability"].sum()
    assert (outbound_mass <= 1.0 + 1e-9).all()
    assert (inbound_mass <= 1.0 + 1e-7).all()
    assert first.diagnostics["target_tail_number_present_but_ignored"]


def test_rotation_risk_uses_supplied_predictions() -> None:
    model = LatentRotationGraph().fit(_history())
    low = model.infer(_target(), inbound_disruption_risk=np.zeros(4)).features
    high = model.infer(_target(), inbound_disruption_risk=np.ones(4)).features
    connected = high["flare24_rotation_predecessor_probability"].gt(0.0)
    assert low.loc[connected, "flare24_rotation_inbound_disruption_risk"].eq(0.0).all()
    assert high.loc[connected, "flare24_rotation_inbound_disruption_risk"].eq(1.0).all()


def test_target_outcomes_are_rejected() -> None:
    model = LatentRotationGraph().fit(_history())
    with pytest.raises(ValueError, match="contains outcomes"):
        model.infer(_target().assign(Cancelled=0))


def test_model_card_discloses_training_only_tail_use() -> None:
    card = LatentRotationGraph().fit(_history()).model_card()
    assert card.learned_connections == 3
    assert card.target_tail_number_policy.startswith("ignored")
