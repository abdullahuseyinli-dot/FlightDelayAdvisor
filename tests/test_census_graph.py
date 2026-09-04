from __future__ import annotations

import numpy as np
import pandas as pd

from flightdelaybench.census_graph import schedule_graph_messages
from flightdelaybench.contracts import CENSUS_GRAPH_MESSAGE_FEATURES


def _lookup(level: str) -> pd.DataFrame:
    key = "Origin" if level == "origin" else "Dest"
    frame = pd.DataFrame(
        {
            "FlightDate": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            key: ["A", "B", "C"],
        }
    )
    for outcome, base in (("delay", 0.1), ("cancel", 0.01)):
        for window in (7, 28, 90):
            frame[f"recent_{level}_{outcome}_rate_{window}d"] = [
                base,
                base + 0.1,
                base + 0.2,
            ]
    return frame


def test_schedule_graph_messages_are_frequency_weighted_and_outcome_blind() -> None:
    schedule = pd.DataFrame(
        {
            "FlightDate": pd.to_datetime(["2024-01-02"] * 4),
            "Origin": ["A", "A", "A", "B"],
            "Dest": ["B", "B", "C", "A"],
            "ArrDel15": [0, 1, 1, 0],
        }
    )
    global_lookup = pd.DataFrame({"FlightDate": pd.to_datetime(["2024-01-02"])})
    for outcome, value in (("delay", 0.15), ("cancel", 0.02)):
        for window in (7, 28, 90):
            global_lookup[f"recent_global_{outcome}_rate_{window}d"] = value

    origin, dest = schedule_graph_messages(
        schedule,
        global_lookup=global_lookup,
        origin_lookup=_lookup("origin"),
        dest_lookup=_lookup("dest"),
    )
    row = origin.loc[origin["Origin"].eq("A")].iloc[0]
    assert np.isclose(row["graph_origin_partner_delay_mean_7d"], (0.2 + 0.2 + 0.3) / 3)
    assert np.isclose(row["graph_origin_partner_delay_max_7d"], 0.3)
    assert np.isclose(row["graph_origin_partner_count_log1p"], np.log1p(2))
    assert set(CENSUS_GRAPH_MESSAGE_FEATURES).issubset(set(origin) | set(dest))

    perturbed = schedule.copy()
    perturbed["ArrDel15"] = 1 - perturbed["ArrDel15"]
    other_origin, other_dest = schedule_graph_messages(
        perturbed,
        global_lookup=global_lookup,
        origin_lookup=_lookup("origin"),
        dest_lookup=_lookup("dest"),
    )
    pd.testing.assert_frame_equal(origin, other_origin)
    pd.testing.assert_frame_equal(dest, other_dest)
