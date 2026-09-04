from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from flightdelaybench.flare_features import SCHEDULE_INPUT_COLUMNS


def test_feature_materializer_schedule_contract_excludes_outcomes() -> None:
    forbidden = {
        "ArrDel15",
        "Cancelled",
        "Tail_Number",
        "disruption_state",
        "delay_label_observed",
        "joint_label_observed",
    }
    assert forbidden.isdisjoint(SCHEDULE_INPUT_COLUMNS)


def test_top100_catalog_has_timezones_for_materializer() -> None:
    path = Path("data/external/forecast24_openmeteo_gfs_v1/airport_catalog.json")
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["airports"])
    assert len(frame) == 100
    assert frame["timezone"].notna().all()
    assert frame["iata"].is_unique
