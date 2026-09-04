from __future__ import annotations

import pandas as pd
import pytest

from flightdelaybench.boundary_rotation_models import (
    _verified_protocol,
    normalize_boundary_rotation_history_chunk,
)


def _raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [2023, 2023],
            "Month": [1, 1],
            "FlightDate": ["2023-01-01", "2023-01-01"],
            "Reporting_Airline": ["ZZ", "ZZ"],
            "Flight_Number_Reporting_Airline": [1, 2],
            "Origin": ["SM1", "SM1"],
            "Dest": ["HUB", "SM2"],
            "CRSDepTime": [700, 800],
            "CRSElapsedTime": [60.0, 70.0],
            "Distance": [300.0, 400.0],
            "Tail_Number": ["N1", "N2"],
        }
    )


def test_boundary_rotation_history_keeps_tail_only_for_prior_fit() -> None:
    result = normalize_boundary_rotation_history_chunk(
        _raw(),
        source_year=2023,
        source_month=1,
        source_offset=10,
        target_airports={"HUB"},
    )
    assert result["sample_id"].tolist() == ["bts-2023-01-0000010"]
    assert result["Tail_Number"].tolist() == ["N1"]
    assert "ArrDel15" not in result


def test_boundary_rotation_history_rejects_outcome_columns() -> None:
    raw = _raw()
    raw["ArrDel15"] = [0, 1]
    with pytest.raises(ValueError, match="outcome"):
        normalize_boundary_rotation_history_chunk(
            raw,
            source_year=2023,
            source_month=1,
            source_offset=0,
            target_airports={"HUB"},
        )


def test_boundary_rotation_protocol_binds_information_boundary(tmp_path) -> None:
    protocol = tmp_path / "protocol.toml"
    protocol.write_text(
        """
[identity]
method = "BC-POT-Rotation-v1"
[information_boundary]
history_years = [2023, 2024]
historical_outcomes_allowed = false
target_year_rows_allowed_during_fit = false
target_year_tail_number_allowed = false
confirmation_gate_opened = false
[model]
minimum_turn_minutes = 20.0
maximum_layover_minutes = 720.0
maximum_candidates = 12
history_end_month = 9
minimum_lag_to_earliest_target_cutoff_days = 91
""".strip(),
        encoding="utf-8",
    )
    result = _verified_protocol(
        protocol,
        history_years=(2023, 2024),
        minimum_turn_minutes=20.0,
        maximum_layover_minutes=720.0,
        maximum_candidates=12,
    )
    assert result["identity"]["method"] == "BC-POT-Rotation-v1"


def test_boundary_rotation_protocol_rejects_setting_drift(tmp_path) -> None:
    protocol = tmp_path / "protocol.toml"
    protocol.write_text(
        """
[identity]
method = "BC-POT-Rotation-v1"
[information_boundary]
history_years = [2023, 2024]
historical_outcomes_allowed = false
target_year_rows_allowed_during_fit = false
target_year_tail_number_allowed = false
confirmation_gate_opened = false
[model]
minimum_turn_minutes = 25.0
maximum_layover_minutes = 720.0
maximum_candidates = 12
history_end_month = 9
minimum_lag_to_earliest_target_cutoff_days = 91
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frozen protocol"):
        _verified_protocol(
            protocol,
            history_years=(2023, 2024),
            minimum_turn_minutes=20.0,
            maximum_layover_minutes=720.0,
            maximum_candidates=12,
        )
