from __future__ import annotations

import pytest

from flightdelaybench.cutoff_weather_pilot import covers_horizon, parse_taf_header


def test_expiry_is_exclusive_and_later_issue_cannot_fill_gap() -> None:
    record = parse_taf_header("KDSM 150532Z 1506/1606 28007KT P6SM FEW250=", "KDSM", "202401150532-KDMX-FTUS43-TAFDSM")
    assert not covers_horizon(record, "2024-01-15T06:00Z", 24)
    assert covers_horizon(record, "2024-01-15T06:00Z", 23)
    assert not covers_horizon(record, "2024-01-15T05:00Z", 24)
    assert not covers_horizon(record, "2024-01-15T06:00Z", 23, latency_minutes=60)


def test_month_rollover_and_hour_24_are_explicit() -> None:
    record = parse_taf_header("KJFK 311730Z 3118/0124 28007KT P6SM FEW250=", "KJFK", "202401311730-KOKX-FTUS41-TAFJFK")
    assert record["valid_end_utc"].startswith("2024-02-02T00:00")
    assert covers_horizon(record, "2024-01-31T18:00Z", 27)


def test_cancelled_taf_is_unavailable() -> None:
    record = parse_taf_header("KDSM 150532Z 1506/1606 CNL=", "KDSM", "202401150532-KDMX-FTUS43-TAFDSM")
    assert not covers_horizon(record, "2024-01-15T06:00Z", 12)


def test_segment_without_header_and_unauthorized_year_rejected() -> None:
    with pytest.raises(ValueError, match="header missing"):
        parse_taf_header("FM151700 28007KT P6SM FEW250", "KDSM", "202401150532-KDMX-FTUS43-TAFDSM")
    with pytest.raises(ValueError, match="2024"):
        parse_taf_header("KDSM 150532Z 1506/1606 CNL=", "KDSM", "202601150532-KDMX-FTUS43-TAFDSM")
