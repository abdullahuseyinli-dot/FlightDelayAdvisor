from __future__ import annotations

import pandas as pd

from flightdelaybench.census_context_audit import (
    _distribution,
    _expected_dates,
    classify_network_scope,
)


def test_classify_network_scope_is_exhaustive() -> None:
    frame = pd.DataFrame(
        {
            "Origin": ["AAA", "AAA", "CCC", "DDD"],
            "Dest": ["BBB", "CCC", "AAA", "EEE"],
        }
    )
    scope = classify_network_scope(frame, {"AAA", "BBB"})
    assert scope.sum().to_dict() == {"induced": 1, "boundary": 2, "outside": 1}
    assert scope.sum(axis=1).eq(1).all()


def test_expected_dates_retains_leap_days() -> None:
    dates = _expected_dates((2023, 2024))
    assert len(dates) == 731
    assert pd.Timestamp("2024-02-29") in dates


def test_distribution_preserves_counts_and_quantiles() -> None:
    result = _distribution(pd.Series([1, 2, 3, 4, 5]))
    assert result["min"] == 1
    assert result["median"] == 3.0
    assert result["max"] == 5
