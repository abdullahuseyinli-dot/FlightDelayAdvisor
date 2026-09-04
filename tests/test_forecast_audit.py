from __future__ import annotations

import numpy as np
import pytest

from flightdelaybench.forecast_audit import (
    _paired_probability_intervals,
    _require_audit_year,
)


def test_forecast_audit_year_is_fail_closed() -> None:
    _require_audit_year(2025)
    with pytest.raises(PermissionError, match="2025"):
        _require_audit_year(2026)
    with pytest.raises(PermissionError, match="2025"):
        _require_audit_year(2024)


def test_paired_probability_intervals_use_candidate_minus_reference() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    candidate = np.where(labels == 1, 0.9, 0.1).astype(np.float64)
    reference = np.full(len(labels), 0.5, dtype=np.float64)
    dates = np.asarray(["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"])
    intervals = _paired_probability_intervals(
        labels,
        candidate,
        reference,
        dates,
        repetitions=100,
        seed=7,
    )
    assert intervals["log_loss"]["estimate"] < 0
    assert intervals["brier"]["estimate"] < 0
