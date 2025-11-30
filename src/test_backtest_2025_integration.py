"""
Smoke tests for the 2025 real-world backtest.

These are intentionally light and are skipped if 2025 data is not present,
so they won't break CI on machines that do not have data/raw_2025/.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RAW_2025_DIR = REPO_ROOT / "data" / "raw_2025"

from backtest_2025_real_world import load_bts_2025_raw  # type: ignore


@pytest.mark.skipif(
    not RAW_2025_DIR.exists(),
    reason="2025 raw data directory data/raw_2025/ not available.",
)
def test_can_load_and_map_2025_raw_columns() -> None:
    """
    Basic integration check that:
    - at least some 2025 data can be read
    - canonical columns exist after mapping
    """
    df = load_bts_2025_raw(RAW_2025_DIR, max_rows=1000)

    expected_cols = [
        "Year",
        "Month",
        "DayOfMonth",
        "DayOfWeek",
        "DepHour",
        "Reporting_Airline",
        "Origin",
        "Dest",
        "Distance",
        "Cancelled",
        "ArrDel15",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Expected column {col!r} missing in 2025 data."


# You could add more tests (e.g. building features + one model prediction),
# but this minimal smoke test is usually enough to guard against schema drift.
