from __future__ import annotations

import numpy as np
import pandas as pd

from flightdelaybench.shift_recalibration import operational_shift_matrix


def _prediction_frame(task: str, rows: int = 5) -> pd.DataFrame:
    outcome = "delay" if task == "delay" else "cancel"
    values: dict[str, object] = {
        "probability": np.linspace(0.1, 0.3, rows),
        "baseline_global": np.repeat(0.18, rows),
        "baseline_route": np.linspace(0.12, 0.22, rows),
        "Month": np.arange(1, rows + 1),
        "ArrDel15": np.arange(rows) % 2,
        "Cancelled": np.zeros(rows),
    }
    for view in ("global", "route", "airline", "origin_outbound", "dest_inbound"):
        for window in (7, 28, 90):
            values[f"recent_{view}_{outcome}_rate_{window}d"] = np.linspace(
                0.08 + window / 10_000,
                0.28 + window / 10_000,
                rows,
            )
    return pd.DataFrame(values)


def test_operational_shift_matrix_is_finite_and_outcome_blind() -> None:
    frame = _prediction_frame("delay")
    original = operational_shift_matrix(frame, "delay")
    frame["ArrDel15"] = 1 - frame["ArrDel15"]
    frame["Cancelled"] = 1
    perturbed = operational_shift_matrix(frame, "delay")
    assert original.shape == (5, 18)
    assert np.isfinite(original).all()
    np.testing.assert_allclose(original, perturbed)
