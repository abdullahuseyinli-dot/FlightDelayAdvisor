from __future__ import annotations

import pandas as pd

from flightdelaybench.forecast_calibration import (
    CALIBRATION_FOLDS,
    _forward_calibration_folds,
    _select_method,
)


def test_forward_calibration_folds_are_disjoint_and_strictly_ordered() -> None:
    dates = pd.Series(pd.date_range("2024-10-01", "2024-12-31", freq="D"))
    folds = _forward_calibration_folds(dates)
    validation_positions: set[int] = set()
    assert len(folds) == len(CALIBRATION_FOLDS) == 3
    for train, validation in folds:
        train_dates = dates.loc[train]
        validation_dates = dates.loc[validation]
        assert train_dates.max() < validation_dates.min()
        current = set(validation_dates.index)
        assert not validation_positions & current
        validation_positions.update(current)


def test_calibration_selection_uses_log_loss_then_simplicity_order() -> None:
    records = [
        {"method": "platt", "pooled_forward_metrics": {"log_loss": 0.20}},
        {"method": "identity", "pooled_forward_metrics": {"log_loss": 0.20}},
        {"method": "intercept", "pooled_forward_metrics": {"log_loss": 0.19}},
    ]
    assert _select_method(records) == "intercept"
    records[2]["pooled_forward_metrics"]["log_loss"] = 0.20
    assert _select_method(records) == "identity"
