from __future__ import annotations

import pytest

from flightdelaybench.flare_boundary_reporting import primary_metrics_table
from flightdelaybench.flare_boundary_study_validation import METHODS


def test_primary_metrics_table_uses_absolute_accuracy_differences() -> None:
    report = {
        "primary_evaluation": {
            "methods": {
                method: {
                    "joint": {
                        "n": 10,
                        "log_loss": 0.5 - 0.001 * index,
                        "multiclass_brier": 0.3 - 0.001 * index,
                    }
                }
                for index, method in enumerate(METHODS)
            }
        },
        "primary_argmax_decision_metrics": {
            method: {"accuracy": 0.75 + 0.01 * index, "balanced_accuracy": 0.5}
            for index, method in enumerate(METHODS)
        },
    }
    table = primary_metrics_table(report).set_index("method")
    selected = table.loc["boundary_gated_ensemble"]
    assert selected["absolute_accuracy_gain_vs_schedule"] == pytest.approx(0.07)
    assert selected["accuracy_percentage_point_gain_vs_schedule"] == pytest.approx(7.0)
    assert selected["absolute_accuracy_gain_vs_previous_meta"] == pytest.approx(0.04)
