from __future__ import annotations

import json
from pathlib import Path

from flightdelaybench.flare_capacity_factorized_reporting import (
    _load_inputs,
    component_grid_table,
    component_weight_table,
    monthly_score_table,
    paired_interval_table,
    primary_score_table,
    q4_selection_table,
    regime_score_table,
)

REPORT_PATH = Path("reports/experiments/flare24_ccrth_task_factorized_v1.json")
VALIDATION_PATH = Path("reports/validation/flare24_ccrth_task_factorized_v2.json")
PARENT_PATH = Path("reports/experiments/flare24_ccrth_2025_retrospective_v5.json")


def _report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_factorized_publication_tables_are_complete_and_explicitly_posthoc() -> None:
    report = _report()
    scores = primary_score_table(report)
    intervals = paired_interval_table(report)
    weights = component_weight_table(report)
    grid = component_grid_table(report)
    monthly = monthly_score_table(report)
    regimes = regime_score_table(report)
    q4 = q4_selection_table(report)

    assert len(scores) == 5
    assert len(intervals) == 8
    assert len(weights) == 50
    assert len(grid) == 25
    assert grid["post_hoc_winner"].sum() == 1
    assert not grid["selection_adjusted_interval_available"].any()
    assert len(monthly) == 60
    assert set(regimes["regime"]) == {"low", "elevated", "severe"}
    assert q4["selected"].sum() == 1
    winner_role = scores.loc[scores["method"].eq(grid.loc[grid["post_hoc_winner"], "method"].item()), "evidence_role"].item()
    assert "post-hoc" in winner_role
    assert "2026" in winner_role


def test_factorized_publication_inputs_are_hash_bound() -> None:
    report, validation, parent = _load_inputs(
        REPORT_PATH,
        VALIDATION_PATH,
        PARENT_PATH,
    )

    assert report["outcomes_accessed"]["2026_accessed"] is False
    assert validation["confirmation_outcomes_accessed"] is False
    assert parent["outcomes_accessed"]["2026_accessed"] is False
