from __future__ import annotations

import json
from pathlib import Path

from flightdelaybench.flare_capacity_metastack import META_METHOD
from flightdelaybench.flare_capacity_metastack_reporting import (
    _load_inputs,
    coefficient_table,
    confirmation_table,
    monthly_score_table,
    paired_interval_table,
    primary_score_table,
    regime_score_table,
    regularization_table,
)

REPORT_PATH = Path("reports/experiments/flare24_ccrth_metastack_v1.json")
VALIDATION_PATH = Path("reports/validation/flare24_ccrth_metastack_v2.json")
PARENT_ASSETS_PATH = Path("manifests/flare24_ccrth_publication_assets_v7.json")


def _report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_metastack_publication_tables_are_complete_and_disclosed() -> None:
    report = _report()
    scores = primary_score_table(report)
    intervals = paired_interval_table(report)
    regularization = regularization_table(report)
    coefficients = coefficient_table(report)
    monthly = monthly_score_table(report)
    regimes = regime_score_table(report)
    confirmation = confirmation_table(report)

    assert len(scores) == 4
    assert len(intervals) == 4
    assert len(regularization) == 7
    assert regularization["selected_on_h1"].sum() == 1
    assert regularization.loc[regularization["selected_on_h1"], "regularization_c"].item() == 0.001
    assert len(coefficients) == 6
    assert len(monthly) == 24
    assert len(regimes) == 6
    assert set(regimes["regime"]) == {"low", "elevated", "severe"}
    assert not intervals["independent_confirmation"].any()
    assert not intervals["selection_adjusted"].any()
    assert confirmation["2026_outcomes_accessed"].item() is False
    assert set(scores["method"]) == {"flare24", META_METHOD}


def test_metastack_publication_inputs_are_hash_bound() -> None:
    report, validation, parent = _load_inputs(
        REPORT_PATH,
        VALIDATION_PATH,
        PARENT_ASSETS_PATH,
    )

    assert report["outcomes_accessed"]["2026_accessed"] is False
    assert validation["confirmation_outcomes_accessed"] is False
    assert parent["confirmation_outcomes_accessed"] is False
