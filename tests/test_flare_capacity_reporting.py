from __future__ import annotations

from flightdelaybench.flare_capacity_reporting import (
    METHOD_ORDER,
    airport_score_table,
    feature_coverage_table,
    feature_importance_table,
    feature_registry_table,
    full_year_point_score_table,
    monthly_score_table,
    proper_score_table,
    regime_score_table,
)


def _method_scores() -> dict[str, dict[str, object]]:
    return {
        name: {
            "joint": {
                "n": 100,
                "log_loss": 0.5 - 0.001 * index,
                "multiclass_brier": 0.3 - 0.001 * index,
            }
        }
        for index, name in enumerate(METHOD_ORDER)
    }


def test_capacity_publication_tables_use_flare24_as_reference() -> None:
    monthly_methods = {
        name: {
            "joint_log_loss": values["joint"]["log_loss"],
            "multiclass_brier": values["joint"]["multiclass_brier"],
        }
        for name, values in _method_scores().items()
    }
    report = {
        "retrospective_evaluation": {"methods": _method_scores()},
        "retrospective_full_year_descriptive_joint_scores": {
            name: {
                "n": values["joint"]["n"],
                "joint_log_loss": values["joint"]["log_loss"],
                "multiclass_brier": values["joint"]["multiclass_brier"],
            }
            for name, values in _method_scores().items()
        },
        "retrospective_monthly_scores": [
            {"month": 1, "n": 100, "methods": monthly_methods}
        ],
        "retrospective_capacity_regime_scores": [
            {"regime": "low", "n": 100, "methods": monthly_methods}
        ],
        "feature_importance": [
            {
                "candidate": "hypergraph",
                "task": "delay",
                "capacity_feature_importance_fraction": 0.2,
                "top_capacity_40": [
                    {"feature": "ccrth_origin_expected_queue", "importance": 4.0}
                ],
            }
        ],
        "retrospective_airport_scores": [
            {
                "role": "origin",
                "airport": "AAA",
                "method": "hypergraph",
                "n": 100,
                "joint_log_loss": 0.4,
                "joint_log_loss_delta_vs_flare24": -0.1,
                "multiclass_brier": 0.2,
                "multiclass_brier_delta_vs_flare24": -0.1,
            }
        ],
        "feature_sets": {
            "flare24": [],
            "raw_demand": ["ccrth_origin_expected_queue"],
            "normalized_capacity": ["ccrth_origin_expected_queue"],
            "queue_shadow": ["ccrth_origin_expected_queue"],
            "hypergraph": ["ccrth_origin_expected_queue"],
        },
        "capacity_feature_nonmissing_fraction": {
            "ccrth_origin_expected_queue": 0.98
        },
    }
    scores = proper_score_table(report)
    assert len(scores) == len(METHOD_ORDER)
    assert scores.loc[scores["method"].eq("flare24"), "joint_log_loss_delta_vs_flare24"].item() == 0.0
    assert monthly_score_table(report)["month"].eq(1).all()
    assert len(full_year_point_score_table(report)) == len(METHOD_ORDER)
    assert regime_score_table(report)["regime"].eq("low").all()
    importance = feature_importance_table(report)
    assert importance.loc[0, "feature"] == "ccrth_origin_expected_queue"
    registry = feature_registry_table()
    assert registry["feature"].is_unique
    assert registry["available_from"].eq("FORECAST_24H").all()
    coverage = feature_coverage_table(report)
    assert coverage.loc[0, "nonmissing_fraction"] == 0.98
    assert bool(coverage.loc[0, "used_by_hypergraph"])
    assert airport_score_table(report).loc[0, "airport"] == "AAA"
