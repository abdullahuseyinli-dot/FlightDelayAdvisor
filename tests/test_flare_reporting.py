from __future__ import annotations

from flightdelaybench.flare_reporting import (
    REPORT_METHODS,
    main_score_table,
    monthly_score_table,
    paired_interval_table,
)


def _binary() -> dict[str, float | int]:
    return {
        "n": 100,
        "log_loss": 0.2,
        "brier": 0.1,
        "roc_auc": 0.7,
        "average_precision": 0.3,
    }


def _audit() -> dict[str, object]:
    methods = {
        method: {
            "joint": {"n": 100, "log_loss": 0.4, "multiclass_brier": 0.2},
            "delay_given_operated": _binary(),
            "cancellation": _binary(),
        }
        for method in REPORT_METHODS
    }
    interval = {
        "estimate": -0.01,
        "lower": -0.02,
        "upper": -0.001,
        "confidence": 0.95,
        "clusters": 365,
        "repetitions": 2_000,
        "seed": 1,
    }
    monthly = []
    for month in range(1, 13):
        monthly.append(
            {
                "year": 2025,
                "month": month,
                "proper_scores": {
                    method: {
                        "n": 100,
                        "joint_log_loss": 0.4 - (0.01 if method != "baseline" else 0.0),
                        "multiclass_brier": 0.2,
                    }
                    for method in REPORT_METHODS
                },
            }
        )
    return {
        "primary_evaluation": {
            "methods": methods,
            "paired_date_cluster_comparisons": {
                f"{method}_minus_baseline": {
                    "joint_log_loss": interval,
                    "multiclass_brier": interval,
                }
                for method in REPORT_METHODS
                if method != "baseline"
            },
        },
        "monthly_proper_scores": monthly,
    }


def test_flare_reporting_tables_preserve_method_and_month_coverage() -> None:
    audit = _audit()
    scores = main_score_table(audit)
    assert scores["method"].tolist() == list(REPORT_METHODS)
    intervals = paired_interval_table(audit)
    assert len(intervals) == (len(REPORT_METHODS) - 1) * 2
    assert intervals["upper"].lt(0.0).all()
    monthly = monthly_score_table(audit)
    assert len(monthly) == 12 * len(REPORT_METHODS)
    assert monthly.loc[
        monthly["method"].eq("weather"), "joint_log_loss_minus_baseline"
    ].lt(0.0).all()
