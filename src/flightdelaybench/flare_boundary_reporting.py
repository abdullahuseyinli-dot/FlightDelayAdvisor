"""Create publication tables, figures, and a concise report for BC-POT-R."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .flare_boundary_study_validation import METHODS
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

DISPLAY_NAMES = {
    "schedule_baseline": "Schedule baseline",
    "flare24": "FLARE-24",
    "capacity_gated_simplex": "CC-RTH gated",
    "meta_current": "Previous meta-stack",
    "boundary_only": "BC-POT full view",
    "counterfactual_residual": "BC-POT residual view",
    "boundary_ensemble": "BC-POT global ensemble",
    "boundary_gated_ensemble": "BC-POT residual-gated ensemble",
}
COLORS = {
    "schedule_baseline": "#6B7280",
    "flare24": "#4C78A8",
    "capacity_gated_simplex": "#72B7B2",
    "meta_current": "#B279A2",
    "boundary_only": "#F2CF5B",
    "counterfactual_residual": "#F58518",
    "boundary_ensemble": "#E45756",
    "boundary_gated_ensemble": "#7A5195",
}


def _self_hashed(path: Path) -> tuple[dict[str, Any], str]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    keys = [key for key in ("report_sha256", "validation_sha256") if key in payload]
    if len(keys) != 1:
        raise ValueError(f"publication source has invalid self-hash keys: {path}")
    key = keys[0]
    body = {name: value for name, value in payload.items() if name != key}
    if canonical_json_sha256(body) != payload[key]:
        raise ValueError(f"publication source self-hash failed: {path}")
    return payload, key


def primary_metrics_table(report: dict[str, Any]) -> pd.DataFrame:
    proper = report["primary_evaluation"]["methods"]
    decisions = report["primary_argmax_decision_metrics"]
    schedule = "schedule_baseline"
    current = "meta_current"
    return pd.DataFrame.from_records(
        [
            {
                "method": method,
                "display_name": DISPLAY_NAMES[method],
                "n": int(proper[method]["joint"]["n"]),
                "joint_log_loss": float(proper[method]["joint"]["log_loss"]),
                "joint_log_loss_delta_vs_schedule": float(
                    proper[method]["joint"]["log_loss"]
                    - proper[schedule]["joint"]["log_loss"]
                ),
                "joint_log_loss_delta_vs_previous_meta": float(
                    proper[method]["joint"]["log_loss"]
                    - proper[current]["joint"]["log_loss"]
                ),
                "multiclass_brier": float(proper[method]["joint"]["multiclass_brier"]),
                "multiclass_brier_delta_vs_schedule": float(
                    proper[method]["joint"]["multiclass_brier"]
                    - proper[schedule]["joint"]["multiclass_brier"]
                ),
                "multiclass_brier_delta_vs_previous_meta": float(
                    proper[method]["joint"]["multiclass_brier"]
                    - proper[current]["joint"]["multiclass_brier"]
                ),
                "argmax_accuracy": float(decisions[method]["accuracy"]),
                "absolute_accuracy_gain_vs_schedule": float(
                    decisions[method]["accuracy"] - decisions[schedule]["accuracy"]
                ),
                "absolute_accuracy_gain_vs_previous_meta": float(
                    decisions[method]["accuracy"] - decisions[current]["accuracy"]
                ),
                "accuracy_percentage_point_gain_vs_schedule": 100.0
                * float(decisions[method]["accuracy"] - decisions[schedule]["accuracy"]),
                "accuracy_percentage_point_gain_vs_previous_meta": 100.0
                * float(decisions[method]["accuracy"] - decisions[current]["accuracy"]),
                "balanced_accuracy": float(decisions[method]["balanced_accuracy"]),
            }
            for method in METHODS
        ]
    )


def monthly_metrics_table(report: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for period in report["retrospective_monthly_scores"]:
        for method in METHODS:
            metrics = period["methods"][method]
            records.append(
                {
                    "month": int(period["month"]),
                    "method": method,
                    "n": int(metrics["n"]),
                    "joint_log_loss": float(metrics["joint_log_loss"]),
                    "multiclass_brier": float(metrics["multiclass_brier"]),
                    "argmax_accuracy": float(metrics["accuracy"]),
                }
            )
    return pd.DataFrame.from_records(records)


def residual_regime_table(report: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    selected = str(report["selected_method_by_2024_forward_score"])
    for regime in report["retrospective_boundary_residual_regime_scores"]:
        for method in ("schedule_baseline", "meta_current", selected):
            metrics = regime["methods"][method]
            records.append(
                {
                    "regime": regime["regime"],
                    "rows": int(regime["rows"]),
                    "observed_rows": int(regime["observed_rows"]),
                    "method": method,
                    "joint_log_loss": float(metrics["joint_log_loss"]),
                    "multiclass_brier": float(metrics["multiclass_brier"]),
                    "argmax_accuracy": float(metrics["accuracy"]),
                    "on_time_prevalence": regime["class_prevalence"]["on_time"],
                    "delay_prevalence": regime["class_prevalence"]["delayed"],
                    "cancellation_prevalence": regime["class_prevalence"]["cancelled"],
                    "gating_feature_mean": regime["gating_feature_mean"],
                    "gating_feature_p90": regime["gating_feature_p90"],
                }
            )
    return pd.DataFrame.from_records(records)


def feature_importance_table(report: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for model in report["model_artifacts"]:
        for rank, feature in enumerate(model["top_30_feature_importance"], start=1):
            name = str(feature["feature"])
            records.append(
                {
                    "candidate": model["candidate"],
                    "task": model["task"],
                    "rank": rank,
                    "feature": name,
                    "importance": float(feature["importance"]),
                    "feature_view": (
                        "boundary_residual"
                        if name.startswith("bcpot_residual__")
                        else "boundary_full"
                        if name.startswith("bcpot_full__")
                        else "other"
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(partial)
    frame.to_csv(partial, index=False)
    partial.replace(path)
    return {
        "path": path.as_posix(),
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _atomic_figure(figure: Any, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(partial)
    figure.savefig(partial, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    partial.replace(path)
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _atomic_text(value: str, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(partial)
    partial.write_text(value, encoding="utf-8")
    partial.replace(path)
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_boundary_publication_assets(
    *,
    report_path: Path,
    validation_path: Path,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_dir.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite BC-POT-R publication assets")
    report, report_key = _self_hashed(report_path)
    validation, validation_key = _self_hashed(validation_path)
    if (
        report.get("status")
        != "COMPLETE_BCPOTR_2024_SELECTION_2025_RETROSPECTIVE_REDEVELOPMENT"
        or validation.get("status") != "PASS_BCPOTR_STUDY_VALIDATION"
        or Path(str(validation.get("report", {}).get("path", ""))).resolve()
        != report_path.resolve()
        or validation.get("report", {}).get("self_hash") != report[report_key]
    ):
        raise ValueError("publication assets require a bound passing study validation")
    output_dir.mkdir(parents=True)
    tables = {
        "primary_metrics": primary_metrics_table(report),
        "monthly_metrics": monthly_metrics_table(report),
        "boundary_residual_regimes": residual_regime_table(report),
        "boundary_signal_outcome_associations": pd.DataFrame.from_records(
            report["primary_boundary_signal_outcome_associations"]
        ),
        "feature_importance": feature_importance_table(report),
        "airport_scores": pd.DataFrame.from_records(report["retrospective_airport_scores"]),
    }
    table_records = [
        {"name": name, **_atomic_csv(frame, output_dir / f"{name}.csv")}
        for name, frame in tables.items()
    ]

    primary = tables["primary_metrics"]
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), sharey=True)
    positions = np.arange(len(primary))
    colors = [COLORS[name] for name in primary["method"]]
    axes[0].barh(positions, primary["joint_log_loss"], color=colors)
    axes[0].set_xlabel("Joint log loss (lower is better)")
    axes[1].barh(positions, primary["argmax_accuracy"], color=colors)
    axes[1].axvline(0.85, color="black", linestyle="--", linewidth=1.0, label="0.85")
    axes[1].set_xlabel("Standard three-state argmax accuracy")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(positions, primary["display_name"])
    axes[0].invert_yaxis()
    figure.suptitle("BC-POT-R retrospective 2025 primary comparison")
    figures = [
        {
            "name": "primary_comparison",
            **_atomic_figure(figure, output_dir / "primary_comparison.png"),
        }
    ]
    plt.close(figure)

    selected = str(report["selected_method_by_2024_forward_score"])
    monthly = tables["monthly_metrics"]
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True)
    for method in ("schedule_baseline", "meta_current", selected):
        subset = monthly.loc[monthly["method"].eq(method)]
        axes[0].plot(
            subset["month"],
            subset["joint_log_loss"],
            marker="o",
            label=DISPLAY_NAMES[method],
            color=COLORS[method],
        )
        axes[1].plot(
            subset["month"],
            subset["argmax_accuracy"],
            marker="o",
            label=DISPLAY_NAMES[method],
            color=COLORS[method],
        )
    axes[0].set_ylabel("Joint log loss")
    axes[1].set_ylabel("Argmax accuracy")
    axes[1].set_xlabel("2025 month")
    axes[1].set_xticks(range(1, 13))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, ncols=3)
    figure.suptitle("Temporal stability of the selected BC-POT-R design")
    figures.append(
        {
            "name": "monthly_stability",
            **_atomic_figure(figure, output_dir / "monthly_stability.png"),
        }
    )
    plt.close(figure)

    selected_row = primary.loc[primary["method"].eq(selected)].iloc[0]
    gate = report["breakthrough_gate"]
    schedule_interval = report["primary_selected_argmax_comparison_vs_schedule"][
        "accuracy_difference_vs_reference"
    ]
    summary = f"""# BC-POT-R results

This is retrospective redevelopment evidence: earlier aggregate 2025 results were known.
The 2026 confirmation gate remains unopened.

## Primary result

- Q4-2024-selected method: `{selected}`
- Standard argmax accuracy: {float(selected_row['argmax_accuracy']):.9f}
- Original schedule-baseline accuracy: {float(primary.loc[primary['method'].eq('schedule_baseline'), 'argmax_accuracy'].iloc[0]):.9f}
- Previous meta-stack accuracy: {float(primary.loc[primary['method'].eq('meta_current'), 'argmax_accuracy'].iloc[0]):.9f}
- Absolute accuracy gain vs original schedule baseline: {float(selected_row['absolute_accuracy_gain_vs_schedule']):+.9f}
- Date-cluster 95% interval for that gain: [{float(schedule_interval['lower']):+.9f}, {float(schedule_interval['upper']):+.9f}]
- Absolute accuracy gain vs previous meta-stack: {float(selected_row['absolute_accuracy_gain_vs_previous_meta']):+.9f}
- Requested +0.05 gate passed: {bool(gate['minimum_gate_passed'])}
- Requested +0.10 gate passed: {bool(gate['stretch_gate_passed'])}
- Joint log loss: {float(selected_row['joint_log_loss']):.9f}
- Multiclass Brier score: {float(selected_row['multiclass_brier']):.9f}

Accuracy changes above are absolute proportions; multiply by 100 for percentage points.
No relative-percentage substitution is used. Boundary residuals are computational
contrasts, not causal effects, and capacity/queue fields are model states rather than
observed airport operations.
"""
    summary_record = _atomic_text(summary, output_dir / "RESULTS.md")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_BCPOTR_PUBLICATION_ASSETS",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_report": {
            "path": report_path.as_posix(),
            "bytes": report_path.stat().st_size,
            "sha256": sha256_file(report_path),
            "self_hash": report[report_key],
        },
        "source_validation": {
            "path": validation_path.as_posix(),
            "bytes": validation_path.stat().st_size,
            "sha256": sha256_file(validation_path),
            "self_hash": validation[validation_key],
        },
        "selected_method": selected,
        "breakthrough_gate": gate,
        "tables": table_records,
        "figures": figures,
        "summary": summary_record,
        "claim_limit": (
            "Retrospective redevelopment only; earlier 2025 aggregates were known and "
            "2026 confirmation outcomes remain unopened."
        ),
        "provenance": capture_provenance((Path(__file__),)),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = build_boundary_publication_assets(
        report_path=args.report,
        validation_path=args.validation,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
    )
    print(json.dumps({"status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
