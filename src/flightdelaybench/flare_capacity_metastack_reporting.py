"""Build the self-contained CC-RTH v8 publication bundle with meta-stack evidence."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .flare_capacity_factorized_reporting import (
    _atomic_csv,
    _atomic_figure,
    _load_self_hashed,
)
from .flare_capacity_metastack import META_METHOD, SELECTION_END
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

EXPECTED_REPORT_STATUS = "COMPLETE_TF_CCRTH_METASTACK_POST_HOC_2025_2026_UNOPENED"
EXPECTED_VALIDATION_STATUS = "PASS_TF_CCRTH_METASTACK_STUDY_VALIDATION"
EXPECTED_PARENT_ASSET_STATUS = "COMPLETE_VALIDATED_TF_CCRTH_PUBLICATION_ASSETS"
DISPLAY_NAME = "TF-CC-RTH cancellation logit stack"
COLOR = "#7A5195"
BASELINE_COLOR = "#8C8C8C"


def _verify_record(record: Mapping[str, Any], *, role: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"{role} byte count differs")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{role} checksum differs")
    return path


def _load_inputs(
    report_path: Path,
    validation_path: Path,
    parent_assets_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    report = _load_self_hashed(report_path, "report_sha256")
    validation = _load_self_hashed(validation_path, "validation_sha256")
    parent_assets = _load_self_hashed(parent_assets_path, "manifest_sha256")
    if report.get("status") != EXPECTED_REPORT_STATUS:
        raise ValueError("meta-stack publication assets require a complete report")
    if validation.get("status") != EXPECTED_VALIDATION_STATUS:
        raise ValueError("meta-stack publication assets require passing validation")
    if parent_assets.get("status") != EXPECTED_PARENT_ASSET_STATUS:
        raise ValueError("meta-stack publication assets require validated v7 assets")
    bound = validation.get("report", {})
    if (
        Path(str(bound.get("path", ""))).resolve() != report_path.resolve()
        or bound.get("sha256") != sha256_file(report_path)
        or bound.get("self_hash") != report["report_sha256"]
    ):
        raise ValueError("meta-stack validation is not bound to the supplied report")
    factorized = report.get("factorized_context", {})
    parent_source = parent_assets.get("source_report", {})
    if factorized.get("sha256") != parent_source.get("sha256") or factorized.get(
        "self_hash"
    ) != parent_source.get("self_hash"):
        raise ValueError("v7 asset source differs from the meta-stack factorized context")
    if (
        report.get("outcomes_accessed", {}).get("2026_accessed") is not False
        or validation.get("confirmation_outcomes_accessed") is not False
        or parent_assets.get("confirmation_outcomes_accessed") is not False
    ):
        raise ValueError("publication source crossed the unopened-2026 boundary")
    for kind in ("tables", "figures"):
        for record in parent_assets[kind]:
            _verify_record(record, role=f"v7 {kind[:-1]} {record.get('name')}")
    return report, validation, parent_assets


def primary_score_table(report: Mapping[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for period, key, evidence_role in (
        (
            "2025_full_primary",
            "descriptive_full_primary_evaluation",
            "post-hoc descriptive; architecture informed by 2025",
        ),
        (
            "2025_h2_held_forward",
            "held_forward_evaluation",
            "execution-held-forward; not epistemically blind",
        ),
    ):
        methods = report[key]["methods"]
        baseline = methods["flare24"]
        for method in ("flare24", META_METHOD):
            values = methods[method]
            records.append(
                {
                    "period": period,
                    "method": method,
                    "display_name": "FLARE-24" if method == "flare24" else DISPLAY_NAME,
                    "evidence_role": evidence_role,
                    "joint_n": int(values["joint"]["n"]),
                    "joint_log_loss": float(values["joint"]["log_loss"]),
                    "joint_log_loss_delta_vs_flare24": float(values["joint"]["log_loss"])
                    - float(baseline["joint"]["log_loss"]),
                    "multiclass_brier": float(values["joint"]["multiclass_brier"]),
                    "multiclass_brier_delta_vs_flare24": float(values["joint"]["multiclass_brier"])
                    - float(baseline["joint"]["multiclass_brier"]),
                    "cancellation_n": int(values["cancellation"]["n"]),
                    "cancellation_log_loss": float(values["cancellation"]["log_loss"]),
                    "cancellation_brier": float(values["cancellation"]["brier"]),
                    "cancellation_roc_auc": float(values["cancellation"]["roc_auc"]),
                    "cancellation_average_precision": float(
                        values["cancellation"]["average_precision"]
                    ),
                    "cancellation_ece_equal_mass": float(values["cancellation"]["ece_equal_mass"]),
                    "delay_log_loss": float(values["delay_given_operated"]["log_loss"]),
                    "delay_brier": float(values["delay_given_operated"]["brier"]),
                }
            )
    return pd.DataFrame.from_records(records)


def paired_interval_table(report: Mapping[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    comparison_key = f"{META_METHOD}_minus_flare24"
    for period, key, independent in (
        ("2025_full_primary", "descriptive_full_primary_evaluation", False),
        ("2025_h2_held_forward", "held_forward_evaluation", False),
    ):
        comparison = report[key]["paired_date_cluster_comparisons"][comparison_key]
        for metric in ("joint_log_loss", "multiclass_brier"):
            interval = comparison[metric]
            records.append(
                {
                    "period": period,
                    "method": META_METHOD,
                    "metric": metric,
                    "estimate": float(interval["estimate"]),
                    "lower": float(interval["lower"]),
                    "upper": float(interval["upper"]),
                    "clusters": int(interval["clusters"]),
                    "repetitions": int(interval["repetitions"]),
                    "seed": int(interval["seed"]),
                    "independent_confirmation": independent,
                    "selection_adjusted": False,
                    "negative_favors_candidate": True,
                }
            )
    return pd.DataFrame.from_records(records)


def regularization_table(report: Mapping[str, Any]) -> pd.DataFrame:
    selected = str(report["selected_model"])
    q4 = report["q4_fit_scores"]
    h1 = report["h1_candidate_scores"]
    records: list[dict[str, Any]] = []
    for model in report["model_artifacts"]:
        name = f"c_{float(model['regularization_c']):.4g}".replace(".", "p")
        records.append(
            {
                "model": name,
                "regularization_c": float(model["regularization_c"]),
                "selected_on_h1": name == selected,
                "q4_n": int(q4[name]["n"]),
                "q4_joint_log_loss": float(q4[name]["joint_log_loss"]),
                "q4_joint_log_loss_delta_vs_flare24": float(q4[name]["joint_log_loss"])
                - float(q4["flare24"]["joint_log_loss"]),
                "q4_multiclass_brier": float(q4[name]["multiclass_brier"]),
                "q4_multiclass_brier_delta_vs_flare24": float(q4[name]["multiclass_brier"])
                - float(q4["flare24"]["multiclass_brier"]),
                "h1_n": int(h1[name]["n"]),
                "h1_joint_log_loss": float(h1[name]["joint_log_loss"]),
                "h1_joint_log_loss_delta_vs_flare24": float(h1[name]["joint_log_loss"])
                - float(h1["flare24"]["joint_log_loss"]),
                "h1_multiclass_brier": float(h1[name]["multiclass_brier"]),
                "h1_multiclass_brier_delta_vs_flare24": float(h1[name]["multiclass_brier"])
                - float(h1["flare24"]["multiclass_brier"]),
            }
        )
    return pd.DataFrame.from_records(records)


def coefficient_table(report: Mapping[str, Any]) -> pd.DataFrame:
    selected_c = float(report["selected_regularization_c"])
    selected = next(
        record
        for record in report["model_artifacts"]
        if float(record["regularization_c"]) == selected_c
    )
    rows = [
        {
            "term": "intercept",
            "term_type": "intercept",
            "coefficient": float(selected["intercept"]),
            "regularization_c": selected_c,
            "interpretation_limit": "descriptive predictive coefficient; not causal",
        }
    ]
    rows.extend(
        {
            "term": name,
            "term_type": "candidate_cancellation_logit",
            "coefficient": float(selected["coefficients"][name]),
            "regularization_c": selected_c,
            "interpretation_limit": "conditional on correlated candidate logits; not causal",
        }
        for name in selected["feature_order"]
    )
    return pd.DataFrame.from_records(rows)


def monthly_score_table(report: Mapping[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for month in report["descriptive_monthly_scores"]:
        baseline = month["methods"]["flare24"]
        for method in ("flare24", META_METHOD):
            metrics = month["methods"][method]
            records.append(
                {
                    "month": int(month["month"]),
                    "n": int(month["n"]),
                    "method": method,
                    "joint_log_loss": float(metrics["joint_log_loss"]),
                    "joint_log_loss_delta_vs_flare24": float(metrics["joint_log_loss"])
                    - float(baseline["joint_log_loss"]),
                    "multiclass_brier": float(metrics["multiclass_brier"]),
                    "multiclass_brier_delta_vs_flare24": float(metrics["multiclass_brier"])
                    - float(baseline["multiclass_brier"]),
                }
            )
    return pd.DataFrame.from_records(records)


def regime_score_table(report: Mapping[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for regime in report["descriptive_capacity_regime_scores"]:
        baseline = regime["methods"]["flare24"]
        for method in ("flare24", META_METHOD):
            metrics = regime["methods"][method]
            records.append(
                {
                    "regime": str(regime["regime"]),
                    "n": int(regime["n"]),
                    "method": method,
                    "joint_log_loss": float(metrics["joint_log_loss"]),
                    "joint_log_loss_delta_vs_flare24": float(metrics["joint_log_loss"])
                    - float(baseline["joint_log_loss"]),
                    "multiclass_brier": float(metrics["multiclass_brier"]),
                    "multiclass_brier_delta_vs_flare24": float(metrics["multiclass_brier"])
                    - float(baseline["multiclass_brier"]),
                }
            )
    return pd.DataFrame.from_records(records)


def confirmation_table(report: Mapping[str, Any]) -> pd.DataFrame:
    lock = report["confirmation_lock"]
    return pd.DataFrame.from_records(
        [
            {
                "method": META_METHOD,
                "status": "LOCKED_FOR_UNOPENED_2026_CONFIRMATION",
                "coefficient_fit_period": "2024-10-03/2024-12-31",
                "regularization_selection_period": f"2025-01-03/{SELECTION_END}",
                "execution_held_forward_period": "2025-07-03/2025-12-29",
                "confirmation_period": "2026-01-03/2026-12-29",
                "2025_architecture_informed": True,
                "2025_independent_confirmation": False,
                "2026_outcomes_accessed": bool(lock["2026_outcomes_accessed"]),
                "lock_sha256": str(lock["sha256"]),
                "lock_self_hash": str(lock["self_hash"]),
            }
        ]
    )


def _copy_parent_asset(
    record: Mapping[str, Any],
    *,
    output_dir: Path,
    role: str,
) -> dict[str, Any]:
    source = _verify_record(record, role=role)
    target = output_dir / source.name
    if target.exists():
        raise FileExistsError(target)
    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        raise FileExistsError(partial)
    shutil.copyfile(source, partial)
    partial.replace(target)
    if sha256_file(target) != record["sha256"]:
        raise RuntimeError(f"copied {role} checksum differs")
    return {
        **dict(record),
        "path": target.as_posix(),
        "inherited_from": source.as_posix(),
    }


def build_metastack_publication_assets(
    report_path: Path,
    *,
    validation_path: Path,
    parent_assets_path: Path,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if output_dir.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite CC-RTH v8 publication assets")
    report, validation, parent_assets = _load_inputs(
        report_path, validation_path, parent_assets_path
    )
    output_dir.mkdir(parents=True)
    table_records = [
        _copy_parent_asset(
            record,
            output_dir=output_dir,
            role=f"v7 table {record['name']}",
        )
        for record in parent_assets["tables"]
    ]
    figure_records = [
        _copy_parent_asset(
            record,
            output_dir=output_dir,
            role=f"v7 figure {record['name']}",
        )
        for record in parent_assets["figures"]
    ]
    tables = {
        "metastack_primary_scores": primary_score_table(report),
        "metastack_paired_intervals": paired_interval_table(report),
        "metastack_regularization_selection": regularization_table(report),
        "metastack_coefficients": coefficient_table(report),
        "metastack_monthly_scores": monthly_score_table(report),
        "metastack_capacity_regime_scores": regime_score_table(report),
        "metastack_airport_scores": pd.DataFrame.from_records(report["descriptive_airport_scores"]),
        "metastack_confirmation_protocol": confirmation_table(report),
    }
    table_records.extend(
        _atomic_csv(frame, output_dir / f"{name}.csv") for name, frame in tables.items()
    )

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    plt.style.use("seaborn-v0_8-whitegrid")
    intervals = tables["metastack_paired_intervals"]
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    for row, (period, period_label) in enumerate(
        (
            ("2025_h2_held_forward", "H2 execution-held-forward"),
            ("2025_full_primary", "Full 2025 descriptive"),
        )
    ):
        for column, (metric, metric_label) in enumerate(
            (
                ("joint_log_loss", "Joint log loss"),
                ("multiclass_brier", "Multiclass Brier"),
            )
        ):
            axis = axes[row, column]
            point = intervals.loc[
                intervals["period"].eq(period) & intervals["metric"].eq(metric)
            ].iloc[0]
            estimate = float(point["estimate"])
            axis.errorbar(
                estimate,
                0,
                xerr=np.asarray(
                    [
                        [estimate - float(point["lower"])],
                        [float(point["upper"]) - estimate],
                    ]
                ),
                fmt="o",
                color=COLOR,
                capsize=5,
                markersize=8,
            )
            axis.axvline(0.0, color="black", linewidth=1.0)
            axis.set_yticks([0], [period_label] if column == 0 else [])
            axis.set_title(metric_label)
            axis.set_xlabel("Stack minus FLARE-24 (negative is better)")
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            axis.xaxis.set_major_formatter(formatter)
    figure.suptitle("TF-CC-RTH logit-stack proper-score effects (95% date-cluster intervals)")
    figure.text(
        0.5,
        0.01,
        "Both periods are retrospective: H2 was held forward in execution, but the architecture was informed by prior 2025 exploration.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.2, bottom=0.16, top=0.89, wspace=0.22, hspace=0.5)
    figure_records.append(
        _atomic_figure(figure, output_dir / "metastack_proper_score_intervals.png")
    )
    plt.close(figure)

    regularization = tables["metastack_regularization_selection"]
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    for axis, metric, title in zip(
        axes,
        (
            "h1_joint_log_loss_delta_vs_flare24",
            "h1_multiclass_brier_delta_vs_flare24",
        ),
        ("H1 joint log-loss delta", "H1 multiclass Brier delta"),
        strict=True,
    ):
        axis.plot(
            regularization["regularization_c"],
            regularization[metric] * 1_000.0,
            marker="o",
            color=COLOR,
            linewidth=2.0,
        )
        selected_row = regularization.loc[regularization["selected_on_h1"]].iloc[0]
        axis.scatter(
            [selected_row["regularization_c"]],
            [float(selected_row[metric]) * 1_000.0],
            s=100,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            zorder=4,
            label="Selected C=0.001",
        )
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_xscale("log")
        axis.set_xlabel("Inverse L2 strength, C (log scale)")
        axis.set_ylabel("Candidate minus FLARE-24 (x1000)")
        axis.set_title(title)
        axis.legend(frameon=False)
    figure.suptitle("January-June 2025 regularization selection")
    figure.tight_layout()
    figure_records.append(_atomic_figure(figure, output_dir / "metastack_regularization_curve.png"))
    plt.close(figure)

    coefficients = tables["metastack_coefficients"]
    feature_coefficients = coefficients.loc[
        coefficients["term_type"].eq("candidate_cancellation_logit")
    ].copy()
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    colors = [COLOR if value >= 0 else "#D45087" for value in feature_coefficients["coefficient"]]
    axis.barh(
        feature_coefficients["term"],
        feature_coefficients["coefficient"],
        color=colors,
    )
    axis.axvline(0.0, color="black", linewidth=1.0)
    axis.set_xlabel("L2-logistic coefficient")
    axis.set_title("Selected cancellation-logit stack coefficients (C=0.001)")
    axis.text(
        0.99,
        0.95,
        f"intercept = {float(coefficients.iloc[0]['coefficient']):.3f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#C7C7C7", "alpha": 0.9},
    )
    figure.tight_layout()
    figure_records.append(_atomic_figure(figure, output_dir / "metastack_coefficients.png"))
    plt.close(figure)

    monthly = tables["metastack_monthly_scores"]
    stack_monthly = monthly.loc[monthly["method"].eq(META_METHOD)]
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.3))
    for axis, metric, title in zip(
        axes,
        (
            "joint_log_loss_delta_vs_flare24",
            "multiclass_brier_delta_vs_flare24",
        ),
        ("Joint log-loss delta", "Multiclass Brier delta"),
        strict=True,
    ):
        axis.plot(
            stack_monthly["month"],
            stack_monthly[metric] * 1_000.0,
            marker="o",
            color=COLOR,
            linewidth=2.0,
        )
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.axvline(6.5, color=BASELINE_COLOR, linestyle="--", linewidth=1.0)
        axis.set_xticks(range(1, 13))
        axis.set_xlabel("2025 month")
        axis.set_ylabel("Stack minus FLARE-24 (x1000)")
        axis.set_title(title)
    figure.suptitle("Descriptive monthly stability of the cancellation logit stack")
    figure.text(
        0.5,
        0.01,
        "Dashed line separates H1 from H2 months; July 1-2 were excluded from formal held-forward scoring.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(bottom=0.17, top=0.87, wspace=0.28)
    figure_records.append(_atomic_figure(figure, output_dir / "metastack_monthly_deltas.png"))
    plt.close(figure)

    regimes = tables["metastack_capacity_regime_scores"]
    stack_regimes = regimes.loc[regimes["method"].eq(META_METHOD)]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.1))
    for axis, metric, title in zip(
        axes,
        (
            "joint_log_loss_delta_vs_flare24",
            "multiclass_brier_delta_vs_flare24",
        ),
        ("Joint log-loss delta", "Multiclass Brier delta"),
        strict=True,
    ):
        axis.bar(
            stack_regimes["regime"],
            stack_regimes[metric] * 1_000.0,
            color=COLOR,
        )
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_ylabel("Stack minus FLARE-24 (x1000)")
        axis.set_title(title)
    figure.suptitle("Improvement across frozen airport capacity-stress regimes")
    figure.tight_layout()
    figure_records.append(_atomic_figure(figure, output_dir / "metastack_regime_deltas.png"))
    plt.close(figure)

    primary = tables["metastack_primary_scores"]
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    metrics = (
        ("cancellation_log_loss", "Cancellation log loss", "lower"),
        ("cancellation_brier", "Cancellation Brier", "lower"),
        ("cancellation_roc_auc", "Cancellation AUROC", "higher"),
        ("cancellation_average_precision", "Cancellation average precision", "higher"),
    )
    period_order = ("2025_h2_held_forward", "2025_full_primary")
    period_labels = ("H2 held-forward", "Full retrospective")
    for axis, (metric, title, direction) in zip(axes.flat, metrics, strict=True):
        values: list[float] = []
        for period in period_order:
            period_rows = primary.loc[primary["period"].eq(period)].set_index("method")
            values.append(
                float(period_rows.loc[META_METHOD, metric])
                - float(period_rows.loc["flare24", metric])
            )
        axis.barh(period_labels, values, color=COLOR)
        axis.axvline(0.0, color="black", linewidth=1.0)
        axis.invert_yaxis()
        axis.set_title(f"{title} ({direction} is better)")
        axis.set_xlabel("Absolute difference vs FLARE-24")
        if metric == "cancellation_brier":
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            axis.xaxis.set_major_formatter(formatter)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.tick_params(axis="x", labelsize=9)
    figure.suptitle("Cancellation-task gains driving the joint improvement")
    figure.tight_layout()
    figure_records.append(_atomic_figure(figure, output_dir / "metastack_cancellation_effects.png"))
    plt.close(figure)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_VALIDATED_TF_CCRTH_METASTACK_PUBLICATION_ASSETS",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_report": {
            "path": report_path.as_posix(),
            "sha256": sha256_file(report_path),
            "self_hash": report["report_sha256"],
        },
        "source_validation": {
            "path": validation_path.as_posix(),
            "sha256": sha256_file(validation_path),
            "self_hash": validation["validation_sha256"],
        },
        "parent_asset_manifest": {
            "path": parent_assets_path.as_posix(),
            "sha256": sha256_file(parent_assets_path),
            "self_hash": parent_assets["manifest_sha256"],
        },
        "tables": table_records,
        "figures": figure_records,
        "inherited_table_count": len(parent_assets["tables"]),
        "inherited_figure_count": len(parent_assets["figures"]),
        "visual_qa_required": True,
        "evidence_note": (
            "The Q4-locked TF-CC-RTH result remains the prospective-style retrospective "
            "result. The stronger cancellation logit stack is post-hoc 2025 development. "
            "Its H2 evaluation was held forward in execution but is not epistemically blind "
            "or selection-adjusted. Independent confirmation remains locked to unopened 2026."
        ),
        "confirmation_outcomes_accessed": False,
        "provenance": capture_provenance((Path(__file__),)),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--parent-assets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = build_metastack_publication_assets(
        args.report,
        validation_path=args.validation,
        parent_assets_path=args.parent_assets,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
    )
    print(json.dumps({"status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
