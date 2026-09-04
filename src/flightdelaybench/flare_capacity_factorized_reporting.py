"""Build publication tables and figures for validated TF-CC-RTH-v1 evidence."""

from __future__ import annotations

import argparse
import json
import textwrap
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .flare_capacity_factorized import FACTORIZED_METHODS
from .flare_capacity_reporting import feature_importance_table
from .flare_capacity_study import BLEND_CANDIDATES, GATING_LABELS
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

EXPECTED_REPORT_STATUS = "COMPLETE_TF_CCRTH_POST_HOC_2025_ANALYSIS_2026_UNOPENED"
EXPECTED_VALIDATION_STATUS = "PASS_TF_CCRTH_STUDY_VALIDATION"
METHOD_COLORS = {
    "capacity_gated_simplex": "#B8B8B8",
    "task_factorized_global_q4": "#4C78A8",
    "task_factorized_capacity_gated_q4": "#54A24B",
    "post_hoc_winner": "#E45756",
}


def _load_self_hashed(path: Path, hash_key: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = str(payload.get(hash_key, ""))
    if canonical_json_sha256(
        {key: value for key, value in payload.items() if key != hash_key}
    ) != recorded:
        raise ValueError(f"self-hash failed: {path}")
    return payload


def _load_inputs(
    report_path: Path,
    validation_path: Path,
    parent_report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    report = _load_self_hashed(report_path, "report_sha256")
    validation = _load_self_hashed(validation_path, "validation_sha256")
    parent = _load_self_hashed(parent_report_path, "report_sha256")
    if report.get("status") != EXPECTED_REPORT_STATUS:
        raise ValueError("TF-CC-RTH publication assets require a complete report")
    if validation.get("status") != EXPECTED_VALIDATION_STATUS:
        raise ValueError("TF-CC-RTH publication assets require passing validation")
    bound = validation.get("report", {})
    if (
        Path(str(bound.get("path", ""))).resolve() != report_path.resolve()
        or bound.get("sha256") != sha256_file(report_path)
        or bound.get("self_hash") != report["report_sha256"]
    ):
        raise ValueError("TF-CC-RTH validation is not bound to the supplied report")
    parent_bound = report.get("parent_report", {})
    if (
        Path(str(parent_bound.get("path", ""))).resolve() != parent_report_path.resolve()
        or parent_bound.get("sha256") != sha256_file(parent_report_path)
        or parent_bound.get("self_hash") != parent["report_sha256"]
    ):
        raise ValueError("TF-CC-RTH report is not bound to the supplied parent report")
    if (
        report.get("outcomes_accessed", {}).get("2026_accessed") is not False
        or validation.get("confirmation_outcomes_accessed") is not False
    ):
        raise ValueError("TF-CC-RTH publication source crossed the 2026 boundary")
    return report, validation, parent


def _winner_name(report: dict[str, Any]) -> str:
    return str(report["post_hoc_component_pair_winner"]["method"])


def _display_name(name: str, winner: str) -> str:
    names = {
        "flare24": "FLARE-24",
        "capacity_gated_simplex": "Parent gated simplex",
        "task_factorized_global_q4": "Task-factorized global (Q4 lock)",
        "task_factorized_capacity_gated_q4": "Task-factorized stress-gated (Q4 lock)",
        winner: "2025 component-pair winner (post-hoc)",
    }
    return names[name]


def primary_score_table(report: dict[str, Any]) -> pd.DataFrame:
    winner = _winner_name(report)
    methods = report["retrospective_evaluation"]["methods"]
    baseline = methods["flare24"]
    order = ("flare24", "capacity_gated_simplex", *FACTORIZED_METHODS, winner)
    records: list[dict[str, Any]] = []
    for name in order:
        values = methods[name]
        joint = values["joint"]
        cancellation = values["cancellation"]
        delay = values["delay_given_operated"]
        records.append(
            {
                "method": name,
                "display_name": _display_name(name, winner),
                "evidence_role": (
                    "post-hoc 2025-selected; requires 2026 confirmation"
                    if name == winner
                    else "Q4-locked retrospective evaluation"
                ),
                "n": int(joint["n"]),
                "joint_log_loss": float(joint["log_loss"]),
                "joint_log_loss_delta_vs_flare24": float(joint["log_loss"])
                - float(baseline["joint"]["log_loss"]),
                "multiclass_brier": float(joint["multiclass_brier"]),
                "multiclass_brier_delta_vs_flare24": float(
                    joint["multiclass_brier"]
                )
                - float(baseline["joint"]["multiclass_brier"]),
                "cancellation_log_loss": float(cancellation["log_loss"]),
                "cancellation_log_loss_delta_vs_flare24": float(
                    cancellation["log_loss"]
                )
                - float(baseline["cancellation"]["log_loss"]),
                "cancellation_brier": float(cancellation["brier"]),
                "cancellation_roc_auc": float(cancellation["roc_auc"]),
                "cancellation_average_precision": float(
                    cancellation["average_precision"]
                ),
                "delay_log_loss": float(delay["log_loss"]),
                "delay_brier": float(delay["brier"]),
            }
        )
    return pd.DataFrame.from_records(records)


def paired_interval_table(report: dict[str, Any]) -> pd.DataFrame:
    winner = _winner_name(report)
    records: list[dict[str, Any]] = []
    comparisons = report["retrospective_evaluation"][
        "paired_date_cluster_comparisons"
    ]
    for name in ("capacity_gated_simplex", *FACTORIZED_METHODS, winner):
        comparison = comparisons[f"{name}_minus_flare24"]
        for metric in ("joint_log_loss", "multiclass_brier"):
            interval = comparison[metric]
            records.append(
                {
                    "method": name,
                    "display_name": _display_name(name, winner),
                    "metric": metric,
                    "estimate": float(interval["estimate"]),
                    "lower": float(interval["lower"]),
                    "upper": float(interval["upper"]),
                    "clusters": int(interval["clusters"]),
                    "repetitions": int(interval["repetitions"]),
                    "selection_adjusted": False if name == winner else None,
                    "negative_favors_candidate": True,
                }
            )
    return pd.DataFrame.from_records(records)


def component_weight_table(report: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    global_selection = report["global_component_simplexes"]
    gated = report["capacity_gated_component_simplexes"]
    for regime, components in (
        ("global", global_selection["components"]),
        *(
            (name, gated["regimes"][name]["components"])
            for name in GATING_LABELS
        ),
    ):
        for task in ("cancellation", "delay_given_operated"):
            selection = components[task]
            for candidate in BLEND_CANDIDATES:
                records.append(
                    {
                        "regime": regime,
                        "task": task,
                        "candidate": candidate,
                        "weight": float(selection["weights"][candidate]),
                        "rows": int(selection["rows"]),
                        "fallback_to_global": bool(
                            selection.get("fallback_to_global", regime == "missing")
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def q4_selection_table(report: dict[str, Any]) -> pd.DataFrame:
    selected = report["selected_method_by_q4_forward_joint_log_loss"]
    return pd.DataFrame.from_records(
        [
            {
                "method": name,
                "selected": name == selected,
                "n": int(values["n"]),
                "joint_log_loss": float(values["joint_log_loss"]),
                "multiclass_brier": float(values["multiclass_brier"]),
            }
            for name, values in report["selection_scores"].items()
        ]
    )


def component_grid_table(report: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(report["post_hoc_component_pair_grid"])
    baseline = frame.loc[
        frame["method"].eq("cancel_flare24__delay_flare24"),
        ["joint_log_loss", "multiclass_brier"],
    ].iloc[0]
    frame["joint_log_loss_delta_vs_flare24"] = (
        frame["joint_log_loss"] - baseline["joint_log_loss"]
    )
    frame["multiclass_brier_delta_vs_flare24"] = (
        frame["multiclass_brier"] - baseline["multiclass_brier"]
    )
    frame["post_hoc_winner"] = frame["method"].eq(_winner_name(report))
    frame["selection_adjusted_interval_available"] = False
    return frame


def monthly_score_table(report: dict[str, Any]) -> pd.DataFrame:
    winner = _winner_name(report)
    order = ("flare24", "capacity_gated_simplex", *FACTORIZED_METHODS, winner)
    records: list[dict[str, Any]] = []
    for month in report["retrospective_monthly_scores"]:
        baseline = month["methods"]["flare24"]
        for name in order:
            metrics = month["methods"][name]
            records.append(
                {
                    "month": int(month["month"]),
                    "n": int(month["n"]),
                    "method": name,
                    "joint_log_loss": float(metrics["joint_log_loss"]),
                    "joint_log_loss_delta_vs_flare24": float(
                        metrics["joint_log_loss"]
                    )
                    - float(baseline["joint_log_loss"]),
                    "multiclass_brier": float(metrics["multiclass_brier"]),
                    "multiclass_brier_delta_vs_flare24": float(
                        metrics["multiclass_brier"]
                    )
                    - float(baseline["multiclass_brier"]),
                }
            )
    return pd.DataFrame.from_records(records)


def regime_score_table(report: dict[str, Any]) -> pd.DataFrame:
    winner = _winner_name(report)
    order = ("flare24", "capacity_gated_simplex", *FACTORIZED_METHODS, winner)
    records: list[dict[str, Any]] = []
    for regime in report["retrospective_capacity_regime_scores"]:
        baseline = regime["methods"]["flare24"]
        for name in order:
            metrics = regime["methods"][name]
            records.append(
                {
                    "regime": regime["regime"],
                    "n": int(regime["n"]),
                    "method": name,
                    "joint_log_loss": float(metrics["joint_log_loss"]),
                    "joint_log_loss_delta_vs_flare24": float(
                        metrics["joint_log_loss"]
                    )
                    - float(baseline["joint_log_loss"]),
                    "multiclass_brier": float(metrics["multiclass_brier"]),
                    "multiclass_brier_delta_vs_flare24": float(
                        metrics["multiclass_brier"]
                    )
                    - float(baseline["multiclass_brier"]),
                }
            )
    return pd.DataFrame.from_records(records)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(partial)
    frame.to_csv(partial, index=False)
    partial.replace(path)
    return {
        "name": path.stem,
        "path": path.as_posix(),
        "rows": len(frame),
        "columns": list(frame.columns),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _atomic_figure(figure: Any, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_suffix(".part.png")
    if partial.exists():
        raise FileExistsError(partial)
    figure.savefig(partial, dpi=190, bbox_inches="tight", facecolor="white")
    partial.replace(path)
    return {
        "name": path.stem,
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _clean_feature_label(value: str) -> str:
    text = value.removeprefix("ccrth_").replace("_", " ")
    return textwrap.fill(text, width=27)


def build_task_factorized_publication_assets(
    report_path: Path,
    *,
    validation_path: Path,
    parent_report_path: Path,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if output_dir.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite TF-CC-RTH publication assets")
    report, validation, parent = _load_inputs(
        report_path, validation_path, parent_report_path
    )
    output_dir.mkdir(parents=True)
    winner = _winner_name(report)
    tables = {
        "primary_scores": primary_score_table(report),
        "paired_intervals": paired_interval_table(report),
        "q4_selection": q4_selection_table(report),
        "component_weights": component_weight_table(report),
        "posthoc_component_grid": component_grid_table(report),
        "monthly_scores": monthly_score_table(report),
        "capacity_regime_scores": regime_score_table(report),
        "airport_scores": pd.DataFrame.from_records(
            report["retrospective_airport_scores"]
        ),
        "parent_feature_importance": feature_importance_table(parent),
    }
    table_records = [
        _atomic_csv(frame, output_dir / f"{name}.csv")
        for name, frame in tables.items()
    ]

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    plt.style.use("seaborn-v0_8-whitegrid")
    figure_records: list[dict[str, Any]] = []
    intervals = tables["paired_intervals"]
    plot_methods = (
        "capacity_gated_simplex",
        "task_factorized_global_q4",
        "task_factorized_capacity_gated_q4",
        winner,
    )
    labels = [_display_name(name, winner) for name in plot_methods]
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for axis, metric, title in zip(
        axes,
        ("joint_log_loss", "multiclass_brier"),
        ("Joint log-loss difference", "Multiclass Brier difference"),
        strict=True,
    ):
        subset = intervals.loc[intervals["metric"].eq(metric)].set_index("method")
        for y, name in enumerate(plot_methods):
            row = subset.loc[name]
            color_key = "post_hoc_winner" if name == winner else name
            axis.errorbar(
                float(row["estimate"]),
                y,
                xerr=np.asarray(
                    [
                        [float(row["estimate"]) - float(row["lower"])],
                        [float(row["upper"]) - float(row["estimate"])],
                    ]
                ),
                fmt="o",
                color=METHOD_COLORS[color_key],
                capsize=4,
                markersize=7,
            )
        axis.axvline(0.0, color="black", linewidth=1.0)
        axis.set_yticks(range(len(labels)), labels if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel("Candidate minus FLARE-24 (negative is better)")
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        axis.xaxis.set_major_formatter(formatter)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.tick_params(axis="x", labelsize=9)
    figure.suptitle("TF-CC-RTH 2025 paired date-cluster effects (95% intervals)")
    figure.text(
        0.5,
        0.01,
        "* Red result was selected post-hoc from 25 pairs; its interval is not selection-adjusted.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(bottom=0.18, wspace=0.18)
    figure_records.append(
        _atomic_figure(figure, output_dir / "validated_proper_score_effects.png")
    )
    plt.close(figure)

    scores = tables["primary_scores"].set_index("method")
    diagnostic_methods = (
        "task_factorized_global_q4",
        "task_factorized_capacity_gated_q4",
        winner,
    )
    metrics = (
        ("cancellation_log_loss", "Cancellation log loss", "lower"),
        ("cancellation_brier", "Cancellation Brier", "lower"),
        ("cancellation_roc_auc", "Cancellation AUROC", "higher"),
        ("cancellation_average_precision", "Cancellation average precision", "higher"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    for axis, (field, title, direction) in zip(axes.flat, metrics, strict=True):
        baseline = float(scores.loc["flare24", field])
        deltas = [float(scores.loc[name, field]) - baseline for name in diagnostic_methods]
        colors = [
            METHOD_COLORS["post_hoc_winner" if name == winner else name]
            for name in diagnostic_methods
        ]
        axis.barh(range(len(diagnostic_methods)), deltas, color=colors)
        axis.axvline(0.0, color="black", linewidth=1.0)
        axis.set_yticks(
            range(len(diagnostic_methods)),
            ["Global Q4", "Stress-gated Q4", "Post-hoc pair"],
        )
        axis.invert_yaxis()
        axis.set_title(f"{title} ({direction} is better)")
        axis.set_xlabel("Absolute difference vs FLARE-24")
    figure.suptitle("Where the TF-CC-RTH gain comes from: cancellation prediction")
    figure.tight_layout()
    figure_records.append(
        _atomic_figure(figure, output_dir / "cancellation_component_effects.png")
    )
    plt.close(figure)

    weights = tables["component_weights"]
    regime_order = ("global", "low", "elevated", "severe")
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    for axis, task in zip(
        axes, ("cancellation", "delay_given_operated"), strict=True
    ):
        subset = weights.loc[
            weights["task"].eq(task) & weights["regime"].isin(regime_order)
        ]
        matrix = (
            subset.pivot(index="regime", columns="candidate", values="weight")
            .reindex(index=regime_order, columns=BLEND_CANDIDATES)
            .to_numpy(dtype=float)
        )
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "black",
                    fontsize=8,
                )
        axis.set_xticks(range(len(BLEND_CANDIDATES)), BLEND_CANDIDATES, rotation=32, ha="right")
        axis.set_yticks(range(len(regime_order)), regime_order)
        axis.set_title(task.replace("_", " ").title())
    figure.suptitle("Task factorization: resource-state signal is used for cancellation, not delay")
    figure.subplots_adjust(bottom=0.27, top=0.80, left=0.08, right=0.88, wspace=0.28)
    colorbar_axis = figure.add_axes((0.91, 0.25, 0.018, 0.48))
    figure.colorbar(
        image,
        cax=colorbar_axis,
        label="Q4-selected component weight",
    )
    figure_records.append(
        _atomic_figure(figure, output_dir / "task_factorized_weights.png")
    )
    plt.close(figure)

    monthly = tables["monthly_scores"]
    figure, axis = plt.subplots(figsize=(11.5, 6.0))
    for name, label in (
        ("capacity_gated_simplex", "Parent gated simplex"),
        ("task_factorized_global_q4", "Task-factorized global (Q4)"),
        ("task_factorized_capacity_gated_q4", "Task-factorized stress-gated (Q4)"),
        (winner, "Post-hoc component pair"),
    ):
        subset = monthly.loc[monthly["method"].eq(name)]
        color_key = "post_hoc_winner" if name == winner else name
        axis.plot(
            subset["month"],
            subset["joint_log_loss_delta_vs_flare24"],
            marker="o",
            linewidth=2.0,
            color=METHOD_COLORS[color_key],
            label=label,
            linestyle="--" if name == winner else "-",
        )
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xticks(range(1, 13))
    axis.set_xlabel("2025 month")
    axis.set_ylabel("Joint log-loss difference vs FLARE-24")
    axis.set_title("Temporal stability of task-factorized improvements")
    axis.legend(frameon=False, ncols=2)
    figure.tight_layout()
    figure_records.append(
        _atomic_figure(figure, output_dir / "task_factorized_monthly_deltas.png")
    )
    plt.close(figure)

    grid = tables["posthoc_component_grid"]
    log_matrix = (
        grid.pivot(
            index="cancellation_source",
            columns="delay_given_operated_source",
            values="joint_log_loss_delta_vs_flare24",
        )
        .reindex(index=BLEND_CANDIDATES, columns=BLEND_CANDIDATES)
        .to_numpy(dtype=float)
        * 1_000.0
    )
    limit = float(np.max(np.abs(log_matrix)))
    figure, axis = plt.subplots(figsize=(9.5, 7.5))
    image = axis.imshow(log_matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
    for row in range(log_matrix.shape[0]):
        for column in range(log_matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{log_matrix[row, column]:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold" if (row, column) == (2, 0) else "normal",
            )
    axis.set_xticks(range(len(BLEND_CANDIDATES)), BLEND_CANDIDATES, rotation=30, ha="right")
    axis.set_yticks(range(len(BLEND_CANDIDATES)), BLEND_CANDIDATES)
    axis.set_xlabel("Delay-given-operation source")
    axis.set_ylabel("Cancellation source")
    axis.set_title("Post-hoc 2025 component swap grid (log-loss delta x1000)")
    figure.colorbar(image, ax=axis, label="Candidate minus FLARE-24; negative is better")
    figure.text(
        0.5,
        0.01,
        "All 25 pairs are shown; the bold cell is hypothesis-generating, not independent confirmation.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(bottom=0.2)
    figure_records.append(
        _atomic_figure(figure, output_dir / "posthoc_component_pair_grid.png")
    )
    plt.close(figure)

    regimes = tables["capacity_regime_scores"]
    plot_regimes = [
        name for name in GATING_LABELS if name in set(regimes["regime"])
    ]
    figure, axis = plt.subplots(figsize=(10.5, 6.0))
    bar_methods = (
        "task_factorized_global_q4",
        "task_factorized_capacity_gated_q4",
        winner,
    )
    width = 0.24
    positions = np.arange(len(plot_regimes))
    for offset, name in zip((-width, 0.0, width), bar_methods, strict=True):
        subset = regimes.loc[regimes["method"].eq(name)].set_index("regime")
        values = [
            float(subset.loc[regime, "joint_log_loss_delta_vs_flare24"])
            for regime in plot_regimes
        ]
        color_key = "post_hoc_winner" if name == winner else name
        axis.bar(
            positions + offset,
            values,
            width=width,
            color=METHOD_COLORS[color_key],
            label=_display_name(name, winner),
        )
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xticks(positions, plot_regimes)
    axis.set_ylabel("Joint log-loss difference vs FLARE-24")
    axis.set_title("Task-factorized effects across frozen capacity-stress regimes")
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure_records.append(
        _atomic_figure(figure, output_dir / "task_factorized_regime_deltas.png")
    )
    plt.close(figure)

    importance = tables["parent_feature_importance"]
    figure, axes = plt.subplots(1, 2, figsize=(16.5, 8.2))
    for axis, task in zip(axes, ("delay", "cancellation"), strict=True):
        subset = importance.loc[
            importance["candidate"].eq("hypergraph")
            & importance["task"].eq(task)
        ].head(15)
        subset = subset.iloc[::-1].copy()
        labels_clean = [_clean_feature_label(str(value)) for value in subset["feature"]]
        axis.barh(labels_clean, subset["importance"], color="#F58518")
        axis.set_title(f"{task.title()} component")
        axis.set_xlabel("CatBoost PredictionValuesChange importance")
        axis.tick_params(axis="y", labelsize=8.5)
    figure.suptitle("Descriptive CC-RTH feature use in the full hypergraph candidate")
    figure.subplots_adjust(left=0.22, right=0.98, bottom=0.1, top=0.9, wspace=0.55)
    figure_records.append(
        _atomic_figure(
            figure, output_dir / "hypergraph_feature_importance_corrected.png"
        )
    )
    plt.close(figure)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_VALIDATED_TF_CCRTH_PUBLICATION_ASSETS",
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
        "parent_report": {
            "path": parent_report_path.as_posix(),
            "sha256": sha256_file(parent_report_path),
            "self_hash": parent["report_sha256"],
        },
        "tables": table_records,
        "figures": figure_records,
        "visual_qa_required": True,
        "evidence_note": (
            "Q4-locked factorized results are retrospective evidence. The 25-pair "
            "winner is explicitly post-hoc, its interval is not selection-adjusted, "
            "and independent confirmation remains locked to unopened 2026 outcomes."
        ),
        "confirmation_outcomes_accessed": False,
        "provenance": capture_provenance(
            (Path(__file__), Path(feature_importance_table.__code__.co_filename))
        ),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = build_task_factorized_publication_assets(
        args.report,
        validation_path=args.validation,
        parent_report_path=args.parent_report,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
    )
    print(json.dumps({"status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
