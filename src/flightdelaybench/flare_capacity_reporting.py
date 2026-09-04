"""Create publication tables and figures from a completed CC-RTH study report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .flare_capacity_contracts import CAPACITY_FEATURE_REGISTRY
from .flare_capacity_study import CAPACITY_CANDIDATES
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

DISPLAY_NAMES = {
    "flare24": "FLARE-24",
    "raw_demand": "+ demand",
    "normalized_capacity": "+ normalized capacity",
    "queue_shadow": "+ queue / shadow price",
    "hypergraph": "+ graph messages",
    "global_simplex": "Global simplex",
    "capacity_gated_simplex": "Capacity-gated simplex",
}
METHOD_ORDER = (
    *CAPACITY_CANDIDATES,
    "global_simplex",
    "capacity_gated_simplex",
)
COLORS = {
    "flare24": "#5B6770",
    "raw_demand": "#4C78A8",
    "normalized_capacity": "#72B7B2",
    "queue_shadow": "#F2CF5B",
    "hypergraph": "#F58518",
    "global_simplex": "#B279A2",
    "capacity_gated_simplex": "#E45756",
}


def _load_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = str(report.get("report_sha256", ""))
    if canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    ) != recorded:
        raise ValueError("CC-RTH report self-hash failed")
    if report.get("status") != "COMPLETE_CCRTH_2024_SELECTION_2025_RETROSPECTIVE_EVALUATION":
        raise ValueError("CC-RTH publication assets require a complete study")
    return report


def proper_score_table(report: dict[str, Any]) -> pd.DataFrame:
    methods = report["retrospective_evaluation"]["methods"]
    baseline = methods["flare24"]["joint"]
    records: list[dict[str, Any]] = []
    for name in METHOD_ORDER:
        joint = methods[name]["joint"]
        log_delta = float(joint["log_loss"]) - float(baseline["log_loss"])
        brier_delta = float(joint["multiclass_brier"]) - float(
            baseline["multiclass_brier"]
        )
        records.append(
            {
                "method": name,
                "display_name": DISPLAY_NAMES[name],
                "n": int(joint["n"]),
                "joint_log_loss": float(joint["log_loss"]),
                "joint_log_loss_delta_vs_flare24": log_delta,
                "joint_log_loss_relative_change_percent": 100.0
                * log_delta
                / float(baseline["log_loss"]),
                "multiclass_brier": float(joint["multiclass_brier"]),
                "multiclass_brier_delta_vs_flare24": brier_delta,
                "multiclass_brier_relative_change_percent": 100.0
                * brier_delta
                / float(baseline["multiclass_brier"]),
            }
        )
    return pd.DataFrame.from_records(records)


def full_year_point_score_table(report: dict[str, Any]) -> pd.DataFrame:
    methods = report["retrospective_full_year_descriptive_joint_scores"]
    baseline = methods["flare24"]
    return pd.DataFrame.from_records(
        [
            {
                "method": name,
                "display_name": DISPLAY_NAMES[name],
                "n": int(methods[name]["n"]),
                "joint_log_loss": float(methods[name]["joint_log_loss"]),
                "joint_log_loss_delta_vs_flare24": float(
                    methods[name]["joint_log_loss"]
                )
                - float(baseline["joint_log_loss"]),
                "multiclass_brier": float(methods[name]["multiclass_brier"]),
                "multiclass_brier_delta_vs_flare24": float(
                    methods[name]["multiclass_brier"]
                )
                - float(baseline["multiclass_brier"]),
                "evidence_role": "descriptive full-year continuity; primary inference is embargoed",
            }
            for name in METHOD_ORDER
        ]
    )


def incremental_interval_table(report: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for comparison in report["retrospective_incremental_comparisons"]:
        for metric in ("joint_log_loss_difference", "multiclass_brier_difference"):
            interval = comparison[metric]
            records.append(
                {
                    "candidate": comparison["candidate"],
                    "reference": comparison["reference"],
                    "metric": metric.removesuffix("_difference"),
                    "estimate": float(interval["estimate"]),
                    "lower": float(interval["lower"]),
                    "upper": float(interval["upper"]),
                    "clusters": int(interval["clusters"]),
                    "repetitions": int(interval["repetitions"]),
                    "negative_favors_candidate": True,
                }
            )
    return pd.DataFrame.from_records(records)


def monthly_score_table(report: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for month_record in report["retrospective_monthly_scores"]:
        baseline = month_record["methods"]["flare24"]
        for name in METHOD_ORDER:
            metrics = month_record["methods"][name]
            records.append(
                {
                    "month": int(month_record["month"]),
                    "n": int(month_record["n"]),
                    "method": name,
                    "joint_log_loss": float(metrics["joint_log_loss"]),
                    "joint_log_loss_delta_vs_flare24": float(metrics["joint_log_loss"])
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
    records: list[dict[str, Any]] = []
    for regime_record in report["retrospective_capacity_regime_scores"]:
        baseline = regime_record["methods"]["flare24"]
        for name in METHOD_ORDER:
            metrics = regime_record["methods"][name]
            records.append(
                {
                    "regime": regime_record["regime"],
                    "n": int(regime_record["n"]),
                    "method": name,
                    "joint_log_loss": float(metrics["joint_log_loss"]),
                    "joint_log_loss_delta_vs_flare24": float(metrics["joint_log_loss"])
                    - float(baseline["joint_log_loss"]),
                    "multiclass_brier": float(metrics["multiclass_brier"]),
                    "multiclass_brier_delta_vs_flare24": float(
                        metrics["multiclass_brier"]
                    )
                    - float(baseline["multiclass_brier"]),
                }
            )
    return pd.DataFrame.from_records(records)


def feature_importance_table(report: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for model in report["feature_importance"]:
        for rank, feature in enumerate(model["top_capacity_40"], start=1):
            records.append(
                {
                    "candidate": model["candidate"],
                    "task": model["task"],
                    "rank": rank,
                    "feature": feature["feature"],
                    "importance": float(feature["importance"]),
                    "capacity_feature_importance_fraction": float(
                        model["capacity_feature_importance_fraction"]
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def feature_registry_table() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "feature": specification.name,
                "available_from": specification.available_from.name,
                "source": specification.source,
                "description": specification.description,
            }
            for specification in CAPACITY_FEATURE_REGISTRY
        ]
    )


def feature_coverage_table(report: dict[str, Any]) -> pd.DataFrame:
    feature_sets = {
        candidate: set(features)
        for candidate, features in report["feature_sets"].items()
    }
    return pd.DataFrame.from_records(
        [
            {
                "feature": feature,
                "nonmissing_fraction": float(fraction),
                **{
                    f"used_by_{candidate}": feature in selected
                    for candidate, selected in feature_sets.items()
                },
            }
            for feature, fraction in report[
                "capacity_feature_nonmissing_fraction"
            ].items()
        ]
    )


def airport_score_table(report: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame.from_records(report["retrospective_airport_scores"])


def _atomic_csv(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
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


def build_capacity_publication_assets(
    report_path: Path,
    *,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse CC-RTH publication directory: {output_dir}")
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite CC-RTH publication manifest: {manifest_path}")
    report = _load_report(report_path)
    output_dir.mkdir(parents=True)
    tables = {
        "proper_scores": proper_score_table(report),
        "full_year_point_scores": full_year_point_score_table(report),
        "incremental_intervals": incremental_interval_table(report),
        "monthly_scores": monthly_score_table(report),
        "capacity_regime_scores": regime_score_table(report),
        "feature_importance": feature_importance_table(report),
        "feature_registry": feature_registry_table(),
        "feature_coverage": feature_coverage_table(report),
        "airport_scores": airport_score_table(report),
    }
    table_records = [
        {"name": name, **_atomic_csv(frame, output_dir / f"{name}.csv")}
        for name, frame in tables.items()
    ]

    score = tables["proper_scores"]
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)
    y = np.arange(len(score))
    for axis, column, title in (
        (axes[0], "joint_log_loss_delta_vs_flare24", "Joint log-loss difference"),
        (axes[1], "multiclass_brier_delta_vs_flare24", "Multiclass Brier difference"),
    ):
        values = score[column].to_numpy(dtype=float)
        axis.barh(y, values, color=[COLORS[name] for name in score["method"]])
        axis.axvline(0.0, color="black", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Difference vs exact FLARE-24 (lower is better)")
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y, score["display_name"])
    axes[0].invert_yaxis()
    figure.suptitle("CC-RTH primary embargoed 2025 proper-score changes")
    figure_records = [
        {
            "name": "proper_score_deltas",
            **_atomic_figure(figure, output_dir / "proper_score_deltas.png"),
        }
    ]
    plt.close(figure)

    airports = tables["airport_scores"]
    selected_method = str(report["selected_method_by_2024_forward_score"])
    selected_airports = airports.loc[airports["method"].eq(selected_method)].copy()
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), sharey=True)
    for axis, role in zip(axes, ("origin", "destination"), strict=True):
        subset = selected_airports.loc[selected_airports["role"].eq(role)].copy()
        axis.scatter(
            subset["n"],
            subset["joint_log_loss_delta_vs_flare24"],
            s=28,
            alpha=0.75,
            color=COLORS.get(selected_method, "#2F4B7C"),
        )
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_xscale("log")
        axis.set_xlabel("Scored flights (log scale)")
        axis.set_title(role.title())
        axis.grid(alpha=0.25)
        extremes = pd.concat(
            [
                subset.nsmallest(3, "joint_log_loss_delta_vs_flare24"),
                subset.nlargest(3, "joint_log_loss_delta_vs_flare24"),
            ]
        ).drop_duplicates("airport")
        for row in extremes.itertuples(index=False):
            axis.annotate(
                row.airport,
                (row.n, row.joint_log_loss_delta_vs_flare24),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
            )
    axes[0].set_ylabel("Joint log-loss difference vs FLARE-24")
    figure.suptitle(
        f"Airport heterogeneity for Q4-selected {DISPLAY_NAMES[selected_method]}"
    )
    figure_records.append(
        {
            "name": "selected_method_airport_heterogeneity",
            **_atomic_figure(
                figure, output_dir / "selected_method_airport_heterogeneity.png"
            ),
        }
    )
    plt.close(figure)

    intervals = tables["incremental_intervals"]
    intervals = intervals.loc[intervals["metric"].eq("joint_log_loss")].reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    y = np.arange(len(intervals))
    estimate = intervals["estimate"].to_numpy(dtype=float)
    lower = intervals["lower"].to_numpy(dtype=float)
    upper = intervals["upper"].to_numpy(dtype=float)
    axis.errorbar(
        estimate,
        y,
        xerr=np.vstack((estimate - lower, upper - estimate)),
        fmt="o",
        color="#2F4B7C",
        ecolor="#7A8BA6",
        capsize=4,
    )
    labels = [
        f"{DISPLAY_NAMES[row.candidate]} vs {DISPLAY_NAMES[row.reference]}"
        for row in intervals.itertuples(index=False)
    ]
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.axvline(0.0, color="black", linewidth=1.0)
    axis.set_xlabel("Paired joint log-loss difference (95% date-cluster interval)")
    axis.set_title("Incremental CC-RTH evidence; negative favors candidate")
    axis.grid(axis="x", alpha=0.25)
    figure_records.append(
        {
            "name": "incremental_log_loss_intervals",
            **_atomic_figure(figure, output_dir / "incremental_log_loss_intervals.png"),
        }
    )
    plt.close(figure)

    monthly = tables["monthly_scores"]
    figure, axis = plt.subplots(figsize=(11.5, 5.8))
    for name in ("normalized_capacity", "queue_shadow", "hypergraph", "capacity_gated_simplex"):
        subset = monthly.loc[monthly["method"].eq(name)]
        axis.plot(
            subset["month"],
            subset["joint_log_loss_delta_vs_flare24"],
            marker="o",
            linewidth=2.0,
            color=COLORS[name],
            label=DISPLAY_NAMES[name],
        )
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xticks(range(1, 13))
    axis.set_xlabel("2025 month")
    axis.set_ylabel("Joint log-loss difference vs FLARE-24")
    axis.set_title("Temporal stability of CC-RTH increments")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncols=2)
    figure_records.append(
        {
            "name": "monthly_log_loss_deltas",
            **_atomic_figure(figure, output_dir / "monthly_log_loss_deltas.png"),
        }
    )
    plt.close(figure)

    regimes = tables["capacity_regime_scores"]
    selected_regimes = regimes.loc[
        regimes["method"].isin(["hypergraph", "capacity_gated_simplex"])
    ].copy()
    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    regime_order = [name for name in ("low", "elevated", "severe", "missing") if name in set(selected_regimes["regime"])]
    width = 0.36
    x = np.arange(len(regime_order))
    for offset, name in zip((-width / 2, width / 2), ("hypergraph", "capacity_gated_simplex"), strict=True):
        indexed = selected_regimes.loc[selected_regimes["method"].eq(name)].set_index("regime")
        values = [indexed.loc[regime, "joint_log_loss_delta_vs_flare24"] for regime in regime_order]
        axis.bar(x + offset, values, width=width, color=COLORS[name], label=DISPLAY_NAMES[name])
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xticks(x, regime_order)
    axis.set_ylabel("Joint log-loss difference vs FLARE-24")
    axis.set_title("CC-RTH performance by Q4-locked shadow-price regime")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure_records.append(
        {
            "name": "capacity_regime_deltas",
            **_atomic_figure(figure, output_dir / "capacity_regime_deltas.png"),
        }
    )
    plt.close(figure)

    importance = tables["feature_importance"]
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 7.2))
    for axis, task in zip(axes, ("delay", "cancellation"), strict=True):
        subset = importance.loc[
            importance["candidate"].eq("hypergraph") & importance["task"].eq(task)
        ].head(15)
        subset = subset.iloc[::-1]
        axis.barh(subset["feature"], subset["importance"], color=COLORS["hypergraph"])
        axis.set_title(f"{task.replace('_', ' ').title()} component")
        axis.set_xlabel("CatBoost PredictionValuesChange importance")
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("Most-used CC-RTH features in the full hypergraph candidate")
    figure_records.append(
        {
            "name": "hypergraph_feature_importance",
            **_atomic_figure(figure, output_dir / "hypergraph_feature_importance.png"),
        }
    )
    plt.close(figure)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_CCRTH_PUBLICATION_ASSETS",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_report": {
            "path": report_path.as_posix(),
            "sha256": sha256_file(report_path),
            "self_hash": report["report_sha256"],
        },
        "tables": table_records,
        "figures": figure_records,
        "figure_note": (
            "All differences use the exact previous FLARE-24 probabilities; negative "
            "proper-score differences favor CC-RTH. Feature importance is descriptive."
        ),
        "provenance": capture_provenance((Path(__file__),)),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = build_capacity_publication_assets(
        args.report,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
    )
    print(json.dumps({"status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
