"""Create immutable tables and figures from validated FLARE-24 reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .flare_study import CANDIDATES
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

REPORT_METHODS = (*CANDIDATES, "ensemble", "reconciled")
METHOD_LABELS = {
    "baseline": "Schedule/history baseline",
    "weather": "+ forecast weather",
    "rotation_structural": "+ latent rotation structure",
    "rotation_risk": "+ propagated predecessor risk",
    "ensemble": "Selected blend (100% structural rotation)",
    "reconciled": "Final output (aggregate alignment rejected)",
}
ABLATION_METHODS = ("weather", "rotation_structural", "rotation_risk")


def _verified_report(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("report_sha256")
    body = {key: value for key, value in payload.items() if key != "report_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"FLARE-24 report self-hash failed: {path}")
    return payload


def main_score_table(audit: dict[str, Any]) -> pd.DataFrame:
    """Flatten the primary proper-score and binary diagnostic views."""

    methods = audit["primary_evaluation"]["methods"]
    if set(methods) != set(REPORT_METHODS):
        raise ValueError("FLARE-24 audit method set is incomplete")
    rows: list[dict[str, Any]] = []
    for method in REPORT_METHODS:
        result = methods[method]
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "joint_n": int(result["joint"]["n"]),
                "joint_log_loss": float(result["joint"]["log_loss"]),
                "joint_multiclass_brier": float(result["joint"]["multiclass_brier"]),
                "delay_n": int(result["delay_given_operated"]["n"]),
                "delay_log_loss": float(result["delay_given_operated"]["log_loss"]),
                "delay_brier": float(result["delay_given_operated"]["brier"]),
                "delay_roc_auc": float(result["delay_given_operated"]["roc_auc"]),
                "delay_average_precision": float(
                    result["delay_given_operated"]["average_precision"]
                ),
                "cancellation_n": int(result["cancellation"]["n"]),
                "cancellation_log_loss": float(result["cancellation"]["log_loss"]),
                "cancellation_brier": float(result["cancellation"]["brier"]),
                "cancellation_roc_auc": float(result["cancellation"]["roc_auc"]),
                "cancellation_average_precision": float(
                    result["cancellation"]["average_precision"]
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_interval_table(audit: dict[str, Any]) -> pd.DataFrame:
    """Flatten paired date-cluster intervals, preserving their direction."""

    comparisons = audit["primary_evaluation"]["paired_date_cluster_comparisons"]
    rows: list[dict[str, Any]] = []
    for comparison, metrics in comparisons.items():
        for metric in ("joint_log_loss", "multiclass_brier"):
            interval = metrics[metric]
            rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "estimate": float(interval["estimate"]),
                    "lower": float(interval["lower"]),
                    "upper": float(interval["upper"]),
                    "confidence": float(interval["confidence"]),
                    "date_clusters": int(interval["clusters"]),
                    "bootstrap_repetitions": int(interval["repetitions"]),
                    "seed": int(interval["seed"]),
                    "direction": "candidate_minus_baseline; negative_favours_candidate",
                }
            )
    return pd.DataFrame(rows)


def monthly_score_table(audit: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month_record in audit["monthly_proper_scores"]:
        scores = month_record["proper_scores"]
        baseline = scores["baseline"]
        for method in REPORT_METHODS:
            result = scores[method]
            rows.append(
                {
                    "year": int(month_record["year"]),
                    "month": int(month_record["month"]),
                    "method": method,
                    "rows": int(result["n"]),
                    "joint_log_loss": float(result["joint_log_loss"]),
                    "multiclass_brier": float(result["multiclass_brier"]),
                    "joint_log_loss_minus_baseline": float(
                        result["joint_log_loss"] - baseline["joint_log_loss"]
                    ),
                    "multiclass_brier_minus_baseline": float(
                        result["multiclass_brier"] - baseline["multiclass_brier"]
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    if set(zip(frame["year"], frame["month"], strict=True)) != {
        (2025, month) for month in range(1, 13)
    }:
        raise ValueError("FLARE-24 monthly reporting table is incomplete")
    return frame


def calibration_table(selection: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for artifact in selection["calibration"]["artifacts"]:
        for candidate in artifact["candidate_methods"]:
            rows.append(
                {
                    "flight_candidate": artifact["candidate"],
                    "task": artifact["task"],
                    "calibration_method": candidate["method"],
                    "selected": candidate["method"] == artifact["selected_method"],
                    "pooled_rows": int(candidate["pooled_rows"]),
                    "pooled_log_loss": float(candidate["pooled_log_loss"]),
                    "pooled_brier": float(candidate["pooled_brier"]),
                }
            )
    return pd.DataFrame(rows)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite FLARE-24 report table: {path}")
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated FLARE-24 report-table partial exists: {partial}")
    frame.to_csv(partial, index=False, lineterminator="\n")
    partial.replace(path)
    return {
        "kind": "table",
        "path": path.as_posix(),
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _save_figure(figure: Any, output_dir: Path, stem: str) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    records: list[dict[str, Any]] = []
    for suffix, options in (("png", {"dpi": 240}), ("pdf", {})):
        path = output_dir / f"{stem}.{suffix}"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite FLARE-24 figure: {path}")
        partial = output_dir / f"{stem}.part.{suffix}"
        figure.savefig(partial, bbox_inches="tight", **options)
        partial.replace(path)
        records.append(
            {
                "kind": "figure",
                "stem": stem,
                "format": suffix,
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    plt.close(figure)
    return records


def _ablation_figure(scores: pd.DataFrame) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = scores.loc[scores["method"].eq("baseline")].iloc[0]
    candidates = scores.loc[scores["method"].isin(ABLATION_METHODS)].reset_index(drop=True)
    labels = candidates["label"].tolist()
    positions = np.arange(len(candidates))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, metric, title in (
        (axes[0], "joint_log_loss", "Joint log loss"),
        (axes[1], "joint_multiclass_brier", "Multiclass Brier"),
    ):
        differences = candidates[metric].to_numpy(dtype=np.float64) - float(baseline[metric])
        colors = np.where(differences < 0.0, "#0072B2", "#D55E00")
        axis.barh(positions, differences, color=colors)
        axis.axvline(0.0, color="#333333", linestyle="--", linewidth=1)
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlabel("Difference from baseline (negative is better)")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("FLARE-24 2025 feature ablations", fontweight="bold")
    figure.tight_layout()
    return figure


def _interval_figure(intervals: pd.DataFrame) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparisons = {f"{method}_minus_baseline" for method in ABLATION_METHODS}
    rows = intervals.loc[
        intervals["metric"].eq("joint_log_loss")
        & intervals["comparison"].isin(comparisons)
    ].reset_index(drop=True)
    estimates = rows["estimate"].to_numpy(dtype=np.float64)
    lower = rows["lower"].to_numpy(dtype=np.float64)
    upper = rows["upper"].to_numpy(dtype=np.float64)
    positions = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    axis.errorbar(
        estimates,
        positions,
        xerr=np.vstack((estimates - lower, upper - estimates)),
        fmt="o",
        color="#0072B2",
        ecolor="#0072B2",
        capsize=4,
    )
    axis.axvline(0.0, color="#333333", linestyle="--", linewidth=1)
    axis.set_yticks(positions, rows["comparison"].str.replace("_", " "))
    axis.invert_yaxis()
    axis.set_xlabel("Candidate minus baseline joint log loss")
    axis.set_title("Paired 95% date-cluster intervals", fontweight="bold")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    return figure


def _monthly_figure(monthly: pd.DataFrame) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10.5, 5.0))
    for method in ABLATION_METHODS:
        rows = monthly.loc[monthly["method"].eq(method)].sort_values("month")
        axis.plot(
            rows["month"],
            rows["joint_log_loss_minus_baseline"],
            marker="o",
            linewidth=1.5,
            label=METHOD_LABELS[method],
        )
    axis.axhline(0.0, color="#333333", linestyle="--", linewidth=1)
    axis.set_xticks(range(1, 13))
    axis.set_xlabel("2025 month")
    axis.set_ylabel("Joint log loss minus baseline")
    axis.set_title("Temporal stability of FLARE-24 gains", fontweight="bold")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    return figure


def generate_flare24_publication_bundle(
    *,
    selection_report_path: Path,
    audit_report_path: Path,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Generate create-only reporting artifacts from frozen FLARE-24 evidence."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse FLARE-24 publication directory: {output_dir}")
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite FLARE-24 figure manifest: {manifest_path}")
    selection = _verified_report(selection_report_path)
    audit = _verified_report(audit_report_path)
    if selection.get("status") != "COMPLETE_2024_FLARE24_SELECTION_NOT_CONFIRMATORY":
        raise ValueError("FLARE-24 reporting requires a completed selection report")
    if audit.get("status") != "COMPLETE_2025_FLARE24_RETROSPECTIVE_AUDIT_NOT_BLIND_CONFIRMATION":
        raise ValueError("FLARE-24 reporting requires a completed audit report")
    linked = audit.get("selection_report", {})
    if linked.get("sha256") != sha256_file(selection_report_path):
        raise ValueError("FLARE-24 audit does not reference the supplied selection report")
    output_dir.mkdir(parents=True)
    scores = main_score_table(audit)
    intervals = paired_interval_table(audit)
    monthly = monthly_score_table(audit)
    calibration = calibration_table(selection)
    reconciliation = pd.DataFrame(selection["reconciliation_selection"]["candidates"])
    ensemble = pd.DataFrame(selection["ensemble_selection"]["grid"])
    if "weights" in ensemble:
        weights = pd.json_normalize(ensemble.pop("weights")).add_prefix("weight_")
        ensemble = pd.concat([ensemble, weights], axis=1)
    artifacts: list[dict[str, Any]] = []
    for name, table in (
        ("flare24_2025_primary_scores.csv", scores),
        ("flare24_2025_paired_intervals.csv", intervals),
        ("flare24_2025_monthly_scores.csv", monthly),
        ("flare24_2024_calibration_selection.csv", calibration),
        ("flare24_2024_reconciliation_selection.csv", reconciliation),
        ("flare24_2024_ensemble_grid.csv", ensemble),
    ):
        artifacts.append(_atomic_csv(table, output_dir / name))
    for stem, figure in (
        ("flare24_2025_ablation", _ablation_figure(scores)),
        ("flare24_2025_paired_intervals", _interval_figure(intervals)),
        ("flare24_2025_monthly_stability", _monthly_figure(monthly)),
    ):
        artifacts.extend(_save_figure(figure, output_dir, stem))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_FLARE24_PUBLICATION_BUNDLE",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_reports": [
            {"path": path.as_posix(), "sha256": sha256_file(path)}
            for path in (selection_report_path, audit_report_path)
        ],
        "artifacts": artifacts,
        "confirmation_outcomes_accessed": False,
        "provenance": capture_provenance((Path(__file__),)),
        "claim_limit": "Figures and tables inherit the retrospective evidence limits of the source reports.",
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = generate_flare24_publication_bundle(
        selection_report_path=args.selection_report,
        audit_report_path=args.audit_report,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
    )
    print(
        json.dumps(
            {
                "manifest": args.manifest.as_posix(),
                "artifacts": len(manifest["artifacts"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
