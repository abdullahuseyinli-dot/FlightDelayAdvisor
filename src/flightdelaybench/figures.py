"""Generate create-only publication figures from self-hashed evidence reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

COLORS = {
    "delay": "#0072B2",
    "cancellation": "#D55E00",
    "prevalence": "#009E73",
}


def _verify_report(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    fields = [name for name in ("report_sha256", "result_sha256") if name in payload]
    if len(fields) != 1:
        raise ValueError(f"expected exactly one report self-hash field: {path}")
    field = fields[0]
    recorded = payload[field]
    body = {key: value for key, value in payload.items() if key != field}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"report self-hash failed: {path}")
    return payload


def _task(report: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in report["task_results"] if item["task"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one task result for {name}")
    result: dict[str, Any] = matches[0]
    return result


def _save_figure(figure: Any, output_dir: Path, stem: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for suffix, kwargs in (
        ("png", {"dpi": 240}),
        ("pdf", {}),
    ):
        path = output_dir / f"{stem}.{suffix}"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite figure: {path}")
        partial = output_dir / f"{stem}.part.{suffix}"
        if partial.exists():
            raise FileExistsError(f"unadjudicated partial figure exists: {partial}")
        figure.savefig(partial, bbox_inches="tight", **kwargs)
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


def _effect_figure(selection: dict[str, Any], audit: dict[str, Any]) -> Any:
    rows: list[tuple[str, str, str, float, float, float]] = []
    for stage, report in (("2024 Q4 selection", selection), ("2025 audit", audit)):
        for task_name in ("delay", "cancellation"):
            task = _task(report, task_name)
            for comparator, field in (
                ("frozen baseline", "paired_forecast_minus_baseline"),
                ("operational residual", "paired_forecast_minus_operational_residual"),
            ):
                interval = task[field]["log_loss"]
                rows.append(
                    (
                        stage,
                        task_name,
                        comparator,
                        float(interval["estimate"]),
                        float(interval["lower"]),
                        float(interval["upper"]),
                    )
                )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=False)
    for axis, task_name in zip(axes, ("delay", "cancellation"), strict=True):
        selected = [row for row in rows if row[1] == task_name]
        positions = np.arange(len(selected))
        estimates = np.asarray([row[3] for row in selected])
        lower = np.asarray([row[4] for row in selected])
        upper = np.asarray([row[5] for row in selected])
        axis.errorbar(
            estimates,
            positions,
            xerr=np.vstack((estimates - lower, upper - estimates)),
            fmt="o",
            color=COLORS[task_name],
            ecolor=COLORS[task_name],
            capsize=4,
            linewidth=1.8,
        )
        axis.axvline(0.0, color="#333333", linewidth=1, linestyle="--")
        axis.set_yticks(
            positions,
            [f"{stage}\nvs {comparator}" for stage, _, comparator, *_ in selected],
        )
        axis.invert_yaxis()
        axis.set_title(task_name.title())
        axis.set_xlabel("PAFRA minus comparator log loss\n(negative favors PAFRA)")
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("Fixed-lead forecast contribution replicates out of time", fontweight="bold")
    figure.tight_layout()
    return figure


def _weather_figure(diagnostics: dict[str, Any]) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    for axis, task_name in zip(axes, ("delay", "cancellation"), strict=True):
        task = _task(diagnostics, task_name)
        rows = [item for item in task["weather_strata"] if item["stratum"] != "missing"]
        labels = [str(item["stratum"]).title() for item in rows]
        intervals = [item["paired_forecast_minus_baseline"]["log_loss"] for item in rows]
        estimates = np.asarray([float(item["estimate"]) for item in intervals])
        lower = np.asarray([float(item["lower"]) for item in intervals])
        upper = np.asarray([float(item["upper"]) for item in intervals])
        positions = np.arange(len(rows))
        axis.errorbar(
            estimates,
            positions,
            xerr=np.vstack((estimates - lower, upper - estimates)),
            fmt="o",
            color=COLORS[task_name],
            ecolor=COLORS[task_name],
            capsize=4,
            linewidth=1.8,
        )
        axis.axvline(0.0, color="#333333", linewidth=1, linestyle="--")
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_title(task_name.title())
        axis.set_xlabel("PAFRA minus baseline log loss")
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("Forecast benefit by pre-outcome weather severity", fontweight="bold")
    figure.tight_layout()
    return figure


def _reliability_figure(diagnostics: dict[str, Any]) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, task_name in zip(axes, ("delay", "cancellation"), strict=True):
        task = _task(diagnostics, task_name)
        predicted = np.asarray([item["probability_mean"] for item in task["reliability_bins"]])
        observed = np.asarray([item["observed_rate"] for item in task["reliability_bins"]])
        upper = float(max(predicted.max(), observed.max()) * 1.08)
        axis.plot([0, upper], [0, upper], "--", color="#666666", label="ideal")
        axis.plot(
            predicted,
            observed,
            marker="o",
            color=COLORS[task_name],
            linewidth=1.6,
            label="PAFRA",
        )
        axis.set_xlim(0, upper)
        axis.set_ylim(0, upper)
        axis.set_title(task_name.title())
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Observed rate")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("2025 equal-mass reliability", fontweight="bold")
    figure.tight_layout()
    return figure


def _rolling_figure(rolling: dict[str, Any]) -> Any:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True)
    years = sorted({int(item["target_year"]) for item in rolling["fold_results"]})
    for column, task_name in enumerate(("delay", "cancellation")):
        rows = sorted(
            (item for item in rolling["fold_results"] if item["task"] == task_name),
            key=lambda item: int(item["target_year"]),
        )
        axes[0, column].plot(
            years,
            [item["metrics"]["roc_auc"] for item in rows],
            marker="o",
            color=COLORS[task_name],
        )
        axes[0, column].set_title(task_name.title())
        axes[0, column].set_ylabel("AUROC")
        axes[0, column].grid(alpha=0.25)
        axes[1, column].plot(
            years,
            [item["metrics"]["prevalence"] for item in rows],
            marker="o",
            color=COLORS["prevalence"],
        )
        axes[1, column].set_ylabel("Outcome prevalence")
        axes[1, column].set_xlabel("Forward evaluation year")
        axes[1, column].grid(alpha=0.25)
        axes[1, column].set_xticks(years)
    figure.suptitle("Rolling-origin discrimination and regime shift", fontweight="bold")
    figure.tight_layout()
    return figure


def generate_publication_figures(
    *,
    selection_report_path: Path,
    audit_report_path: Path,
    diagnostics_report_path: Path,
    rolling_report_path: Path,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse figure directory: {output_dir}")
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite figure manifest: {manifest_path}")
    paths = (
        selection_report_path,
        audit_report_path,
        diagnostics_report_path,
        rolling_report_path,
    )
    selection, audit, diagnostics, rolling = (_verify_report(path) for path in paths)
    output_dir.mkdir(parents=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    artifacts: list[dict[str, Any]] = []
    for stem, figure in (
        ("pafra_paired_effects", _effect_figure(selection, audit)),
        ("pafra_weather_strata", _weather_figure(diagnostics)),
        ("pafra_reliability", _reliability_figure(diagnostics)),
        ("hmop_rolling_regimes", _rolling_figure(rolling)),
    ):
        artifacts.extend(_save_figure(figure, output_dir, stem))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_PUBLICATION_FIGURE_BUNDLE",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_reports": [
            {"path": path.as_posix(), "sha256": sha256_file(path)} for path in paths
        ],
        "artifacts": artifacts,
        "provenance": capture_provenance((Path(__file__),)),
        "claim_limit": "Figures inherit the evidence roles and limitations of their source reports.",
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--diagnostics-report", type=Path, required=True)
    parser.add_argument("--rolling-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = generate_publication_figures(
        selection_report_path=args.selection_report,
        audit_report_path=args.audit_report,
        diagnostics_report_path=args.diagnostics_report,
        rolling_report_path=args.rolling_report,
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
