"""Paired nested-ablation analysis for the frozen FLARE-24 audit."""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .bootstrap import paired_cluster_mean_difference
from .flare_audit_recovery import AUDIT_METHODS, JOINT_STATES, normalize_persisted_joint
from .flare_evaluation import joint_loss_rows
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

NESTED_COMPARISONS = (
    ("weather", "baseline", "increment from fixed-vintage weather and aviation features"),
    (
        "rotation_structural",
        "weather",
        "increment from schedule-only latent rotation structure",
    ),
    (
        "rotation_risk",
        "rotation_structural",
        "increment from propagated closed-left predecessor risk",
    ),
    ("ensemble", "rotation_structural", "increment from selected convex blending"),
    ("reconciled", "ensemble", "increment from selected aggregate alignment"),
)


def binary_loss_rows(
    labels: ArrayLike,
    probabilities: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return decomposable binary log and Brier losses with project clipping."""

    y = np.asarray(labels, dtype=np.int8)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if y.ndim != 1 or p.ndim != 1 or len(y) == 0 or len(y) != len(p):
        raise ValueError("binary labels and probabilities must align")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("binary ablation inputs are invalid")
    log_rows = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    return np.asarray(log_rows, dtype=np.float64), np.square(p - y)


def _self_hashed_report(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("report_sha256")
    body = {name: value for name, value in payload.items() if name != "report_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"FLARE-24 audit report self-hash failed: {path}")
    return payload


def _atomic_csv(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite FLARE-24 ablation table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated FLARE-24 ablation table partial: {partial}")
    frame.to_csv(partial, index=False, lineterminator="\n")
    partial.replace(path)
    return {
        "path": path.as_posix(),
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _paired(
    candidate: NDArray[np.float64],
    reference: NDArray[np.float64],
    clusters: NDArray[Any],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    return asdict(
        paired_cluster_mean_difference(
            candidate,
            reference,
            clusters,
            repetitions=repetitions,
            seed=seed,
        )
    )


def run_nested_ablation(
    *,
    audit_report_path: Path,
    output_path: Path,
    table_path: Path,
) -> dict[str, Any]:
    """Compute all prespecified incremental contrasts from saved audit probabilities."""

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite FLARE-24 ablation report: {output_path}")
    audit = _self_hashed_report(audit_report_path)
    if audit.get("status") != (
        "COMPLETE_2025_FLARE24_RETROSPECTIVE_AUDIT_NOT_BLIND_CONFIRMATION"
    ):
        raise ValueError("nested ablation requires a completed FLARE-24 audit")
    repetitions = int(audit["primary_evaluation"]["bootstrap_repetitions"])
    seed = int(audit["primary_evaluation"]["bootstrap_seed"])
    columns = [
        "FlightDate",
        "joint_label_observed",
        "disruption_state",
        "Cancelled",
        "delay_label_observed",
        "ArrDel15",
        *(
            f"prob_{method}_{state}"
            for method in AUDIT_METHODS
            for state in JOINT_STATES
        ),
    ]
    parts: list[pd.DataFrame] = []
    source_predictions: list[dict[str, Any]] = []
    for record in audit["prediction_artifacts"]:
        path = Path(str(record["path"]))
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"FLARE-24 nested-ablation prediction changed: {path}")
        frame = pd.read_parquet(path, columns=columns)
        if len(frame) != int(record["rows"]):
            raise ValueError(f"FLARE-24 nested-ablation prediction row count failed: {path}")
        parts.append(frame)
        source_predictions.append(
            {
                "path": path.as_posix(),
                "sha256": record["sha256"],
                "rows": len(frame),
                "year": int(record["year"]),
                "month": int(record["month"]),
            }
        )
    evidence = pd.concat(parts, ignore_index=True)
    del parts
    probabilities: dict[str, NDArray[np.float64]] = {}
    normalization: dict[str, Any] = {}
    for method in AUDIT_METHODS:
        method_columns = [f"prob_{method}_{state}" for state in JOINT_STATES]
        probabilities[method], normalization[method] = normalize_persisted_joint(
            evidence.loc[:, method_columns].to_numpy(dtype=np.float32),
            method=method,
        )
    joint_mask = evidence["joint_label_observed"].eq(1).to_numpy()
    joint_labels = evidence.loc[joint_mask, "disruption_state"].to_numpy(dtype=np.int64)
    joint_clusters = evidence.loc[joint_mask, "FlightDate"].to_numpy()
    cancellation_mask = evidence["Cancelled"].isin([0, 1]).to_numpy()
    cancellation_labels = evidence.loc[cancellation_mask, "Cancelled"].to_numpy(
        dtype=np.int8
    )
    cancellation_clusters = evidence.loc[cancellation_mask, "FlightDate"].to_numpy()
    delay_mask = (
        evidence["Cancelled"].eq(0) & evidence["delay_label_observed"].eq(1)
    ).to_numpy()
    delay_labels = evidence.loc[delay_mask, "ArrDel15"].to_numpy(dtype=np.int8)
    delay_clusters = evidence.loc[delay_mask, "FlightDate"].to_numpy()

    losses: dict[str, dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]] = {}
    for method, values in probabilities.items():
        operated = values[:, 0] + values[:, 1]
        conditional_delay = np.divide(
            values[:, 1],
            operated,
            out=np.zeros_like(operated),
            where=operated > 1e-12,
        )
        losses[method] = {
            "joint": joint_loss_rows(joint_labels, values[joint_mask]),
            "delay_given_operated": binary_loss_rows(
                delay_labels, conditional_delay[delay_mask]
            ),
            "cancellation": binary_loss_rows(
                cancellation_labels, values[cancellation_mask, 2]
            ),
        }
    clusters = {
        "joint": joint_clusters,
        "delay_given_operated": delay_clusters,
        "cancellation": cancellation_clusters,
    }
    metric_names = {
        "joint": ("joint_log_loss", "multiclass_brier"),
        "delay_given_operated": ("log_loss", "brier"),
        "cancellation": ("log_loss", "brier"),
    }
    comparison_records: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for candidate, reference, interpretation in NESTED_COMPARISONS:
        endpoint_results: dict[str, Any] = {}
        for endpoint, names in metric_names.items():
            endpoint_results[endpoint] = {}
            for metric_index, metric_name in enumerate(names):
                interval = _paired(
                    losses[candidate][endpoint][metric_index],
                    losses[reference][endpoint][metric_index],
                    clusters[endpoint],
                    repetitions=repetitions,
                    seed=seed,
                )
                endpoint_results[endpoint][metric_name] = interval
                table_rows.append(
                    {
                        "candidate": candidate,
                        "reference": reference,
                        "endpoint": endpoint,
                        "metric": metric_name,
                        **interval,
                        "direction": "candidate_minus_reference; negative_favours_candidate",
                    }
                )
        comparison_records.append(
            {
                "candidate": candidate,
                "reference": reference,
                "interpretation": interpretation,
                "endpoints": endpoint_results,
            }
        )
    table_artifact = _atomic_csv(pd.DataFrame(table_rows), table_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_2025_FLARE24_NESTED_ABLATION_ANALYSIS",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_audit": {
            "path": audit_report_path.as_posix(),
            "sha256": sha256_file(audit_report_path),
            "self_hash": audit["report_sha256"],
        },
        "source_predictions": source_predictions,
        "evaluation_year": 2025,
        "comparison_order": [
            {"candidate": candidate, "reference": reference}
            for candidate, reference, _ in NESTED_COMPARISONS
        ],
        "comparisons": comparison_records,
        "table_artifact": table_artifact,
        "probability_normalization_audit": normalization,
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
        "confirmation_outcomes_accessed": False,
        "provenance": capture_provenance(
            (
                Path(__file__),
                Path(__file__).with_name("bootstrap.py"),
                Path(__file__).with_name("flare_evaluation.py"),
                Path(__file__).with_name("flare_audit_recovery.py"),
            )
        ),
        "claim_limit": (
            "These are prespecified paired nested contrasts on a retrospective audit. "
            "A confidence interval crossing zero is retained and is not relabelled as a gain."
        ),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(output_path, report)
    del evidence, probabilities, losses
    gc.collect()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_nested_ablation(
        audit_report_path=args.audit_report,
        output_path=args.output,
        table_path=args.table,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": args.output.as_posix(),
                "report_sha256": report["report_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
