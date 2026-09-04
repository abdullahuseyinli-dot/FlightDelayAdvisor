"""Compare coherent hurdle and direct three-state rolling probabilities."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bootstrap import paired_cluster_mean_difference
from .calibration import compose_hurdle_probabilities
from .ensemble import _prediction_path, _verify_report
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .metrics import multiclass_brier


def _multiclass_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0)
    clipped /= clipped.sum(axis=1, keepdims=True)
    row_log = -np.log(clipped[np.arange(len(labels)), labels])
    return {
        "n": len(labels),
        "log_loss": float(row_log.mean()),
        "multiclass_brier": multiclass_brier(labels, clipped),
        "class_prevalence": {
            str(label): float(np.mean(labels == label)) for label in range(3)
        },
    }


def analyse_joint_probabilities(
    rolling_report_path: Path,
    *,
    output_path: Path,
    years: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023),
    bootstrap_repetitions: int = 2_000,
    seed: int = 20260903,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite joint analysis: {output_path}")
    report = _verify_report(rolling_report_path)
    if "joint" not in report.get("tasks", []):
        raise ValueError("rolling report does not contain a direct joint candidate")
    if not set(years).issubset(int(value) for value in report["target_years"]):
        raise ValueError("requested joint-analysis years are unavailable")

    records: list[dict[str, Any]] = []
    for year in years:
        delay = pd.read_parquet(_prediction_path(report, "delay", year))
        cancellation = pd.read_parquet(_prediction_path(report, "cancellation", year))
        direct = pd.read_parquet(_prediction_path(report, "joint", year))
        identifiers = delay["sample_id"].to_numpy()
        if not np.array_equal(identifiers, cancellation["sample_id"].to_numpy()) or not np.array_equal(
            identifiers,
            direct["sample_id"].to_numpy(),
        ):
            raise ValueError(f"binary and joint samples are not aligned for {year}")
        mask = direct["joint_label_observed"].eq(1).to_numpy()
        labels = direct.loc[mask, "disruption_state"].to_numpy(dtype=np.int64)
        hurdle = compose_hurdle_probabilities(
            cancellation.loc[mask, "probability"].to_numpy(dtype=np.float64),
            delay.loc[mask, "probability"].to_numpy(dtype=np.float64),
        )
        direct_probabilities = direct.loc[
            mask,
            ["prob_on_time", "prob_delayed", "prob_cancelled"],
        ].to_numpy(dtype=np.float64)
        direct_probabilities = np.clip(direct_probabilities, 1e-6, 1.0)
        direct_probabilities /= direct_probabilities.sum(axis=1, keepdims=True)
        observed = np.eye(3, dtype=np.float64)[labels]
        hurdle_log = -np.log(hurdle[np.arange(len(labels)), labels])
        direct_log = -np.log(direct_probabilities[np.arange(len(labels)), labels])
        hurdle_brier = np.square(hurdle - observed).sum(axis=1)
        direct_brier = np.square(direct_probabilities - observed).sum(axis=1)
        clusters = direct.loc[mask, "FlightDate"].to_numpy()
        log_interval = paired_cluster_mean_difference(
            direct_log,
            hurdle_log,
            clusters,
            repetitions=bootstrap_repetitions,
            seed=seed,
        )
        brier_interval = paired_cluster_mean_difference(
            direct_brier,
            hurdle_brier,
            clusters,
            repetitions=bootstrap_repetitions,
            seed=seed + 1,
        )
        records.append(
            {
                "year": year,
                "hurdle": _multiclass_metrics(labels, hurdle),
                "direct": _multiclass_metrics(labels, direct_probabilities),
                "paired_direct_minus_hurdle": {
                    "direction": "negative favours direct joint model",
                    "log_loss": {
                        "estimate": log_interval.estimate,
                        "lower": log_interval.lower,
                        "upper": log_interval.upper,
                        "clusters": log_interval.clusters,
                        "repetitions": log_interval.repetitions,
                    },
                    "multiclass_brier": {
                        "estimate": brier_interval.estimate,
                        "lower": brier_interval.lower,
                        "upper": brier_interval.upper,
                        "clusters": brier_interval.clusters,
                        "repetitions": brier_interval.repetitions,
                    },
                },
            }
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_DEVELOPMENT_ANALYSIS_NOT_CONFIRMATORY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "rolling_report": rolling_report_path.as_posix(),
        "rolling_report_sha256": sha256_file(rolling_report_path),
        "years": list(years),
        "methods": {
            "hurdle": "P(cancel), then P(delay | not cancelled); probabilities sum to one",
            "direct": "one native three-class CatBoost model",
        },
        "records": records,
        "aggregate": {
            method: {
                "mean_year_log_loss": float(
                    np.mean([record[method]["log_loss"] for record in records])
                ),
                "mean_year_multiclass_brier": float(
                    np.mean([record[method]["multiclass_brier"] for record in records])
                ),
            }
            for method in ("hurdle", "direct")
        },
        "bootstrap_repetitions": bootstrap_repetitions,
        "seed": seed,
    }
    result["result_sha256"] = canonical_json_sha256(result)
    write_canonical_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rolling_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2019, 2020, 2021, 2022, 2023])
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyse_joint_probabilities(
        args.rolling_report,
        output_path=args.output,
        years=tuple(args.years),
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    print(json.dumps({"output": args.output.as_posix(), "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
