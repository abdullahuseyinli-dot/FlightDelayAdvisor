"""Prospective recalibration with closed-left operational shift signals."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import fit_binary_calibrator
from .ensemble import _eligible, _labels, _prediction_path, _verify_report
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .metrics import binary_metrics, clip_probabilities
from .rolling import _atomic_joblib, _atomic_parquet


def _logit(values: NDArray[np.float64]) -> NDArray[np.float64]:
    probabilities = clip_probabilities(values)
    return np.asarray(np.log(probabilities / (1.0 - probabilities)), dtype=np.float64)


def _rate(frame: pd.DataFrame, column: str) -> NDArray[np.float64]:
    if column not in frame:
        raise ValueError(f"shift-aware prediction artifact is missing {column}")
    return np.asarray(pd.to_numeric(frame[column], errors="raise"), dtype=np.float64)


def operational_shift_matrix(frame: pd.DataFrame, task: str) -> NDArray[np.float64]:
    """Construct context using only probabilities and information through D-1."""

    if task not in {"delay", "cancellation"}:
        raise ValueError(f"unsupported shift-recalibration task: {task}")
    outcome = "delay" if task == "delay" else "cancel"
    probability = _rate(frame, "probability")
    baseline_global = _rate(frame, "baseline_global")
    baseline_route = _rate(frame, "baseline_route")
    global_rates = {
        window: _rate(frame, f"recent_global_{outcome}_rate_{window}d")
        for window in (7, 28, 90)
    }
    route_rates = {
        window: _rate(frame, f"recent_route_{outcome}_rate_{window}d")
        for window in (7, 28, 90)
    }
    airline_rates = {
        window: _rate(frame, f"recent_airline_{outcome}_rate_{window}d")
        for window in (7, 28, 90)
    }
    origin_rates = {
        window: _rate(frame, f"recent_origin_outbound_{outcome}_rate_{window}d")
        for window in (7, 28, 90)
    }
    dest_rates = {
        window: _rate(frame, f"recent_dest_inbound_{outcome}_rate_{window}d")
        for window in (7, 28, 90)
    }
    month = _rate(frame, "Month")
    columns: list[NDArray[np.float64]] = [
        _logit(probability),
        _logit(global_rates[7]) - _logit(baseline_global),
        _logit(global_rates[28]) - _logit(baseline_global),
        _logit(global_rates[90]) - _logit(baseline_global),
        _logit(route_rates[7]) - _logit(baseline_route),
        _logit(route_rates[28]) - _logit(baseline_route),
        _logit(route_rates[90]) - _logit(baseline_route),
        _logit(airline_rates[7]) - _logit(baseline_global),
        _logit(airline_rates[28]) - _logit(baseline_global),
        _logit(origin_rates[7]) - _logit(baseline_global),
        _logit(origin_rates[28]) - _logit(baseline_global),
        _logit(dest_rates[7]) - _logit(baseline_global),
        _logit(dest_rates[28]) - _logit(baseline_global),
        global_rates[7] - global_rates[90],
        global_rates[7] - 2.0 * global_rates[28] + global_rates[90],
        (origin_rates[28] + dest_rates[28]) / 2.0 - global_rates[28],
        np.sin(2.0 * np.pi * month / 12.0),
        np.cos(2.0 * np.pi * month / 12.0),
    ]
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise ValueError("operational shift matrix contains non-finite values")
    return np.asarray(matrix, dtype=np.float64)


def _fit_shift_calibrator(
    matrix: NDArray[np.float64], labels: NDArray[np.int64]
) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(C=0.25, solver="lbfgs", max_iter=1_000)),
        ]
    )
    model.fit(matrix, labels)
    return model


def _loaded_year(
    report: dict[str, Any], task: str, year: int
) -> tuple[pd.DataFrame, NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    frame = pd.read_parquet(_prediction_path(report, task, year))
    mask = _eligible(frame, task)
    eligible = frame.loc[mask].reset_index(drop=True)
    labels = _labels(frame, task, mask)
    probabilities = _rate(eligible, "probability")
    matrix = operational_shift_matrix(eligible, task)
    return eligible, labels, probabilities, matrix


def analyse_shift_recalibration(
    rolling_report_path: Path,
    *,
    run_dir: Path,
    output_path: Path,
    years: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023),
) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse recalibration run directory: {run_dir}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite recalibration report: {output_path}")
    if tuple(sorted(set(years))) != years or len(years) < 3:
        raise ValueError("years must be unique, chronological, and contain at least three periods")
    report = _verify_report(rolling_report_path)
    available_years = set(int(value) for value in report["target_years"])
    if not set(years).issubset(available_years):
        raise ValueError("requested recalibration years are missing from the rolling report")
    run_dir.mkdir(parents=True)

    all_records: dict[str, Any] = {}
    artifacts: list[dict[str, Any]] = []
    for task in ("delay", "cancellation"):
        data = {year: _loaded_year(report, task, year) for year in years}
        records: list[dict[str, Any]] = []
        for year in years:
            _, labels, probabilities, _ = data[year]
            records.append(
                {"method": "raw", "year": year, "metrics": binary_metrics(labels, probabilities).as_dict()}
            )
        for target_year in years[1:]:
            prior_years = tuple(year for year in years if year < target_year)
            prior_labels = np.concatenate([data[year][1] for year in prior_years])
            prior_probabilities = np.concatenate([data[year][2] for year in prior_years])
            prior_matrix = np.concatenate([data[year][3] for year in prior_years])
            eligible, target_labels, target_probabilities, target_matrix = data[target_year]

            platt = fit_binary_calibrator("platt", prior_probabilities, prior_labels)
            platt_probabilities = platt.predict(target_probabilities)
            expanding = _fit_shift_calibrator(prior_matrix, prior_labels)
            expanding_probabilities = np.asarray(
                expanding.predict_proba(target_matrix)[:, 1], dtype=np.float64
            )
            last_year = target_year - 1
            last = _fit_shift_calibrator(data[last_year][3], data[last_year][1])
            last_probabilities = np.asarray(
                last.predict_proba(target_matrix)[:, 1], dtype=np.float64
            )
            records.extend(
                [
                    {
                        "method": "expanding_platt",
                        "year": target_year,
                        "fit_years": list(prior_years),
                        "metrics": binary_metrics(target_labels, platt_probabilities).as_dict(),
                    },
                    {
                        "method": "expanding_operational_shift",
                        "year": target_year,
                        "fit_years": list(prior_years),
                        "metrics": binary_metrics(
                            target_labels, expanding_probabilities
                        ).as_dict(),
                    },
                    {
                        "method": "last_year_operational_shift",
                        "year": target_year,
                        "fit_years": [last_year],
                        "metrics": binary_metrics(target_labels, last_probabilities).as_dict(),
                    },
                ]
            )
            for name, model in (("expanding", expanding), ("last_year", last)):
                artifacts.append(
                    {
                        "kind": "calibrator",
                        "task": task,
                        "year": target_year,
                        "method": name,
                        **_atomic_joblib(
                            model,
                            run_dir / "models" / f"{task}_{target_year}_{name}.joblib",
                        ),
                    }
                )
            prediction_frame = eligible.loc[:, ["sample_id", "FlightDate", "Month"]].copy()
            prediction_frame["label"] = target_labels.astype("int8")
            prediction_frame["prob_raw"] = target_probabilities.astype("float32")
            prediction_frame["prob_expanding_platt"] = platt_probabilities.astype("float32")
            prediction_frame["prob_expanding_shift"] = expanding_probabilities.astype("float32")
            prediction_frame["prob_last_year_shift"] = last_probabilities.astype("float32")
            artifacts.append(
                {
                    "kind": "predictions",
                    "task": task,
                    "year": target_year,
                    **_atomic_parquet(
                        prediction_frame,
                        run_dir / "predictions" / f"{task}_{target_year}.parquet",
                    ),
                }
            )

        aggregate: list[dict[str, Any]] = []
        for method in sorted({str(record["method"]) for record in records}):
            selected = [
                record
                for record in records
                if record["method"] == method and int(record["year"]) in years[1:]
            ]
            if len(selected) != len(years) - 1:
                continue
            aggregate.append(
                {
                    "method": method,
                    "years": list(years[1:]),
                    "mean_year_log_loss": float(
                        np.mean([record["metrics"]["log_loss"] for record in selected])
                    ),
                    "mean_year_brier": float(
                        np.mean([record["metrics"]["brier"] for record in selected])
                    ),
                    "mean_year_roc_auc": float(
                        np.mean([record["metrics"]["roc_auc"] for record in selected])
                    ),
                }
            )
        aggregate.sort(key=lambda item: item["mean_year_log_loss"])
        all_records[task] = {"records": records, "aggregate": aggregate}

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_DEVELOPMENT_ANALYSIS_NOT_CONFIRMATORY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "closed_left_operational_shift_recalibration",
        "rolling_report": rolling_report_path.as_posix(),
        "rolling_report_sha256": sha256_file(rolling_report_path),
        "years": list(years),
        "adaptation_rule": (
            "For target year Y, all fitted coefficients use rolling predictions and labels "
            "strictly before Y. Target-year covariates are closed-left statistics through D-1."
        ),
        "regularization_C": 0.25,
        "tasks": all_records,
        "artifacts": artifacts,
    }
    result["result_sha256"] = canonical_json_sha256(result)
    write_canonical_json(output_path, result)
    write_canonical_json(run_dir / "run_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rolling_report", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2019, 2020, 2021, 2022, 2023])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyse_shift_recalibration(
        args.rolling_report,
        run_dir=args.run_dir,
        output_path=args.output,
        years=tuple(args.years),
    )
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "top": {task: value["aggregate"][:3] for task, value in result["tasks"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
