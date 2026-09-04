"""Prospective calibration and adaptive ensemble analysis of rolling predictions."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from .calibration import fit_binary_calibrator
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .metrics import binary_metrics


def _verify_report(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("report_sha256")
    body = {key: value for key, value in payload.items() if key != "report_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"rolling report self-hash failed: {path}")
    return payload


def _prediction_path(report: dict[str, Any], task: str, year: int) -> Path:
    matches = [
        artifact
        for artifact in report["artifacts"]
        if artifact["kind"] == "predictions"
        and artifact["task"] == task
        and int(artifact["year"]) == year
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one prediction artifact for {task}, {year}")
    record = matches[0]
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"prediction checksum failed: {path}")
    return path


def _eligible(frame: pd.DataFrame, task: str) -> NDArray[np.bool_]:
    if task == "delay":
        return np.asarray(
            frame["Cancelled"].eq(0) & frame["delay_label_observed"].eq(1),
            dtype=np.bool_,
        )
    return np.asarray(frame["Cancelled"].isin([0, 1]), dtype=np.bool_)


def _labels(frame: pd.DataFrame, task: str, mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    column = "ArrDel15" if task == "delay" else "Cancelled"
    return np.asarray(frame.loc[mask, column], dtype=np.int64)


def _clip(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.clip(probabilities, 1e-6, 1.0 - 1e-6)


def _logit(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    values = _clip(probabilities)
    return np.asarray(np.log(values / (1.0 - values)), dtype=np.float64)


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(1.0 / (1.0 + np.exp(-values)), dtype=np.float64)


def _fit_logit_blend(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    penalty: float = 1e-4,
) -> NDArray[np.float64]:
    logits = _logit(probabilities)
    count = probabilities.shape[1]
    uniform = np.repeat(1.0 / count, count)

    def objective(weights: NDArray[np.float64]) -> float:
        prediction = _sigmoid(logits @ weights)
        loss = -np.mean(labels * np.log(prediction) + (1 - labels) * np.log1p(-prediction))
        return float(loss + penalty * np.square(weights - uniform).sum())

    result = minimize(
        objective,
        uniform,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        options={"maxiter": 100, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"logit blend optimization failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def _apply_logit_blend(
    probabilities: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    return _sigmoid(_logit(probabilities) @ weights)


def _metric_record(
    *,
    method: str,
    year: int,
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "method": method,
        "year": year,
        "metrics": binary_metrics(labels, probabilities).as_dict(),
        **({} if detail is None else detail),
    }


def _load_aligned_year(
    reports: list[dict[str, Any]],
    candidate_names: list[str],
    *,
    task: str,
    year: int,
) -> tuple[pd.DataFrame, NDArray[np.int64], NDArray[np.float64]]:
    base = pd.read_parquet(_prediction_path(reports[0], task, year))
    mask = _eligible(base, task)
    labels = _labels(base, task, mask)
    columns: list[NDArray[np.float64]] = [
        np.asarray(base.loc[mask, "probability"], dtype=np.float64)
    ]
    base_ids = base["sample_id"].to_numpy()
    for report in reports[1:]:
        other = pd.read_parquet(_prediction_path(report, task, year))
        if not np.array_equal(base_ids, other["sample_id"].to_numpy()):
            raise ValueError(f"candidate samples are not aligned for {task}, {year}")
        if not np.array_equal(
            base.loc[:, ["ArrDel15", "Cancelled"]].to_numpy(),
            other.loc[:, ["ArrDel15", "Cancelled"]].to_numpy(),
            equal_nan=True,
        ):
            raise ValueError(f"candidate labels differ for {task}, {year}")
        columns.append(np.asarray(other.loc[mask, "probability"], dtype=np.float64))
    metadata = base.loc[mask, ["sample_id", "FlightDate", "Month"]].reset_index(drop=True)
    matrix = np.column_stack(columns)
    if matrix.shape[1] != len(candidate_names):
        raise AssertionError("candidate probability matrix has wrong width")
    return metadata, labels, matrix


def analyse_ensemble(
    report_paths: tuple[Path, ...],
    *,
    output_path: Path,
    years: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023),
) -> dict[str, Any]:
    """Fit each adaptation only on earlier out-of-time predictions."""

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite ensemble evidence: {output_path}")
    if len(report_paths) < 2:
        raise ValueError("ensemble analysis requires at least two candidate reports")
    if tuple(sorted(years)) != years or len(years) < 3:
        raise ValueError("years must be chronological and contain at least three periods")
    reports = [_verify_report(path) for path in report_paths]
    candidate_names = [str(report["candidate_name"]) for report in reports]
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("candidate names must be unique")

    task_results: dict[str, Any] = {}
    for task in ("delay", "cancellation"):
        data: dict[int, tuple[pd.DataFrame, NDArray[np.int64], NDArray[np.float64]]] = {
            year: _load_aligned_year(
                reports,
                candidate_names,
                task=task,
                year=year,
            )
            for year in years
        }
        records: list[dict[str, Any]] = []
        for year in years:
            _, labels, matrix = data[year]
            for index, candidate in enumerate(candidate_names):
                records.append(
                    _metric_record(
                        method=f"raw::{candidate}",
                        year=year,
                        labels=labels,
                        probabilities=matrix[:, index],
                    )
                )

        adaptation_years = years[1:]
        for target_year in adaptation_years:
            prior_years = tuple(year for year in years if year < target_year)
            prior_labels = np.concatenate([data[year][1] for year in prior_years])
            prior_matrix = np.concatenate([data[year][2] for year in prior_years])
            _, target_labels, target_matrix = data[target_year]

            weights = _fit_logit_blend(prior_matrix, prior_labels)
            records.append(
                _metric_record(
                    method="prospective_logit_blend",
                    year=target_year,
                    labels=target_labels,
                    probabilities=_apply_logit_blend(target_matrix, weights),
                    detail={
                        "fit_years": list(prior_years),
                        "weights": {
                            candidate: float(weight)
                            for candidate, weight in zip(candidate_names, weights, strict=True)
                        },
                    },
                )
            )

            stacker = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            stacker.fit(_logit(prior_matrix), prior_labels)
            stacked = stacker.predict_proba(_logit(target_matrix))[:, 1]
            records.append(
                _metric_record(
                    method="prospective_logit_stacker",
                    year=target_year,
                    labels=target_labels,
                    probabilities=np.asarray(stacked, dtype=np.float64),
                    detail={
                        "fit_years": list(prior_years),
                        "coefficients": {
                            candidate: float(coefficient)
                            for candidate, coefficient in zip(
                                candidate_names, stacker.coef_[0], strict=True
                            )
                        },
                        "intercept": float(stacker.intercept_[0]),
                    },
                )
            )

            for index, candidate in enumerate(candidate_names):
                for calibration_method in ("intercept", "platt", "beta", "isotonic"):
                    calibrator = fit_binary_calibrator(
                        calibration_method,
                        prior_matrix[:, index],
                        prior_labels,
                    )
                    records.append(
                        _metric_record(
                            method=f"expanding_{calibration_method}::{candidate}",
                            year=target_year,
                            labels=target_labels,
                            probabilities=calibrator.predict(target_matrix[:, index]),
                            detail={"fit_years": list(prior_years)},
                        )
                    )
                    last_year = target_year - 1
                    last_calibrator = fit_binary_calibrator(
                        calibration_method,
                        data[last_year][2][:, index],
                        data[last_year][1],
                    )
                    records.append(
                        _metric_record(
                            method=f"last_year_{calibration_method}::{candidate}",
                            year=target_year,
                            labels=target_labels,
                            probabilities=last_calibrator.predict(target_matrix[:, index]),
                            detail={"fit_years": [last_year]},
                        )
                    )

        hindsight: list[dict[str, Any]] = []
        for year in years:
            _, labels, matrix = data[year]
            losses = np.column_stack(
                [
                    -(
                        labels * np.log(_clip(matrix[:, index]))
                        + (1 - labels) * np.log1p(-_clip(matrix[:, index]))
                    )
                    for index in range(matrix.shape[1])
                ]
            )
            chosen = np.argmin(losses, axis=1)
            oracle_probabilities = matrix[np.arange(len(labels)), chosen]
            hindsight.append(
                _metric_record(
                    method="HINDSIGHT_ROW_ORACLE_NOT_DEPLOYABLE",
                    year=year,
                    labels=labels,
                    probabilities=oracle_probabilities,
                )
            )

        aggregate: list[dict[str, Any]] = []
        methods = sorted({str(record["method"]) for record in records})
        for method in methods:
            selected = [
                record
                for record in records
                if record["method"] == method and int(record["year"]) in adaptation_years
            ]
            if len(selected) != len(adaptation_years):
                continue
            aggregate.append(
                {
                    "method": method,
                    "years": list(adaptation_years),
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
        task_results[task] = {
            "records": records,
            "aggregate_2020_2023": aggregate,
            "hindsight_bound": hindsight,
        }

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_DEVELOPMENT_ANALYSIS_NOT_CONFIRMATORY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "candidate_reports": [
            {"path": path.as_posix(), "sha256": sha256_file(path)} for path in report_paths
        ],
        "candidate_names": candidate_names,
        "years": list(years),
        "adaptation_rule": (
            "For target year Y, ensemble/calibration fits use only out-of-time predictions "
            "and labels from listed years strictly before Y. Target features and scores are "
            "not used for fitting."
        ),
        "task_results": task_results,
    }
    result["result_sha256"] = canonical_json_sha256(result)
    write_canonical_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2019, 2020, 2021, 2022, 2023])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyse_ensemble(tuple(args.reports), output_path=args.output, years=tuple(args.years))
    top = {
        task: payload["aggregate_2020_2023"][:5] for task, payload in result["task_results"].items()
    }
    print(json.dumps({"output": args.output.as_posix(), "top_methods": top}, indent=2))


if __name__ == "__main__":
    main()
