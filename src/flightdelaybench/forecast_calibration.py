"""Select a robust probability calibrator using only the 2024 selection quarter."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .bootstrap import paired_cluster_mean_difference
from .calibration import CalibrationMethod, fit_binary_calibrator
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .metrics import binary_metrics, clip_probabilities
from .provenance import capture_provenance
from .rolling import _atomic_joblib

CALIBRATION_METHODS: tuple[CalibrationMethod, ...] = (
    "identity",
    "intercept",
    "platt",
    "beta",
    "isotonic",
)

# Expanding, forward-only folds.  No observation is calibrated using a label
# from its own date or a later date.
CALIBRATION_FOLDS = (
    ("2024-10-01", "2024-10-31", "2024-11-01", "2024-11-15"),
    ("2024-10-01", "2024-11-15", "2024-11-16", "2024-11-30"),
    ("2024-10-01", "2024-11-30", "2024-12-01", "2024-12-31"),
)


def _verify_selection_report(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("report_sha256")
    body = {key: value for key, value in payload.items() if key != "report_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"forecast-adaptation report self-hash failed: {path}")
    if payload.get("status") != "COMPLETE_2024_SELECTION_NOT_CONFIRMATORY":
        raise ValueError("calibration requires a completed 2024 selection report")
    if payload.get("outcomes_accessed", {}).get("maximum_calendar_date") != "2024-12-31":
        raise ValueError("selection report has an unexpected outcome boundary")
    return payload


def _prediction_artifact(report: dict[str, Any], task: str) -> tuple[Path, dict[str, Any]]:
    matches = [
        record
        for record in report["artifacts"]
        if record.get("kind") == "predictions" and record.get("task") == task
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one 2024 prediction artifact for {task}")
    record = matches[0]
    path = Path(record["path"])
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError(f"selection prediction checksum failed: {path}")
    return path, record


def _eligible(frame: pd.DataFrame, task: str) -> tuple[pd.DataFrame, NDArray[np.int64]]:
    if task == "delay":
        mask = frame["Cancelled"].eq(0) & frame["delay_label_observed"].eq(1)
        label_column = "ArrDel15"
    elif task == "cancellation":
        mask = frame["Cancelled"].isin([0, 1])
        label_column = "Cancelled"
    else:
        raise ValueError(f"unsupported calibration task: {task}")
    selected = frame.loc[mask].reset_index(drop=True)
    labels = np.asarray(selected[label_column], dtype=np.int64)
    if selected.empty or np.unique(labels).size != 2:
        raise ValueError(f"calibration population for {task} lacks both classes")
    return selected, labels


def _forward_calibration_folds(
    dates: pd.Series,
) -> list[tuple[NDArray[np.bool_], NDArray[np.bool_]]]:
    normalized = pd.to_datetime(dates, errors="raise").dt.normalize()
    folds: list[tuple[NDArray[np.bool_], NDArray[np.bool_]]] = []
    validation_dates: set[pd.Timestamp] = set()
    for train_start, train_end, valid_start, valid_end in CALIBRATION_FOLDS:
        train = np.asarray(
            normalized.between(pd.Timestamp(train_start), pd.Timestamp(train_end)),
            dtype=np.bool_,
        )
        valid = np.asarray(
            normalized.between(pd.Timestamp(valid_start), pd.Timestamp(valid_end)),
            dtype=np.bool_,
        )
        if not train.any() or not valid.any():
            raise ValueError("a forward calibration fold is empty")
        if normalized.loc[train].max() >= normalized.loc[valid].min():
            raise AssertionError("calibration fold is not strictly forward in time")
        current_dates = set(normalized.loc[valid])
        if validation_dates & current_dates:
            raise AssertionError("calibration validation dates overlap")
        validation_dates.update(current_dates)
        folds.append((train, valid))
    return folds


def _losses(labels: NDArray[np.int64], probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    values = clip_probabilities(probabilities)
    return np.asarray(
        -(labels * np.log(values) + (1 - labels) * np.log1p(-values)),
        dtype=np.float64,
    )


def _interval_dict(interval: Any) -> dict[str, Any]:
    return {
        "estimate": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
        "confidence": interval.confidence,
        "clusters": interval.clusters,
        "repetitions": interval.repetitions,
        "seed": interval.seed,
    }


def _select_method(records: list[dict[str, Any]]) -> CalibrationMethod:
    order: dict[str, int] = {
        method: index for index, method in enumerate(CALIBRATION_METHODS)
    }
    selected = min(
        records,
        key=lambda record: (
            float(record["pooled_forward_metrics"]["log_loss"]),
            order[str(record["method"])],
        ),
    )
    return cast(CalibrationMethod, str(selected["method"]))


def select_forecast_calibration(
    selection_report_path: Path,
    *,
    run_dir: Path,
    output_path: Path,
    bootstrap_repetitions: int = 2_000,
    seed: int = 20260903,
) -> dict[str, Any]:
    """Cross-fit calibration on Q4 2024, then freeze it for a later-year audit."""

    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse calibration run directory: {run_dir}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite calibration report: {output_path}")
    report = _verify_selection_report(selection_report_path)
    if bootstrap_repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions are required")
    run_dir.mkdir(parents=True)

    task_results: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for task in ("delay", "cancellation"):
        prediction_path, prediction_record = _prediction_artifact(report, task)
        frame, labels = _eligible(pd.read_parquet(prediction_path), task)
        raw = np.asarray(frame["prob_forecast_residual"], dtype=np.float64)
        folds = _forward_calibration_folds(frame["FlightDate"])
        method_records: list[dict[str, Any]] = []
        pooled: dict[str, tuple[list[NDArray[np.int64]], list[NDArray[np.float64]], list[np.ndarray]]] = {
            method: ([], [], []) for method in CALIBRATION_METHODS
        }
        for fold_index, (train_mask, valid_mask) in enumerate(folds, start=1):
            train_labels = labels[train_mask]
            valid_labels = labels[valid_mask]
            if np.unique(train_labels).size != 2 or np.unique(valid_labels).size != 2:
                raise ValueError(f"calibration fold {fold_index} for {task} lacks both classes")
            for method in CALIBRATION_METHODS:
                calibrator = fit_binary_calibrator(method, raw[train_mask], train_labels)
                calibrated = calibrator.predict(raw[valid_mask])
                pooled_labels, pooled_probabilities, pooled_dates = pooled[method]
                pooled_labels.append(valid_labels)
                pooled_probabilities.append(calibrated)
                pooled_dates.append(frame.loc[valid_mask, "FlightDate"].to_numpy())

        for method in CALIBRATION_METHODS:
            pooled_labels, pooled_probabilities, pooled_dates = pooled[method]
            method_records.append(
                {
                    "method": method,
                    "folds": [
                        {
                            "fold": index,
                            "train_dates": list(CALIBRATION_FOLDS[index - 1][:2]),
                            "validation_dates": list(CALIBRATION_FOLDS[index - 1][2:]),
                            "training_rows": int(train_mask.sum()),
                            "validation_rows": len(fold_labels),
                            "metrics": binary_metrics(fold_labels, fold_probabilities).as_dict(),
                        }
                        for index, ((train_mask, _), fold_labels, fold_probabilities) in enumerate(
                            zip(folds, pooled_labels, pooled_probabilities, strict=True), start=1
                        )
                    ],
                    "pooled_forward_metrics": binary_metrics(
                        np.concatenate(pooled_labels),
                        np.concatenate(pooled_probabilities),
                    ).as_dict(),
                }
            )
        selected_method = _select_method(method_records)
        selected_record = next(
            record for record in method_records if record["method"] == selected_method
        )
        identity_record = next(record for record in method_records if record["method"] == "identity")
        selected_labels = np.concatenate(pooled[selected_method][0])
        selected_probabilities = np.concatenate(pooled[selected_method][1])
        selected_dates = np.concatenate(pooled[selected_method][2])
        identity_probabilities = np.concatenate(pooled["identity"][1])
        interval = paired_cluster_mean_difference(
            _losses(selected_labels, selected_probabilities),
            _losses(selected_labels, identity_probabilities),
            selected_dates,
            repetitions=bootstrap_repetitions,
            seed=seed + (0 if task == "delay" else 100),
        )
        final_calibrator = fit_binary_calibrator(selected_method, raw, labels)
        calibrated_full = final_calibrator.predict(raw)
        artifact = _atomic_joblib(
            final_calibrator,
            run_dir / "models" / f"{task}_{selected_method}_q4_2024.joblib",
        )
        artifact_record = {
            "kind": "calibrator",
            "task": task,
            "method": selected_method,
            **artifact,
        }
        artifacts.append(artifact_record)
        task_results.append(
            {
                "task": task,
                "prediction_artifact": {
                    "path": prediction_path.as_posix(),
                    "sha256": prediction_record["sha256"],
                },
                "eligible_rows": len(frame),
                "methods": method_records,
                "selection_rule": "minimum pooled forward-fold log loss; fixed simplicity order breaks exact ties",
                "selected_method": selected_method,
                "selected_pooled_forward_metrics": selected_record["pooled_forward_metrics"],
                "identity_pooled_forward_metrics": identity_record["pooled_forward_metrics"],
                "paired_selected_minus_identity_log_loss": _interval_dict(interval),
                "full_q4_refit_descriptive_metrics": binary_metrics(labels, calibrated_full).as_dict(),
                "calibrator_artifact": artifact_record,
            }
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_2024_CALIBRATION_SELECTION_NOT_CONFIRMATORY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "forward_cross_fitted_probability_calibration",
        "selection_report": selection_report_path.as_posix(),
        "selection_report_sha256": sha256_file(selection_report_path),
        "selection_report_self_hash": report["report_sha256"],
        "forecast_candidate": "forecast_residual",
        "folds": [list(fold) for fold in CALIBRATION_FOLDS],
        "candidate_methods": list(CALIBRATION_METHODS),
        "task_results": task_results,
        "artifacts": artifacts,
        "bootstrap_repetitions": bootstrap_repetitions,
        "seed": seed,
        "outcomes_accessed": {"maximum_calendar_date": "2024-12-31"},
        "provenance": capture_provenance((Path(__file__), Path(__file__).with_name("calibration.py"))),
        "claim_limit": (
            "Calibration family selection and fitting use the 2024 selection quarter. "
            "Performance must be assessed on the separately labelled 2025 retrospective audit; "
            "the audit is not a never-seen confirmation because 2025 was present in the legacy repository."
        ),
    }
    result["report_sha256"] = canonical_json_sha256(result)
    write_canonical_json(output_path, result)
    write_canonical_json(run_dir / "run_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection_report", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = select_forecast_calibration(
        args.selection_report,
        run_dir=args.run_dir,
        output_path=args.output,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "selected": {
                    item["task"]: item["selected_method"] for item in result["task_results"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
