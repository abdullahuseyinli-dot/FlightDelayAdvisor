"""Mechanism, reliability, and operational-subgroup diagnostics for a locked audit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .forecast_audit import _paired_probability_intervals
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .metrics import binary_metrics, clip_probabilities
from .provenance import capture_provenance

WEATHER_VARIABLES = ("prcp_sum", "gust_max", "cape_max")
WEATHER_STRATA = ("typical", "adverse", "extreme", "missing")


def _verify_report(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("report_sha256")
    body = {key: value for key, value in payload.items() if key != "report_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"audit report self-hash failed: {path}")
    if payload.get("status") != "COMPLETE_2025_RETROSPECTIVE_AUDIT_NOT_CONFIRMATORY":
        raise ValueError("diagnostics require a completed 2025 retrospective audit")
    return payload


def _verify_manifest(path: Path, expected_hash: str) -> dict[str, Any]:
    if sha256_file(path) != expected_hash:
        raise ValueError(f"forecast manifest no longer matches audit report: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("manifest_sha256")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"forecast manifest self-hash failed: {path}")
    return payload


def _forecast_record(manifest: dict[str, Any], year: int) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = manifest["outputs"]
    matches = [record for record in outputs if int(record["year"]) == year]
    if len(matches) != 1:
        raise ValueError(f"expected one daily forecast artifact for {year}")
    record = matches[0]
    path = Path(record["path"])
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError(f"daily forecast checksum failed: {path}")
    return record


def _prediction_records(report: dict[str, Any], task: str) -> list[dict[str, Any]]:
    records = sorted(
        (
            record
            for record in report["artifacts"]
            if record.get("kind") == "predictions" and record.get("task") == task
        ),
        key=lambda record: int(record["month"]),
    )
    if [int(record["month"]) for record in records] != list(range(1, 13)):
        raise ValueError(f"audit report lacks twelve monthly prediction artifacts for {task}")
    for record in records:
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"audit prediction checksum failed: {path}")
    return records


def _weather_thresholds(history: pd.DataFrame) -> dict[str, dict[str, float]]:
    complete = history.loc[history["forecast24_missing"].eq(0)]
    if complete.empty:
        raise ValueError("weather-threshold history has no complete forecast days")
    thresholds: dict[str, dict[str, float]] = {}
    for variable in WEATHER_VARIABLES:
        values = pd.to_numeric(complete[f"forecast24_{variable}"], errors="raise")
        thresholds[variable] = {
            "q90": float(values.quantile(0.90)),
            "q99": float(values.quantile(0.99)),
        }
    return thresholds


def _attach_weather(frame: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Airport",
        "FlightDate",
        "forecast24_missing",
        *(f"forecast24_{variable}" for variable in WEATHER_VARIABLES),
    ]
    lookup = forecasts.loc[:, columns].copy()
    lookup["FlightDate"] = pd.to_datetime(lookup["FlightDate"], errors="raise")
    if lookup.duplicated(["Airport", "FlightDate"]).any():
        raise ValueError("daily forecast table has duplicate airport dates")
    output = frame.copy()
    output["FlightDate"] = pd.to_datetime(output["FlightDate"], errors="raise")
    for side, airport in (("origin", "Origin"), ("dest", "Dest")):
        renamed = lookup.rename(
            columns={
                "Airport": airport,
                **{
                    name: f"{side}_{name}"
                    for name in columns
                    if name not in {"Airport", "FlightDate"}
                },
            }
        )
        output = output.merge(
            renamed,
            on=[airport, "FlightDate"],
            how="left",
            sort=False,
            validate="many_to_one",
        )
    return output


def _weather_strata(
    frame: pd.DataFrame, thresholds: dict[str, dict[str, float]]
) -> pd.Series:
    missing = frame["origin_forecast24_missing"].ne(0) | frame[
        "dest_forecast24_missing"
    ].ne(0)
    extreme = np.zeros(len(frame), dtype=np.bool_)
    adverse = np.zeros(len(frame), dtype=np.bool_)
    for variable in WEATHER_VARIABLES:
        endpoint = np.maximum(
            pd.to_numeric(frame[f"origin_forecast24_{variable}"], errors="coerce"),
            pd.to_numeric(frame[f"dest_forecast24_{variable}"], errors="coerce"),
        )
        extreme |= np.asarray(endpoint.ge(thresholds[variable]["q99"]), dtype=np.bool_)
        adverse |= np.asarray(endpoint.ge(thresholds[variable]["q90"]), dtype=np.bool_)
        missing |= endpoint.isna()
    strata = np.full(len(frame), "typical", dtype=object)
    strata[adverse] = "adverse"
    strata[extreme] = "extreme"
    strata[np.asarray(missing, dtype=np.bool_)] = "missing"
    return pd.Series(strata, index=frame.index, dtype="string")


def _losses(labels: NDArray[np.int64], probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    values = clip_probabilities(probabilities)
    return np.asarray(
        -(labels * np.log(values) + (1 - labels) * np.log1p(-values)),
        dtype=np.float64,
    )


def _group_score_sums(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    labels = np.asarray(frame["label"], dtype=np.int64)
    baseline = np.asarray(frame["prob_baseline"], dtype=np.float64)
    forecast = np.asarray(frame["prob_forecast_residual"], dtype=np.float64)
    values = pd.DataFrame(
        {
            "group": frame[group].astype("string"),
            "n": 1,
            "positives": labels,
            "baseline_log": _losses(labels, baseline),
            "forecast_log": _losses(labels, forecast),
            "baseline_brier": np.square(baseline - labels),
            "forecast_brier": np.square(forecast - labels),
        }
    )
    return (
        values.groupby("group", sort=True, observed=True)
        .agg(
            n=("n", "sum"),
            positives=("positives", "sum"),
            baseline_log=("baseline_log", "sum"),
            forecast_log=("forecast_log", "sum"),
            baseline_brier=("baseline_brier", "sum"),
            forecast_brier=("forecast_brier", "sum"),
        )
        .reset_index()
    )


def _combine_group_sums(pieces: list[pd.DataFrame], group_name: str) -> list[dict[str, Any]]:
    combined = (
        pd.concat(pieces, ignore_index=True)
        .groupby("group", sort=True, observed=True)
        .sum(numeric_only=True)
        .reset_index()
    )
    records: list[dict[str, Any]] = []
    for row in combined.itertuples(index=False):
        n = int(row.n)
        positives = int(row.positives)
        records.append(
            {
                group_name: str(row.group),
                "n": n,
                "positives": positives,
                "prevalence": positives / n,
                "baseline_log_loss": float(row.baseline_log / n),
                "forecast_log_loss": float(row.forecast_log / n),
                "forecast_minus_baseline_log_loss": float(
                    (row.forecast_log - row.baseline_log) / n
                ),
                "baseline_brier": float(row.baseline_brier / n),
                "forecast_brier": float(row.forecast_brier / n),
                "forecast_minus_baseline_brier": float(
                    (row.forecast_brier - row.baseline_brier) / n
                ),
                "stable_summary_eligible": n >= 10_000 and positives >= 50,
            }
        )
    return records


def _reliability_bins(
    labels: NDArray[np.int64], probabilities: NDArray[np.float64], bins: int = 20
) -> list[dict[str, Any]]:
    order = np.argsort(probabilities, kind="stable")
    records: list[dict[str, Any]] = []
    for index, selected in enumerate(np.array_split(order, bins), start=1):
        records.append(
            {
                "bin": index,
                "n": len(selected),
                "probability_min": float(probabilities[selected].min()),
                "probability_max": float(probabilities[selected].max()),
                "probability_mean": float(probabilities[selected].mean()),
                "observed_rate": float(labels[selected].mean()),
            }
        )
    return records


def analyse_forecast_audit(
    audit_report_path: Path,
    *,
    output_path: Path,
    bootstrap_repetitions: int = 2_000,
    seed: int = 20260903,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite forecast diagnostics: {output_path}")
    report = _verify_report(audit_report_path)
    forecast_manifest_path = Path(report["forecast_manifest"])
    forecast_manifest = _verify_manifest(
        forecast_manifest_path,
        str(report["forecast_manifest_sha256"]),
    )
    history_record = _forecast_record(forecast_manifest, 2024)
    audit_record = _forecast_record(forecast_manifest, 2025)
    history_weather = pd.read_parquet(Path(history_record["path"]))
    audit_weather = pd.read_parquet(Path(audit_record["path"]))
    thresholds = _weather_thresholds(history_weather)

    task_results: list[dict[str, Any]] = []
    for task_index, task in enumerate(("delay", "cancellation")):
        weather_store: dict[str, dict[str, list[np.ndarray]]] = {
            stratum: {"labels": [], "baseline": [], "forecast": [], "dates": []}
            for stratum in WEATHER_STRATA
        }
        global_store: dict[str, list[np.ndarray]] = {
            "labels": [],
            "forecast": [],
        }
        group_pieces: dict[str, list[pd.DataFrame]] = {
            "Reporting_Airline": [],
            "Origin": [],
            "Dest": [],
        }
        for record in _prediction_records(report, task):
            frame = pd.read_parquet(Path(record["path"]))
            frame = _attach_weather(frame, audit_weather)
            frame["weather_stratum"] = _weather_strata(frame, thresholds)
            labels = np.asarray(frame["label"], dtype=np.int64)
            baseline = np.asarray(frame["prob_baseline"], dtype=np.float64)
            forecast = np.asarray(frame["prob_forecast_residual"], dtype=np.float64)
            global_store["labels"].append(labels)
            global_store["forecast"].append(forecast)
            for stratum in WEATHER_STRATA:
                mask = np.asarray(frame["weather_stratum"].eq(stratum), dtype=np.bool_)
                if not mask.any():
                    continue
                weather_store[stratum]["labels"].append(labels[mask])
                weather_store[stratum]["baseline"].append(baseline[mask])
                weather_store[stratum]["forecast"].append(forecast[mask])
                weather_store[stratum]["dates"].append(frame.loc[mask, "FlightDate"].to_numpy())
            for group in group_pieces:
                group_pieces[group].append(_group_score_sums(frame, group))

        strata_results: list[dict[str, Any]] = []
        for stratum_index, stratum in enumerate(WEATHER_STRATA):
            stored = weather_store[stratum]
            if not stored["labels"]:
                strata_results.append({"stratum": stratum, "n": 0})
                continue
            labels = np.concatenate(stored["labels"]).astype(np.int64)
            baseline = np.concatenate(stored["baseline"]).astype(np.float64)
            forecast = np.concatenate(stored["forecast"]).astype(np.float64)
            dates = np.concatenate(stored["dates"])
            intervals = _paired_probability_intervals(
                labels,
                forecast,
                baseline,
                dates,
                repetitions=bootstrap_repetitions,
                seed=seed + task_index * 100 + stratum_index * 10,
            )
            strata_results.append(
                {
                    "stratum": stratum,
                    "n": len(labels),
                    "prevalence": float(labels.mean()),
                    "baseline_metrics": binary_metrics(labels, baseline).as_dict(),
                    "forecast_metrics": binary_metrics(labels, forecast).as_dict(),
                    "paired_forecast_minus_baseline": intervals,
                }
            )
        global_labels = np.concatenate(global_store["labels"]).astype(np.int64)
        global_forecast = np.concatenate(global_store["forecast"]).astype(np.float64)
        task_results.append(
            {
                "task": task,
                "weather_strata": strata_results,
                "reliability_bins": _reliability_bins(global_labels, global_forecast),
                "operational_subgroups": {
                    "airline": _combine_group_sums(
                        group_pieces["Reporting_Airline"], "Reporting_Airline"
                    ),
                    "origin": _combine_group_sums(group_pieces["Origin"], "Origin"),
                    "destination": _combine_group_sums(group_pieces["Dest"], "Dest"),
                },
            }
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_2025_RETROSPECTIVE_DIAGNOSTICS_NOT_CONFIRMATORY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "audit_report": audit_report_path.as_posix(),
        "audit_report_sha256": sha256_file(audit_report_path),
        "audit_report_self_hash": report["report_sha256"],
        "weather_threshold_reference": {
            "year": 2024,
            "definition": (
                "Airport-day q90/q99 thresholds computed without outcomes; a flight uses the "
                "maximum of its origin and destination forecast. Extreme overrides adverse."
            ),
            "variables": list(WEATHER_VARIABLES),
            "thresholds": thresholds,
            "artifact": history_record,
        },
        "audit_weather_artifact": audit_record,
        "minimum_operational_subgroup_rows": 10_000,
        "minimum_operational_subgroup_positives": 50,
        "bootstrap_repetitions": bootstrap_repetitions,
        "seed": seed,
        "task_results": task_results,
        "provenance": capture_provenance((Path(__file__),)),
        "claim_limit": (
            "Weather strata and carrier/airport results are descriptive heterogeneity analyses. "
            "They are not causal effects, demographic fairness audits, or independently powered "
            "multiple-comparison claims."
        ),
    }
    result["report_sha256"] = canonical_json_sha256(result)
    write_canonical_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = analyse_forecast_audit(
        args.audit_report,
        output_path=args.output,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "weather_strata": {
                    task["task"]: [
                        {
                            "stratum": item["stratum"],
                            "n": item["n"],
                            "log_loss_difference": item.get(
                                "paired_forecast_minus_baseline", {}
                            )
                            .get("log_loss", {})
                            .get("estimate"),
                        }
                        for item in task["weather_strata"]
                    ]
                    for task in result["task_results"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
