"""Independent acceptance checks for strict prior-day feature lookups."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import RECENT_WINDOWS_DAYS
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .recent import BASE_LEVEL_KEYS, SMOOTHING_STRENGTHS

DEFAULT_CHECK_DATES = (
    "2018-06-15",
    "2020-04-15",
    "2024-07-15",
    "2025-06-15",
)


def _validate_self_hash(payload: dict[str, Any], field: str) -> bool:
    declared = str(payload.get(field, ""))
    candidate = payload.copy()
    candidate.pop(field, None)
    return bool(declared) and canonical_json_sha256(candidate) == declared


def _source_window(
    *,
    repository_root: Path,
    source_records: list[dict[str, Any]],
    start: pd.Timestamp,
    stop: pd.Timestamp,
) -> pd.DataFrame:
    columns = [
        "FlightDate",
        "Route",
        "Reporting_Airline",
        "Origin",
        "Dest",
        "ArrDel15",
        "Cancelled",
        "delay_label_observed",
    ]
    pieces: list[pd.DataFrame] = []
    for record in source_records:
        first = pd.Timestamp(record["first_date"])
        last = pd.Timestamp(record["last_date"])
        if last < start or first >= stop:
            continue
        frame = pd.read_parquet(repository_root / str(record["path"]), columns=columns)
        frame["FlightDate"] = pd.to_datetime(frame["FlightDate"], errors="raise").dt.normalize()
        mask = frame["FlightDate"].ge(start) & frame["FlightDate"].lt(stop)
        pieces.append(frame.loc[mask])
    if not pieces:
        raise ValueError(f"no source outcomes found in [{start.date()}, {stop.date()})")
    return pd.concat(pieces, ignore_index=True)


def _outcome_sums(frame: pd.DataFrame) -> dict[str, float]:
    delay_support = float(frame["delay_label_observed"].sum())
    delay_events = float(
        (
            pd.to_numeric(frame["ArrDel15"], errors="coerce").fillna(0.0)
            * frame["delay_label_observed"]
        ).sum()
    )
    return {
        "flight_count": float(len(frame)),
        "cancel_events": float(frame["Cancelled"].sum()),
        "delay_support": delay_support,
        "delay_events": delay_events,
    }


def _assert_close(actual: float, expected: float, *, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=2e-6):
        raise AssertionError(f"{name}: expected {expected}, observed {actual}")


def _check_target_date(
    *,
    date: pd.Timestamp,
    repository_root: Path,
    source_records: list[dict[str, Any]],
    output_paths: dict[tuple[str, int], Path],
) -> list[dict[str, Any]]:
    history = _source_window(
        repository_root=repository_root,
        source_records=source_records,
        start=date - pd.Timedelta(days=max(RECENT_WINDOWS_DAYS)),
        stop=date,
    )
    checks: list[dict[str, Any]] = []
    global_table = pd.read_parquet(output_paths[("global", int(date.year))])
    global_row = global_table.loc[global_table["FlightDate"].eq(date)]
    if len(global_row) != 1:
        raise AssertionError(f"expected one global lookup row for {date.date()}")
    global_target = global_row.iloc[0]

    for window in RECENT_WINDOWS_DAYS:
        window_history = history.loc[
            history["FlightDate"].ge(date - pd.Timedelta(days=window))
        ]
        sums = _outcome_sums(window_history)
        expected_delay = sums["delay_events"] / sums["delay_support"]
        expected_cancel = sums["cancel_events"] / sums["flight_count"]
        expected = {
            f"recent_global_delay_rate_{window}d": expected_delay,
            f"recent_global_cancel_rate_{window}d": expected_cancel,
            f"recent_global_count_log1p_{window}d": math.log1p(sums["flight_count"]),
            f"recent_global_delay_support_log1p_{window}d": math.log1p(
                sums["delay_support"]
            ),
        }
        for name, value in expected.items():
            _assert_close(float(global_target[name]), value, name=f"{date.date()} {name}")
        checks.append(
            {
                "date": str(date.date()),
                "level": "global",
                "window_days": window,
                "history_rows": len(window_history),
            }
        )

    for level in ("route", "airline", "origin", "dest"):
        keys = BASE_LEVEL_KEYS[level]
        table = pd.read_parquet(output_paths[(level, int(date.year))])
        target_rows = table.loc[table["FlightDate"].eq(date)].sort_values(
            list(keys), kind="mergesort"
        )
        if target_rows.empty:
            raise AssertionError(f"no {level} lookup row for {date.date()}")
        target = target_rows.iloc[0]
        key_mask = np.ones(len(history), dtype=bool)
        key_values: dict[str, str] = {}
        for key in keys:
            key_value = str(target[key])
            key_values[key] = key_value
            key_mask &= history[key].astype(str).eq(key_value).to_numpy()
        group_history = history.loc[key_mask]

        for window in RECENT_WINDOWS_DAYS:
            start = date - pd.Timedelta(days=window)
            global_window = history.loc[history["FlightDate"].ge(start)]
            group_window = group_history.loc[group_history["FlightDate"].ge(start)]
            global_sums = _outcome_sums(global_window)
            group_sums = _outcome_sums(group_window)
            global_delay = global_sums["delay_events"] / global_sums["delay_support"]
            global_cancel = global_sums["cancel_events"] / global_sums["flight_count"]
            alpha = SMOOTHING_STRENGTHS[level]
            expected_delay = (
                group_sums["delay_events"] + alpha * global_delay
            ) / (group_sums["delay_support"] + alpha)
            expected_cancel = (
                group_sums["cancel_events"] + alpha * global_cancel
            ) / (group_sums["flight_count"] + alpha)
            expected = {
                f"recent_{level}_delay_rate_{window}d": expected_delay,
                f"recent_{level}_cancel_rate_{window}d": expected_cancel,
                f"recent_{level}_count_log1p_{window}d": math.log1p(
                    group_sums["flight_count"]
                ),
                f"recent_{level}_delay_support_log1p_{window}d": math.log1p(
                    group_sums["delay_support"]
                ),
            }
            for name, value in expected.items():
                _assert_close(float(target[name]), value, name=f"{date.date()} {name}")
            checks.append(
                {
                    "date": str(date.date()),
                    "level": level,
                    "key": key_values,
                    "window_days": window,
                    "history_rows": len(group_window),
                }
            )
    return checks


def validate_recent_features(
    *,
    recent_manifest: Path,
    report_path: Path,
    check_dates: tuple[str, ...] = DEFAULT_CHECK_DATES,
) -> dict[str, Any]:
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite recent validation report: {report_path}")
    payload: dict[str, Any] = json.loads(recent_manifest.read_text(encoding="utf-8"))
    repository_root = recent_manifest.resolve().parent.parent
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    if not _validate_self_hash(payload, "manifest_sha256"):
        failures.append("recent feature manifest self-hash is invalid")
    source_manifest_path = repository_root / str(payload["source_feature_manifest"])
    if sha256_file(source_manifest_path) != payload["source_feature_manifest_sha256"]:
        failures.append("source feature manifest file hash mismatch")
    source_payload: dict[str, Any] = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not _validate_self_hash(source_payload, "manifest_sha256"):
        failures.append("source feature manifest self-hash is invalid")

    output_paths: dict[tuple[str, int], Path] = {}
    total_rows = 0
    for record in payload["outputs"]:
        level = str(record["level"])
        year = int(record["year"])
        path = repository_root / str(record["path"])
        output_paths[(level, year)] = path
        if sha256_file(path) != record["sha256"]:
            failures.append(f"output hash mismatch: {record['path']}")
            continue
        table = pd.read_parquet(path)
        total_rows += len(table)
        keys = [*BASE_LEVEL_KEYS[level], "FlightDate"]
        if len(table) != int(record["rows"]):
            failures.append(f"row count mismatch: {record['path']}")
        if table.duplicated(keys).any():
            failures.append(f"duplicate lookup keys: {record['path']}")
        if not pd.to_datetime(table["FlightDate"]).dt.year.eq(year).all():
            failures.append(f"partition year mismatch: {record['path']}")
        rate_columns = [name for name in table if "_rate_" in name]
        support_columns = [name for name in table if name.endswith(tuple(f"_{w}d" for w in RECENT_WINDOWS_DAYS)) and "_log1p_" in name]
        rate_values = table[rate_columns].to_numpy(dtype=np.float64)
        finite_rates = rate_values[np.isfinite(rate_values)]
        if ((finite_rates < 0.0) | (finite_rates > 1.0)).any():
            failures.append(f"rate outside [0,1]: {record['path']}")
        support_values = table[support_columns].to_numpy(dtype=np.float64)
        finite_support = support_values[np.isfinite(support_values)]
        if (finite_support < 0.0).any():
            failures.append(f"negative log support: {record['path']}")

    if total_rows != int(payload["total_lookup_rows"]):
        failures.append("total lookup row count mismatch")

    if not failures:
        try:
            source_records = list(source_payload["outputs"])
            for value in check_dates:
                checks.extend(
                    _check_target_date(
                        date=pd.Timestamp(value),
                        repository_root=repository_root,
                        source_records=source_records,
                        output_paths=output_paths,
                    )
                )
        except (AssertionError, KeyError, ValueError) as error:
            failures.append(f"independent temporal reconstruction failed: {error}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "recent_manifest": recent_manifest.as_posix(),
        "recent_manifest_sha256": sha256_file(recent_manifest),
        "outputs_verified": len(payload["outputs"]),
        "lookup_rows_verified": total_rows,
        "domain_checks": [
            "unique level/date keys",
            "partition year",
            "rates in [0,1] or documented cold-start null",
            "non-negative log supports",
        ],
        "independent_cutoff_checks": checks,
        "failures": failures,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(report_path, report)
    if failures:
        raise ValueError("recent feature acceptance failed: " + "; ".join(failures))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--check-dates", nargs="+", default=list(DEFAULT_CHECK_DATES))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = validate_recent_features(
        recent_manifest=args.manifest,
        report_path=args.report,
        check_dates=tuple(args.check_dates),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "outputs_verified": report["outputs_verified"],
                "cutoff_checks": len(report["independent_cutoff_checks"]),
                "report": args.report.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
