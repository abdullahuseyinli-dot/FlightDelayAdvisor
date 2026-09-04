"""Acceptance checks for closed-left census flight-number history tables."""

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

from .census_recent import FLIGHT_KEYS, FLIGHT_SMOOTHING_STRENGTH
from .contracts import RECENT_WINDOWS_DAYS
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

DEFAULT_CHECK_DATES = (
    "2018-06-15",
    "2020-04-15",
    "2024-08-15",
    "2025-06-15",
)


def _resolve_evidence(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _verify_self_hash(payload: dict[str, Any], field: str = "manifest_sha256") -> None:
    declared = str(payload.get(field, ""))
    body = {key: value for key, value in payload.items() if key != field}
    if not declared or canonical_json_sha256(body) != declared:
        raise ValueError(f"invalid {field}")


def _source_window(
    *,
    root: Path,
    records: list[dict[str, Any]],
    start: pd.Timestamp,
    stop: pd.Timestamp,
) -> pd.DataFrame:
    columns = [
        "FlightDate",
        "ScheduledFlightId",
        "ArrDel15",
        "Cancelled",
        "delay_label_observed",
    ]
    pieces: list[pd.DataFrame] = []
    for record in records:
        if "first_date" in record and "last_date" in record:
            if pd.Timestamp(record["last_date"]) < start or pd.Timestamp(record["first_date"]) >= stop:
                continue
        path = _resolve_evidence(root, str(record["path"]))
        frame = pd.read_parquet(path, columns=columns)
        frame["FlightDate"] = pd.to_datetime(frame["FlightDate"], errors="raise").dt.normalize()
        mask = frame["FlightDate"].ge(start) & frame["FlightDate"].lt(stop)
        if mask.any():
            pieces.append(frame.loc[mask])
    if not pieces:
        raise ValueError(f"no source rows in [{start.date()}, {stop.date()})")
    return pd.concat(pieces, ignore_index=True)


def _sums(frame: pd.DataFrame) -> dict[str, float]:
    support = float(frame["delay_label_observed"].sum())
    return {
        "flights": float(len(frame)),
        "cancellations": float(frame["Cancelled"].sum()),
        "delay_support": support,
        "delays": float(
            (
                pd.to_numeric(frame["ArrDel15"], errors="coerce").fillna(0.0)
                * frame["delay_label_observed"]
            ).sum()
        ),
    }


def _check_date(
    *,
    date: pd.Timestamp,
    root: Path,
    source_records: list[dict[str, Any]],
    lookup_path: Path,
) -> list[dict[str, Any]]:
    history = _source_window(
        root=root,
        records=source_records,
        start=date - pd.Timedelta(days=max(RECENT_WINDOWS_DAYS)),
        stop=date,
    )
    lookup = pd.read_parquet(lookup_path)
    target_rows = lookup.loc[pd.to_datetime(lookup["FlightDate"]).eq(date)].sort_values(
        list(FLIGHT_KEYS), kind="mergesort"
    )
    if target_rows.empty:
        raise ValueError(f"no flight-number lookup row for {date.date()}")
    target = target_rows.iloc[0]
    identifier = str(target["ScheduledFlightId"])
    flight_history = history.loc[history["ScheduledFlightId"].astype(str).eq(identifier)]
    checks: list[dict[str, Any]] = []
    for window in RECENT_WINDOWS_DAYS:
        start = date - pd.Timedelta(days=window)
        global_values = _sums(history.loc[history["FlightDate"].ge(start)])
        flight_values = _sums(flight_history.loc[flight_history["FlightDate"].ge(start)])
        global_delay = global_values["delays"] / global_values["delay_support"]
        global_cancel = global_values["cancellations"] / global_values["flights"]
        alpha = FLIGHT_SMOOTHING_STRENGTH
        expected = {
            f"recent_flight_delay_rate_{window}d": (
                flight_values["delays"] + alpha * global_delay
            )
            / (flight_values["delay_support"] + alpha),
            f"recent_flight_cancel_rate_{window}d": (
                flight_values["cancellations"] + alpha * global_cancel
            )
            / (flight_values["flights"] + alpha),
            f"recent_flight_count_log1p_{window}d": math.log1p(flight_values["flights"]),
            f"recent_flight_delay_support_log1p_{window}d": math.log1p(
                flight_values["delay_support"]
            ),
        }
        for name, value in expected.items():
            if not math.isclose(float(target[name]), value, rel_tol=1e-5, abs_tol=2e-6):
                raise AssertionError(
                    f"{date.date()} {identifier} {name}: expected {value}, observed {target[name]}"
                )
        checks.append(
            {
                "date": date.date().isoformat(),
                "ScheduledFlightId": identifier,
                "window_days": window,
                "history_rows": int(flight_values["flights"]),
            }
        )
    return checks


def validate_flight_recent_features(
    *,
    manifest_path: Path,
    report_path: Path,
    check_dates: tuple[str, ...] = DEFAULT_CHECK_DATES,
) -> dict[str, Any]:
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite flight-history validation: {report_path}")
    root = manifest_path.resolve().parent.parent
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    checks: list[dict[str, Any]] = []
    try:
        _verify_self_hash(payload)
        source_path = _resolve_evidence(root, str(payload["source_feature_manifest"]))
        if sha256_file(source_path) != payload["source_feature_manifest_sha256"]:
            raise ValueError("source census manifest file hash mismatch")
        source: dict[str, Any] = json.loads(source_path.read_text(encoding="utf-8"))
        _verify_self_hash(source)
        output_paths: dict[int, Path] = {}
        rows = 0
        for record in payload["outputs"]:
            if record["level"] != "flight":
                raise ValueError("flight-history manifest contains a non-flight output")
            year = int(record["year"])
            path = _resolve_evidence(root, str(record["path"]))
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"output hash mismatch: {path}")
            table = pd.read_parquet(path)
            if len(table) != int(record["rows"]):
                raise ValueError(f"output row count mismatch: {path}")
            if table.duplicated([*FLIGHT_KEYS, "FlightDate"]).any():
                raise ValueError(f"duplicate flight/date lookup key: {path}")
            if not pd.to_datetime(table["FlightDate"]).dt.year.eq(year).all():
                raise ValueError(f"lookup partition year mismatch: {path}")
            rates = table.filter(regex=r"_rate_\d+d$").to_numpy(dtype=np.float64)
            supports = table.filter(regex=r"_(?:count|delay_support)_log1p_\d+d$").to_numpy(
                dtype=np.float64
            )
            finite_rates = rates[np.isfinite(rates)]
            finite_supports = supports[np.isfinite(supports)]
            if ((finite_rates < 0.0) | (finite_rates > 1.0)).any():
                raise ValueError(f"rate outside [0,1]: {path}")
            if (finite_supports < 0.0).any():
                raise ValueError(f"negative log support: {path}")
            output_paths[year] = path
            rows += len(table)
        if rows != int(payload["total_lookup_rows"]):
            raise ValueError("total lookup row count mismatch")
        for value in check_dates:
            date = pd.Timestamp(value)
            checks.extend(
                _check_date(
                    date=date,
                    root=root,
                    source_records=list(source["outputs"]),
                    lookup_path=output_paths[int(date.year)],
                )
            )
    except (AssertionError, KeyError, ValueError) as error:
        failures.append(f"{type(error).__name__}: {error}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "flight_recent_manifest": manifest_path.as_posix(),
        "flight_recent_manifest_sha256": sha256_file(manifest_path),
        "outputs_verified": len(payload.get("outputs", [])),
        "lookup_rows_verified": int(payload.get("total_lookup_rows", 0)),
        "independent_cutoff_checks": checks,
        "failures": failures,
        "provenance": capture_provenance(
            (
                Path(__file__),
                Path(__file__).with_name("census_recent.py"),
                Path(__file__).with_name("recent.py"),
            )
        ),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(report_path, report)
    if failures:
        raise ValueError("flight-history acceptance failed: " + "; ".join(failures))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--check-dates", nargs="+", default=list(DEFAULT_CHECK_DATES))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = validate_flight_recent_features(
        manifest_path=args.manifest,
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
