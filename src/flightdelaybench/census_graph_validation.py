"""Acceptance checks for closed-left schedule-graph message features."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .census_graph import _resolve, schedule_graph_messages
from .census_normalization import _verify_json_self_hash
from .contracts import CENSUS_GRAPH_MESSAGE_FEATURES
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

DEFAULT_CHECK_DATES = (
    "2018-06-15",
    "2020-04-15",
    "2024-08-15",
    "2025-06-15",
)


def _level_key(level: str) -> str:
    if level == "origin":
        return "Origin"
    if level == "dest":
        return "Dest"
    raise ValueError(f"unknown graph level: {level}")


def validate_census_graph_features(
    *,
    graph_manifest: Path,
    report_path: Path,
    check_dates: tuple[str, ...] = DEFAULT_CHECK_DATES,
) -> dict[str, Any]:
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite graph validation report: {report_path}")
    root = graph_manifest.resolve().parent.parent
    graph = _verify_json_self_hash(graph_manifest)
    failures: list[str] = []
    output_checks: list[dict[str, Any]] = []
    reconstruction_checks: list[dict[str, Any]] = []
    output_paths: dict[tuple[str, int], Path] = {}
    verified_rows = 0
    for record in graph["outputs"]:
        level = str(record["level"])
        year = int(record["year"])
        path = _resolve(root, str(record["path"]))
        try:
            if sha256_file(path) != record["sha256"]:
                raise ValueError("output SHA-256 mismatch")
            table = pd.read_parquet(path)
            key = _level_key(level)
            if len(table) != int(record["rows"]):
                raise ValueError("output row count mismatch")
            if table.duplicated(["FlightDate", key]).any():
                raise ValueError("duplicate graph lookup key")
            if not pd.to_datetime(table["FlightDate"]).dt.year.eq(year).all():
                raise ValueError("graph output year mismatch")
            missing = sorted(
                {
                    name
                    for name in CENSUS_GRAPH_MESSAGE_FEATURES
                    if name.startswith(f"graph_{level}_")
                }
                - set(table.columns)
            )
            if missing:
                raise ValueError(f"graph output lacks features: {missing}")
            values = table.filter(regex=rf"^graph_{level}_").to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("graph output contains non-finite values")
            rate_columns = [name for name in table if any(token in name for token in ("_mean_", "_max_"))]
            rate_values = table.loc[:, rate_columns].to_numpy(dtype=np.float64)
            if ((rate_values < 0.0) | (rate_values > 1.0)).any():
                raise ValueError("graph mean/max rate outside [0,1]")
            std_values = table.filter(regex=r"_std_\d+d$").to_numpy(dtype=np.float64)
            if (std_values < 0.0).any():
                raise ValueError("negative graph dispersion")
            output_paths[(level, year)] = path
            verified_rows += len(table)
            output_checks.append({"level": level, "year": year, "rows": len(table), "status": "PASS"})
        except (KeyError, ValueError) as error:
            failures.append(f"{level}/{year}: {type(error).__name__}: {error}")
            output_checks.append({"level": level, "year": year, "status": "FAIL"})
    if verified_rows != int(graph["total_lookup_rows"]):
        failures.append("total graph lookup row count mismatch")

    try:
        census_path = _resolve(root, str(graph["census_manifest"]))
        recent_path = _resolve(root, str(graph["recent_manifest"]))
        if sha256_file(census_path) != graph["census_manifest_sha256"]:
            raise ValueError("bound census manifest hash mismatch")
        if sha256_file(recent_path) != graph["recent_manifest_sha256"]:
            raise ValueError("bound recent manifest hash mismatch")
        census = _verify_json_self_hash(census_path)
        recent = _verify_json_self_hash(recent_path)
        if census["manifest_sha256"] != graph["census_manifest_self_hash"]:
            raise ValueError("bound census manifest self-hash mismatch")
        if recent["manifest_sha256"] != graph["recent_manifest_self_hash"]:
            raise ValueError("bound recent manifest self-hash mismatch")
        census_periods = {
            (int(record["year"]), int(record["month"])): _resolve(root, str(record["path"]))
            for record in census["outputs"]
        }
        recent_outputs = {
            (str(record["level"]), int(record["year"])): _resolve(root, str(record["path"]))
            for record in recent["outputs"]
        }
        for value in check_dates:
            date = pd.Timestamp(value)
            year, month = int(date.year), int(date.month)
            schedule = pd.read_parquet(
                census_periods[(year, month)], columns=["FlightDate", "Origin", "Dest"]
            )
            reconstructed = schedule_graph_messages(
                schedule,
                global_lookup=pd.read_parquet(recent_outputs[("global", year)]),
                origin_lookup=pd.read_parquet(recent_outputs[("origin", year)]),
                dest_lookup=pd.read_parquet(recent_outputs[("dest", year)]),
            )
            for level, expected in zip(("origin", "dest"), reconstructed, strict=True):
                key = _level_key(level)
                expected = expected.loc[pd.to_datetime(expected["FlightDate"]).eq(date)].sort_values(key)
                stored = pd.read_parquet(output_paths[(level, year)])
                stored = stored.loc[pd.to_datetime(stored["FlightDate"]).eq(date)].sort_values(key)
                pd.testing.assert_frame_equal(
                    stored.reset_index(drop=True),
                    expected.reset_index(drop=True),
                    check_dtype=False,
                    check_exact=False,
                    rtol=1e-6,
                    atol=1e-6,
                )
                reconstruction_checks.append(
                    {"date": date.date().isoformat(), "level": level, "rows": len(expected), "status": "PASS"}
                )
    except (AssertionError, KeyError, ValueError) as error:
        failures.append(f"independent reconstruction: {type(error).__name__}: {error}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "graph_manifest": graph_manifest.as_posix(),
        "graph_manifest_sha256": sha256_file(graph_manifest),
        "graph_manifest_self_hash": graph["manifest_sha256"],
        "outputs": output_checks,
        "lookup_rows_verified": verified_rows,
        "independent_reconstructions": reconstruction_checks,
        "failures": failures,
        "checks": [
            "all output hashes and row counts",
            "unique airport/date keys",
            "finite registered graph features",
            "bounded rate summaries and non-negative dispersion",
            "independent schedule-plus-prior reconstruction",
        ],
        "provenance": capture_provenance(
            (Path(__file__), Path(__file__).with_name("census_graph.py"))
        ),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(report_path, report)
    if failures:
        raise ValueError("schedule-graph acceptance failed: " + "; ".join(failures))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--check-dates", nargs="+", default=list(DEFAULT_CHECK_DATES))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = validate_census_graph_features(
        graph_manifest=args.manifest,
        report_path=args.report,
        check_dates=tuple(args.check_dates),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "outputs": len(report["outputs"]),
                "reconstructions": len(report["independent_reconstructions"]),
                "report": args.report.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
