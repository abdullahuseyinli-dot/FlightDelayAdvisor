"""Build a create-only top-airport BTS census track with rich schedule fields."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from .bts import materialize_archive
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance


def _verify_json_self_hash(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("manifest_sha256")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"manifest self-hash failed: {path}")
    return payload


def _resolve_evidence_path(value: str, repository_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def _summarize_output(
    path: Path,
    *,
    year: int,
    month: int,
    raw_sha256: str,
) -> dict[str, Any]:
    frame = pd.read_parquet(
        path,
        columns=[
            "FlightDate",
            "ArrDel15",
            "Cancelled",
            "Diverted",
            "delay_label_observed",
            "joint_label_observed",
        ],
    )
    dates = pd.to_datetime(frame["FlightDate"], errors="raise")
    cancellations = int(frame["Cancelled"].fillna(0).sum())
    diversions = int(frame["Diverted"].fillna(0).sum())
    delays = int(frame["ArrDel15"].fillna(0).sum())
    delay_support = int(frame["delay_label_observed"].sum())
    return {
        "year": year,
        "month": month,
        "path": path.resolve().as_posix(),
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "raw_sha256": raw_sha256,
        "first_date": dates.min().date().isoformat(),
        "last_date": dates.max().date().isoformat(),
        "cancellations": cancellations,
        "diversions": diversions,
        "delays": delays,
        "delay_support": delay_support,
        "joint_label_ineligible": int(frame["joint_label_observed"].eq(0).sum()),
        "cancellation_rate": cancellations / len(frame),
        "diversion_rate": diversions / len(frame),
        "delay_rate_observed": delays / max(1, delay_support),
    }


def normalize_census_track(
    *,
    raw_manifest_paths: tuple[Path, ...],
    airport_config: Path,
    output_dir: Path,
    manifest_path: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Normalize complete raw years while retaining diverted rows for schedule counts."""

    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite census manifest: {manifest_path}")
    if output_dir.exists() and not resume:
        raise FileExistsError(f"census output exists; pass --resume to verify it: {output_dir}")
    if not raw_manifest_paths:
        raise ValueError("at least one raw manifest is required")
    repository_root = Path(__file__).resolve().parents[2]
    airports_payload: dict[str, Any] = json.loads(airport_config.read_text(encoding="utf-8"))
    airports = tuple(str(code) for code in airports_payload.get("airports", []))
    if len(airports) != 100 or len(set(airports)) != 100:
        raise ValueError("the frozen census airport cohort must contain 100 unique codes")
    source_manifest = repository_root / str(airports_payload["source_feature_manifest"])
    if sha256_file(source_manifest) != airports_payload["source_feature_manifest_sha256"]:
        raise ValueError("the airport cohort source-manifest hash no longer matches")

    raw_sources: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    seen_periods: set[tuple[int, int]] = set()
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    for raw_manifest_path in raw_manifest_paths:
        raw = _verify_json_self_hash(raw_manifest_path)
        raw_records = list(raw.get("records", []))
        years = sorted({int(record["year"]) for record in raw_records})
        if len(years) != 1 or len(raw_records) != 12:
            raise ValueError(f"raw manifest must contain one complete 12-month year: {raw_manifest_path}")
        year = years[0]
        if sorted(int(record["month"]) for record in raw_records) != list(range(1, 13)):
            raise ValueError(f"raw manifest months are incomplete: {raw_manifest_path}")
        raw_sources.append(
            {
                "path": raw_manifest_path.as_posix(),
                "sha256": sha256_file(raw_manifest_path),
                "self_hash": raw["manifest_sha256"],
                "year": year,
            }
        )
        for raw_record in sorted(raw_records, key=lambda item: int(item["month"])):
            month = int(raw_record["month"])
            period = (year, month)
            if period in seen_periods:
                raise ValueError(f"duplicate raw census period: {period}")
            seen_periods.add(period)
            raw_path = _resolve_evidence_path(str(raw_record["local_path"]), repository_root)
            if sha256_file(raw_path) != raw_record["sha256"]:
                raise ValueError(f"raw census archive hash mismatch: {raw_path}")
            output_path = output_dir / f"year={year}" / f"month={month:02d}.parquet"
            if output_path.exists():
                if not resume:
                    raise FileExistsError(f"normalized census output exists: {output_path}")
            else:
                materialize_archive(
                    raw_path,
                    output_path,
                    year=year,
                    month=month,
                    allowed_airports=airports,
                    retain_diverted=True,
                )
            records.append(
                _summarize_output(
                    output_path,
                    year=year,
                    month=month,
                    raw_sha256=str(raw_record["sha256"]),
                )
            )
            print(
                f"normalized census {year}-{month:02d}: {records[-1]['rows']} rows",
                flush=True,
            )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_TOP100_SCHEDULE_COHORT_WITH_DIVERTED_ROWS_RETAINED",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "feature_variant": "official_bts_top100_census_rich_schedule",
        "airport_config": airport_config.as_posix(),
        "airport_config_sha256": sha256_file(airport_config),
        "airports": list(airports),
        "raw_manifests": raw_sources,
        "outputs": records,
        "years": sorted({int(record["year"]) for record in records}),
        "total_rows": sum(int(record["rows"]) for record in records),
        "target_policy": {
            "schedule_context": "all retained rows, including later diversions",
            "cancellation": "Cancelled in {0,1}; diverted non-cancellations remain eligible",
            "delay": "non-cancelled, non-diverted rows with observed ArrDel15",
            "joint": "cancelled rows or eligible delay rows; diversions are ineligible",
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": version("pandas"),
            "pyarrow": version("pyarrow"),
        },
        "provenance": capture_provenance(
            (Path(__file__), Path(__file__).with_name("bts.py"))
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--airport-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = normalize_census_track(
        raw_manifest_paths=tuple(args.raw_manifests),
        airport_config=args.airport_config,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "manifest": args.manifest.as_posix(),
                "years": result["years"],
                "total_rows": result["total_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
