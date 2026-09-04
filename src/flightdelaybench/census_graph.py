"""Closed-left schedule-graph message passing for the official census track."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .census_normalization import _verify_json_self_hash
from .contracts import CENSUS_GRAPH_MESSAGE_FEATURES, RECENT_WINDOWS_DAYS
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

GRAPH_METHOD_NAME = "Closed-Left Schedule Graph Message Passing"
GRAPH_METHOD_ACRONYM = "CL-SGMP"


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _atomic_graph_table(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite graph evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated graph partial exists: {partial}")
    frame.to_parquet(partial, index=False, compression="zstd", row_group_size=100_000)
    partial.replace(path)
    return {
        "path": path.as_posix(),
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "first_date": pd.to_datetime(frame["FlightDate"]).min().date().isoformat(),
        "last_date": pd.to_datetime(frame["FlightDate"]).max().date().isoformat(),
    }


def _lookup_paths(
    recent: dict[str, Any],
    *,
    root: Path,
    recent_dir: Path,
) -> tuple[dict[tuple[str, int], Path], list[dict[str, Any]]]:
    paths: dict[tuple[str, int], Path] = {}
    verified: list[dict[str, Any]] = []
    requested_root = recent_dir.resolve()
    for record in recent["outputs"]:
        level = str(record["level"])
        if level not in {"global", "origin", "dest"}:
            continue
        path = _resolve(root, str(record["path"]))
        if not path.resolve().is_relative_to(requested_root):
            raise ValueError(f"recent lookup is outside requested directory: {path}")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise ValueError(f"recent lookup hash mismatch: {path}")
        key = (level, int(record["year"]))
        if key in paths:
            raise ValueError(f"duplicate recent lookup: {key}")
        paths[key] = path
        verified.append({"level": level, "year": key[1], "path": path.as_posix(), "sha256": actual})
    return paths, verified


def _load_lookup(path: Path) -> pd.DataFrame:
    table = pd.read_parquet(path)
    table["FlightDate"] = pd.to_datetime(table["FlightDate"], errors="raise").dt.normalize()
    return table


def _aggregate_messages(
    edges: pd.DataFrame,
    *,
    side: str,
    key: str,
    partner: str,
    partner_level: str,
) -> pd.DataFrame:
    input_columns: list[str] = []
    specs: list[tuple[str, int, str]] = []
    for outcome in ("delay", "cancel"):
        for window in RECENT_WINDOWS_DAYS:
            name = f"recent_{partner_level}_{outcome}_rate_{window}d"
            input_columns.append(name)
            specs.append((outcome, window, name))
            fallback = f"recent_global_{outcome}_rate_{window}d"
            edges[name] = pd.to_numeric(edges[name], errors="coerce").fillna(edges[fallback]).fillna(0.0)
    grouped = edges.groupby(["FlightDate", key], observed=True, sort=False)
    means = grouped[input_columns].mean()
    maxima = grouped[input_columns].max()
    deviations = grouped[input_columns].std(ddof=0).fillna(0.0)
    result = means.index.to_frame(index=False)
    result[f"graph_{side}_partner_count_log1p"] = np.log1p(
        grouped[partner].nunique().reindex(means.index).to_numpy(dtype=np.float64)
    ).astype("float32")
    for outcome, window, source in specs:
        result[f"graph_{side}_partner_{outcome}_mean_{window}d"] = means[source].to_numpy(
            dtype=np.float32
        )
        result[f"graph_{side}_partner_{outcome}_max_{window}d"] = maxima[source].to_numpy(
            dtype=np.float32
        )
        result[f"graph_{side}_partner_{outcome}_std_{window}d"] = deviations[source].to_numpy(
            dtype=np.float32
        )
    feature_columns = [
        name for name in CENSUS_GRAPH_MESSAGE_FEATURES if name.startswith(f"graph_{side}_")
    ]
    values = result.loc[:, feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("schedule-graph aggregation produced non-finite features")
    return result.sort_values(["FlightDate", key], kind="mergesort").reset_index(drop=True)


def schedule_graph_messages(
    schedule: pd.DataFrame,
    *,
    global_lookup: pd.DataFrame,
    origin_lookup: pd.DataFrame,
    dest_lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute one-hop messages without reading a target-day outcome column."""

    required = {"FlightDate", "Origin", "Dest"}
    missing = sorted(required - set(schedule.columns))
    if missing:
        raise ValueError(f"schedule graph lacks columns: {missing}")
    edges = schedule.loc[:, ["FlightDate", "Origin", "Dest"]].copy()
    edges["FlightDate"] = pd.to_datetime(edges["FlightDate"], errors="raise").dt.normalize()
    global_rates = [name for name in global_lookup if "_rate_" in name]
    origin_rates = [name for name in origin_lookup if "_rate_" in name]
    dest_rates = [name for name in dest_lookup if "_rate_" in name]
    edges = edges.merge(
        global_lookup.loc[:, ["FlightDate", *global_rates]],
        on="FlightDate",
        how="left",
        validate="many_to_one",
    )
    edges = edges.merge(
        origin_lookup.loc[:, ["FlightDate", "Origin", *origin_rates]],
        on=["FlightDate", "Origin"],
        how="left",
        validate="many_to_one",
    )
    edges = edges.merge(
        dest_lookup.loc[:, ["FlightDate", "Dest", *dest_rates]],
        on=["FlightDate", "Dest"],
        how="left",
        validate="many_to_one",
    )
    origin_messages = _aggregate_messages(
        edges.copy(),
        side="origin",
        key="Origin",
        partner="Dest",
        partner_level="dest",
    )
    dest_messages = _aggregate_messages(
        edges,
        side="dest",
        key="Dest",
        partner="Origin",
        partner_level="origin",
    )
    return origin_messages, dest_messages


def build_census_graph_features(
    *,
    census_manifest: Path,
    recent_manifest: Path,
    recent_dir: Path,
    output_dir: Path,
    output_manifest: Path,
    verbose: bool = False,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse graph output directory: {output_dir}")
    if output_manifest.exists():
        raise FileExistsError(f"refusing to overwrite graph manifest: {output_manifest}")
    root = census_manifest.resolve().parent.parent
    census = _verify_json_self_hash(census_manifest)
    recent = _verify_json_self_hash(recent_manifest)
    if recent["source_feature_manifest_sha256"] != sha256_file(census_manifest):
        raise ValueError("recent lookup is not bound to the requested census manifest")
    lookup_paths, verified_lookups = _lookup_paths(recent, root=root, recent_dir=recent_dir)
    output_dir.mkdir(parents=True)
    by_level_year: dict[tuple[str, int], list[pd.DataFrame]] = {}
    verified_census: list[dict[str, Any]] = []
    lookup_cache: dict[int, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    records = sorted(census["outputs"], key=lambda record: (int(record["year"]), int(record["month"])))
    for index, record in enumerate(records, start=1):
        year = int(record["year"])
        path = _resolve(root, str(record["path"]))
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise ValueError(f"census schedule hash mismatch: {path}")
        if year not in lookup_cache:
            lookup_cache[year] = (
                _load_lookup(lookup_paths[("global", year)]),
                _load_lookup(lookup_paths[("origin", year)]),
                _load_lookup(lookup_paths[("dest", year)]),
            )
        schedule = pd.read_parquet(path, columns=["FlightDate", "Origin", "Dest"])
        origin_messages, dest_messages = schedule_graph_messages(
            schedule,
            global_lookup=lookup_cache[year][0],
            origin_lookup=lookup_cache[year][1],
            dest_lookup=lookup_cache[year][2],
        )
        by_level_year.setdefault(("origin", year), []).append(origin_messages)
        by_level_year.setdefault(("dest", year), []).append(dest_messages)
        verified_census.append(
            {"year": year, "month": int(record["month"]), "path": path.as_posix(), "sha256": actual}
        )
        if verbose:
            print(f"schedule-graph source={index}/{len(records)}", flush=True)

    outputs: list[dict[str, Any]] = []
    for (level, year), parts in sorted(by_level_year.items()):
        table = pd.concat(parts, ignore_index=True).sort_values(
            ["FlightDate", "Origin" if level == "origin" else "Dest"], kind="mergesort"
        )
        outputs.append(
            {
                "level": level,
                "year": year,
                **_atomic_graph_table(
                    table.reset_index(drop=True),
                    output_dir / f"level={level}" / f"year={year}.parquet",
                ),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "DERIVED_CLOSED_LEFT_SCHEDULE_GRAPH_MESSAGES",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": GRAPH_METHOD_NAME,
        "acronym": GRAPH_METHOD_ACRONYM,
        "census_manifest": census_manifest.as_posix(),
        "census_manifest_sha256": sha256_file(census_manifest),
        "census_manifest_self_hash": census["manifest_sha256"],
        "recent_manifest": recent_manifest.as_posix(),
        "recent_manifest_sha256": sha256_file(recent_manifest),
        "recent_manifest_self_hash": recent["manifest_sha256"],
        "census_partitions_verified": verified_census,
        "recent_lookups_verified": verified_lookups,
        "target_day_outcomes_read": False,
        "node_state_cutoff": "closed-left windows [target_date-window, target_date)",
        "edge_definition": "target-day census schedule; one row supplies one frequency weight",
        "aggregation": ["schedule-frequency-weighted mean", "maximum", "population standard deviation"],
        "features": list(CENSUS_GRAPH_MESSAGE_FEATURES),
        "outputs": outputs,
        "total_lookup_rows": sum(int(record["rows"]) for record in outputs),
        "claim_limit": "The graph uses a retrospective BTS schedule census proxy, not a validated advance schedule feed or real-time airport state.",
        "provenance": capture_provenance((Path(__file__),)),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(output_manifest, manifest)
    return manifest


def attach_census_graph_features(frame: pd.DataFrame, *, graph_dir: Path) -> pd.DataFrame:
    """Attach precomputed graph messages without changing flight order or cardinality."""

    result = frame.copy()
    result["__graph_order"] = np.arange(len(result), dtype=np.int64)
    pieces: list[pd.DataFrame] = []
    for year, current in result.groupby("Year", sort=False, observed=True):
        origin = _load_lookup(graph_dir / "level=origin" / f"year={int(year)}.parquet")
        dest = _load_lookup(graph_dir / "level=dest" / f"year={int(year)}.parquet")
        current = current.merge(
            origin,
            on=["FlightDate", "Origin"],
            how="left",
            sort=False,
            validate="many_to_one",
        ).merge(
            dest,
            on=["FlightDate", "Dest"],
            how="left",
            sort=False,
            validate="many_to_one",
        )
        pieces.append(current)
    attached = pd.concat(pieces, ignore_index=True).sort_values("__graph_order", kind="mergesort")
    if len(attached) != len(frame):
        raise AssertionError("schedule-graph attachment changed flight cardinality")
    values = attached.loc[:, list(CENSUS_GRAPH_MESSAGE_FEATURES)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("schedule-graph attachment produced missing or non-finite features")
    return attached.drop(columns="__graph_order").reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-manifest", type=Path, required=True)
    parser.add_argument("--recent-manifest", type=Path, required=True)
    parser.add_argument("--recent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = build_census_graph_features(
        census_manifest=args.census_manifest,
        recent_manifest=args.recent_manifest,
        recent_dir=args.recent_dir,
        output_dir=args.output_dir,
        output_manifest=args.manifest,
        verbose=True,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "outputs": len(result["outputs"]),
                "lookup_rows": result["total_lookup_rows"],
                "manifest": args.manifest.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
