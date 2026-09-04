"""Build and attach strict prior-day multi-timescale operating-history features.

The builder aggregates outcomes by operating day and applies closed-left calendar
windows.  A flight on date ``D`` can therefore use dates in ``[D-window, D)`` but
never another outcome from date ``D``.  The resulting tables are compact lookups,
not copies of the full flight-level feature cohort.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .contracts import (
    RECENT_OPERATIONAL_FEATURES,
    RECENT_OPERATIONAL_STATISTICS,
    RECENT_OPERATIONAL_VIEWS,
    RECENT_WINDOWS_DAYS,
)
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json

BASE_LEVEL_KEYS: dict[str, tuple[str, ...]] = {
    "global": (),
    "route": ("Route",),
    "airline": ("Reporting_Airline",),
    "origin": ("Origin",),
    "dest": ("Dest",),
}

SMOOTHING_STRENGTHS: dict[str, float] = {
    "global": 0.0,
    "route": 50.0,
    "airline": 200.0,
    "origin": 200.0,
    "dest": 200.0,
}

_VALUE_COLUMNS = (
    "flight_count",
    "cancel_events",
    "delay_support",
    "delay_events",
)


def _manifest_outputs(feature_manifest: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    payload: dict[str, Any] = json.loads(feature_manifest.read_text(encoding="utf-8"))
    declared_hash = str(payload.get("manifest_sha256", ""))
    hash_payload = payload.copy()
    hash_payload.pop("manifest_sha256", None)
    if not declared_hash or canonical_json_sha256(hash_payload) != declared_hash:
        raise ValueError("source feature manifest self-hash is invalid")
    records = list(payload.get("outputs", []))
    if not records:
        raise ValueError("feature manifest contains no output partitions")
    repository_root = feature_manifest.resolve().parent.parent
    return repository_root, payload, records


def _daily_aggregate(frame: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    work = frame.loc[
        :,
        ["FlightDate", "ArrDel15", "Cancelled", "delay_label_observed", *keys],
    ].copy()
    work["FlightDate"] = pd.to_datetime(work["FlightDate"], errors="raise").dt.normalize()
    work["flight_count"] = np.float32(1.0)
    work["cancel_events"] = pd.to_numeric(work["Cancelled"], errors="raise").astype("float32")
    work["delay_support"] = pd.to_numeric(
        work["delay_label_observed"], errors="raise"
    ).astype("float32")
    work["delay_events"] = (
        pd.to_numeric(work["ArrDel15"], errors="coerce").fillna(0.0)
        * work["delay_support"]
    ).astype("float32")
    group_columns = [*keys, "FlightDate"]
    return (
        work.groupby(group_columns, observed=True, sort=False)[list(_VALUE_COLUMNS)]
        .sum()
        .reset_index()
    )


def _rolling_sums(
    daily: pd.DataFrame,
    keys: tuple[str, ...],
    window_days: int,
) -> pd.DataFrame:
    ordered = daily.sort_values([*keys, "FlightDate"], kind="mergesort").reset_index(drop=True)
    window = f"{window_days}D"
    if keys:
        result = (
            ordered.groupby(list(keys), observed=True, sort=False)
            .rolling(
                window,
                on="FlightDate",
                closed="left",
                min_periods=1,
            )[list(_VALUE_COLUMNS)]
            .sum()
            .reset_index()
        )
    else:
        values = ordered.rolling(
            window,
            on="FlightDate",
            closed="left",
            min_periods=1,
        )[list(_VALUE_COLUMNS)].sum()
        values = values.drop(columns="FlightDate", errors="ignore")
        result = pd.concat(
            [ordered[["FlightDate"]].reset_index(drop=True), values.reset_index(drop=True)],
            axis=1,
        )
    return result


def _safe_rate(events: pd.Series, support: pd.Series) -> pd.Series:
    denominator = support.astype("float64")
    return (events.astype("float64") / denominator.where(denominator > 0)).astype("float32")


def _global_features(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    result: pd.DataFrame | None = None
    raw_by_window: dict[int, pd.DataFrame] = {}
    for window in RECENT_WINDOWS_DAYS:
        raw = _rolling_sums(daily, (), window)
        raw_by_window[window] = raw
        if result is None:
            result = raw[["FlightDate"]].copy()
        additions = pd.DataFrame(
            {
                f"recent_global_delay_rate_{window}d": _safe_rate(
                    raw["delay_events"], raw["delay_support"]
                ),
                f"recent_global_cancel_rate_{window}d": _safe_rate(
                    raw["cancel_events"], raw["flight_count"]
                ),
                f"recent_global_count_log1p_{window}d": np.log1p(
                    raw["flight_count"]
                ).astype("float32"),
                f"recent_global_delay_support_log1p_{window}d": np.log1p(
                    raw["delay_support"]
                ).astype("float32"),
            }
        )
        result = pd.concat([result, additions], axis=1)
    if result is None:
        raise ValueError("global daily aggregate is empty")
    return result, raw_by_window


def _group_features(
    daily: pd.DataFrame,
    *,
    level: str,
    keys: tuple[str, ...],
    global_raw: dict[int, pd.DataFrame],
    smoothing_strength: float,
) -> pd.DataFrame:
    result: pd.DataFrame | None = None
    for window in RECENT_WINDOWS_DAYS:
        raw = _rolling_sums(daily, keys, window)
        if result is None:
            result = raw.loc[:, [*keys, "FlightDate"]].copy()
        global_window = global_raw[window].loc[
            :,
            ["FlightDate", "delay_events", "delay_support", "cancel_events", "flight_count"],
        ].rename(
            columns={
                "delay_events": "global_delay_events",
                "delay_support": "global_delay_support",
                "cancel_events": "global_cancel_events",
                "flight_count": "global_flight_count",
            }
        )
        raw = raw.merge(global_window, on="FlightDate", how="left", validate="many_to_one")
        global_delay_rate = _safe_rate(
            raw["global_delay_events"], raw["global_delay_support"]
        )
        global_cancel_rate = _safe_rate(
            raw["global_cancel_events"], raw["global_flight_count"]
        )
        delay_support = raw["delay_support"].fillna(0.0).astype("float64")
        delay_events = raw["delay_events"].fillna(0.0).astype("float64")
        flight_count = raw["flight_count"].fillna(0.0).astype("float64")
        cancel_events = raw["cancel_events"].fillna(0.0).astype("float64")
        additions = pd.DataFrame(
            {
                f"recent_{level}_delay_rate_{window}d": (
                    (delay_events + smoothing_strength * global_delay_rate)
                    / (delay_support + smoothing_strength)
                ).astype("float32"),
                f"recent_{level}_cancel_rate_{window}d": (
                    (cancel_events + smoothing_strength * global_cancel_rate)
                    / (flight_count + smoothing_strength)
                ).astype("float32"),
                f"recent_{level}_count_log1p_{window}d": np.log1p(flight_count).astype(
                    "float32"
                ),
                f"recent_{level}_delay_support_log1p_{window}d": np.log1p(
                    delay_support
                ).astype("float32"),
            }
        )
        result = pd.concat([result, additions], axis=1)
    if result is None:
        raise ValueError(f"{level} daily aggregate is empty")
    return result


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite recent-feature evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated recent-feature partial exists: {partial}")
    frame.to_parquet(partial, index=False, compression="zstd", row_group_size=100_000)
    partial.replace(path)
    return {
        "path": path.as_posix(),
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "first_date": str(frame["FlightDate"].min().date()),
        "last_date": str(frame["FlightDate"].max().date()),
    }


def _write_level_partitions(
    table: pd.DataFrame,
    *,
    level: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    years = table["FlightDate"].dt.year.astype("int16")
    for year in sorted(years.unique()):
        partition = table.loc[years.eq(year)].reset_index(drop=True)
        record = _atomic_parquet(
            partition,
            output_dir / f"level={level}" / f"year={int(year)}.parquet",
        )
        records.append({"level": level, "year": int(year), **record})
    return records


def build_recent_feature_tables(
    *,
    feature_manifest: Path,
    output_dir: Path,
    output_manifest: Path,
    verbose: bool = False,
) -> dict[str, Any]:
    """Build all lookup tables once, refusing to reuse any output location."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse recent-feature directory: {output_dir}")
    if output_manifest.exists():
        raise FileExistsError(f"refusing to overwrite recent-feature manifest: {output_manifest}")
    repository_root, source_manifest, source_records = _manifest_outputs(feature_manifest)
    output_dir.mkdir(parents=True)

    daily_parts: dict[str, list[pd.DataFrame]] = {level: [] for level in BASE_LEVEL_KEYS}
    verified_sources: list[dict[str, Any]] = []
    for source_index, record in enumerate(source_records, start=1):
        source_path = repository_root / str(record["path"])
        if verbose:
            print(
                f"recent aggregation source={source_index}/{len(source_records)} "
                f"path={record['path']}",
                flush=True,
            )
        actual_hash = sha256_file(source_path)
        if actual_hash != record["sha256"]:
            raise ValueError(f"source feature hash mismatch: {source_path}")
        frame = pd.read_parquet(
            source_path,
            columns=[
                "FlightDate",
                "Route",
                "Reporting_Airline",
                "Origin",
                "Dest",
                "ArrDel15",
                "Cancelled",
                "delay_label_observed",
            ],
        )
        for level, keys in BASE_LEVEL_KEYS.items():
            daily_parts[level].append(_daily_aggregate(frame, keys))
        verified_sources.append(
            {
                "path": str(record["path"]),
                "rows": int(record["rows"]),
                "sha256": actual_hash,
            }
        )
        del frame

    daily: dict[str, pd.DataFrame] = {}
    for level, keys in BASE_LEVEL_KEYS.items():
        combined = pd.concat(daily_parts[level], ignore_index=True)
        daily[level] = (
            combined.groupby([*keys, "FlightDate"], observed=True, sort=False)[
                list(_VALUE_COLUMNS)
            ]
            .sum()
            .reset_index()
        )
        del combined, daily_parts[level]

    global_table, global_raw = _global_features(daily["global"])
    if verbose:
        print("recent rolling level=global", flush=True)
    output_records = _write_level_partitions(
        global_table,
        level="global",
        output_dir=output_dir,
    )
    feature_names: dict[str, list[str]] = {
        "global": [name for name in global_table.columns if name.startswith("recent_")]
    }
    del global_table

    for level in ("route", "airline", "origin", "dest"):
        if verbose:
            print(f"recent rolling level={level}", flush=True)
        table = _group_features(
            daily[level],
            level=level,
            keys=BASE_LEVEL_KEYS[level],
            global_raw=global_raw,
            smoothing_strength=SMOOTHING_STRENGTHS[level],
        )
        feature_names[level] = [name for name in table.columns if name.startswith("recent_")]
        output_records.extend(
            _write_level_partitions(table, level=level, output_dir=output_dir)
        )
        del table

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "builder_version": 1,
        "status": "DERIVED_STRICT_PRIOR_DAY_OPERATIONAL_PROXY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_feature_manifest": feature_manifest.as_posix(),
        "source_feature_manifest_sha256": sha256_file(feature_manifest),
        "source_feature_variant": source_manifest["feature_variant"],
        "source_partitions_verified": verified_sources,
        "cutoff_rule": "closed-left calendar windows [target_date-window, target_date)",
        "target_day_outcomes_excluded": True,
        "live_availability_claimed": False,
        "availability_note": (
            "Historical BTS outcomes proxy an operational feed; production use requires "
            "a separately validated prior-day data source and reporting-lag policy."
        ),
        "windows_days": list(RECENT_WINDOWS_DAYS),
        "base_level_keys": {name: list(keys) for name, keys in BASE_LEVEL_KEYS.items()},
        "smoothing_strengths": SMOOTHING_STRENGTHS,
        "base_table_features": feature_names,
        "attached_model_features": list(RECENT_OPERATIONAL_FEATURES),
        "outputs": output_records,
        "total_lookup_rows": sum(int(record["rows"]) for record in output_records),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(output_manifest, manifest)
    return manifest


def _load_level_year(recent_dir: Path, level: str, year: int) -> pd.DataFrame:
    path = recent_dir / f"level={level}" / f"year={year}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing recent-feature lookup: {path}")
    table = pd.read_parquet(path)
    table["FlightDate"] = pd.to_datetime(table["FlightDate"], errors="raise").dt.normalize()
    return table


def _rename_feature_prefix(table: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    return table.rename(
        columns={
            name: name.replace(f"recent_{source}_", f"recent_{target}_", 1)
            for name in table.columns
            if name.startswith(f"recent_{source}_")
        }
    )


def _merge_lookup(
    frame: pd.DataFrame,
    table: pd.DataFrame,
    *,
    left_key: str | None,
    right_key: str | None,
) -> pd.DataFrame:
    left_on = ["FlightDate"] if left_key is None else ["FlightDate", left_key]
    lookup = table
    if right_key is None:
        right_on = ["FlightDate"]
        drop_after: list[str] = []
    elif left_key == right_key:
        right_on = ["FlightDate", right_key]
        drop_after = []
    else:
        lookup_key = "__recent_lookup_key"
        if lookup_key in frame.columns or lookup_key in table.columns:
            raise ValueError("reserved recent lookup key is already present")
        lookup = table.rename(columns={right_key: lookup_key})
        right_on = ["FlightDate", lookup_key]
        drop_after = [lookup_key]
    return frame.merge(
        lookup,
        how="left",
        left_on=left_on,
        right_on=right_on,
        sort=False,
        validate="many_to_one",
        suffixes=("", "__lookup"),
    ).drop(columns=drop_after, errors="raise")


def _fallback_prior(view: str, outcome: str) -> str:
    level = {
        "global": "global",
        "route": "route",
        "airline": "airline",
        "origin_outbound": "origin",
        "dest_inbound": "dest",
        # Cross-direction histories have no matching earlier-year feature for the
        # opposite endpoint, so the strictly prior global rate is the neutral fallback.
        "origin_inbound": "global",
        "dest_outbound": "global",
    }[view]
    return f"prior_{level}_{outcome}_rate"


def attach_recent_features(
    frame: pd.DataFrame,
    *,
    recent_dir: Path,
    fallback_strategy: Literal["earlier_year", "recent_global"] = "earlier_year",
) -> pd.DataFrame:
    """Attach direct and cross-direction lookup views to one or more years."""

    pieces: list[pd.DataFrame] = []
    normalized = frame.copy()
    normalized["__recent_row_order"] = np.arange(len(normalized), dtype=np.int64)
    normalized["FlightDate"] = pd.to_datetime(
        normalized["FlightDate"], errors="raise"
    ).dt.normalize()
    for year, year_frame in normalized.groupby("Year", sort=False, observed=True):
        current = year_frame.copy()
        cache = {
            level: _load_level_year(recent_dir, level, int(year)) for level in BASE_LEVEL_KEYS
        }
        current = _merge_lookup(current, cache["global"], left_key=None, right_key=None)
        current = _merge_lookup(
            current,
            cache["route"],
            left_key="Route",
            right_key="Route",
        )
        current = _merge_lookup(
            current,
            cache["airline"],
            left_key="Reporting_Airline",
            right_key="Reporting_Airline",
        )
        origin_outbound = _rename_feature_prefix(cache["origin"], "origin", "origin_outbound")
        current = _merge_lookup(
            current,
            origin_outbound,
            left_key="Origin",
            right_key="Origin",
        )
        dest_inbound = _rename_feature_prefix(cache["dest"], "dest", "dest_inbound")
        current = _merge_lookup(
            current,
            dest_inbound,
            left_key="Dest",
            right_key="Dest",
        )
        origin_inbound = _rename_feature_prefix(cache["dest"], "dest", "origin_inbound")
        current = _merge_lookup(
            current,
            origin_inbound,
            left_key="Origin",
            right_key="Dest",
        )
        dest_outbound = _rename_feature_prefix(cache["origin"], "origin", "dest_outbound")
        current = _merge_lookup(
            current,
            dest_outbound,
            left_key="Dest",
            right_key="Origin",
        )

        if fallback_strategy == "recent_global":
            for window in RECENT_WINDOWS_DAYS:
                current[f"recent_global_delay_rate_{window}d"] = pd.to_numeric(
                    current[f"recent_global_delay_rate_{window}d"], errors="coerce"
                ).fillna(0.20)
                current[f"recent_global_cancel_rate_{window}d"] = pd.to_numeric(
                    current[f"recent_global_cancel_rate_{window}d"], errors="coerce"
                ).fillna(0.02)
        elif fallback_strategy != "earlier_year":
            raise ValueError(f"unknown recent fallback strategy: {fallback_strategy}")

        for view in RECENT_OPERATIONAL_VIEWS:
            for window in RECENT_WINDOWS_DAYS:
                for outcome in ("delay", "cancel"):
                    name = f"recent_{view}_{outcome}_rate_{window}d"
                    fallback = (
                        current[f"recent_global_{outcome}_rate_{window}d"]
                        if fallback_strategy == "recent_global"
                        else current[_fallback_prior(view, outcome)]
                    )
                    current[name] = pd.to_numeric(current[name], errors="coerce").fillna(fallback)
                for statistic in ("count_log1p", "delay_support_log1p"):
                    name = f"recent_{view}_{statistic}_{window}d"
                    current[name] = pd.to_numeric(current[name], errors="coerce").fillna(0.0)

        recent_values = current.loc[:, list(RECENT_OPERATIONAL_FEATURES)].to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(recent_values).all():
            raise ValueError(f"recent feature join produced non-finite values for {int(year)}")
        rate_names = [name for name in RECENT_OPERATIONAL_FEATURES if "_rate_" in name]
        rates = current.loc[:, rate_names].to_numpy(dtype=np.float64)
        if ((rates < 0.0) | (rates > 1.0)).any():
            raise ValueError(f"recent feature join produced invalid rates for {int(year)}")
        pieces.append(current)

    attached = pd.concat(pieces, axis=0).sort_values(
        "__recent_row_order", kind="mergesort"
    )
    if len(attached) != len(frame):
        raise AssertionError("recent feature attachment changed the flight row count")
    return attached.drop(columns="__recent_row_order").reset_index(drop=True)


def recent_feature_columns() -> tuple[str, ...]:
    expected = tuple(
        f"recent_{view}_{statistic}_{window}d"
        for view in RECENT_OPERATIONAL_VIEWS
        for statistic in RECENT_OPERATIONAL_STATISTICS
        for window in RECENT_WINDOWS_DAYS
    )
    if expected != RECENT_OPERATIONAL_FEATURES:
        raise AssertionError("recent feature registry and builder naming diverged")
    return expected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = build_recent_feature_tables(
        feature_manifest=args.feature_manifest,
        output_dir=args.output_dir,
        output_manifest=args.manifest,
        verbose=True,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "outputs": len(manifest["outputs"]),
                "total_lookup_rows": manifest["total_lookup_rows"],
                "manifest": args.manifest.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
