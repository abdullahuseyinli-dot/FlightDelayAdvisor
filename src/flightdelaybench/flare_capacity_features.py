"""Materialize the capacity-conditioned Resource-Time Flight Hypergraph.

For target year Y, scheduling frontiers and latent-rotation kernels come from
Y-1.  Target/context files are projected to schedule columns before processing;
no target-year tail identifiers or outcomes are read.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import joblib  # type: ignore[import-untyped]
import pandas as pd

from .flare_capacity import (
    CapacityConfig,
    build_capacity_hypergraph,
    fit_schedule_frontier,
    resolve_scheduled_elapsed,
)
from .flare_capacity_contracts import CAPACITY_ALL_FEATURES
from .flare_resource_catalog import load_resource_catalog
from .flare_rotation import LatentRotationGraph
from .flare_weather import timezone_catalog
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

SCHEDULE_COLUMNS = (
    "sample_id",
    "FlightDate",
    "Origin",
    "Dest",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "CRSDepMinutes",
    "CRSElapsedTime",
    "Distance",
)

CAPACITY_WEATHER_COLUMNS = tuple(
    [
        f"flare24_{side}_{timing}_{variable}"
        for side, timing in (("origin", "departure"), ("dest", "arrival"))
        for variable in (
            "wind_direction",
            "wind_speed",
            "wind_gust",
            "precipitation",
        )
    ]
    + [
        f"flare24_{side}_{variable}"
        for side in ("origin", "dest")
        for variable in (
            "convective_index",
            "icing_environment_index",
            "wind_optimal_gust_crosswind_knots",
            "gust_excess_knots",
            "visibility_hazard",
        )
    ]
)


def _verified_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    hash_key = "manifest_sha256" if "manifest_sha256" in payload else "catalog_sha256"
    if hash_key not in payload:
        raise ValueError(f"JSON artifact has no self-hash: {path}")
    recorded = payload[hash_key]
    body = {key: value for key, value in payload.items() if key != hash_key}
    if canonical_json_sha256(body) != recorded:
        raise ValueError(f"JSON artifact self-hash failed: {path}")
    return payload


def _partition_path(census_dir: Path, year: int, month: int) -> Path:
    if year not in {2023, 2024, 2025}:
        raise ValueError(f"CC-RTH refuses census year {year}")
    path = census_dir / f"year={year}" / f"month={month:02d}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing census schedule partition: {path}")
    return path


def _period_paths(census_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> tuple[Path, ...]:
    periods = pd.period_range(start=start, end=end, freq="M")
    return tuple(
        _partition_path(census_dir, int(period.year), int(period.month)) for period in periods
    )


def _read_period_schedule(
    census_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, tuple[Path, ...]]:
    paths = _period_paths(census_dir, start, end)
    parts = [pd.read_parquet(path, columns=list(SCHEDULE_COLUMNS)) for path in paths]
    frame = pd.concat(parts, ignore_index=True)
    dates = pd.to_datetime(frame["FlightDate"], errors="raise").dt.normalize()
    frame["FlightDate"] = dates
    selected = frame.loc[dates.between(start.normalize(), end.normalize())].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"empty schedule from {start.date()} through {end.date()}")
    if selected["sample_id"].isna().any() or selected["sample_id"].duplicated().any():
        raise ValueError("schedule period contains invalid sample ids")
    return selected, paths


def _read_year_schedule(census_dir: Path, year: int) -> tuple[pd.DataFrame, tuple[Path, ...]]:
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31)
    return _read_period_schedule(census_dir, start, end)


def _attach_weather(target: pd.DataFrame, feature_path: Path) -> pd.DataFrame:
    weather = pd.read_parquet(
        feature_path,
        columns=["sample_id", *CAPACITY_WEATHER_COLUMNS],
    )
    if weather["sample_id"].isna().any() or weather["sample_id"].duplicated().any():
        raise ValueError(f"weather capacity partition has invalid sample ids: {feature_path}")
    result = target.merge(
        weather,
        on="sample_id",
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if not result["_merge"].eq("both").all():
        raise ValueError(f"weather capacity partition omits target flights: {feature_path}")
    return result.drop(columns="_merge")


def _load_rotation_model(
    manifest_path: Path,
    target_year: int,
) -> tuple[LatentRotationGraph, dict[str, Any]]:
    manifest = _verified_json(manifest_path)
    matches = [
        record
        for record in manifest.get("models", [])
        if int(record.get("target_year", -1)) == target_year
    ]
    if len(matches) != 1:
        raise ValueError(f"rotation manifest has {len(matches)} models for {target_year}")
    record = dict(matches[0])
    path = Path(record["path"])
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError(f"rotation model checksum failed: {path}")
    model = joblib.load(path)
    if not isinstance(model, LatentRotationGraph):
        raise TypeError(f"rotation artifact has unexpected type: {type(model)!r}")
    return model, record


def infer_rotation_edges(
    model: LatentRotationGraph,
    target: pd.DataFrame,
    context: pd.DataFrame,
    *,
    timezone_by_airport: dict[str, str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Emit schedule-only candidate predecessor edges for target-day successors."""

    target_dates = pd.to_datetime(target["FlightDate"], errors="raise").dt.normalize()
    context_dates = pd.to_datetime(context["FlightDate"], errors="raise").dt.normalize()
    target_ids = set(target["sample_id"].astype(str))
    outputs: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for target_date in sorted(target_dates.unique()):
        timestamp = pd.Timestamp(target_date)
        mask = context_dates.between(
            timestamp - pd.Timedelta(days=1),
            timestamp + pd.Timedelta(days=1),
        )
        window = context.loc[mask].reset_index(drop=True)
        if window.empty:
            raise ValueError(f"empty rotation window for {timestamp.date()}")
        inference = model.infer(window, timezone_by_airport=timezone_by_airport)
        edges = inference.edges.copy()
        if not edges.empty:
            predecessor = edges["predecessor_position"].to_numpy(dtype="int64")
            successor = edges["successor_position"].to_numpy(dtype="int64")
            edges["predecessor_sample_id"] = (
                window.iloc[predecessor]["sample_id"].astype(str).to_numpy()
            )
            edges["successor_sample_id"] = (
                window.iloc[successor]["sample_id"].astype(str).to_numpy()
            )
            successor_dates = pd.to_datetime(
                window.iloc[successor]["FlightDate"].to_numpy()
            ).normalize()
            emit = edges["successor_sample_id"].isin(target_ids) & (successor_dates == timestamp)
            outputs.append(
                edges.loc[
                    emit,
                    [
                        "predecessor_sample_id",
                        "successor_sample_id",
                        "turn_minutes",
                        "score",
                        "probability",
                    ],
                ].reset_index(drop=True)
            )
        diagnostic = dict(inference.diagnostics)
        diagnostic["target_date"] = timestamp.date().isoformat()
        diagnostic["emitted_edges"] = 0 if edges.empty else int(emit.sum())
        diagnostics.append(diagnostic)
    result = (
        pd.concat(outputs, ignore_index=True)
        if outputs
        else pd.DataFrame(
            columns=[
                "predecessor_sample_id",
                "successor_sample_id",
                "turn_minutes",
                "score",
                "probability",
            ]
        )
    )
    if not result.empty and not set(result["successor_sample_id"]).issubset(target_ids):
        raise RuntimeError("rotation edge materialization escaped target flights")
    return result, diagnostics


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite CC-RTH artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated CC-RTH partial exists: {partial}")
    frame.to_parquet(partial, index=False, compression="zstd")
    partial.replace(path)
    return {
        "path": path.as_posix(),
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def materialize_capacity_hypergraph(
    *,
    census_dir: Path,
    context_census_dir: Path | None = None,
    context_manifest_path: Path | None = None,
    weather_feature_dir: Path,
    weather_feature_manifest_path: Path,
    airport_catalog_path: Path,
    context_airport_catalog_path: Path | None = None,
    resource_catalog_path: Path,
    resource_catalog_manifest_path: Path,
    rotation_manifest_path: Path,
    output_dir: Path,
    graph_dir: Path,
    frontier_dir: Path,
    manifest_path: Path,
    target_years: tuple[int, ...],
    target_months: tuple[int, ...] = tuple(range(1, 13)),
    operational_constraints_path: Path | None = None,
    context_buffer_days: int = 2,
    artifact_mode: Literal["full", "features-only"] = "full",
    config: CapacityConfig | None = None,
) -> dict[str, Any]:
    """Build versioned CC-RTH features and sparse graph artifacts."""

    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite CC-RTH manifest: {manifest_path}")
    if not target_years or any(year not in {2024, 2025} for year in target_years):
        raise ValueError("CC-RTH target years must be a subset of 2024-2025")
    if (
        not target_months
        or len(target_months) != len(set(target_months))
        or any(month < 1 or month > 12 for month in target_months)
    ):
        raise ValueError("CC-RTH target months must be unique values in 1..12")
    if context_buffer_days < 1:
        raise ValueError("capacity context buffer must be at least one day")
    if artifact_mode not in {"full", "features-only"}:
        raise ValueError("artifact_mode must be 'full' or 'features-only'")
    active_context_dir = context_census_dir or census_dir
    context_is_external = active_context_dir.resolve() != census_dir.resolve()
    if context_is_external and context_manifest_path is None:
        raise ValueError("external context census requires a context manifest")
    if context_is_external and context_airport_catalog_path is None:
        raise ValueError("external context census requires its airport catalog")
    settings = config or CapacityConfig()
    weather_manifest = _verified_json(weather_feature_manifest_path)
    resource_manifest = _verified_json(resource_catalog_manifest_path)
    context_manifest = (
        None if context_manifest_path is None else _verified_json(context_manifest_path)
    )
    if context_manifest is not None:
        if context_manifest.get("outcome_columns_read") != []:
            raise ValueError("context manifest records outcome access")
        if context_manifest.get("tail_number_read") is not False:
            raise ValueError("context manifest does not prove target-tail exclusion")
        if context_manifest.get("confirmation_outcomes_accessed") is not False:
            raise ValueError("context manifest records confirmation-outcome access")
        if context_is_external and context_manifest.get("method") != (
            "FLARE24-BOUNDARY-COMPLETE-SCHEDULE-CONTEXT"
        ):
            raise ValueError("external context is not the boundary-complete schedule artifact")
    if sha256_file(resource_catalog_path) != resource_manifest["output"]["sha256"]:
        raise ValueError("resource catalog does not match its manifest")
    resource_catalog = load_resource_catalog(resource_catalog_path)
    resource_airports = set(resource_catalog)
    if not resource_airports:
        raise ValueError("resource catalog contains no airports")
    active_airport_catalog = context_airport_catalog_path or airport_catalog_path
    timezones = timezone_catalog(active_airport_catalog)
    constraints = (
        None
        if operational_constraints_path is None
        else pd.read_parquet(operational_constraints_path)
    )
    started = time.perf_counter()
    outputs: list[dict[str, Any]] = []
    frontier_outputs: list[dict[str, Any]] = []
    input_records: dict[str, dict[str, Any]] = {}
    total_rows = 0
    total_nodes = 0
    total_incidence = 0
    total_rotation_edges = 0
    coverage_sums = {feature: 0 for feature in CAPACITY_ALL_FEATURES}
    diagnostics: list[dict[str, Any]] = []

    for target_year in target_years:
        history_year = target_year - 1
        history, history_paths = _read_year_schedule(active_context_dir, history_year)
        frontier = fit_schedule_frontier(
            history,
            timezone_by_airport=timezones,
            resource_airports=resource_airports,
        )
        frontier_path = frontier_dir / f"history={history_year}_for={target_year}.parquet"
        frontier_record = _atomic_parquet(frontier, frontier_path)
        frontier_outputs.append(
            {
                "history_year": history_year,
                "target_year": target_year,
                **frontier_record,
            }
        )
        for path in history_paths:
            input_records.setdefault(
                f"{path.as_posix()}|frontier-{target_year}",
                {
                    "path": path.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "columns_read": list(SCHEDULE_COLUMNS),
                    "role": f"strictly-prior-year scheduling frontier for {target_year}",
                },
            )
        del history
        gc.collect()
        rotation_model, rotation_model_record = _load_rotation_model(
            rotation_manifest_path, target_year
        )

        for month in target_months:
            month_start = pd.Timestamp(year=target_year, month=month, day=1)
            month_end = month_start + pd.offsets.MonthEnd(0)
            context_start = max(
                month_start - pd.Timedelta(days=context_buffer_days),
                pd.Timestamp(year=2023, month=1, day=1),
            )
            context_end = min(
                month_end + pd.Timedelta(days=context_buffer_days),
                pd.Timestamp(year=2025, month=12, day=31),
            )
            target, target_paths = _read_period_schedule(census_dir, month_start, month_end)
            context, context_paths = _read_period_schedule(
                active_context_dir, context_start, context_end
            )
            weather_path = (
                weather_feature_dir / f"year={target_year}" / f"month={month:02d}.parquet"
            )
            if not weather_path.is_file():
                raise FileNotFoundError(f"missing FLARE weather features: {weather_path}")
            target_weather = _attach_weather(target, weather_path)
            rotation_context, rotation_elapsed_imputations = resolve_scheduled_elapsed(context)
            rotation_edges, rotation_diagnostics = infer_rotation_edges(
                rotation_model,
                target,
                rotation_context,
                timezone_by_airport=timezones,
            )
            result = build_capacity_hypergraph(
                target_weather,
                context,
                frontier,
                timezone_by_airport=timezones,
                resource_catalog=resource_catalog,
                resource_airports=resource_airports,
                rotation_edges=rotation_edges,
                operational_constraints=constraints,
                config=settings,
            )
            feature_path = output_dir / f"year={target_year}" / f"month={month:02d}.parquet"
            graph_month = graph_dir / f"year={target_year}" / f"month={month:02d}"
            feature_record = _atomic_parquet(result.features, feature_path)
            if artifact_mode == "full":
                flight_record = _atomic_parquet(
                    result.flight_nodes, graph_month / "flight_nodes.parquet"
                )
                resource_record = _atomic_parquet(
                    result.resource_nodes, graph_month / "resource_nodes.parquet"
                )
                incidence_record = _atomic_parquet(
                    result.incidence_edges, graph_month / "incidence_edges.parquet"
                )
                rotation_record = _atomic_parquet(
                    result.rotation_edges, graph_month / "rotation_edges.parquet"
                )
            else:
                flight_record = None
                resource_record = None
                incidence_record = None
                rotation_record = None
            for feature in CAPACITY_ALL_FEATURES:
                coverage_sums[feature] += int(result.features[feature].notna().sum())
            month_record = {
                "year": target_year,
                "month": month,
                "features": feature_record,
                "flight_nodes": flight_record,
                "resource_nodes": resource_record,
                "incidence_edges": incidence_record,
                "rotation_edges": rotation_record,
                "diagnostics": result.diagnostics,
                "rotation_schedule_elapsed_imputations": rotation_elapsed_imputations,
                "rotation_diagnostics": rotation_diagnostics,
                "rotation_model": rotation_model_record,
            }
            outputs.append(month_record)
            total_rows += len(result.features)
            total_nodes += len(result.resource_nodes)
            total_incidence += len(result.incidence_edges)
            total_rotation_edges += len(result.rotation_edges)
            diagnostics.append(
                {
                    "year": target_year,
                    "month": month,
                    **result.diagnostics,
                }
            )
            for path in target_paths:
                input_records.setdefault(
                    f"{path.as_posix()}|target",
                    {
                        "path": path.as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "columns_read": list(SCHEDULE_COLUMNS),
                        "role": "frozen scored target schedule; no outcomes or tails",
                    },
                )
            for path in context_paths:
                input_records.setdefault(
                    f"{path.as_posix()}|context",
                    {
                        "path": path.as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "columns_read": list(SCHEDULE_COLUMNS),
                        "role": (
                            "boundary-complete adjacent-day schedule context; no outcomes or tails"
                            if context_is_external
                            else "adjacent-day schedule context; no outcomes or tails"
                        ),
                    },
                )
            input_records.setdefault(
                f"{weather_path.as_posix()}|weather",
                {
                    "path": weather_path.as_posix(),
                    "bytes": weather_path.stat().st_size,
                    "sha256": sha256_file(weather_path),
                    "columns_read": ["sample_id", *CAPACITY_WEATHER_COLUMNS],
                    "role": "cutoff-coherent FLARE-24 weather covariates",
                },
            )
            print(
                f"materialized CC-RTH {target_year}-{month:02d}: "
                f"flights={len(result.features)} resources={len(result.resource_nodes)} "
                f"rotation_edges={len(result.rotation_edges)}",
                flush=True,
            )
            del target, context, target_weather, rotation_context, rotation_edges, result
            gc.collect()
        del frontier, rotation_model
        gc.collect()

    coverage = {feature: coverage_sums[feature] / total_rows for feature in CAPACITY_ALL_FEATURES}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "method": (
            "FLARE-24-BOUNDARY-COMPLETE-PROBABILISTIC-OPERATIONS-TWIN"
            if context_is_external
            else "FLARE-24-CAPACITY-CONDITIONED-RESOURCE-TIME-HYPERGRAPH"
        ),
        "short_name": "BC-POT-v1" if context_is_external else "CC-RTH-v1",
        "status": (
            (
                "COMPLETE_COVARIATE_GRAPH_NO_TARGET_OUTCOMES_ACCESSED"
                if artifact_mode == "full"
                else "COMPLETE_COVARIATE_FEATURES_NO_TARGET_OUTCOMES_ACCESSED"
            )
            if set(target_months) == set(range(1, 13))
            else (
                "PARTIAL_SMOKE_COVARIATE_GRAPH_NO_TARGET_OUTCOMES_ACCESSED"
                if artifact_mode == "full"
                else "PARTIAL_SMOKE_COVARIATE_FEATURES_NO_TARGET_OUTCOMES_ACCESSED"
            )
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target_years": list(target_years),
        "target_months": list(target_months),
        "rows": total_rows,
        "resource_nodes": total_nodes,
        "incidence_edges": total_incidence,
        "rotation_edges": total_rotation_edges,
        "artifact_mode": artifact_mode,
        "scored_population": "unchanged frozen top100-to-top100 sample ids",
        "context_population": (
            "all BTS flights touching a frozen top-100 endpoint"
            if context_is_external
            else "frozen top100-to-top100 sample ids"
        ),
        "features": list(CAPACITY_ALL_FEATURES),
        "feature_nonmissing_fraction": coverage,
        "frontier_outputs": frontier_outputs,
        "outputs": outputs,
        "schedule_inputs": list(input_records.values()),
        "weather_feature_manifest": {
            "path": weather_feature_manifest_path.as_posix(),
            "sha256": sha256_file(weather_feature_manifest_path),
            "self_hash": weather_manifest["manifest_sha256"],
        },
        "resource_catalog_manifest": {
            "path": resource_catalog_manifest_path.as_posix(),
            "sha256": sha256_file(resource_catalog_manifest_path),
            "self_hash": resource_manifest["manifest_sha256"],
        },
        "rotation_manifest": {
            "path": rotation_manifest_path.as_posix(),
            "sha256": sha256_file(rotation_manifest_path),
        },
        "context_schedule_manifest": (
            None
            if context_manifest_path is None or context_manifest is None
            else {
                "path": context_manifest_path.as_posix(),
                "sha256": sha256_file(context_manifest_path),
                "self_hash": context_manifest["manifest_sha256"],
            }
        ),
        "context_airport_catalog": {
            "path": active_airport_catalog.as_posix(),
            "sha256": sha256_file(active_airport_catalog),
        },
        "operational_constraints": (
            None
            if operational_constraints_path is None
            else {
                "path": operational_constraints_path.as_posix(),
                "sha256": sha256_file(operational_constraints_path),
            }
        ),
        "configuration": {name: getattr(settings, name) for name in settings.__dataclass_fields__},
        "diagnostics": diagnostics,
        "outcome_columns_read": [],
        "target_tail_number_read": False,
        "confirmation_outcomes_accessed": False,
        "split_unit_required": "whole operational date with graph-boundary purge",
        "environment": {
            "python": platform.python_version(),
            "joblib": version("joblib"),
            "pandas": version("pandas"),
            "pyarrow": version("pyarrow"),
        },
        "provenance": capture_provenance(
            (
                Path(__file__),
                Path(__file__).with_name("flare_capacity.py"),
                Path(__file__).with_name("flare_capacity_contracts.py"),
                Path(__file__).with_name("flare_resource_catalog.py"),
            )
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "The BTS final schedule is a retrospective T-24 schedule proxy. Prior-year "
            "frontiers are empirical scheduling envelopes, not physical AAR/ADR. NASR gives "
            "static geometry, not active configurations. Optional operational constraints "
            "have zero coverage unless an authentic issue-time-vintaged feed is supplied. "
            "Boundary flights are context-only and do not change the scored population."
        ),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-dir", type=Path, required=True)
    parser.add_argument("--context-census-dir", type=Path)
    parser.add_argument("--context-manifest", type=Path)
    parser.add_argument("--weather-feature-dir", type=Path, required=True)
    parser.add_argument("--weather-feature-manifest", type=Path, required=True)
    parser.add_argument("--airport-catalog", type=Path, required=True)
    parser.add_argument("--context-airport-catalog", type=Path)
    parser.add_argument("--resource-catalog", type=Path, required=True)
    parser.add_argument("--resource-catalog-manifest", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--frontier-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-years", type=int, nargs="+", required=True)
    parser.add_argument("--target-months", type=int, nargs="+", default=list(range(1, 13)))
    parser.add_argument("--operational-constraints", type=Path)
    parser.add_argument("--context-buffer-days", type=int, default=2)
    parser.add_argument(
        "--artifact-mode",
        choices=("full", "features-only"),
        default="full",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = materialize_capacity_hypergraph(
        census_dir=args.census_dir,
        context_census_dir=args.context_census_dir,
        context_manifest_path=args.context_manifest,
        weather_feature_dir=args.weather_feature_dir,
        weather_feature_manifest_path=args.weather_feature_manifest,
        airport_catalog_path=args.airport_catalog,
        context_airport_catalog_path=args.context_airport_catalog,
        resource_catalog_path=args.resource_catalog,
        resource_catalog_manifest_path=args.resource_catalog_manifest,
        rotation_manifest_path=args.rotation_manifest,
        output_dir=args.output_dir,
        graph_dir=args.graph_dir,
        frontier_dir=args.frontier_dir,
        manifest_path=args.manifest,
        target_years=tuple(args.target_years),
        target_months=tuple(args.target_months),
        operational_constraints_path=args.operational_constraints,
        context_buffer_days=args.context_buffer_days,
        artifact_mode=args.artifact_mode,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": result["rows"],
                "resource_nodes": result["resource_nodes"],
                "incidence_edges": result["incidence_edges"],
                "rotation_edges": result["rotation_edges"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
