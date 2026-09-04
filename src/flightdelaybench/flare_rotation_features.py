"""Materialize expanding, schedule-only FLARE-24 latent-rotation features.

For target year ``Y``, the connection kernel is supervised with tail numbers
from ``Y-1`` only.  Target-year files are read with an explicit schedule-only
column projection that excludes tails and all outcomes.  Inference uses a
three-calendar-day schedule window so flights near midnight can compete for
the same incoming aircraft without crossing the information boundary.
"""

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

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .contracts import FLARE24_ROTATION_FEATURES, RECENT_WINDOWS_DAYS
from .flare_rotation import LatentRotationGraph
from .flare_weather import timezone_catalog
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance
from .recent import attach_recent_features

ROTATION_TARGET_COLUMNS = (
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
ROTATION_HISTORY_COLUMNS = (*ROTATION_TARGET_COLUMNS, "Tail_Number")
RISK_DELAY_VIEWS = ("route", "airline", "origin_outbound", "dest_inbound")
RISK_CANCEL_VIEWS = (
    "route",
    "airline",
    "origin_outbound",
    "origin_inbound",
    "dest_inbound",
    "dest_outbound",
)


def _partition_path(census_dir: Path, year: int, month: int) -> Path:
    path = census_dir / f"year={year}" / f"month={month:02d}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing census schedule partition: {path}")
    return path


def _year_paths(census_dir: Path, year: int) -> tuple[Path, ...]:
    paths = tuple(_partition_path(census_dir, year, month) for month in range(1, 13))
    return paths


def _prepare_schedule(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    result = frame.copy()
    dates = pd.to_datetime(result["FlightDate"], errors="raise").dt.normalize()
    result["FlightDate"] = dates
    elapsed = pd.to_numeric(result["CRSElapsedTime"], errors="coerce")
    distance = pd.to_numeric(result["Distance"], errors="raise")
    invalid = elapsed.isna() | elapsed.le(0.0)
    elapsed = elapsed.where(~invalid, 30.0 + distance.clip(lower=0.0) / 8.0)
    if elapsed.isna().any() or elapsed.le(0.0).any():
        raise ValueError("rotation schedule block time could not be resolved")
    result["CRSElapsedTime"] = elapsed.astype("float32")
    if result["sample_id"].isna().any() or result["sample_id"].duplicated().any():
        raise ValueError("rotation schedule requires unique non-missing sample_id values")
    return result, int(invalid.sum())


def _load_year(
    census_dir: Path,
    year: int,
    *,
    history: bool,
) -> tuple[pd.DataFrame, tuple[Path, ...], int]:
    paths = _year_paths(census_dir, year)
    columns = ROTATION_HISTORY_COLUMNS if history else ROTATION_TARGET_COLUMNS
    parts = [pd.read_parquet(path, columns=list(columns)) for path in paths]
    frame, imputed = _prepare_schedule(pd.concat(parts, ignore_index=True))
    if not frame["FlightDate"].dt.year.eq(year).all():
        raise ValueError(f"rotation census partitions escape target year {year}")
    return frame, paths, imputed


def _read_date_context(
    census_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, tuple[Path, ...], int]:
    periods = pd.period_range(start=start, end=end, freq="M")
    paths = tuple(
        _partition_path(census_dir, int(period.year), int(period.month))
        for period in periods
    )
    parts = [
        pd.read_parquet(path, columns=list(ROTATION_TARGET_COLUMNS)) for path in paths
    ]
    frame, imputed = _prepare_schedule(pd.concat(parts, ignore_index=True))
    selected = frame.loc[frame["FlightDate"].between(start, end)].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"rotation context is empty for {start.date()} through {end.date()}")
    return selected, paths, imputed


def _pooled_hmop_probability(
    frame: pd.DataFrame,
    *,
    outcome: str,
    views: tuple[str, ...],
) -> NDArray[np.float64]:
    columns = [
        f"recent_{view}_{outcome}_rate_{window}d"
        for view in views
        for window in RECENT_WINDOWS_DAYS
    ]
    probabilities = frame.loc[:, columns].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0.0) | (probabilities > 1.0)
    ).any():
        raise ValueError("closed-left HMOP rotation-risk inputs must be in [0, 1]")
    clipped = np.clip(probabilities, 1e-5, 1.0 - 1e-5)
    logits = np.log(clipped) - np.log1p(-clipped)
    pooled = 1.0 / (1.0 + np.exp(-logits.mean(axis=1)))
    return np.asarray(pooled, dtype=np.float64)


def closed_left_disruption_risk(
    schedule: pd.DataFrame,
    *,
    recent_dir: Path,
) -> NDArray[np.float64]:
    """Compute an outcome-free three-state disruption prior for rotation propagation."""

    required = {"FlightDate", "Origin", "Dest", "Reporting_Airline"}
    missing = sorted(required - set(schedule.columns))
    if missing:
        raise ValueError(f"rotation-risk schedule is missing columns: {missing}")
    working = schedule.loc[:, sorted(required)].copy()
    dates = pd.to_datetime(working["FlightDate"], errors="raise").dt.normalize()
    working["FlightDate"] = dates
    working["Year"] = dates.dt.year.astype("int16")
    working["Route"] = (
        working["Origin"].astype("string") + "-" + working["Dest"].astype("string")
    )
    attached = attach_recent_features(
        working,
        recent_dir=recent_dir,
        fallback_strategy="recent_global",
    )
    delay = _pooled_hmop_probability(
        attached,
        outcome="delay",
        views=RISK_DELAY_VIEWS,
    )
    cancellation = _pooled_hmop_probability(
        attached,
        outcome="cancel",
        views=RISK_CANCEL_VIEWS,
    )
    disruption = cancellation + (1.0 - cancellation) * delay
    if not np.isfinite(disruption).all() or (
        (disruption < 0.0) | (disruption > 1.0)
    ).any():
        raise RuntimeError("closed-left HMOP disruption prior is invalid")
    return np.asarray(disruption, dtype=np.float64)


def infer_rotation_period(
    model: LatentRotationGraph,
    target: pd.DataFrame,
    context: pd.DataFrame,
    *,
    timezone_by_airport: dict[str, str],
    context_disruption_risk: ArrayLike | None = None,
) -> tuple[pd.DataFrame, tuple[dict[str, Any], ...]]:
    """Infer each target day from its previous/current/next schedule window."""

    target_prepared, _ = _prepare_schedule(target)
    context_prepared, _ = _prepare_schedule(context)
    target_dates = pd.to_datetime(
        target_prepared["FlightDate"], errors="raise"
    ).dt.normalize()
    context_dates = pd.to_datetime(
        context_prepared["FlightDate"], errors="raise"
    ).dt.normalize()
    if context_disruption_risk is None:
        risk = None
    else:
        risk = np.asarray(context_disruption_risk, dtype=np.float64)
        if risk.shape != (len(context_prepared),) or not np.isfinite(risk).all():
            raise ValueError("context disruption risk must be finite and align with context")
        if ((risk < 0.0) | (risk > 1.0)).any():
            raise ValueError("context disruption risk must be in [0, 1]")
    target_ids = set(target_prepared["sample_id"].astype(str))
    if not target_ids.issubset(set(context_prepared["sample_id"].astype(str))):
        raise ValueError("rotation context omits target flights")

    outputs: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for target_date in sorted(target_dates.unique()):
        timestamp = pd.Timestamp(target_date)
        window_mask = context_dates.between(
            timestamp - pd.Timedelta(days=1),
            timestamp + pd.Timedelta(days=1),
        ).to_numpy()
        window = context_prepared.loc[window_mask].reset_index(drop=True)
        if window.empty:
            raise ValueError(f"empty latent-rotation schedule window for {timestamp.date()}")
        inference = model.infer(
            window,
            timezone_by_airport=timezone_by_airport,
            inbound_disruption_risk=(risk[window_mask] if risk is not None else None),
        )
        window_dates = pd.to_datetime(window["FlightDate"], errors="raise").dt.normalize()
        emit = window_dates.eq(timestamp).to_numpy()
        day = pd.concat(
            [
                window.loc[emit, ["sample_id"]].reset_index(drop=True),
                inference.features.loc[emit].reset_index(drop=True),
            ],
            axis=1,
        )
        expected = int(target_dates.eq(timestamp).sum())
        if len(day) != expected:
            raise AssertionError(
                f"latent-rotation output row mismatch for {timestamp.date()}: "
                f"{len(day)} != {expected}"
            )
        record = dict(inference.diagnostics)
        record.update(
            {
                "FlightDate": timestamp.date().isoformat(),
                "emitted_rows": len(day),
                "window_start": (timestamp - pd.Timedelta(days=1)).date().isoformat(),
                "window_end": (timestamp + pd.Timedelta(days=1)).date().isoformat(),
            }
        )
        diagnostics.append(record)
        outputs.append(day)

    inferred = pd.concat(outputs, ignore_index=True)
    if inferred["sample_id"].duplicated().any() or set(
        inferred["sample_id"].astype(str)
    ) != target_ids:
        raise RuntimeError("latent-rotation period output is not one-to-one with target")
    ordered = target_prepared.loc[:, ["sample_id"]].merge(
        inferred,
        how="left",
        on="sample_id",
        sort=False,
        validate="one_to_one",
    )
    if ordered[list(FLARE24_ROTATION_FEATURES[:-1])].isna().all(axis=1).any():
        raise RuntimeError("latent-rotation structural features are unexpectedly all missing")
    connected = ordered["flare24_rotation_predecessor_probability"].gt(0.0)
    propagated = ordered["flare24_rotation_inbound_disruption_risk"]
    if risk is None and propagated.notna().any():
        raise RuntimeError("structural rotation materialization unexpectedly contains risk")
    if risk is not None and (propagated.loc[connected].isna().any() or not connected.any()):
        raise RuntimeError("risk-aware rotation materialization omitted connected-flight risk")
    return ordered, tuple(diagnostics)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite rotation features: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated rotation partial exists: {partial}")
    frame.to_parquet(partial, index=False, compression="zstd")
    partial.replace(path)
    return {
        "path": path.as_posix(),
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _atomic_joblib(value: Any, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite rotation model: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated rotation model partial exists: {partial}")
    joblib.dump(value, partial)
    partial.replace(path)
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def materialize_rotation_features(
    *,
    census_dir: Path,
    airport_catalog_path: Path,
    output_dir: Path,
    model_dir: Path,
    manifest_path: Path,
    target_years: tuple[int, ...],
    recent_dir: Path | None = None,
    recent_manifest_path: Path | None = None,
    minimum_turn_minutes: float = 20.0,
    maximum_layover_minutes: float = 720.0,
    maximum_candidates: int = 12,
) -> dict[str, Any]:
    """Fit a prior-year rotation kernel and materialize each target year."""

    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite rotation manifest: {manifest_path}")
    if not target_years or any(year not in {2024, 2025} for year in target_years):
        raise ValueError("FLARE-24 rotation target years must be a subset of 2024-2025")
    if (recent_dir is None) != (recent_manifest_path is None):
        raise ValueError("recent_dir and recent_manifest_path must be supplied together")
    if recent_manifest_path is not None:
        recent_payload: dict[str, Any] = json.loads(
            recent_manifest_path.read_text(encoding="utf-8")
        )
        recorded = recent_payload.get("manifest_sha256")
        body = {
            key: value
            for key, value in recent_payload.items()
            if key != "manifest_sha256"
        }
        if recorded != canonical_json_sha256(body):
            raise ValueError("closed-left recent-feature manifest self-hash failed")
    started = time.perf_counter()
    timezones = timezone_catalog(airport_catalog_path)
    outputs: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    source_files: dict[str, dict[str, Any]] = {}
    total_rows = 0
    total_imputed = 0
    for target_year in target_years:
        history_year = target_year - 1
        history, history_paths, history_imputed = _load_year(
            census_dir,
            history_year,
            history=True,
        )
        model = LatentRotationGraph(
            minimum_turn_minutes=minimum_turn_minutes,
            maximum_layover_minutes=maximum_layover_minutes,
            maximum_candidates=maximum_candidates,
        ).fit(history, timezone_by_airport=timezones)
        model_artifact = _atomic_joblib(
            model,
            model_dir / f"latent_rotation_history_{history_year}_for_{target_year}.joblib",
        )
        models.append(
            {
                "target_year": target_year,
                "history_year": history_year,
                "model_card": model.model_card().as_dict(),
                "scheduled_block_imputations": history_imputed,
                **model_artifact,
            }
        )
        for path in history_paths:
            source_files.setdefault(
                f"{path.as_posix()}|history-for-{target_year}",
                {
                    "path": path.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "columns_read": list(ROTATION_HISTORY_COLUMNS),
                    "role": "prior-year tail-supervised fit",
                },
            )

        for month in range(1, 13):
            month_start = pd.Timestamp(year=target_year, month=month, day=1)
            month_end = month_start + pd.offsets.MonthEnd(0)
            target, target_paths, target_imputed = _read_date_context(
                census_dir,
                month_start,
                month_end,
            )
            context_end = min(
                month_end + pd.Timedelta(days=1),
                pd.Timestamp(year=target_year, month=12, day=31),
            )
            context, context_paths, context_imputed = _read_date_context(
                census_dir,
                month_start - pd.Timedelta(days=1),
                context_end,
            )
            context_risk = (
                closed_left_disruption_risk(context, recent_dir=recent_dir)
                if recent_dir is not None
                else None
            )
            features, diagnostics = infer_rotation_period(
                model,
                target,
                context,
                timezone_by_airport=timezones,
                context_disruption_risk=context_risk,
            )
            for feature in FLARE24_ROTATION_FEATURES:
                features[feature] = pd.to_numeric(
                    features[feature], errors="raise"
                ).astype("float32")
            artifact = _atomic_parquet(
                features,
                output_dir / f"year={target_year}" / f"month={month:02d}.parquet",
            )
            outputs.append(
                {
                    "year": target_year,
                    "month": month,
                    "diagnostics": list(diagnostics),
                    **artifact,
                }
            )
            total_rows += len(features)
            total_imputed += target_imputed + context_imputed
            for path in (*target_paths, *context_paths):
                source_files.setdefault(
                    f"{path.as_posix()}|schedule-for-{target_year}",
                    {
                        "path": path.as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "columns_read": list(ROTATION_TARGET_COLUMNS),
                        "role": "target schedule or adjacent-day context; no tail/outcome columns",
                    },
                )
            print(
                f"materialized latent rotations {target_year}-{month:02d}: "
                f"rows={len(features)}",
                flush=True,
            )
        del history, model

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_SCHEDULE_ONLY_LATENT_ROTATION_FEATURES",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "FLARE24-EXPANDING-PRIOR-YEAR-CAPACITATED-LATENT-ROTATIONS",
        "target_years": list(target_years),
        "rows": total_rows,
        "features": list(FLARE24_ROTATION_FEATURES),
        "inbound_risk_source": (
            "closed-left HMOP logit pool propagated through latent predecessor edges"
            if recent_dir is not None
            else "none; structural rotation ablation"
        ),
        "inbound_risk_formula": (
            "cancel_prior + (1-cancel_prior)*delay_prior; each prior is the equal "
            "logit pool over registered 7/28/90-day closed-left group rates"
            if recent_dir is not None
            else None
        ),
        "models": models,
        "source_files": sorted(source_files.values(), key=lambda item: item["path"]),
        "outputs": outputs,
        "target_tail_number_read": False,
        "target_outcome_columns_read": [],
        "historical_tail_number_role": "supervision in the immediately prior year only",
        "scheduled_block_imputations_including_repeated_context_reads": total_imputed,
        "airport_catalog": {
            "path": airport_catalog_path.as_posix(),
            "sha256": sha256_file(airport_catalog_path),
        },
        "recent_feature_manifest": (
            {
                "path": recent_manifest_path.as_posix(),
                "sha256": sha256_file(recent_manifest_path),
            }
            if recent_manifest_path is not None
            else None
        ),
        "recent_lookup_files": (
            [
                {
                    "path": path.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for year in sorted({*target_years, min(target_years) - 1})
                for level in ("global", "route", "airline", "origin", "dest")
                for path in (recent_dir / f"level={level}" / f"year={year}.parquet",)
            ]
            if recent_dir is not None
            else []
        ),
        "environment": {
            "python": platform.python_version(),
            "joblib": version("joblib"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
        },
        "provenance": capture_provenance(
            (
                Path(__file__),
                Path(__file__).with_name("flare_rotation.py"),
                Path(__file__).with_name("contracts.py"),
            )
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "The inferred edge weights are latent schedule-based hypotheses, not observed "
            "target-year aircraft assignments or operational tail-tracking claims."
        ),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-dir", type=Path, required=True)
    parser.add_argument("--airport-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-years", nargs="+", type=int, required=True)
    parser.add_argument("--recent-dir", type=Path)
    parser.add_argument("--recent-manifest", type=Path)
    parser.add_argument("--minimum-turn-minutes", type=float, default=20.0)
    parser.add_argument("--maximum-layover-minutes", type=float, default=720.0)
    parser.add_argument("--maximum-candidates", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = materialize_rotation_features(
        census_dir=args.census_dir,
        airport_catalog_path=args.airport_catalog,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        manifest_path=args.manifest,
        target_years=tuple(args.target_years),
        recent_dir=args.recent_dir,
        recent_manifest_path=args.recent_manifest,
        minimum_turn_minutes=args.minimum_turn_minutes,
        maximum_layover_minutes=args.maximum_layover_minutes,
        maximum_candidates=args.maximum_candidates,
    )
    print(
        json.dumps(
            {
                "manifest": args.manifest.as_posix(),
                "rows": result["rows"],
                "partitions": len(result["outputs"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
