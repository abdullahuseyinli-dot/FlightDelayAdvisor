"""Build leakage-resistant, point-in-time flight features.

Every outcome-derived value for a target year is computed from strictly earlier
calendar years.  Realised weather is retained only under an explicit ``oracle_``
name; deployable weather variables are expanding, prior-year climatologies.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .contracts import (
    AvailabilityHorizon,
    core_point_in_time_features_at,
    validate_predictors,
)
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json

BUILDER_VERSION = 2

GROUP_LEVELS: dict[str, tuple[str, ...]] = {
    "route": ("Route",),
    "airline": ("Reporting_Airline",),
    "origin": ("Origin",),
    "dest": ("Dest",),
    "slot": ("Origin", "Month", "DayOfWeek", "DepHour"),
}

SMOOTHING_STRENGTH: dict[str, float] = {
    "route": 200.0,
    "airline": 1_000.0,
    "origin": 1_000.0,
    "dest": 1_000.0,
    "slot": 300.0,
}

WEATHER_VARIABLES = ("tavg", "prcp", "snow", "wspd")
ORACLE_SOURCE_COLUMNS: dict[str, str] = {
    f"{side}_{variable}": f"oracle_{side.lower()}_{variable}"
    for side in ("Origin", "Dest")
    for variable in WEATHER_VARIABLES
}

BASE_SOURCE_COLUMNS = (
    "sample_id",
    "Year",
    "Month",
    "DayOfMonth",
    "DayOfWeek",
    "DayOfYear",
    "FlightDate",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "Distance",
    "DepHour",
    "IsWeekend",
    "IsHolidaySeason",
    "DepHour_sin",
    "DepHour_cos",
    "Month_sin",
    "Month_cos",
    "Route",
    "DistanceBand",
    "ArrDel15",
    "Cancelled",
)

LEGACY_READ_COLUMNS = tuple(
    dict.fromkeys(
        [
            column
            for column in BASE_SOURCE_COLUMNS
            if column not in {"sample_id", "Month_sin", "Month_cos"}
        ]
        + list(ORACLE_SOURCE_COLUMNS)
    )
)

_OUTCOME_COLUMNS = ("scheduled_count", "delay_support", "delay_sum", "cancel_sum")


def _decay_factor(half_life_years: float | None, elapsed_years: int) -> float:
    if half_life_years is None:
        return 1.0
    if half_life_years <= 0:
        raise ValueError("half_life_years must be positive")
    return math.exp(-math.log(2.0) * elapsed_years / half_life_years)


def _add_tables(left: pd.DataFrame | None, right: pd.DataFrame) -> pd.DataFrame:
    if left is None:
        return right.copy()
    return left.add(right, fill_value=0.0).loc[:, list(right.columns)]


@dataclass(slots=True)
class OutcomeSummary:
    """Sufficient statistics for one or more years of schedule outcomes."""

    global_stats: dict[str, float]
    group_stats: dict[str, pd.DataFrame]


def summarise_outcomes(frame: pd.DataFrame) -> OutcomeSummary:
    """Reduce raw outcomes without retaining individual records."""

    required = {"ArrDel15", "Cancelled"}.union(
        key for keys in GROUP_LEVELS.values() for key in keys
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"cannot summarise outcomes; missing columns: {missing}")

    working = frame.loc[:, sorted(required)].copy()
    cancel = pd.to_numeric(working["Cancelled"], errors="coerce")
    if cancel.isna().any():
        raise ValueError("Cancelled contains missing or non-numeric labels")
    delay = pd.to_numeric(working["ArrDel15"], errors="coerce")
    bad_cancel = ~cancel.isin([0, 1])
    bad_delay = delay.notna() & ~delay.isin([0, 1])
    if bad_cancel.any() or bad_delay.any():
        raise ValueError(
            "outcome labels must be binary; "
            f"bad Cancelled={int(bad_cancel.sum())}, bad ArrDel15={int(bad_delay.sum())}"
        )

    working["scheduled_count"] = 1.0
    delay_eligible = cancel.eq(0) & delay.notna()
    working["delay_support"] = delay_eligible.astype("float64")
    working["delay_sum"] = delay.where(delay_eligible, 0.0).astype("float64")
    working["cancel_sum"] = cancel.astype("float64")
    global_stats = {column: float(working[column].sum()) for column in _OUTCOME_COLUMNS}
    grouped: dict[str, pd.DataFrame] = {}
    for name, keys in GROUP_LEVELS.items():
        grouped[name] = (
            working.groupby(list(keys), observed=True, sort=False)[list(_OUTCOME_COLUMNS)]
            .sum()
            .astype("float64")
        )
    return OutcomeSummary(global_stats=global_stats, group_stats=grouped)


def combine_outcome_summaries(summaries: Iterable[OutcomeSummary]) -> OutcomeSummary:
    """Combine independently accumulated partitions, such as months of one year."""

    global_stats = {column: 0.0 for column in _OUTCOME_COLUMNS}
    group_stats: dict[str, pd.DataFrame | None] = {name: None for name in GROUP_LEVELS}
    seen = False
    for summary in summaries:
        seen = True
        for column in _OUTCOME_COLUMNS:
            global_stats[column] += summary.global_stats[column]
        for name in GROUP_LEVELS:
            group_stats[name] = _add_tables(group_stats[name], summary.group_stats[name])
    if not seen:
        raise ValueError("at least one outcome summary is required")
    return OutcomeSummary(
        global_stats=global_stats,
        group_stats={name: table for name, table in group_stats.items() if table is not None},
    )


@dataclass(slots=True)
class OutcomePriorState:
    """Expanding or exponentially decayed outcome history."""

    half_life_years: float | None = None
    global_stats: dict[str, float] = field(
        default_factory=lambda: {column: 0.0 for column in _OUTCOME_COLUMNS}
    )
    group_stats: dict[str, pd.DataFrame] = field(default_factory=dict)
    current_year: int | None = None

    def advance_to(self, year: int) -> None:
        """Age existing evidence to the prediction year, enforcing chronology."""

        if self.current_year is None:
            self.current_year = year
            return
        if year <= self.current_year:
            raise ValueError(f"year must increase beyond {self.current_year}; got {year}")
        factor = _decay_factor(self.half_life_years, year - self.current_year)
        if factor != 1.0:
            self.global_stats = {
                column: value * factor for column, value in self.global_stats.items()
            }
            for table in self.group_stats.values():
                table.loc[:, list(_OUTCOME_COLUMNS)] *= factor
        self.current_year = year

    def add_summary(self, summary: OutcomeSummary, *, year: int) -> None:
        """Add outcomes after (never before) features for ``year`` were generated."""

        if self.current_year is None:
            self.current_year = year
        if year != self.current_year:
            raise ValueError(
                f"state is positioned at {self.current_year}; cannot add summary for {year}"
            )
        for column in _OUTCOME_COLUMNS:
            self.global_stats[column] += summary.global_stats[column]
        for name, table in summary.group_stats.items():
            self.group_stats[name] = _add_tables(self.group_stats.get(name), table)

    def update(self, frame: pd.DataFrame, *, year: int) -> None:
        self.add_summary(summarise_outcomes(frame), year=year)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Look up smoothed priors using only evidence already in this state."""

        delay_support = self.global_stats["delay_support"]
        scheduled = self.global_stats["scheduled_count"]
        if delay_support <= 0 or scheduled <= 0:
            raise ValueError("outcome prior state has no historical support")
        global_delay = self.global_stats["delay_sum"] / delay_support
        global_cancel = self.global_stats["cancel_sum"] / scheduled

        result = pd.DataFrame(index=frame.index)
        result["prior_global_delay_rate"] = np.float32(global_delay)
        result["prior_global_cancel_rate"] = np.float32(global_cancel)
        result["prior_global_count"] = np.float32(scheduled)

        for name, keys in GROUP_LEVELS.items():
            if name not in self.group_stats:
                raise ValueError(f"outcome state lacks the {name!r} aggregation")
            table = self.group_stats[name]
            if len(keys) == 1:
                lookup_index: pd.Index[Any] = pd.Index(frame[keys[0]], name=keys[0])
            else:
                lookup_index = pd.MultiIndex.from_frame(frame.loc[:, list(keys)])
            matched = table.reindex(lookup_index)
            count = matched["scheduled_count"].fillna(0.0).to_numpy(dtype="float64")
            local_delay_support = matched["delay_support"].fillna(0.0).to_numpy(dtype="float64")
            delay_sum = matched["delay_sum"].fillna(0.0).to_numpy(dtype="float64")
            cancel_sum = matched["cancel_sum"].fillna(0.0).to_numpy(dtype="float64")
            strength = SMOOTHING_STRENGTH[name]
            result[f"prior_{name}_delay_rate"] = (
                (delay_sum + strength * global_delay) / (local_delay_support + strength)
            ).astype("float32")
            result[f"prior_{name}_cancel_rate"] = (
                (cancel_sum + strength * global_cancel) / (count + strength)
            ).astype("float32")
            result[f"prior_{name}_count"] = count.astype("float32")
            result[f"prior_{name}_delay_support"] = local_delay_support.astype("float32")
        return result


def _weather_daily_observations(frame: pd.DataFrame) -> pd.DataFrame:
    sides: list[pd.DataFrame] = []
    for side in ("Origin", "Dest"):
        source_columns = [f"{side}_{variable}" for variable in WEATHER_VARIABLES]
        if not set(source_columns).issubset(frame.columns):
            continue
        side_frame = frame.loc[:, ["FlightDate", "Month", side, *source_columns]].rename(
            columns={
                side: "airport",
                **{f"{side}_{variable}": variable for variable in WEATHER_VARIABLES},
            }
        )
        side_frame = side_frame.dropna(subset=["airport"])
        daily = (
            side_frame.groupby(["airport", "FlightDate", "Month"], observed=True, sort=False)[
                list(WEATHER_VARIABLES)
            ]
            .mean()
            .reset_index()
        )
        sides.append(daily)
    if not sides:
        return pd.DataFrame(columns=["airport", "FlightDate", "Month", *WEATHER_VARIABLES])
    combined = pd.concat(sides, ignore_index=True)
    return (
        combined.groupby(["airport", "FlightDate", "Month"], observed=True, sort=False)[
            list(WEATHER_VARIABLES)
        ]
        .mean()
        .reset_index()
    )


def _summarise_weather(daily: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    grouped = daily.groupby(group_keys, observed=True, sort=False)
    for variable in WEATHER_VARIABLES:
        piece = (
            grouped[variable]
            .agg(["sum", "count"])
            .rename(columns={"sum": f"{variable}_sum", "count": f"{variable}_count"})
        )
        pieces.append(piece.astype("float64"))
    return pd.concat(pieces, axis=1)


@dataclass(slots=True)
class ClimatologyState:
    """Prior-year airport/month weather normals based on unique airport-days."""

    half_life_years: float | None = None
    airport_month: pd.DataFrame | None = None
    month: pd.DataFrame | None = None
    current_year: int | None = None

    @property
    def statistic_columns(self) -> list[str]:
        return [
            f"{variable}_{statistic}"
            for variable in WEATHER_VARIABLES
            for statistic in ("sum", "count")
        ]

    def advance_to(self, year: int) -> None:
        if self.current_year is None:
            self.current_year = year
            return
        if year <= self.current_year:
            raise ValueError(f"year must increase beyond {self.current_year}; got {year}")
        factor = _decay_factor(self.half_life_years, year - self.current_year)
        if factor != 1.0:
            for table in (self.airport_month, self.month):
                if table is not None:
                    table.loc[:, self.statistic_columns] *= factor
        self.current_year = year

    def update(self, frame: pd.DataFrame, *, year: int) -> None:
        if self.current_year is None:
            self.current_year = year
        if year != self.current_year:
            raise ValueError(f"climatology state is positioned at {self.current_year}; got {year}")
        daily = _weather_daily_observations(frame)
        if daily.empty:
            return
        airport_month = _summarise_weather(daily, ["airport", "Month"])
        month = _summarise_weather(daily, ["Month"])
        self.airport_month = _add_tables(self.airport_month, airport_month)
        self.month = _add_tables(self.month, month)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.airport_month is None or self.month is None:
            raise ValueError("climatology state has no historical observations")
        result = pd.DataFrame(index=frame.index)
        month_index = pd.Index(frame["Month"], name="Month")
        month_match = self.month.reindex(month_index)

        for side in ("origin", "dest"):
            airport_column = side.capitalize()
            airport_index = pd.MultiIndex.from_arrays(
                [frame[airport_column], frame["Month"]], names=["airport", "Month"]
            )
            airport_match = self.airport_month.reindex(airport_index)
            for variable in WEATHER_VARIABLES:
                airport_count = airport_match[f"{variable}_count"].to_numpy(dtype="float64")
                airport_sum = airport_match[f"{variable}_sum"].to_numpy(dtype="float64")
                airport_value = np.divide(
                    airport_sum,
                    airport_count,
                    out=np.full(len(frame), np.nan, dtype="float64"),
                    where=np.isfinite(airport_count) & (airport_count > 0),
                )
                month_count = month_match[f"{variable}_count"].to_numpy(dtype="float64")
                month_sum = month_match[f"{variable}_sum"].to_numpy(dtype="float64")
                month_value = np.divide(
                    month_sum,
                    month_count,
                    out=np.full(len(frame), np.nan, dtype="float64"),
                    where=np.isfinite(month_count) & (month_count > 0),
                )
                overall_count = float(self.month[f"{variable}_count"].sum())
                overall_value = (
                    float(self.month[f"{variable}_sum"].sum()) / overall_count
                    if overall_count > 0
                    else 0.0
                )
                missing = ~np.isfinite(airport_value)
                imputed = np.where(
                    missing,
                    np.where(np.isfinite(month_value), month_value, overall_value),
                    airport_value,
                )
                result[f"clim_{side}_{variable}"] = imputed.astype("float32")
                result[f"clim_{side}_{variable}_missing"] = missing.astype("int8")
        return result


def prepare_base(frame: pd.DataFrame, *, source_cohort: str) -> pd.DataFrame:
    """Normalize legacy and current BTS rows to one research schema."""

    required = {
        column
        for column in BASE_SOURCE_COLUMNS
        if column not in {"sample_id", "Month_sin", "Month_cos"}
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"input frame is missing base columns: {missing}")
    base = frame.loc[:, [column for column in BASE_SOURCE_COLUMNS if column in frame]].copy()
    base = base.reset_index(drop=True)
    base["FlightDate"] = pd.to_datetime(base["FlightDate"], errors="raise")
    base["Year"] = pd.to_numeric(base["Year"], errors="raise").astype("int16")
    base["Month"] = pd.to_numeric(base["Month"], errors="raise").astype("int8")
    base["Month_sin"] = np.sin(2 * np.pi * base["Month"] / 12).astype("float32")
    base["Month_cos"] = np.cos(2 * np.pi * base["Month"] / 12).astype("float32")
    if "sample_id" not in base:
        years = base["Year"].astype(str)
        ordinal = pd.Series(np.arange(len(base)), index=base.index).astype(str).str.zfill(7)
        base.insert(0, "sample_id", "legacy-" + years + "-" + ordinal)
    if base["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique within a partition")

    cancelled = pd.to_numeric(base["Cancelled"], errors="coerce")
    delay = pd.to_numeric(base["ArrDel15"], errors="coerce")
    if cancelled.isna().any():
        raise ValueError("Cancelled contains missing labels")
    bad_cancel = ~cancelled.isin([0, 1])
    bad_delay = delay.notna() & ~delay.isin([0, 1])
    if bad_cancel.any() or bad_delay.any():
        raise ValueError(
            "outcome labels must be binary; "
            f"bad Cancelled={int(bad_cancel.sum())}, bad ArrDel15={int(bad_delay.sum())}"
        )
    base["Cancelled"] = cancelled.astype("int8")
    base["ArrDel15"] = delay.astype("float32")
    base.loc[base["Cancelled"].eq(1), "ArrDel15"] = np.nan
    base["delay_label_observed"] = base["ArrDel15"].notna().astype("int8")
    base["joint_label_observed"] = (base["Cancelled"].eq(1) | base["ArrDel15"].notna()).astype(
        "int8"
    )
    base["disruption_state"] = np.select(
        [
            base["Cancelled"].eq(1),
            base["ArrDel15"].eq(1),
            base["ArrDel15"].eq(0),
        ],
        [2, 1, 0],
        default=-1,
    ).astype("int8")
    base["source_cohort"] = source_cohort

    for source, target in ORACLE_SOURCE_COLUMNS.items():
        if source in frame:
            base[target] = pd.to_numeric(frame[source], errors="coerce").astype("float32")
    return base


def transform_point_in_time(
    frame: pd.DataFrame,
    *,
    source_cohort: str,
    outcome_state: OutcomePriorState,
    climatology_state: ClimatologyState,
) -> pd.DataFrame:
    """Create one partition from states that contain earlier years only."""

    base = prepare_base(frame, source_cohort=source_cohort)
    prior = outcome_state.transform(base)
    climatology = climatology_state.transform(base)
    result = pd.concat([base, prior, climatology], axis=1)
    deployable = core_point_in_time_features_at(AvailabilityHorizon.SCHEDULE_CLIMATOLOGY)
    validate_predictors(deployable, AvailabilityHorizon.SCHEDULE_CLIMATOLOGY)
    missing = sorted(set(deployable) - set(result.columns))
    if missing:
        raise ValueError(f"builder failed to create registered deployable features: {missing}")
    if result.loc[:, list(deployable)].isna().any().any():
        null_counts = result.loc[:, list(deployable)].isna().sum()
        offenders = {name: int(count) for name, count in null_counts.items() if count}
        raise ValueError(f"deployable features contain missing values: {offenders}")
    return result


def _load_legacy_year(path: Path, year: int) -> pd.DataFrame:
    table = pq.read_table(path, columns=list(LEGACY_READ_COLUMNS), filters=[("Year", "=", year)])
    frame = table.to_pandas()
    if frame.empty:
        raise ValueError(f"legacy input has no rows for {year}")
    if not frame["Year"].eq(year).all():
        raise ValueError(f"legacy year filter returned out-of-year rows for {year}")
    return frame


def _write_partition(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite derived evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated partial feature file exists: {partial}")
    frame.to_parquet(
        partial,
        engine="pyarrow",
        compression="zstd",
        index=False,
        row_group_size=100_000,
    )
    partial.replace(path)
    cancelled = int(frame["Cancelled"].sum())
    delayed = int(frame["ArrDel15"].fillna(0).sum())
    missing_delay = int((frame["Cancelled"].eq(0) & frame["ArrDel15"].isna()).sum())
    rows = len(frame)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": rows,
        "delays": delayed,
        "cancellations": cancelled,
        "noncancelled_missing_delay_label": missing_delay,
        "joint_label_ineligible": int(frame["joint_label_observed"].eq(0).sum()),
        "delay_rate_non_cancelled": delayed / max(rows - cancelled, 1),
        "cancellation_rate": cancelled / max(rows, 1),
        "first_date": frame["FlightDate"].min().date().isoformat(),
        "last_date": frame["FlightDate"].max().date().isoformat(),
    }


def _write_state(
    path: Path, outcome_state: OutcomePriorState, climatology_state: ClimatologyState
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite fitted feature state: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated partial state exists: {partial}")
    joblib.dump(
        {
            "builder_version": BUILDER_VERSION,
            "outcome_state": outcome_state,
            "climatology_state": climatology_state,
        },
        partial,
        compress=3,
    )
    partial.replace(path)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _source_record(
    path: Path,
    *,
    year: int | None = None,
    years: tuple[int, int] | None = None,
    month: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if year is not None:
        record["year"] = year
    if years is not None:
        record["years"] = list(years)
    if month is not None:
        record["month"] = month
    return record


def build_dataset(
    *,
    legacy_path: Path,
    normalized_2025_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    half_life_years: float | None = None,
    start_year: int = 2011,
    end_year: int = 2025,
) -> dict[str, Any]:
    """Materialize a complete study feature variant without overwriting evidence."""

    if start_year < 2011 or end_year < start_year or end_year > 2025:
        raise ValueError("supported study range is 2011 through 2025")
    if half_life_years is not None and half_life_years <= 0:
        raise ValueError("half_life_years must be positive")
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite feature manifest: {manifest_path}")

    outcome_state = OutcomePriorState(half_life_years=half_life_years)
    climatology_state = ClimatologyState(half_life_years=half_life_years)
    outputs: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = [_source_record(legacy_path, years=(2010, 2024))]

    for year in range(2010, min(end_year, 2024) + 1):
        frame = _load_legacy_year(legacy_path, year)
        if year == 2010:
            outcome_state.update(frame, year=year)
            climatology_state.update(frame, year=year)
            continue
        outcome_state.advance_to(year)
        climatology_state.advance_to(year)
        if year >= start_year:
            featured = transform_point_in_time(
                frame,
                source_cohort="legacy-top100-weather-v1",
                outcome_state=outcome_state,
                climatology_state=climatology_state,
            )
            record = _write_partition(featured, output_dir / f"year={year}.parquet")
            record["year"] = year
            record["source"] = "legacy_sample"
            outputs.append(record)
        outcome_state.update(frame, year=year)
        climatology_state.update(frame, year=year)

    if end_year >= 2025:
        outcome_state.advance_to(2025)
        climatology_state.advance_to(2025)
        monthly_summaries: list[OutcomeSummary] = []
        for month in range(1, 13):
            source = normalized_2025_dir / f"month={month:02d}.parquet"
            if not source.exists():
                raise FileNotFoundError(f"missing normalized 2025 partition: {source}")
            sources.append(_source_record(source, year=2025, month=month))
            frame = pd.read_parquet(source)
            if not frame["Year"].eq(2025).all() or not frame["Month"].eq(month).all():
                raise ValueError(f"normalized partition does not match 2025-{month:02d}")
            featured = transform_point_in_time(
                frame,
                source_cohort="bts-top100-census-2025-v1",
                outcome_state=outcome_state,
                climatology_state=climatology_state,
            )
            record = _write_partition(
                featured, output_dir / "year=2025" / f"month={month:02d}.parquet"
            )
            record.update({"year": 2025, "month": month, "source": "bts_census"})
            outputs.append(record)
            monthly_summaries.append(summarise_outcomes(frame))
        outcome_state.add_summary(combine_outcome_summaries(monthly_summaries), year=2025)

    state_record = _write_state(
        output_dir / f"state_after_{end_year}.joblib", outcome_state, climatology_state
    )
    deployable_features = list(
        core_point_in_time_features_at(AvailabilityHorizon.SCHEDULE_CLIMATOLOGY)
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "builder_version": BUILDER_VERSION,
        "feature_variant": (
            "all_history" if half_life_years is None else f"exponential_decay_{half_life_years:g}y"
        ),
        "half_life_years": half_life_years,
        "strictly_prior_calendar_years": True,
        "warmup_year": 2010,
        "modeling_years": [start_year, end_year],
        "deployable_horizon": AvailabilityHorizon.SCHEDULE_CLIMATOLOGY.name,
        "deployable_features": deployable_features,
        "oracle_features_retained_for_diagnostic_only": list(ORACLE_SOURCE_COLUMNS.values()),
        "smoothing_strengths": SMOOTHING_STRENGTH,
        "weather_unit_of_aggregation": "unique airport-date before airport-month pooling",
        "label_policy": {
            "delay": "ArrDel15 observed and flight non-cancelled",
            "cancellation": "all retained rows",
            "joint": (
                "cancelled rows or non-cancelled rows with ArrDel15 observed; "
                "unknown legacy arrival outcomes are state -1 and ineligible"
            ),
        },
        "sources": sources,
        "outputs": outputs,
        "state": state_record,
        "total_output_rows": sum(int(record["rows"]) for record in outputs),
        "evidence_status": "DERIVED_REPRODUCIBLE_POINT_IN_TIME_FEATURES",
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy",
        type=Path,
        default=Path("data/processed/bts_delay_2010_2024_balanced_research_weather.parquet"),
    )
    parser.add_argument(
        "--normalized-2025-dir",
        type=Path,
        default=Path("data/derived/bts_normalized/2025"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--half-life-years", type=float)
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=2025)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_dataset(
        legacy_path=args.legacy,
        normalized_2025_dir=args.normalized_2025_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        half_life_years=args.half_life_years,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print(
        json.dumps(
            {
                "manifest": args.manifest.as_posix(),
                "variant": manifest["feature_variant"],
                "rows": manifest["total_output_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
