"""Outcome-blind target-day schedule context for the 24-hour proxy track."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .contracts import (
    SCHEDULE_CONTEXT_FEATURES,
    AvailabilityHorizon,
    validate_predictors,
)
from .modeling import _stable_sample, load_feature_year
from .recent import attach_recent_features

SCHEDULE_KEYS = (
    "FlightDate",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "Route",
    "DepHour",
)


def _count(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    return frame.groupby(keys, observed=True, sort=False)["sample_id"].transform("size")


def _mapped_airport_count(
    frame: pd.DataFrame,
    *,
    aggregate_column: str,
    lookup_column: str,
) -> NDArray[np.float64]:
    counts = frame.groupby(["FlightDate", aggregate_column], observed=True, sort=False).size()
    lookup = pd.MultiIndex.from_arrays(
        [frame["FlightDate"].to_numpy(), frame[lookup_column].astype(str).to_numpy()],
        names=["FlightDate", aggregate_column],
    )
    values = counts.reindex(lookup, fill_value=0).to_numpy(dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def attach_schedule_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach features that use schedule rows and never inspect outcome values."""

    validate_predictors(SCHEDULE_CONTEXT_FEATURES, AvailabilityHorizon.FORECAST_24H)
    missing = sorted(set((*SCHEDULE_KEYS, "sample_id")) - set(frame.columns))
    if missing:
        raise ValueError(f"schedule-context input is missing columns: {missing}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("schedule-context input sample_id values must be unique")

    global_day = _count(frame, ["FlightDate"]).to_numpy(dtype=np.float64)
    airline_day = _count(frame, ["FlightDate", "Reporting_Airline"]).to_numpy(dtype=np.float64)
    route_day = _count(frame, ["FlightDate", "Route"]).to_numpy(dtype=np.float64)
    origin_out = _count(frame, ["FlightDate", "Origin"]).to_numpy(dtype=np.float64)
    dest_in = _count(frame, ["FlightDate", "Dest"]).to_numpy(dtype=np.float64)
    origin_in = _mapped_airport_count(
        frame,
        aggregate_column="Dest",
        lookup_column="Origin",
    )
    dest_out = _mapped_airport_count(
        frame,
        aggregate_column="Origin",
        lookup_column="Dest",
    )
    origin_bank = _count(frame, ["FlightDate", "Origin", "DepHour"]).to_numpy(
        dtype=np.float64
    )
    airline_origin = _count(
        frame,
        ["FlightDate", "Reporting_Airline", "Origin"],
    ).to_numpy(dtype=np.float64)
    airline_dest = _count(
        frame,
        ["FlightDate", "Reporting_Airline", "Dest"],
    ).to_numpy(dtype=np.float64)

    def share(
        numerator: NDArray[np.float64],
        denominator: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return np.asarray(
            np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator, dtype=np.float64),
                where=denominator > 0,
            ),
            dtype=np.float64,
        )

    features = pd.DataFrame(
        {
            "schedule_global_day_log1p": np.log1p(global_day),
            "schedule_airline_day_log1p": np.log1p(airline_day),
            "schedule_route_day_log1p": np.log1p(route_day),
            "schedule_origin_outbound_day_log1p": np.log1p(origin_out),
            "schedule_origin_inbound_day_log1p": np.log1p(origin_in),
            "schedule_dest_inbound_day_log1p": np.log1p(dest_in),
            "schedule_dest_outbound_day_log1p": np.log1p(dest_out),
            "schedule_origin_departure_bank_log1p": np.log1p(origin_bank),
            "schedule_airline_origin_day_log1p": np.log1p(airline_origin),
            "schedule_airline_dest_day_log1p": np.log1p(airline_dest),
            "schedule_airline_network_share": share(airline_day, global_day),
            "schedule_route_network_share": share(route_day, global_day),
            "schedule_origin_network_share": share(origin_out, global_day),
            "schedule_dest_network_share": share(dest_in, global_day),
            "schedule_origin_bank_share": share(origin_bank, origin_out),
            "schedule_airline_origin_share": share(airline_origin, origin_out),
            "schedule_airline_dest_share": share(airline_dest, dest_in),
            "schedule_route_origin_share": share(route_day, origin_out),
            "schedule_route_dest_share": share(route_day, dest_in),
            "schedule_origin_flow_imbalance": share(origin_out - origin_in, origin_out + origin_in),
            "schedule_dest_flow_imbalance": share(dest_in - dest_out, dest_in + dest_out),
        },
        index=frame.index,
        dtype="float32",
    )
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError("schedule-context features contain non-finite values")
    return pd.concat([frame, features], axis=1, copy=False)


def load_context_recent_feature_year(
    feature_dir: Path,
    recent_dir: Path,
    year: int,
    *,
    limit: int | None = None,
    seed: int = 20260903,
) -> pd.DataFrame:
    """Compute schedule census features before sampling, then add prior-day history."""

    full = load_feature_year(feature_dir, year, limit=None, seed=seed)
    enriched = attach_schedule_context(full)
    sampled = _stable_sample(enriched, limit, seed=seed + year)
    return attach_recent_features(sampled, recent_dir=recent_dir)


def load_context_recent_feature_years(
    feature_dir: Path,
    recent_dir: Path,
    years: tuple[int, ...],
    *,
    rows_per_year: int | None = None,
    seed: int = 20260903,
) -> pd.DataFrame:
    if not years:
        raise ValueError("at least one schedule-context year is required")
    return pd.concat(
        [
            load_context_recent_feature_year(
                feature_dir,
                recent_dir,
                year,
                limit=rows_per_year,
                seed=seed,
            )
            for year in years
        ],
        ignore_index=True,
    )


def schedule_context_profile() -> dict[str, object]:
    return {
        "availability_horizon": AvailabilityHorizon.FORECAST_24H.name,
        "construction": "full target-day cohort schedule census before model sampling",
        "outcomes_read_by_transform": False,
        "claim_limit": (
            "Retrospective cohort-density proxy only: the 2011-2024 legacy source is a "
            "research sample and the 2025 cohort excludes diverted flights. It must be "
            "validated against a true advance schedule feed before operational use."
        ),
        "features": list(SCHEDULE_CONTEXT_FEATURES),
    }
