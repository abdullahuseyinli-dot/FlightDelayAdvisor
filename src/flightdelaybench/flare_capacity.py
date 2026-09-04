"""Capacity-conditioned Resource-Time Flight Hypergraph (CC-RTH).

The runnable implementation uses only schedule fields, a strictly prior-year
empirical scheduling frontier, cycle-dated FAA NASR geometry, and weather known
at the FLARE-24 cutoff.  The frontier is a relative BTS scheduling envelope,
not a claim about FAA-declared AAR/ADR or realised runway configuration.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .flare_capacity_contracts import CAPACITY_ALL_FEATURES, WINDOWS_MINUTES
from .flare_weather import scheduled_flight_times

BUCKET_MINUTES = 15
SCENARIOS = ("constrained", "marginal", "good")
QUANTILES = ("p50", "p75", "p90")

METRO_GROUPS: dict[str, tuple[str, ...]] = {
    "CHI": ("MDW", "ORD"),
    "DFW": ("DAL", "DFW"),
    "HOU": ("HOU", "IAH"),
    "LAX_BASIN": ("BUR", "LAX", "LGB", "ONT", "SNA"),
    "NYC": ("EWR", "JFK", "LGA"),
    "SF_BAY": ("OAK", "SFO", "SJC"),
    "SOUTH_FL": ("FLL", "MIA", "PBI"),
    "WAS": ("BWI", "DCA", "IAD"),
}
AIRPORT_TO_METRO = {
    airport: metro for metro, airports in METRO_GROUPS.items() for airport in airports
}

FORBIDDEN_OPERATIONAL_COLUMNS = frozenset(
    {
        "ActualElapsedTime",
        "AirTime",
        "ArrDelay",
        "ArrDel15",
        "Cancelled",
        "CancellationCode",
        "DepDelay",
        "DepDel15",
        "Diverted",
        "TaxiIn",
        "TaxiOut",
        "WheelsOff",
        "WheelsOn",
    }
)


@dataclass(frozen=True, slots=True)
class CapacityConfig:
    """Predeclared mechanics for relative capacity scenarios."""

    bucket_minutes: int = BUCKET_MINUTES
    crosswind_feasible_knots: float = 20.0
    tailwind_feasible_knots: float = 7.0
    crosswind_soft_scale_knots: float = 12.0
    tailwind_soft_scale_knots: float = 5.0
    convection_scale: float = 5.0
    precipitation_scale: float = 3.0
    crosswind_stress_scale_knots: float = 25.0
    gust_stress_scale_knots: float = 25.0
    scenario_logit_scale: float = 2.0
    shadow_price_temperature: float = 0.10

    def __post_init__(self) -> None:
        values = (
            self.crosswind_feasible_knots,
            self.tailwind_feasible_knots,
            self.crosswind_soft_scale_knots,
            self.tailwind_soft_scale_knots,
            self.convection_scale,
            self.precipitation_scale,
            self.crosswind_stress_scale_knots,
            self.gust_stress_scale_knots,
            self.scenario_logit_scale,
            self.shadow_price_temperature,
        )
        if self.bucket_minutes != BUCKET_MINUTES:
            raise ValueError("CC-RTH v1 requires 15-minute resource buckets")
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("capacity configuration values must be finite and positive")


@dataclass(slots=True)
class CapacityHypergraphResult:
    """Flight features plus the explicit sparse hypergraph tables."""

    features: pd.DataFrame
    flight_nodes: pd.DataFrame
    resource_nodes: pd.DataFrame
    incidence_edges: pd.DataFrame
    rotation_edges: pd.DataFrame
    frontier: pd.DataFrame
    diagnostics: dict[str, Any]


def _node_id(resource_type: str, resource_key: str, bucket: pd.Timestamp) -> str:
    token = f"{resource_type}|{resource_key}|{bucket.isoformat()}".encode()
    return "r_" + hashlib.sha256(token).hexdigest()[:24]


def resolve_scheduled_elapsed(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Resolve missing block time from scheduled distance without using outcomes."""

    result = frame.copy()
    elapsed = pd.to_numeric(result["CRSElapsedTime"], errors="coerce")
    invalid = elapsed.isna() | elapsed.le(0.0)
    if invalid.any():
        if "Distance" not in result:
            raise ValueError("invalid scheduled block time requires Distance for imputation")
        distance = pd.to_numeric(result["Distance"], errors="coerce")
        elapsed = elapsed.where(~invalid, 30.0 + distance.clip(lower=0.0) / 8.0)
    if elapsed.isna().any() or elapsed.le(0.0).any():
        raise ValueError("scheduled block time could not be resolved")
    result["CRSElapsedTime"] = elapsed.astype("float32")
    return result, int(invalid.sum())


def prepare_scheduled_events(
    frame: pd.DataFrame,
    *,
    timezone_by_airport: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Create arrival/departure event nodes without consulting outcomes."""

    required = {
        "sample_id",
        "FlightDate",
        "Origin",
        "Dest",
        "CRSDepMinutes",
        "CRSElapsedTime",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"capacity schedule is missing columns: {missing}")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("capacity schedule requires unique non-missing sample_id values")
    schedule_columns = sorted(required | ({"Distance"} & set(frame.columns)))
    schedule, imputed = resolve_scheduled_elapsed(frame.loc[:, schedule_columns])
    times = scheduled_flight_times(schedule, timezone_by_airport=timezone_by_airport)
    if times[["departure_time_utc", "arrival_time_utc", "cutoff_time_utc"]].isna().any().any():
        raise ValueError("capacity schedule contains unresolved timestamps")

    flights = pd.DataFrame(
        {
            "flight_row": np.arange(len(frame), dtype=np.int64),
            "sample_id": frame["sample_id"].astype("string").to_numpy(),
            "origin": frame["Origin"].astype("string").to_numpy(),
            "dest": frame["Dest"].astype("string").to_numpy(),
            "departure_time_utc": times["departure_time_utc"].to_numpy(),
            "arrival_time_utc": times["arrival_time_utc"].to_numpy(),
            "cutoff_time_utc": times["cutoff_time_utc"].to_numpy(),
        }
    )
    departure = pd.DataFrame(
        {
            "flight_row": flights["flight_row"],
            "sample_id": flights["sample_id"],
            "side": "origin",
            "role": "origin_departure",
            "airport": flights["origin"],
            "direction": "departure",
            "event_time_utc": flights["departure_time_utc"],
            "cutoff_time_utc": flights["cutoff_time_utc"],
        }
    )
    arrival = pd.DataFrame(
        {
            "flight_row": flights["flight_row"],
            "sample_id": flights["sample_id"],
            "side": "dest",
            "role": "destination_arrival",
            "airport": flights["dest"],
            "direction": "arrival",
            "event_time_utc": flights["arrival_time_utc"],
            "cutoff_time_utc": flights["cutoff_time_utc"],
        }
    )
    events = pd.concat([departure, arrival], ignore_index=True)
    events["event_time_utc"] = pd.to_datetime(events["event_time_utc"], errors="raise")
    events["cutoff_time_utc"] = pd.to_datetime(events["cutoff_time_utc"], errors="raise")
    events["bucket_time_utc"] = events["event_time_utc"].dt.floor(f"{BUCKET_MINUTES}min")
    events["metro"] = events["airport"].map(AIRPORT_TO_METRO).astype("string")
    return flights, events, imputed


def _local_calendar(
    frame: pd.DataFrame,
    *,
    time_column: str,
    airport_column: str,
    timezone_by_airport: dict[str, str],
) -> pd.DataFrame:
    local_date = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    local_hour = pd.Series(-1, index=frame.index, dtype="int16")
    local_minute = pd.Series(-1, index=frame.index, dtype="int16")
    airports = frame[airport_column].astype("string")
    missing = sorted(set(airports.dropna().astype(str)) - set(timezone_by_airport))
    if missing:
        raise ValueError(f"capacity timezones are missing airports: {missing}")
    utc = pd.to_datetime(frame[time_column], errors="raise", utc=True)
    for airport, indexes in airports.groupby(airports, sort=False).groups.items():
        local = pd.DatetimeIndex(utc.loc[indexes]).tz_convert(timezone_by_airport[str(airport)])
        local_date.loc[indexes] = local.tz_localize(None).normalize().to_numpy()
        local_hour.loc[indexes] = local.hour.astype("int16")
        local_minute.loc[indexes] = local.minute.astype("int16")
    if local_date.isna().any() or local_hour.lt(0).any():
        raise RuntimeError("capacity local calendar conversion failed")
    month = local_date.dt.month.astype("int8")
    return pd.DataFrame(
        {
            "local_date": local_date,
            "local_hour": local_hour.astype("int8"),
            "local_minute": local_minute.astype("int8"),
            "local_season": (((month - 1) // 3) + 1).astype("int8"),
        },
        index=frame.index,
    )


def _centered_counts(
    query_times: NDArray[np.int64],
    source_times: NDArray[np.int64],
    window_minutes: int,
) -> NDArray[np.float64]:
    half_window = int(window_minutes * 60 * 1_000_000_000 / 2)
    left = np.searchsorted(source_times, query_times - half_window, side="left")
    right = np.searchsorted(source_times, query_times + half_window, side="right")
    return np.asarray(right - left, dtype=np.float64)


def attach_event_demand(events: pd.DataFrame) -> pd.DataFrame:
    """Attach multi-scale same-airport and metro schedule pressure."""

    result = events.copy().reset_index(drop=True)
    count_columns = [
        f"{direction}_demand_{window}m"
        for direction in ("arrival", "departure", "movement")
        for window in WINDOWS_MINUTES
    ]
    arrays = {name: np.zeros(len(result), dtype=np.float64) for name in count_columns}
    event_ns = result["event_time_utc"].astype("int64").to_numpy()
    for _, indexes in result.groupby("airport", sort=False, observed=True).groups.items():
        positions = np.asarray(indexes, dtype=np.int64)
        query = event_ns[positions]
        directions = result.loc[positions, "direction"].astype(str).to_numpy()
        sources = {
            "arrival": np.sort(query[directions == "arrival"]),
            "departure": np.sort(query[directions == "departure"]),
            "movement": np.sort(query),
        }
        for direction, source in sources.items():
            for window in WINDOWS_MINUTES:
                arrays[f"{direction}_demand_{window}m"][positions] = _centered_counts(
                    query, source, window
                )

    for name, values in arrays.items():
        result[name] = values
    same_direction_30 = np.where(
        result["direction"].eq("arrival"),
        result["arrival_demand_30m"],
        result["departure_demand_30m"],
    )
    result["same_direction_loo_30m"] = np.maximum(same_direction_30 - 1.0, 0.0)
    result["movement_loo_30m"] = np.maximum(result["movement_demand_30m"] - 1.0, 0.0)

    ordered = result.sort_values(
        ["airport", "direction", "event_time_utc", "sample_id"], kind="mergesort"
    )
    group = ordered.groupby(["airport", "direction"], sort=False, observed=True)
    previous = group["event_time_utc"].shift(1)
    following = group["event_time_utc"].shift(-1)
    ordered["minutes_since_previous_same_direction"] = (
        (ordered["event_time_utc"] - previous).dt.total_seconds() / 60.0
    ).fillna(1_440.0)
    ordered["minutes_until_next_same_direction"] = (
        (following - ordered["event_time_utc"]).dt.total_seconds() / 60.0
    ).fillna(1_440.0)
    gaps = ordered.loc[
        :,
        [
            "minutes_since_previous_same_direction",
            "minutes_until_next_same_direction",
        ],
    ].sort_index()
    result[gaps.columns] = gaps
    total_60 = result["arrival_demand_60m"] + result["departure_demand_60m"]
    result["flow_imbalance_60m"] = np.divide(
        result["departure_demand_60m"] - result["arrival_demand_60m"],
        total_60,
        out=np.zeros(len(result), dtype=np.float64),
        where=total_60.to_numpy() > 0.0,
    )
    result["burstiness_15_to_60"] = np.divide(
        4.0 * result["movement_demand_15m"],
        result["movement_demand_60m"],
        out=np.zeros(len(result), dtype=np.float64),
        where=result["movement_demand_60m"].to_numpy() > 0.0,
    )

    result["metro_movement_demand_60m"] = 0.0
    metro_events = result.loc[result["metro"].notna()]
    for _, indexes in metro_events.groupby("metro", sort=False, observed=True).groups.items():
        positions = np.asarray(indexes, dtype=np.int64)
        query = event_ns[positions]
        all_times = np.sort(query)
        metro_counts = _centered_counts(query, all_times, 60)
        own_counts = result.loc[positions, "movement_demand_60m"].to_numpy(dtype=np.float64)
        result.loc[positions, "metro_movement_demand_60m"] = np.maximum(
            metro_counts - own_counts, 0.0
        )
    return result


def _padded_quantiles(
    values: NDArray[np.float64], expected_bins: int
) -> tuple[float, float, float, float]:
    if expected_bins < len(values):
        raise ValueError("observed frontier bins exceed expected bins")
    padded = np.concatenate([np.zeros(expected_bins - len(values), dtype=np.float64), values])
    quantiles = np.quantile(padded, [0.50, 0.75, 0.90], method="linear")
    observed_fraction = float(len(values) / expected_bins) if expected_bins else 0.0
    return float(quantiles[0]), float(quantiles[1]), float(quantiles[2]), observed_fraction


def fit_schedule_frontier(
    history_schedule: pd.DataFrame,
    *,
    timezone_by_airport: dict[str, str],
    resource_airports: Collection[str] | None = None,
) -> pd.DataFrame:
    """Fit a prior-year airport/direction/season/hour scheduling envelope."""

    _, events, _ = prepare_scheduled_events(
        history_schedule,
        timezone_by_airport=timezone_by_airport,
    )
    if resource_airports is not None:
        resource_set = {str(airport) for airport in resource_airports}
        if not resource_set:
            raise ValueError("resource_airports must not be empty")
        events = events.loc[events["airport"].astype(str).isin(resource_set)].reset_index(drop=True)
        if events.empty:
            raise ValueError("no history events remain in the requested resource airports")
    calendar = _local_calendar(
        events,
        time_column="event_time_utc",
        airport_column="airport",
        timezone_by_airport=timezone_by_airport,
    )
    events = pd.concat([events, calendar], axis=1)
    # FAA capacity is normally expressed hourly.  Estimate an hourly schedule
    # frontier across prior dates, then convert to service per 15-minute bucket.
    # This avoids treating an unused quarter within an otherwise busy hour as
    # zero physical service capacity.
    counts = (
        events.groupby(
            [
                "airport",
                "direction",
                "local_date",
                "local_season",
                "local_hour",
            ],
            sort=False,
            observed=True,
        )
        .size()
        .rename("demand")
        .reset_index()
    )
    movement = (
        events.groupby(
            ["airport", "local_date", "local_season", "local_hour"],
            sort=False,
            observed=True,
        )
        .size()
        .rename("demand")
        .reset_index()
    )
    movement["direction"] = "movement"
    counts = pd.concat([counts, movement], ignore_index=True)

    # Arrival events can legitimately fall on the day after the final scheduled
    # departure date.  The padding denominator therefore follows event-local
    # dates rather than origin FlightDate.
    date_range = pd.date_range(calendar["local_date"].min(), calendar["local_date"].max(), freq="D")
    season_days = pd.Series(((date_range.month - 1) // 3) + 1).value_counts().to_dict()
    airports = sorted(events["airport"].astype(str).unique())
    records: list[dict[str, Any]] = []
    grouped = {
        (str(airport), str(direction), int(season), int(hour)): group["demand"].to_numpy(
            dtype=np.float64
        )
        for (airport, direction, season, hour), group in counts.groupby(
            ["airport", "direction", "local_season", "local_hour"],
            sort=False,
            observed=True,
        )
    }
    for airport in airports:
        for direction in ("arrival", "departure", "movement"):
            for season in sorted(season_days):
                support_days = int(season_days[season])
                for hour in range(24):
                    values = grouped.get(
                        (airport, direction, int(season), hour),
                        np.zeros(0, dtype=np.float64),
                    )
                    p50, p75, p90, observed_fraction = _padded_quantiles(values, support_days)
                    buckets_per_hour = 60 // BUCKET_MINUTES
                    records.append(
                        {
                            "airport": airport,
                            "direction": direction,
                            "local_season": int(season),
                            "local_hour": hour,
                            "frontier_p50": p50 / buckets_per_hour,
                            "frontier_p75": p75 / buckets_per_hour,
                            "frontier_p90": p90 / buckets_per_hour,
                            "support_days": support_days,
                            "observed_fraction": observed_fraction,
                        }
                    )
    frontier = pd.DataFrame.from_records(records)
    if (
        frontier.empty
        or frontier.duplicated(["airport", "direction", "local_season", "local_hour"]).any()
    ):
        raise RuntimeError("prior-year scheduling frontier is invalid")
    for quantile in QUANTILES:
        frontier[f"frontier_{quantile}"] = frontier[f"frontier_{quantile}"].astype("float32")
    return frontier


def _frontier_view(frontier: pd.DataFrame, direction: str, prefix: str) -> pd.DataFrame:
    selected = frontier.loc[frontier["direction"].eq(direction)].copy()
    return selected.rename(
        columns={
            "frontier_p50": f"{prefix}_frontier_p50",
            "frontier_p75": f"{prefix}_frontier_p75",
            "frontier_p90": f"{prefix}_frontier_p90",
            "support_days": f"{prefix}_support_days",
            "observed_fraction": f"{prefix}_observed_fraction",
        }
    ).drop(columns="direction")


def _queue_series(
    demand: NDArray[np.float64], capacity: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    queue = np.zeros(len(demand), dtype=np.float64)
    previous = 0.0
    for index, (arrivals, service) in enumerate(zip(demand, capacity, strict=True)):
        previous = max(0.0, previous + float(arrivals) - max(float(service), 0.0))
        queue[index] = previous
    recovery = np.zeros(len(queue), dtype=np.float64)
    next_zero = -1
    for index in range(len(queue) - 1, -1, -1):
        if queue[index] <= 1e-9:
            next_zero = index
        elif next_zero >= 0:
            recovery[index] = (next_zero - index) * BUCKET_MINUTES
        else:
            recovery[index] = 24.0 * 60.0
    return queue, recovery


def build_resource_grid(
    events: pd.DataFrame,
    frontier: pd.DataFrame,
    *,
    timezone_by_airport: dict[str, str],
) -> pd.DataFrame:
    """Build shared airport-time nodes and scenario queue trajectories."""

    airports = sorted(events["airport"].astype(str).unique())
    start = events["bucket_time_utc"].min().floor("D") - pd.Timedelta(days=1)
    end = events["bucket_time_utc"].max().ceil("D") + pd.Timedelta(days=1)
    buckets = pd.date_range(start, end, freq=f"{BUCKET_MINUTES}min", inclusive="left")
    grid = pd.MultiIndex.from_product(
        [airports, buckets], names=["airport", "bucket_time_utc"]
    ).to_frame(index=False)
    observed = (
        events.groupby(["airport", "bucket_time_utc", "direction"], observed=True)
        .size()
        .unstack("direction", fill_value=0)
        .rename(columns={"arrival": "arrival_demand", "departure": "departure_demand"})
        .reset_index()
    )
    grid = grid.merge(
        observed,
        on=["airport", "bucket_time_utc"],
        how="left",
        validate="one_to_one",
    )
    for column in ("arrival_demand", "departure_demand"):
        if column not in grid:
            grid[column] = 0.0
        grid[column] = grid[column].fillna(0.0).astype("float32")
    grid["movement_demand"] = grid["arrival_demand"] + grid["departure_demand"]
    calendar = _local_calendar(
        grid,
        time_column="bucket_time_utc",
        airport_column="airport",
        timezone_by_airport=timezone_by_airport,
    )
    grid = pd.concat([grid, calendar], axis=1)
    keys = ["airport", "local_season", "local_hour"]
    for direction in ("arrival", "departure", "movement"):
        view = _frontier_view(frontier, direction, direction)
        grid = grid.merge(view, on=keys, how="left", validate="many_to_one")
        value_columns = {
            f"{direction}_frontier_p50": "median",
            f"{direction}_frontier_p75": "median",
            f"{direction}_frontier_p90": "median",
            f"{direction}_support_days": "max",
            f"{direction}_observed_fraction": "median",
        }
        fallback = (
            view.groupby(["airport", "local_hour"], observed=True, sort=False)
            .agg(value_columns)
            .reset_index()
            .rename(columns={column: f"{column}_fallback" for column in value_columns})
        )
        grid = grid.merge(
            fallback,
            on=["airport", "local_hour"],
            how="left",
            validate="many_to_one",
        )
        for column in value_columns:
            grid[column] = grid[column].fillna(grid[f"{column}_fallback"])
            grid = grid.drop(columns=f"{column}_fallback")
    frontier_columns = [column for column in grid if "_frontier_" in column]
    if grid[frontier_columns].isna().any().any():
        missing_airports = sorted(
            grid.loc[grid[frontier_columns].isna().any(axis=1), "airport"].unique()
        )
        raise ValueError(f"frontier omits target airport-time cells: {missing_airports}")

    for direction in ("arrival", "departure", "movement"):
        for scenario, quantile in zip(SCENARIOS, QUANTILES, strict=True):
            queue_name = f"{direction}_queue_{scenario}"
            recovery_name = f"{direction}_recovery_{scenario}"
            grid[queue_name] = 0.0
            grid[recovery_name] = 0.0
            for _, indexes in grid.groupby(
                ["airport", "local_date"], sort=False, observed=True
            ).groups.items():
                positions = np.asarray(indexes, dtype=np.int64)
                demand = grid.loc[positions, f"{direction}_demand"].to_numpy(dtype=np.float64)
                capacity = grid.loc[positions, f"{direction}_frontier_{quantile}"].to_numpy(
                    dtype=np.float64
                )
                queue, recovery = _queue_series(demand, capacity)
                grid.loc[positions, queue_name] = queue
                grid.loc[positions, recovery_name] = recovery
    grid["resource_type"] = "airport_runway_system"
    grid["resource_key"] = grid["airport"].astype(str)
    grid["resource_node_id"] = [
        _node_id("airport_runway_system", str(airport), pd.Timestamp(bucket))
        for airport, bucket in zip(grid["airport"], grid["bucket_time_utc"], strict=True)
    ]
    return grid


def _metro_grid(resource_grid: pd.DataFrame) -> pd.DataFrame:
    member = resource_grid.loc[resource_grid["airport"].isin(AIRPORT_TO_METRO)].copy()
    if member.empty:
        return pd.DataFrame()
    member["metro"] = member["airport"].map(AIRPORT_TO_METRO)
    aggregate_columns = [
        "arrival_demand",
        "departure_demand",
        "movement_demand",
        *[f"movement_frontier_{quantile}" for quantile in QUANTILES],
    ]
    metro = (
        member.groupby(["metro", "bucket_time_utc"], sort=False, observed=True)[aggregate_columns]
        .sum()
        .reset_index()
    )
    metro["service_date_utc"] = pd.to_datetime(
        metro["bucket_time_utc"], errors="raise"
    ).dt.normalize()
    for scenario, quantile in zip(SCENARIOS, QUANTILES, strict=True):
        metro[f"movement_queue_{scenario}"] = 0.0
        metro[f"movement_recovery_{scenario}"] = 0.0
        for _, indexes in metro.groupby(
            ["metro", "service_date_utc"], sort=False, observed=True
        ).groups.items():
            positions = np.asarray(indexes, dtype=np.int64)
            queue, recovery = _queue_series(
                metro.loc[positions, "movement_demand"].to_numpy(dtype=np.float64),
                metro.loc[positions, f"movement_frontier_{quantile}"].to_numpy(dtype=np.float64),
            )
            metro.loc[positions, f"movement_queue_{scenario}"] = queue
            metro.loc[positions, f"movement_recovery_{scenario}"] = recovery
    metro["resource_type"] = "metro_schedule_coupling"
    metro["resource_key"] = metro["metro"]
    metro["resource_node_id"] = [
        _node_id("metro_schedule_coupling", str(name), pd.Timestamp(bucket))
        for name, bucket in zip(metro["metro"], metro["bucket_time_utc"], strict=True)
    ]
    return metro


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return np.asarray(exponential / exponential.sum(axis=1, keepdims=True), dtype=np.float64)


def _endpoint_values(frame: pd.DataFrame, events: pd.DataFrame, suffix: str) -> NDArray[np.float64]:
    values = np.full(len(events), np.nan, dtype=np.float64)
    for side, prefix in (("origin", "flare24_origin"), ("dest", "flare24_dest")):
        column = f"{prefix}_{suffix}"
        if column not in frame:
            continue
        mask = events["side"].eq(side).to_numpy()
        rows = events.loc[mask, "flight_row"].to_numpy(dtype=np.int64)
        values[mask] = pd.to_numeric(frame.iloc[rows][column], errors="coerce").to_numpy(
            dtype=np.float64
        )
    return values


def _static_and_runway_state(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    resource_catalog: dict[str, dict[str, Any]],
    *,
    config: CapacityConfig,
) -> pd.DataFrame:
    output = pd.DataFrame(index=events.index)
    static_fields = {
        "physical_runway_count": "physical_runway_count",
        "eligible_runway_end_count": "eligible_runway_end_count",
        "orientation_family_count": "orientation_family_count",
        "max_parallel_runways": "max_parallel_runways",
        "ils_end_fraction": "ils_end_fraction",
    }
    for name, source in static_fields.items():
        output[name] = events["airport"].map(
            lambda airport, field=source: resource_catalog.get(str(airport), {}).get(field)
        )
    output["minimum_runway_length_kft"] = (
        events["airport"].map(
            lambda airport: resource_catalog.get(str(airport), {}).get("minimum_runway_length_ft")
        )
        / 1_000.0
    )
    output["maximum_runway_length_kft"] = (
        events["airport"].map(
            lambda airport: resource_catalog.get(str(airport), {}).get("maximum_runway_length_ft")
        )
        / 1_000.0
    )
    annual = events["airport"].map(
        lambda airport: resource_catalog.get(str(airport), {}).get("annual_operations")
    )
    output["airport_annual_operations_log1p"] = np.log1p(pd.to_numeric(annual, errors="coerce"))
    output["runway_catalog_missing"] = events["airport"].map(
        lambda airport: float(str(airport) not in resource_catalog)
    )

    wind_direction = _endpoint_values(frame, events, "departure_wind_direction")
    wind_speed = _endpoint_values(frame, events, "departure_wind_speed")
    wind_gust = _endpoint_values(frame, events, "departure_wind_gust")
    dest_mask = events["side"].eq("dest").to_numpy()
    dest_direction = _endpoint_values(frame, events, "arrival_wind_direction")
    dest_speed = _endpoint_values(frame, events, "arrival_wind_speed")
    dest_gust = _endpoint_values(frame, events, "arrival_wind_gust")
    wind_direction[dest_mask] = dest_direction[dest_mask]
    wind_speed[dest_mask] = dest_speed[dest_mask]
    wind_gust[dest_mask] = dest_gust[dest_mask]

    entropy = np.full(len(events), np.nan, dtype=np.float64)
    effective = np.full(len(events), np.nan, dtype=np.float64)
    feasible = np.full(len(events), np.nan, dtype=np.float64)
    for airport, indexes in events.groupby("airport", sort=False, observed=True).groups.items():
        positions = np.asarray(indexes, dtype=np.int64)
        record = resource_catalog.get(str(airport))
        if record is None:
            continue
        headings = np.asarray(
            [
                float(end["heading_true_degrees"])
                for runway in record.get("runways", [])
                for end in runway.get("ends", [])
            ],
            dtype=np.float64,
        )
        if headings.size == 0:
            continue
        valid = (
            np.isfinite(wind_direction[positions])
            & np.isfinite(wind_speed[positions])
            & np.isfinite(wind_gust[positions])
        )
        if (~valid).any():
            entropy[positions[~valid]] = 1.0
            effective[positions[~valid]] = float(headings.size)
        if not valid.any():
            continue
        chosen = positions[valid]
        angle = np.deg2rad(wind_direction[chosen, None] - headings[None, :])
        gust = np.maximum(wind_gust[chosen], wind_speed[chosen])[:, None]
        headwind = gust * np.cos(angle)
        crosswind = np.abs(gust * np.sin(angle))
        tailwind = np.maximum(-headwind, 0.0)
        scores = -crosswind / config.crosswind_soft_scale_knots
        scores -= tailwind / config.tailwind_soft_scale_knots
        probability = _softmax(scores)
        raw_entropy = -(probability * np.log(np.clip(probability, 1e-12, 1.0))).sum(axis=1)
        entropy[chosen] = raw_entropy / max(math.log(headings.size), 1.0)
        effective[chosen] = np.exp(raw_entropy)
        feasible[chosen] = (
            (crosswind <= config.crosswind_feasible_knots)
            & (tailwind <= config.tailwind_feasible_knots)
        ).mean(axis=1)
    output["runway_configuration_entropy"] = entropy
    output["runway_effective_end_count"] = effective
    output["runway_feasible_end_fraction"] = feasible
    return output


def _weather_stress(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    *,
    config: CapacityConfig,
) -> NDArray[np.float64]:
    convection = _endpoint_values(frame, events, "convective_index")
    icing = _endpoint_values(frame, events, "icing_environment_index")
    crosswind = _endpoint_values(frame, events, "wind_optimal_gust_crosswind_knots")
    gust = _endpoint_values(frame, events, "gust_excess_knots")
    precipitation = _endpoint_values(frame, events, "departure_precipitation")
    dest_mask = events["side"].eq("dest").to_numpy()
    dest_precipitation = _endpoint_values(frame, events, "arrival_precipitation")
    precipitation[dest_mask] = dest_precipitation[dest_mask]
    visibility = _endpoint_values(frame, events, "visibility_hazard")
    components = np.column_stack(
        [
            1.0 - np.exp(-np.maximum(convection, 0.0) / config.convection_scale),
            np.clip(icing, 0.0, 1.0),
            np.clip(crosswind / config.crosswind_stress_scale_knots, 0.0, 1.0),
            np.clip(gust / config.gust_stress_scale_knots, 0.0, 1.0),
            1.0 - np.exp(-np.maximum(precipitation, 0.0) / config.precipitation_scale),
            np.clip(visibility, 0.0, 1.0),
        ]
    )
    present = np.isfinite(components)
    numerator = np.nansum(components, axis=1)
    denominator = present.sum(axis=1)
    return np.asarray(
        np.divide(
            numerator,
            denominator,
            out=np.full(len(events), np.nan, dtype=np.float64),
            where=denominator > 0,
        ),
        dtype=np.float64,
    )


def _scenario_probabilities(
    stress: NDArray[np.float64],
    feasible: NDArray[np.float64],
    *,
    config: CapacityConfig,
) -> NDArray[np.float64]:
    weather = np.where(np.isfinite(stress), np.clip(stress, 0.0, 1.0), 0.5)
    runway_penalty = np.where(np.isfinite(feasible), 1.0 - np.clip(feasible, 0.0, 1.0), 0.5)
    severity = np.clip(0.75 * weather + 0.25 * runway_penalty, 0.0, 1.0)
    logits = config.scenario_logit_scale * np.column_stack(
        [severity, 1.0 - np.abs(severity - 0.5) * 2.0, 1.0 - severity]
    )
    return _softmax(logits)


def _weighted_quantile_rows(
    values: NDArray[np.float64], weights: NDArray[np.float64], quantile: float
) -> NDArray[np.float64]:
    output = np.zeros(len(values), dtype=np.float64)
    for row in range(len(values)):
        order = np.argsort(values[row])
        ordered_values = values[row, order]
        cumulative = np.cumsum(weights[row, order])
        output[row] = ordered_values[min(np.searchsorted(cumulative, quantile), 2)]
    return output


def _operational_constraint_view(
    events: pd.DataFrame,
    constraints: pd.DataFrame | None,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "available": np.zeros(len(events), dtype=np.float64),
            "capacity_constrained": np.full(len(events), np.nan),
            "capacity_marginal": np.full(len(events), np.nan),
            "capacity_good": np.full(len(events), np.nan),
        }
    )
    if constraints is None or constraints.empty:
        return output
    required = {
        "constraint_id",
        "airport",
        "direction",
        "issue_time_utc",
        "valid_from_utc",
        "valid_to_utc",
        "capacity_low",
        "capacity_median",
        "capacity_high",
    }
    missing = sorted(required - set(constraints.columns))
    if missing:
        raise ValueError(f"operational constraints are missing columns: {missing}")
    if constraints["constraint_id"].isna().any() or constraints["constraint_id"].duplicated().any():
        raise ValueError("operational constraint ids must be unique and non-missing")
    table = constraints.copy()
    for column in ("issue_time_utc", "valid_from_utc", "valid_to_utc"):
        table[column] = pd.to_datetime(table[column], errors="raise", utc=True).dt.tz_localize(None)
    if table["valid_to_utc"].le(table["valid_from_utc"]).any():
        raise ValueError("operational constraint validity intervals must be positive")
    for column in ("capacity_low", "capacity_median", "capacity_high"):
        table[column] = pd.to_numeric(table[column], errors="raise")
    if (
        table[["capacity_low", "capacity_median", "capacity_high"]].lt(0.0).any().any()
        or table["capacity_low"].gt(table["capacity_median"]).any()
        or table["capacity_median"].gt(table["capacity_high"]).any()
    ):
        raise ValueError("operational constraint capacities must be ordered and non-negative")
    for airport, indexes in events.groupby("airport", sort=False, observed=True).groups.items():
        candidate = table.loc[table["airport"].astype(str).eq(str(airport))]
        if candidate.empty:
            continue
        for position in indexes:
            event = events.loc[position]
            eligible = candidate.loc[
                candidate["direction"].astype(str).isin([str(event["direction"]), "movement"])
                & candidate["issue_time_utc"].le(event["cutoff_time_utc"])
                & candidate["valid_from_utc"].le(event["event_time_utc"])
                & candidate["valid_to_utc"].gt(event["event_time_utc"])
            ]
            if eligible.empty:
                continue
            selected = eligible.sort_values("issue_time_utc", kind="mergesort").iloc[-1]
            output.loc[position, "available"] = 1.0
            output.loc[position, "capacity_constrained"] = selected["capacity_low"]
            output.loc[position, "capacity_marginal"] = selected["capacity_median"]
            output.loc[position, "capacity_good"] = selected["capacity_high"]
    return output


def _event_capacity_state(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    resource_grid: pd.DataFrame,
    metro_grid: pd.DataFrame,
    resource_catalog: dict[str, dict[str, Any]],
    *,
    constraints: pd.DataFrame | None,
    config: CapacityConfig,
) -> pd.DataFrame:
    grid_columns = [
        "airport",
        "bucket_time_utc",
        "resource_node_id",
        "arrival_demand",
        "departure_demand",
        "movement_demand",
        *[
            f"{direction}_frontier_{quantile}"
            for direction in ("arrival", "departure", "movement")
            for quantile in QUANTILES
        ],
        "movement_support_days",
        "movement_observed_fraction",
        *[
            f"{direction}_{kind}_{scenario}"
            for direction in ("arrival", "departure", "movement")
            for kind in ("queue", "recovery")
            for scenario in SCENARIOS
        ],
    ]
    attached = events.merge(
        resource_grid.loc[:, grid_columns],
        on=["airport", "bucket_time_utc"],
        how="left",
        validate="many_to_one",
    )
    if attached["resource_node_id"].isna().any():
        raise RuntimeError("event could not attach to an airport resource node")
    static = _static_and_runway_state(frame, attached, resource_catalog, config=config)
    state = pd.concat([attached, static], axis=1)
    stress = _weather_stress(frame, state, config=config)
    probabilities = _scenario_probabilities(
        stress,
        state["runway_feasible_end_fraction"].to_numpy(dtype=np.float64),
        config=config,
    )
    state["weather_capacity_stress"] = stress
    for index, scenario in enumerate(SCENARIOS):
        state[f"capacity_scenario_{scenario}_probability"] = probabilities[:, index]

    direction_capacity = np.column_stack(
        [
            np.where(
                state["direction"].eq("arrival"),
                state[f"arrival_frontier_{quantile}"],
                state[f"departure_frontier_{quantile}"],
            )
            for quantile in QUANTILES
        ]
    ).astype(np.float64)
    movement_capacity = state[[f"movement_frontier_{quantile}" for quantile in QUANTILES]].to_numpy(
        dtype=np.float64
    )
    constraints_view = _operational_constraint_view(state, constraints)
    optional_capacity = constraints_view[
        [f"capacity_{scenario}" for scenario in SCENARIOS]
    ].to_numpy(dtype=np.float64)
    optional_available = constraints_view["available"].to_numpy(dtype=np.float64) > 0.0
    direction_capacity[optional_available] = np.minimum(
        direction_capacity[optional_available], optional_capacity[optional_available]
    )
    movement_capacity[optional_available] = np.minimum(
        movement_capacity[optional_available], optional_capacity[optional_available]
    )

    direction_demand = np.where(
        state["direction"].eq("arrival"),
        state["arrival_demand"],
        state["departure_demand"],
    ).astype(np.float64)
    movement_demand = state["movement_demand"].to_numpy(dtype=np.float64)
    safe_direction_capacity = np.maximum(direction_capacity, 0.25)
    safe_movement_capacity = np.maximum(movement_capacity, 0.25)
    direction_utilization = direction_demand[:, None] / safe_direction_capacity
    movement_utilization = movement_demand[:, None] / safe_movement_capacity
    overload = (direction_demand[:, None] > direction_capacity) | (
        movement_demand[:, None] > movement_capacity
    )
    slack = np.minimum(
        direction_capacity - direction_demand[:, None],
        movement_capacity - movement_demand[:, None],
    )
    queue = np.zeros((len(state), 3), dtype=np.float64)
    recovery = np.zeros((len(state), 3), dtype=np.float64)
    for index, scenario in enumerate(SCENARIOS):
        direction_queue = np.where(
            state["direction"].eq("arrival"),
            state[f"arrival_queue_{scenario}"],
            state[f"departure_queue_{scenario}"],
        )
        direction_recovery = np.where(
            state["direction"].eq("arrival"),
            state[f"arrival_recovery_{scenario}"],
            state[f"departure_recovery_{scenario}"],
        )
        queue[:, index] = np.maximum(
            direction_queue, state[f"movement_queue_{scenario}"].to_numpy()
        )
        recovery[:, index] = np.maximum(
            direction_recovery, state[f"movement_recovery_{scenario}"].to_numpy()
        )

    expected_direction_capacity = (probabilities * direction_capacity).sum(axis=1)
    expected_movement_capacity = (probabilities * movement_capacity).sum(axis=1)
    state["expected_direction_capacity"] = expected_direction_capacity
    state["expected_movement_capacity"] = expected_movement_capacity
    state["direction_utilization"] = (probabilities * direction_utilization).sum(axis=1)
    state["movement_utilization"] = (probabilities * movement_utilization).sum(axis=1)
    state["overload_probability"] = (probabilities * overload).sum(axis=1)
    state["expected_slack"] = (probabilities * slack).sum(axis=1)
    state["expected_queue"] = (probabilities * queue).sum(axis=1)
    state["queue_p90"] = _weighted_quantile_rows(queue, probabilities, 0.90)
    state["recovery_minutes"] = (probabilities * recovery).sum(axis=1)
    maximum_utilization = np.maximum(direction_utilization, movement_utilization)
    state["shadow_price"] = (
        probabilities
        * (
            np.logaddexp(
                0.0,
                (maximum_utilization - 1.0) / config.shadow_price_temperature,
            )
            * config.shadow_price_temperature
        )
    ).sum(axis=1)
    counterfactual_direction = np.maximum(direction_demand - 1.0, 0.0)[:, None]
    counterfactual_movement = np.maximum(movement_demand - 1.0, 0.0)[:, None]
    observed_excess = np.maximum(maximum_utilization - 1.0, 0.0)
    counterfactual_excess = np.maximum(
        np.maximum(
            counterfactual_direction / safe_direction_capacity,
            counterfactual_movement / safe_movement_capacity,
        )
        - 1.0,
        0.0,
    )
    state["marginal_overload"] = (probabilities * (observed_excess - counterfactual_excess)).sum(
        axis=1
    )
    state["historical_frontier_support_days_log1p"] = np.log1p(state["movement_support_days"])
    state["historical_frontier_observed_fraction"] = state["movement_observed_fraction"]
    state["optional_constraint_available"] = constraints_view["available"].to_numpy()
    state["optional_constraint_overload_probability"] = np.where(
        optional_available,
        (probabilities * (movement_demand[:, None] > optional_capacity)).sum(axis=1),
        np.nan,
    )

    state["metro_utilization"] = 0.0
    state["metro_queue"] = 0.0
    state["metro_resource_node_id"] = pd.NA
    if not metro_grid.empty:
        metro_columns = [
            "metro",
            "bucket_time_utc",
            "resource_node_id",
            "movement_demand",
            *[f"movement_frontier_{quantile}" for quantile in QUANTILES],
            *[f"movement_queue_{scenario}" for scenario in SCENARIOS],
        ]
        joined = state.loc[:, ["metro", "bucket_time_utc"]].merge(
            metro_grid.loc[:, metro_columns],
            on=["metro", "bucket_time_utc"],
            how="left",
            validate="many_to_one",
            suffixes=("", "_metro"),
        )
        metro_capacity = joined[
            [f"movement_frontier_{quantile}" for quantile in QUANTILES]
        ].to_numpy(dtype=np.float64)
        metro_demand = joined["movement_demand"].to_numpy(dtype=np.float64)
        metro_queue = joined[[f"movement_queue_{scenario}" for scenario in SCENARIOS]].to_numpy(
            dtype=np.float64
        )
        has_metro = joined["resource_node_id"].notna().to_numpy()
        state.loc[has_metro, "metro_utilization"] = (
            probabilities[has_metro]
            * (metro_demand[has_metro, None] / np.maximum(metro_capacity[has_metro], 0.25))
        ).sum(axis=1)
        state.loc[has_metro, "metro_queue"] = (
            probabilities[has_metro] * metro_queue[has_metro]
        ).sum(axis=1)
        state.loc[has_metro, "metro_resource_node_id"] = joined.loc[
            has_metro, "resource_node_id"
        ].to_numpy()
    return state


def _event_to_flight_features(state: pd.DataFrame, flight_count: int) -> pd.DataFrame:
    output_data: dict[str, NDArray[np.float64]] = {}
    neutral_static = (
        "physical_runway_count",
        "eligible_runway_end_count",
        "orientation_family_count",
        "max_parallel_runways",
        "ils_end_fraction",
        "minimum_runway_length_kft",
        "maximum_runway_length_kft",
        "airport_annual_operations_log1p",
        "runway_catalog_missing",
    )
    neutral_demand = tuple(
        [
            f"{direction}_demand_{window}m"
            for direction in ("arrival", "departure", "movement")
            for window in WINDOWS_MINUTES
        ]
        + [
            "same_direction_loo_30m",
            "movement_loo_30m",
            "minutes_since_previous_same_direction",
            "minutes_until_next_same_direction",
            "flow_imbalance_60m",
            "burstiness_15_to_60",
            "metro_movement_demand_60m",
        ]
    )
    neutral_state = tuple(
        [
            f"{kind}_frontier_{quantile}"
            for kind in ("direction", "movement")
            for quantile in QUANTILES
        ]
        + [
            "historical_frontier_support_days_log1p",
            "historical_frontier_observed_fraction",
            "runway_configuration_entropy",
            "runway_effective_end_count",
            "runway_feasible_end_fraction",
            "weather_capacity_stress",
            "capacity_scenario_good_probability",
            "capacity_scenario_marginal_probability",
            "capacity_scenario_constrained_probability",
            "expected_direction_capacity",
            "expected_movement_capacity",
            "direction_utilization",
            "movement_utilization",
            "overload_probability",
            "expected_slack",
            "expected_queue",
            "queue_p90",
            "recovery_minutes",
            "shadow_price",
            "marginal_overload",
            "metro_utilization",
            "metro_queue",
            "optional_constraint_available",
            "optional_constraint_overload_probability",
        ]
    )
    state = state.copy()
    direction_arrival = state["direction"].eq("arrival")
    for quantile in QUANTILES:
        state[f"direction_frontier_{quantile}"] = np.where(
            direction_arrival,
            state[f"arrival_frontier_{quantile}"],
            state[f"departure_frontier_{quantile}"],
        )
    for side in ("origin", "dest"):
        selected = state.loc[state["side"].eq(side)].sort_values("flight_row")
        if len(selected) != flight_count or not np.array_equal(
            selected["flight_row"].to_numpy(dtype=np.int64), np.arange(flight_count)
        ):
            raise RuntimeError(
                f"capacity state does not contain exactly one {side} event per flight"
            )
        for name in neutral_static:
            output_data[f"ccrth_{side}_{name}"] = pd.to_numeric(
                selected[name], errors="coerce"
            ).to_numpy(dtype=np.float64)
        for name in neutral_demand:
            values = pd.to_numeric(selected[name], errors="coerce").to_numpy(dtype=np.float64)
            contract_name = name
            if (
                name.endswith("_demand_15m")
                or name.endswith("_demand_30m")
                or name.endswith("_demand_60m")
                or name.endswith("_demand_120m")
            ):
                contract_name = f"{name}_log1p"
                values = np.log1p(values)
            elif name in {
                "same_direction_loo_30m",
                "movement_loo_30m",
                "minutes_since_previous_same_direction",
                "minutes_until_next_same_direction",
                "metro_movement_demand_60m",
            }:
                contract_name = f"{name}_log1p"
                values = np.log1p(np.maximum(values, 0.0))
            output_data[f"ccrth_{side}_{contract_name}"] = values
        for name in neutral_state:
            output_data[f"ccrth_{side}_{name}"] = pd.to_numeric(
                selected[name], errors="coerce"
            ).to_numpy(dtype=np.float64)
    result = pd.DataFrame(output_data, index=np.arange(flight_count))
    result["ccrth_route_max_overload_probability"] = result[
        ["ccrth_origin_overload_probability", "ccrth_dest_overload_probability"]
    ].max(axis=1)
    result["ccrth_route_max_queue_p90"] = result[
        ["ccrth_origin_queue_p90", "ccrth_dest_queue_p90"]
    ].max(axis=1)
    result["ccrth_route_min_expected_slack"] = result[
        ["ccrth_origin_expected_slack", "ccrth_dest_expected_slack"]
    ].min(axis=1)
    result["ccrth_route_sum_shadow_price"] = result[
        ["ccrth_origin_shadow_price", "ccrth_dest_shadow_price"]
    ].sum(axis=1)
    result["ccrth_route_bottleneck_is_destination"] = (
        result["ccrth_dest_shadow_price"] > result["ccrth_origin_shadow_price"]
    ).astype("float64")
    result["ccrth_route_max_weather_capacity_stress"] = result[
        ["ccrth_origin_weather_capacity_stress", "ccrth_dest_weather_capacity_stress"]
    ].max(axis=1)
    result["ccrth_route_max_metro_utilization"] = result[
        ["ccrth_origin_metro_utilization", "ccrth_dest_metro_utilization"]
    ].max(axis=1)
    return result


def _attach_rotation_messages(
    features: pd.DataFrame,
    flights: pd.DataFrame,
    rotation_edges: pd.DataFrame | None,
    source_frame: pd.DataFrame,
    *,
    predecessor_resource_state: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = features.copy()
    result["ccrth_rotation_connected_probability"] = 0.0
    result["ccrth_rotation_resource_message_coverage"] = 0.0
    result["ccrth_rotation_predecessor_capacity_overload"] = np.nan
    result["ccrth_rotation_predecessor_capacity_queue"] = np.nan
    result["ccrth_rotation_predecessor_capacity_shadow_price"] = np.nan
    if rotation_edges is None or rotation_edges.empty:
        if "flare24_rotation_predecessor_probability" in source_frame:
            result["ccrth_rotation_connected_probability"] = pd.to_numeric(
                source_frame["flare24_rotation_predecessor_probability"], errors="coerce"
            ).fillna(0.0)
        return result, pd.DataFrame(
            columns=[
                "predecessor_sample_id",
                "successor_sample_id",
                "probability",
                "turn_minutes",
            ]
        )
    required = {"predecessor_sample_id", "successor_sample_id", "probability"}
    missing = sorted(required - set(rotation_edges.columns))
    if missing:
        raise ValueError(f"rotation edges are missing columns: {missing}")
    edges = rotation_edges.copy()
    edges["probability"] = pd.to_numeric(edges["probability"], errors="raise")
    if (~edges["probability"].between(0.0, 1.0)).any():
        raise ValueError("rotation edge probabilities must be in [0, 1]")
    predecessor = pd.DataFrame(
        {
            "predecessor_sample_id": flights["sample_id"].astype(str),
            "predecessor_overload": features["ccrth_dest_overload_probability"],
            "predecessor_queue": features["ccrth_dest_queue_p90"],
            "predecessor_shadow": features["ccrth_dest_shadow_price"],
        }
    )
    if predecessor_resource_state is not None and not predecessor_resource_state.empty:
        required_state = {
            "sample_id",
            "overload_probability",
            "queue_p90",
            "shadow_price",
        }
        missing_state = sorted(required_state - set(predecessor_resource_state.columns))
        if missing_state:
            raise ValueError(f"predecessor resource state is missing columns: {missing_state}")
        external = predecessor_resource_state.loc[
            :,
            [
                "sample_id",
                "overload_probability",
                "queue_p90",
                "shadow_price",
            ],
        ].rename(
            columns={
                "sample_id": "predecessor_sample_id",
                "overload_probability": "predecessor_overload",
                "queue_p90": "predecessor_queue",
                "shadow_price": "predecessor_shadow",
            }
        )
        external["predecessor_sample_id"] = external["predecessor_sample_id"].astype(str)
        if external["predecessor_sample_id"].duplicated().any():
            raise ValueError("predecessor resource state sample ids must be unique")
        # Target predecessors have cutoff-valid weather-aware state and therefore
        # take precedence over the weather-neutral context state.
        external = external.loc[
            ~external["predecessor_sample_id"].isin(set(predecessor["predecessor_sample_id"]))
        ]
        predecessor = pd.concat([predecessor, external], ignore_index=True)
    for column in ("predecessor_overload", "predecessor_queue", "predecessor_shadow"):
        predecessor[column] = pd.to_numeric(predecessor[column], errors="coerce")
    total_mass = edges.groupby("successor_sample_id", sort=False, observed=True)[
        "probability"
    ].sum()
    messages = edges.merge(
        predecessor,
        on="predecessor_sample_id",
        how="left",
        validate="many_to_one",
    )
    messages = messages.dropna(
        subset=["predecessor_overload", "predecessor_queue", "predecessor_shadow"]
    ).copy()
    for source, target in (
        ("predecessor_overload", "overload"),
        ("predecessor_queue", "queue"),
        ("predecessor_shadow", "shadow"),
    ):
        messages[f"weighted_{target}"] = messages["probability"] * messages[source]
    aggregated = messages.groupby("successor_sample_id", sort=False, observed=True).agg(
        known_probability=("probability", "sum"),
        weighted_overload=("weighted_overload", "sum"),
        weighted_queue=("weighted_queue", "sum"),
        weighted_shadow=("weighted_shadow", "sum"),
    )
    lookup = flights["sample_id"].astype(str)
    known_lookup = lookup.map(aggregated.to_dict("index"))
    total_lookup = lookup.map(total_mass).fillna(0.0).to_numpy(dtype=np.float64)
    result["ccrth_rotation_connected_probability"] = np.minimum(total_lookup, 1.0)
    for row, payload in enumerate(known_lookup):
        total = total_lookup[row]
        if not isinstance(payload, dict) or total <= 0.0:
            continue
        mass = max(float(payload["known_probability"]), 1e-12)
        result.loc[row, "ccrth_rotation_resource_message_coverage"] = min(mass / total, 1.0)
        result.loc[row, "ccrth_rotation_predecessor_capacity_overload"] = (
            float(payload["weighted_overload"]) / mass
        )
        result.loc[row, "ccrth_rotation_predecessor_capacity_queue"] = (
            float(payload["weighted_queue"]) / mass
        )
        result.loc[row, "ccrth_rotation_predecessor_capacity_shadow_price"] = (
            float(payload["weighted_shadow"]) / mass
        )
    return result, edges


def build_capacity_hypergraph(
    target_frame: pd.DataFrame,
    context_schedule: pd.DataFrame,
    frontier: pd.DataFrame,
    *,
    timezone_by_airport: dict[str, str],
    resource_catalog: dict[str, dict[str, Any]],
    resource_airports: Collection[str] | None = None,
    rotation_edges: pd.DataFrame | None = None,
    operational_constraints: pd.DataFrame | None = None,
    config: CapacityConfig | None = None,
) -> CapacityHypergraphResult:
    """Construct CC-RTH nodes, edges, and flight-level resource messages."""

    settings = config or CapacityConfig()
    flights, target_events, target_imputed = prepare_scheduled_events(
        target_frame,
        timezone_by_airport=timezone_by_airport,
    )
    _, context_events, context_imputed = prepare_scheduled_events(
        context_schedule,
        timezone_by_airport=timezone_by_airport,
    )
    raw_context_events = len(context_events)
    resource_set: set[str] | None = None
    if resource_airports is not None:
        resource_set = {str(airport) for airport in resource_airports}
        if not resource_set:
            raise ValueError("resource_airports must not be empty")
        target_airports = set(target_events["airport"].astype(str))
        outside = sorted(target_airports - resource_set)
        if outside:
            raise ValueError(f"target events fall outside requested resource airports: {outside}")
        context_events = context_events.loc[
            context_events["airport"].astype(str).isin(resource_set)
        ].reset_index(drop=True)
        if context_events.empty:
            raise ValueError("no context events remain in the requested resource airports")
    context_events = attach_event_demand(context_events)
    target_ids = set(flights["sample_id"].astype(str))
    target_event_state = target_events.loc[:, ["sample_id", "side"]].merge(
        context_events,
        on=["sample_id", "side"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_context"),
    )
    if target_event_state["flight_row"].isna().any():
        raise ValueError("context schedule omits target flights")
    target_event_state = target_event_state.drop(
        columns=[column for column in target_event_state if column.endswith("_context")]
    )
    target_event_state["flight_row"] = (
        target_event_state["sample_id"]
        .astype(str)
        .map({sample_id: row for row, sample_id in enumerate(flights["sample_id"].astype(str))})
    )
    resource_grid = build_resource_grid(
        context_events,
        frontier,
        timezone_by_airport=timezone_by_airport,
    )
    metro_grid = _metro_grid(resource_grid)
    state = _event_capacity_state(
        target_frame,
        target_event_state,
        resource_grid,
        metro_grid,
        resource_catalog,
        constraints=operational_constraints,
        config=settings,
    )
    predecessor_state = pd.DataFrame()
    context_only_predecessor_ids: set[str] = set()
    if rotation_edges is not None and not rotation_edges.empty:
        requested_predecessors = set(rotation_edges["predecessor_sample_id"].astype(str))
        predecessor_events = context_events.loc[
            context_events["side"].eq("dest")
            & context_events["sample_id"].astype(str).isin(requested_predecessors)
        ].copy()
        if not predecessor_events.empty:
            predecessor_state = _event_capacity_state(
                context_schedule,
                predecessor_events,
                resource_grid,
                metro_grid,
                resource_catalog,
                constraints=operational_constraints,
                config=settings,
            )
            context_only_predecessor_ids = (
                set(predecessor_state["sample_id"].astype(str)) - target_ids
            )
    features = _event_to_flight_features(state, len(flights))
    features, normalized_rotation_edges = _attach_rotation_messages(
        features,
        flights,
        rotation_edges,
        target_frame,
        predecessor_resource_state=predecessor_state,
    )
    missing_contract = sorted(set(CAPACITY_ALL_FEATURES) - set(features.columns))
    extras = sorted(set(features.columns) - set(CAPACITY_ALL_FEATURES))
    if missing_contract or extras:
        raise RuntimeError(
            f"capacity feature contract mismatch; missing={missing_contract}, extras={extras}"
        )
    features = features.loc[:, list(CAPACITY_ALL_FEATURES)].copy()
    features.insert(0, "sample_id", flights["sample_id"].astype(str).to_numpy())
    numeric = features.loc[:, list(CAPACITY_ALL_FEATURES)].to_numpy(dtype=np.float64)
    if np.isinf(numeric).any():
        raise ValueError("capacity features contain infinity")
    for column in CAPACITY_ALL_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="raise").astype("float32")

    airport_incidence = state.loc[
        :,
        [
            "sample_id",
            "role",
            "direction",
            "resource_node_id",
            "event_time_utc",
            "cutoff_time_utc",
            "overload_probability",
            "shadow_price",
        ],
    ].copy()
    airport_incidence["resource_type"] = "airport_runway_system"
    airport_incidence["incidence_probability"] = 1.0
    metro_incidence = state.loc[state["metro_resource_node_id"].notna(), :].copy()
    if not metro_incidence.empty:
        metro_incidence = metro_incidence.loc[
            :,
            [
                "sample_id",
                "role",
                "direction",
                "metro_resource_node_id",
                "event_time_utc",
                "cutoff_time_utc",
                "overload_probability",
                "shadow_price",
            ],
        ].rename(columns={"metro_resource_node_id": "resource_node_id"})
        metro_incidence["resource_type"] = "metro_schedule_coupling"
        metro_incidence["incidence_probability"] = 1.0
    incidence = pd.concat([airport_incidence, metro_incidence], ignore_index=True)
    incidence["source_vintage_policy"] = "flight_specific_cutoff"

    airport_nodes = resource_grid.copy()
    metro_nodes = metro_grid.copy()
    resource_nodes = pd.concat([airport_nodes, metro_nodes], ignore_index=True, sort=False)
    keep_nodes = set(incidence["resource_node_id"].astype(str))
    resource_nodes = resource_nodes.loc[
        resource_nodes["resource_node_id"].astype(str).isin(keep_nodes)
    ].reset_index(drop=True)
    if resource_nodes["resource_node_id"].duplicated().any():
        raise RuntimeError("resource node ids are not unique")
    if set(incidence["sample_id"].astype(str)) != target_ids:
        raise RuntimeError("hypergraph incidence omits target flights")

    diagnostics = {
        "target_flights": len(flights),
        "target_events": len(state),
        "resource_nodes": len(resource_nodes),
        "incidence_edges": len(incidence),
        "rotation_edges": len(normalized_rotation_edges),
        "target_block_time_imputations": target_imputed,
        "context_block_time_imputations": context_imputed,
        "context_events_before_resource_filter": raw_context_events,
        "context_events_after_resource_filter": len(context_events),
        "resource_airport_count": (
            len(set(context_events["airport"].astype(str)))
            if resource_set is None
            else len(resource_set)
        ),
        "rotation_context_predecessor_states": len(predecessor_state),
        "rotation_context_only_predecessor_states": len(context_only_predecessor_ids),
        "rotation_context_only_edges": int(
            normalized_rotation_edges["predecessor_sample_id"]
            .astype(str)
            .isin(context_only_predecessor_ids)
            .sum()
        ),
        "rotation_context_only_probability_mass": float(
            normalized_rotation_edges.loc[
                normalized_rotation_edges["predecessor_sample_id"]
                .astype(str)
                .isin(context_only_predecessor_ids),
                "probability",
            ].sum()
        ),
        "rotation_resource_message_mean_coverage": float(
            features["ccrth_rotation_resource_message_coverage"].mean()
        ),
        "operational_constraint_coverage": float(
            features[
                [
                    "ccrth_origin_optional_constraint_available",
                    "ccrth_dest_optional_constraint_available",
                ]
            ]
            .to_numpy(dtype=np.float64)
            .mean()
        ),
        "frontier_role": "strictly-prior-year empirical BTS scheduling envelope",
        "active_runway_claimed": False,
        "declared_aar_adr_claimed": False,
        "gate_assignment_claimed": False,
        "target_outcomes_read": False,
    }
    return CapacityHypergraphResult(
        features=features,
        flight_nodes=flights,
        resource_nodes=resource_nodes,
        incidence_edges=incidence,
        rotation_edges=normalized_rotation_edges,
        frontier=frontier,
        diagnostics=diagnostics,
    )
