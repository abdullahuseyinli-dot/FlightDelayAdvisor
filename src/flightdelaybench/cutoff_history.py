"""Observation-time history for a new, explicitly versioned T-24 study.

Legacy daily summaries cannot be repaired by relabelling their timestamps. This
module accepts individual observations with evidenced event/publication/availability
times and computes windows of availability time. It never invents those timestamps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .contracts import CUTOFF_HISTORY_FEATURES, RECENT_WINDOWS_DAYS

_VIEW_KEYS = {
    "global": (None, None),
    "route": ("Route", "Route"),
    "airline": ("Reporting_Airline", "Reporting_Airline"),
    "origin_outbound": ("Origin", "Origin"),
    "origin_inbound": ("Origin", "Dest"),
    "dest_inbound": ("Dest", "Dest"),
    "dest_outbound": ("Dest", "Origin"),
    "flight": ("ScheduledFlightId", "ScheduledFlightId"),
}
_KEYS = ("Route", "Reporting_Airline", "Origin", "Dest", "ScheduledFlightId")
_TIMES = ("event_time_utc", "source_published_at_utc", "available_at_utc")


def utc_timestamps(values: pd.Series, *, name: str) -> pd.Series:
    """Reject missing or naive timestamps rather than assuming a local timezone."""
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.isna().any() or not isinstance(parsed.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{name} requires complete timezone-aware timestamps")
    # Pandas preserves input resolution; integer microseconds cannot be compared
    # with Timedelta.value (nanoseconds). Normalize before any searchsorted query.
    return parsed.dt.tz_convert("UTC").astype("datetime64[ns, UTC]")


def _dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.isna().any() or parsed.dt.tz is not None:
        raise ValueError("FlightDate must be a complete timezone-naive operating date")
    if not parsed.eq(parsed.dt.normalize()).all():
        raise ValueError("FlightDate must be normalized to midnight")
    if parsed.dt.year.ge(2026).any():
        raise ValueError("cutoff redevelopment refuses 2026 or later outcomes/targets")
    return parsed


def validate_history_inputs(
    targets: pd.DataFrame, observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_targets = {"sample_id", "FlightDate", "cutoff_time_utc", "departure_time_utc", *_KEYS}
    required_observations = {
        "sample_id", "FlightDate", "outcome", "value", "source_id", *_TIMES, *_KEYS,
    }
    for frame, required, name in (
        (targets, required_targets, "targets"),
        (observations, required_observations, "observations"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")
        if frame.loc[:, ["sample_id", *_KEYS]].isna().any().any():
            raise ValueError(f"{name} contain missing identity/group keys")
    if targets.empty or targets["sample_id"].duplicated().any():
        raise ValueError("targets must be nonempty with unique sample IDs")
    t, e = targets.reset_index(drop=True).copy(), observations.reset_index(drop=True).copy()
    t["FlightDate"], e["FlightDate"] = _dates(t["FlightDate"]), _dates(e["FlightDate"])
    for name in ("cutoff_time_utc", "departure_time_utc"):
        t[name] = utc_timestamps(t[name], name=name)
    if not (t["departure_time_utc"] - t["cutoff_time_utc"]).eq(pd.Timedelta(hours=24)).all():
        raise ValueError("target cutoff must equal scheduled departure minus 24 hours")
    for name in _TIMES:
        e[name] = utc_timestamps(e[name], name=name)
    if e["source_id"].isna().any() or e["source_id"].astype(str).str.strip().eq("").any():
        raise ValueError("every observation requires a source identity")
    if not e["outcome"].isin(["delay", "cancel"]).all():
        raise ValueError("outcome must be delay or cancel")
    e["value"] = pd.to_numeric(e["value"], errors="raise")
    if not e["value"].isin([0, 1]).all():
        raise ValueError("observed values must be binary, without imputed missing labels")
    if e.duplicated(["sample_id", "outcome"]).any():
        raise ValueError("duplicate observation or unresolved label revision")
    if (e["source_published_at_utc"] < e["event_time_utc"]).any() or (
        e["available_at_utc"] < e["source_published_at_utc"]
    ).any():
        raise ValueError("observation timestamps violate event <= publication <= availability")
    # Even a malformed earlier-date copy of the target may never enter its history.
    if set(t["sample_id"]) & set(e["sample_id"]):
        # Overlap is normal for a multi-day batch; validate date identity before
        # excluding each target's own operating date during construction.
        matched = t[["sample_id", "FlightDate"]].merge(
            e[["sample_id", "FlightDate"]], on="sample_id", suffixes=("_target", "_event")
        )
        if not matched["FlightDate_target"].eq(matched["FlightDate_event"]).all():
            raise ValueError("target/observation sample ID has conflicting operating dates")
    return t, e


def build_cutoff_history(
    targets: pd.DataFrame, observations: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build 7/28/90-day histories from observations available at each cutoff.

    Windows are [cutoff-window, cutoff] in *consumer availability time*. Thus a
    late-published historical observation only enters when actually available.
    Outcomes from the target operating date are excluded independently. Missing
    support produces NaN rates and zero counts, never a fabricated on-time prior.
    """
    t, e = validate_history_inputs(targets, observations)
    output = pd.DataFrame(np.nan, index=t.index, columns=list(CUTOFF_HISTORY_FEATURES), dtype="float32")
    maximum_used = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns, UTC]")
    eligible_pairs = 0
    for flight_date, day_targets in t.groupby("FlightDate", sort=True):
        min_cutoff = day_targets["cutoff_time_utc"].min() - pd.Timedelta(days=90)
        max_cutoff = day_targets["cutoff_time_utc"].max()
        day_events = e.loc[
            e["FlightDate"].lt(flight_date)
            & e["available_at_utc"].between(min_cutoff, max_cutoff)
        ]
        for view, (target_key, event_key) in _VIEW_KEYS.items():
            groups = [(None, day_targets)] if target_key is None else day_targets.groupby(target_key, sort=False)
            # Index once per view/day: scanning every observation again for each
            # scheduled-flight ID is quadratic on a census-sized context table.
            event_groups = None if event_key is None else day_events.groupby(event_key, sort=False).indices
            for key, rows in groups:
                candidates = day_events if event_groups is None else day_events.iloc[event_groups.get(key, [])]
                query = rows["cutoff_time_utc"].astype("int64").to_numpy()
                for outcome in ("delay", "cancel"):
                    subset = candidates.loc[candidates["outcome"].eq(outcome)].sort_values("available_at_utc")
                    times = subset["available_at_utc"].astype("int64").to_numpy()
                    values = subset["value"].to_numpy(dtype=np.float64)
                    sums = np.r_[0.0, values.cumsum()]
                    right = np.searchsorted(times, query, side="right")
                    for window in RECENT_WINDOWS_DAYS:
                        left = np.searchsorted(times, query - pd.Timedelta(days=window).value, side="left")
                        count = right - left
                        total = sums[right] - sums[left]
                        rate = np.divide(total, count, out=np.full(len(rows), np.nan), where=count > 0)
                        output.loc[rows.index, f"asof_{view}_{outcome}_rate_{window}d"] = rate.astype("float32")
                        output.loc[rows.index, f"asof_{view}_{outcome}_support_log1p_{window}d"] = np.log1p(count).astype("float32")
                        if view == "global" and window == 90:
                            eligible_pairs += int(count.sum())
                            present = count > 0
                            indices = rows.index[present]
                            stamps = pd.Series(pd.to_datetime(times[right[present] - 1], utc=True), index=indices)
                            prior = maximum_used.loc[indices]
                            replace = prior.isna() | stamps.gt(prior)
                            maximum_used.loc[indices[replace]] = stamps.loc[replace]
    if (maximum_used > t["cutoff_time_utc"]).any():
        raise RuntimeError("history availability exceeded target cutoff")
    result = pd.concat([t[["sample_id", "FlightDate", "cutoff_time_utc"]], output], axis=1)
    result["history_max_available_at_utc"] = maximum_used
    audit = {
        "schema_version": 1,
        "status": "PASS_OBSERVATION_TIME_HISTORY",
        "target_rows": len(t),
        "source_observations": len(e),
        "feature_count": len(CUTOFF_HISTORY_FEATURES),
        "history_window_clock": "consumer availability time",
        "global_90d_observation_target_pairs": eligible_pairs,
        "rows_without_history": int(maximum_used.isna().sum()),
        "source_ids": sorted(e["source_id"].astype(str).unique().tolist()),
        "maximum_availability_excess_seconds": 0.0,
        "same_operating_day_outcomes_allowed": False,
        "source_timestamps_synthesized": False,
        "confirmation_outcomes_accessed": False,
    }
    return result, audit


def assert_safe_history_attachment(targets: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Join by ID and require identical prediction cutoffs, never by date alone."""
    columns = {"sample_id", "cutoff_time_utc", "history_max_available_at_utc", *CUTOFF_HISTORY_FEATURES}
    if columns - set(history.columns):
        raise ValueError("incomplete cutoff history partition")
    if set(CUTOFF_HISTORY_FEATURES) & set(targets):
        raise ValueError("target already contains history; replacement must be explicit")
    if targets["sample_id"].isna().any() or history["sample_id"].isna().any():
        raise ValueError("missing target/history sample IDs")
    if history["sample_id"].duplicated().any() or targets["sample_id"].duplicated().any():
        raise ValueError("duplicate target/history sample IDs")
    attached = targets.merge(history.drop(columns="FlightDate", errors="ignore"), on="sample_id", how="left", validate="one_to_one", suffixes=("", "_history"), indicator=True)
    if not attached["_merge"].eq("both").all():
        raise ValueError("history omits requested targets")
    cutoff = utc_timestamps(attached["cutoff_time_utc"], name="target cutoff")
    history_cutoff = utc_timestamps(attached["cutoff_time_utc_history"], name="history cutoff")
    if not cutoff.eq(history_cutoff).all():
        raise ValueError("history was computed for a different prediction cutoff")
    present = attached["history_max_available_at_utc"].notna()
    used = pd.Series(pd.NaT, index=attached.index, dtype="datetime64[ns, UTC]")
    if present.any():
        used.loc[present] = utc_timestamps(attached.loc[present, "history_max_available_at_utc"], name="history availability")
    if (used > cutoff).any():
        raise ValueError("attached history contains post-cutoff observations")
    for feature in CUTOFF_HISTORY_FEATURES:
        values = pd.to_numeric(attached[feature], errors="raise")
        if "_rate_" in feature:
            support = attached[feature.replace("_rate_", "_support_log1p_")]
            if not values.isna().eq(support.eq(0)).all() or not values.dropna().between(0, 1).all():
                raise ValueError("invalid history rate/support pair")
        elif not np.isfinite(values).all() or values.lt(0).any():
            raise ValueError("invalid history support")
    global_support = attached[[f"asof_global_{outcome}_support_log1p_90d" for outcome in ("delay", "cancel")]].gt(0).any(axis=1)
    if not global_support.eq(present).all():
        raise ValueError("history support lacks matching availability evidence")
    return attached.drop(columns=["cutoff_time_utc_history", "_merge"])
