"""Independent hierarchical aggregate forecasts for FLARE-24 reconciliation.

The reconciler must not derive its margins by summing the same per-flight model
that it adjusts.  This module supplies genuinely separate airport-, carrier-,
and route-level forecasts using a recency-weighted hierarchical
Dirichlet-multinomial model.  Its uncertainty is propagated through the
hierarchy and controls reconciliation strength.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

STATE_NAMES = ("on_time", "delayed", "cancelled")
DEFAULT_GROUP_COLUMNS: dict[str, tuple[str, ...]] = {
    "origin_hour": ("Origin", "DepHour"),
    "destination_hour": ("Dest", "ArrHour"),
    "carrier_day": ("Reporting_Airline",),
    "route_day": ("Route",),
}
_TARGET_OUTCOME_COLUMNS = frozenset(
    {
        "ArrDel15",
        "Cancelled",
        "disruption_state",
        "delay_label_observed",
        "joint_label_observed",
    }
)


def _canonical_key(value: object) -> str:
    if pd.isna(value):
        raise ValueError("aggregate group keys cannot be missing")
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _key_tuple(values: tuple[object, ...]) -> tuple[str, ...]:
    return tuple(_canonical_key(value) for value in values)


def _weighted_state_map(
    frame: pd.DataFrame,
    keys: tuple[str, ...],
) -> dict[tuple[str, ...], NDArray[np.float64]]:
    grouped = (
        frame.groupby([*keys, "_state"], sort=False, observed=True)["_weight"]
        .sum()
        .unstack("_state", fill_value=0.0)
        .reindex(columns=range(len(STATE_NAMES)), fill_value=0.0)
    )
    result: dict[tuple[str, ...], NDArray[np.float64]] = {}
    for index, row in grouped.iterrows():
        raw = index if isinstance(index, tuple) else (index,)
        result[_key_tuple(tuple(raw))] = row.to_numpy(dtype=np.float64)
    return result


def _posterior_update(
    counts: NDArray[np.float64],
    prior_probability: NDArray[np.float64],
    prior_variance: NDArray[np.float64],
    prior_strength: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    support = float(counts.sum())
    concentration = support + prior_strength
    probability = (counts + prior_strength * prior_probability) / concentration
    inherited_weight = prior_strength / concentration
    parameter_variance = (
        probability * (1.0 - probability) / (concentration + 1.0)
        + np.square(inherited_weight) * prior_variance
    )
    return probability, parameter_variance


@dataclass(frozen=True, slots=True)
class AggregateModelCard:
    method: str
    reference_date: str
    maximum_history_date: str
    half_life_days: float
    global_prior_strength: float
    group_prior_strength: float
    seasonal_prior_strength: float
    history_rows: int
    history_dates: int
    state_names: tuple[str, ...]
    group_columns: dict[str, tuple[str, ...]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "reference_date": self.reference_date,
            "maximum_history_date": self.maximum_history_date,
            "half_life_days": self.half_life_days,
            "global_prior_strength": self.global_prior_strength,
            "group_prior_strength": self.group_prior_strength,
            "seasonal_prior_strength": self.seasonal_prior_strength,
            "history_rows": self.history_rows,
            "history_dates": self.history_dates,
            "state_names": list(self.state_names),
            "group_columns": {
                name: list(columns) for name, columns in self.group_columns.items()
            },
        }


class HierarchicalAggregateForecaster:
    """Recency-weighted hierarchical forecast of three-state group counts."""

    def __init__(
        self,
        *,
        group_columns: dict[str, tuple[str, ...]] | None = None,
        half_life_days: float = 90.0,
        global_prior_strength: float = 3.0,
        group_prior_strength: float = 20.0,
        seasonal_prior_strength: float = 15.0,
    ) -> None:
        groups = dict(DEFAULT_GROUP_COLUMNS if group_columns is None else group_columns)
        if not groups or any(not columns for columns in groups.values()):
            raise ValueError("group_columns must define at least one non-empty group")
        parameters = (
            half_life_days,
            global_prior_strength,
            group_prior_strength,
            seasonal_prior_strength,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in parameters):
            raise ValueError("aggregate hierarchy parameters must be finite and positive")
        self.group_columns = groups
        self.half_life_days = float(half_life_days)
        self.global_prior_strength = float(global_prior_strength)
        self.group_prior_strength = float(group_prior_strength)
        self.seasonal_prior_strength = float(seasonal_prior_strength)
        self._fitted = False
        self._reference_date = pd.Timestamp.min
        self._maximum_history_date = pd.Timestamp.min
        self._global = np.zeros(len(STATE_NAMES), dtype=np.float64)
        self._global_seasonal: dict[tuple[str, ...], NDArray[np.float64]] = {}
        self._group_general: dict[str, dict[tuple[str, ...], NDArray[np.float64]]] = {}
        self._group_seasonal: dict[str, dict[tuple[str, ...], NDArray[np.float64]]] = {}
        self._history_rows = 0
        self._history_dates = 0

    @staticmethod
    def _derived_schedule(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        if "Route" not in result and {"Origin", "Dest"}.issubset(result.columns):
            result["Route"] = result["Origin"].astype("string") + "-" + result[
                "Dest"
            ].astype("string")
        if "DepHour" not in result and "CRSDepMinutes" in result:
            result["DepHour"] = (
                pd.to_numeric(result["CRSDepMinutes"], errors="raise") // 60
            ).astype("int16")
        if "ArrHour" not in result and "CRSArrMinutes" in result:
            result["ArrHour"] = (
                pd.to_numeric(result["CRSArrMinutes"], errors="raise") // 60
            ).astype("int16")
        return result

    def fit(
        self,
        history: pd.DataFrame,
        *,
        date_column: str = "FlightDate",
        state_column: str = "disruption_state",
    ) -> HierarchicalAggregateForecaster:
        """Fit only from observed historical days; prediction never updates this state."""

        frame = self._derived_schedule(history)
        required = {
            date_column,
            state_column,
            *(column for columns in self.group_columns.values() for column in columns),
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"aggregate history is missing columns: {missing}")
        if frame.empty:
            raise ValueError("aggregate history cannot be empty")
        dates = pd.to_datetime(frame[date_column], errors="raise").dt.normalize()
        states = pd.to_numeric(frame[state_column], errors="raise").astype("int8")
        if not states.isin(range(len(STATE_NAMES))).all():
            raise ValueError("aggregate history states must be 0, 1, or 2")
        for columns in self.group_columns.values():
            if frame.loc[:, list(columns)].isna().any().any():
                raise ValueError("aggregate history group keys cannot be missing")
        maximum = dates.max()
        reference = maximum + pd.Timedelta(days=1)
        age_days = (reference - dates).dt.total_seconds().to_numpy(dtype=np.float64) / 86_400.0
        weights = np.exp2(-age_days / self.half_life_days)
        working = frame.assign(
            _date=dates,
            _state=states,
            _weight=weights,
            _month=dates.dt.month.astype("int8"),
            _day_of_week=dates.dt.dayofweek.astype("int8"),
        )
        self._global = np.bincount(
            states.to_numpy(dtype=np.int64),
            weights=weights,
            minlength=len(STATE_NAMES),
        ).astype(np.float64)
        self._global_seasonal = _weighted_state_map(
            working, ("_month", "_day_of_week")
        )
        self._group_general = {}
        self._group_seasonal = {}
        for group_type, columns in self.group_columns.items():
            self._group_general[group_type] = _weighted_state_map(working, columns)
            self._group_seasonal[group_type] = _weighted_state_map(
                working, (*columns, "_month", "_day_of_week")
            )
        self._reference_date = reference
        self._maximum_history_date = maximum
        self._history_rows = len(frame)
        self._history_dates = int(dates.nunique())
        self._fitted = True
        return self

    def model_card(self) -> AggregateModelCard:
        if not self._fitted:
            raise RuntimeError("aggregate forecaster has not been fitted")
        return AggregateModelCard(
            method="FLARE24-HIERARCHICAL-DECAYED-DIRICHLET-MULTINOMIAL",
            reference_date=self._reference_date.date().isoformat(),
            maximum_history_date=self._maximum_history_date.date().isoformat(),
            half_life_days=self.half_life_days,
            global_prior_strength=self.global_prior_strength,
            group_prior_strength=self.group_prior_strength,
            seasonal_prior_strength=self.seasonal_prior_strength,
            history_rows=self._history_rows,
            history_dates=self._history_dates,
            state_names=STATE_NAMES,
            group_columns=dict(self.group_columns),
        )

    def _hierarchical_probability(
        self,
        *,
        group_type: str,
        group_key: tuple[str, ...],
        month: int,
        day_of_week: int,
        decay: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
        base_alpha = np.full(len(STATE_NAMES), 0.5, dtype=np.float64)
        global_counts = self._global * decay
        global_alpha = global_counts + base_alpha
        global_probability = global_alpha / global_alpha.sum()
        global_variance = (
            global_probability
            * (1.0 - global_probability)
            / (global_alpha.sum() + 1.0)
        )

        season_key = (str(month), str(day_of_week))
        global_season_counts = self._global_seasonal.get(
            season_key, np.zeros(len(STATE_NAMES), dtype=np.float64)
        ) * decay
        global_season_probability, global_season_variance = _posterior_update(
            global_season_counts,
            global_probability,
            global_variance,
            self.global_prior_strength,
        )
        group_counts = self._group_general[group_type].get(
            group_key, np.zeros(len(STATE_NAMES), dtype=np.float64)
        ) * decay
        group_probability, group_variance = _posterior_update(
            group_counts,
            global_season_probability,
            global_season_variance,
            self.group_prior_strength,
        )
        seasonal_key = (*group_key, str(month), str(day_of_week))
        seasonal_counts = self._group_seasonal[group_type].get(
            seasonal_key, np.zeros(len(STATE_NAMES), dtype=np.float64)
        ) * decay
        probability, parameter_variance = _posterior_update(
            seasonal_counts,
            group_probability,
            group_variance,
            self.seasonal_prior_strength,
        )
        return probability, parameter_variance, float(group_counts.sum()), float(
            seasonal_counts.sum()
        )

    def predict(
        self,
        schedule: pd.DataFrame,
        *,
        date_column: str = "FlightDate",
    ) -> pd.DataFrame:
        """Predict independent group counts using schedule fields only."""

        if not self._fitted:
            raise RuntimeError("aggregate forecaster has not been fitted")
        leaked = sorted(_TARGET_OUTCOME_COLUMNS & set(schedule.columns))
        if leaked:
            raise ValueError(f"target aggregate schedule contains outcome columns: {leaked}")
        frame = self._derived_schedule(schedule)
        required = {
            date_column,
            *(column for columns in self.group_columns.values() for column in columns),
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"aggregate target schedule is missing columns: {missing}")
        if frame.empty:
            raise ValueError("aggregate target schedule cannot be empty")
        dates = pd.to_datetime(frame[date_column], errors="raise").dt.normalize()
        if (dates <= self._maximum_history_date).any():
            raise ValueError("aggregate target dates must be strictly after all fitted history")
        frame = frame.assign(_date=dates)
        rows: list[dict[str, Any]] = []
        for target_date, day in frame.groupby("_date", sort=True, observed=True):
            target_timestamp = pd.Timestamp(target_date)
            elapsed = (target_timestamp - self._reference_date).days
            decay = float(np.exp2(-elapsed / self.half_life_days))
            month = int(target_timestamp.month)
            day_of_week = int(target_timestamp.dayofweek)
            for group_type, columns in self.group_columns.items():
                grouped = day.groupby(list(columns), sort=True, observed=True).size()
                for raw_index, count_value in grouped.items():
                    raw_key = raw_index if isinstance(raw_index, tuple) else (raw_index,)
                    group_key = _key_tuple(tuple(raw_key))
                    probability, parameter_variance, group_support, seasonal_support = (
                        self._hierarchical_probability(
                            group_type=group_type,
                            group_key=group_key,
                            month=month,
                            day_of_week=day_of_week,
                            decay=decay,
                        )
                    )
                    scheduled_count = int(count_value)
                    mean = scheduled_count * probability
                    count_variance = (
                        scheduled_count * probability * (1.0 - probability)
                        + scheduled_count
                        * max(scheduled_count - 1, 0)
                        * parameter_variance
                    )
                    record: dict[str, Any] = {
                        "FlightDate": target_timestamp,
                        "group_type": group_type,
                        "scheduled_count": scheduled_count,
                        "history_effective_support": group_support,
                        "seasonal_effective_support": seasonal_support,
                        "history_decay_factor": decay,
                    }
                    record.update(
                        {
                            column: value
                            for column, value in zip(columns, raw_key, strict=True)
                        }
                    )
                    for state_index, state in enumerate(STATE_NAMES):
                        record[f"mean_{state}"] = float(mean[state_index])
                        record[f"variance_{state}"] = float(
                            max(count_variance[state_index], 1e-6)
                        )
                    rows.append(record)
        result = pd.DataFrame(rows)
        if result.empty:
            raise RuntimeError("aggregate forecaster produced no group rows")
        means = result.loc[:, [f"mean_{state}" for state in STATE_NAMES]].sum(axis=1)
        if not np.allclose(
            means.to_numpy(dtype=np.float64),
            result["scheduled_count"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-8,
        ):
            raise AssertionError("aggregate state means do not sum to scheduled counts")
        return result.sort_values(
            ["FlightDate", "group_type"], kind="mergesort"
        ).reset_index(drop=True)
