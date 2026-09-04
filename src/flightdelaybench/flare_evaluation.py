"""Chronological proper-score evaluation and scale selection for FLARE-24."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .bootstrap import paired_cluster_mean_difference
from .flare_aggregate import DEFAULT_GROUP_COLUMNS
from .flare_reconciliation import (
    ReconciliationConstraints,
    constraints_from_frame,
    joint_binary_marginals,
    reconcile_probabilities,
)
from .metrics import binary_metrics, multiclass_brier


def joint_loss_rows(
    labels: ArrayLike,
    probabilities: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(y) != len(p) or len(y) == 0:
        raise ValueError("joint labels and probabilities must align with three states")
    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("joint labels must be 0, 1, or 2")
    if not np.isfinite(p).all() or (p < 0.0).any() or not np.allclose(
        p.sum(axis=1), 1.0, atol=1e-8
    ):
        raise ValueError("joint probabilities must be finite and sum to one")
    clipped = np.clip(p, 1e-12, 1.0)
    clipped /= clipped.sum(axis=1, keepdims=True)
    log_score = -np.log(clipped[np.arange(len(y)), y])
    observed = np.eye(3, dtype=np.float64)[y]
    brier = np.square(clipped - observed).sum(axis=1)
    return log_score, brier


@dataclass(frozen=True, slots=True)
class DateReconciliationResult:
    probabilities: NDArray[np.float64]
    date_diagnostics: tuple[dict[str, Any], ...]


def reconcile_joint_by_date(
    flights: pd.DataFrame,
    base_probabilities: ArrayLike,
    aggregate_forecasts: pd.DataFrame,
    *,
    variance_multiplier: float = 1.0,
    group_columns: dict[str, tuple[str, ...]] | None = None,
) -> DateReconciliationResult:
    """Reconcile each operational date independently to avoid cross-date leakage."""

    base = np.asarray(base_probabilities, dtype=np.float64)
    if base.shape != (len(flights), 3):
        raise ValueError("base joint probabilities must align with flights")
    if not np.isfinite(variance_multiplier) or variance_multiplier <= 0.0:
        raise ValueError("variance_multiplier must be finite and positive")
    required = {"FlightDate", *(column for columns in (group_columns or DEFAULT_GROUP_COLUMNS).values() for column in columns)}
    missing = sorted(required - set(flights.columns))
    if missing:
        raise ValueError(f"reconciliation flights are missing columns: {missing}")
    if "FlightDate" not in aggregate_forecasts:
        raise ValueError("aggregate forecasts are missing FlightDate")
    groups = dict(DEFAULT_GROUP_COLUMNS if group_columns is None else group_columns)
    dates = pd.to_datetime(flights["FlightDate"], errors="raise").dt.normalize()
    aggregate_dates = pd.to_datetime(
        aggregate_forecasts["FlightDate"], errors="raise"
    ).dt.normalize()
    output = np.empty_like(base)
    diagnostics: list[dict[str, Any]] = []
    for target_date in sorted(dates.unique()):
        mask = dates.eq(target_date).to_numpy()
        positions = np.flatnonzero(mask)
        day_flights = flights.loc[mask].reset_index(drop=True)
        day_aggregate = aggregate_forecasts.loc[
            aggregate_dates.eq(target_date)
        ].reset_index(drop=True)
        if day_aggregate.empty:
            raise ValueError(f"no aggregate forecasts for {pd.Timestamp(target_date).date()}")
        compiled = constraints_from_frame(
            day_flights,
            day_aggregate,
            group_columns=groups,
        )
        scaled = ReconciliationConstraints(
            names=compiled.names,
            members=compiled.members,
            mean_counts=compiled.mean_counts,
            variance_counts=compiled.variance_counts * variance_multiplier,
        )
        result = reconcile_probabilities(base[positions], scaled)
        output[positions] = result.probabilities
        date_record = result.diagnostics.as_dict()
        date_record.update(
            {
                "FlightDate": pd.Timestamp(target_date).date().isoformat(),
                "variance_multiplier": variance_multiplier,
            }
        )
        diagnostics.append(date_record)
    if not np.isfinite(output).all() or not np.allclose(output.sum(axis=1), 1.0, atol=1e-9):
        raise RuntimeError("date reconciliation produced incoherent output")
    return DateReconciliationResult(output, tuple(diagnostics))


def select_reconciliation_variance_multiplier(
    flights: pd.DataFrame,
    base_probabilities: ArrayLike,
    aggregate_forecasts: pd.DataFrame,
    labels: ArrayLike,
    *,
    candidates: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
) -> dict[str, Any]:
    """Select reconciliation strength on a labelled development period only."""

    y = np.asarray(labels, dtype=np.int64)
    if len(y) != len(flights) or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("reconciliation selection labels must align and have three states")
    if not candidates or any(not np.isfinite(value) or value <= 0.0 for value in candidates):
        raise ValueError("reconciliation scale candidates must be finite and positive")
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        result = reconcile_joint_by_date(
            flights,
            base_probabilities,
            aggregate_forecasts,
            variance_multiplier=float(candidate),
        )
        log_rows, brier_rows = joint_loss_rows(y, result.probabilities)
        records.append(
            {
                "variance_multiplier": float(candidate),
                "joint_log_loss": float(log_rows.mean()),
                "multiclass_brier": float(brier_rows.mean()),
                "mean_probability_shift": float(
                    np.mean(
                        np.abs(
                            result.probabilities
                            - np.asarray(base_probabilities, dtype=np.float64)
                        )
                    )
                ),
            }
        )
    selected = min(records, key=lambda record: (record["joint_log_loss"], record["variance_multiplier"]))
    return {
        "selection_metric": "joint_log_loss",
        "selected_variance_multiplier": selected["variance_multiplier"],
        "candidates": records,
        "selection_outcomes_scope": "development period supplied by caller",
    }


def evaluate_joint_probabilities(
    frame: pd.DataFrame,
    methods: dict[str, ArrayLike],
    *,
    reference_method: str,
    bootstrap_repetitions: int = 2_000,
    bootstrap_seed: int = 20260903,
) -> dict[str, Any]:
    """Evaluate joint and hurdle views using proper scores and paired date clusters."""

    required = {
        "FlightDate",
        "disruption_state",
        "joint_label_observed",
        "Cancelled",
        "delay_label_observed",
        "ArrDel15",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"joint evaluation frame is missing columns: {missing}")
    if reference_method not in methods:
        raise ValueError(f"unknown reference method: {reference_method}")
    observed = frame["joint_label_observed"].eq(1).to_numpy()
    if not observed.any():
        raise ValueError("joint evaluation has no observed outcomes")
    labels = frame.loc[observed, "disruption_state"].to_numpy(dtype=np.int64)
    clusters = frame.loc[observed, "FlightDate"].to_numpy()
    per_method_losses: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    results: dict[str, Any] = {}
    for name, values in methods.items():
        probabilities = np.asarray(values, dtype=np.float64)
        if probabilities.shape != (len(frame), 3):
            raise ValueError(f"joint method {name} probabilities do not align")
        scored = probabilities[observed]
        log_rows, brier_rows = joint_loss_rows(labels, scored)
        per_method_losses[name] = (log_rows, brier_rows)
        views = joint_binary_marginals(probabilities)
        cancellation_mask = frame["Cancelled"].isin([0, 1]).to_numpy()
        operated_mask = (
            frame["Cancelled"].eq(0) & frame["delay_label_observed"].eq(1)
        ).to_numpy()
        results[name] = {
            "joint": {
                "n": len(labels),
                "log_loss": float(log_rows.mean()),
                "multiclass_brier": multiclass_brier(labels, scored),
            },
            "cancellation": binary_metrics(
                frame.loc[cancellation_mask, "Cancelled"].to_numpy(dtype=np.int64),
                views["cancel_probability"][cancellation_mask],
            ).as_dict(),
            "delay_given_operated": binary_metrics(
                frame.loc[operated_mask, "ArrDel15"].to_numpy(dtype=np.int64),
                views["delay_given_operated_probability"][operated_mask],
            ).as_dict(),
        }
    reference_log, reference_brier = per_method_losses[reference_method]
    comparisons: dict[str, Any] = {}
    for name, (candidate_log, candidate_brier) in per_method_losses.items():
        if name == reference_method:
            continue
        log_interval = paired_cluster_mean_difference(
            candidate_log,
            reference_log,
            clusters,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        )
        brier_interval = paired_cluster_mean_difference(
            candidate_brier,
            reference_brier,
            clusters,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        )
        comparisons[f"{name}_minus_{reference_method}"] = {
            "joint_log_loss": asdict(log_interval),
            "multiclass_brier": asdict(brier_interval),
        }
    return {
        "reference_method": reference_method,
        "methods": results,
        "paired_date_cluster_comparisons": comparisons,
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_seed": bootstrap_seed,
    }
