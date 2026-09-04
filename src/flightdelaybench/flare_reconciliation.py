"""Uncertainty-aware probabilistic reconciliation for FLARE-24.

The base model predicts a three-state distribution for every scheduled flight.
Independent aggregate models predict expected state counts for overlapping
airport, carrier, and route groups.  This module computes the unique convex
compromise between the two information sources:

    sum_i KL(p_i || q_i)
      + 1/2 sum_(g,c) (sum_(i in g) p_ic - mu_gc)^2 / variance_gc.

Large aggregate variance therefore weakens a constraint rather than forcing an
unreliable total.  The solver works in the smooth dual and never uses realised
target-day outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

_EPSILON = 1e-12


def _probability_matrix(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(f"{name} must have shape (n_flights, n_states>=2)")
    if probabilities.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError(f"{name} must contain finite non-negative values")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} rows must sum to one")
    clipped = np.clip(probabilities, _EPSILON, 1.0)
    return np.asarray(clipped / clipped.sum(axis=1, keepdims=True), dtype=np.float64)


def hurdle_joint_probabilities(
    cancel_probability: ArrayLike,
    delay_given_operated_probability: ArrayLike,
) -> NDArray[np.float64]:
    """Convert cancellation and conditional-delay probabilities to three states.

    The state order is ``on_time, delayed, cancelled``.  Delay is explicitly
    conditional on the flight operating, avoiding the incoherent sum of two
    independently estimated marginal probabilities.
    """

    cancel = np.asarray(cancel_probability, dtype=np.float64)
    delay = np.asarray(delay_given_operated_probability, dtype=np.float64)
    if cancel.ndim != 1 or delay.ndim != 1 or cancel.shape != delay.shape:
        raise ValueError("hurdle inputs must be equally sized one-dimensional arrays")
    if cancel.size == 0:
        raise ValueError("hurdle inputs cannot be empty")
    if (
        not np.isfinite(cancel).all()
        or not np.isfinite(delay).all()
        or (cancel < 0.0).any()
        or (cancel > 1.0).any()
        or (delay < 0.0).any()
        or (delay > 1.0).any()
    ):
        raise ValueError("hurdle probabilities must be finite and in [0, 1]")
    operated = 1.0 - cancel
    return np.column_stack((operated * (1.0 - delay), operated * delay, cancel))


def joint_binary_marginals(probabilities: ArrayLike) -> dict[str, NDArray[np.float64]]:
    """Return cancellation and delay-given-operation views of a joint forecast."""

    joint = _probability_matrix(probabilities, name="probabilities")
    if joint.shape[1] != 3:
        raise ValueError("FLARE-24 joint probabilities must have exactly three states")
    operated = joint[:, 0] + joint[:, 1]
    conditional_delay = np.divide(
        joint[:, 1],
        operated,
        out=np.zeros_like(operated),
        where=operated > _EPSILON,
    )
    return {
        "cancel_probability": joint[:, 2],
        "delay_given_operated_probability": conditional_delay,
        "unconditional_delay_probability": joint[:, 1],
    }


@dataclass(frozen=True, slots=True)
class ReconciliationConstraints:
    """Sparse overlapping aggregate-count forecasts.

    Each constraint has one integer member array and one mean/variance per
    state.  Membership may overlap arbitrarily across constraints.
    """

    names: tuple[str, ...]
    members: tuple[NDArray[np.int64], ...]
    mean_counts: NDArray[np.float64]
    variance_counts: NDArray[np.float64]

    def validate(self, *, n_flights: int, n_states: int) -> None:
        n_constraints = len(self.names)
        if not self.names or len(self.members) != n_constraints:
            raise ValueError("constraints must contain equally sized names and members")
        if len(set(self.names)) != n_constraints:
            raise ValueError("constraint names must be unique")
        expected_shape = (n_constraints, n_states)
        if self.mean_counts.shape != expected_shape:
            raise ValueError(f"mean_counts must have shape {expected_shape}")
        if self.variance_counts.shape != expected_shape:
            raise ValueError(f"variance_counts must have shape {expected_shape}")
        if not np.isfinite(self.mean_counts).all() or (self.mean_counts < 0.0).any():
            raise ValueError("aggregate means must be finite and non-negative")
        if not np.isfinite(self.variance_counts).all() or (self.variance_counts <= 0.0).any():
            raise ValueError("aggregate variances must be finite and strictly positive")
        for index, member_values in enumerate(self.members):
            member = np.asarray(member_values)
            if member.ndim != 1 or member.size == 0:
                raise ValueError(f"constraint {self.names[index]} has no members")
            if not np.issubdtype(member.dtype, np.integer):
                raise ValueError(f"constraint {self.names[index]} members must be integers")
            if (member < 0).any() or (member >= n_flights).any():
                raise ValueError(f"constraint {self.names[index]} member index is out of range")
            if np.unique(member).size != member.size:
                raise ValueError(f"constraint {self.names[index]} repeats a flight")
            group_size = float(member.size)
            if (self.mean_counts[index] > group_size + 1e-8).any():
                raise ValueError(f"constraint {self.names[index]} mean exceeds group size")
            if not np.isclose(
                self.mean_counts[index].sum(), group_size, rtol=0.0, atol=1e-6
            ):
                raise ValueError(
                    f"constraint {self.names[index]} state means must sum to group size"
                )


@dataclass(frozen=True, slots=True)
class ReconciliationDiagnostics:
    converged: bool
    optimizer_message: str
    iterations: int
    objective_before: float
    objective_after: float
    mean_kl_shift: float
    maximum_probability_shift: float
    base_margin_standardized_rmse: float
    reconciled_margin_standardized_rmse: float
    maximum_simplex_error: float
    constraint_count: int
    flight_count: int
    state_count: int
    maximum_absolute_dual_gradient: float
    rms_dual_gradient: float
    dual_preconditioner_minimum: float
    dual_preconditioner_maximum: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "optimizer_message": self.optimizer_message,
            "iterations": self.iterations,
            "objective_before": self.objective_before,
            "objective_after": self.objective_after,
            "mean_kl_shift": self.mean_kl_shift,
            "maximum_probability_shift": self.maximum_probability_shift,
            "base_margin_standardized_rmse": self.base_margin_standardized_rmse,
            "reconciled_margin_standardized_rmse": (
                self.reconciled_margin_standardized_rmse
            ),
            "maximum_simplex_error": self.maximum_simplex_error,
            "constraint_count": self.constraint_count,
            "flight_count": self.flight_count,
            "state_count": self.state_count,
            "maximum_absolute_dual_gradient": self.maximum_absolute_dual_gradient,
            "rms_dual_gradient": self.rms_dual_gradient,
            "dual_preconditioner_minimum": self.dual_preconditioner_minimum,
            "dual_preconditioner_maximum": self.dual_preconditioner_maximum,
        }


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    probabilities: NDArray[np.float64]
    dual_values: NDArray[np.float64]
    base_margins: NDArray[np.float64]
    reconciled_margins: NDArray[np.float64]
    diagnostics: ReconciliationDiagnostics


def _constraint_margins(
    probabilities: NDArray[np.float64],
    members: tuple[NDArray[np.int64], ...],
    *,
    membership: csr_matrix | None = None,
) -> NDArray[np.float64]:
    operator = (
        _membership_matrix(members, n_flights=len(probabilities))
        if membership is None
        else membership
    )
    return np.asarray(operator @ probabilities, dtype=np.float64)


def _membership_matrix(
    members: tuple[NDArray[np.int64], ...],
    *,
    n_flights: int,
) -> csr_matrix:
    """Build the sparse group-by-flight incidence matrix once per solve."""

    lengths = np.fromiter((len(member) for member in members), dtype=np.int64)
    rows = np.repeat(np.arange(len(members), dtype=np.int64), lengths)
    columns = np.concatenate(members).astype(np.int64, copy=False)
    data = np.ones(columns.size, dtype=np.float64)
    return csr_matrix(
        (data, (rows, columns)),
        shape=(len(members), n_flights),
        dtype=np.float64,
    )


def _primal_objective(
    probabilities: NDArray[np.float64],
    base: NDArray[np.float64],
    constraints: ReconciliationConstraints,
    *,
    membership: csr_matrix | None = None,
) -> float:
    divergence = np.sum(probabilities * (np.log(probabilities) - np.log(base)))
    residual = (
        _constraint_margins(
            probabilities,
            constraints.members,
            membership=membership,
        )
        - constraints.mean_counts
    )
    penalty = 0.5 * np.sum(np.square(residual) / constraints.variance_counts)
    return float(divergence + penalty)


def reconcile_probabilities(
    base_probabilities: ArrayLike,
    constraints: ReconciliationConstraints,
    *,
    maximum_iterations: int = 500,
    gradient_tolerance: float = 1e-8,
    require_convergence: bool = True,
) -> ReconciliationResult:
    """Reconcile flight probabilities with uncertain overlapping count forecasts."""

    base = _probability_matrix(base_probabilities, name="base_probabilities")
    n_flights, n_states = base.shape
    constraints.validate(n_flights=n_flights, n_states=n_states)
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    if not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0.0:
        raise ValueError("gradient_tolerance must be finite and positive")

    log_base = np.log(base)
    n_constraints = len(constraints.names)
    membership = _membership_matrix(constraints.members, n_flights=n_flights)

    # L-BFGS sees constraint/state coordinates with radically different curvature:
    # a major-airport on-time total can involve thousands of flights, while a rare
    # route cancellation coordinate may have a variance near zero.  Scale each dual
    # coordinate by the square root of its Hessian diagonal at the base forecast.
    # This is an exact reparameterization of the same convex objective, not a change
    # to the reconciliation estimand.
    base_bernoulli_variance = base * (1.0 - base)
    curvature_at_base = constraints.variance_counts + np.asarray(
        membership @ base_bernoulli_variance,
        dtype=np.float64,
    )
    dual_scale = np.sqrt(np.maximum(curvature_at_base, _EPSILON))

    def probabilities_from_scaled_dual(
        flat_scaled_dual: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        dual = flat_scaled_dual.reshape(n_constraints, n_states) / dual_scale
        logits = log_base - np.asarray(membership.T @ dual, dtype=np.float64)
        logits -= logsumexp(logits, axis=1, keepdims=True)
        return np.asarray(np.exp(logits), dtype=np.float64)

    def objective_and_gradient(
        flat_scaled_dual: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        dual = flat_scaled_dual.reshape(n_constraints, n_states) / dual_scale
        logits = log_base - np.asarray(membership.T @ dual, dtype=np.float64)
        log_partition = logsumexp(logits, axis=1)
        probabilities = np.exp(logits - log_partition[:, None])
        margins = _constraint_margins(
            probabilities,
            constraints.members,
            membership=membership,
        )
        objective = (
            log_partition.sum()
            + np.sum(dual * constraints.mean_counts)
            + 0.5 * np.sum(constraints.variance_counts * np.square(dual))
        )
        gradient = (
            constraints.mean_counts
            + constraints.variance_counts * dual
            - margins
        )
        return float(objective), (gradient / dual_scale).ravel()

    initial_scaled_dual = np.zeros(n_constraints * n_states, dtype=np.float64)
    optimized = minimize(
        objective_and_gradient,
        initial_scaled_dual,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maximum_iterations, "gtol": gradient_tolerance, "ftol": 1e-12},
    )
    converged = bool(optimized.success)
    if require_convergence and not converged:
        raise RuntimeError(f"FLARE-24 reconciliation did not converge: {optimized.message}")
    dual_values = (
        np.asarray(optimized.x, dtype=np.float64).reshape(n_constraints, n_states)
        / dual_scale
    )
    reconciled = probabilities_from_scaled_dual(np.asarray(optimized.x, dtype=np.float64))
    base_margins = _constraint_margins(
        base,
        constraints.members,
        membership=membership,
    )
    reconciled_margins = _constraint_margins(
        reconciled,
        constraints.members,
        membership=membership,
    )
    objective_before = _primal_objective(
        base,
        base,
        constraints,
        membership=membership,
    )
    objective_after = _primal_objective(
        reconciled,
        base,
        constraints,
        membership=membership,
    )
    if objective_after > objective_before + 1e-8:
        raise RuntimeError("FLARE-24 reconciliation increased its convex primal objective")
    simplex_error = float(np.max(np.abs(reconciled.sum(axis=1) - 1.0)))
    if not np.isfinite(reconciled).all() or (reconciled <= 0.0).any() or simplex_error > 1e-9:
        raise RuntimeError("FLARE-24 reconciliation produced invalid probabilities")
    kl_by_flight = np.sum(reconciled * (np.log(reconciled) - log_base), axis=1)
    base_z = (base_margins - constraints.mean_counts) / np.sqrt(
        constraints.variance_counts
    )
    reconciled_z = (reconciled_margins - constraints.mean_counts) / np.sqrt(
        constraints.variance_counts
    )
    dual_gradient = (
        constraints.mean_counts
        + constraints.variance_counts * dual_values
        - reconciled_margins
    )
    diagnostics = ReconciliationDiagnostics(
        converged=converged,
        optimizer_message=str(optimized.message),
        iterations=int(optimized.nit),
        objective_before=objective_before,
        objective_after=objective_after,
        mean_kl_shift=float(kl_by_flight.mean()),
        maximum_probability_shift=float(np.max(np.abs(reconciled - base))),
        base_margin_standardized_rmse=float(np.sqrt(np.mean(np.square(base_z)))),
        reconciled_margin_standardized_rmse=float(
            np.sqrt(np.mean(np.square(reconciled_z)))
        ),
        maximum_simplex_error=simplex_error,
        constraint_count=n_constraints,
        flight_count=n_flights,
        state_count=n_states,
        maximum_absolute_dual_gradient=float(np.max(np.abs(dual_gradient))),
        rms_dual_gradient=float(np.sqrt(np.mean(np.square(dual_gradient)))),
        dual_preconditioner_minimum=float(dual_scale.min()),
        dual_preconditioner_maximum=float(dual_scale.max()),
    )
    return ReconciliationResult(
        probabilities=reconciled,
        dual_values=dual_values,
        base_margins=base_margins,
        reconciled_margins=reconciled_margins,
        diagnostics=diagnostics,
    )


def constraints_from_frame(
    flights: pd.DataFrame,
    aggregate_forecasts: pd.DataFrame,
    *,
    group_columns: dict[str, tuple[str, ...]],
    state_names: tuple[str, ...] = ("on_time", "delayed", "cancelled"),
) -> ReconciliationConstraints:
    """Compile independently predicted group totals into sparse constraints.

    ``aggregate_forecasts`` has one row per group and columns named
    ``mean_<state>`` and ``variance_<state>``.  Its ``group_type`` selects the
    flight key declared in ``group_columns``.  Missing group forecasts are
    simply absent; duplicate or zero-member forecasts are rejected.
    """

    if not group_columns:
        raise ValueError("group_columns cannot be empty")
    required_forecast = {"group_type"}
    for state in state_names:
        required_forecast.update({f"mean_{state}", f"variance_{state}"})
    all_key_columns = {column for columns in group_columns.values() for column in columns}
    missing_flight = sorted(all_key_columns - set(flights.columns))
    missing_forecast = sorted(required_forecast - set(aggregate_forecasts.columns))
    if missing_flight:
        raise ValueError(f"flights are missing reconciliation keys: {missing_flight}")
    if missing_forecast:
        raise ValueError(f"aggregate forecasts are missing columns: {missing_forecast}")

    unknown_types = sorted(
        set(aggregate_forecasts["group_type"].astype(str)) - set(group_columns)
    )
    if unknown_types:
        raise ValueError(f"unknown reconciliation group_type: {unknown_types[0]}")

    names: list[str] = []
    member_arrays: list[NDArray[np.int64]] = []
    mean_rows: list[list[float]] = []
    variance_rows: list[list[float]] = []
    seen: set[str] = set()
    for group_type, keys in group_columns.items():
        forecast_rows = aggregate_forecasts.loc[
            aggregate_forecasts["group_type"].astype(str).eq(group_type)
        ]
        if forecast_rows.empty:
            continue
        missing_keys = sorted(set(keys) - set(aggregate_forecasts.columns))
        if missing_keys:
            raise ValueError(
                f"aggregate forecasts are missing {group_type} keys: {missing_keys}"
            )
        flight_members: dict[tuple[str, ...], NDArray[np.int64]] = {}
        grouped = flights.groupby(list(keys), sort=False, observed=True).indices
        for raw_key, indexes in grouped.items():
            values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            canonical = tuple(_canonical_group_value(value) for value in values)
            flight_members[canonical] = np.asarray(indexes, dtype=np.int64)

        for row_index, row in forecast_rows.iterrows():
            if pd.isna(row_index):
                raise ValueError("aggregate forecast index cannot be missing")
            values = tuple(row[key] for key in keys)
            if any(pd.isna(value) for value in values):
                missing_key = next(
                    key for key, value in zip(keys, values, strict=True) if pd.isna(value)
                )
                raise ValueError(
                    f"aggregate forecast has a missing {group_type} key: {missing_key}"
                )
            canonical = tuple(_canonical_group_value(value) for value in values)
            name = "|".join(
                [group_type]
                + [
                    f"{key}={value}"
                    for key, value in zip(keys, canonical, strict=True)
                ]
            )
            if name in seen:
                raise ValueError(f"duplicate aggregate forecast: {name}")
            seen.add(name)
            members = flight_members.get(canonical)
            if members is None or members.size == 0:
                raise ValueError(f"aggregate forecast has no matching flights: {name}")
            names.append(name)
            member_arrays.append(members)
            mean_rows.append([float(row[f"mean_{state}"]) for state in state_names])
            variance_rows.append(
                [float(row[f"variance_{state}"]) for state in state_names]
            )
    constraints = ReconciliationConstraints(
        names=tuple(names),
        members=tuple(member_arrays),
        mean_counts=np.asarray(mean_rows, dtype=np.float64),
        variance_counts=np.asarray(variance_rows, dtype=np.float64),
    )
    constraints.validate(n_flights=len(flights), n_states=len(state_names))
    return constraints


def _canonical_group_value(value: object) -> str:
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)
