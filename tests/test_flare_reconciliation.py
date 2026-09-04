from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flightdelaybench.flare_reconciliation import (
    ReconciliationConstraints,
    constraints_from_frame,
    hurdle_joint_probabilities,
    joint_binary_marginals,
    reconcile_probabilities,
)


def _one_group(variance: float) -> ReconciliationConstraints:
    return ReconciliationConstraints(
        names=("all",),
        members=(np.arange(4, dtype=np.int64),),
        mean_counts=np.array([[1.0, 2.0, 1.0]]),
        variance_counts=np.full((1, 3), variance),
    )


def test_hurdle_joint_round_trip() -> None:
    cancel = np.array([0.1, 0.25])
    conditional_delay = np.array([0.2, 0.8])
    joint = hurdle_joint_probabilities(cancel, conditional_delay)
    np.testing.assert_allclose(joint.sum(axis=1), 1.0)
    views = joint_binary_marginals(joint)
    np.testing.assert_allclose(views["cancel_probability"], cancel)
    np.testing.assert_allclose(
        views["delay_given_operated_probability"], conditional_delay
    )


def test_low_variance_moves_margins_and_improves_convex_objective() -> None:
    base = np.tile(np.array([0.8, 0.15, 0.05]), (4, 1))
    result = reconcile_probabilities(base, _one_group(variance=0.001))
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(result.reconciled_margins[0], [1.0, 2.0, 1.0], atol=0.01)
    assert result.diagnostics.converged
    assert result.diagnostics.objective_after < result.diagnostics.objective_before
    assert (
        result.diagnostics.reconciled_margin_standardized_rmse
        < result.diagnostics.base_margin_standardized_rmse
    )


def test_large_variance_has_less_influence() -> None:
    base = np.tile(np.array([0.8, 0.15, 0.05]), (4, 1))
    strong = reconcile_probabilities(base, _one_group(variance=0.01)).probabilities
    weak = reconcile_probabilities(base, _one_group(variance=1_000.0)).probabilities
    assert np.max(np.abs(strong - base)) > np.max(np.abs(weak - base)) * 100.0


def test_overlapping_reconciliation_is_permutation_equivariant() -> None:
    base = np.array(
        [
            [0.70, 0.25, 0.05],
            [0.60, 0.30, 0.10],
            [0.50, 0.35, 0.15],
            [0.75, 0.20, 0.05],
        ]
    )
    constraints = ReconciliationConstraints(
        names=("origin_A", "carrier_X"),
        members=(np.array([0, 1]), np.array([1, 2, 3])),
        mean_counts=np.array([[1.0, 0.8, 0.2], [1.5, 1.1, 0.4]]),
        variance_counts=np.full((2, 3), 0.2),
    )
    original = reconcile_probabilities(base, constraints).probabilities
    permutation = np.array([2, 0, 3, 1])
    inverse = np.argsort(permutation)
    remapped_members = tuple(
        np.flatnonzero(np.isin(permutation, member)).astype(np.int64)
        for member in constraints.members
    )
    permuted_constraints = ReconciliationConstraints(
        names=constraints.names,
        members=remapped_members,
        mean_counts=constraints.mean_counts,
        variance_counts=constraints.variance_counts,
    )
    permuted = reconcile_probabilities(
        base[permutation], permuted_constraints
    ).probabilities
    np.testing.assert_allclose(permuted[inverse], original, atol=1e-8)


def test_compile_constraints_from_multiple_group_systems() -> None:
    flights = pd.DataFrame(
        {
            "Origin": ["A", "A", "B"],
            "DepHour": [8, 8, 9],
            "Carrier": ["X", "Y", "X"],
        }
    )
    forecasts = pd.DataFrame(
        [
            {
                "group_type": "origin_hour",
                "Origin": "A",
                "DepHour": 8,
                "mean_on_time": 1.0,
                "mean_delayed": 0.8,
                "mean_cancelled": 0.2,
                "variance_on_time": 0.5,
                "variance_delayed": 0.5,
                "variance_cancelled": 0.2,
            },
            {
                "group_type": "carrier",
                "Carrier": "X",
                "mean_on_time": 1.2,
                "mean_delayed": 0.6,
                "mean_cancelled": 0.2,
                "variance_on_time": 0.5,
                "variance_delayed": 0.5,
                "variance_cancelled": 0.2,
            },
        ]
    )
    constraints = constraints_from_frame(
        flights,
        forecasts,
        group_columns={"origin_hour": ("Origin", "DepHour"), "carrier": ("Carrier",)},
    )
    assert constraints.names == (
        "origin_hour|Origin=A|DepHour=8",
        "carrier|Carrier=X",
    )
    np.testing.assert_array_equal(constraints.members[0], [0, 1])
    np.testing.assert_array_equal(constraints.members[1], [0, 2])


def test_rejects_means_that_do_not_form_group_size() -> None:
    bad = ReconciliationConstraints(
        names=("all",),
        members=(np.array([0, 1]),),
        mean_counts=np.array([[1.0, 0.5, 0.1]]),
        variance_counts=np.ones((1, 3)),
    )
    with pytest.raises(ValueError, match="sum to group size"):
        reconcile_probabilities(np.full((2, 3), 1.0 / 3.0), bad)
