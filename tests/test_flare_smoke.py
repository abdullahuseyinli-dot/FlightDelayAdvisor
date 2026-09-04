from __future__ import annotations

import numpy as np
import pytest

from flightdelaybench.flare_smoke import validate_smoke_probabilities


def test_validate_smoke_probabilities_builds_simplex() -> None:
    joint, checks = validate_smoke_probabilities(
        np.array([0.2, 0.8]),
        np.array([0.1, 0.25]),
    )

    assert joint.shape == (2, 3)
    np.testing.assert_allclose(joint, [[0.72, 0.18, 0.1], [0.15, 0.6, 0.25]])
    np.testing.assert_allclose(joint.sum(axis=1), 1.0)
    assert checks["rows"] == 2
    assert checks["maximum_simplex_error"] <= 1e-12


def test_smoke_preserves_endpoint_semantics_at_probability_extremes() -> None:
    joint, _ = validate_smoke_probabilities([0.2, 0.8, 0.0, 1.0], [0.01, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        joint, [[0.792, 0.198, 0.01], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )


@pytest.mark.parametrize(
    ("delay", "cancellation"),
    [
        ([], []),
        ([0.1], [0.1, 0.2]),
        ([np.nan], [0.1]),
        ([1.1], [0.1]),
    ],
)
def test_validate_smoke_probabilities_rejects_invalid_input(
    delay: list[float], cancellation: list[float]
) -> None:
    with pytest.raises(ValueError):
        validate_smoke_probabilities(delay, cancellation)
