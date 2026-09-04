from __future__ import annotations

import numpy as np
import pytest

from flightdelaybench.calibration import (
    compose_hurdle_probabilities,
    fit_binary_calibrator,
)
from flightdelaybench.metrics import binary_metrics


@pytest.mark.parametrize("method", ["identity", "intercept", "platt", "beta", "isotonic"])
def test_calibrators_return_bounded_probabilities(method: str) -> None:
    probabilities = np.linspace(0.02, 0.98, 100)
    labels = (probabilities > 0.65).astype(int)
    calibrator = fit_binary_calibrator(method, probabilities, labels)  # type: ignore[arg-type]
    calibrated = calibrator.predict(probabilities)
    assert calibrated.shape == probabilities.shape
    assert np.isfinite(calibrated).all()
    assert ((calibrated > 0) & (calibrated < 1)).all()


def test_intercept_update_matches_observed_prevalence_and_improves_shifted_score() -> None:
    probabilities = np.repeat(0.1, 100)
    labels = np.array([1] * 30 + [0] * 70)
    calibrated = fit_binary_calibrator("intercept", probabilities, labels).predict(probabilities)
    assert calibrated.mean() == pytest.approx(0.3)
    assert (
        binary_metrics(labels, calibrated).log_loss < binary_metrics(labels, probabilities).log_loss
    )


def test_hurdle_composition_is_coherent_and_semantically_ordered() -> None:
    result = compose_hurdle_probabilities([0.1, 0.2], [0.25, 0.5])
    np.testing.assert_allclose(result.sum(axis=1), 1.0)
    np.testing.assert_allclose(result[0], [0.675, 0.225, 0.1])
    np.testing.assert_allclose(result[1], [0.4, 0.4, 0.2])
