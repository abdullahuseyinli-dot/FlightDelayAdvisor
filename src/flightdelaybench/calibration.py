"""Prospective probability calibration and coherent disruption probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CalibrationMethod = Literal["identity", "intercept", "platt", "beta", "isotonic"]


def _clip(probabilities: ArrayLike, epsilon: float = 1e-6) -> NDArray[np.float64]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("binary probabilities must be a finite one-dimensional array")
    return np.clip(values, epsilon, 1.0 - epsilon)


def _logit(probabilities: ArrayLike) -> NDArray[np.float64]:
    values = _clip(probabilities)
    return np.asarray(np.log(values / (1.0 - values)), dtype=np.float64)


class BinaryCalibrator(Protocol):
    def predict(self, probabilities: ArrayLike) -> NDArray[np.float64]: ...


@dataclass(frozen=True, slots=True)
class IdentityCalibrator:
    def predict(self, probabilities: ArrayLike) -> NDArray[np.float64]:
        return _clip(probabilities)


@dataclass(frozen=True, slots=True)
class InterceptCalibrator:
    offset: float

    def predict(self, probabilities: ArrayLike) -> NDArray[np.float64]:
        logits = _logit(probabilities) + self.offset
        return np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=np.float64)


@dataclass(slots=True)
class LogisticCalibrator:
    model: LogisticRegression
    method: Literal["platt", "beta"]

    def _matrix(self, probabilities: ArrayLike) -> NDArray[np.float64]:
        values = _clip(probabilities)
        if self.method == "platt":
            return _logit(values).reshape(-1, 1)
        return np.column_stack((np.log(values), -np.log1p(-values)))

    def predict(self, probabilities: ArrayLike) -> NDArray[np.float64]:
        return _clip(self.model.predict_proba(self._matrix(probabilities))[:, 1])


@dataclass(slots=True)
class IsotonicCalibrator:
    model: IsotonicRegression

    def predict(self, probabilities: ArrayLike) -> NDArray[np.float64]:
        return _clip(self.model.predict(_clip(probabilities)))


def _fit_intercept(probabilities: NDArray[np.float64], target_mean: float) -> float:
    logits = _logit(probabilities)
    low, high = -20.0, 20.0
    for _ in range(100):
        middle = (low + high) / 2.0
        fitted_mean = float(np.mean(1.0 / (1.0 + np.exp(-(logits + middle)))))
        if fitted_mean < target_mean:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def fit_binary_calibrator(
    method: CalibrationMethod,
    probabilities: ArrayLike,
    labels: ArrayLike,
) -> BinaryCalibrator:
    """Fit a calibrator on an earlier labeled period only."""

    values = _clip(probabilities)
    y = np.asarray(labels, dtype=np.int64)
    if len(values) != len(y) or len(y) == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("calibration labels must be aligned, non-empty, and binary")
    if method == "identity":
        return IdentityCalibrator()
    if method == "intercept":
        return InterceptCalibrator(offset=_fit_intercept(values, float(y.mean())))
    if method == "platt":
        if np.unique(y).size < 2:
            raise ValueError(f"{method} calibration requires both outcome classes")
        matrix = _logit(values).reshape(-1, 1)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(matrix, y)
        return LogisticCalibrator(model=model, method=method)
    if method == "beta":
        if np.unique(y).size < 2:
            raise ValueError(f"{method} calibration requires both outcome classes")
        matrix = np.column_stack((np.log(values), -np.log1p(-values)))
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(matrix, y)
        return LogisticCalibrator(model=model, method=method)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
        model.fit(values, y)
        return IsotonicCalibrator(model=model)
    raise ValueError(f"unknown calibration method: {method}")


def compose_hurdle_probabilities(
    cancellation_probability: ArrayLike,
    conditional_delay_probability: ArrayLike,
) -> NDArray[np.float64]:
    """Return coherent [on-time, delayed, cancelled] probabilities."""

    cancel = _clip(cancellation_probability)
    delay = _clip(conditional_delay_probability)
    if len(cancel) != len(delay):
        raise ValueError("cancellation and conditional-delay probabilities must align")
    not_cancelled = 1.0 - cancel
    result = np.column_stack((not_cancelled * (1.0 - delay), not_cancelled * delay, cancel))
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-10):
        raise AssertionError("hurdle probability composition violated the simplex")
    return np.asarray(result, dtype=np.float64)
