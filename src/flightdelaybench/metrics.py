"""Metrics for probabilistic disruption prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    n: int
    prevalence: float
    log_loss: float
    brier: float
    brier_skill: float
    roc_auc: float
    average_precision: float
    ece_equal_mass: float
    calibration_intercept: float
    calibration_slope: float
    top_decile_rate: float
    top_decile_lift: float
    top_decile_capture: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def clip_probabilities(probabilities: ArrayLike, epsilon: float = 1e-6) -> NDArray[np.float64]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("binary probabilities must be one-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("probabilities contain non-finite values")
    return np.clip(values, epsilon, 1.0 - epsilon)


def equal_mass_ece(y_true: ArrayLike, probabilities: ArrayLike, bins: int = 15) -> float:
    """Expected calibration error using approximately equal-count bins."""

    y = np.asarray(y_true, dtype=np.int8)
    p = clip_probabilities(probabilities)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("labels and probabilities must have the same non-zero length")
    order = np.argsort(p, kind="mergesort")
    chunks = np.array_split(order, min(bins, len(order)))
    return float(
        sum(len(chunk) * abs(float(y[chunk].mean()) - float(p[chunk].mean())) for chunk in chunks)
        / len(y)
    )


def calibration_intercept_slope(
    y_true: ArrayLike,
    probabilities: ArrayLike,
) -> tuple[float, float]:
    """Estimate calibration intercept and slope by logistic recalibration."""

    y = np.asarray(y_true, dtype=np.int8)
    p = clip_probabilities(probabilities)
    if np.unique(y).size != 2:
        return float("nan"), float("nan")
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logits, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def binary_metrics(y_true: ArrayLike, probabilities: ArrayLike) -> BinaryMetrics:
    """Compute the prespecified binary probabilistic metrics."""

    y = np.asarray(y_true, dtype=np.int8)
    p = clip_probabilities(probabilities)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("labels and probabilities must have the same non-zero length")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary")

    prevalence = float(y.mean())
    brier = float(np.mean(np.square(p - y)))
    reference_brier = float(np.mean(np.square(prevalence - y)))
    brier_skill = float(1.0 - brier / reference_brier) if reference_brier > 0 else float("nan")
    intercept, slope = calibration_intercept_slope(y, p)

    top_n = max(1, int(np.ceil(len(y) * 0.10)))
    top = np.argsort(p, kind="mergesort")[-top_n:]
    top_rate = float(y[top].mean())
    positives = int(y.sum())
    capture = float(y[top].sum() / positives) if positives else float("nan")

    return BinaryMetrics(
        n=len(y),
        prevalence=prevalence,
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        brier=brier,
        brier_skill=brier_skill,
        roc_auc=float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan"),
        average_precision=float(average_precision_score(y, p)) if positives else float("nan"),
        ece_equal_mass=equal_mass_ece(y, p),
        calibration_intercept=intercept,
        calibration_slope=slope,
        top_decile_rate=top_rate,
        top_decile_lift=float(top_rate / prevalence) if prevalence > 0 else float("nan"),
        top_decile_capture=capture,
    )


def multiclass_brier(y_true: ArrayLike, probabilities: ArrayLike) -> float:
    """Mean multiclass Brier score, averaged across observations."""

    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or len(y) != p.shape[0]:
        raise ValueError("probabilities must have shape (n_observations, n_classes)")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = p.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("multiclass probability rows must sum to one")
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("class labels are outside the probability matrix")
    observed = np.eye(p.shape[1], dtype=np.float64)[y]
    return float(np.mean(np.sum(np.square(p - observed), axis=1)))
