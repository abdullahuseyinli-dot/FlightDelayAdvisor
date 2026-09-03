"""Cluster-aware paired uncertainty for decomposable proper scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class PairedInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    clusters: int
    repetitions: int
    seed: int


def paired_cluster_mean_difference(
    losses_candidate: ArrayLike,
    losses_reference: ArrayLike,
    clusters: ArrayLike,
    *,
    repetitions: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260903,
) -> PairedInterval:
    """Bootstrap candidate-minus-reference mean loss by resampling clusters.

    Cluster means are weighted by their original row counts after resampling, so the
    estimand remains the row-level mean while dependence within a cluster is retained.
    Negative estimates favour the candidate.
    """

    candidate = np.asarray(losses_candidate, dtype=np.float64)
    reference = np.asarray(losses_reference, dtype=np.float64)
    cluster_values = np.asarray(clusters)
    if not (len(candidate) == len(reference) == len(cluster_values)) or len(candidate) == 0:
        raise ValueError("loss and cluster arrays must have the same non-zero length")
    differences = candidate - reference
    if not np.isfinite(differences).all():
        raise ValueError("loss differences contain non-finite values")
    if repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions are required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    grouped = (
        pd.DataFrame({"cluster": cluster_values, "difference": differences})
        .groupby("cluster", sort=False, observed=True)["difference"]
        .agg(["sum", "count"])
    )
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.int64)
    cluster_count = len(grouped)
    if cluster_count < 2:
        raise ValueError("cluster bootstrap requires at least two clusters")

    generator = np.random.default_rng(seed)
    replicates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = generator.integers(0, cluster_count, size=cluster_count)
        replicates[index] = sums[sampled].sum() / counts[sampled].sum()

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(replicates, [alpha, 1.0 - alpha])
    return PairedInterval(
        estimate=float(differences.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        clusters=cluster_count,
        repetitions=repetitions,
        seed=seed,
    )
