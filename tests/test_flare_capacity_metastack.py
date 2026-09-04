from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flightdelaybench.flare_capacity_metastack import (
    _frozen_metastack_implementation_files,
    _load_protocol,
    _metastack_joint,
    cancellation_logit_design,
    fit_cancellation_metastack,
)
from flightdelaybench.flare_capacity_study import BLEND_CANDIDATES
from flightdelaybench.flare_reconciliation import hurdle_joint_probabilities


def _candidate_probabilities(labels: np.ndarray) -> dict[str, np.ndarray]:
    weak_cancel = np.where(labels == 1, 0.58, 0.42)
    delay = np.full(len(labels), 0.30)
    probabilities = {
        name: hurdle_joint_probabilities(weak_cancel, delay) for name in BLEND_CANDIDATES
    }
    strong_cancel = np.where(labels == 1, 0.92, 0.08)
    probabilities["queue_shadow"] = hurdle_joint_probabilities(strong_cancel, delay)
    return probabilities


def test_cancellation_logit_design_preserves_registered_candidate_order() -> None:
    labels = np.tile([0, 1], 100)
    probabilities = _candidate_probabilities(labels)

    design = cancellation_logit_design(probabilities)

    assert design.shape == (len(labels), len(BLEND_CANDIDATES))
    queue_position = BLEND_CANDIDATES.index("queue_shadow")
    assert np.abs(design[:, queue_position]).mean() > np.abs(design[:, 0]).mean()
    with pytest.raises(ValueError, match="out of order"):
        cancellation_logit_design(dict(reversed(tuple(probabilities.items()))))


def test_cancellation_metastack_fits_signal_and_recombines_coherently() -> None:
    labels = np.tile([0, 1], 400)
    probabilities = _candidate_probabilities(labels)
    model = fit_cancellation_metastack(
        cancellation_logit_design(probabilities),
        labels,
        regularization_c=0.01,
    )

    cancellation, joint = _metastack_joint(model, probabilities)

    assert model.coef_[0, BLEND_CANDIDATES.index("queue_shadow")] > 0.0
    assert np.mean(np.where(labels == 1, cancellation, 1.0 - cancellation)) > 0.85
    assert np.allclose(joint.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    assert np.allclose(joint[:, 2], cancellation, rtol=0.0, atol=1e-12)


def test_metastack_fit_rejects_degenerate_labels() -> None:
    design = np.zeros((20, len(BLEND_CANDIDATES)))

    with pytest.raises(ValueError, match="fit inputs are invalid"):
        fit_cancellation_metastack(
            design,
            np.zeros(20, dtype=int),
            regularization_c=0.001,
        )


def test_metastack_protocol_matches_executable_contract() -> None:
    protocol, record = _load_protocol(
        Path("configs/flare24_ccrth_metastack_v1.toml"),
        repetitions=2_000,
        seed=20260905,
    )

    assert protocol["identity"]["method"] == "TF-CC-RTH-LogitStack-v1"
    assert protocol["information_boundary"]["confirmation_outcomes_accessed"] is False
    assert len(record["sha256"]) == 64


def test_metastack_provenance_binds_all_scoring_sources() -> None:
    paths = _frozen_metastack_implementation_files()
    names = {path.name for path in paths}

    assert len(names) == len(paths)
    assert all(path.is_file() for path in paths)
    assert {
        "flare_capacity_metastack.py",
        "flare_capacity_factorized.py",
        "flare_capacity_study.py",
        "flare_evaluation.py",
        "flare_reconciliation.py",
        "bootstrap.py",
        "metrics.py",
        "hashing.py",
        "provenance.py",
    }.issubset(names)
