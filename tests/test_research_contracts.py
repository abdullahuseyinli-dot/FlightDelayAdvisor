from __future__ import annotations

import numpy as np
import pytest

from flightdelaybench.bootstrap import paired_cluster_mean_difference
from flightdelaybench.contracts import (
    AvailabilityHorizon,
    features_available_at,
    validate_predictors,
)
from flightdelaybench.metrics import binary_metrics, multiclass_brier
from flightdelaybench.splits import DEFAULT_PROTOCOL, EvidenceRole


def test_schedule_contract_excludes_future_and_oracle_features() -> None:
    features = features_available_at(AvailabilityHorizon.SCHEDULE_CLIMATOLOGY)
    assert "prior_route_delay_rate" in features
    assert "clim_origin_tavg" in features
    assert "forecast24_origin_t2m" not in features
    assert "oracle_origin_tavg" not in features


def test_oracle_requires_two_explicit_permissions() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        validate_predictors(
            ["oracle_origin_tavg"],
            AvailabilityHorizon.SCHEDULE_CLIMATOLOGY,
        )
    with pytest.raises(ValueError, match="allow_oracle"):
        validate_predictors(
            ["oracle_origin_tavg"],
            AvailabilityHorizon.ORACLE_REALISED,
        )
    validate_predictors(
        ["oracle_origin_tavg"],
        AvailabilityHorizon.ORACLE_REALISED,
        allow_oracle=True,
    )


def test_post_event_and_unregistered_features_fail_closed() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validate_predictors(["DepDelay"], AvailabilityHorizon.FORECAST_1H)
    with pytest.raises(ValueError, match="unregistered"):
        validate_predictors(["mystery_feature"], AvailabilityHorizon.FORECAST_1H)


def test_evidence_roles_and_strictly_earlier_training_years() -> None:
    assert DEFAULT_PROTOCOL.role_for(2010) is EvidenceRole.WARMUP
    assert DEFAULT_PROTOCOL.role_for(2024) is EvidenceRole.SELECTION_CALIBRATION
    assert DEFAULT_PROTOCOL.role_for(2025) is EvidenceRole.RETROSPECTIVE_AUDIT
    assert DEFAULT_PROTOCOL.role_for(2026, 4) is EvidenceRole.LOCKED_CONFIRMATION
    assert DEFAULT_PROTOCOL.role_for(2026, 7) is EvidenceRole.OUT_OF_SCOPE
    years = DEFAULT_PROTOCOL.training_years_for(2023)
    assert years[0] == 2011
    assert years[-1] == 2022
    assert all(year < 2023 for year in years)


def test_probabilistic_metrics_reward_better_predictions() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    good = np.array([0.05, 0.10, 0.20, 0.80, 0.90, 0.95])
    weak = np.repeat(0.5, len(y))
    good_metrics = binary_metrics(y, good)
    weak_metrics = binary_metrics(y, weak)
    assert good_metrics.log_loss < weak_metrics.log_loss
    assert good_metrics.brier < weak_metrics.brier
    assert good_metrics.roc_auc == 1.0
    assert good_metrics.brier_skill > 0.0


def test_multiclass_brier_validates_probability_simplex() -> None:
    y = np.array([0, 1, 2])
    perfect = np.eye(3)
    assert multiclass_brier(y, perfect) == 0.0
    with pytest.raises(ValueError, match="sum to one"):
        multiclass_brier(y, perfect * 0.9)


def test_paired_cluster_bootstrap_is_reproducible_and_directional() -> None:
    reference = np.array([0.4, 0.5, 0.7, 0.8, 0.2, 0.3])
    candidate = reference - 0.1
    clusters = np.array(["a", "a", "b", "b", "c", "c"])
    first = paired_cluster_mean_difference(
        candidate,
        reference,
        clusters,
        repetitions=200,
        seed=12,
    )
    second = paired_cluster_mean_difference(
        candidate,
        reference,
        clusters,
        repetitions=200,
        seed=12,
    )
    assert first == second
    assert first.estimate == pytest.approx(-0.1)
    assert first.upper < 0.0

