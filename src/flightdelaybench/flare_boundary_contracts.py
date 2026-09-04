"""Feature contracts for the boundary-complete operations-twin experiment."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import AvailabilityHorizon
from .flare_capacity_contracts import (
    CAPACITY_ALL_FEATURES,
    CAPACITY_STATIC_FEATURES,
)

BOUNDARY_DYNAMIC_SOURCE_FEATURES = tuple(
    feature for feature in CAPACITY_ALL_FEATURES if feature not in CAPACITY_STATIC_FEATURES
)
BOUNDARY_FULL_FEATURES = tuple(
    f"bcpot_full__{feature.removeprefix('ccrth_')}" for feature in BOUNDARY_DYNAMIC_SOURCE_FEATURES
)
BOUNDARY_RESIDUAL_FEATURES = tuple(
    f"bcpot_residual__{feature.removeprefix('ccrth_')}"
    for feature in BOUNDARY_DYNAMIC_SOURCE_FEATURES
)
BOUNDARY_OBSERVATION_FEATURES = (
    "bcpot_rotation_predecessor_state_newly_observed",
    "bcpot_rotation_predecessor_state_lost",
)
BOUNDARY_ALL_FEATURES = (
    *BOUNDARY_FULL_FEATURES,
    *BOUNDARY_RESIDUAL_FEATURES,
    *BOUNDARY_OBSERVATION_FEATURES,
)

BOUNDARY_FULL_BY_SOURCE = dict(
    zip(BOUNDARY_DYNAMIC_SOURCE_FEATURES, BOUNDARY_FULL_FEATURES, strict=True)
)
BOUNDARY_RESIDUAL_BY_SOURCE = dict(
    zip(BOUNDARY_DYNAMIC_SOURCE_FEATURES, BOUNDARY_RESIDUAL_FEATURES, strict=True)
)


@dataclass(frozen=True, slots=True)
class BoundaryFeatureSpec:
    name: str
    available_from: AvailabilityHorizon
    source: str
    description: str


BOUNDARY_FEATURE_REGISTRY = tuple(
    BoundaryFeatureSpec(
        name=name,
        available_from=AvailabilityHorizon.FORECAST_24H,
        source=("paired induced and boundary-complete advance-schedule operations twins"),
        description=(
            "Outcome-blind boundary-complete state or its signed residual from the "
            "otherwise identical top-100-induced graph. Missing numeric states are "
            "zero-imputed only for subtraction and paired with observation-change flags. "
            "The residual is descriptive, not a causal treatment effect."
        ),
    )
    for name in BOUNDARY_ALL_FEATURES
)


def validate_boundary_predictors(
    names: tuple[str, ...] | list[str],
    horizon: AvailabilityHorizon = AvailabilityHorizon.FORECAST_24H,
) -> None:
    if len(names) != len(set(names)):
        raise ValueError("boundary predictor list contains duplicates")
    registry = {spec.name: spec for spec in BOUNDARY_FEATURE_REGISTRY}
    unknown = sorted(set(names) - set(registry))
    if unknown:
        raise ValueError(f"unregistered boundary predictors: {unknown}")
    unavailable = sorted(name for name in names if registry[name].available_from > horizon)
    if unavailable:
        raise ValueError(f"boundary predictors are unavailable at {horizon.name}: {unavailable}")


validate_boundary_predictors(list(BOUNDARY_ALL_FEATURES))
