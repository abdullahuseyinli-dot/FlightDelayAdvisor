"""Point-in-time contracts for the capacity-conditioned FLARE hypergraph.

This registry is intentionally separate from the frozen FLARE-24 registry.  The
published FLARE-24 implementation and its checksums remain untouched while this
research extension is developed and evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import AvailabilityHorizon


@dataclass(frozen=True, slots=True)
class CapacityFeatureSpec:
    """Availability and source contract for one capacity-hypergraph feature."""

    name: str
    available_from: AvailabilityHorizon
    source: str
    description: str


SIDES = ("origin", "dest")
WINDOWS_MINUTES = (15, 30, 60, 120)
FRONTIER_QUANTILES = ("p50", "p75", "p90")

CAPACITY_STATIC_FEATURES = tuple(
    f"ccrth_{side}_{name}"
    for side in SIDES
    for name in (
        "physical_runway_count",
        "eligible_runway_end_count",
        "orientation_family_count",
        "max_parallel_runways",
        "ils_end_fraction",
        "minimum_runway_length_kft",
        "maximum_runway_length_kft",
        "airport_annual_operations_log1p",
        "runway_catalog_missing",
    )
)

CAPACITY_DEMAND_FEATURES = tuple(
    [
        f"ccrth_{side}_{direction}_demand_{window}m_log1p"
        for side in SIDES
        for direction in ("arrival", "departure", "movement")
        for window in WINDOWS_MINUTES
    ]
    + [
        f"ccrth_{side}_{name}"
        for side in SIDES
        for name in (
            "same_direction_loo_30m_log1p",
            "movement_loo_30m_log1p",
            "minutes_since_previous_same_direction_log1p",
            "minutes_until_next_same_direction_log1p",
            "flow_imbalance_60m",
            "burstiness_15_to_60",
            "metro_movement_demand_60m_log1p",
        )
    ]
)

CAPACITY_STATE_FEATURES = tuple(
    [
        f"ccrth_{side}_{kind}_frontier_{quantile}"
        for side in SIDES
        for kind in ("direction", "movement")
        for quantile in FRONTIER_QUANTILES
    ]
    + [
        f"ccrth_{side}_{name}"
        for side in SIDES
        for name in (
            "historical_frontier_support_days_log1p",
            "historical_frontier_observed_fraction",
            "runway_configuration_entropy",
            "runway_effective_end_count",
            "runway_feasible_end_fraction",
            "weather_capacity_stress",
            "capacity_scenario_good_probability",
            "capacity_scenario_marginal_probability",
            "capacity_scenario_constrained_probability",
            "expected_direction_capacity",
            "expected_movement_capacity",
            "direction_utilization",
            "movement_utilization",
            "overload_probability",
            "expected_slack",
            "expected_queue",
            "queue_p90",
            "recovery_minutes",
            "shadow_price",
            "marginal_overload",
            "metro_utilization",
            "metro_queue",
            "optional_constraint_available",
            "optional_constraint_overload_probability",
        )
    ]
)

CAPACITY_ROTATION_MESSAGE_FEATURES = (
    "ccrth_rotation_connected_probability",
    "ccrth_rotation_resource_message_coverage",
    "ccrth_rotation_predecessor_capacity_overload",
    "ccrth_rotation_predecessor_capacity_queue",
    "ccrth_rotation_predecessor_capacity_shadow_price",
)

CAPACITY_ROUTE_FEATURES = (
    "ccrth_route_max_overload_probability",
    "ccrth_route_max_queue_p90",
    "ccrth_route_min_expected_slack",
    "ccrth_route_sum_shadow_price",
    "ccrth_route_bottleneck_is_destination",
    "ccrth_route_max_weather_capacity_stress",
    "ccrth_route_max_metro_utilization",
)

CAPACITY_ALL_FEATURES = (
    *CAPACITY_STATIC_FEATURES,
    *CAPACITY_DEMAND_FEATURES,
    *CAPACITY_STATE_FEATURES,
    *CAPACITY_ROTATION_MESSAGE_FEATURES,
    *CAPACITY_ROUTE_FEATURES,
)


def _source_for(name: str) -> tuple[str, str]:
    if any(token in name for token in ("runway_", "physical_runway", "annual_operations")):
        return (
            "cycle-dated FAA NASR static airport/runway geometry",
            "Static airport resource geometry; not the realised active configuration.",
        )
    if "rotation_" in name:
        return (
            "prior-year-supervised, target-year schedule-only latent rotation graph",
            "One-hop probability-weighted message from candidate predecessor resources.",
        )
    if "optional_constraint" in name:
        return (
            "optional issue-time-vintaged operational constraint adapter",
            "Constraint admitted only when issued no later than the flight cutoff.",
        )
    if any(token in name for token in ("demand_", "flow_", "burstiness", "minutes_")):
        return (
            "target-day advance-schedule proxy",
            "Outcome-blind airport resource demand computed before row sampling.",
        )
    if "frontier" in name:
        return (
            "strictly prior-year schedule",
            "Hierarchically backed-off empirical scheduling frontier, not declared AAR/ADR.",
        )
    return (
        "capacity-conditioned resource-time hypergraph",
        "Derived from registered schedule, prior-year frontier, static geometry, and T-24 weather.",
    )


CAPACITY_FEATURE_REGISTRY = tuple(
    CapacityFeatureSpec(
        name=name,
        available_from=AvailabilityHorizon.FORECAST_24H,
        source=_source_for(name)[0],
        description=_source_for(name)[1],
    )
    for name in CAPACITY_ALL_FEATURES
)


def validate_capacity_predictors(
    names: tuple[str, ...] | list[str],
    horizon: AvailabilityHorizon = AvailabilityHorizon.FORECAST_24H,
) -> None:
    """Reject unregistered, duplicate, or too-late capacity predictors."""

    if len(names) != len(set(names)):
        raise ValueError("capacity predictor list contains duplicates")
    registry = {spec.name: spec for spec in CAPACITY_FEATURE_REGISTRY}
    unknown = sorted(set(names) - set(registry))
    if unknown:
        raise ValueError(f"unregistered capacity predictors: {unknown}")
    unavailable = sorted(
        name for name in names if registry[name].available_from > horizon
    )
    if unavailable:
        raise ValueError(
            f"capacity predictors are unavailable at {horizon.name}: {unavailable}"
        )


validate_capacity_predictors(list(CAPACITY_ALL_FEATURES))
