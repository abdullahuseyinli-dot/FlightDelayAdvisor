from __future__ import annotations

import numpy as np
import pandas as pd

from flightdelaybench.flare_capacity import (
    CapacityConfig,
    build_capacity_hypergraph,
    fit_schedule_frontier,
    prepare_scheduled_events,
    resolve_scheduled_elapsed,
)
from flightdelaybench.flare_capacity_contracts import CAPACITY_ALL_FEATURES


def _schedule(year: int, days: int = 8) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day in range(1, days + 1):
        for number, (origin, dest, departure) in enumerate(
            (("AAA", "BBB", 8 * 60), ("BBB", "AAA", 8 * 60 + 20)), start=1
        ):
            rows.append(
                {
                    "sample_id": f"{year}-{day}-{number}",
                    "FlightDate": pd.Timestamp(year=year, month=1, day=day),
                    "Origin": origin,
                    "Dest": dest,
                    "CRSDepMinutes": departure,
                    "CRSElapsedTime": 60.0,
                    "Distance": 300.0,
                }
            )
    return pd.DataFrame(rows)


def _weather(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for side, timing in (("origin", "departure"), ("dest", "arrival")):
        prefix = f"flare24_{side}"
        result[f"{prefix}_{timing}_wind_direction"] = 90.0
        result[f"{prefix}_{timing}_wind_speed"] = 12.0
        result[f"{prefix}_{timing}_wind_gust"] = 18.0
        result[f"{prefix}_{timing}_precipitation"] = 0.5
        result[f"{prefix}_convective_index"] = 0.2
        result[f"{prefix}_icing_environment_index"] = 0.0
        result[f"{prefix}_wind_optimal_gust_crosswind_knots"] = 3.0
        result[f"{prefix}_gust_excess_knots"] = 2.0
        result[f"{prefix}_visibility_hazard"] = 0.1
    result["flare24_rotation_predecessor_probability"] = 0.0
    return result


def _resources() -> dict[str, dict[str, object]]:
    runway = {
        "runway_id": "09/27",
        "length_ft": 9_000.0,
        "width_ft": 150.0,
        "surface_type": "ASPH",
        "lighting_code": "HIGH",
        "ends": [
            {"runway_end_id": "09", "heading_true_degrees": 90.0, "ils_type": "ILS"},
            {"runway_end_id": "27", "heading_true_degrees": 270.0, "ils_type": "ILS"},
        ],
    }
    return {
        airport: {
            "iata": airport,
            "physical_runway_count": 1,
            "eligible_runway_end_count": 2,
            "orientation_family_count": 1,
            "max_parallel_runways": 1,
            "ils_end_fraction": 1.0,
            "minimum_runway_length_ft": 9_000.0,
            "maximum_runway_length_ft": 9_000.0,
            "annual_operations": None,
            "runways": [runway],
            "missing": False,
        }
        for airport in ("AAA", "BBB")
    }


def test_prepare_events_never_uses_outcomes() -> None:
    schedule = _schedule(2023, days=1)
    schedule["ArrDel15"] = [0.0, 1.0]
    _, first, _ = prepare_scheduled_events(
        schedule, timezone_by_airport={"AAA": "UTC", "BBB": "UTC"}
    )
    schedule["ArrDel15"] = [1.0, 0.0]
    _, second, _ = prepare_scheduled_events(
        schedule, timezone_by_airport={"AAA": "UTC", "BBB": "UTC"}
    )
    pd.testing.assert_frame_equal(first, second)


def test_missing_scheduled_elapsed_uses_distance_only() -> None:
    schedule = _schedule(2024, days=1)
    schedule.loc[0, "CRSElapsedTime"] = np.nan
    schedule["ArrDelay"] = [999.0, -999.0]
    resolved, count = resolve_scheduled_elapsed(schedule)
    assert count == 1
    assert resolved.loc[0, "CRSElapsedTime"] == 30.0 + 300.0 / 8.0
    assert resolved.loc[0, "ArrDelay"] == 999.0


def test_capacity_hypergraph_contract_and_rotation_message() -> None:
    timezones = {"AAA": "UTC", "BBB": "UTC"}
    history = _schedule(2023)
    frontier = fit_schedule_frontier(history, timezone_by_airport=timezones)
    target = _weather(_schedule(2024, days=1))
    rotation = pd.DataFrame(
        {
            "predecessor_sample_id": ["2024-1-1"],
            "successor_sample_id": ["2024-1-2"],
            "probability": [0.8],
            "turn_minutes": [80.0],
        }
    )
    result = build_capacity_hypergraph(
        target,
        target.loc[
            :,
            [
                "sample_id",
                "FlightDate",
                "Origin",
                "Dest",
                "CRSDepMinutes",
                "CRSElapsedTime",
                "Distance",
            ],
        ],
        frontier,
        timezone_by_airport=timezones,
        resource_catalog=_resources(),
        rotation_edges=rotation,
        config=CapacityConfig(),
    )
    assert tuple(result.features.columns) == ("sample_id", *CAPACITY_ALL_FEATURES)
    assert len(result.flight_nodes) == 2
    assert set(result.incidence_edges["sample_id"]) == set(target["sample_id"])
    assert result.resource_nodes["resource_node_id"].is_unique
    assert np.isclose(result.features.loc[1, "ccrth_rotation_connected_probability"], 0.8)
    assert np.isfinite(result.features.loc[1, "ccrth_rotation_predecessor_capacity_shadow_price"])
    scenario_columns = [
        "ccrth_origin_capacity_scenario_constrained_probability",
        "ccrth_origin_capacity_scenario_marginal_probability",
        "ccrth_origin_capacity_scenario_good_probability",
    ]
    assert np.allclose(result.features[scenario_columns].sum(axis=1), 1.0)


def test_boundary_context_contributes_demand_and_rotation_state() -> None:
    timezones = {"AAA": "UTC", "BBB": "UTC", "CCC": "UTC"}
    history = pd.concat(
        [
            _schedule(2023),
            pd.DataFrame(
                {
                    "sample_id": ["history-boundary"],
                    "FlightDate": [pd.Timestamp("2023-01-01")],
                    "Origin": ["CCC"],
                    "Dest": ["BBB"],
                    "CRSDepMinutes": [7 * 60],
                    "CRSElapsedTime": [60.0],
                    "Distance": [300.0],
                }
            ),
        ],
        ignore_index=True,
    )
    frontier = fit_schedule_frontier(
        history,
        timezone_by_airport=timezones,
        resource_airports={"AAA", "BBB"},
    )
    assert set(frontier["airport"]) == {"AAA", "BBB"}

    target = _weather(_schedule(2024, days=1))
    boundary = pd.DataFrame(
        {
            "sample_id": ["boundary-predecessor"],
            "FlightDate": [pd.Timestamp("2024-01-01")],
            "Origin": ["CCC"],
            "Dest": ["BBB"],
            "CRSDepMinutes": [7 * 60],
            "CRSElapsedTime": [60.0],
            "Distance": [300.0],
        }
    )
    context = pd.concat(
        [
            target.loc[
                :,
                [
                    "sample_id",
                    "FlightDate",
                    "Origin",
                    "Dest",
                    "CRSDepMinutes",
                    "CRSElapsedTime",
                    "Distance",
                ],
            ],
            boundary,
        ],
        ignore_index=True,
    )
    rotation = pd.DataFrame(
        {
            "predecessor_sample_id": ["boundary-predecessor"],
            "successor_sample_id": ["2024-1-2"],
            "probability": [0.8],
            "turn_minutes": [20.0],
        }
    )
    result = build_capacity_hypergraph(
        target,
        context,
        frontier,
        timezone_by_airport=timezones,
        resource_catalog=_resources(),
        resource_airports={"AAA", "BBB"},
        rotation_edges=rotation,
    )
    assert set(result.incidence_edges["sample_id"]) == set(target["sample_id"])
    assert set(result.resource_nodes["resource_key"]) <= {"AAA", "BBB"}
    assert np.isclose(result.features.loc[1, "ccrth_rotation_connected_probability"], 0.8)
    assert np.isclose(result.features.loc[1, "ccrth_rotation_resource_message_coverage"], 1.0)
    assert np.isfinite(result.features.loc[1, "ccrth_rotation_predecessor_capacity_shadow_price"])
    assert result.diagnostics["rotation_context_only_predecessor_states"] == 1
    assert result.diagnostics["rotation_context_only_edges"] == 1


def test_future_issued_operational_constraint_is_rejected_by_cutoff() -> None:
    timezones = {"AAA": "UTC", "BBB": "UTC"}
    history = _schedule(2023)
    frontier = fit_schedule_frontier(history, timezone_by_airport=timezones)
    target = _weather(_schedule(2024, days=1).iloc[[0]].reset_index(drop=True))
    constraints = pd.DataFrame(
        {
            "constraint_id": ["future"],
            "airport": ["AAA"],
            "direction": ["departure"],
            "issue_time_utc": ["2024-01-01T07:00:00Z"],
            "valid_from_utc": ["2024-01-01T07:00:00Z"],
            "valid_to_utc": ["2024-01-01T10:00:00Z"],
            "capacity_low": [0.0],
            "capacity_median": [0.0],
            "capacity_high": [0.0],
        }
    )
    # The flight cutoff is 2023-12-31 08:00 UTC, before the constraint issue time.
    result = build_capacity_hypergraph(
        target,
        target,
        frontier,
        timezone_by_airport=timezones,
        resource_catalog=_resources(),
        operational_constraints=constraints,
    )
    assert result.features.loc[0, "ccrth_origin_optional_constraint_available"] == 0.0
