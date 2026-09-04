"""Point-in-time feature contracts.

The registry is intentionally explicit. A column that is not registered cannot be
used merely because it happens to be present in a retrospective table.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum


class AvailabilityHorizon(IntEnum):
    """Ordered prediction horizons, from earliest/deployable to retrospective."""

    SCHEDULE_CLIMATOLOGY = 0
    FORECAST_24H = 1
    FORECAST_6H = 2
    FORECAST_1H = 3
    ORACLE_REALISED = 99


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """A declared model input and the earliest study horizon at which it is allowed."""

    name: str
    available_from: AvailabilityHorizon
    source: str
    description: str
    categorical: bool = False


_SCHEDULE = AvailabilityHorizon.SCHEDULE_CLIMATOLOGY

RECENT_WINDOWS_DAYS = (7, 28, 90)
CUTOFF_HISTORY_VIEWS = (
    "global", "route", "airline", "origin_outbound", "origin_inbound",
    "dest_inbound", "dest_outbound", "flight",
)
CUTOFF_HISTORY_FEATURES = tuple(
    f"asof_{view}_{outcome}_{statistic}_{window}d"
    for view in CUTOFF_HISTORY_VIEWS
    for outcome in ("delay", "cancel")
    for statistic in ("rate", "support_log1p")
    for window in RECENT_WINDOWS_DAYS
)
RECENT_OPERATIONAL_VIEWS = (
    "global",
    "route",
    "airline",
    "origin_outbound",
    "origin_inbound",
    "dest_inbound",
    "dest_outbound",
)
RECENT_OPERATIONAL_STATISTICS = (
    "delay_rate",
    "cancel_rate",
    "count_log1p",
    "delay_support_log1p",
)
RECENT_OPERATIONAL_FEATURES = tuple(
    f"recent_{view}_{statistic}_{window}d"
    for view in RECENT_OPERATIONAL_VIEWS
    for statistic in RECENT_OPERATIONAL_STATISTICS
    for window in RECENT_WINDOWS_DAYS
)

CENSUS_RICH_SCHEDULE_FEATURES = (
    "ArrHour",
    "CRSDepMinutes",
    "CRSArrMinutes",
    "CRSElapsedTime",
    "ScheduledFlightId",
)
CENSUS_FLIGHT_RECENT_FEATURES = tuple(
    f"recent_flight_{statistic}_{window}d"
    for statistic in RECENT_OPERATIONAL_STATISTICS
    for window in RECENT_WINDOWS_DAYS
)

CENSUS_GRAPH_MESSAGE_FEATURES = tuple(
    [
        f"graph_{side}_partner_{outcome}_{statistic}_{window}d"
        for side in ("origin", "dest")
        for outcome in ("delay", "cancel")
        for statistic in ("mean", "max", "std")
        for window in RECENT_WINDOWS_DAYS
    ]
    + [f"graph_{side}_partner_count_log1p" for side in ("origin", "dest")]
)

# Counts and concentration measures computed from all rows in the target-day
# cohort. Their interpretation depends on the source: the legacy research sample
# is only a density proxy, while the census track retains diversions but still
# needs validation against a genuine advance schedule snapshot.
SCHEDULE_CONTEXT_FEATURES = (
    "schedule_global_day_log1p",
    "schedule_airline_day_log1p",
    "schedule_route_day_log1p",
    "schedule_origin_outbound_day_log1p",
    "schedule_origin_inbound_day_log1p",
    "schedule_dest_inbound_day_log1p",
    "schedule_dest_outbound_day_log1p",
    "schedule_origin_departure_bank_log1p",
    "schedule_airline_origin_day_log1p",
    "schedule_airline_dest_day_log1p",
    "schedule_airline_network_share",
    "schedule_route_network_share",
    "schedule_origin_network_share",
    "schedule_dest_network_share",
    "schedule_origin_bank_share",
    "schedule_airline_origin_share",
    "schedule_airline_dest_share",
    "schedule_route_origin_share",
    "schedule_route_dest_share",
    "schedule_origin_flow_imbalance",
    "schedule_dest_flow_imbalance",
)

FORECAST24_DAILY_STATISTICS = (
    "tavg",
    "tmin",
    "tmax",
    "prcp_sum",
    "prcp_max",
    "rh_mean",
    "rh_max",
    "cloud_mean",
    "cloud_max",
    "pressure_mean",
    "pressure_range",
    "wspd_mean",
    "wspd_max",
    "gust_max",
    "cape_max",
    "min_variable_coverage",
    "missing",
)
FORECAST24_DAILY_FEATURES = tuple(
    f"forecast24_{side}_{statistic}"
    for side in ("origin", "dest")
    for statistic in FORECAST24_DAILY_STATISTICS
)

# FLARE-24 weather variables are selected from an issuance-aware weather cube.
# Unlike the legacy daily summaries, every value used by these features must
# have an explicit (or source-contract-implied) issue time no later than the
# flight-specific prediction cutoff.
FLARE24_WEATHER_VARIABLES = (
    "temperature",
    "precipitation",
    "humidity",
    "cloud",
    "pressure",
    "wind_speed",
    "wind_gust",
    "cape",
    "wind_direction",
    "visibility",
    "ceiling",
    "snowfall",
    "freezing_level",
    "reflectivity",
)
FLARE24_ORIGIN_DEPARTURE_FEATURES = tuple(
    f"flare24_origin_departure_{variable}" for variable in FLARE24_WEATHER_VARIABLES
)
FLARE24_DEST_ARRIVAL_FEATURES = tuple(
    f"flare24_dest_arrival_{variable}" for variable in FLARE24_WEATHER_VARIABLES
)
FLARE24_ORIGIN_WINDOW_FEATURES = (
    "flare24_origin_window_temperature_mean",
    "flare24_origin_window_temperature_change",
    "flare24_origin_window_precipitation_sum",
    "flare24_origin_window_precipitation_max",
    "flare24_origin_window_humidity_max",
    "flare24_origin_window_cloud_max",
    "flare24_origin_window_pressure_change",
    "flare24_origin_window_wind_speed_max",
    "flare24_origin_window_wind_gust_max",
    "flare24_origin_window_cape_max",
    "flare24_origin_window_visibility_min",
    "flare24_origin_window_ceiling_min",
    "flare24_origin_window_snowfall_sum",
    "flare24_origin_window_reflectivity_max",
)
FLARE24_WEATHER_DIAGNOSTIC_FEATURES = (
    "flare24_origin_selected_lead_hours",
    "flare24_dest_selected_lead_hours",
    "flare24_origin_weather_age_hours",
    "flare24_dest_weather_age_hours",
    "flare24_origin_weather_missing",
    "flare24_dest_weather_missing",
    "flare24_origin_variable_coverage",
    "flare24_dest_variable_coverage",
    "flare24_origin_window_coverage",
    "flare24_cutoff_coherent_valid",
)
FLARE24_WEATHER_INTERACTION_FEATURES = (
    "flare24_od_temperature_gap",
    "flare24_od_wind_speed_max",
    "flare24_od_wind_gust_max",
    "flare24_od_cape_max",
    "flare24_od_visibility_min",
    "flare24_od_ceiling_min",
)
FLARE24_FORECAST_REVISION_VARIABLES = (
    "temperature",
    "precipitation",
    "humidity",
    "cloud",
    "pressure",
    "wind_speed",
    "wind_gust",
    "cape",
)
FLARE24_FORECAST_REVISION_FEATURES = (
    *(
        f"flare24_origin_revision_{variable}_fresh_minus_day2"
        for variable in FLARE24_FORECAST_REVISION_VARIABLES
    ),
    "flare24_origin_revision_available",
)
FLARE24_WEATHER_FEATURES = (
    *FLARE24_ORIGIN_DEPARTURE_FEATURES,
    *FLARE24_DEST_ARRIVAL_FEATURES,
    *FLARE24_ORIGIN_WINDOW_FEATURES,
    *FLARE24_WEATHER_DIAGNOSTIC_FEATURES,
    *FLARE24_WEATHER_INTERACTION_FEATURES,
    *FLARE24_FORECAST_REVISION_FEATURES,
)
FLARE24_ROTATION_FEATURES = (
    "flare24_rotation_candidate_count",
    "flare24_rotation_predecessor_probability",
    "flare24_rotation_max_predecessor_probability",
    "flare24_rotation_entropy",
    "flare24_rotation_expected_turn_minutes",
    "flare24_rotation_tight_connection_probability",
    "flare24_rotation_competition",
    "flare24_rotation_inbound_disruption_risk",
)
FLARE24_AVIATION_WEATHER_FEATURES = (
    *(
        f"flare24_{side}_{feature}"
        for side in ("origin", "dest")
        for feature in (
            "wind_optimal_headwind_knots",
            "wind_optimal_crosswind_knots",
            "wind_optimal_gust_crosswind_knots",
            "runway_heading_available",
            "visibility_hazard",
            "low_visibility",
            "severe_visibility",
            "convective_index",
            "icing_environment_index",
            "snow_intensity",
            "gust_excess_knots",
        )
    ),
    "flare24_route_endpoint_convective_max",
    "flare24_route_endpoint_icing_max",
    "flare24_route_endpoint_crosswind_max",
    "flare24_route_endpoint_visibility_hazard_max",
    "flare24_origin_convective_bank_load",
    "flare24_origin_crosswind_bank_load",
    "flare24_dest_visibility_inbound_load",
    "flare24_dest_icing_inbound_load",
)
FLARE24_CORRIDOR_FEATURES = (
    "flare24_route_corridor_proxy_cape_max",
    "flare24_route_corridor_proxy_precipitation_max",
    "flare24_route_corridor_proxy_wind_gust_max",
    "flare24_route_corridor_proxy_visibility_min",
    "flare24_route_corridor_proxy_freezing_level_min",
    "flare24_route_corridor_proxy_convective_index_max",
    "flare24_route_corridor_proxy_icing_index_max",
    "flare24_route_corridor_proxy_coverage",
    "flare24_route_corridor_proxy_distance_mean_km",
    "flare24_route_corridor_proxy_distance_max_km",
    "flare24_route_corridor_proxy_airport_count",
)

FEATURE_REGISTRY: tuple[FeatureSpec, ...] = (
    FeatureSpec("Year", _SCHEDULE, "BTS schedule", "Calendar year"),
    FeatureSpec("Month", _SCHEDULE, "BTS schedule", "Calendar month"),
    FeatureSpec("DayOfMonth", _SCHEDULE, "BTS schedule", "Calendar day"),
    FeatureSpec("DayOfWeek", _SCHEDULE, "BTS schedule", "Scheduled day of week"),
    FeatureSpec("DayOfYear", _SCHEDULE, "derived schedule", "Calendar day of year"),
    FeatureSpec("DepHour", _SCHEDULE, "BTS schedule", "Scheduled departure hour"),
    FeatureSpec("DepHour_sin", _SCHEDULE, "derived schedule", "Cyclic departure hour"),
    FeatureSpec("DepHour_cos", _SCHEDULE, "derived schedule", "Cyclic departure hour"),
    FeatureSpec("Month_sin", _SCHEDULE, "derived schedule", "Cyclic calendar month"),
    FeatureSpec("Month_cos", _SCHEDULE, "derived schedule", "Cyclic calendar month"),
    FeatureSpec("IsWeekend", _SCHEDULE, "derived schedule", "Weekend indicator"),
    FeatureSpec(
        "IsHolidaySeason", _SCHEDULE, "derived schedule", "Prespecified holiday-season proxy"
    ),
    FeatureSpec("Distance", _SCHEDULE, "BTS schedule", "Scheduled route distance"),
    FeatureSpec("Reporting_Airline", _SCHEDULE, "BTS schedule", "Reporting carrier code", True),
    FeatureSpec("Origin", _SCHEDULE, "BTS schedule", "Origin airport code", True),
    FeatureSpec("Dest", _SCHEDULE, "BTS schedule", "Destination airport code", True),
    FeatureSpec("Route", _SCHEDULE, "derived schedule", "Directed origin-destination pair", True),
    FeatureSpec("DistanceBand", _SCHEDULE, "derived schedule", "Prespecified distance band", True),
    FeatureSpec(
        "prior_global_delay_rate",
        _SCHEDULE,
        "prior BTS outcomes",
        "Strictly earlier-year delay rate",
    ),
    FeatureSpec(
        "prior_global_cancel_rate",
        _SCHEDULE,
        "prior BTS outcomes",
        "Strictly earlier-year cancellation rate",
    ),
    FeatureSpec(
        "prior_global_count",
        _SCHEDULE,
        "prior BTS schedules",
        "Effective earlier-year global support",
    ),
    FeatureSpec(
        "prior_route_delay_rate",
        _SCHEDULE,
        "prior BTS outcomes",
        "Smoothed earlier-year route delay rate",
    ),
    FeatureSpec(
        "prior_route_cancel_rate",
        _SCHEDULE,
        "prior BTS outcomes",
        "Smoothed earlier-year route cancellation rate",
    ),
    FeatureSpec(
        "prior_route_count", _SCHEDULE, "prior BTS schedules", "Earlier-year route support"
    ),
    FeatureSpec(
        "prior_route_delay_support",
        _SCHEDULE,
        "prior BTS outcomes",
        "Earlier-year route support with observed delay outcomes",
    ),
    FeatureSpec(
        "prior_airline_delay_rate",
        _SCHEDULE,
        "prior BTS outcomes",
        "Smoothed earlier-year carrier delay rate",
    ),
    FeatureSpec(
        "prior_airline_cancel_rate",
        _SCHEDULE,
        "prior BTS outcomes",
        "Smoothed earlier-year carrier cancellation rate",
    ),
    FeatureSpec(
        "prior_airline_count", _SCHEDULE, "prior BTS schedules", "Earlier-year carrier support"
    ),
    FeatureSpec(
        "prior_airline_delay_support",
        _SCHEDULE,
        "prior BTS outcomes",
        "Earlier-year carrier support with observed delay outcomes",
    ),
    FeatureSpec(
        "prior_origin_delay_rate",
        _SCHEDULE,
        "prior BTS outcomes",
        "Smoothed earlier-year origin delay rate",
    ),
    FeatureSpec(
        "prior_origin_cancel_rate",
        _SCHEDULE,
        "prior BTS outcomes",
        "Smoothed earlier-year origin cancellation rate",
    ),
    FeatureSpec(
        "prior_origin_count", _SCHEDULE, "prior BTS schedules", "Earlier-year origin support"
    ),
    FeatureSpec(
        "prior_origin_delay_support",
        _SCHEDULE,
        "prior BTS outcomes",
        "Earlier-year origin support with observed delay outcomes",
    ),
    FeatureSpec(
        "prior_dest_delay_rate",
        _SCHEDULE,
        "prior BTS outcomes",
        "Smoothed earlier-year destination delay rate",
    ),
    FeatureSpec(
        "prior_dest_cancel_rate",
        _SCHEDULE,
        "prior BTS outcomes",
        "Smoothed earlier-year destination cancellation rate",
    ),
    FeatureSpec(
        "prior_dest_count", _SCHEDULE, "prior BTS schedules", "Earlier-year destination support"
    ),
    FeatureSpec(
        "prior_dest_delay_support",
        _SCHEDULE,
        "prior BTS outcomes",
        "Earlier-year destination support with observed delay outcomes",
    ),
    FeatureSpec(
        "prior_slot_delay_rate",
        _SCHEDULE,
        "prior BTS outcomes",
        "Smoothed origin/month/day/hour delay rate",
    ),
    FeatureSpec(
        "prior_slot_cancel_rate",
        _SCHEDULE,
        "prior BTS outcomes",
        "Smoothed origin/month/day/hour cancellation rate",
    ),
    FeatureSpec("prior_slot_count", _SCHEDULE, "prior BTS schedules", "Earlier-year slot support"),
    FeatureSpec(
        "prior_slot_delay_support",
        _SCHEDULE,
        "prior BTS outcomes",
        "Earlier-year slot support with observed delay outcomes",
    ),
    FeatureSpec(
        "clim_origin_tavg",
        _SCHEDULE,
        "prior weather",
        "Earlier-year origin monthly mean temperature",
    ),
    FeatureSpec(
        "clim_origin_prcp",
        _SCHEDULE,
        "prior weather",
        "Earlier-year origin monthly precipitation climatology",
    ),
    FeatureSpec(
        "clim_origin_snow",
        _SCHEDULE,
        "prior weather",
        "Earlier-year origin monthly snow climatology",
    ),
    FeatureSpec(
        "clim_origin_wspd",
        _SCHEDULE,
        "prior weather",
        "Earlier-year origin monthly wind climatology",
    ),
    FeatureSpec(
        "clim_dest_tavg",
        _SCHEDULE,
        "prior weather",
        "Earlier-year destination monthly mean temperature",
    ),
    FeatureSpec(
        "clim_dest_prcp",
        _SCHEDULE,
        "prior weather",
        "Earlier-year destination monthly precipitation climatology",
    ),
    FeatureSpec(
        "clim_dest_snow",
        _SCHEDULE,
        "prior weather",
        "Earlier-year destination monthly snow climatology",
    ),
    FeatureSpec(
        "clim_dest_wspd",
        _SCHEDULE,
        "prior weather",
        "Earlier-year destination monthly wind climatology",
    ),
    FeatureSpec(
        "clim_origin_tavg_missing",
        _SCHEDULE,
        "derived prior weather",
        "Origin airport-month temperature climatology required fallback",
    ),
    FeatureSpec(
        "clim_origin_prcp_missing",
        _SCHEDULE,
        "derived prior weather",
        "Origin airport-month precipitation climatology required fallback",
    ),
    FeatureSpec(
        "clim_origin_snow_missing",
        _SCHEDULE,
        "derived prior weather",
        "Origin airport-month snow climatology required fallback",
    ),
    FeatureSpec(
        "clim_origin_wspd_missing",
        _SCHEDULE,
        "derived prior weather",
        "Origin airport-month wind climatology required fallback",
    ),
    FeatureSpec(
        "clim_dest_tavg_missing",
        _SCHEDULE,
        "derived prior weather",
        "Destination airport-month temperature climatology required fallback",
    ),
    FeatureSpec(
        "clim_dest_prcp_missing",
        _SCHEDULE,
        "derived prior weather",
        "Destination airport-month precipitation climatology required fallback",
    ),
    FeatureSpec(
        "clim_dest_snow_missing",
        _SCHEDULE,
        "derived prior weather",
        "Destination airport-month snow climatology required fallback",
    ),
    FeatureSpec(
        "clim_dest_wspd_missing",
        _SCHEDULE,
        "derived prior weather",
        "Destination airport-month wind climatology required fallback",
    ),
    FeatureSpec(
        "forecast24_origin_t2m",
        AvailabilityHorizon.FORECAST_24H,
        "archived NWP forecast",
        "Origin 2 m temperature forecast",
    ),
    FeatureSpec(
        "forecast24_origin_prcp",
        AvailabilityHorizon.FORECAST_24H,
        "archived NWP forecast",
        "Origin precipitation forecast",
    ),
    FeatureSpec(
        "forecast24_origin_wspd",
        AvailabilityHorizon.FORECAST_24H,
        "archived NWP forecast",
        "Origin wind-speed forecast",
    ),
    FeatureSpec(
        "forecast24_dest_t2m",
        AvailabilityHorizon.FORECAST_24H,
        "archived NWP forecast",
        "Destination 2 m temperature forecast",
    ),
    FeatureSpec(
        "forecast24_dest_prcp",
        AvailabilityHorizon.FORECAST_24H,
        "archived NWP forecast",
        "Destination precipitation forecast",
    ),
    FeatureSpec(
        "forecast24_dest_wspd",
        AvailabilityHorizon.FORECAST_24H,
        "archived NWP forecast",
        "Destination wind-speed forecast",
    ),
    FeatureSpec(
        "forecast6_origin_t2m",
        AvailabilityHorizon.FORECAST_6H,
        "archived NWP forecast",
        "Origin 2 m temperature forecast",
    ),
    FeatureSpec(
        "forecast6_origin_prcp",
        AvailabilityHorizon.FORECAST_6H,
        "archived NWP forecast",
        "Origin precipitation forecast",
    ),
    FeatureSpec(
        "forecast6_origin_wspd",
        AvailabilityHorizon.FORECAST_6H,
        "archived NWP forecast",
        "Origin wind-speed forecast",
    ),
    FeatureSpec(
        "forecast6_dest_t2m",
        AvailabilityHorizon.FORECAST_6H,
        "archived NWP forecast",
        "Destination 2 m temperature forecast",
    ),
    FeatureSpec(
        "forecast6_dest_prcp",
        AvailabilityHorizon.FORECAST_6H,
        "archived NWP forecast",
        "Destination precipitation forecast",
    ),
    FeatureSpec(
        "forecast6_dest_wspd",
        AvailabilityHorizon.FORECAST_6H,
        "archived NWP forecast",
        "Destination wind-speed forecast",
    ),
    FeatureSpec(
        "forecast1_origin_t2m",
        AvailabilityHorizon.FORECAST_1H,
        "archived NWP forecast",
        "Origin 2 m temperature forecast",
    ),
    FeatureSpec(
        "forecast1_origin_prcp",
        AvailabilityHorizon.FORECAST_1H,
        "archived NWP forecast",
        "Origin precipitation forecast",
    ),
    FeatureSpec(
        "forecast1_origin_wspd",
        AvailabilityHorizon.FORECAST_1H,
        "archived NWP forecast",
        "Origin wind-speed forecast",
    ),
    FeatureSpec(
        "forecast1_dest_t2m",
        AvailabilityHorizon.FORECAST_1H,
        "archived NWP forecast",
        "Destination 2 m temperature forecast",
    ),
    FeatureSpec(
        "forecast1_dest_prcp",
        AvailabilityHorizon.FORECAST_1H,
        "archived NWP forecast",
        "Destination precipitation forecast",
    ),
    FeatureSpec(
        "forecast1_dest_wspd",
        AvailabilityHorizon.FORECAST_1H,
        "archived NWP forecast",
        "Destination wind-speed forecast",
    ),
    FeatureSpec(
        "oracle_origin_tavg",
        AvailabilityHorizon.ORACLE_REALISED,
        "realised daily weather",
        "Diagnostic origin observed temperature",
    ),
    FeatureSpec(
        "oracle_origin_prcp",
        AvailabilityHorizon.ORACLE_REALISED,
        "realised daily weather",
        "Diagnostic origin observed precipitation",
    ),
    FeatureSpec(
        "oracle_origin_snow",
        AvailabilityHorizon.ORACLE_REALISED,
        "realised daily weather",
        "Diagnostic origin observed snow",
    ),
    FeatureSpec(
        "oracle_origin_wspd",
        AvailabilityHorizon.ORACLE_REALISED,
        "realised daily weather",
        "Diagnostic origin observed wind speed",
    ),
    FeatureSpec(
        "oracle_dest_tavg",
        AvailabilityHorizon.ORACLE_REALISED,
        "realised daily weather",
        "Diagnostic destination observed temperature",
    ),
    FeatureSpec(
        "oracle_dest_prcp",
        AvailabilityHorizon.ORACLE_REALISED,
        "realised daily weather",
        "Diagnostic destination observed precipitation",
    ),
    FeatureSpec(
        "oracle_dest_snow",
        AvailabilityHorizon.ORACLE_REALISED,
        "realised daily weather",
        "Diagnostic destination observed snow",
    ),
    FeatureSpec(
        "oracle_dest_wspd",
        AvailabilityHorizon.ORACLE_REALISED,
        "realised daily weather",
        "Diagnostic destination observed wind speed",
    ),
)

# Historical registrations are retained to reproduce the calendar-day proxy study.
# Their horizon label is NOT timestamp evidence: [D-window,D) can include outcomes
# later than a flight's D-1 cutoff. New cutoff research excludes these columns and
# uses separately registered asof_* features with per-observation availability.
FEATURE_REGISTRY += tuple(
    FeatureSpec(
        name,
        AvailabilityHorizon.FORECAST_24H,
        "strictly prior BTS operating days",
        "Closed-left multi-timescale empirical-Bayes operating-history feature",
    )
    for name in RECENT_OPERATIONAL_FEATURES
)

FEATURE_REGISTRY += (
    FeatureSpec("ArrHour", _SCHEDULE, "BTS schedule", "Scheduled arrival hour"),
    FeatureSpec(
        "CRSDepMinutes",
        _SCHEDULE,
        "BTS schedule",
        "Scheduled local departure minutes after midnight",
    ),
    FeatureSpec(
        "CRSArrMinutes",
        _SCHEDULE,
        "BTS schedule",
        "Scheduled local arrival minutes after midnight",
    ),
    FeatureSpec(
        "CRSElapsedTime",
        _SCHEDULE,
        "BTS schedule",
        "Scheduled block time in minutes",
    ),
    FeatureSpec(
        "ScheduledFlightId",
        _SCHEDULE,
        "derived BTS schedule",
        "Reporting carrier and published flight-number identifier",
        True,
    ),
)

FEATURE_REGISTRY += tuple(
    FeatureSpec(
        name,
        AvailabilityHorizon.FORECAST_24H,
        "strictly prior BTS operating days",
        "Closed-left flight-number operating-history feature",
    )
    for name in CENSUS_FLIGHT_RECENT_FEATURES
)

FEATURE_REGISTRY += tuple(
    FeatureSpec(
        name,
        AvailabilityHorizon.FORECAST_24H,
        "target-day schedule graph with strictly prior BTS node states",
        "One-hop schedule-frequency-weighted closed-left graph message feature",
    )
    for name in CENSUS_GRAPH_MESSAGE_FEATURES
)

FEATURE_REGISTRY += tuple(
    FeatureSpec(
        name,
        AvailabilityHorizon.FORECAST_24H,
        "target-day BTS schedule-census proxy",
        "Outcome-blind schedule volume, concentration, bank, or network-flow feature",
    )
    for name in SCHEDULE_CONTEXT_FEATURES
)

FEATURE_REGISTRY += tuple(
    FeatureSpec(
        name,
        AvailabilityHorizon.FORECAST_24H,
        "Open-Meteo archived GFS fixed previous_day1 forecast",
        "Daily summary of a numerical forecast issued 24 hours before valid time",
    )
    for name in FORECAST24_DAILY_FEATURES
)

FEATURE_REGISTRY += tuple(
    FeatureSpec(
        name,
        AvailabilityHorizon.FORECAST_24H,
        "issuance-aware archived numerical weather forecast",
        (
            "FLARE-24 schedule-aligned aviation-weather feature selected only from "
            "records issued no later than the flight-specific 24-hour cutoff"
        ),
    )
    for name in FLARE24_WEATHER_FEATURES
)

FEATURE_REGISTRY += tuple(
    FeatureSpec(
        name,
        AvailabilityHorizon.FORECAST_24H,
        "schedule-only capacitated latent rotation graph",
        (
            "FLARE-24 aircraft-connection uncertainty inferred from the target-day "
            "schedule; historical tail identifiers supervise fit but are unavailable at inference"
        ),
    )
    for name in FLARE24_ROTATION_FEATURES
)

FEATURE_REGISTRY += tuple(
    FeatureSpec(
        name,
        AvailabilityHorizon.FORECAST_24H,
        "cutoff-coherent numerical forecast plus published schedule/runway geometry",
        (
            "FLARE-24 aviation transformation; runway components use a wind-optimal "
            "available-runway envelope, not a realised active-runway assignment"
        ),
    )
    for name in FLARE24_AVIATION_WEATHER_FEATURES
)

FEATURE_REGISTRY += tuple(
    FeatureSpec(
        name,
        AvailabilityHorizon.FORECAST_24H,
        "cutoff-coherent forecasts at great-circle nearest-airport corridor proxies",
        (
            "FLARE-24 route proxy sampled along the scheduled great-circle path; spatial "
            "proxy distances are retained and no exact gridded-corridor equivalence is claimed"
        ),
    )
    for name in FLARE24_CORRIDOR_FEATURES
)

FEATURE_REGISTRY += tuple(
    FeatureSpec(
        name,
        AvailabilityHorizon.FORECAST_24H,
        "timestamp-validated observations from strictly earlier operating dates",
        "History over observation-availability windows ending at the target cutoff",
    )
    for name in CUTOFF_HISTORY_FEATURES
)

FEATURE_BY_NAME = {feature.name: feature for feature in FEATURE_REGISTRY}

FORBIDDEN_PREDICTORS = frozenset(
    {
        "ArrDel15",
        "Cancelled",
        "delay_label_observed",
        "joint_label_observed",
        "disruption_state",
        "CancellationCode",
        "ArrDelay",
        "ArrDelayMinutes",
        "DepDelay",
        "DepDelayMinutes",
        "DepDel15",
        "DepTime",
        "ArrTime",
        "ActualElapsedTime",
        "AirTime",
        "TaxiIn",
        "TaxiOut",
        "CarrierDelay",
        "WeatherDelay",
        "NASDelay",
        "SecurityDelay",
        "LateAircraftDelay",
        "OriginDailyFlights",
        "OriginDailyFlightsAirline",
        "Diverted",
        "Tail_Number",
    }
)


def features_available_at(
    horizon: AvailabilityHorizon,
    *,
    include_oracle: bool = False,
) -> tuple[str, ...]:
    """Return registered features available by ``horizon``.

    Oracle variables require an explicit opt-in even if the oracle horizon is passed.
    """

    return tuple(
        feature.name
        for feature in FEATURE_REGISTRY
        if feature.available_from <= horizon
        and (include_oracle or feature.available_from is not AvailabilityHorizon.ORACLE_REALISED)
    )


def core_point_in_time_features_at(
    horizon: AvailabilityHorizon,
    *,
    include_oracle: bool = False,
) -> tuple[str, ...]:
    """Return fields materialized by the cross-cohort point-in-time builder.

    Rich census schedule columns are registered predictors, but the historical
    research-sample table does not contain them.  Keeping this capability-specific
    view separate avoids making one track's optional schema mandatory in another.
    """

    return tuple(
        name
        for name in features_available_at(horizon, include_oracle=include_oracle)
        if name not in CENSUS_RICH_SCHEDULE_FEATURES
        and name not in CUTOFF_HISTORY_FEATURES
    )


def validate_predictors(
    predictors: Iterable[str],
    horizon: AvailabilityHorizon,
    *,
    allow_oracle: bool = False,
) -> None:
    """Raise ``ValueError`` when a predictor violates the point-in-time contract."""

    requested = tuple(predictors)
    forbidden = sorted(set(requested) & FORBIDDEN_PREDICTORS)
    unknown = sorted(set(requested) - FEATURE_BY_NAME.keys())
    too_late = sorted(
        name
        for name in requested
        if name in FEATURE_BY_NAME and FEATURE_BY_NAME[name].available_from > horizon
    )
    oracle = sorted(
        name
        for name in requested
        if name in FEATURE_BY_NAME
        and FEATURE_BY_NAME[name].available_from is AvailabilityHorizon.ORACLE_REALISED
        and not allow_oracle
    )

    messages: list[str] = []
    if forbidden:
        messages.append(f"forbidden outcome/post-event predictors: {forbidden}")
    if unknown:
        messages.append(f"unregistered predictors: {unknown}")
    if too_late:
        messages.append(f"predictors unavailable at {horizon.name}: {too_late}")
    if oracle:
        messages.append(f"oracle predictors require allow_oracle=True: {oracle}")
    if messages:
        raise ValueError("; ".join(messages))
