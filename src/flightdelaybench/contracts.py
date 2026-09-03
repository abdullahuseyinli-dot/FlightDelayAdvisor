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
    FeatureSpec("IsHolidaySeason", _SCHEDULE, "derived schedule", "Prespecified holiday-season proxy"),
    FeatureSpec("Distance", _SCHEDULE, "BTS schedule", "Scheduled route distance"),
    FeatureSpec("Reporting_Airline", _SCHEDULE, "BTS schedule", "Reporting carrier code", True),
    FeatureSpec("Origin", _SCHEDULE, "BTS schedule", "Origin airport code", True),
    FeatureSpec("Dest", _SCHEDULE, "BTS schedule", "Destination airport code", True),
    FeatureSpec("Route", _SCHEDULE, "derived schedule", "Directed origin-destination pair", True),
    FeatureSpec("DistanceBand", _SCHEDULE, "derived schedule", "Prespecified distance band", True),
    FeatureSpec("prior_global_delay_rate", _SCHEDULE, "prior BTS outcomes", "Strictly earlier-year delay rate"),
    FeatureSpec("prior_global_cancel_rate", _SCHEDULE, "prior BTS outcomes", "Strictly earlier-year cancellation rate"),
    FeatureSpec("prior_route_delay_rate", _SCHEDULE, "prior BTS outcomes", "Smoothed earlier-year route delay rate"),
    FeatureSpec("prior_route_cancel_rate", _SCHEDULE, "prior BTS outcomes", "Smoothed earlier-year route cancellation rate"),
    FeatureSpec("prior_route_count", _SCHEDULE, "prior BTS schedules", "Earlier-year route support"),
    FeatureSpec("prior_airline_delay_rate", _SCHEDULE, "prior BTS outcomes", "Smoothed earlier-year carrier delay rate"),
    FeatureSpec("prior_airline_cancel_rate", _SCHEDULE, "prior BTS outcomes", "Smoothed earlier-year carrier cancellation rate"),
    FeatureSpec("prior_airline_count", _SCHEDULE, "prior BTS schedules", "Earlier-year carrier support"),
    FeatureSpec("prior_origin_delay_rate", _SCHEDULE, "prior BTS outcomes", "Smoothed earlier-year origin delay rate"),
    FeatureSpec("prior_origin_cancel_rate", _SCHEDULE, "prior BTS outcomes", "Smoothed earlier-year origin cancellation rate"),
    FeatureSpec("prior_origin_count", _SCHEDULE, "prior BTS schedules", "Earlier-year origin support"),
    FeatureSpec("prior_dest_delay_rate", _SCHEDULE, "prior BTS outcomes", "Smoothed earlier-year destination delay rate"),
    FeatureSpec("prior_dest_cancel_rate", _SCHEDULE, "prior BTS outcomes", "Smoothed earlier-year destination cancellation rate"),
    FeatureSpec("prior_dest_count", _SCHEDULE, "prior BTS schedules", "Earlier-year destination support"),
    FeatureSpec("prior_slot_delay_rate", _SCHEDULE, "prior BTS outcomes", "Smoothed origin/month/day/hour delay rate"),
    FeatureSpec("prior_slot_cancel_rate", _SCHEDULE, "prior BTS outcomes", "Smoothed origin/month/day/hour cancellation rate"),
    FeatureSpec("prior_slot_count", _SCHEDULE, "prior BTS schedules", "Earlier-year slot support"),
    FeatureSpec("clim_origin_tavg", _SCHEDULE, "prior weather", "Earlier-year origin monthly mean temperature"),
    FeatureSpec("clim_origin_prcp", _SCHEDULE, "prior weather", "Earlier-year origin monthly precipitation climatology"),
    FeatureSpec("clim_origin_snow", _SCHEDULE, "prior weather", "Earlier-year origin monthly snow climatology"),
    FeatureSpec("clim_origin_wspd", _SCHEDULE, "prior weather", "Earlier-year origin monthly wind climatology"),
    FeatureSpec("clim_dest_tavg", _SCHEDULE, "prior weather", "Earlier-year destination monthly mean temperature"),
    FeatureSpec("clim_dest_prcp", _SCHEDULE, "prior weather", "Earlier-year destination monthly precipitation climatology"),
    FeatureSpec("clim_dest_snow", _SCHEDULE, "prior weather", "Earlier-year destination monthly snow climatology"),
    FeatureSpec("clim_dest_wspd", _SCHEDULE, "prior weather", "Earlier-year destination monthly wind climatology"),
    FeatureSpec("forecast24_origin_t2m", AvailabilityHorizon.FORECAST_24H, "archived NWP forecast", "Origin 2 m temperature forecast"),
    FeatureSpec("forecast24_origin_prcp", AvailabilityHorizon.FORECAST_24H, "archived NWP forecast", "Origin precipitation forecast"),
    FeatureSpec("forecast24_origin_wspd", AvailabilityHorizon.FORECAST_24H, "archived NWP forecast", "Origin wind-speed forecast"),
    FeatureSpec("forecast24_dest_t2m", AvailabilityHorizon.FORECAST_24H, "archived NWP forecast", "Destination 2 m temperature forecast"),
    FeatureSpec("forecast24_dest_prcp", AvailabilityHorizon.FORECAST_24H, "archived NWP forecast", "Destination precipitation forecast"),
    FeatureSpec("forecast24_dest_wspd", AvailabilityHorizon.FORECAST_24H, "archived NWP forecast", "Destination wind-speed forecast"),
    FeatureSpec("forecast6_origin_t2m", AvailabilityHorizon.FORECAST_6H, "archived NWP forecast", "Origin 2 m temperature forecast"),
    FeatureSpec("forecast6_origin_prcp", AvailabilityHorizon.FORECAST_6H, "archived NWP forecast", "Origin precipitation forecast"),
    FeatureSpec("forecast6_origin_wspd", AvailabilityHorizon.FORECAST_6H, "archived NWP forecast", "Origin wind-speed forecast"),
    FeatureSpec("forecast6_dest_t2m", AvailabilityHorizon.FORECAST_6H, "archived NWP forecast", "Destination 2 m temperature forecast"),
    FeatureSpec("forecast6_dest_prcp", AvailabilityHorizon.FORECAST_6H, "archived NWP forecast", "Destination precipitation forecast"),
    FeatureSpec("forecast6_dest_wspd", AvailabilityHorizon.FORECAST_6H, "archived NWP forecast", "Destination wind-speed forecast"),
    FeatureSpec("forecast1_origin_t2m", AvailabilityHorizon.FORECAST_1H, "archived NWP forecast", "Origin 2 m temperature forecast"),
    FeatureSpec("forecast1_origin_prcp", AvailabilityHorizon.FORECAST_1H, "archived NWP forecast", "Origin precipitation forecast"),
    FeatureSpec("forecast1_origin_wspd", AvailabilityHorizon.FORECAST_1H, "archived NWP forecast", "Origin wind-speed forecast"),
    FeatureSpec("forecast1_dest_t2m", AvailabilityHorizon.FORECAST_1H, "archived NWP forecast", "Destination 2 m temperature forecast"),
    FeatureSpec("forecast1_dest_prcp", AvailabilityHorizon.FORECAST_1H, "archived NWP forecast", "Destination precipitation forecast"),
    FeatureSpec("forecast1_dest_wspd", AvailabilityHorizon.FORECAST_1H, "archived NWP forecast", "Destination wind-speed forecast"),
    FeatureSpec("oracle_origin_tavg", AvailabilityHorizon.ORACLE_REALISED, "realised daily weather", "Diagnostic origin observed temperature"),
    FeatureSpec("oracle_origin_prcp", AvailabilityHorizon.ORACLE_REALISED, "realised daily weather", "Diagnostic origin observed precipitation"),
    FeatureSpec("oracle_origin_snow", AvailabilityHorizon.ORACLE_REALISED, "realised daily weather", "Diagnostic origin observed snow"),
    FeatureSpec("oracle_origin_wspd", AvailabilityHorizon.ORACLE_REALISED, "realised daily weather", "Diagnostic origin observed wind speed"),
    FeatureSpec("oracle_dest_tavg", AvailabilityHorizon.ORACLE_REALISED, "realised daily weather", "Diagnostic destination observed temperature"),
    FeatureSpec("oracle_dest_prcp", AvailabilityHorizon.ORACLE_REALISED, "realised daily weather", "Diagnostic destination observed precipitation"),
    FeatureSpec("oracle_dest_snow", AvailabilityHorizon.ORACLE_REALISED, "realised daily weather", "Diagnostic destination observed snow"),
    FeatureSpec("oracle_dest_wspd", AvailabilityHorizon.ORACLE_REALISED, "realised daily weather", "Diagnostic destination observed wind speed"),
)

FEATURE_BY_NAME = {feature.name: feature for feature in FEATURE_REGISTRY}

FORBIDDEN_PREDICTORS = frozenset(
    {
        "ArrDel15",
        "Cancelled",
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
        if name in FEATURE_BY_NAME
        and FEATURE_BY_NAME[name].available_from > horizon
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
