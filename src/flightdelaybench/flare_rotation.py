"""Schedule-only latent aircraft-rotation graph for FLARE-24.

Historical tail numbers supervise connection-pattern estimation, but inference
uses only the published schedule.  Candidate predecessor edges are reconciled
by an entropy-regularized matching whose inbound capacity is at most one.  This
prevents several outbound flights from independently assigning high
probability to the same inbound aircraft.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .contracts import FLARE24_ROTATION_FEATURES
from .flare_weather import scheduled_flight_times

ROTATION_FEATURES = FLARE24_ROTATION_FEATURES

_TARGET_OUTCOMES = frozenset(
    {
        "ArrDel15",
        "Cancelled",
        "disruption_state",
        "delay_label_observed",
        "joint_label_observed",
        "DepDelay",
        "ArrDelay",
    }
)


def _canonical(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _flight_number(frame: pd.DataFrame) -> pd.Series:
    for column in (
        "Flight_Number_Reporting_Airline",
        "Flight_Number_Marketing_Airline",
        "ScheduledFlightId",
    ):
        if column in frame:
            return frame[column].map(_canonical).astype("string")
    return pd.Series("", index=frame.index, dtype="string")


def _timed_schedule(
    frame: pd.DataFrame,
    *,
    timezone_by_airport: dict[str, str] | None,
) -> pd.DataFrame:
    required = {"FlightDate", "Origin", "Dest", "Reporting_Airline"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"rotation schedule is missing columns: {missing}")
    result = frame.copy()
    if {"departure_time_utc", "arrival_time_utc"}.issubset(result.columns):
        departure = pd.to_datetime(result["departure_time_utc"], errors="raise")
        arrival = pd.to_datetime(result["arrival_time_utc"], errors="raise")
    else:
        if timezone_by_airport is None:
            raise ValueError(
                "timezone_by_airport is required when UTC schedule timestamps are absent"
            )
        schedule = scheduled_flight_times(
            result,
            timezone_by_airport=timezone_by_airport,
        )
        departure = schedule["departure_time_utc"]
        arrival = schedule["arrival_time_utc"]
    if departure.isna().any() or arrival.isna().any():
        raise ValueError("rotation schedule contains unresolved timestamps")
    if (arrival <= departure).any():
        raise ValueError("rotation scheduled arrival must occur after departure")
    result["_departure_utc"] = departure.to_numpy()
    result["_arrival_utc"] = arrival.to_numpy()
    result["_carrier"] = result["Reporting_Airline"].map(_canonical).astype("string")
    result["_origin"] = result["Origin"].map(_canonical).astype("string")
    result["_dest"] = result["Dest"].map(_canonical).astype("string")
    result["_flight_number"] = _flight_number(result)
    if (result[["_carrier", "_origin", "_dest"]] == "").any().any():
        raise ValueError("rotation schedule identifiers cannot be missing")
    return result


@dataclass(frozen=True, slots=True)
class RotationModelCard:
    method: str
    historical_rows: int
    labelled_tail_rows: int
    learned_connections: int
    connection_prevalence: float
    global_turn_median_minutes: float
    global_turn_scale_minutes: float
    minimum_turn_minutes: float
    maximum_layover_minutes: float
    target_tail_number_policy: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "historical_rows": self.historical_rows,
            "labelled_tail_rows": self.labelled_tail_rows,
            "learned_connections": self.learned_connections,
            "connection_prevalence": self.connection_prevalence,
            "global_turn_median_minutes": self.global_turn_median_minutes,
            "global_turn_scale_minutes": self.global_turn_scale_minutes,
            "minimum_turn_minutes": self.minimum_turn_minutes,
            "maximum_layover_minutes": self.maximum_layover_minutes,
            "target_tail_number_policy": self.target_tail_number_policy,
        }


@dataclass(frozen=True, slots=True)
class RotationInference:
    features: pd.DataFrame
    edges: pd.DataFrame
    diagnostics: dict[str, Any]


class LatentRotationGraph:
    """Learn historical rotations and infer a schedule-only soft matching."""

    def __init__(
        self,
        *,
        minimum_turn_minutes: float = 20.0,
        maximum_layover_minutes: float = 720.0,
        tight_connection_minutes: float = 90.0,
        maximum_candidates: int = 12,
        smoothing: float = 2.0,
    ) -> None:
        numeric = (
            minimum_turn_minutes,
            maximum_layover_minutes,
            tight_connection_minutes,
            smoothing,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in numeric):
            raise ValueError("rotation parameters must be finite and positive")
        if minimum_turn_minutes >= maximum_layover_minutes:
            raise ValueError("minimum turn must be shorter than maximum layover")
        if maximum_candidates < 1:
            raise ValueError("maximum_candidates must be positive")
        self.minimum_turn_minutes = float(minimum_turn_minutes)
        self.maximum_layover_minutes = float(maximum_layover_minutes)
        self.tight_connection_minutes = float(tight_connection_minutes)
        self.maximum_candidates = int(maximum_candidates)
        self.smoothing = float(smoothing)
        self._fitted = False
        self._historical_rows = 0
        self._labelled_tail_rows = 0
        self._learned_connections = 0
        self._connection_prevalence = 0.0
        self._global_turn = (90.0, 45.0)
        self._turn_by_carrier_hub: dict[tuple[str, str], tuple[float, float, int]] = {}
        self._transition_counts: dict[tuple[str, str, str, str], int] = {}
        self._transition_parent: dict[tuple[str, str], int] = {}
        self._flight_pair_counts: dict[tuple[str, str, str], int] = {}
        self._flight_pair_parent: dict[tuple[str, str], int] = {}

    @staticmethod
    def _robust_turn(values: NDArray[np.float64]) -> tuple[float, float]:
        median = float(np.median(values))
        median_absolute_deviation = float(np.median(np.abs(values - median)))
        scale = max(1.4826 * median_absolute_deviation, 20.0)
        return median, scale

    def fit(
        self,
        history: pd.DataFrame,
        *,
        timezone_by_airport: dict[str, str] | None = None,
        tail_column: str = "Tail_Number",
    ) -> LatentRotationGraph:
        """Learn connection kernels; tail identity is used only in this fit step."""

        if tail_column not in history:
            raise ValueError(f"rotation history is missing supervision column: {tail_column}")
        timed = _timed_schedule(history, timezone_by_airport=timezone_by_airport)
        tails = timed[tail_column].map(_canonical).astype("string")
        labelled = timed.loc[tails.ne("")].copy()
        labelled["_tail"] = tails.loc[labelled.index]
        if labelled.empty:
            raise ValueError("rotation history has no labelled tail rows")
        ordered = labelled.sort_values(
            ["_carrier", "_tail", "_departure_utc"], kind="mergesort"
        ).reset_index(drop=True)
        groups = ordered.groupby(["_carrier", "_tail"], sort=False, observed=True)
        for column in (
            "_origin",
            "_dest",
            "_arrival_utc",
            "_flight_number",
        ):
            ordered[f"_previous{column}"] = groups[column].shift(1)
        slack = (
            ordered["_departure_utc"]
            - pd.to_datetime(ordered["_previous_arrival_utc"], errors="coerce")
        ).dt.total_seconds() / 60.0
        connection = (
            ordered["_previous_dest"].eq(ordered["_origin"])
            & slack.ge(self.minimum_turn_minutes)
            & slack.le(self.maximum_layover_minutes)
        )
        connected = ordered.loc[connection].copy()
        connected["_turn_minutes"] = slack.loc[connection].to_numpy(dtype=np.float64)
        if connected.empty:
            raise ValueError("rotation history contains no valid consecutive connections")
        turn_values = connected["_turn_minutes"].to_numpy(dtype=np.float64)
        self._global_turn = self._robust_turn(turn_values)
        self._turn_by_carrier_hub = {}
        for raw_key, values in connected.groupby(
            ["_carrier", "_origin"], sort=False, observed=True
        )["_turn_minutes"]:
            key = tuple(str(value) for value in raw_key)
            numeric_values = values.to_numpy(dtype=np.float64)
            median, scale = self._robust_turn(numeric_values)
            self._turn_by_carrier_hub[(key[0], key[1])] = (
                median,
                scale,
                len(numeric_values),
            )
        transition_group = connected.groupby(
            ["_carrier", "_previous_origin", "_origin", "_dest"],
            sort=False,
            observed=True,
        ).size()
        self._transition_counts = {}
        for raw_key, count in transition_group.items():
            carrier, previous_origin, hub, next_dest = (
                str(value) for value in raw_key
            )
            self._transition_counts[(carrier, previous_origin, hub, next_dest)] = int(
                count
            )
        transition_parent = connected.groupby(
            ["_carrier", "_origin"], sort=False, observed=True
        ).size()
        self._transition_parent = {}
        for raw_key, count in transition_parent.items():
            carrier, hub = (str(value) for value in raw_key)
            self._transition_parent[(carrier, hub)] = int(count)
        with_flight_numbers = connected.loc[
            connected["_previous_flight_number"].astype("string").ne("")
            & connected["_flight_number"].astype("string").ne("")
        ]
        pair_group = with_flight_numbers.groupby(
            ["_carrier", "_previous_flight_number", "_flight_number"],
            sort=False,
            observed=True,
        ).size()
        self._flight_pair_counts = {}
        for raw_key, count in pair_group.items():
            carrier, previous_flight, next_flight = (
                str(value) for value in raw_key
            )
            self._flight_pair_counts[(carrier, previous_flight, next_flight)] = int(count)
        pair_parent = with_flight_numbers.groupby(
            ["_carrier", "_flight_number"], sort=False, observed=True
        ).size()
        self._flight_pair_parent = {}
        for raw_key, count in pair_parent.items():
            carrier, next_flight = (str(value) for value in raw_key)
            self._flight_pair_parent[(carrier, next_flight)] = int(count)
        self._historical_rows = len(history)
        self._labelled_tail_rows = len(labelled)
        self._learned_connections = len(connected)
        self._connection_prevalence = len(connected) / len(labelled)
        self._fitted = True
        return self

    def model_card(self) -> RotationModelCard:
        if not self._fitted:
            raise RuntimeError("latent rotation graph has not been fitted")
        return RotationModelCard(
            method="FLARE24-CAPACITATED-LATENT-ROTATION-MATCHING",
            historical_rows=self._historical_rows,
            labelled_tail_rows=self._labelled_tail_rows,
            learned_connections=self._learned_connections,
            connection_prevalence=self._connection_prevalence,
            global_turn_median_minutes=self._global_turn[0],
            global_turn_scale_minutes=self._global_turn[1],
            minimum_turn_minutes=self.minimum_turn_minutes,
            maximum_layover_minutes=self.maximum_layover_minutes,
            target_tail_number_policy="ignored; unavailable to inference",
        )

    def _candidate_score(
        self,
        *,
        carrier: str,
        hub: str,
        previous_origin: str,
        next_dest: str,
        previous_flight: str,
        next_flight: str,
        turn_minutes: float,
    ) -> float:
        median, scale, support = self._turn_by_carrier_hub.get(
            (carrier, hub), (*self._global_turn, 0)
        )
        shrinkage = support / (support + 20.0)
        median = shrinkage * median + (1.0 - shrinkage) * self._global_turn[0]
        scale = max(shrinkage * scale + (1.0 - shrinkage) * self._global_turn[1], 20.0)
        turn_likelihood = float(np.exp(-0.5 * np.square((turn_minutes - median) / scale)))
        parent_count = self._transition_parent.get((carrier, hub), 0)
        transition_count = self._transition_counts.get(
            (carrier, previous_origin, hub, next_dest), 0
        )
        transition_rate = (transition_count + self.smoothing) / (
            parent_count + 25.0 * self.smoothing
        )
        flight_count = self._flight_pair_counts.get(
            (carrier, previous_flight, next_flight), 0
        )
        flight_parent = self._flight_pair_parent.get((carrier, next_flight), 0)
        flight_rate = (flight_count + self.smoothing) / (
            flight_parent + 10.0 * self.smoothing
        )
        prior_odds = np.clip(self._connection_prevalence, 0.01, 0.99) / np.clip(
            1.0 - self._connection_prevalence, 0.01, 0.99
        )
        score = (
            prior_odds
            * max(turn_likelihood, 1e-6)
            * (0.25 + 12.0 * transition_rate)
            * (0.5 + 5.0 * flight_rate)
        )
        return float(max(score, 1e-12))

    def _candidate_edges(self, schedule: pd.DataFrame) -> pd.DataFrame:
        predecessor_chunks: list[NDArray[np.int64]] = []
        successor_chunks: list[NDArray[np.int64]] = []
        turn_chunks: list[NDArray[np.float64]] = []
        score_chunks: list[NDArray[np.float64]] = []
        prior_odds = np.clip(self._connection_prevalence, 0.01, 0.99) / np.clip(
            1.0 - self._connection_prevalence, 0.01, 0.99
        )
        carriers = sorted(schedule["_carrier"].unique())
        for carrier in carriers:
            carrier_frame = schedule.loc[schedule["_carrier"].eq(carrier)]
            hubs = sorted(
                set(carrier_frame["_origin"].astype(str))
                & set(carrier_frame["_dest"].astype(str))
            )
            for hub in hubs:
                inbound = carrier_frame.loc[carrier_frame["_dest"].eq(hub)].sort_values(
                    "_arrival_utc", kind="mergesort"
                )
                outgoing = carrier_frame.loc[carrier_frame["_origin"].eq(hub)]
                if inbound.empty or outgoing.empty:
                    continue
                arrival_ns = inbound["_arrival_utc"].to_numpy(dtype="datetime64[ns]").astype(
                    np.int64
                )
                inbound_positions = inbound["_position"].to_numpy(dtype=np.int64)
                inbound_origins = inbound["_origin"].astype(str).to_numpy()
                inbound_flights = inbound["_flight_number"].astype(str).to_numpy()
                departure_ns = outgoing["_departure_utc"].to_numpy(
                    dtype="datetime64[ns]"
                ).astype(np.int64)
                outgoing_positions = outgoing["_position"].to_numpy(dtype=np.int64)
                outgoing_destinations = outgoing["_dest"].astype(str).to_numpy()
                outgoing_flights = outgoing["_flight_number"].astype(str).to_numpy()
                lower = departure_ns - int(self.maximum_layover_minutes * 60e9)
                upper = departure_ns - int(self.minimum_turn_minutes * 60e9)
                left_edges = np.searchsorted(arrival_ns, lower, side="left")
                right_edges = np.searchsorted(arrival_ns, upper, side="right")
                median, scale, support = self._turn_by_carrier_hub.get(
                    (str(carrier), str(hub)), (*self._global_turn, 0)
                )
                shrinkage = support / (support + 20.0)
                median = shrinkage * median + (1.0 - shrinkage) * self._global_turn[0]
                scale = max(
                    shrinkage * scale + (1.0 - shrinkage) * self._global_turn[1],
                    20.0,
                )
                transition_parent = self._transition_parent.get(
                    (str(carrier), str(hub)), 0
                )
                for outgoing_index in range(len(outgoing)):
                    right = int(right_edges[outgoing_index])
                    left = max(
                        int(left_edges[outgoing_index]),
                        right - self.maximum_candidates,
                    )
                    if right <= left:
                        continue
                    candidate_indexes = np.arange(left, right, dtype=np.int64)
                    turn = (
                        departure_ns[outgoing_index] - arrival_ns[candidate_indexes]
                    ) / 60e9
                    next_destination = outgoing_destinations[outgoing_index]
                    next_flight = outgoing_flights[outgoing_index]
                    transition_count = np.fromiter(
                        (
                            self._transition_counts.get(
                                (
                                    str(carrier),
                                    previous_origin,
                                    str(hub),
                                    next_destination,
                                ),
                                0,
                            )
                            for previous_origin in inbound_origins[candidate_indexes]
                        ),
                        dtype=np.float64,
                        count=len(candidate_indexes),
                    )
                    transition_rate = (transition_count + self.smoothing) / (
                        transition_parent + 25.0 * self.smoothing
                    )
                    flight_parent = self._flight_pair_parent.get(
                        (str(carrier), next_flight), 0
                    )
                    flight_count = np.fromiter(
                        (
                            self._flight_pair_counts.get(
                                (str(carrier), previous_flight, next_flight), 0
                            )
                            for previous_flight in inbound_flights[candidate_indexes]
                        ),
                        dtype=np.float64,
                        count=len(candidate_indexes),
                    )
                    flight_rate = (flight_count + self.smoothing) / (
                        flight_parent + 10.0 * self.smoothing
                    )
                    turn_likelihood = np.exp(
                        -0.5 * np.square((turn - median) / scale)
                    )
                    score = (
                        prior_odds
                        * np.maximum(turn_likelihood, 1e-6)
                        * (0.25 + 12.0 * transition_rate)
                        * (0.5 + 5.0 * flight_rate)
                    )
                    predecessor_chunks.append(inbound_positions[candidate_indexes])
                    successor_chunks.append(
                        np.full(
                            len(candidate_indexes),
                            outgoing_positions[outgoing_index],
                            dtype=np.int64,
                        )
                    )
                    turn_chunks.append(np.asarray(turn, dtype=np.float64))
                    score_chunks.append(
                        np.maximum(score, 1e-12).astype(np.float64, copy=False)
                    )
        if not predecessor_chunks:
            return pd.DataFrame(
                columns=(
                    "predecessor_position",
                    "successor_position",
                    "turn_minutes",
                    "score",
                ),
            )
        return pd.DataFrame(
            {
                "predecessor_position": np.concatenate(predecessor_chunks),
                "successor_position": np.concatenate(successor_chunks),
                "turn_minutes": np.concatenate(turn_chunks),
                "score": np.concatenate(score_chunks),
            }
        )

    @staticmethod
    def _capacity_match(
        edges: pd.DataFrame,
        *,
        schedule_rows: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
        if edges.empty:
            return (
                np.zeros(0, dtype=np.float64),
                np.zeros(schedule_rows, dtype=np.float64),
                {"converged": True, "iterations": 0, "maximum_inbound_mass": 0.0},
            )
        predecessor = edges["predecessor_position"].to_numpy(dtype=np.int64)
        successor = edges["successor_position"].to_numpy(dtype=np.int64)
        scores = edges["score"].to_numpy(dtype=np.float64)
        unique_predecessor, predecessor_compact = np.unique(predecessor, return_inverse=True)
        del unique_predecessor

        def objective_gradient(
            dual: NDArray[np.float64],
        ) -> tuple[float, NDArray[np.float64]]:
            adjusted = scores * np.exp(-dual[predecessor_compact])
            denominator = np.ones(schedule_rows, dtype=np.float64)
            denominator += np.bincount(successor, weights=adjusted, minlength=schedule_rows)
            probabilities = adjusted / denominator[successor]
            inbound_mass = np.bincount(
                predecessor_compact,
                weights=probabilities,
                minlength=dual.size,
            )
            objective = float(np.log(denominator).sum() + dual.sum())
            gradient = np.asarray(1.0 - inbound_mass, dtype=np.float64)
            return objective, gradient

        initial = np.zeros(int(predecessor_compact.max()) + 1, dtype=np.float64)
        optimized = minimize(
            objective_gradient,
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, None)] * initial.size,
            options={"maxiter": 1_000, "gtol": 1e-11, "ftol": 1e-15, "maxls": 100},
        )
        if not optimized.success:
            raise RuntimeError(f"latent rotation capacity solver failed: {optimized.message}")
        dual = np.asarray(optimized.x, dtype=np.float64)
        repair_iterations = 0
        repair_target_tolerance = 5e-9
        acceptance_tolerance = 1e-8
        maximum_inbound_mass_before_repair = 0.0
        while True:
            adjusted = scores * np.exp(-dual[predecessor_compact])
            denominator = np.ones(schedule_rows, dtype=np.float64)
            denominator += np.bincount(
                successor,
                weights=adjusted,
                minlength=schedule_rows,
            )
            probabilities = adjusted / denominator[successor]
            inbound_mass = np.bincount(
                predecessor_compact,
                weights=probabilities,
                minlength=dual.size,
            )
            maximum_inbound_mass = float(inbound_mass.max(initial=0.0))
            if repair_iterations == 0:
                maximum_inbound_mass_before_repair = maximum_inbound_mass
            if maximum_inbound_mass <= 1.0 + repair_target_tolerance:
                break
            if repair_iterations >= 200:
                # The repair target is intentionally tighter than the declared
                # floating-point acceptance tolerance.  Exhausting repair at a
                # residual between those bounds is acceptable; the explicit
                # post-loop gate below still rejects a material violation.
                break
            # L-BFGS can stop on relative objective tolerance while an active
            # capacity has a tiny positive residual.  Increasing only the
            # violated non-negative dual coordinates preserves the KKT sign
            # and monotonically reduces their incident edge weights.  Repeating
            # is necessary because successors can have several predecessors.
            dual += 4.0 * np.log(
                np.maximum(inbound_mass / (1.0 - 1e-10), 1.0)
            )
            repair_iterations += 1
        maximum_inbound_mass = float(inbound_mass.max(initial=0.0))
        maximum_inbound_mass_before_projection = maximum_inbound_mass
        projection_target = 1.0 - 1e-12
        projection_factors = np.ones_like(inbound_mass)
        projected = inbound_mass > 1.0
        projection_factors[projected] = (
            projection_target / inbound_mass[projected]
        )
        probability_mass_before_projection = float(probabilities.sum())
        probabilities *= projection_factors[predecessor_compact]
        inbound_mass = np.bincount(
            predecessor_compact,
            weights=probabilities,
            minlength=dual.size,
        )
        maximum_inbound_mass = float(inbound_mass.max(initial=0.0))
        if maximum_inbound_mass > 1.0 + acceptance_tolerance:
            raise RuntimeError(
                "latent rotation solution violates inbound capacity: "
                f"maximum_mass={maximum_inbound_mass:.12g}, "
                f"optimizer={optimized.message}, iterations={optimized.nit}"
            )
        outbound_mass = np.bincount(
            successor,
            weights=probabilities,
            minlength=schedule_rows,
        )
        dummy_probability = 1.0 - outbound_mass
        if (dummy_probability < -1e-12).any():
            raise RuntimeError("latent rotation feasibility projection broke outbound mass")
        dummy_probability = np.clip(dummy_probability, 0.0, 1.0)
        diagnostics = {
            "converged": True,
            "iterations": int(optimized.nit),
            "candidate_edges": len(edges),
            "capacity_dual_nonzero": int((dual > 1e-9).sum()),
            "capacity_repair_iterations": repair_iterations,
            "capacity_repair_target_tolerance": repair_target_tolerance,
            "capacity_acceptance_tolerance": acceptance_tolerance,
            "postsolve_projection_predecessors": int(projected.sum()),
            "postsolve_probability_mass_removed": (
                probability_mass_before_projection - float(probabilities.sum())
            ),
            "maximum_inbound_mass_before_projection": (
                maximum_inbound_mass_before_projection
            ),
            "maximum_inbound_mass_before_repair": (
                maximum_inbound_mass_before_repair
            ),
            "maximum_inbound_mass": maximum_inbound_mass,
        }
        return probabilities, dummy_probability, diagnostics

    def infer(
        self,
        target_schedule: pd.DataFrame,
        *,
        timezone_by_airport: dict[str, str] | None = None,
        inbound_disruption_risk: ArrayLike | None = None,
    ) -> RotationInference:
        """Infer rotations without reading target tails or realised outcomes."""

        if not self._fitted:
            raise RuntimeError("latent rotation graph has not been fitted")
        leaked = sorted(_TARGET_OUTCOMES & set(target_schedule.columns))
        if leaked:
            raise ValueError(f"target rotation schedule contains outcomes: {leaked}")
        inference_schedule = target_schedule.drop(columns=["Tail_Number"], errors="ignore")
        timed = _timed_schedule(
            inference_schedule,
            timezone_by_airport=timezone_by_airport,
        ).reset_index(drop=False, names="_original_index")
        timed["_position"] = np.arange(len(timed), dtype=np.int64)
        if inbound_disruption_risk is None:
            risk = np.full(len(timed), np.nan, dtype=np.float64)
        else:
            risk = np.asarray(inbound_disruption_risk, dtype=np.float64)
            if risk.ndim != 1 or risk.size != len(timed):
                raise ValueError("inbound disruption risk must match the target schedule")
            if not np.isfinite(risk).all() or (risk < 0.0).any() or (risk > 1.0).any():
                raise ValueError("inbound disruption risk must be finite and in [0, 1]")
        edges = self._candidate_edges(timed)
        probabilities, dummy, diagnostics = self._capacity_match(
            edges, schedule_rows=len(timed)
        )
        edges = edges.copy()
        edges["probability"] = probabilities

        candidate_count = np.zeros(len(timed), dtype=np.int32)
        predecessor_probability = np.zeros(len(timed), dtype=np.float64)
        maximum_probability = np.zeros(len(timed), dtype=np.float64)
        entropy = np.zeros(len(timed), dtype=np.float64)
        expected_turn = np.full(len(timed), np.nan, dtype=np.float64)
        tight_probability = np.zeros(len(timed), dtype=np.float64)
        competition = np.zeros(len(timed), dtype=np.float64)
        inbound_risk_feature = np.full(len(timed), np.nan, dtype=np.float64)
        if not edges.empty:
            predecessor = edges["predecessor_position"].to_numpy(dtype=np.int64)
            successor = edges["successor_position"].to_numpy(dtype=np.int64)
            turn = edges["turn_minutes"].to_numpy(dtype=np.float64)
            score = edges["score"].to_numpy(dtype=np.float64)
            probability = edges["probability"].to_numpy(dtype=np.float64)
            independent_denominator = np.ones(len(timed), dtype=np.float64)
            independent_denominator += np.bincount(
                successor, weights=score, minlength=len(timed)
            )
            independent_probability = score / independent_denominator[successor]
            independent_inbound_mass = np.bincount(
                predecessor, weights=independent_probability, minlength=len(timed)
            )
            candidate_count = np.bincount(
                successor,
                minlength=len(timed),
            ).astype(np.int32, copy=False)
            predecessor_probability = np.asarray(
                np.bincount(
                    successor,
                    weights=probability,
                    minlength=len(timed),
                ),
                dtype=np.float64,
            )
            np.maximum.at(maximum_probability, successor, probability)
            entropy = np.asarray(
                np.bincount(
                    successor,
                    weights=-probability * np.log(probability),
                    minlength=len(timed),
                ),
                dtype=np.float64,
            )
            entropy -= dummy * np.log(dummy)
            weighted_turn = np.bincount(
                successor,
                weights=probability * turn,
                minlength=len(timed),
            )
            connected = predecessor_probability > 0.0
            expected_turn[connected] = (
                weighted_turn[connected] / predecessor_probability[connected]
            )
            tight_probability = np.asarray(
                np.bincount(
                    successor,
                    weights=probability * (turn <= self.tight_connection_minutes),
                    minlength=len(timed),
                ),
                dtype=np.float64,
            )
            competition_numerator = np.bincount(
                successor,
                weights=(
                    probability
                    * np.maximum(independent_inbound_mass[predecessor] - 1.0, 0.0)
                ),
                minlength=len(timed),
            )
            competition[connected] = (
                competition_numerator[connected]
                / predecessor_probability[connected]
            )
            if np.isfinite(risk).all():
                weighted_risk = np.bincount(
                    successor,
                    weights=probability * risk[predecessor],
                    minlength=len(timed),
                )
                inbound_risk_feature[connected] = (
                    weighted_risk[connected] / predecessor_probability[connected]
                )
        feature_frame = pd.DataFrame(
            {
                "flare24_rotation_candidate_count": candidate_count,
                "flare24_rotation_predecessor_probability": predecessor_probability,
                "flare24_rotation_max_predecessor_probability": maximum_probability,
                "flare24_rotation_entropy": entropy,
                "flare24_rotation_expected_turn_minutes": expected_turn,
                "flare24_rotation_tight_connection_probability": tight_probability,
                "flare24_rotation_competition": competition,
                "flare24_rotation_inbound_disruption_risk": inbound_risk_feature,
            },
            index=timed["_original_index"],
        ).loc[:, list(ROTATION_FEATURES)]
        if not np.allclose(predecessor_probability + dummy, 1.0, atol=1e-9):
            raise RuntimeError("latent rotation outbound distributions are incoherent")
        diagnostics.update(
            {
                "schedule_rows": len(timed),
                "target_tail_number_present_but_ignored": "Tail_Number" in target_schedule,
                "target_outcomes_accessed": False,
                "inbound_risk_source": (
                    "none" if inbound_disruption_risk is None else "supplied model prediction"
                ),
            }
        )
        return RotationInference(
            features=feature_frame,
            edges=edges,
            diagnostics=diagnostics,
        )
