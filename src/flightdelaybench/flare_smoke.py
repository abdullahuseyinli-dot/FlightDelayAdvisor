"""Run a create-only FLARE-24 wheel and frozen-model inference smoke check."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .flare_reconciliation import hurdle_joint_probabilities
from .flare_study import load_enriched_month
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json


def validate_smoke_probabilities(
    delay_probability: ArrayLike,
    cancellation_probability: ArrayLike,
) -> tuple[NDArray[np.float64], dict[str, float | int]]:
    """Build and validate the three-state hurdle distribution."""

    delay = np.asarray(delay_probability, dtype=np.float64)
    cancellation = np.asarray(cancellation_probability, dtype=np.float64)
    if delay.ndim != 1 or cancellation.shape != delay.shape or len(delay) == 0:
        raise ValueError("smoke probabilities must be nonempty aligned vectors")
    if not np.isfinite(delay).all() or not np.isfinite(cancellation).all():
        raise ValueError("smoke probabilities must be finite")
    if ((delay < 0.0) | (delay > 1.0)).any() or (
        (cancellation < 0.0) | (cancellation > 1.0)
    ).any():
        raise ValueError("smoke probabilities must lie in [0, 1]")
    joint = hurdle_joint_probabilities(
        cancel_probability=cancellation,
        delay_given_operated_probability=delay,
    )
    if joint.shape != (len(delay), 3) or not np.isfinite(joint).all():
        raise ValueError("smoke joint probabilities are invalid")
    maximum_simplex_error = float(np.max(np.abs(joint.sum(axis=1) - 1.0)))
    if maximum_simplex_error > 1e-12 or (joint < 0.0).any() or (joint > 1.0).any():
        raise ValueError("smoke joint probabilities violate the simplex")
    if not np.allclose(joint[:, 2], cancellation, rtol=0.0, atol=1e-12) or not np.allclose(
        joint[:, 1], (1.0 - cancellation) * delay, rtol=0.0, atol=1e-12
    ):
        raise ValueError("smoke probabilities violate cancellation/conditional-delay semantics")
    return joint, {
        "rows": len(joint),
        "minimum_probability": float(joint.min()),
        "maximum_probability": float(joint.max()),
        "maximum_simplex_error": maximum_simplex_error,
    }


def _load_verified_model(path: Path, expected_sha256: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen FLARE-24 model: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(f"frozen model hash mismatch: {path}")
    model = joblib.load(path)
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError(f"frozen model has no predict_proba method: {path}")
    return model


def run_flare24_smoke(
    *,
    census_dir: Path,
    recent_dir: Path,
    flight_recent_dir: Path,
    graph_dir: Path,
    weather_feature_dir: Path,
    rotation_feature_dir: Path,
    delay_model_path: Path,
    cancellation_model_path: Path,
    delay_model_sha256: str,
    cancellation_model_sha256: str,
    output_path: Path,
    year: int = 2024,
    month: int = 10,
    limit: int = 128,
    seed: int = 20260904,
    execution_context: str = "source",
) -> dict[str, Any]:
    """Verify an installed package and frozen model pair on historical inputs."""

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite FLARE-24 smoke report: {output_path}")
    if year != 2024:
        raise ValueError("release smoke is deliberately restricted to the 2024 selection year")
    if month < 1 or month > 12 or limit < 1:
        raise ValueError("month must be in 1..12 and limit must be positive")
    if execution_context not in {"source", "fresh_wheel"}:
        raise ValueError("execution context must be source or fresh_wheel")

    delay_model = _load_verified_model(delay_model_path, delay_model_sha256)
    cancellation_model = _load_verified_model(
        cancellation_model_path, cancellation_model_sha256
    )
    frame = load_enriched_month(
        census_dir=census_dir,
        recent_dir=recent_dir,
        flight_recent_dir=flight_recent_dir,
        graph_dir=graph_dir,
        weather_feature_dir=weather_feature_dir,
        rotation_feature_dir=rotation_feature_dir,
        year=year,
        month=month,
        limit=limit,
        seed=seed,
    )
    delay_probability = np.asarray(delay_model.predict_proba(frame), dtype=np.float64)
    cancellation_probability = np.asarray(
        cancellation_model.predict_proba(frame), dtype=np.float64
    )
    joint, checks = validate_smoke_probabilities(delay_probability, cancellation_probability)

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS_FLARE24_FRESH_WHEEL_INFERENCE_SMOKE" if execution_context == "fresh_wheel" else "PASS_FLARE24_SOURCE_INFERENCE_SMOKE",
        "execution_context": execution_context,
        "fresh_wheel_is_caller_attestation": execution_context == "fresh_wheel",
        "package": {
            "distribution": "flightdelaybench",
            "version": importlib.metadata.version("flightdelaybench"),
            "module_path": str(Path(__file__).resolve()),
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "joblib": importlib.metadata.version("joblib"),
        },
        "sample": {
            "year": year,
            "month": month,
            "requested_limit": limit,
            "rows": len(frame),
            "seed": seed,
            "outcome_columns_used_for_prediction": False,
        },
        "models": {
            "delay": {
                "path": delay_model_path.as_posix(),
                "sha256": delay_model_sha256.lower(),
            },
            "cancellation": {
                "path": cancellation_model_path.as_posix(),
                "sha256": cancellation_model_sha256.lower(),
            },
        },
        "prediction_checks": {
            **checks,
            "joint_columns": ["on_time", "delayed", "cancelled"],
            "mean_on_time": float(joint[:, 0].mean()),
            "mean_delayed": float(joint[:, 1].mean()),
            "mean_cancelled": float(joint[:, 2].mean()),
        },
        "outcomes_accessed": {
            "2026": False,
            "maximum_calendar_year_loaded": year,
        },
        "claim_limit": (
            "This smoke check verifies package/model compatibility and numerical inference only; "
            "it does not re-estimate performance or authorize confirmation-outcome access."
        ),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-dir", type=Path, required=True)
    parser.add_argument("--recent-dir", type=Path, required=True)
    parser.add_argument("--flight-recent-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--weather-feature-dir", type=Path, required=True)
    parser.add_argument("--rotation-feature-dir", type=Path, required=True)
    parser.add_argument("--delay-model", type=Path, required=True)
    parser.add_argument("--cancellation-model", type=Path, required=True)
    parser.add_argument("--delay-model-sha256", required=True)
    parser.add_argument("--cancellation-model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=10)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--execution-context", choices=("source", "fresh_wheel"), default="source")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_flare24_smoke(
        census_dir=args.census_dir,
        recent_dir=args.recent_dir,
        flight_recent_dir=args.flight_recent_dir,
        graph_dir=args.graph_dir,
        weather_feature_dir=args.weather_feature_dir,
        rotation_feature_dir=args.rotation_feature_dir,
        delay_model_path=args.delay_model,
        cancellation_model_path=args.cancellation_model,
        delay_model_sha256=args.delay_model_sha256,
        cancellation_model_sha256=args.cancellation_model_sha256,
        output_path=args.output,
        year=args.year,
        month=args.month,
        limit=args.limit,
        seed=args.seed,
        execution_context=args.execution_context,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
