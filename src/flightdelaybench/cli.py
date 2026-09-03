"""Command-line entry point for lightweight contract inspection."""

from __future__ import annotations

import argparse
import json

from .contracts import AvailabilityHorizon, features_available_at
from .splits import DEFAULT_PROTOCOL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flightdelaybench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    features = subparsers.add_parser("features", help="list allowed features at a horizon")
    features.add_argument(
        "horizon",
        choices=[horizon.name.lower() for horizon in AvailabilityHorizon],
    )

    subparsers.add_parser("protocol", help="print the built-in split contract")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "features":
        horizon = AvailabilityHorizon[args.horizon.upper()]
        print(json.dumps(features_available_at(horizon), indent=2))
        return
    if args.command == "protocol":
        payload = {
            "warmup_year": DEFAULT_PROTOCOL.warmup_year,
            "rolling_evaluation_years": DEFAULT_PROTOCOL.rolling_evaluation_years,
            "selection_year": DEFAULT_PROTOCOL.selection_year,
            "retrospective_year": DEFAULT_PROTOCOL.retrospective_year,
            "confirmation_year": DEFAULT_PROTOCOL.confirmation_year,
            "confirmation_months": DEFAULT_PROTOCOL.confirmation_months,
        }
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
