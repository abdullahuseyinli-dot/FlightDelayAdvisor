"""FlightDelayBench research utilities."""

from .contracts import AvailabilityHorizon, FeatureSpec
from .splits import DEFAULT_PROTOCOL, StudyProtocol

__all__ = [
    "DEFAULT_PROTOCOL",
    "AvailabilityHorizon",
    "FeatureSpec",
    "StudyProtocol",
]

__version__ = "0.1.0.dev0"
