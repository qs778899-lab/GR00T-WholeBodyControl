"""SONIC integration helpers that keep embodiment knowledge outside the core."""

from .frames import common_to_world_vectors, world_to_common_positions
from .mapping import BodyIndexMap, resolve_body_index_map
from .state import ComplianceCommandState, ComplianceSamplingSpec
from .wrench import ResidualWrenchLimiter, WrenchLimitResult

__all__ = [
    "BodyIndexMap",
    "ComplianceCommandState",
    "ComplianceSamplingSpec",
    "ResidualWrenchLimiter",
    "WrenchLimitResult",
    "common_to_world_vectors",
    "resolve_body_index_map",
    "world_to_common_positions",
]
