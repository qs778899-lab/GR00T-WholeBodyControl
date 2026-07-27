"""SONIC integration helpers that keep embodiment knowledge outside the core."""

from .deployment import (
    SonicActionResidualDeployment,
    SonicResidualContextContract,
    assemble_release_action_context,
    compose_optional_sonic_action_residual,
    encode_condition,
    load_sonic_action_residual_deployment,
)
from .frames import common_to_world_vectors, world_to_common_positions
from .mapping import BodyIndexMap, resolve_body_index_map
from .state import ComplianceCommandState, ComplianceSamplingSpec
from .wrench import ResidualWrenchLimiter, WrenchLimitResult

__all__ = [
    "BodyIndexMap",
    "ComplianceCommandState",
    "ComplianceSamplingSpec",
    "ResidualWrenchLimiter",
    "SonicActionResidualDeployment",
    "SonicResidualContextContract",
    "WrenchLimitResult",
    "assemble_release_action_context",
    "common_to_world_vectors",
    "compose_optional_sonic_action_residual",
    "encode_condition",
    "load_sonic_action_residual_deployment",
    "resolve_body_index_map",
    "world_to_common_positions",
]
