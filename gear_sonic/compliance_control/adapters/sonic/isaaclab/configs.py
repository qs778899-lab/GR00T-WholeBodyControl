"""Opt-in config classes so release manager classes remain untouched."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils import configclass

from gear_sonic.envs.manager_env.mdp.events import EventCfg
from gear_sonic.envs.manager_env.mdp.observations import ObservationsCfg, TokenizerCfg


@configclass
class ComplianceEventsCfg(EventCfg):
    """Release events plus the explicit compliance reset slot."""

    compliance_force_reset = None


@configclass
class ComplianceTokenizerCfg(TokenizerCfg):
    """Release tokenizer observations plus one unique compliance target."""

    chip_compliance_target = None


@configclass
class ComplianceTargetObservationCfg(ObsGroup):
    """Standalone flattened hindsight target consumed only by the residual."""

    chip_compliance_target = None


@configclass
class ComplianceCommandObservationCfg(ObsGroup):
    """Actor-safe hard gate, ordered site mask, and compliance values."""

    chip_compliance_command = None


@configclass
class ComplianceForceObservationCfg(ObsGroup):
    """Privileged final applied force consumed only by the critic residual."""

    chip_compliance_force = None


@configclass
class ComplianceObservationsCfg(ObservationsCfg):
    """Release observation groups plus isolated compliance branch inputs."""

    compliance_target: ComplianceTargetObservationCfg = None
    compliance_command: ComplianceCommandObservationCfg = None
    compliance_force: ComplianceForceObservationCfg = None
