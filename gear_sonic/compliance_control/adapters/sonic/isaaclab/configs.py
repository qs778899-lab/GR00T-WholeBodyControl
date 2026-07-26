"""Opt-in config classes so release manager classes remain untouched."""

from isaaclab.utils import configclass

from gear_sonic.envs.manager_env.mdp.events import EventCfg
from gear_sonic.envs.manager_env.mdp.observations import TokenizerCfg


@configclass
class ComplianceEventsCfg(EventCfg):
    """Release events plus explicit compliance sampling/reset slots."""

    compliance_force_push = None
    compliance_force_reset = None


@configclass
class ComplianceTokenizerCfg(TokenizerCfg):
    """Release tokenizer observations plus one unique compliance target."""

    chip_compliance_target = None
