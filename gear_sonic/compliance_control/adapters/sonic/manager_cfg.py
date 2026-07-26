"""IsaacLab manager configuration classes for the opt-in SONIC adapter."""

from __future__ import annotations

import dataclasses

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .command import MotionComplianceCommand


@configclass
class MotionComplianceCommandCfg(CommandTermCfg):
    """Configuration for the thin SONIC compliance command."""

    class_type: type = MotionComplianceCommand
    enabled: bool = False
    asset_name: str = "robot"
    tracking_command_name: str = "motion"
    reference_body_names: list[str] = dataclasses.MISSING
    site_body_names: list[str] = dataclasses.MISSING
    anchor_body_name: str = dataclasses.MISSING
    site_body_offsets: list[list[float]] = dataclasses.MISSING
    common_frame: str = "current_anchor"
    num_future_frames: int = dataclasses.MISSING
    seed: int = 0
    enable_probability: float = 0.75
    site_activation_probability: float = 0.5
    force_threshold_range_n: tuple[float, float] = (10.0, 20.0)
    reference_displacement_m: float = 0.05
    reference_offset_range_m: tuple[float, float] = (0.02, 0.05)
    tracking_gain_n_per_m: float = 100.0
    tracking_force_cap_n: float = 5.0
    max_net_force_n: float = 20.0
    max_net_torque_nm: float = 10.0
    resampling_time_range: tuple[float, float] = (2.0, 16.0)
    debug_vis: bool = False


@configclass
class ComplianceCommandsCfg:
    """Commands composition that leaves the existing tracking term intact."""

    motion = None
    motion_compliance = None


@configclass
class ComplianceEventsCfg:
    """Existing tracking events plus narrow compliance apply/reset terms."""

    physics_material = None
    add_joint_default_pos = None
    add_hand_joint_default_pos = None
    base_com = None
    push_robot = None
    randomize_rigid_body_mass = None
    motion_compliance_apply = None
    motion_compliance_reset = None
