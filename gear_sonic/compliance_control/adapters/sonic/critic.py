"""Thin SONIC critic glue with a privileged compliance-only value residual."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from gear_sonic.trl.modules.actor_critic_modules import Critic

from ...core import ComplianceResidualMLP
from .checkpoint import (
    CheckpointMigrationReport,
    classify_checkpoint_state,
    migrate_legacy_state_dict,
)
from .policy import _feature_width, _instantiate_residual, _ordered_site_names


class SonicComplianceCritic(Critic):
    """Unchanged release value path plus a gated privileged scalar branch."""

    _NEW_STATE_PREFIXES = ("compliance_value_residual.",)

    def __init__(
        self,
        *args,
        compliance_site_names: Sequence[str],
        compliance_cartesian_dim: int,
        compliance_num_future_frames: int,
        compliance_value_residual,
        compliance_target_key: str = "compliance_target",
        compliance_command_key: str = "compliance_command",
        compliance_force_key: str = "compliance_force",
        freeze_base_critic: bool = True,
        allow_legacy_checkpoint_migration: bool = False,
        **kwargs,
    ) -> None:
        obs_dim_dict = kwargs.get("obs_dim_dict")
        if obs_dim_dict is None:
            env_config = kwargs.get("env_config")
            if env_config is None and args:
                env_config = args[0]
            obs_dim_dict = env_config.robot.algo_obs_dim_dict
        super().__init__(*args, **kwargs)
        if type(freeze_base_critic) is not bool:
            raise TypeError("freeze_base_critic must be a bool")
        self.freeze_base_critic = freeze_base_critic
        if self.freeze_base_critic:
            for parameter in self.parameters():
                parameter.requires_grad = False
            if self.running_mean_std is not None:
                self.running_mean_std.freeze()

        self.compliance_site_names = _ordered_site_names(compliance_site_names)
        if type(compliance_cartesian_dim) is not int or compliance_cartesian_dim <= 0:
            raise ValueError("compliance_cartesian_dim must be a positive integer")
        if type(compliance_num_future_frames) is not int or compliance_num_future_frames <= 0:
            raise ValueError("compliance_num_future_frames must be a positive integer")
        self.compliance_target_key = compliance_target_key
        self.compliance_command_key = compliance_command_key
        self.compliance_force_key = compliance_force_key
        target_dim = (
            compliance_num_future_frames
            * len(self.compliance_site_names)
            * compliance_cartesian_dim
        )
        force_dim = len(self.compliance_site_names) * compliance_cartesian_dim
        command_dim = 1 + len(self.compliance_site_names) + force_dim
        if _feature_width(obs_dim_dict, compliance_target_key) != target_dim:
            raise ValueError("compliance target observation width does not match site layout")
        if _feature_width(obs_dim_dict, compliance_command_key) != command_dim:
            raise ValueError("compliance command observation width does not match site layout")
        if _feature_width(obs_dim_dict, compliance_force_key) != force_dim:
            raise ValueError("compliance force observation width does not match site layout")
        critic_context_dim = _feature_width(obs_dim_dict, "critic_obs")
        self.compliance_value_residual = _instantiate_residual(
            compliance_value_residual,
            condition_dim=target_dim + force_dim,
            num_sites=len(self.compliance_site_names),
            cartesian_dim=compliance_cartesian_dim,
            context_dim=critic_context_dim,
            output_dim=1,
        )
        self.allow_legacy_checkpoint_migration = bool(allow_legacy_checkpoint_migration)
        self._legacy_migration_consumed = False
        self.last_migration_report: CheckpointMigrationReport | None = None

    def evaluate(self, obs_dict, **kwargs):
        """Normalize once, then share one critic context across both paths."""

        normalized_obs = obs_dict.copy()
        if self.running_mean_std is not None:
            if self.use_batch_norm:
                normalized_obs["critic_obs"] = self.running_mean_std(
                    normalized_obs["critic_obs"]
                )
            else:
                with torch.no_grad():
                    normalized_obs["critic_obs"] = self.running_mean_std(
                        normalized_obs["critic_obs"]
                    )
        base_value = self.critic(normalized_obs, **kwargs)
        condition = torch.cat(
            (
                normalized_obs[self.compliance_target_key],
                normalized_obs[self.compliance_force_key],
            ),
            dim=-1,
        )
        residual = self.compliance_value_residual(
            condition,
            normalized_obs[self.compliance_command_key],
            normalized_obs["critic_obs"],
        )
        return base_value + residual

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Migrate one released state or strictly resume a migrated state."""

        _, _, source_new = classify_checkpoint_state(
            self,
            state_dict,
            new_key_prefixes=self._NEW_STATE_PREFIXES,
        )
        if source_new:
            return nn.Module.load_state_dict(self, state_dict, strict=True, assign=assign)
        if not self.allow_legacy_checkpoint_migration:
            raise RuntimeError("legacy checkpoint migration is disabled for this critic")
        if self._legacy_migration_consumed:
            raise RuntimeError("legacy checkpoint migration may be used only once")
        report, result = migrate_legacy_state_dict(
            self,
            state_dict,
            new_key_prefixes=self._NEW_STATE_PREFIXES,
            assign=assign,
        )
        self.last_migration_report = report
        self._legacy_migration_consumed = True
        return result
