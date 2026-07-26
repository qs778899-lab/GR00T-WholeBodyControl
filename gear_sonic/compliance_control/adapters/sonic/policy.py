"""Thin SONIC policy glue for a post-quantization compliance residual."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from gear_sonic.trl.modules.actor_critic_modules import Actor
from gear_sonic.trl.modules.universal_token_modules import UniversalTokenModule
from gear_sonic.trl.utils import common

from ...core import ComplianceResidualMLP
from .checkpoint import (
    CheckpointMigrationReport,
    classify_checkpoint_state,
    migrate_legacy_state_dict,
)


_FORBIDDEN_ACTOR_OBSERVATION_KEYS = frozenset(("compliance_force",))


def _ordered_site_names(site_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(site_names, str | bytes):
        raise TypeError("compliance_site_names must be a sequence")
    names = tuple(site_names)
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("compliance_site_names must contain non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("compliance_site_names must be unique and ordered")
    return names


def _ordered_observation_keys(observation_keys: Sequence[str]) -> tuple[str, ...]:
    if isinstance(observation_keys, str | bytes):
        raise TypeError("allowed_observation_keys must be a sequence")
    keys = tuple(observation_keys)
    if not keys or any(not isinstance(key, str) or not key.strip() for key in keys):
        raise ValueError("allowed_observation_keys must contain non-empty strings")
    if len(keys) != len(set(keys)):
        raise ValueError("allowed_observation_keys must be unique and ordered")
    forbidden = _FORBIDDEN_ACTOR_OBSERVATION_KEYS.intersection(keys)
    if forbidden:
        raise ValueError(
            "allowed_observation_keys contains privileged observations: "
            f"{sorted(forbidden)}"
        )
    return keys


def _feature_width(obs_dim_dict, feature_name: str) -> int:
    if feature_name not in obs_dim_dict:
        raise KeyError(f"missing observation dimension for {feature_name!r}")
    width = obs_dim_dict[feature_name]
    if type(width) is not int or width <= 0:
        raise ValueError(f"observation dimension for {feature_name!r} must be positive")
    return width


def _instantiate_residual(config, **dimensions) -> ComplianceResidualMLP:
    if isinstance(config, nn.Module):
        residual = config
    else:
        # Residual construction is opt-in and must not perturb the release
        # actor/critic initialization or subsequent stochastic sequence.
        with torch.random.fork_rng(devices=[]):
            residual = common.custom_instantiate(
                config,
                _resolve=False,
                **dimensions,
            )
    if not isinstance(residual, ComplianceResidualMLP):
        raise TypeError("compliance_residual must instantiate ComplianceResidualMLP")
    return residual


class SonicComplianceUniversalTokenModule(UniversalTokenModule):
    """Release UniversalTokenModule plus one actor-safe 64D latent branch."""

    def __init__(
        self,
        *,
        compliance_site_names: Sequence[str],
        compliance_cartesian_dim: int,
        compliance_num_future_frames: int | None = None,
        compliance_residual,
        compliance_target_key: str = "compliance_target",
        compliance_command_key: str = "compliance_command",
        **kwargs,
    ) -> None:
        self.compliance_site_names = _ordered_site_names(compliance_site_names)
        if type(compliance_cartesian_dim) is not int or compliance_cartesian_dim <= 0:
            raise ValueError("compliance_cartesian_dim must be a positive integer")
        self.compliance_cartesian_dim = compliance_cartesian_dim
        if compliance_num_future_frames is not None and (
            type(compliance_num_future_frames) is not int
            or compliance_num_future_frames <= 0
        ):
            raise ValueError("compliance_num_future_frames must be a positive integer")
        self.compliance_target_key = compliance_target_key
        self.compliance_command_key = compliance_command_key
        super().__init__(**kwargs)
        self.compliance_num_future_frames = (
            self.num_future_frames
            if compliance_num_future_frames is None
            else compliance_num_future_frames
        )

        target_dim = (
            self.compliance_num_future_frames
            * len(self.compliance_site_names)
            * self.compliance_cartesian_dim
        )
        command_dim = (
            1
            + len(self.compliance_site_names)
            + len(self.compliance_site_names) * self.compliance_cartesian_dim
        )
        if _feature_width(self.obs_dim_dict, compliance_target_key) != target_dim:
            raise ValueError("compliance target observation width does not match site layout")
        if _feature_width(self.obs_dim_dict, compliance_command_key) != command_dim:
            raise ValueError("compliance command observation width does not match site layout")
        context_dim = sum(
            _feature_width(self.obs_dim_dict, name) for name in self.proprioception_features
        )
        self.compliance_residual = _instantiate_residual(
            compliance_residual,
            condition_dim=target_dim,
            num_sites=len(self.compliance_site_names),
            cartesian_dim=self.compliance_cartesian_dim,
            context_dim=context_dim,
            output_dim=self.token_total_dim,
        )
        self._last_compliance_residual = None
        self._pending_compliance_residual = None

    def assemble_all_tokens(self, encoded_tokens, encoder_masks, batch_size, seq_len):
        """Inject the per-timestep residual after FSQ without flattening time.

        ``UniversalTokenModule`` first encodes the flattened ``B*S`` rows and
        then calls this virtual method to restore ``(B, S, token, channel)``.
        Adding the adapter-owned residual here keeps the released module
        untouched and makes the temporal alignment explicit: residual
        ``[b, s]`` can affect only token ``[b, s]``.
        """

        tokens = super().assemble_all_tokens(
            encoded_tokens,
            encoder_masks,
            batch_size,
            seq_len,
        )
        residual = self._pending_compliance_residual
        if residual is None:
            return tokens
        expected_shape = (batch_size, seq_len, self.token_total_dim)
        if tuple(residual.shape) != expected_shape:
            raise RuntimeError(
                "compliance residual sequence shape mismatch: "
                f"expected {expected_shape}, got {tuple(residual.shape)}"
            )
        return tokens + residual.reshape(
            batch_size,
            seq_len,
            self.max_num_tokens,
            self.token_dim,
        )

    def forward(
        self,
        input_data,
        compute_aux_loss=False,
        return_dict=False,
        latent_residual=None,
        latent_residual_mode="post_quantization",
        **kwargs,
    ):
        """Generate the sole latent residual from observed target and actor state."""

        if latent_residual is not None:
            raise ValueError("external latent_residual cannot be combined with compliance")
        if latent_residual_mode != "post_quantization":
            raise ValueError("SONIC compliance residual must be post_quantization")
        target = input_data[self.compliance_target_key]
        actor_command = input_data[self.compliance_command_key]
        if target.ndim != 3 or actor_command.ndim != 3:
            raise ValueError(
                "SONIC compliance policy inputs must preserve (batch, sequence, feature)"
            )
        batch_sequence = target.shape[:2]
        if actor_command.shape[:2] != batch_sequence:
            raise ValueError("SONIC compliance policy batch/sequence dimensions must match")
        context_inputs = [input_data[name] for name in self.proprioception_features]
        if any(
            tensor.ndim != 3 or tensor.shape[:2] != batch_sequence
            for tensor in context_inputs
        ):
            raise ValueError(
                "SONIC compliance context must preserve the target batch/sequence dimensions"
            )
        context = torch.cat(
            context_inputs,
            dim=-1,
        )
        delta_z = self.compliance_residual(target, actor_command, context)
        expected_residual_shape = (*batch_sequence, self.token_total_dim)
        if tuple(delta_z.shape) != expected_residual_shape:
            raise RuntimeError(
                "compliance residual changed the batch/sequence layout: "
                f"expected {expected_residual_shape}, got {tuple(delta_z.shape)}"
            )
        self._last_compliance_residual = delta_z.detach()
        if self._pending_compliance_residual is not None:
            raise RuntimeError("nested SONIC compliance policy forwards are unsupported")
        self._pending_compliance_residual = delta_z
        try:
            return super().forward(
                input_data,
                compute_aux_loss=compute_aux_loss,
                return_dict=return_dict,
                latent_residual=None,
                latent_residual_mode="post_quantization",
                **kwargs,
            )
        finally:
            self._pending_compliance_residual = None


class SonicComplianceActor(Actor):
    """Opt-in Actor whose released checkpoint migration is explicit and strict."""

    _NEW_STATE_PREFIXES = ("actor_module.compliance_residual.",)

    def __init__(
        self,
        *args,
        allowed_observation_keys: Sequence[str],
        allow_legacy_checkpoint_migration: bool = False,
        **kwargs,
    ) -> None:
        self._allowed_observation_keys = _ordered_observation_keys(
            allowed_observation_keys
        )
        super().__init__(*args, **kwargs)
        self.allow_legacy_checkpoint_migration = bool(allow_legacy_checkpoint_migration)
        self._legacy_migration_consumed = False
        self.last_migration_report: CheckpointMigrationReport | None = None

    @property
    def allowed_observation_keys(self) -> tuple[str, ...]:
        """Return the immutable public actor boundary."""

        return self._allowed_observation_keys

    @property
    def get_std(self):
        """Clamp a frozen direct std without mutating its checkpoint tensor."""

        if self.use_log_std or self.std.requires_grad:
            return super().get_std

        std = self.std
        if self.algo_config.get("use_clampped_std", False):
            std = torch.clamp(
                std,
                min=self.algo_config.std_clamp_min,
                max=self.algo_config.std_clamp_max,
            )
        if self.clamp_noise_std:
            std = torch.clamp(std, max=self.max_noise_std)
        return std

    def _public_observations(self, obs_dict) -> dict[str, torch.Tensor]:
        """Copy only explicitly allowlisted observations into the actor path."""

        public_keys = _ordered_observation_keys(self._allowed_observation_keys)
        missing = [key for key in public_keys if key not in obs_dict]
        if missing:
            raise KeyError(f"actor is missing public observations: {missing}")
        return {key: obs_dict[key] for key in public_keys}

    def forward(self, obs_dict, is_training=False, **kwargs):
        """Remove every non-public observation before invoking the backbone."""

        return super().forward(
            self._public_observations(obs_dict),
            is_training=is_training,
            **kwargs,
        )

    def _update_obs_buffer(self, obs_dict, episode_attnmask=None, cur_dones=None):
        """Keep privileged groups out of the actor's rollout history as well."""

        return super()._update_obs_buffer(
            self._public_observations(obs_dict),
            episode_attnmask=episode_attnmask,
            cur_dones=cur_dones,
        )

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
            raise RuntimeError("legacy checkpoint migration is disabled for this actor")
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
