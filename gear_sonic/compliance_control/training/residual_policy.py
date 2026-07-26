"""Frozen-release SONIC policy with isolated hard-gated residual adapters."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn

from gear_sonic.compliance_control.core import hard_gate_residual
from gear_sonic.trl.modules.actor_critic_modules import Critic
from gear_sonic.trl.modules.universal_token_modules import UniversalTokenModule


MOTION_COMPLIANCE_BACKBONE_TARGET = (
    "gear_sonic.compliance_control.training.residual_policy."
    "MotionComplianceUniversalTokenModule"
)
MOTION_COMPLIANCE_CRITIC_TARGET = (
    "gear_sonic.compliance_control.training.residual_policy."
    "MotionComplianceResidualCritic"
)
CONDITION_OBSERVATION_KEY = "motion_compliance_condition"
PRIVILEGED_OBSERVATION_KEY = "motion_compliance_privileged"
CONDITION_DIM = 3
class ZeroInitializedResidualMLP(nn.Module):
    """Small residual MLP whose initial output is exactly zero."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("residual input/output dimensions must be positive")
        if not hidden_dims or any(type(width) is not int or width <= 0 for width in hidden_dims):
            raise ValueError("residual hidden_dims must contain positive integers")
        # Adapter construction must not perturb release initialization or
        # environment sampling streams.  The generated weights remain random,
        # but the caller's CPU RNG state is restored on exit.
        with torch.random.fork_rng(devices=[]):
            layers: list[nn.Module] = []
            previous = input_dim
            for width in hidden_dims:
                layers.extend((nn.Linear(previous, width), nn.SiLU()))
                previous = width
            output = nn.Linear(previous, output_dim)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
            layers.append(output)
        self.module = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.module(value)


def _require_obs_width(obs_dim_dict, key: str, expected: int | None = None) -> int:
    width = obs_dim_dict.get(key, None)
    if type(width) is not int or width <= 0:
        raise ValueError(f"{key} must have a positive integer width; got {width!r}")
    if expected is not None and width != expected:
        raise ValueError(f"{key} must have width {expected}; got {width}")
    return width


def motion_compliance_residual_parameters(
    policy: nn.Module,
    value_model: nn.Module,
) -> tuple[nn.Parameter, ...]:
    """Return residual parameters after proving no released parameter is trainable.

    This is an optimizer-boundary check, not a name-prefix filter: expected
    parameters are identified from the two registered residual modules and
    compared by object identity with every parameter whose ``requires_grad``
    bit is set on the complete policy/value models.
    """

    policy_backbone = getattr(policy, "actor_module", policy)
    action_residual = getattr(
        policy_backbone,
        "motion_compliance_action_residual",
        None,
    )
    value_residual = getattr(
        value_model,
        "motion_compliance_value_residual",
        None,
    )
    if not isinstance(action_residual, nn.Module):
        raise TypeError("policy lacks a motion-compliance action residual module")
    if not isinstance(value_residual, nn.Module):
        raise TypeError("value model lacks a motion-compliance value residual module")

    expected = tuple(action_residual.parameters()) + tuple(value_residual.parameters())
    if not expected:
        raise ValueError("motion-compliance residual modules have no parameters")
    expected_ids = {id(parameter) for parameter in expected}
    if len(expected_ids) != len(expected):
        raise ValueError("motion-compliance residual modules share parameters")

    named_parameters = tuple(
        (f"policy.{name}", parameter) for name, parameter in policy.named_parameters()
    ) + tuple(
        (f"value.{name}", parameter) for name, parameter in value_model.named_parameters()
    )
    actual = tuple(parameter for _, parameter in named_parameters if parameter.requires_grad)
    actual_ids = {id(parameter) for parameter in actual}
    if actual_ids != expected_ids or len(actual) != len(expected):
        unexpected = sorted(
            name
            for name, parameter in named_parameters
            if parameter.requires_grad and id(parameter) not in expected_ids
        )
        missing = sorted(
            name
            for name, parameter in named_parameters
            if id(parameter) in expected_ids and not parameter.requires_grad
        )
        raise RuntimeError(
            "only motion-compliance residual parameters may be trainable; "
            f"unexpected={unexpected}, frozen_residual={missing}"
        )
    return expected


class MotionComplianceUniversalTokenModule(UniversalTokenModule):
    """Run the byte-preserved release backbone plus an enabled-only action delta.

    The environment exposes the public condition as a separate observation
    group.  It is appended only inside this wrapper so the released
    ``actor_obs`` remains 930 columns and ``g1_dyn`` remains 994 columns.
    """

    def __init__(
        self,
        *args,
        obs_dim_dict=None,
        motion_compliance_condition_key: str = CONDITION_OBSERVATION_KEY,
        motion_compliance_residual_hidden_dims: Sequence[int] = (256, 256),
        motion_compliance_action_delta_limit: float = 0.25,
        **kwargs,
    ) -> None:
        if obs_dim_dict is None:
            env_config = kwargs.get("env_config", args[0] if args else None)
            obs_dim_dict = env_config.robot.algo_obs_dim_dict
        self.motion_compliance_condition_key = motion_compliance_condition_key
        if (
            isinstance(motion_compliance_action_delta_limit, bool)
            or not isinstance(motion_compliance_action_delta_limit, (int, float))
            or not math.isfinite(float(motion_compliance_action_delta_limit))
            or motion_compliance_action_delta_limit <= 0.0
        ):
            raise ValueError("motion_compliance_action_delta_limit must be finite and positive")
        self.motion_compliance_action_delta_limit = float(
            motion_compliance_action_delta_limit
        )
        self.release_actor_obs_dim = _require_obs_width(obs_dim_dict, "actor_obs", 930)
        _require_obs_width(obs_dim_dict, motion_compliance_condition_key, CONDITION_DIM)

        super().__init__(*args, obs_dim_dict=obs_dim_dict, **kwargs)
        if tuple(self.proprioception_features) != ("actor_obs",):
            raise ValueError(
                "motion-compliance release path requires proprioception_features=['actor_obs']"
            )
        if tuple(self.decoder_input_features.get("g1_dyn", ())) != (
            "token_flattened",
            "proprioception",
        ):
            raise ValueError("g1_dyn release input contract changed")
        first_layer = self.decoders["g1_dyn"].module[0]
        expected_release_width = self.token_total_dim + self.release_actor_obs_dim
        if (
            not isinstance(first_layer, nn.Linear)
            or first_layer.in_features != expected_release_width
        ):
            raise ValueError(
                f"g1_dyn must retain released input width {expected_release_width}"
            )

        # Freeze every released encoder, quantizer, and decoder parameter before
        # registering the sole trainable action adapter.
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.motion_compliance_action_residual = ZeroInitializedResidualMLP(
            expected_release_width + CONDITION_DIM,
            self.actions_dim,
            motion_compliance_residual_hidden_dims,
        )

    def _validated_actor_inputs(
        self,
        input_data: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actor_input_keys = {
            "actor_obs",
            "tokenizer",
            self.motion_compliance_condition_key,
        }
        unexpected = sorted(set(input_data) - actor_input_keys)
        if unexpected:
            raise ValueError(f"actor input contains non-allowlisted groups: {unexpected}")
        missing = sorted(actor_input_keys - set(input_data))
        if missing:
            raise KeyError(f"actor input lacks required groups: {missing}")
        actor_obs = input_data["actor_obs"]
        condition = input_data[self.motion_compliance_condition_key]
        if actor_obs.shape[:-1] != condition.shape[:-1]:
            raise ValueError("actor_obs and motion compliance condition leading shapes differ")
        if actor_obs.shape[-1] != self.release_actor_obs_dim:
            raise ValueError("actor_obs no longer matches the released 930-column contract")
        if condition.shape[-1] != CONDITION_DIM:
            raise ValueError("motion compliance condition must have three columns")
        if actor_obs.dtype != condition.dtype or actor_obs.device != condition.device:
            raise ValueError("actor_obs and motion compliance condition dtype/device differ")
        return actor_obs, condition

    def _apply_action_residual(
        self,
        base_action: torch.Tensor,
        token_flattened: torch.Tensor,
        actor_obs: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if base_action.shape[:-1] != actor_obs.shape[:-1] or (
            token_flattened.shape[:-1] != actor_obs.shape[:-1]
        ):
            raise ValueError("action residual context leading shapes differ")
        if base_action.shape[-1] != self.actions_dim:
            raise ValueError("released action width differs from configured action width")
        if token_flattened.shape[-1] != self.token_total_dim:
            raise ValueError("action residual token width differs from release")
        enabled = condition[..., 0].gt(0.5)
        residual_context = torch.cat(
            (
                token_flattened,
                actor_obs,
                condition,
            ),
            dim=-1,
        )
        # Rejected rows are sanitized before the MLP.  Otherwise NaN in a
        # disabled condition can poison shared weight gradients even though
        # the final hard gate selects the release branch.
        residual_context = torch.where(
            enabled.unsqueeze(-1),
            residual_context,
            torch.zeros_like(residual_context),
        )
        residual = torch.tanh(
            self.motion_compliance_action_residual(residual_context)
        ) * self.motion_compliance_action_delta_limit
        return hard_gate_residual(
            base_action,
            residual,
            enabled,
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
        actor_obs, condition = self._validated_actor_inputs(input_data)
        # The release module sees its original dict and original 930-column
        # proprioception.  Requesting its rich result lets this wrapper compose
        # a separate final action without changing any decoder input.
        output = super().forward(
            input_data,
            compute_aux_loss=compute_aux_loss,
            return_dict=True,
            latent_residual=latent_residual,
            latent_residual_mode=latent_residual_mode,
            **kwargs,
        )
        base_action = output["action_mean"]
        action = self._apply_action_residual(
            base_action,
            self._last_full_latent_flat,
            actor_obs,
            condition,
        )
        output["action_mean"] = action
        g1_output = output["decoded_outputs"].get("g1_dyn")
        if g1_output is not None and "action" in g1_output:
            g1_output["action"] = action
        if compute_aux_loss or return_dict:
            return output
        return action

    def forward_with_external_tokens(self, input_data, external_tokens, **kwargs):
        actor_obs, condition = self._validated_actor_inputs(input_data)
        base_action = super().forward_with_external_tokens(
            input_data,
            external_tokens,
            **kwargs,
        )
        tokens = external_tokens
        if tokens.ndim == 3:
            tokens = tokens.unsqueeze(1)
        token_flattened = tokens.reshape(*tokens.shape[:-2], -1)
        if base_action.ndim == actor_obs.ndim - 1:
            if actor_obs.shape[-2] != 1 or token_flattened.shape[-2] != 1:
                raise ValueError("external-token base action dropped a non-singleton sequence")
            actor_obs = actor_obs.squeeze(-2)
            condition = condition.squeeze(-2)
            token_flattened = token_flattened.squeeze(-2)
        return self._apply_action_residual(
            base_action,
            token_flattened,
            actor_obs,
            condition,
        )


class MotionComplianceResidualCritic(Critic):
    """Frozen released value model plus an enabled-only privileged residual."""

    def __init__(
        self,
        env_config,
        algo_config,
        backbone,
        obs_dim_dict=None,
        module_dim_dict={},
        running_mean_std=False,
        use_batch_norm=False,
        backbone_kwargs={},
        motion_compliance_condition_key: str = CONDITION_OBSERVATION_KEY,
        motion_compliance_privileged_key: str = PRIVILEGED_OBSERVATION_KEY,
        motion_compliance_residual_hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        if obs_dim_dict is None:
            obs_dim_dict = env_config.robot.algo_obs_dim_dict
        self.motion_compliance_condition_key = motion_compliance_condition_key
        self.motion_compliance_privileged_key = motion_compliance_privileged_key
        self.release_critic_obs_dim = _require_obs_width(obs_dim_dict, "critic_obs", 1645)
        condition_dim = _require_obs_width(
            obs_dim_dict,
            motion_compliance_condition_key,
            CONDITION_DIM,
        )
        privileged_dim = _require_obs_width(obs_dim_dict, motion_compliance_privileged_key)
        self.motion_compliance_privileged_dim = privileged_dim
        super().__init__(
            env_config=env_config,
            algo_config=algo_config,
            backbone=backbone,
            obs_dim_dict=obs_dim_dict,
            module_dim_dict=module_dim_dict,
            running_mean_std=running_mean_std,
            use_batch_norm=use_batch_norm,
            backbone_kwargs=backbone_kwargs,
        )
        first_layer = self.critic_module.module[0]
        if not isinstance(first_layer, nn.Linear) or first_layer.in_features != 1645:
            raise ValueError("critic must retain released input width 1645")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if self.running_mean_std is not None and hasattr(self.running_mean_std, "freeze"):
            self.running_mean_std.freeze()
        self.motion_compliance_value_residual = ZeroInitializedResidualMLP(
            self.release_critic_obs_dim + condition_dim + privileged_dim,
            1,
            motion_compliance_residual_hidden_dims,
        )

    def evaluate(self, obs_dict, **kwargs):
        critic_obs = obs_dict["critic_obs"]
        condition = obs_dict[self.motion_compliance_condition_key]
        privileged = obs_dict[self.motion_compliance_privileged_key]
        if critic_obs.shape[:-1] != condition.shape[:-1] or (
            critic_obs.shape[:-1] != privileged.shape[:-1]
        ):
            raise ValueError("critic compliance observation leading shapes differ")
        if critic_obs.shape[-1] != self.release_critic_obs_dim:
            raise ValueError("critic_obs no longer matches the released contract")
        if condition.shape[-1] != CONDITION_DIM:
            raise ValueError("motion compliance condition must have three columns")
        if privileged.shape[-1] != self.motion_compliance_privileged_dim:
            raise ValueError("motion compliance privileged width changed")
        if (
            critic_obs.dtype != condition.dtype
            or critic_obs.dtype != privileged.dtype
            or critic_obs.device != condition.device
            or critic_obs.device != privileged.device
        ):
            raise ValueError("critic compliance observations dtype/device differ")

        base_value = super().evaluate({"critic_obs": critic_obs}, **kwargs)
        enabled = condition[..., 0].gt(0.5)
        residual_context = torch.cat((critic_obs, condition, privileged), dim=-1)
        residual_context = torch.where(
            enabled.unsqueeze(-1),
            residual_context,
            torch.zeros_like(residual_context),
        )
        residual = self.motion_compliance_value_residual(
            residual_context
        )
        return hard_gate_residual(
            base_value,
            residual,
            enabled,
        )
