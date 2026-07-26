"""Actor boundary used only by motion-compliance finetuning."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch.distributions import Normal

from gear_sonic.trl.modules.actor_critic_modules import Actor


MOTION_COMPLIANCE_ACTOR_TARGET = (
    "gear_sonic.compliance_control.training.actor.MotionComplianceFrozenNoiseActor"
)
_MOTION_COMPLIANCE_POLICY_KEYS = (
    "actor_obs",
    "tokenizer",
    "motion_compliance_condition",
)


class MotionComplianceFrozenNoiseActor(Actor):
    """Preserve frozen direct-std checkpoint state while retaining effective clamping."""

    @staticmethod
    def _policy_only_observations(obs_dict):
        missing = sorted(set(_MOTION_COMPLIANCE_POLICY_KEYS) - set(obs_dict))
        if missing:
            raise KeyError(f"motion-compliance actor lacks required groups: {missing}")
        return {key: obs_dict[key] for key in _MOTION_COMPLIANCE_POLICY_KEYS}

    def forward(self, obs_dict, is_training=False, **kwargs):
        """Pass an explicit allowlist to the policy backbone."""

        return super().forward(
            self._policy_only_observations(obs_dict),
            is_training=is_training,
            **kwargs,
        )

    def _update_obs_buffer(self, obs_dict, episode_attnmask=None, cur_dones=None):
        """Keep critic/privileged groups out of temporal policy history."""

        return super()._update_obs_buffer(
            self._policy_only_observations(obs_dict),
            episode_attnmask=episode_attnmask,
            cur_dones=cur_dones,
        )

    def rollout_with_tokens(
        self,
        obs_dict,
        external_tokens,
        episode_attnmask=None,
        cur_dones=None,
        **kwargs,
    ):
        """Use the same frozen, out-of-place noise clamp on token bypass."""

        self._update_obs_buffer(obs_dict, episode_attnmask, cur_dones)
        if not hasattr(self.actor_module, "forward_with_external_tokens"):
            raise NotImplementedError(
                "actor_module does not have forward_with_external_tokens"
            )
        obs_dict_last = {key: value[:, -1:] for key, value in self.obs_dict_buffer.items()}
        action_mean = self.actor_module.forward_with_external_tokens(
            input_data=obs_dict_last,
            external_tokens=external_tokens,
            **kwargs,
        )
        std = self.get_std
        self.distribution = Normal(
            action_mean,
            (action_mean * 0.0 + std).clamp(min=1.0e-6),
        )
        self.steps += 1
        return TensorDict(
            {
                "actions": self.distribution.sample(),
                "action_mean": self.action_mean,
                "action_sigma": self.action_std,
            }
        )

    @property
    def get_std(self):
        """Return release-equivalent noise without modifying the direct-std parameter."""

        if self.use_log_std:
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
