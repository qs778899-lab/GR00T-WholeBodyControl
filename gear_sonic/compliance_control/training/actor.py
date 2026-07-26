"""Actor boundary used only by motion-compliance finetuning."""

from __future__ import annotations

import torch

from gear_sonic.trl.modules.actor_critic_modules import Actor


MOTION_COMPLIANCE_ACTOR_TARGET = (
    "gear_sonic.compliance_control.training.actor.MotionComplianceFrozenNoiseActor"
)


class MotionComplianceFrozenNoiseActor(Actor):
    """Preserve frozen direct-std checkpoint state while retaining effective clamping."""

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
