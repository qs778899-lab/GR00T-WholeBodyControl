"""Compliance-only PPO trainer with strict checkpoint semantics."""

from __future__ import annotations

from gear_sonic.trl.trainer.ppo_trainer_aux_loss import TRLAuxLossPPOTrainer

from .checkpoint import (
    load_trl_checkpoint,
    strict_load_policy_value_state,
    validate_strict_resume_payload,
)


class MotionCompliancePPOTrainer(TRLAuxLossPPOTrainer):
    """Keep strict migration/resume rules isolated from SONIC's generic trainer."""

    _tag_names = ["trl", "aux_loss_ppo", "motion_compliance"]

    def load_checkpoint(self, checkpoint_path, resume=False):  # noqa: D417
        """Strict-load migrated initialization or a complete branch checkpoint."""

        print(f"Loading motion-compliance checkpoint from {checkpoint_path}")  # noqa: T201
        checkpoint = load_trl_checkpoint(
            checkpoint_path,
            map_location=self.accelerator.device,
        )

        model = self.accelerator.unwrap_model(self.model)
        if model.value_model is None:
            raise ValueError("motion-compliance trainer requires a critic value model")
        load_report = strict_load_policy_value_state(
            model.policy,
            model.value_model,
            checkpoint,
            resume=resume,
        )

        if resume:
            payload = validate_strict_resume_payload(checkpoint)
            self.optimizer.load_state_dict(payload.optimizer_state_dict)
            self.lr_scheduler.load_state_dict(payload.lr_scheduler_state_dict)
            self.env.load_env_state_dict(payload.env_state_dict)

            if "args" in checkpoint and hasattr(checkpoint["args"], "learning_rate"):
                self.args.learning_rate = checkpoint["args"].learning_rate
                for parameter_group in self.optimizer.param_groups:
                    parameter_group["lr"] = self.args.learning_rate

            for key, value in payload.state.__dict__.items():
                if key in {"cur_reward_sum", "cur_episode_length"}:
                    current_value = getattr(self, key, None)
                    if current_value is None or current_value.shape != value.shape:
                        raise ValueError(
                            f"strict resume trainer tensor shape differs for {key}: "
                            f"current={getattr(current_value, 'shape', None)}, "
                            f"checkpoint={value.shape}"
                        )
                    setattr(self, key, value)
                if key not in {
                    "stateful_callbacks",
                    "is_local_process_zero",
                    "is_world_process_zero",
                    "log_history",
                }:
                    setattr(self.state, key, value)

        loaded_step = getattr(checkpoint.get("state"), "global_step", None)
        print(  # noqa: T201
            "Loaded motion-compliance checkpoint "
            f"from step {loaded_step}: policy_key={load_report.policy_key}, "
            f"strict={load_report.strict}, migrated_init={load_report.migrated_init}"
        )
        return checkpoint
