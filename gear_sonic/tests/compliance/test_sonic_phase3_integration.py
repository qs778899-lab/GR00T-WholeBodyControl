"""Resolved Phase-3 model and official-checkpoint integration contracts."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import torch


HYDRA_AVAILABLE = importlib.util.find_spec("hydra") is not None
OFFICIAL_CHECKPOINT = Path(
    "/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/"
    "official_assets/sonic_release/last.pt"
)
OFFICIAL_SHA256 = "e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909"

TOKENIZER_SHAPES = {
    "encoder_index": (3,),
    "command_multi_future_nonflat": (10, 58),
    "command_z_multi_future_nonflat": (10, 1),
    "motion_anchor_ori_b_mf_nonflat": (10, 6),
    "command_multi_future_lower_body": (240,),
    "vr_3point_local_target": (9,),
    "vr_3point_local_orn_target": (12,),
    "motion_anchor_ori_b": (6,),
    "command_z": (1,),
    "smpl_joints_multi_future_local_nonflat": (10, 72),
    "smpl_root_ori_b_multi_future": (10, 6),
    "joint_pos_multi_future_wrist_for_smpl": (10, 6),
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _byte_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_bytes = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    right_bytes = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(left_bytes, right_bytes)


def _load_trusted_official_checkpoint(path: Path):
    """Load the pinned local artifact with its historical TRL pickle symbols."""

    from trl.experimental.ppo import ppo_trainer
    import trl.trainer.utils

    trl.trainer.utils.OnlineTrainerState = ppo_trainer.OnlineTrainerState
    trl.trainer.utils.exact_div = ppo_trainer.exact_div
    sys.modules["trl.trainer.utils"].OnlineTrainerState = (
        ppo_trainer.OnlineTrainerState
    )
    sys.modules["trl.trainer.utils"].exact_div = ppo_trainer.exact_div
    return torch.load(path, map_location="cpu", weights_only=False)


def _fake_env_config(*, num_sites: int = 2, num_future_frames: int = 10):
    from omegaconf import OmegaConf

    obs_dims = {
        "actor_obs": 930,
        "critic_obs": 1645,
        "compliance_target": num_future_frames * num_sites * 3,
        "compliance_command": 1 + num_sites + num_sites * 3,
        "compliance_force": num_sites * 3,
        "tokenizer": sum(math.prod(shape) for shape in TOKENIZER_SHAPES.values()),
    }
    return OmegaConf.create(
        {
            "robot": {"algo_obs_dim_dict": obs_dims, "actions_dim": 29},
            "obs": {
                "group_obs_dims": {
                    "tokenizer": {
                        name: list(shape) for name, shape in TOKENIZER_SHAPES.items()
                    }
                },
                "group_obs_names": {"tokenizer": list(TOKENIZER_SHAPES)},
            },
        }
    )


@unittest.skipUnless(HYDRA_AVAILABLE, "hydra-core unavailable in portable environment")
class SonicPhase3ResolvedIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from hydra import compose, initialize_config_dir
        from gear_sonic.trl.utils.common import custom_instantiate
        from gear_sonic.utils.config_utils import register_rl_resolvers

        if not OFFICIAL_CHECKPOINT.is_file():
            raise AssertionError(f"missing pinned checkpoint: {OFFICIAL_CHECKPOINT}")
        if _file_sha256(OFFICIAL_CHECKPOINT) != OFFICIAL_SHA256:
            raise AssertionError("official SONIC checkpoint SHA-256 mismatch")

        register_rl_resolvers()
        config_dir = Path(__file__).resolve().parents[2] / "config"
        with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
            common = [
                "num_envs=1",
                "exp_base=chip_phase3_integration",
                "experiment_name=chip_phase3_integration",
                "experiment_dir=/tmp/chip_phase3_integration",
            ]
            cls.cfg = compose(
                config_name="base",
                overrides=[
                    "+exp=manager/universal_token/all_modes/"
                    "sonic_release_compliance_residual",
                    *common,
                ],
            )
            cls.release_cfg = compose(
                config_name="base",
                overrides=[
                    "+exp=manager/universal_token/all_modes/sonic_release",
                    *common,
                ],
            )
        cls.env_config = _fake_env_config()
        cls._custom_instantiate = staticmethod(custom_instantiate)
        cls.actor = cls._instantiate_actor()
        cls.critic = cls._instantiate_critic()
        cls.actor_new_before = {
            name: tensor.clone()
            for name, tensor in cls.actor.state_dict().items()
            if name.startswith("actor_module.compliance_residual.")
        }
        cls.critic_new_before = {
            name: tensor.clone()
            for name, tensor in cls.critic.state_dict().items()
            if name.startswith("compliance_value_residual.")
        }
        cls.checkpoint = _load_trusted_official_checkpoint(OFFICIAL_CHECKPOINT)
        cls.actor.load_state_dict(cls.checkpoint["policy_state_dict"])
        cls.critic.load_state_dict(cls.checkpoint["value_state_dict"])
        cls.actor.eval()
        cls.critic.eval()

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.actor
        del cls.critic
        del cls.checkpoint
        gc.collect()

    @classmethod
    def _instantiate_actor(
        cls,
        *,
        actor_config=None,
        env_config=None,
        algo_config=None,
    ):
        return cls._custom_instantiate(
            cls.cfg.algo.config.actor if actor_config is None else actor_config,
            env_config=cls.env_config if env_config is None else env_config,
            algo_config=cls.cfg.algo.config if algo_config is None else algo_config,
            _resolve=False,
        )

    @classmethod
    def _instantiate_critic(
        cls,
        *,
        critic_config=None,
        env_config=None,
        algo_config=None,
    ):
        return cls._custom_instantiate(
            cls.cfg.algo.config.critic if critic_config is None else critic_config,
            env_config=cls.env_config if env_config is None else env_config,
            algo_config=cls.cfg.algo.config if algo_config is None else algo_config,
            _resolve=False,
        )

    @staticmethod
    def _actor_inputs(*, batch: int = 3):
        components = {
            name: torch.randn(batch, 1, *shape)
            for name, shape in TOKENIZER_SHAPES.items()
        }
        components["encoder_index"] = torch.tensor(
            [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[1.0, 0.0, 1.0]]]
        )[:batch]
        tokenizer = torch.cat(
            [components[name].reshape(batch, 1, -1) for name in TOKENIZER_SHAPES],
            dim=-1,
        )
        public = {
            "actor_obs": torch.randn(batch, 1, 930),
            "tokenizer": tokenizer,
            "compliance_target": torch.randn(batch, 1, 60),
            "compliance_command": torch.zeros(batch, 1, 9),
        }
        return public, components

    def test_official_migration_is_complete_and_bitwise_exact(self) -> None:
        actor_source = self.checkpoint["policy_state_dict"]
        critic_source = self.checkpoint["value_state_dict"]
        self.assertEqual(len(actor_source), 55)
        self.assertEqual(len(critic_source), 17)

        actor_report = self.actor.last_migration_report
        critic_report = self.critic.last_migration_report
        self.assertEqual(set(actor_report.legacy_keys), set(actor_source))
        self.assertEqual(set(critic_report.legacy_keys), set(critic_source))
        self.assertEqual(
            set(actor_report.initialized_new_keys), set(self.actor_new_before)
        )
        self.assertEqual(
            set(critic_report.initialized_new_keys), set(self.critic_new_before)
        )
        self.assertEqual(len(actor_report.initialized_new_keys), 6)
        self.assertEqual(len(critic_report.initialized_new_keys), 6)

        actor_loaded = self.actor.state_dict()
        critic_loaded = self.critic.state_dict()
        for name, source in actor_source.items():
            self.assertTrue(_byte_equal(actor_loaded[name], source), msg=name)
        for name, source in critic_source.items():
            self.assertTrue(_byte_equal(critic_loaded[name], source), msg=name)
        for name, initialized in self.actor_new_before.items():
            self.assertTrue(_byte_equal(actor_loaded[name], initialized), msg=name)
        for name, initialized in self.critic_new_before.items():
            self.assertTrue(_byte_equal(critic_loaded[name], initialized), msg=name)

        self.assertTrue(
            torch.equal(
                self.actor.actor_module.compliance_residual.output_layer.weight,
                torch.zeros_like(
                    self.actor.actor_module.compliance_residual.output_layer.weight
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                self.actor.actor_module.compliance_residual.output_layer.bias,
                torch.zeros_like(
                    self.actor.actor_module.compliance_residual.output_layer.bias
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                self.critic.compliance_value_residual.output_layer.weight,
                torch.zeros_like(
                    self.critic.compliance_value_residual.output_layer.weight
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                self.critic.compliance_value_residual.output_layer.bias,
                torch.zeros_like(
                    self.critic.compliance_value_residual.output_layer.bias
                ),
            )
        )

    def test_branch_checkpoints_resume_strictly_even_if_caller_requests_non_strict(self) -> None:
        actor_clone = self._instantiate_actor()
        actor_clone.load_state_dict(self.actor.state_dict(), strict=False)
        partial_actor = dict(self.actor.state_dict())
        partial_actor.pop(next(iter(self.actor_new_before)))
        with self.assertRaises(RuntimeError):
            actor_clone.load_state_dict(partial_actor, strict=False)
        del actor_clone
        gc.collect()

        critic_clone = self._instantiate_critic()
        critic_clone.load_state_dict(self.critic.state_dict(), strict=False)
        partial_critic = dict(self.critic.state_dict())
        partial_critic.pop(next(iter(self.critic_new_before)))
        with self.assertRaises(RuntimeError):
            critic_clone.load_state_dict(partial_critic, strict=False)
        del critic_clone
        gc.collect()

    def test_privileged_force_is_a_non_overridable_actor_rejection(self) -> None:
        from omegaconf import OmegaConf
        from tensordict import TensorDict

        bad_actor_config = OmegaConf.create(
            OmegaConf.to_container(self.cfg.algo.config.actor, resolve=True)
        )
        bad_actor_config.allowed_observation_keys.append("compliance_force")
        with self.assertRaisesRegex(ValueError, "privileged observations"):
            self._instantiate_actor(actor_config=bad_actor_config)

        public, _ = self._actor_inputs()
        full = {**public, "compliance_force": torch.randn(3, 1, 6)}
        original_keys = self.actor._allowed_observation_keys  # noqa: SLF001
        original_steps = self.actor.steps
        try:
            # Defense in depth: even mutation of the internal tuple cannot make
            # direct or rollout paths accept the hard-coded privileged key.
            self.actor._allowed_observation_keys = (  # noqa: SLF001
                *original_keys,
                "compliance_force",
            )
            with self.assertRaisesRegex(ValueError, "privileged observations"):
                self.actor(full)
            self.actor.obs_dict_buffer = TensorDict()
            with self.assertRaisesRegex(ValueError, "privileged observations"):
                self.actor.rollout(
                    {name: tensor[:, 0] for name, tensor in full.items()}
                )
        finally:
            self.actor._allowed_observation_keys = original_keys  # noqa: SLF001
            self.actor.obs_dict_buffer = TensorDict()
            self.actor.steps = original_steps

    def test_actor_and_critic_construct_for_variable_site_and_future_counts(self) -> None:
        from omegaconf import OmegaConf

        variants = ((1, 1), (2, 10), (5, 3), (14, 4), (17, 7))
        for num_sites, num_future_frames in variants:
            names = [f"site_{index}" for index in range(num_sites)]
            actor_config = OmegaConf.create(
                OmegaConf.to_container(self.cfg.algo.config.actor, resolve=True)
            )
            critic_config = OmegaConf.create(
                OmegaConf.to_container(self.cfg.algo.config.critic, resolve=True)
            )
            actor_config.backbone.compliance_site_names = names
            actor_config.backbone.compliance_num_future_frames = num_future_frames
            critic_config.compliance_site_names = names
            critic_config.compliance_num_future_frames = num_future_frames
            env_config = _fake_env_config(
                num_sites=num_sites,
                num_future_frames=num_future_frames,
            )

            actor = self._instantiate_actor(
                actor_config=actor_config,
                env_config=env_config,
            )
            critic = self._instantiate_critic(
                critic_config=critic_config,
                env_config=env_config,
            )
            self.assertEqual(actor.actor_module.compliance_site_names, tuple(names))
            self.assertEqual(
                actor.actor_module.compliance_num_future_frames,
                num_future_frames,
            )
            self.assertEqual(actor.actor_module.token_total_dim, 64)
            self.assertEqual(
                actor.actor_module.compliance_residual.layout.condition_dim,
                num_sites * num_future_frames * 3,
            )
            self.assertEqual(
                actor.actor_module.compliance_residual.layout.command_dim,
                1 + num_sites + num_sites * 3,
            )
            self.assertEqual(critic.compliance_site_names, tuple(names))
            self.assertEqual(
                critic.compliance_value_residual.layout.condition_dim,
                num_sites * num_future_frames * 3 + num_sites * 3,
            )
            del actor
            del critic
            gc.collect()

    def test_residual_construction_preserves_release_rng_sequence(self) -> None:
        from omegaconf import OmegaConf

        entry_cpu_state = torch.random.get_rng_state().clone()
        entry_cuda_states = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        )

        def construct_and_capture(factory):
            torch.random.set_rng_state(entry_cpu_state)
            if entry_cuda_states is not None:
                torch.cuda.set_rng_state_all(entry_cuda_states)
            model = factory()
            cpu_after = torch.random.get_rng_state().clone()
            cuda_after = (
                [state.clone() for state in torch.cuda.get_rng_state_all()]
                if entry_cuda_states is not None
                else None
            )
            cpu_sequence = torch.rand(16)
            cuda_sequence = (
                torch.rand(16, device="cuda")
                if entry_cuda_states is not None
                else None
            )
            del model
            gc.collect()
            return cpu_after, cuda_after, cpu_sequence, cuda_sequence

        try:
            release_actor_config = OmegaConf.create(
                OmegaConf.to_container(
                    self.release_cfg.algo.config.actor,
                    resolve=True,
                )
            )
            compliance_actor_config = OmegaConf.create(
                OmegaConf.to_container(self.cfg.algo.config.actor, resolve=True)
            )
            release_actor_result = construct_and_capture(
                lambda: self._instantiate_actor(
                    actor_config=release_actor_config,
                    algo_config=self.release_cfg.algo.config,
                )
            )
            compliance_actor_result = construct_and_capture(
                lambda: self._instantiate_actor(actor_config=compliance_actor_config)
            )

            release_critic_config = OmegaConf.create(
                OmegaConf.to_container(
                    self.release_cfg.algo.config.critic,
                    resolve=True,
                )
            )
            compliance_critic_config = OmegaConf.create(
                OmegaConf.to_container(self.cfg.algo.config.critic, resolve=True)
            )
            release_critic_result = construct_and_capture(
                lambda: self._instantiate_critic(
                    critic_config=release_critic_config,
                    algo_config=self.release_cfg.algo.config,
                )
            )
            compliance_critic_result = construct_and_capture(
                lambda: self._instantiate_critic(critic_config=compliance_critic_config)
            )

            for release_result, compliance_result in (
                (release_actor_result, compliance_actor_result),
                (release_critic_result, compliance_critic_result),
            ):
                self.assertTrue(torch.equal(release_result[0], compliance_result[0]))
                self.assertTrue(_byte_equal(release_result[2], compliance_result[2]))
                if entry_cuda_states is not None:
                    self.assertTrue(
                        all(
                            torch.equal(left, right)
                            for left, right in zip(
                                release_result[1], compliance_result[1]
                            )
                        )
                    )
                    self.assertTrue(
                        _byte_equal(release_result[3], compliance_result[3])
                    )
        finally:
            torch.random.set_rng_state(entry_cpu_state)
            if entry_cuda_states is not None:
                torch.cuda.set_rng_state_all(entry_cuda_states)

    def test_only_declared_residual_parameters_are_trainable(self) -> None:
        actor_trainable = {
            name for name, parameter in self.actor.named_parameters() if parameter.requires_grad
        }
        critic_trainable = {
            name for name, parameter in self.critic.named_parameters() if parameter.requires_grad
        }
        self.assertEqual(actor_trainable, set(self.actor_new_before))
        self.assertEqual(critic_trainable, set(self.critic_new_before))
        self.assertFalse(self.actor.get_std.requires_grad)
        for module in (
            *self.actor.actor_module.encoders.values(),
            *self.actor.actor_module.decoders.values(),
            self.actor.actor_module.quantizer,
        ):
            self.assertTrue(all(not parameter.requires_grad for parameter in module.parameters()))
        self.assertTrue(self.critic.running_mean_std.frozen)

    def test_disabled_actor_is_release_exact_and_privileged_force_never_enters(self) -> None:
        from tensordict import TensorDict

        from gear_sonic.trl.modules.universal_token_modules import UniversalTokenModule

        public, _ = self._actor_inputs()
        privileged_force = torch.randn(3, 1, 6, requires_grad=True)
        full = {**public, "compliance_force": privileged_force}
        seen_actor_keys = []
        seen_residual_inputs = []

        def actor_pre_hook(_module, args):
            seen_actor_keys.append(tuple(args[0]))

        def residual_pre_hook(_module, args):
            seen_residual_inputs.append((args[0].clone(), args[1].clone()))

        actor_hook = self.actor.actor_module.register_forward_pre_hook(actor_pre_hook)
        residual_hook = self.actor.actor_module.compliance_residual.register_forward_pre_hook(
            residual_pre_hook
        )
        noise_parameter = self.actor.log_std if self.actor.use_log_std else self.actor.std
        noise_before = noise_parameter.detach().clone()
        cpu_rng_before = torch.random.get_rng_state().clone()
        cuda_rng_before = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        )
        original_buffer = self.actor.obs_dict_buffer
        original_dones_buffer = self.actor.dones_buffer
        original_distribution = self.actor.distribution
        original_steps = self.actor.steps
        try:
            with torch.no_grad():
                release_output = UniversalTokenModule.forward(
                    self.actor.actor_module, public
                )
                compliance_output = self.actor(full)
                poisoned_output = self.actor(
                    {**full, "compliance_force": torch.full_like(privileged_force, float("nan"))}
                )
                self.actor.obs_dict_buffer = TensorDict()
                self.actor.rollout(
                    {name: tensor[:, 0] for name, tensor in full.items()}
                )

            rollout_buffer_keys = tuple(self.actor.obs_dict_buffer.keys())
        finally:
            actor_hook.remove()
            residual_hook.remove()
            with torch.no_grad():
                noise_parameter.copy_(noise_before)
            torch.random.set_rng_state(cpu_rng_before)
            if cuda_rng_before is not None:
                torch.cuda.set_rng_state_all(cuda_rng_before)
            self.actor.obs_dict_buffer = original_buffer
            self.actor.dones_buffer = original_dones_buffer
            self.actor.distribution = original_distribution
            self.actor.steps = original_steps

        self.assertTrue(_byte_equal(compliance_output, release_output))
        self.assertTrue(_byte_equal(poisoned_output, release_output))
        self.assertEqual(
            seen_actor_keys,
            [self.actor.allowed_observation_keys] * 3,
        )
        self.assertEqual(len(seen_residual_inputs), 3)
        for seen_target, seen_command in seen_residual_inputs:
            self.assertTrue(torch.equal(seen_target, public["compliance_target"]))
            self.assertTrue(torch.equal(seen_command, public["compliance_command"]))

        self.assertEqual(
            rollout_buffer_keys, self.actor.allowed_observation_keys
        )

    def test_nonzero_residual_still_hard_gates_complete_actor_outputs(self) -> None:
        from torch import nn

        from gear_sonic.trl.modules.universal_token_modules import UniversalTokenModule

        residual = self.actor.actor_module.compliance_residual
        snapshot = {
            name: tensor.clone() for name, tensor in residual.state_dict().items()
        }
        try:
            with torch.no_grad():
                for layer in residual.trunk:
                    if isinstance(layer, nn.Linear):
                        layer.weight.fill_(0.002)
                        layer.bias.fill_(0.05)
                residual.output_layer.weight.fill_(0.01)
                residual.output_layer.bias.fill_(0.02)

                public, _ = self._actor_inputs()
                public["compliance_target"].fill_(float("nan"))
                release_off = UniversalTokenModule.forward(
                    self.actor.actor_module, public
                )
                complete_off = self.actor(
                    {**public, "compliance_force": torch.randn(3, 1, 6)}
                )
                self.assertTrue(_byte_equal(complete_off, release_off))

                zero_compliance = public["compliance_command"].clone()
                zero_compliance[..., 0] = 1.0
                zero_compliance[..., 1] = 1.0
                zero_public = {**public, "compliance_command": zero_compliance}
                release_zero = UniversalTokenModule.forward(
                    self.actor.actor_module, zero_public
                )
                complete_zero = self.actor(
                    {**zero_public, "compliance_force": torch.randn(3, 1, 6)}
                )
                self.assertTrue(_byte_equal(complete_zero, release_zero))

                mixed_target = public["compliance_target"].clone()
                mixed_target[0].zero_()
                mixed_command = public["compliance_command"].clone()
                mixed_command[0, ..., 0] = 1.0
                mixed_command[0, ..., 1] = 1.0
                mixed_command[0, ..., 3:6] = 0.02
                mixed_command[2, ..., 0] = 1.0
                mixed_command[2, ..., 1] = 1.0
                mixed_public = {
                    **public,
                    "compliance_target": mixed_target,
                    "compliance_command": mixed_command,
                }
                release_mixed = UniversalTokenModule.forward(
                    self.actor.actor_module, mixed_public
                )
                complete_mixed = self.actor(
                    {**mixed_public, "compliance_force": torch.randn(3, 1, 6)}
                )
                self.assertFalse(_byte_equal(complete_mixed[0], release_mixed[0]))
                self.assertTrue(_byte_equal(complete_mixed[1], release_mixed[1]))
                self.assertTrue(_byte_equal(complete_mixed[2], release_mixed[2]))
        finally:
            residual.load_state_dict(snapshot, strict=True)

    def test_critic_normalizes_once_and_shares_the_exact_base_context(self) -> None:
        critic_obs = {
            "critic_obs": torch.randn(3, 1, 1645),
            "compliance_target": torch.randn(3, 1, 60),
            "compliance_command": torch.zeros(3, 1, 9),
            "compliance_force": torch.randn(3, 1, 6),
        }
        critic_before = critic_obs["critic_obs"].clone()
        expected_obs = critic_obs.copy()
        with torch.no_grad():
            expected_obs["critic_obs"] = self.critic.running_mean_std(
                expected_obs["critic_obs"]
            )
            expected_base = self.critic.critic(expected_obs)

        normalizer_calls = []
        base_contexts = []
        residual_contexts = []
        normalizer_hook = self.critic.running_mean_std.register_forward_hook(
            lambda *_args: normalizer_calls.append(True)
        )
        base_hook = self.critic.critic.register_forward_pre_hook(
            lambda _module, args: base_contexts.append(args[0]["critic_obs"])
        )
        residual_hook = self.critic.compliance_value_residual.register_forward_pre_hook(
            lambda _module, args: residual_contexts.append(args[2])
        )
        try:
            with torch.no_grad():
                actual = self.critic.evaluate(critic_obs)
        finally:
            normalizer_hook.remove()
            base_hook.remove()
            residual_hook.remove()

        self.assertEqual(len(normalizer_calls), 1)
        self.assertEqual(len(base_contexts), 1)
        self.assertEqual(len(residual_contexts), 1)
        self.assertIs(base_contexts[0], residual_contexts[0])
        self.assertTrue(torch.equal(base_contexts[0], expected_obs["critic_obs"]))
        self.assertTrue(_byte_equal(actual, expected_base))
        self.assertTrue(torch.equal(critic_obs["critic_obs"], critic_before))

    def test_release_encoder_selection_and_auxiliary_losses_remain_valid(self) -> None:
        public, _ = self._actor_inputs()
        with torch.no_grad():
            result = self.actor.actor_module(public, compute_aux_loss=True)
        expected_masks = {
            "g1": [True, False, True],
            "teleop": [False, True, False],
            "smpl": [False, False, True],
        }
        for name, expected in expected_masks.items():
            self.assertEqual(result["encoder_masks"][name].tolist(), expected)
        self.assertEqual(
            set(result["aux_losses"]), set(self.actor.actor_module.aux_loss_coef)
        )
        for name, value in result["aux_losses"].items():
            self.assertTrue(torch.isfinite(value).all(), msg=name)

    def test_first_backward_updates_heads_without_privileged_actor_gradient(self) -> None:
        public, _ = self._actor_inputs()
        command = public["compliance_command"].clone()
        command[..., 0] = 1.0
        command[..., 1] = 1.0
        command[..., 3:6] = 0.02
        force = torch.randn(3, 1, 6, requires_grad=True)
        full = {**public, "compliance_command": command, "compliance_force": force}

        self.actor.zero_grad(set_to_none=True)
        action = self.actor(full)
        weights = torch.linspace(0.1, 1.0, action.numel()).reshape_as(action)
        (action * weights).sum().backward()
        actor_residual = self.actor.actor_module.compliance_residual
        for name, parameter in actor_residual.output_layer.named_parameters():
            self.assertIsNotNone(parameter.grad, msg=name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, msg=name)
        for name, parameter in actor_residual.trunk.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(
                    torch.equal(parameter.grad, torch.zeros_like(parameter.grad)),
                    msg=name,
                )
        self.assertIsNone(force.grad)
        for name, parameter in self.actor.named_parameters():
            if name not in self.actor_new_before:
                self.assertIsNone(parameter.grad, msg=name)

        self.critic.zero_grad(set_to_none=True)
        critic_obs = {
            "critic_obs": torch.randn(3, 1, 1645),
            "compliance_target": torch.randn(3, 1, 60),
            "compliance_command": command,
            "compliance_force": torch.randn(3, 1, 6),
        }
        value = self.critic.evaluate(critic_obs)
        (value * torch.tensor([[[1.0]], [[2.0]], [[3.0]]])).sum().backward()
        critic_residual = self.critic.compliance_value_residual
        for name, parameter in critic_residual.output_layer.named_parameters():
            self.assertIsNotNone(parameter.grad, msg=name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, msg=name)
        for name, parameter in critic_residual.trunk.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(
                    torch.equal(parameter.grad, torch.zeros_like(parameter.grad)),
                    msg=name,
                )
        for name, parameter in self.critic.named_parameters():
            if name not in self.critic_new_before:
                self.assertIsNone(parameter.grad, msg=name)

    def test_frozen_official_std_is_immutable_through_distribution_and_optimizer(self) -> None:
        public, _ = self._actor_inputs()
        official_std = self.checkpoint["policy_state_dict"]["std"]
        expected_effective_std = torch.clamp(
            official_std,
            min=self.actor.algo_config.std_clamp_min,
            max=self.actor.algo_config.std_clamp_max,
        )
        for _ in range(3):
            with torch.no_grad():
                self.actor.update_distribution(public)
            self.assertTrue(_byte_equal(self.actor.get_std, expected_effective_std))
            self.assertTrue(_byte_equal(self.actor.std, official_std))

        residual = self.actor.actor_module.compliance_residual
        residual_before = {
            name: tensor.clone() for name, tensor in residual.state_dict().items()
        }
        try:
            command = public["compliance_command"].clone()
            command[..., 0] = 1.0
            command[..., 1] = 1.0
            command[..., 3:6] = 0.02
            self.actor.zero_grad(set_to_none=True)
            self.actor({**public, "compliance_command": command}).sum().backward()
            optimizer = torch.optim.AdamW(self.actor.parameters(), lr=1.0e-3)
            optimizer.step()

            self.assertFalse(self.actor.std.requires_grad)
            self.assertIsNone(self.actor.std.grad)
            self.assertTrue(_byte_equal(self.actor.std, official_std))
        finally:
            residual.load_state_dict(residual_before, strict=True)
            self.actor.zero_grad(set_to_none=True)


if __name__ == "__main__":
    unittest.main()
