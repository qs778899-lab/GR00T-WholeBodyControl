"""Portable Phase-3 residual, migration, and actor-privilege contracts."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

import torch
from torch import nn

from gear_sonic.compliance_control import ComplianceResidualLayout, ComplianceResidualMLP
from gear_sonic.compliance_control.adapters.sonic.checkpoint import (
    migrate_legacy_state_dict,
)


class ComplianceResidualTest(unittest.TestCase):
    def _module(self, *, sites: int = 4) -> ComplianceResidualMLP:
        return ComplianceResidualMLP(
            condition_dim=sites * 6,
            num_sites=sites,
            cartesian_dim=3,
            context_dim=11,
            output_dim=13,
            hidden_dims=(17, 19),
            residual_limit=0.2,
        )

    def _inputs(self, module: ComplianceResidualMLP, *, batch: int = 3):
        layout = module.layout
        condition = torch.randn(batch, layout.condition_dim)
        command = torch.zeros(batch, layout.command_dim)
        context = torch.randn(batch, layout.context_dim)
        return condition, command, context

    def test_layout_and_forward_support_arbitrary_site_count(self) -> None:
        sites = 17
        module = self._module(sites=sites)
        self.assertEqual(module.layout.num_sites, sites)
        self.assertEqual(module.layout.command_dim, 1 + sites + sites * 3)
        condition, command, context = self._inputs(module)
        command[:, 0] = 1.0
        command[:, 1 : 1 + sites] = 1.0
        command[:, 1 + sites :] = 0.02
        output = module(condition, command, context)
        self.assertEqual(output.shape, (3, 13))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_hard_off_and_zero_compliance_are_exact_even_with_nan_condition(self) -> None:
        module = self._module()
        condition, command, context = self._inputs(module)
        condition.fill_(float("nan"))
        command[:, 1 : 1 + module.layout.num_sites] = 1.0
        output_global_off = module(condition, command, context)
        self.assertTrue(torch.equal(output_global_off, torch.zeros_like(output_global_off)))

        command[:, 0] = 1.0
        output_zero_compliance = module(condition, command, context)
        self.assertTrue(
            torch.equal(output_zero_compliance, torch.zeros_like(output_zero_compliance))
        )

    def test_enabled_output_is_bounded_and_mixed_rows_are_hard_gated(self) -> None:
        module = self._module()
        with torch.no_grad():
            module.output_layer.weight.fill_(10.0)
            module.output_layer.bias.fill_(10.0)
        condition, command, context = self._inputs(module)
        sites = module.layout.num_sites
        command[0, 0] = 1.0
        command[0, 1] = 1.0
        command[0, 1 + sites : 1 + sites + 3] = 0.02
        command[1, 0] = 1.0
        command[1, 1] = 1.0
        output = module(condition, command, context)
        self.assertTrue(torch.equal(output[1:], torch.zeros_like(output[1:])))
        maximum = output[0].detach().abs().max()
        self.assertLessEqual(
            float(maximum),
            module.residual_limit + torch.finfo(maximum.dtype).eps,
        )
        self.assertGreater(float(maximum), 0.0)

    def test_zero_initialized_head_has_nonzero_enabled_gradient(self) -> None:
        module = self._module()
        for layer in module.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0.05)
                nn.init.constant_(layer.bias, 0.1)
        condition, command, context = self._inputs(module)
        sites = module.layout.num_sites
        command[:, 0] = 1.0
        command[:, 1] = 1.0
        command[:, 1 + sites : 1 + sites + 3] = 0.05
        output = module(condition, command, context)
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        output.sum().backward()
        for name, parameter in module.output_layer.named_parameters():
            self.assertIsNotNone(parameter.grad, msg=name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, msg=name)

        # A zero output head deliberately blocks the first-step gradient from
        # reaching the trunk.  The head learns on step one; the trunk can only
        # start learning after the head becomes non-zero.
        for name, parameter in module.trunk.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(
                    torch.equal(parameter.grad, torch.zeros_like(parameter.grad)),
                    msg=name,
                )

    def test_shape_dtype_and_layout_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_sites"):
            ComplianceResidualLayout(4, 0, 3, 5, 6)
        module = self._module()
        condition, command, context = self._inputs(module)
        with self.assertRaisesRegex(ValueError, "condition final dimension"):
            module(condition[..., :-1], command, context)
        with self.assertRaisesRegex(TypeError, "one dtype"):
            module(condition.double(), command, context)
        with self.assertRaisesRegex(ValueError, "leading dimensions"):
            module(condition[:1], command, context)


class _ToyMigratedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.release_path = nn.Linear(5, 4)
        self.compliance_residual = ComplianceResidualMLP(
            condition_dim=6,
            num_sites=2,
            cartesian_dim=3,
            context_dim=4,
            output_dim=7,
            hidden_dims=(8,),
            residual_limit=0.1,
        )


class CheckpointMigrationTest(unittest.TestCase):
    def test_only_new_branch_keys_are_initialized_and_legacy_is_exact(self) -> None:
        torch.manual_seed(3)
        source_module = _ToyMigratedModule()
        full_source = source_module.state_dict()
        source = {
            name: tensor.clone()
            for name, tensor in full_source.items()
            if not name.startswith("compliance_residual.")
        }
        target = _ToyMigratedModule()
        report, result = migrate_legacy_state_dict(
            target,
            source,
            new_key_prefixes=("compliance_residual.",),
        )
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(set(report.legacy_keys), set(source))
        self.assertTrue(report.initialized_new_keys)
        self.assertTrue(
            all(name.startswith("compliance_residual.") for name in report.initialized_new_keys)
        )
        for name, tensor in source.items():
            self.assertTrue(torch.equal(target.state_dict()[name], tensor), msg=name)
        self.assertEqual(set(source), set(report.legacy_keys))

    def test_partial_or_mismatched_legacy_state_is_rejected_before_mutation(self) -> None:
        target = _ToyMigratedModule()
        initial = {name: tensor.clone() for name, tensor in target.state_dict().items()}
        source = {
            name: tensor.clone()
            for name, tensor in initial.items()
            if not name.startswith("compliance_residual.")
        }
        source.pop(next(iter(source)))
        with self.assertRaisesRegex(RuntimeError, "schema mismatch"):
            migrate_legacy_state_dict(
                target,
                source,
                new_key_prefixes=("compliance_residual.",),
            )
        for name, tensor in initial.items():
            self.assertTrue(torch.equal(target.state_dict()[name], tensor), msg=name)


class ActorPrivilegeSourceTest(unittest.TestCase):
    def test_actor_hard_codes_privileged_force_rejection(self) -> None:
        root = Path(__file__).resolve().parents[2]
        policy_path = root / "compliance_control/adapters/sonic/policy.py"
        tree = ast.parse(policy_path.read_text(encoding="utf-8"), filename=str(policy_path))
        strings = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("compliance_force", strings)
        self.assertFalse(any("force_on_robot" in value for value in strings))

        forbidden_assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_FORBIDDEN_ACTOR_OBSERVATION_KEYS"
                for target in node.targets
            )
        )
        self.assertIn("compliance_force", ast.unparse(forbidden_assignment))

        policy_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "SonicComplianceUniversalTokenModule"
        )
        forward = next(
            node
            for node in policy_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        subscript_names = {
            node.slice.value
            for node in ast.walk(forward)
            if isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        }
        self.assertTrue({"actor_obs"}.isdisjoint(subscript_names))
        self.assertNotIn("compliance_force", subscript_names)

        actor_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SonicComplianceActor"
        )
        method_names = {
            node.name for node in actor_class.body if isinstance(node, ast.FunctionDef)
        }
        self.assertTrue(
            {"_public_observations", "forward", "_update_obs_buffer"}.issubset(
                method_names
            )
        )
        actor_init = next(
            node
            for node in actor_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        init_parameters = {
            argument.arg
            for argument in (
                *actor_init.args.posonlyargs,
                *actor_init.args.args,
                *actor_init.args.kwonlyargs,
            )
        }
        self.assertNotIn("forbidden_observation_keys", init_parameters)
        self.assertNotIn("privileged_observation_keys", init_parameters)


if __name__ == "__main__":
    unittest.main()
