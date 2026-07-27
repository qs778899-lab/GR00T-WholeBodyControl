"""Phase-5 tests for portable export/runtime and the thin SONIC adapter."""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import pytest
import torch
import yaml

from gear_sonic.compliance_control.adapters.sonic.deployment import (
    SonicActionResidualDeployment,
    SonicResidualContextContract,
    assemble_release_action_context,
    compose_optional_sonic_action_residual,
    encode_condition,
    load_sonic_action_residual_deployment,
)
from gear_sonic.compliance_control.deployment import (
    ActionResidualRuntime,
    ArtifactExpectation,
    ExportableActionResidual,
    ReleaseArtifactPin,
    ResidualExportSpec,
    export_action_residual_bundle,
    load_artifact_metadata,
    load_onnxruntime_session,
)


SOURCE_NAMES = (
    "adapter.module.0.weight",
    "adapter.module.0.bias",
    "adapter.module.2.weight",
    "adapter.module.2.bias",
    "adapter.module.4.weight",
    "adapter.module.4.bias",
)


def test_portable_python_and_cpp_layers_are_tracker_agnostic():
    repo_root = Path(__file__).parents[2]
    portable_paths = sorted(
        (repo_root / "gear_sonic/compliance_control/deployment").glob("*.py")
    ) + sorted(
        (repo_root / "gear_sonic_deploy/src/motion_compliance").glob("**/*.[ch]pp")
    )
    assert portable_paths
    forbidden = (
        "sonic",
        "g1",
        "isaaclab",
        "wrist",
        "robot_motion_token",
        "actor_observation",
    )
    for path in portable_paths:
        lowered = path.read_text(encoding="utf-8").lower()
        assert not any(word in lowered for word in forbidden), path
    host_header = (
        repo_root
        / "gear_sonic_deploy/src/motion_compliance/include/action_residual_overlay.hpp"
    ).read_text(encoding="utf-8")
    assert "std::vector<ReleaseArtifactPin> release_artifacts" in host_header
    assert "ReleaseArtifactPaths" not in host_header
    assert "ReleaseArtifactDigests" not in host_header


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _acceptance_module():
    path = (
        Path(__file__).parents[2]
        / "tasks"
        / "motion_compliance_finetune"
        / "artifacts"
        / "phase5_export_acceptance.py"
    )
    name = "motion_compliance_phase5_export_acceptance_for_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_tensors() -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(104729)
    shapes = ((4, 10), (4,), (6, 4), (6,), (5, 6), (5,))
    return {
        name: torch.randn(shape, generator=generator, dtype=torch.float32) * 0.1
        for name, shape in zip(SOURCE_NAMES, shapes, strict=True)
    }


def _spec() -> ResidualExportSpec:
    return ResidualExportSpec(
        checkpoint_sha256="a" * 64,
        checkpoint_global_step=7,
        policy_state_key="policy_state_dict",
        residual_tensor_names=SOURCE_NAMES,
        release_context_width=7,
        condition_width=3,
        action_width=5,
        hidden_dims=(4, 6),
        max_abs_delta=0.25,
        context_layout=(("token", 2), ("actor_observation", 5)),
        site_layout=("site_a", "site_b", "site_c"),
        action_layout=tuple(f"action_{index}" for index in range(5)),
    )


def _export(tmp_path: Path):
    output = tmp_path / "bundle"
    metadata = export_action_residual_bundle(_fixture_tensors(), output, _spec())
    return output, metadata


def _expectation(metadata: dict) -> ArtifactExpectation:
    host_base = Path(__file__).resolve()
    return ArtifactExpectation(
        metadata_sha256=metadata["metadata_sha256"],
        checkpoint_sha256="a" * 64,
        checkpoint_global_step=7,
        release_context_width=7,
        condition_width=3,
        action_layout=tuple(f"action_{index}" for index in range(5)),
        site_layout=("site_a", "site_b", "site_c"),
        max_abs_delta=0.25,
        release_artifacts=(
            ReleaseArtifactPin(
                name="fixture_base",
                path=host_base,
                sha256=_sha256(host_base),
            ),
        ),
    )


def _pytorch_module() -> ExportableActionResidual:
    module = ExportableActionResidual(
        7,
        3,
        5,
        hidden_dims=(4, 6),
        max_abs_delta=0.25,
    )
    module.load_linear_state(_fixture_tensors(), source_names=SOURCE_NAMES)
    return module


def _condition(shape: tuple[int, int], pattern: str) -> tuple[np.ndarray, np.ndarray]:
    gate = np.zeros(shape, dtype=np.bool_)
    if pattern == "on":
        gate[...] = True
    elif pattern == "mixed":
        gate.flat[::2] = True
    elif pattern != "off":
        raise AssertionError(pattern)
    condition = np.zeros((*shape, 3), dtype=np.float32)
    condition[..., 0] = gate
    condition[..., 1] = gate * 10.0
    condition[..., 2] = gate * 200.0
    return condition, gate


def test_portable_package_ast_has_no_simulator_or_robot_contracts():
    package = Path(__file__).parents[1] / "compliance_control" / "deployment"
    forbidden_fragments = (
        "isaaclab",
        "left_wrist",
        "right_wrist",
        "torso_link",
        "g1_",
        "14-keypoint",
    )
    for path in sorted(package.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        lowered = source.lower()
        for forbidden in forbidden_fragments:
            assert forbidden not in lowered, (path, forbidden)


def test_export_is_atomic_and_records_complete_contract(tmp_path: Path):
    output, metadata = _export(tmp_path)
    assert sorted(path.name for path in output.iterdir()) == [
        "action_residual.metadata.json",
        "action_residual.onnx",
    ]
    assert metadata["inputs"]["release_action_context"]["shape"] == ["B", "S", 7]
    assert metadata["inputs"]["motion_compliance_condition"]["shape"] == [
        "B",
        "S",
        3,
    ]
    assert metadata["output"]["shape"] == ["B", "S", 5]
    assert metadata["network"]["residual_context_width"] == 10
    assert metadata["source_checkpoint"]["residual_tensor_names"] == list(SOURCE_NAMES)
    assert metadata["model"]["sha256"] == _sha256(output / "action_residual.onnx")
    exported_graph = onnx.load(output / "action_residual.onnx").graph
    assert len(exported_graph.initializer) == 6
    assert {value.name for value in exported_graph.input} == {
        "release_action_context",
        "motion_compliance_condition",
    }
    assert [value.name for value in exported_graph.output] == ["action_delta"]
    loaded = load_artifact_metadata(output, _expectation(metadata))
    assert loaded == metadata


def test_failed_export_publishes_no_final_file(tmp_path: Path, monkeypatch):
    output = tmp_path / "failed_bundle"

    def fail_export(*_args, **_kwargs):
        raise RuntimeError("intentional export failure")

    monkeypatch.setattr(torch.onnx, "export", fail_export)
    with pytest.raises(RuntimeError, match="intentional"):
        export_action_residual_bundle(_fixture_tensors(), output, _spec())
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_export_refuses_to_overwrite_an_artifact(tmp_path: Path):
    output, _ = _export(tmp_path)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        export_action_residual_bundle(_fixture_tensors(), output, _spec())
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


@pytest.mark.parametrize("opset", [16, 18])
def test_schema_v1_export_rejects_other_opsets(tmp_path: Path, opset: int):
    output = tmp_path / f"opset_{opset}"
    with pytest.raises(ValueError, match="opset exactly 17"):
        export_action_residual_bundle(
            _fixture_tensors(),
            output,
            replace(_spec(), opset=opset),
        )
    assert not output.exists()


def test_acceptance_report_is_fail_closed_against_input_and_bundle_collisions(
    tmp_path: Path,
):
    acceptance = _acceptance_module()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    model = bundle / "action_residual.onnx"
    metadata = bundle / "action_residual.metadata.json"
    checkpoint = tmp_path / "last.pt"
    overlay = tmp_path / "overlay.json"
    release_base = tmp_path / "release_base.onnx"
    for path in (model, metadata, checkpoint, overlay, release_base):
        path.write_bytes(b"fixture")

    common = {
        "artifact_directory": bundle,
        "checkpoint": checkpoint,
        "deployment_overlay": overlay,
        "model": model,
        "metadata": metadata,
        "release_artifacts": (release_base,),
    }
    for collision in (
        checkpoint,
        overlay,
        model,
        metadata,
        release_base,
        bundle / "other.json",
    ):
        with pytest.raises(ValueError):
            acceptance._validate_report_destination(
                report=collision,
                overwrite=True,
                **common,
            )

    report = tmp_path / "acceptance.json"
    report.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite-report"):
        acceptance._validate_report_destination(
            report=report,
            overwrite=False,
            **common,
        )
    assert (
        acceptance._validate_report_destination(
            report=report,
            overwrite=True,
            **common,
        )
        == report.resolve()
    )


def test_acceptance_atomic_report_refuses_racing_overwrite(tmp_path: Path):
    acceptance = _acceptance_module()
    report = tmp_path / "acceptance.json"
    acceptance._atomic_json(report, {"first": True}, overwrite=False)
    before = report.read_bytes()
    with pytest.raises(FileExistsError):
        acceptance._atomic_json(report, {"second": True}, overwrite=False)
    assert report.read_bytes() == before


@pytest.mark.parametrize("shape", [(2, 3), (1, 5)])
@pytest.mark.parametrize("pattern", ["off", "on", "mixed"])
def test_pytorch_and_onnxruntime_dynamic_parity(
    tmp_path: Path,
    shape: tuple[int, int],
    pattern: str,
):
    output, metadata = _export(tmp_path)
    session = load_onnxruntime_session(output, metadata)
    generator = np.random.default_rng(65537 + shape[0] + shape[1])
    context = generator.normal(size=(*shape, 7)).astype(np.float32)
    condition, gate = _condition(shape, pattern)
    if pattern == "mixed":
        context[~gate] = np.nan
        condition[~gate] = np.nan
    module = _pytorch_module()
    with torch.no_grad():
        expected = module(torch.from_numpy(context), torch.from_numpy(condition)).numpy()
    actual = session.run(
        ["action_delta"],
        {
            "release_action_context": context,
            "motion_compliance_condition": condition,
        },
    )[0]
    assert np.isfinite(actual).all()
    assert np.allclose(actual[gate], expected[gate], atol=1.0e-5, rtol=1.0e-5)
    assert np.array_equal(actual[~gate], np.zeros_like(actual[~gate]))
    assert np.max(np.abs(actual)) <= 0.25


class _CountingSession:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.calls = 0

    def run(self, output_names, input_feed):
        self.calls += 1
        return self.wrapped.run(output_names, input_feed)


def test_runtime_hard_switch_and_mixed_row_isolation(tmp_path: Path):
    output, metadata = _export(tmp_path)
    session = _CountingSession(load_onnxruntime_session(output, metadata))
    shape = (2, 4)
    release = np.linspace(-0.5, 0.5, 2 * 4 * 5, dtype=np.float32).reshape(*shape, 5)
    release[0, 1, 0] = -0.0
    context = np.ones((*shape, 7), dtype=np.float32)
    condition, gate = _condition(shape, "mixed")
    context[~gate] = np.nan
    condition[~gate] = np.nan

    disabled = ActionResidualRuntime(metadata, enabled=False, session=session)
    disabled_action = disabled.compose(release, context, condition, gate)
    assert session.calls == 0
    assert np.array_equal(disabled_action.view(np.uint32), release.view(np.uint32))

    enabled = ActionResidualRuntime(metadata, enabled=True, session=session)
    all_off = enabled.compose(release, context, condition, np.zeros(shape, dtype=np.bool_))
    assert session.calls == 0
    assert np.array_equal(all_off.view(np.uint32), release.view(np.uint32))

    composed = enabled.compose(release, context, condition, gate)
    assert session.calls == 1
    assert np.array_equal(composed[~gate].view(np.uint32), release[~gate].view(np.uint32))
    assert np.max(np.abs(composed[gate] - release[gate])) <= 0.25
    assert np.isfinite(composed).all()


@pytest.mark.parametrize(
    "gate",
    [
        np.array([[0.0, np.nan]], dtype=np.float32),
        np.array([[0.0, 0.5]], dtype=np.float32),
        np.array([[0, 2]], dtype=np.int64),
    ],
)
def test_runtime_rejects_nonfinite_or_nonbinary_gate(tmp_path: Path, gate: np.ndarray):
    _, metadata = _export(tmp_path)
    runtime = ActionResidualRuntime(metadata, enabled=False)
    with pytest.raises(ValueError, match="finite binary"):
        runtime.compose(
            np.zeros((1, 2, 5), dtype=np.float32),
            np.zeros((1, 2, 7), dtype=np.float32),
            np.zeros((1, 2, 3), dtype=np.float32),
            gate,
        )


def test_runtime_rejects_invalid_enabled_condition(tmp_path: Path):
    output, metadata = _export(tmp_path)
    runtime = ActionResidualRuntime(
        metadata,
        enabled=True,
        session=load_onnxruntime_session(output, metadata),
    )
    release = np.zeros((1, 1, 5), dtype=np.float32)
    context = np.zeros((1, 1, 7), dtype=np.float32)
    gate = np.ones((1, 1), dtype=np.bool_)
    for condition in (
        np.array([[[0.0, 10.0, 200.0]]], dtype=np.float32),
        np.array([[[1.0, 0.0, 200.0]]], dtype=np.float32),
        np.array([[[1.0, 10.0, np.nan]]], dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            runtime.compose(release, context, condition, gate)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("metadata_sha256", "0" * 64), "metadata digest"),
        (
            lambda value: value["source_checkpoint"].__setitem__("global_step", 9),
            "metadata digest",
        ),
        (lambda value: value.__setitem__("schema", "bad"), "unsupported artifact"),
        (lambda value: value["model"].__setitem__("opset", 18), "opset exactly 17"),
        (lambda value: value.__setitem__("site_layout", ["wrong"]), "metadata digest"),
        (lambda value: value.__setitem__("action_layout", ["wrong"] * 5), "duplicate"),
    ],
)
def test_metadata_tampering_is_rejected(tmp_path: Path, mutation, message: str):
    output, _ = _export(tmp_path)
    metadata_path = output / "action_residual.metadata.json"
    value = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutation(value)
    metadata_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_artifact_metadata(output, _expectation(value))


def test_host_expectation_rejects_digest_site_action_and_checkpoint(tmp_path: Path):
    output, metadata = _export(tmp_path)
    base = _expectation(metadata)
    mutations = (
        {"metadata_sha256": "0" * 64},
        {"checkpoint_sha256": "b" * 64},
        {"checkpoint_global_step": 8},
        {"site_layout": ("other",)},
        {"action_layout": tuple(reversed(base.action_layout))},
        {"release_context_width": 8},
    )
    for values in mutations:
        expectation = ArtifactExpectation(**{**base.__dict__, **values})
        with pytest.raises(ValueError):
            load_artifact_metadata(output, expectation)


def test_host_expectation_pins_arbitrary_named_release_artifacts(tmp_path: Path):
    output, metadata = _export(tmp_path)
    base_files = []
    for index in range(3):
        path = tmp_path / f"base_{index}.bin"
        path.write_bytes(f"base-{index}\n".encode())
        base_files.append(path)
    pins = tuple(
        ReleaseArtifactPin(
            name=f"component_{index}",
            path=path,
            sha256=_sha256(path),
        )
        for index, path in enumerate(base_files)
    )
    expectation = replace(_expectation(metadata), release_artifacts=pins)
    assert load_artifact_metadata(output, expectation) == metadata

    base_files[1].write_bytes(b"incompatible base\n")
    with pytest.raises(ValueError, match="component_1.*host-pinned base"):
        load_artifact_metadata(output, expectation)


def test_model_digest_tampering_is_rejected(tmp_path: Path):
    output, metadata = _export(tmp_path)
    with (output / "action_residual.onnx").open("ab") as model:
        model.write(b"tamper")
    with pytest.raises(ValueError, match="model digest"):
        load_artifact_metadata(output, _expectation(metadata))


def _sonic_overlay() -> dict:
    path = (
        Path(__file__).parents[2]
        / "gear_sonic_deploy"
        / "policy"
        / "motion_compliance"
        / "action_residual_overlay.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))[
        "motion_compliance_action_residual"
    ]


def test_sonic_adapter_assembles_true_training_order_and_condition():
    from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER

    config = _sonic_overlay()
    contract = SonicResidualContextContract.from_mapping(config)
    assert contract.release_context_width == 994
    assert contract.condition_width == 3
    command_config_path = (
        Path(__file__).parents[1]
        / "config"
        / "manager_env"
        / "commands"
        / "terms"
        / "motion_compliance.yaml"
    )
    command_sites = yaml.safe_load(command_config_path.read_text(encoding="utf-8"))[
        "motion_compliance"
    ]["site_body_names"]
    assert list(contract.site_layout) == command_sites
    assert list(contract.action_layout) == G1_ISAACLab_ORDER
    token = np.full((2, 3, 64), 1.0, dtype=np.float32)
    actor_obs = np.full((2, 3, 930), 2.0, dtype=np.float32)
    context = assemble_release_action_context(token, actor_obs, contract)
    assert context.shape == (2, 3, 994)
    assert np.array_equal(context[..., :64], token)
    assert np.array_equal(context[..., 64:], actor_obs)
    gate = np.array([[True, False, True], [False, True, False]])
    threshold = np.full((2, 3), 10.0, dtype=np.float32)
    condition = encode_condition(gate, threshold, contract)
    assert np.array_equal(condition[..., 0], gate.astype(np.float32))
    assert np.array_equal(condition[..., 1], gate.astype(np.float32) * 10.0)
    assert np.array_equal(condition[..., 2], gate.astype(np.float32) * 200.0)


def test_sonic_config_is_opt_in_and_rejects_invalid_physics_contract():
    config = _sonic_overlay()
    assert config["enabled"] is False
    assert config["artifact_directory"] is None
    assert config["metadata_sha256"] is None
    session_calls = []

    def forbidden_session_factory(*args):
        session_calls.append(args)
        raise AssertionError("disabled overlay must not load a session")

    assert (
        load_sonic_action_residual_deployment(
            config,
            session_factory=forbidden_session_factory,
        )
        is None
    )
    assert session_calls == []
    release = np.array([[[-0.0, 1.0]]], dtype=np.float32)
    hard_off = compose_optional_sonic_action_residual(
        None,
        release,
        np.full((1, 1, 1), np.nan, dtype=np.float32),
        np.full((1, 1, 1), np.nan, dtype=np.float32),
        np.full((1, 1, 3), np.nan, dtype=np.float32),
        np.zeros((1, 1), dtype=np.bool_),
    )
    assert np.array_equal(hard_off.view(np.uint32), release.view(np.uint32))
    with pytest.raises(ValueError, match="finite binary"):
        compose_optional_sonic_action_residual(
            None,
            release,
            np.zeros((1, 1, 1), dtype=np.float32),
            np.zeros((1, 1, 1), dtype=np.float32),
            np.zeros((1, 1, 3), dtype=np.float32),
            np.full((1, 1), np.nan, dtype=np.float32),
        )
    for value in (
        [0.0, 10.0, 200.0],
        [1.0, 0.0, 200.0],
        [1.0, 10.0, 0.0],
        [1.0, 10.0, float("nan")],
    ):
        invalid = deepcopy(config)
        invalid["default_enabled_condition"] = value
        with pytest.raises(ValueError):
            SonicResidualContextContract.from_mapping(invalid)
    invalid = deepcopy(config)
    invalid["condition_width"] = 4
    with pytest.raises(ValueError, match="condition width must be exactly 3"):
        SonicResidualContextContract.from_mapping(invalid)
    invalid = deepcopy(config)
    invalid["context_layout"] = list(reversed(invalid["context_layout"]))
    with pytest.raises(ValueError, match="order or names"):
        SonicResidualContextContract.from_mapping(invalid)


def test_sonic_deployment_validates_artifact_layout(tmp_path: Path):
    _, metadata = _export(tmp_path)
    config = {
        "context_layout": [
            {"name": "robot_motion_token", "width": 2},
            {"name": "actor_observation", "width": 5},
        ],
        "condition_width": 3,
        "default_enabled_condition": [1.0, 1.0, 10.0],
        "site_layout": ["site_a", "site_b", "site_c"],
        "action_layout": [f"action_{index}" for index in range(5)],
    }
    contract = SonicResidualContextContract.from_mapping(config)
    runtime = ActionResidualRuntime(metadata, enabled=False)
    deployment = SonicActionResidualDeployment(runtime, contract)
    assert deployment.contract is contract
    invalid = deepcopy(config)
    invalid["site_layout"] = ["site_a"]
    with pytest.raises(ValueError, match="site layout"):
        SonicActionResidualDeployment(
            runtime,
            SonicResidualContextContract.from_mapping(invalid),
        )


def test_sonic_enabled_factory_pins_artifact_before_session(tmp_path: Path):
    output, metadata = _export(tmp_path)
    host_base = Path(__file__).resolve()
    release_pins = (
        ReleaseArtifactPin(
            name="fixture_base",
            path=host_base,
            sha256=_sha256(host_base),
        ),
    )
    config = {
        "enabled": True,
        "artifact_directory": str(output),
        "metadata_sha256": metadata["metadata_sha256"],
        "checkpoint_sha256": "a" * 64,
        "checkpoint_global_step": 7,
        "schema": metadata["schema"],
        "max_abs_delta": 0.25,
        "context_layout": [
            {"name": "robot_motion_token", "width": 2},
            {"name": "actor_observation", "width": 5},
        ],
        "condition_width": 3,
        "default_enabled_condition": [1.0, 1.0, 10.0],
        "release_artifacts": [
            {"name": "fixture_base", "sha256": _sha256(host_base)}
        ],
        "site_layout": ["site_a", "site_b", "site_c"],
        "action_layout": [f"action_{index}" for index in range(5)],
    }
    sessions = []

    def session_factory(artifact_directory, validated_metadata):
        assert Path(artifact_directory) == output
        assert validated_metadata == metadata
        session = load_onnxruntime_session(artifact_directory, validated_metadata)
        sessions.append(session)
        return session

    deployment = load_sonic_action_residual_deployment(
        config,
        release_artifacts=release_pins,
        session_factory=session_factory,
    )
    assert isinstance(deployment, SonicActionResidualDeployment)
    assert len(sessions) == 1

    invalid = deepcopy(config)
    invalid["metadata_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="host-pinned"):
        load_sonic_action_residual_deployment(
            invalid,
            release_artifacts=release_pins,
            session_factory=lambda *_args: pytest.fail("must validate before session"),
        )

    invalid = deepcopy(config)
    invalid["release_artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="host-owned pins"):
        load_sonic_action_residual_deployment(
            invalid,
            release_artifacts=release_pins,
            session_factory=lambda *_args: pytest.fail("must validate before session"),
        )

    invalid = deepcopy(config)
    invalid["artifact_directory"] = None
    with pytest.raises(ValueError, match="artifact_directory"):
        load_sonic_action_residual_deployment(
            invalid,
            release_artifacts=release_pins,
        )
