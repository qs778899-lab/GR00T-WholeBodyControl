#!/usr/bin/env python3
"""Compile and run the production C++ residual against system ORT 1.16."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_METADATA_SHA256 = (
    "e954d093603d910e8cde4c2a5842db4d734d1ec8fbc3180f03a9399b5c17d8c5"
)


def _regular(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _enabled_config(source: Path, target: Path, bundle: Path) -> None:
    text = source.read_text(encoding="utf-8")
    replacements = {
        "  enabled: false\n": "  enabled: true\n",
        "  artifact_directory: null\n": f"  artifact_directory: {bundle}\n",
        "  metadata_sha256: null\n": (
            f"  metadata_sha256: {EXPECTED_METADATA_SHA256}\n"
        ),
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError(f"overlay template field is not unique: {old.strip()}")
        text = text.replace(old, new)
    target.write_text(text, encoding="utf-8")


def _assert_production_hook() -> None:
    source = (
        REPO_ROOT
        / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp"
    ).read_text(encoding="utf-8")
    required = (
        "--motion-compliance-overlay",
        "SonicMotionComplianceActionResidual::Load(",
        "motion_compliance_action_residual_->Compose(",
        "selected_action[isaaclab_to_mujoco[i]]",
        "last_action[i] = static_cast<double>(selected_action[i])",
    )
    missing = [needle for needle in required if needle not in source]
    if missing:
        raise AssertionError(f"production G1 hook is incomplete: {missing}")
    compose_at = source.index("motion_compliance_action_residual_->Compose(")
    remap_at = source.index("selected_action[isaaclab_to_mujoco[i]]")
    if compose_at >= remap_at:
        raise AssertionError("residual must compose before IsaacLab-to-MuJoCo remap")

    # The command-line value and keyboard adjustments must reach the same
    # storage read by each composite input manager.  These source-level checks
    # complement the compiled adapter test without constructing networked
    # manager objects (which would open sockets and worker threads).
    proxy_contracts = {
        "interface_manager.hpp": (
            "void SetVR3PointCompliance(",
            "keyboard_->SetVR3PointCompliance(compliance)",
            "gamepad_->SetVR3PointCompliance(compliance)",
            "zmq_->SetVR3PointCompliance(compliance)",
            "ros2_->SetVR3PointCompliance(compliance)",
        ),
        "gamepad_manager.hpp": (
            "void SetVR3PointCompliance(",
            "InputInterface::SetVR3PointCompliance(compliance)",
            "zmq_->SetVR3PointCompliance(compliance)",
            "AdjustLeftHandCompliance(0.1)",
            "AdjustRightHandCompliance(-0.1)",
        ),
        "zmq_manager.hpp": (
            "void SetVR3PointCompliance(",
            "InputInterface::SetVR3PointCompliance(compliance)",
            "pose_interface_->SetVR3PointCompliance(compliance)",
        ),
    }
    input_headers = (
        REPO_ROOT
        / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface"
    )
    for filename, needles in proxy_contracts.items():
        header = (input_headers / filename).read_text(encoding="utf-8")
        missing = [needle for needle in needles if needle not in header]
        if missing:
            raise AssertionError(
                f"compliance input proxy is incomplete in {filename}: {missing}"
            )

    cli_contract = (
        "0x7ff0000000000000ULL",
        "std::memcpy(&bits, &value, sizeof(bits))",
        "both zero = off",
    )
    missing = [needle for needle in cli_contract if needle not in source]
    if missing:
        raise AssertionError(f"operator gate contract is incomplete: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--observation-config", type=Path, required=True)
    args = parser.parse_args()

    template = _regular(
        REPO_ROOT
        / "gear_sonic_deploy/policy/motion_compliance/action_residual_overlay.yaml",
        "overlay template",
    )
    bundle = args.bundle.resolve(strict=True)
    if not bundle.is_dir() or bundle.is_symlink():
        raise ValueError("bundle must be a real directory")
    metadata = _regular(bundle / "action_residual.metadata.json", "metadata")
    if _sha256(metadata) == EXPECTED_METADATA_SHA256:
        raise AssertionError(
            "external metadata pin is a canonical payload digest, not file SHA-256"
        )
    decoder = _regular(args.decoder, "release decoder")
    encoder = _regular(args.encoder, "release encoder")
    observation_config = _regular(args.observation_config, "observation config")
    _assert_production_hook()

    portable_source = (
        REPO_ROOT
        / "gear_sonic_deploy/src/motion_compliance/src/action_residual_overlay.cpp"
    )
    adapter_source = (
        REPO_ROOT
        / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/"
        "motion_compliance_action_residual.cpp"
    )
    smoke_source = Path(__file__).with_suffix(".cpp")
    for source in (portable_source, adapter_source, smoke_source):
        _regular(source, "C++ source")

    with tempfile.TemporaryDirectory(prefix="motion_compliance_cpp_ort_") as directory:
        temporary = Path(directory)
        enabled = temporary / "enabled_overlay.yaml"
        wrong_decoder = temporary / "wrong_decoder.onnx"
        binary = temporary / "phase5_cpp_ort_smoke"
        missing = temporary / "must_not_exist"
        _enabled_config(template, enabled, bundle)
        wrong_decoder.write_bytes(b"not the pinned release decoder\n")
        command = [
            "g++",
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-ffast-math",
            "-fno-fast-math",
            "-I",
            str(REPO_ROOT / "gear_sonic_deploy/src/motion_compliance/include"),
            "-I",
            str(
                REPO_ROOT
                / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include"
            ),
            "-I",
            "/opt/onnxruntime/include",
            str(portable_source),
            str(adapter_source),
            str(smoke_source),
            "-L",
            "/opt/onnxruntime/lib",
            "-Wl,-rpath,/opt/onnxruntime/lib",
            "-lonnxruntime",
            "-lyaml-cpp",
            "-lmd",
            "-pthread",
            "-o",
            str(binary),
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        subprocess.run(
            [
                str(binary),
                str(template),
                str(enabled),
                str(decoder),
                str(encoder),
                str(observation_config),
                str(wrong_decoder),
                str(missing),
            ],
            cwd=REPO_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
