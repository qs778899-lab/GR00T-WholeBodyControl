#!/usr/bin/env python3
"""Validate the production deploy CLI without opening DDS or loading models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def _run(binary: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(binary), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--compile-commands", type=Path, required=True)
    args = parser.parse_args()

    binary = args.binary.resolve(strict=True)
    if not binary.is_file() or binary.is_symlink():
        raise ValueError(f"binary must be a regular non-symlink file: {binary}")

    compile_commands_path = args.compile_commands.resolve(strict=True)
    compile_commands = json.loads(compile_commands_path.read_text(encoding="utf-8"))
    overlay_entries = [
        entry
        for entry in compile_commands
        if Path(entry.get("file", "")).name == "action_residual_overlay.cpp"
    ]
    if len(overlay_entries) != 1:
        raise AssertionError(
            f"expected one portable overlay compile command, got {len(overlay_entries)}"
        )
    command = overlay_entries[0].get("command", "")
    if "-fno-fast-math" not in command or command.rfind(
        "-fno-fast-math"
    ) < command.rfind("-ffast-math"):
        raise AssertionError(
            "portable residual finite checks are not protected from global -ffast-math"
        )

    help_result = _run(binary)
    if help_result.returncode != 0:
        raise AssertionError(f"bare help failed: {help_result.stderr}")
    help_text = help_result.stdout + help_result.stderr
    for required in (
        "--motion-compliance-overlay",
        "use --set-compliance 0 for off",
        "0.0=off/rigid",
    ):
        if required not in help_text:
            raise AssertionError(f"deploy help is missing: {required}")

    invalid_values = (
        "nan",
        "inf",
        "-0.1",
        "0.6",
        "0.1foo",
        "0.1,0.2",
        "0.1,",
        "0.1,0.2,0.3,",
    )
    for value in invalid_values:
        result = _run(
            binary,
            "unused_network",
            "unused_policy.onnx",
            "unused_motion",
            "--set-compliance",
            value,
        )
        output = result.stdout + result.stderr
        if result.returncode != 1:
            raise AssertionError(
                f"invalid compliance {value!r} returned {result.returncode}: {output}"
            )
        if "compliance" not in output.lower():
            raise AssertionError(f"invalid compliance {value!r} lacks diagnostic")
        if "Creating G1Deploy object" in output:
            raise AssertionError(
                f"invalid compliance {value!r} reached DDS/model initialization"
            )

    missing_result = _run(
        binary,
        "unused_network",
        "unused_policy.onnx",
        "unused_motion",
        "--set-compliance",
    )
    if missing_result.returncode != 1:
        raise AssertionError("missing --set-compliance argument was not rejected")

    print(
        "MOTION_COMPLIANCE_PHASE5_DEPLOY_CLI_PASS "
        f"invalid_values={len(invalid_values)} dds_initializations=0 "
        "portable_fast_math=off"
    )


if __name__ == "__main__":
    main()
