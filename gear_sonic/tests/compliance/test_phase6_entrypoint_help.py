"""Focused regression for the Phase-6 Isaac entrypoint help repair."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = {
    "run_chip_compliance_smoke.py": {
        "required": ("--motion-file", "--smpl-motion-dir"),
        "main_ast_sha256": (
            "1eb15ec20f859b19732fd3a0fb000a59c27c67bf0f68bcc9144ceb5c24d7d947"
        ),
    },
    "run_chip_phase3_shape_smoke.py": {
        "required": ("--motion-file", "--smpl-motion-dir", "--checkpoint"),
        "main_ast_sha256": (
            "024584ec5abc96261f06578197ff56af8b518b64c50fee922ef4a7fea26e2c0b"
        ),
    },
    "run_chip_phase5_rollout.py": {
        "required": (
            "--mode",
            "--motion-file",
            "--smpl-motion-dir",
            "--checkpoint",
            "--trace",
            "--summary",
        ),
        "main_ast_sha256": (
            "3b591c16c506f4de63e12ea49a20f7dc6f90291cf07761d2adbddf781639be54"
        ),
    },
}


def _isaaclab_available() -> bool:
    try:
        return importlib.util.find_spec("isaaclab.app") is not None
    except ModuleNotFoundError:
        return False


class IsaacEntrypointHelpRegressionTest(unittest.TestCase):
    def test_runtime_main_ast_is_unchanged_by_cli_only_repair(self) -> None:
        for script_name, contract in _SCRIPTS.items():
            with self.subTest(script=script_name):
                path = _REPOSITORY_ROOT / "gear_sonic/scripts" / script_name
                source = path.read_text(encoding="utf-8")
                module = ast.parse(source)
                main = next(
                    node
                    for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main"
                )
                digest = hashlib.sha256(
                    ast.dump(main, include_attributes=False).encode("utf-8")
                ).hexdigest()
                self.assertEqual(digest, contract["main_ast_sha256"])
                self.assertLess(
                    source.index("except SystemExit as error:"),
                    source.rindex("except BaseException:"),
                )

    @unittest.skipUnless(_isaaclab_available(), "Isaac Lab unavailable")
    def test_help_is_zero_non_writing_and_required_contract_remains(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        for script_name, contract in _SCRIPTS.items():
            path = _REPOSITORY_ROOT / "gear_sonic/scripts" / script_name
            with self.subTest(script=script_name, mode="help"):
                help_result = subprocess.run(
                    [sys.executable, "-B", str(path), "--help"],
                    cwd=_REPOSITORY_ROOT,
                    env=environment,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=15,
                )
                self.assertEqual(help_result.returncode, 0, help_result.stderr)
                self.assertIn("usage:", help_result.stdout)
                self.assertIn("--device", help_result.stdout)
                self.assertNotIn("Traceback", help_result.stdout + help_result.stderr)
                self.assertNotIn("[WARN][AppLauncher]", help_result.stdout)
                for argument in contract["required"]:
                    self.assertIn(argument, help_result.stdout)
            with self.subTest(script=script_name, mode="required"):
                missing_result = subprocess.run(
                    [sys.executable, "-B", str(path)],
                    cwd=_REPOSITORY_ROOT,
                    env=environment,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=15,
                )
                self.assertEqual(missing_result.returncode, 2)
                self.assertIn("required", missing_result.stderr)
                self.assertNotIn(
                    "Traceback", missing_result.stdout + missing_result.stderr
                )


if __name__ == "__main__":
    unittest.main()
