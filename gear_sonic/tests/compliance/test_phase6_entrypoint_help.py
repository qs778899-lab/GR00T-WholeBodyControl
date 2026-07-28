"""Focused regression for the Phase-6 Isaac entrypoint help repair."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
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
_FINAL_AUDIT_PATH = (
    _REPOSITORY_ROOT
    / "tasks/chip_compliance_finetune/artifacts/phase6_final_audit.py"
)


def _load_final_audit_module():
    spec = importlib.util.spec_from_file_location("chip_phase6_final_audit", _FINAL_AUDIT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - import machinery failure
        raise ImportError(_FINAL_AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _initialize_test_repository(repository: Path) -> str:
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Phase6 Test")
    _git(repository, "config", "user.email", "phase6@example.invalid")
    (repository / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    _git(repository, "add", "--", "baseline.txt")
    _git(repository, "commit", "--quiet", "-m", "baseline")
    return _git(repository, "rev-parse", "HEAD")


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


class ProtectedRefAdvanceAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = _load_final_audit_module()

    def _snapshot_values(self) -> dict[str, str]:
        values = {
            ref_name: self.audit.PRE_EXPERIMENT_MAIN_COMMIT
            for ref_name in self.audit.DOCUMENTATION_ADVANCED_REFS
        }
        values["refs/heads/unrelated"] = "1" * 40
        return values

    def test_unchanged_ref_snapshot_remains_accepted(self) -> None:
        expected = self._snapshot_values()
        self.assertEqual(
            self.audit._classify_ref_state(expected, dict(expected)),
            "UNCHANGED",
        )

    def test_exact_current_documentation_advance_is_accepted(self) -> None:
        # Resolve from the shared workspace explicitly: the experiment worktree
        # intentionally does not duplicate official assets or the immutable ref snapshot.
        snapshot = Path(
            "/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/"
            "existing_refs_before.txt"
        )
        self.assertEqual(
            self.audit._audit_ref_snapshot(_REPOSITORY_ROOT, snapshot),
            "PINNED_DOCS_ONLY_FAST_FORWARD",
        )

    def test_partial_documentation_ref_advance_is_rejected(self) -> None:
        expected = self._snapshot_values()
        actual = dict(expected)
        one_ref = sorted(self.audit.DOCUMENTATION_ADVANCED_REFS)[0]
        actual[one_ref] = self.audit.DOCUMENTATION_MAIN_COMMIT
        with self.assertRaisesRegex(AssertionError, "outside pinned exception"):
            self.audit._classify_ref_state(expected, actual)

    def test_future_or_unrelated_ref_move_is_rejected(self) -> None:
        expected = self._snapshot_values()
        cases = []
        future = dict(expected)
        for ref_name in self.audit.DOCUMENTATION_ADVANCED_REFS:
            future[ref_name] = "f" * 40
        cases.append(future)
        unrelated = dict(expected)
        for ref_name in self.audit.DOCUMENTATION_ADVANCED_REFS:
            unrelated[ref_name] = self.audit.DOCUMENTATION_MAIN_COMMIT
        unrelated["refs/heads/unrelated"] = "2" * 40
        cases.append(unrelated)
        for actual in cases:
            with self.subTest(actual=actual), self.assertRaisesRegex(
                AssertionError,
                "outside pinned exception",
            ):
                self.audit._classify_ref_state(expected, actual)

    def test_single_addition_commit_contract_accepts_exact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            old_commit = _initialize_test_repository(repository)
            expected_paths = frozenset({"docs/a.md", "tests/test_a.py"})
            for relative in expected_paths:
                path = repository / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{relative}\n", encoding="utf-8")
            _git(repository, "add", "--", *sorted(expected_paths))
            _git(repository, "commit", "--quiet", "-m", "documentation")
            new_commit = _git(repository, "rev-parse", "HEAD")
            self.audit._validate_single_addition_commit(
                repository,
                old_commit,
                new_commit,
                expected_paths,
            )

    def test_multi_commit_advance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            old_commit = _initialize_test_repository(repository)
            for index in range(2):
                path = repository / f"docs/{index}.md"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{index}\n", encoding="utf-8")
                _git(repository, "add", "--", path.relative_to(repository).as_posix())
                _git(repository, "commit", "--quiet", "-m", f"documentation {index}")
            with self.assertRaisesRegex(AssertionError, "one direct commit"):
                self.audit._validate_single_addition_commit(
                    repository,
                    old_commit,
                    _git(repository, "rev-parse", "HEAD"),
                    frozenset({"docs/0.md", "docs/1.md"}),
                )

    def test_modified_or_extra_path_is_rejected(self) -> None:
        for mode in ("modified", "extra"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                repository = Path(directory)
                old_commit = _initialize_test_repository(repository)
                expected_paths = frozenset({"docs/a.md"})
                if mode == "modified":
                    path = repository / "baseline.txt"
                    path.write_text("modified\n", encoding="utf-8")
                    _git(repository, "add", "--", "baseline.txt")
                else:
                    for relative in ("docs/a.md", "docs/extra.md"):
                        path = repository / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(f"{relative}\n", encoding="utf-8")
                    _git(repository, "add", "--", "docs/a.md", "docs/extra.md")
                _git(repository, "commit", "--quiet", "-m", mode)
                with self.assertRaisesRegex(AssertionError, "path/status mismatch"):
                    self.audit._validate_single_addition_commit(
                        repository,
                        old_commit,
                        _git(repository, "rev-parse", "HEAD"),
                        expected_paths,
                    )


if __name__ == "__main__":
    unittest.main()
