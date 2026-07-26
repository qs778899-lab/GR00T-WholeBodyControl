# CHIP compliance finetune test matrix

Every test listed for the active phase is mandatory. A later phase must not start until all tests for the current phase pass and `status.md` is updated.

## Phase 1 — Baseline, contracts, and architecture skeleton

Run from the repository root:

1. Contract unit tests

   ```bash
   PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s gear_sonic/tests/compliance -p 'test_*.py' -v
   ```

   Required coverage: structured frame/anchor/rotation and mismatch validation; global and mixed-batch/per-site hard-off forward/backward identity; global-off NaN operand isolation; strict finite reference/force and finite non-negative compliance validation when active; zero-compliance identity; CHIP sign; static/future force broadcasting; isotropic and Cartesian-axis anisotropic compliance; site masking; arbitrary 17-site support; displacement limiting; differentiability; no input mutation; target-damper update/full reset/partial reset; exposure metrics using enable, compliance, and site mask; dual reference/articulation name resolution with different runtime orders; `str`/`bytes` sequence rejection; and rejection of invalid/ambiguous/five-dimensional shapes.

2. Import and syntax smoke

   ```bash
   PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
   from pathlib import Path

   for root in (Path("gear_sonic/compliance_control"), Path("gear_sonic/tests/compliance")):
       for path in root.rglob("*.py"):
           compile(path.read_bytes(), str(path), "exec")
   PY
   PYTHONDONTWRITEBYTECODE=1 python -c "from gear_sonic.compliance_control import CartesianFrameSpec, ComplianceTargetSpec, TargetDamper, apply_hindsight_target; print(CartesianFrameSpec, ComplianceTargetSpec, TargetDamper, apply_hindsight_target)"
   ```

   The unit suite must also verify by AST inspection that tracker-agnostic core modules have no Isaac Lab/SONIC environment imports, and that the SONIC name resolver contains no literal `14`/`29` index contract. Successful execution in this environment is the no-IsaacLab runtime proof.

3. Patch hygiene

   ```bash
   PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
   from pathlib import Path

   roots = (
       Path("gear_sonic/compliance_control"),
       Path("gear_sonic/tests/compliance"),
       Path("tasks/chip_compliance_finetune"),
   )
   for root in roots:
       for path in root.rglob("*"):
           assert path.name not in {"__pycache__", ".pytest_cache"}, \
               f"cache directory remains: {path}"
           assert path.suffix not in {".pyc", ".pyo"}, f"compiled Python remains: {path}"
           if not path.is_file():
               continue
           text = path.read_text(encoding="utf-8")
           assert text.endswith("\n"), f"missing final newline: {path}"
           assert not text.endswith("\n\n"), f"extra EOF blank line: {path}"
           assert all(line == line.rstrip() for line in text.splitlines()), \
               f"trailing whitespace: {path}"
   PY
   git diff --check
   git diff --cached --check
   ```

The training entrypoint `--help` smoke requires Isaac Lab, which is not installed in the Phase 1 CPU environment and is mandatory in Phase 4 before any finetune result can pass.

## Phase 2 — Simulator command/event and observation integration

- Phase 1 matrix.
- Isaac Lab config composition for the new compliance experiment only.
- CPU/GPU tensor tests for `[env, future, site, xyz]` force/target alignment.
- Resolve the same ordered sites from deliberately different reference-motion and articulation body orders; assert each consumer receives only its typed index space.
- With a known non-zero yaw/full rotation, transform a basis force and reference target into the declared common frame and numerically verify CHIP sign and axis; identity-only tests do not satisfy this check.
- Target-damper state/update plus full and partial environment reset tests; no stale goal may survive reset.
- One-environment simulator smoke with force disabled, then enabled.
- Assertions that reference buffers are unchanged after observation construction.
- Hydra body-name resolver tests for full and partial reference-body sets without fixed indices.

## Phase 3 — Checkpoint-compatible policy/critic integration

- Phase 2 matrix.
- Released checkpoint load with documented missing/new keys only.
- Disabled-mode policy output parity against the release path.
- Nonzero compliance-adapter gradient and no actor access to privileged force.
- Encoder selection and auxiliary-loss tests for G1/teleop/SMPL samples.
- Assert the compliance-only actor path consumes the Phase 2 damped/hindsight target while the released dense reference/reward path remains unchanged.

## Phase 4 — Low-resource finetune smoke and parity regression

- Phase 3 matrix.
- `train_agent_trl.py --help` in the pinned Isaac Lab training environment.
- The exact `sonic_backup` 16-environment, 5-iteration release-config command in `plan.md`, using the pinned single robot PKL, SMPL sample directory, official checkpoint, and `use_wandb=false`.
- Compliance-enabled 16-environment smoke with finite losses and checkpoint resume.
- Bounded log/artifact size and cleanup verification.

The 2026-07-27 `580.159` kernel / `580.173` userspace NVIDIA mismatch is an explicit blocker, not a waiver for these tests.

## Phase 5 — Tracking/compliance evaluation and export

- Phase 4 matrix.
- Frame/time-aligned stiff baseline versus compliant checkpoint evaluation.
- Upper-limb endpoint, local/global MPJPE, success/fall, displacement, and force metrics.
- ONNX export plus disabled-mode inference parity.

## Phase 6 — Final regression and handoff

- All earlier matrices.
- Full documented regression/golden comparison and output-layout check.
- Final artifact/temporary-file audit.
