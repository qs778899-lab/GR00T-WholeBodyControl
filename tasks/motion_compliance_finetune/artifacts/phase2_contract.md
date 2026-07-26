# Phase-2 SONIC/IsaacLab adapter contract

## Boundary and ownership

- `core/` remains simulator-, robot-, and tracker-independent.
- `adapters/sonic/mapping.py` is the sole owner of configured SONIC body names.
  Reference-motion indices and articulation indices are resolved independently
  by exact, unique body name.
- `command.py` owns persistent per-environment sampling/reference/wrench state,
  calls the portable force formula, and invokes the narrow writer immediately
  after its command update.
- `wrench.py` owns replaceable whole-robot residual limiting.  It leaves all
  requested site wrenches intact and adds an anchor compensation wrench.
- `event.py` owns the physical writer/reset primitives.  It feature-detects
  `permanent_wrench_composer` first and isolates the deprecated setter fallback;
  command update reuses this boundary rather than duplicating writer logic.

## Frames and sign

Reference endpoints include their configured body-local offsets.  Reference
and current endpoint positions are expressed relative to the current anchor
pose, including its full WXYZ rotation, before entering the core.  The core
returns `force_on_robot` in that common frame.  The adapter rotates it into the
world frame, converts each endpoint force into an equivalent body-origin force
and torque, then applies anchor compensation and net-wrench limiting in world
coordinates.  Immediately before writing, it uses each site's and the anchor's
current link quaternion to convert the final force and torque into that link's
local frame and writes with `is_global=false`.  This avoids IsaacLab's modern
composer reusing a link pose cached on its first global write while the robot
moves.  Non-identity rotation, a changing 90-degree body orientation, and
multi-future broadcasting are covered by pure tests.

## Manager lifecycle

The host-side operational switch defaults to `false`.  In that state the
command skips reference/force computation, exposes an exactly zero cached
condition, and leaves an already-clean composer untouched.  If it is switched
off after a wrench was applied, the setter immediately zeros only its owned
body rows and then becomes inert before another physics step.  It never resets
another module's composer rows outside environment reset.  When enabled, each environment
samples one persistent enable bit, an independent site mask, a force threshold,
derived `Kp`, per-site reference offset, and duration using its command-owned
generator.  Host-off reset uses a stable finite timer and no random operation.
The tracking command updates first; the compliance command computes and writes
its tensors for the next physics loop.  No additional interval event exists,
so the release interval-event names, ranges, order, and global RNG sequence stay
unchanged.  Static configuration is validated at construction/resampling
boundaries; private adapter-only unchecked tensor kernels avoid CUDA scalar
extraction in the per-step command path.  Environment reset clears every
dynamic command tensor and the composer, so a previous episode cannot leak a
wrench.  A full-environment reset while host-disabled releases the global dirty
ownership flag; partial or active reset retains it for environments that may
still need a later write.

## Physical-force contract

For every active site and future frame, the portable core separately limits
the nominal compliance term and the 5 N tracking correction before summing.
The physical writer uses future frame zero.  The residual limiter computes net
force and net torque about the configured anchor and adds compensation there so
the applied whole-robot residual is at most 20 N and 10 Nm.  These are synthetic
training limits, not certified contact-force limits.

## Opt-in configuration

The adapter is selected only by the new
`manager_env/commands=tracking/motion_compliance` and
`manager_env/events=tracking/motion_compliance` groups.  Existing release
experiments, `commands.py`, and the robot-motion encoder contract are unchanged.
Even the opt-in composition is physically inert until a later experiment
explicitly sets the host-side command switch to `enabled=true`.

## Real simulator acceptance

`phase2_isaaclab_smoke.py` instantiates a raw `ManagerBasedRLEnv` with one
official robot-motion PKL and its official SMPL directory.  It runs 100 policy
steps with the host-side switch disabled and asserts the composer remains
inactive.  Before stepping, it proves real disabled reset, repeated compute,
and reset-event calls preserve the next CPU and CUDA global RNG samples bit for
bit.  It then enables the command and forces probability one/all sites for
another 100 steps.  It checks returned and command/composer tensors for
finiteness, requires nonzero applied force, and verifies reset clears force and
torque in both command and composer buffers.  The enabled-to-disabled setter is
also checked to clear the real composer immediately, before another step.
