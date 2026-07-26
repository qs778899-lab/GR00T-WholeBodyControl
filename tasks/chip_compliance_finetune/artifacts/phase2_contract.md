# Phase 2 simulator contract

## Scope and opt-in boundary

Phase 2 adds only the SONIC/Isaac Lab adapter, an independent command/event state, a sparse compliance-target observation, and derived Hydra groups. The released `sonic_release.yaml`, policy, critic, checkpoint, reward, training, and deployment paths remain unchanged. The derived experiment starts with `force.enabled: false`.

## Target and frame semantics

- Reference-motion and articulation bodies are resolved independently by name and kept in typed index spaces.
- Reference sites and actual application points use the same configured link-local offsets.
- Reference position, current end-effector position, and force are transformed into one declared world/anchor-local/heading-local common frame before target math.
- The nominal target is the original reference unless the optional target damper is enabled for an active compliant site. A newly active damper site is seeded from the current end-effector position, preventing a stale stiff-mode goal jump.
- The CHIP hindsight target is `g_observed = g_nominal - C * f_on_robot`. Inactive or globally disabled entries select the original target exactly.
- The simulator exposes only the current final applied force. It is explicitly repeated over the future target horizon for tensor alignment; this repetition is not a prediction of future force.
- Observation construction does not mutate the motion command's reference buffers.

## CHIP sampling fidelity and SONIC safety adaptation

The default training envelope follows CHIP: force magnitude is sampled uniformly from `0–40 N`, duration uniformly from `1–3 s`, and inverse Cartesian stiffness is sampled discretely from `{0, 0.02, 0.05} m/N`. The discrete set is configurable as `compliance_values_m_per_n`.

Enabled control steps draw fixed `[num_envs, ...]` candidates from a command-owned generator and combine them with fixed-size `due_mask`/`start_mask` tensors. This avoids dynamic CUDA index construction while keeping per-environment pulse timing asynchronous. Disabled control steps draw nothing from either the private generator or process-global CPU/CUDA generators.

SONIC additionally limits the resultant wrench around the configured anchor on every simulation step. All selected site forces in an environment are uniformly scaled by

`min(1, 30 N / |sum(F)|, 20 N·m / |sum(r × F)|)`.

Consequently a CHIP-envelope sample can be smaller than its requested magnitude after the `30 N` force or `20 N·m` torque safety cap. The state and hindsight observation retain the final re-limited force actually sent to the simulator, not the requested peak.

## Wrench ownership and lifecycle

- Every world force is converted with the current link quaternion to `R_link^T F_world` immediately before writing.
- The configured application offset is passed in link-local coordinates with `is_global=False`; this avoids stale cached link poses in the permanent wrench composer.
- A command that starts globally disabled never touches the composer.
- If an enabled command previously owned wrench rows and is switched off, it clears those rows once and releases the host-side ownership gate. Later disabled steps perform no writer call.
- Full or partial resets clear enable/mask/compliance/force/pulse state and reseed the corresponding damper target from the current end effector. Selected composer rows are cleared whenever the command owns them.

## Hot-path boundary

Public checked APIs retain shape/device/finiteness validation. The lifecycle-validated simulator path uses explicit prevalidated tensor functions, and the command mixin overrides inherited `CommandTerm.compute(dt)` so no dynamic resample IDs are created. Countdown, pulse start/completion, and damper seeding use fixed-shape masks with `torch.where`/`copy_`; the per-step CUDA path performs no `.item()`, tensor truth conversion, `aten::nonzero`, or `aten::_local_scalar_dense`.

The synchronization audit has two deliberately different layers. A portable 4096-environment × 14-site test binds `ComplianceOperationalControl.compute(dt)` to deterministic tensor fixtures and a fake composer; it exercises scale, arbitrary-site shapes, and target-damper updates without requiring Isaac Lab. The one-environment AppLauncher smoke separately profiles the actual `SonicComplianceCommand` instance, articulation pose/index and link-frame rotation helpers, `ArticulationWrenchAdapter`, and Isaac Lab `WrenchComposer`. Only after 100 disabled steps pass does that smoke force the private countdown due and call the bound `command.compute(dt)` under both `TorchDispatchMode` and CPU/CUDA profiler activities. It then disables immediately, before another environment step, and proves the real composer's owned force and torque rows are zero. Both layers reject `aten::nonzero` and `aten::_local_scalar_dense`; the large fixture is not represented as a real Isaac-bound instance.
