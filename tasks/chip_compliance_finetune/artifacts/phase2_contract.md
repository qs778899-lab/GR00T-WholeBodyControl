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

Public checked APIs retain shape/device/finiteness validation. The lifecycle-validated simulator path uses explicit prevalidated tensor functions so its per-step CUDA path performs no `.item()`, tensor truth conversion, or `aten::_local_scalar_dense`. CPU and CUDA dispatch/profiler tests enforce this boundary.
