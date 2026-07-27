# Phase-5 standalone action-residual export

The Phase-4 residual head was trained on a 997-column context ordered as the
64-column robot-motion token, the unchanged 930-column actor observation, and
the 3-column public condition.  The standalone graph therefore accepts the
first 994 columns as `release_action_context`, accepts the condition separately,
and concatenates the condition exactly once inside the graph.
`S` in `[B,S,*]` denotes a dynamic sequence/control-step axis, not the number of
physical contact sites; the physical site layout is validated independently in
the artifact metadata.

The release encoder and decoder remain the source of the 64-column token and
release action.  The new graph only emits a bounded 29-column delta.  The host
must preserve this order:

1. run the unmodified release encoder and decoder;
2. assemble `token + actor_observation` through the thin SONIC adapter;
3. only when the optional host switch and row gate are enabled, run the
   standalone residual model;
4. compose `release_action + action_delta` for enabled rows only, while both
   tensors are still in the decoder's IsaacLab/BFS order;
5. retain the existing deploy-side `isaaclab_to_mujoco` remap after composition.

The repository overlay at
`gear_sonic_deploy/policy/motion_compliance/action_residual_overlay.yaml` is
disabled by default.  Enabling it requires an artifact directory and the
externally pinned metadata digest printed by the export command.  The runtime
rejects checkpoint, step, model, schema, site-layout, action-layout, and width
mismatches before creating a deployment session.
`load_sonic_action_residual_deployment` consumes this overlay: disabled mode
returns no plugin without importing ORT or touching an artifact, while enabled
mode requires caller-owned named release-file pins, checks their real SHA-256
and exact agreement with the YAML declarations, then validates every residual
pin before calling the supplied session factory.

The production C++ path uses the same manifest and graph through
`gear_sonic_deploy/src/motion_compliance`; that directory is tracker-neutral and
the G1-specific release hashes/layouts live only in the adjacent thin SONIC
adapter.  Pass the overlay explicitly with:

```text
--motion-compliance-overlay gear_sonic_deploy/policy/motion_compliance/action_residual_overlay.yaml
```

Changing the YAML switch to `enabled: true` opts into the residual.  With the
existing operator defaults it starts active; add `--set-compliance 0` to start
with an exact release-action bypass.  The first two values are left/right wrist
mode controls for one global residual, not independent per-wrist learned
conditions: both zero disables it and either positive enables it.  Values must
be finite and in `[0.0, 0.5]`.  Composite input managers preserve this setting
through interface switches and route `g/h/b/v` adjustments to the live
delegate.  Those shortcuts are limited to `manager`, `gamepad_manager`, and
`zmq_manager`; direct keyboard/gamepad modes can still select the startup state
with `--set-compliance`.

Use `/home/lab/miniconda3/envs/sonic/bin/python` for ONNX Runtime validation;
that environment contains torch 2.7, ONNX 1.21, and ONNX Runtime 1.25.  Earlier
IsaacLab/config regressions continue to use `sonic_backup`.  Neither interpreter
is modified by this workflow.
