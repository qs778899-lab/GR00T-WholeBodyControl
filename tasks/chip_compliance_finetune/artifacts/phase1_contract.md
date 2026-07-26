# Phase 1 portable compliance contract

## Ordered sites and index spaces

`ComplianceTargetSpec.site_names` is the sole semantic site order. The SONIC boundary resolves that order twice: once against reference-motion bodies and once against articulation bodies. Each result carries an explicit `SiteIndexSpace`; integer indices must never cross consumers. No robot-specific names, counts, or index tables exist in the core.

## Common Cartesian frame

Targets and force-on-robot vectors must declare equal `CartesianFrameSpec` values. A frame consists of a restricted kind (`world`, `anchor_local`, or `heading_local`), an explicit semantic anchor where required, and its permitted rotation (`identity`, `full_3d`, or `yaw_only`). Phase 2 owns the actual transform and must validate a non-zero rotation numerically before calling hindsight math.

## Values, gating, and exposure

- Reference positions are finite floating tensors in `[batch, future, site, xyz]` order.
- Enabled force vectors are finite, use the reference dtype/device, and mean force applied to the robot in the common frame.
- Compliance is finite, non-negative inverse stiffness in `m/N`, isotropic or Cartesian-axis anisotropic.
- Global `enabled=False` returns a non-aliased exact reference and does not touch optional force/compliance operands. Mixed gates require finite operands; disabled elements receive exact reference values and zero force/compliance gradients.
- Metric exposure means `enabled AND requested_site_mask AND any(compliance_xyz > 0)`, not merely a requested site bit.

## Target damping

`TargetDamper` implements `g_t = alpha * x_eef + (1 - alpha) * g_prev`. It requires explicit initialization, supports full and per-environment reset, returns a target differentiable with respect to the current EEF position, and detaches stored state between control steps. Phase 2 owns lifecycle/reset calls; Phase 3 connects only this compliance goal path to the adapter while retaining the released dense tracking path.
