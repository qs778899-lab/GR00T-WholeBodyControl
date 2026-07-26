# Phase-3 observation/reward/config contract

## Release backbone and actor boundary

The robot-motion tokenizer remains byte-for-byte configuration-equivalent to
`sonic_release`.  Its G1 inputs remain, in order,
`command_multi_future_nonflat [batch,10,58]` and
`motion_anchor_ori_b_mf_nonflat [batch,10,6]`.  Compliance is not a tokenizer
term.  The released policy group remains exactly 930 columns, and the released
`g1_dyn` continues to consume `[token64 | proprioception930] = 994` columns.

The actor-visible
`[enable, enable*threshold_N, enable*Kp_N_per_m]` is a separate 3D observation
group.  An independent zero-initialized residual head reads
`[token64 | actor_obs930 | condition3]`; it never changes a release decoder
input.  The final action is selected per row as `base` or `base + bounded_delta`
with a boolean hard gate.  Disabled residual inputs are cleared before the MLP,
so NaN in a rejected condition cannot poison shared mixed-batch gradients.
Release encoders, quantizer, every decoder, and action-noise state are frozen.
The actor boundary forwards only `actor_obs`, `tokenizer`, and the public
condition; critic and privileged groups cannot enter direct or temporal policy
history.

## Asymmetric critic state

The released critic group, first-layer input, running mean, and running variance
remain exactly 1645 columns.  Its frozen release value is composed with an
independent residual head.  That head reads the separate public condition plus
the separate privileged scalar threshold, applied site force from future frame
zero in current-anchor common coordinates, and active-site mask.  For `S`
configured sites the privileged group width is `1 + 3*S + S = 1+4*S`; no
production code fixes `S=2`.  The default two-wrist composition is therefore
condition width 3 and privileged width 9.  Force and mask never enter the actor.
The critic residual uses the same pre-MLP disabled-row sanitization and final
hard gate as the actor.

The earlier 933/1657 expanded-input design is invalidated: it trained release
`g1_dyn`/critic parameters and could not make hard-off structurally identical
to the official model.  Its preserved run artifacts are diagnostic evidence,
not an initialization or resume source for this architecture.

## Reference and reward semantics

Position selection recomputes current-anchor endpoint/reference tensors locally
when IsaacLab evaluates the reward, then uses future index zero.  This avoids a
one-physics-step lag because reward computation precedes the next command
update; reward evaluation does not mutate command-owned force/reference caches.
For sampled enabled environments, active sites select the yielded candidate
while inactive sites are copied from the original reference via `torch.where`.
Orientation always tracks the original future-zero quaternion; rotational
compliance is outside this phase.

Both new rewards use a boolean `torch.where` hard gate after masking disabled
errors, rather than arithmetic multiplication.  Thus a host-off or sampled-off
environment contributes exactly zero even if a disabled error is non-finite,
and the released total reward is unchanged.  Existing dense full-body and
endpoint rewards remain present and unchanged, including the release
experiment's `feet_acc=-2.5e-6` override.
Enabled position/orientation terms use conservative `2.0/0.1` and `0.5/0.4`
weight/std pairs.

Per configured site, command metrics retain selected-target position error,
original-target position error, and original-target orientation error.  This
keeps correct yielding distinguishable from tracking degradation and prevents
one hand from being hidden by the cross-site reward mean.

## Opt-in and real resolved shape

`sonic_release_motion_compliance` inherits `sonic_release`, retains its policy
and critic observation groups, and adds separate condition/privileged groups
while overriding only opt-in command, event, reward, and model-wrapper targets.
It explicitly defaults `motion_compliance.enabled=false`; no release YAML is
modified.  The resolved termination subtree remains exactly equal to release.
Its resolved interval-event names and ranges also exactly match release: wrench
application is part of command update, and the only additional event is reset
cleanup.  Disabled command reset/compute uses no global CPU or CUDA RNG.
The CPU official audit pins base shapes `2048x994` and `2048x1645`, RMS width
1645, and proves residual keys are target-initialized rather than migrated
columns.  The one-environment Phase-3 smoke resolves actual policy/critic shapes
`[1,930]`/`[1,1645]`, separate condition/privileged shapes `[1,3]`/`[1,9]`,
and unchanged tokenizer tensors.  It instantiates the resolved actor/value,
loads every official tensor byte-exact with only residual keys missing, and
exercises zero-init parity, mixed `[off,on,off]`, privileged poison, aux,
external-token, frozen-noise, and residual-gradient paths.  The same smoke
independently reconstructs tracking future-zero reference, checks hard-off
rewards and exact released shared reward totals, then poisons the command's
prior cache to prove reward-time recomputation does not mutate caches or apply
a wrench.
