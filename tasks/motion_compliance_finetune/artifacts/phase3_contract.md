# Phase-3 observation/reward/config contract

## Encoder and actor boundary

The robot-motion tokenizer remains byte-for-byte configuration-equivalent to
`sonic_release`.  Its G1 inputs remain, in order,
`command_multi_future_nonflat [batch,10,58]` and
`motion_anchor_ori_b_mf_nonflat [batch,10,6]`.  Compliance is not a tokenizer
term.  The released 930 policy-proprioception columns remain a prefix and the
actor receives only the appended public condition
`[enable, enable*threshold_N, enable*Kp_N_per_m]`, producing width 933.

## Asymmetric critic state

The critic sees the same public condition plus raw threshold, the applied site
force from future frame zero in current-anchor common coordinates, and the
active-site mask.  For `S` configured sites the addition is `3 + 1 + 3*S + S`;
no production code fixes `S=2`.  The default two-wrist composition therefore
grows 1645 to 1657.  Force and mask never enter the actor group.

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

`sonic_release_motion_compliance` inherits `sonic_release` and overrides only
the command, event, policy observation, critic observation, and reward groups.
It explicitly defaults `motion_compliance.enabled=false`; no release YAML is
modified.  The resolved termination subtree remains exactly equal to release.
The one-environment Phase-3 smoke resolves actual policy/critic shapes,
tokenizer tensor shapes, an independently reconstructed tracking future-zero
reference, hard-off rewards, bitwise original target selection, and exact
equality between the manager total and released shared reward contributions
using the audited official robot/SMPL sample.  It also poisons the real
command's prior cache before an enabled reward call and proves that production
recomputes current state without mutating cache or activating a wrench.
