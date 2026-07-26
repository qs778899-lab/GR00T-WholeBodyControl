# Phase 3 policy/checkpoint contract

## Data boundary

The actor receives only four explicitly allowlisted groups:
`actor_obs`, the unchanged release `tokenizer`, `compliance_target`, and
`compliance_command`. `compliance_target` is the Phase-2 configurable-site,
future-aligned hindsight/damped target. `compliance_command` contains the hard
enable bit, ordered site mask, and per-site Cartesian compliance. The actor
filters both direct forwards and rollout history through this allowlist.

Applied `compliance_force` is deliberately absent from the actor. It is an
isolated privileged group used only by the critic residual. Its rejection is a
hard-coded actor invariant, so neither Hydra configuration nor mutation of the
internal actor allowlist can admit it through direct or rollout paths. The
release dense policy/critic observations, tokenizer, tracking rewards, and
termination path are unchanged and byte-audited against the pinned baseline.

## Model boundary

The release G1/teleop/SMPL encoders and FSQ first produce the normal two-token,
32D-per-token latent. A tracker-agnostic `ComplianceResidualMLP` generates one
hard-gated 64D correction from the public target, public command, and actor
proprioception. The correction is added only after quantization and before the
unchanged G1 dynamic decoder.

The residual output head is exactly zero initialized. Therefore default-off,
zero-compliance, and initial enabled inference preserve the released output.
On the first enabled backward pass the output head has a nonzero gradient; the
zero head intentionally blocks trunk gradients until a later optimizer step.

The critic retains its complete released value path and adds a separate scalar
residual conditioned on target, command, critic context, and privileged applied
force. Site count and order come only from configuration; neither residual
contains a fixed wrist/two-site or 14-body index table. Resolved construction is
covered for 1, 2, 5, 14, and 17 sites and independent future horizons. Local
site offsets default to `null`, so a caller either uses all-zero offsets or
provides exactly one offset per configured site.

The critic running-statistics module is invoked exactly once. Its resulting
frozen normalized `critic_obs` tensor is the same object supplied to the release
value path and the residual context, and a zero residual therefore preserves
the release value bytes.

## Finetune ownership

Phase 3 freezes all release encoders, the FSQ quantizer, both G1 decoders, actor
noise, the base critic, and critic running statistics. Only the six actor
residual parameters and six critic residual parameters are trainable. Phase 4
may change this ownership only through an explicit, reviewed config change.

Residual module construction runs in a forked RNG scope. Relative to the
release actor and critic constructors it consumes no additional process-global
CPU or CUDA random state, and the following stochastic sequence remains exact.
The official checkpoint uses a direct action std with four values outside the
runtime clamp. The opt-in actor computes the same effective clamp out of place
when that std is frozen, so repeated distribution construction and optimizer
steps cannot mutate the checkpoint tensor; log-std and trainable direct-std
behavior remain inherited from the release Actor.

## Checkpoint boundary

The pinned release checkpoint is accepted only as one complete legacy schema:
55 policy keys and 17 value keys. Missing, unexpected, shape-mismatched, or
dtype-mismatched legacy tensors are rejected before mutation. Migration retains
every old tensor byte-for-byte and initializes exactly the six residual keys in
each model. A checkpoint containing residual keys is treated as a branch
checkpoint and always loaded strictly, even if a caller requests non-strict
loading.

The release `sonic_release.yaml` and checkpoint artifacts are not modified.

## Resolved validation

The final portable suite passed 79 tests with 26 expected dependency skips; the
resolved CPU/Hydra/official-checkpoint suite passed 79 tests with four expected
CUDA skips. The focused resolved integration suite passed all 12 tests.

The final one-environment Isaac Lab smoke passed in 16.48 seconds. It resolved
actor/critic/tokenizer/target/command/force widths to
`930/1645/1761/60/9/6`, produced action/value shapes
`(1,1,29)/(1,1,1)`, preserved default-off action bytes, and preserved the
official frozen std bytes through three real distribution constructions.
