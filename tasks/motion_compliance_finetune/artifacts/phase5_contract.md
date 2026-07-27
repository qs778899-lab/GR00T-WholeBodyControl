# Phase-5 export/deployment contract audit

## Release boundary

- `control_policy.hpp` loads one decoder input named `obs_dict` and one output
  named `action`; it is the released-action owner.
- `encoder.hpp` loads one encoder input named `obs_dict` and one output named
  `encoded_tokens`; it is the 64-D token owner.
- The released decoder emits the 29 actions in IsaacLab/BFS order.  The existing
  host indexes that decoder output with `isaaclab_to_mujoco` before applying
  MuJoCo-ordered scales and motor targets.  The new overlay therefore pins the full
  `joint_utils.G1_ISAACLab_ORDER`, and the residual is added to the release
  action before this existing remap.  It also pins the two configured
  compliance sites.
- The release observation YAML, release encoder, release decoder, and their
  inference-owner headers are not edited or copied.  The production executable
  receives one reviewed additive CLI/load/compose hook; it does not change the
  release model or observation math.  The optional residual is a second model
  whose only composition rule is `release_action + bounded_delta` on enabled
  rows.

## True trained input

`MotionComplianceUniversalTokenModule._apply_action_residual` constructs the
trained MLP input as `token_flattened + actor_obs + condition`, with widths
`64 + 930 + 3 = 997`.  Consequently the export graph accepts a 994-D release
context (`token + actor_obs`) and a separate 3-D condition, then concatenates
them once.  An interface accepting 997 plus another 3 columns would not match
the trained first-layer weight shape `[256,997]` and is explicitly rejected.
In the exported `[B,S,*]` tensors, `S` is the dynamic rollout/control-sequence
axis; it is not the physical compliance-site count.  Physical sites remain a
separate metadata layout, matching the training architecture's global 3-D
public condition and critic-only per-site state.

## Porting boundary

`gear_sonic.compliance_control.deployment` and
`gear_sonic_deploy/src/motion_compliance` are the reusable Python and C++
layers.  Their widths,
site identifiers, action identifiers, context segments, hidden sizes, and
delta limit are artifact metadata rather than robot constants.  They import no
IsaacLab module; the Python layer lazily imports ONNX Runtime only when an
enabled host asks for a session, while the C++ layer hides ORT behind a PImpl.
The C++ host contract accepts arbitrary ordered context fields and arbitrary
named/hash-pinned release artifacts rather than SONIC's fixed three files.
The Python host expectation uses the same arbitrary non-empty
`ReleaseArtifactPin(name, path, sha256)` sequence and hashes every real base
file before loading the residual session.  The thin SONIC loader additionally
requires its YAML declarations to match those caller-owned pins exactly.
Artifact schema v1 pins ONNX opset 17 identically in Python and C++ so a bundle
accepted during export cannot be rejected only after migration to production.
The C++ target explicitly cancels the repository-wide `-ffast-math` flag so
finite-value validation remains a real production boundary; this does not
change the released policy target's arithmetic flags.
Another universal tracker supplies:

1. its unchanged release action;
2. the ordered release-policy context segments declared by metadata;
3. the public condition and a separately validated host gate;
4. an ONNX-compatible session implementing the small `run` protocol.

The thin Python and C++ SONIC adapters own the concrete token/observation
ordering, wrist-site names, decoder-output (IsaacLab/BFS) action order,
threshold range, and reference displacement.  The production adapter maps the first two wrist
controls to the current model's one global hard gate; both zero is off and
either positive is on.  Composite input managers synchronize the initial value
and their `g/h/b/v` keyboard changes into the delegate read by the control
loop.  Direct keyboard/gamepad modes support the startup CLI value but do not
claim these manager-only shortcuts.  Disabled
hosts and all-off batches return a byte copy of the release action without
calling the optional session.  In mixed batches, rejected rows are sanitized
before inference and selected back to the original release bytes afterward.
Its overlay loader returns `None` without loading ONNX Runtime or reading an
artifact when `enabled: false`; when enabled, it validates all external pins
before invoking the session factory.
