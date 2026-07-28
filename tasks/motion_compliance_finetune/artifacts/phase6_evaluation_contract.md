# Phase 6 aligned evaluation contract

This layer evaluates standardized traces after a simulator- or tracker-specific
collector has translated its output.  Portable code under
`gear_sonic/compliance_control/evaluation/` owns no body names, action order,
keypoint count, simulator import, or concrete endpoint role.  A collector must
supply ordered `site_ids` and `point_ids`; any robot-specific mapping stays at
that adapter boundary.

## Alignment

Every paired row must match exactly on ordered motion ID, sequence ID, seed,
frame index, and floating timestamp.  Ordered site and tracking-point layouts
must also match exactly. Reference global/local point arrays and original site
positions/orientations match on dtype, shape, and bytes, preventing a changed
reference from being hidden as a candidate result. Frame indices and timestamps
increase strictly within each `(motion, sequence, seed)` stream. A one-bit
timestamp difference rejects the pair instead of being silently resampled.

## Standard trace fields

- Original, selected, and measured site positions are `[rows, sites, 3]` in
  metres in one adapter-selected Cartesian frame.
- Original/measured site orientations are `[rows, sites, 4]` quaternions in
  `xyzw` order.
- Reference/measured global and local tracking points are
  `[rows, points, 3]`; the point count is caller owned.
- Force on the robot is `[rows, sites, 3]` in newtons.  Enabled and active-site
  masks are separate so enabled/no-contact is distinguishable from hard off.
- Terminal, success, fall, and post-reset snapshot masks are explicit.  Force
  on reset rows measures stale-wrench persistence.  Every
  `(motion, sequence, seed)` stream has exactly one reset on its first row and
  exactly one terminal on its last row; a fall is terminal-only.

The report includes per-site original/selected endpoint RMSE and P95,
orientation error, force peak/RMS, reference yield, candidate measured yield
against overlay-off, measured-yield projection along actual force,
active-contact-window endpoint/orientation/force/yield summaries,
inactive-site force/yield, inactive-hand RMSE/P95 position shift against both the
released baseline and overlay-off trace, local/global MPJPE, paired pose shift,
success/fall/reset counts, and input/derived finiteness.  Endpoint roles used
for the stiff-mode threshold are supplied on the CLI; they are not embedded in
the portable package.  Every declared interaction site must exceed
caller-selected active-force and active-yield minima; every inactive site is
independently bounded for residual force and unintended reference yield. Every
formal trial also requires zero falls and success rate exactly one.

## Persistence and runner

`write_trace_npz_atomic` and `write_report_json_atomic` publish only complete
files, refuse overwrite by default, reject oversized compressed and
uncompressed traces, and disable NumPy pickle loading.  Loading uses one
`O_NOFOLLOW` regular-file descriptor from ZIP/member/schema checks through
NumPy decode, and rejects duplicate archive members or non-Unicode/rank-invalid
name arrays.  The thin runner is:

```bash
python -B tasks/motion_compliance_finetune/artifacts/phase6_evaluate_aligned_traces.py \
  --trial released baseline /path/released.npz - \
  --trial overlay_off off /path/off.npz - \
  --trial no_contact no_contact /path/no_contact.npz - \
  --trial single_a single_site /path/single_a.npz endpoint_a \
  --trial single_b single_site /path/single_b.npz endpoint_b \
  --trial simultaneous multi_site /path/simultaneous.npz endpoint_a,endpoint_b \
  --baseline released \
  --endpoint-site endpoint_a --endpoint-site endpoint_b \
  --output /path/evaluation.json
```

The runner uses `load_trace_npz_with_sha256`, so the report records the full
file hash from the same verified descriptor used for decoding. The separate
SONIC validator reloads the six trace files, binds those hashes to collection
summaries, and recomputes the complete report under the fixed formal criteria;
it does not trust a self-declared acceptance flag or check list.

The implemented SONIC collector is
`phase6_collect_sonic_trace.py`; concrete body/checkpoint/motion/IsaacLab
semantics stay there and in `phase6_validate_sonic_collection_reports.py`, not
in the portable package. At the 2026-07-28 pause boundary, their CPU contract
passes but no formal simulator trace exists. Before collection, exact
termination/event function targets and parameters plus a configured reset event
after nonzero force still need Phase-6-specific provenance coverage.

The CPU contract does not manufacture performance evidence.  Real Phase 6
acceptance still requires strictly paired simulator traces plus the prescribed
GPU smoke and performance measurements.
