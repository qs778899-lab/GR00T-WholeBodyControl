# Official asset manifest

Source repository: `nvidia/GEAR-SONIC` on Hugging Face, audited revision
`7c90a56cfe04788c4f041daeef5b1e12930675ad`.

| Path | SHA-256 |
|---|---|
| `sonic_release/last.pt` | `e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909` |
| `sonic_release/config.yaml` | `f08187795fa16a839a28bc1c18e0555d38d9420e03733744341cdcb56ab629c7` |

Sample data contains six PKL files: original and mirrored walking sequences for
the robot, SMPL and SOMA representations. Large binary assets in this directory
are local test inputs and must not be staged or committed.

`last.pt` is 469,418,283 bytes and reports training step 41,550. After importing
the repository's TRL 0.28 compatibility shim it loads on CPU with 55 policy
state entries and 25,870,714 policy parameters. A direct bare `torch.load`
without that shim is not a valid integrity check because the checkpoint refers
to the legacy `trl.trainer.utils.OnlineTrainerState` pickle path.
