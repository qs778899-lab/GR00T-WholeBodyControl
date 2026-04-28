#!/usr/bin/env python3
"""Inspect SONIC motion_lib PKL preprocessing and encoder-input tensors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from tools.sonic_eval.motionlib_provider import load_motionlib_sequence


def _summary(name: str, arr) -> None:
    x = arr.detach().cpu().numpy()
    print(
        f"{name}: shape={tuple(x.shape)} dtype={x.dtype} "
        f"min={float(np.nanmin(x)):.6f} max={float(np.nanmax(x)):.6f} "
        f"mean={float(np.nanmean(x)):.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--motion-name", type=str, default=None)
    parser.add_argument("--target-fps", type=int, default=50)
    parser.add_argument("--num-future-frames", type=int, default=10)
    parser.add_argument("--dt-future-ref-frames", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-motionlib-robot", action="store_true")
    parser.add_argument(
        "--use-isaacsim-app",
        action="store_true",
        help="start IsaacSim SimulationApp before official TrackingCommand preprocessing",
    )
    parser.add_argument("--max-print-frames", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seq = load_motionlib_sequence(
        motion_file=args.motion_file,
        motion_name=args.motion_name,
        target_fps=args.target_fps,
        num_future_frames=args.num_future_frames,
        dt_future_ref_frames=args.dt_future_ref_frames,
        device=args.device,
        prefer_motionlib_robot=not args.no_motionlib_robot,
        use_isaacsim_app=args.use_isaacsim_app,
    )
    print(f"motion_name: {seq.motion_name}")
    print(f"source: {seq.source}")
    print(f"original_fps: {seq.original_fps}")
    print(f"target_fps: {seq.fps}")
    print(f"num_frames: {seq.num_frames}")
    _summary("dof_pos_il", seq.dof_pos)
    _summary("dof_vel_il", seq.dof_vel)
    _summary("root_pos_w", seq.root_pos_w)
    _summary("root_quat_w", seq.root_quat_w)
    _summary("body_pos_w", seq.body_pos_w)
    _summary("body_quat_w", seq.body_quat_w)
    _summary("command_multi_future_nonflat", seq.command_multi_future_nonflat)
    _summary("motion_anchor_ori_b_mf_nonflat", seq.motion_anchor_ori_b_mf_nonflat)

    n = min(args.max_print_frames, seq.num_frames)
    if n > 0:
        print("first dof_pos_il rows:")
        print(seq.dof_pos[:n].detach().cpu().numpy())
        print("first command_multi_future_nonflat[0] window:")
        print(seq.command_multi_future_nonflat[0].detach().cpu().numpy())


if __name__ == "__main__":
    main()
