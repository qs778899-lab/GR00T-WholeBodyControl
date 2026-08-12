"""Command-line inspection for portable review artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .video import probe_video_with_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a bounded human-review video without loading a simulator.",
    )
    subparsers = parser.add_subparsers(dest="command")
    probe = subparsers.add_parser("probe", help="hash and ffprobe one video")
    probe.add_argument("video", type=Path)
    probe.add_argument("--ffprobe", default="ffprobe")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "probe":
        probe, digest = probe_video_with_sha256(args.video, ffprobe=args.ffprobe)
        print(
            json.dumps(
                {"probe": probe, "video_sha256": digest},
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0
    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
