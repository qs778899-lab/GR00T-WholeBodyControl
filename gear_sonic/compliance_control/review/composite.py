"""Portable, bounded side-by-side H.264 composition for review panels."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess


def _open_regular_input(path: Path, *, max_bytes: int) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"panel must be a regular non-symlink file: {path}") from exc
    status = os.fstat(descriptor)
    if not stat.S_ISREG(status.st_mode) or status.st_size <= 0:
        os.close(descriptor)
        raise ValueError(f"panel must be a non-empty regular file: {path}")
    if status.st_size > max_bytes:
        os.close(descriptor)
        raise ValueError(f"panel exceeds max_bytes: {path}")
    return descriptor


def compose_review_panels(
    left_panel: str | Path,
    right_panel: str | Path,
    output_path: str | Path,
    *,
    ffmpeg: str | Path = "ffmpeg",
    fps: int = 50,
    max_input_bytes: int = 256 * 1024 * 1024,
    max_output_bytes: int = 512 * 1024 * 1024,
) -> Path:
    """H-stack two aligned panels through an atomic same-directory partial."""

    if type(fps) is not int or fps <= 0:
        raise ValueError("fps must be a positive integer")
    for value in (max_input_bytes, max_output_bytes):
        if type(value) is not int or value <= 0:
            raise ValueError("byte caps must be positive integers")
    left = Path(left_panel)
    right = Path(right_panel)
    output = Path(output_path)
    if output.suffix.lower() != ".mp4":
        raise ValueError("composite output must end in .mp4")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise NotADirectoryError("composite output parent must be a real directory")
    partial = output.parent / f".{output.stem}.partial.mp4"
    if os.path.lexists(output) or os.path.lexists(partial):
        raise FileExistsError("composite output or partial already exists")
    left_descriptor = _open_regular_input(left, max_bytes=max_input_bytes)
    try:
        right_descriptor = _open_regular_input(right, max_bytes=max_input_bytes)
    except BaseException:
        os.close(left_descriptor)
        raise
    try:
        subprocess.run(
            (
                str(ffmpeg),
                "-v",
                "error",
                "-nostdin",
                "-i",
                f"/proc/self/fd/{left_descriptor}",
                "-i",
                f"/proc/self/fd/{right_descriptor}",
                "-filter_complex",
                "[0:v][1:v]hstack=inputs=2[v]",
                "-map",
                "[v]",
                "-an",
                "-r",
                str(fps),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-map_metadata",
                "-1",
                str(partial),
            ),
            check=True,
            pass_fds=(left_descriptor, right_descriptor),
        )
        status = partial.stat()
        if not stat.S_ISREG(status.st_mode) or status.st_size <= 0:
            raise ValueError("ffmpeg did not produce a non-empty regular video")
        if status.st_size > max_output_bytes:
            raise ValueError("composite review video exceeds max_output_bytes")
        os.link(partial, output, follow_symlinks=False)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        partial.unlink()
        return output
    finally:
        os.close(left_descriptor)
        os.close(right_descriptor)
        partial.unlink(missing_ok=True)
