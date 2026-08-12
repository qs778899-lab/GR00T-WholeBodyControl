"""Bounded, trace-bound metadata validation for human-review videos."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any

from ._hashing import sha256_stream
from .alignment import assert_strict_alignment
from .io import (
    load_trace_npz_with_sha256,
    write_report_json_atomic,
)

_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_VIDEO_BYTES = 512 * 1024 * 1024


def _nonempty(label: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha256_string(label: str, value: object) -> str:
    result = _nonempty(label, value)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return result


def _positive_int(label: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ReviewPanelSpec:
    """Caller-owned trace and provenance for one video panel."""

    role: str
    trace_path: Path
    summary_path: Path
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _nonempty("role", self.role))
        object.__setattr__(self, "trace_path", Path(self.trace_path))
        object.__setattr__(self, "summary_path", Path(self.summary_path))
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _sha256_string("checkpoint_sha256", self.checkpoint_sha256),
        )


@dataclass(frozen=True, slots=True)
class ReviewVideoSpec:
    """Expected identity, layout, and encoding of one paired review video."""

    comparison_name: str
    motion_id: str
    seed: int
    branch_commit: str
    left: ReviewPanelSpec
    right: ReviewPanelSpec
    metrics_path: Path
    video_path: Path
    width: int
    height: int
    fps: int = 50

    def __post_init__(self) -> None:
        for field_name in ("comparison_name", "motion_id", "branch_commit"):
            object.__setattr__(
                self,
                field_name,
                _nonempty(field_name, getattr(self, field_name)),
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.left, ReviewPanelSpec) or not isinstance(
            self.right,
            ReviewPanelSpec,
        ):
            raise TypeError("left and right must be ReviewPanelSpec values")
        if self.left.role == self.right.role:
            raise ValueError("left and right panel roles must be distinct")
        object.__setattr__(self, "metrics_path", Path(self.metrics_path))
        object.__setattr__(self, "video_path", Path(self.video_path))
        for field_name in ("width", "height", "fps"):
            value = _positive_int(field_name, getattr(self, field_name))
            if field_name != "fps" and value % 2:
                raise ValueError(f"{field_name} must be even for yuv420p")


def _open_regular(path: Path, *, max_bytes: int):
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"path must be a regular non-symlink file: {path}") from exc
    status = os.fstat(descriptor)
    if not stat.S_ISREG(status.st_mode):
        os.close(descriptor)
        raise ValueError(f"path must be a regular non-symlink file: {path}")
    if status.st_size > max_bytes:
        os.close(descriptor)
        raise ValueError(f"file exceeds byte cap: {path}")
    return os.fdopen(descriptor, "rb")


def _load_json_with_sha256(
    path: Path,
    *,
    max_bytes: int = _MAX_JSON_BYTES,
) -> tuple[Any, str]:
    with _open_regular(path, max_bytes=max_bytes) as stream:
        digest = sha256_stream(stream)
        stream.seek(0)
        try:
            payload = json.load(stream)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON file: {path}") from exc
        if stream.read(1):
            raise ValueError(f"JSON decoder did not consume the complete file: {path}")
    return payload, digest


def _fraction(value: object, *, label: str) -> Fraction:
    text = _nonempty(label, value)
    try:
        result = Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{label} is not a finite rational") from exc
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _parse_probe_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("ffprobe output must be a JSON object")
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise ValueError("video must contain exactly one stream")
    stream = streams[0]
    if stream.get("codec_type") != "video":
        raise ValueError("the sole stream must be video")
    for field_name in ("width", "height"):
        _positive_int(field_name, stream.get(field_name))
    frames_text = _nonempty("nb_frames", stream.get("nb_frames"))
    if not frames_text.isdigit() or int(frames_text) <= 0:
        raise ValueError("nb_frames must be a positive decimal integer")
    duration_text = _nonempty("duration", stream.get("duration"))
    try:
        duration = float(duration_text)
    except ValueError as exc:
        raise ValueError("duration must be numeric") from exc
    if not duration > 0.0:
        raise ValueError("duration must be positive")
    return {
        "codec_name": _nonempty("codec_name", stream.get("codec_name")),
        "pixel_format": _nonempty("pix_fmt", stream.get("pix_fmt")),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": str(_fraction(stream.get("avg_frame_rate"), label="avg_frame_rate")),
        "frame_count": int(frames_text),
        "duration_s": duration,
    }


def probe_video_with_sha256(
    path: str | Path,
    *,
    ffprobe: str | Path = "ffprobe",
    max_bytes: int = _MAX_VIDEO_BYTES,
) -> tuple[dict[str, Any], str]:
    """Hash and probe one video through the same verified open descriptor."""

    source = Path(path)
    executable = str(ffprobe)
    with _open_regular(source, max_bytes=max_bytes) as stream:
        digest = sha256_stream(stream)
        stream.seek(0)
        descriptor_path = f"/proc/self/fd/{stream.fileno()}"
        try:
            completed = subprocess.run(
                (
                    executable,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    (
                        "stream=codec_type,codec_name,pix_fmt,width,height,"
                        "avg_frame_rate,nb_read_frames,duration"
                    ),
                    "-of",
                    "json",
                    descriptor_path,
                ),
                check=True,
                capture_output=True,
                text=True,
                pass_fds=(stream.fileno(),),
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(f"ffprobe failed for {source}") from exc
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("ffprobe did not return JSON") from exc
    streams = raw.get("streams") if isinstance(raw, dict) else None
    if isinstance(streams, list) and len(streams) == 1 and isinstance(streams[0], dict):
        streams[0]["nb_frames"] = streams[0].pop("nb_read_frames", None)
    return _parse_probe_payload(raw), digest


def build_review_video_manifest(
    spec: ReviewVideoSpec,
    *,
    ffprobe: str | Path = "ffprobe",
) -> dict[str, Any]:
    """Validate one paired video and return its complete immutable bindings."""

    if not isinstance(spec, ReviewVideoSpec):
        raise TypeError("spec must be a ReviewVideoSpec")
    left_trace, left_trace_sha = load_trace_npz_with_sha256(spec.left.trace_path)
    right_trace, right_trace_sha = load_trace_npz_with_sha256(spec.right.trace_path)
    assert_strict_alignment(left_trace, right_trace)
    if left_trace.trial_name != spec.left.role or right_trace.trial_name != spec.right.role:
        raise ValueError("panel role does not match trace trial_name")
    if set(left_trace.motion_ids) != {spec.motion_id}:
        raise ValueError("left trace motion does not match the video spec")
    if set(right_trace.motion_ids) != {spec.motion_id}:
        raise ValueError("right trace motion does not match the video spec")
    if set(left_trace.seed_ids.tolist()) != {spec.seed}:
        raise ValueError("left trace seed does not match the video spec")
    if set(right_trace.seed_ids.tolist()) != {spec.seed}:
        raise ValueError("right trace seed does not match the video spec")
    left_summary, left_summary_sha = _load_json_with_sha256(spec.left.summary_path)
    right_summary, right_summary_sha = _load_json_with_sha256(spec.right.summary_path)
    metrics, metrics_sha = _load_json_with_sha256(spec.metrics_path)
    for label, payload in (
        ("left summary", left_summary),
        ("right summary", right_summary),
        ("metrics", metrics),
    ):
        if not isinstance(payload, dict):
            raise ValueError(f"{label} must be a JSON object")
    probe, video_sha = probe_video_with_sha256(spec.video_path, ffprobe=ffprobe)
    expected_frames = len(left_trace.motion_ids)
    expected_duration = expected_frames / spec.fps
    if probe["codec_name"] != "h264":
        raise ValueError("review video codec must be h264")
    if probe["pixel_format"] != "yuv420p":
        raise ValueError("review video pixel format must be yuv420p")
    if (probe["width"], probe["height"]) != (spec.width, spec.height):
        raise ValueError("review video dimensions do not match the spec")
    if Fraction(probe["frame_rate"]) != Fraction(spec.fps, 1):
        raise ValueError("review video frame rate does not match the policy clock")
    if probe["frame_count"] != expected_frames:
        raise ValueError("review video frame count does not match the trace")
    if abs(probe["duration_s"] - expected_duration) > (0.5 / spec.fps):
        raise ValueError("review video duration does not match the trace")

    def panel_payload(
        panel: ReviewPanelSpec,
        trace_sha: str,
        summary_sha: str,
    ) -> dict[str, Any]:
        return {
            "role": panel.role,
            "checkpoint_sha256": panel.checkpoint_sha256,
            "trace": str(panel.trace_path.resolve()),
            "trace_sha256": trace_sha,
            "summary": str(panel.summary_path.resolve()),
            "summary_sha256": summary_sha,
        }

    return {
        "schema_version": "compliance_review_video_v1",
        "comparison_name": spec.comparison_name,
        "motion_id": spec.motion_id,
        "seed": spec.seed,
        "branch_commit": spec.branch_commit,
        "panel_order": [spec.left.role, spec.right.role],
        "panels": [
            panel_payload(spec.left, left_trace_sha, left_summary_sha),
            panel_payload(spec.right, right_trace_sha, right_summary_sha),
        ],
        "metrics": str(spec.metrics_path.resolve()),
        "metrics_sha256": metrics_sha,
        "video": str(spec.video_path.resolve()),
        "video_sha256": video_sha,
        "video_probe": probe,
        "trace_frame_count": expected_frames,
        "policy_fps": spec.fps,
        "frame_contract": "video frame k equals aligned trace sample k",
    }


def write_review_video_manifest_atomic(
    spec: ReviewVideoSpec,
    output_path: str | Path,
    *,
    ffprobe: str | Path = "ffprobe",
    max_bytes: int = _MAX_JSON_BYTES,
) -> Path:
    """Validate live artifacts, then publish one immutable manifest."""

    manifest = build_review_video_manifest(spec, ffprobe=ffprobe)
    return write_report_json_atomic(
        manifest,
        output_path,
        max_bytes=max_bytes,
        overwrite=False,
    )


def validate_review_video_manifest(
    manifest_path: str | Path,
    spec: ReviewVideoSpec,
    *,
    ffprobe: str | Path = "ffprobe",
    max_bytes: int = _MAX_JSON_BYTES,
) -> dict[str, Any]:
    """Rebind a manifest to live artifacts and reject any changed field/file."""

    recorded, _ = _load_json_with_sha256(Path(manifest_path), max_bytes=max_bytes)
    if not isinstance(recorded, dict):
        raise ValueError("review manifest must be a JSON object")
    expected = build_review_video_manifest(spec, ffprobe=ffprobe)
    if recorded != expected:
        raise ValueError("review manifest contains extra, missing, or rebound artifacts")
    return recorded
