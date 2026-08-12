"""Frame-exact, bounded camera evidence for one SONIC review panel."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import stat

import numpy as np

REVIEW_CAMERA_EYE_OFFSET_M = (3.2, -2.8, 1.8)
REVIEW_CAMERA_LOOKAT_HEIGHT_M = 0.9
REVIEW_PANEL_WIDTH = 960
REVIEW_PANEL_HEIGHT = 720
REVIEW_VIDEO_FPS = 50
DEFAULT_MAX_PANEL_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ReviewFrameMetadata:
    """Visible provenance and protocol state for one trace-aligned frame."""

    role: str
    branch_commit: str
    checkpoint_sha256: str
    motion_id: str
    seed: int
    frame_index: int
    timestamp_s: float
    active_site_names: tuple[str, ...]
    force_norms_n: tuple[float, ...]
    compliance_m_per_n: float

    def __post_init__(self) -> None:
        for field_name in ("role", "branch_commit", "checkpoint_sha256", "motion_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ValueError("timestamp_s must be finite and non-negative")
        if isinstance(self.active_site_names, (str, bytes)):
            raise TypeError("active_site_names must be a sequence")
        if len(self.force_norms_n) != 2 or any(
            not math.isfinite(value) or value < 0.0 for value in self.force_norms_n
        ):
            raise ValueError("force_norms_n must contain two finite non-negative values")
        if not math.isfinite(self.compliance_m_per_n) or self.compliance_m_per_n < 0.0:
            raise ValueError("compliance_m_per_n must be finite and non-negative")


def normalize_rgb_frame(
    frame: object,
    *,
    width: int = REVIEW_PANEL_WIDTH,
    height: int = REVIEW_PANEL_HEIGHT,
) -> np.ndarray:
    """Validate one uint8 RGB/RGBA camera frame and drop alpha explicitly."""

    array = np.asarray(frame)
    if array.dtype != np.uint8:
        raise TypeError("camera frame must use uint8")
    if array.shape not in ((height, width, 3), (height, width, 4)):
        raise ValueError(
            f"camera frame must have shape {(height, width, 3)} or {(height, width, 4)}"
        )
    return np.ascontiguousarray(array[..., :3]).copy()


def overlay_review_metadata(frame: np.ndarray, metadata: ReviewFrameMetadata) -> np.ndarray:
    """Draw compact, fixed-position provenance without changing dimensions."""

    if not isinstance(metadata, ReviewFrameMetadata):
        raise TypeError("metadata must be ReviewFrameMetadata")
    array = np.asarray(frame)
    if array.ndim != 3:
        raise ValueError("metadata overlay requires a three-dimensional image")
    result = normalize_rgb_frame(
        array,
        width=array.shape[1],
        height=array.shape[0],
    )
    import cv2

    active = ",".join(metadata.active_site_names) or "none"
    lines = (
        f"role={metadata.role}",
        (
            f"commit={metadata.branch_commit[:12]} "
            f"ckpt={metadata.checkpoint_sha256[:12]}"
        ),
        (
            f"motion={metadata.motion_id} seed={metadata.seed} "
            f"frame={metadata.frame_index} t={metadata.timestamp_s:.2f}s"
        ),
        (
            f"active={active} force(L/R)={metadata.force_norms_n[0]:.2f}/"
            f"{metadata.force_norms_n[1]:.2f}N C={metadata.compliance_m_per_n:.3f}m/N"
        ),
    )
    overlay = result.copy()
    cv2.rectangle(overlay, (8, 8), (result.shape[1] - 8, 112), (0, 0, 0), -1)
    result = cv2.addWeighted(overlay, 0.62, result, 0.38, 0.0)
    for index, line in enumerate(lines):
        cv2.putText(
            result,
            line,
            (18, 31 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return result


def capture_review_frame(
    raw_env: object,
    motion: object,
    *,
    width: int = REVIEW_PANEL_WIDTH,
    height: int = REVIEW_PANEL_HEIGHT,
) -> np.ndarray:
    """Capture the current pre-transition state using the pinned call order."""

    import torch

    sensors = raw_env.scene.sensors
    if "eval_camera" not in sensors:
        raise RuntimeError("review config did not create eval_camera")
    camera = raw_env.scene["eval_camera"]
    root_position = motion.robot_body_pos_w[:, 0]
    eye_offset = torch.tensor(
        REVIEW_CAMERA_EYE_OFFSET_M,
        dtype=root_position.dtype,
        device=root_position.device,
    )
    lookat_offset = torch.tensor(
        (0.0, 0.0, REVIEW_CAMERA_LOOKAT_HEIGHT_M),
        dtype=root_position.dtype,
        device=root_position.device,
    )
    camera.set_world_poses_from_view(root_position + eye_offset, root_position + lookat_offset)
    raw_env.sim.render()
    camera.update(dt=0.0)
    rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
    return normalize_rgb_frame(rgb, width=width, height=height)


def _open_imageio_writer(path: Path, *, fps: int):
    import imageio.v2 as imageio

    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=5,
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_log_level="error",
    )


class AtomicReviewVideoWriter:
    """Write one H.264 panel through a hidden same-directory partial file."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        width: int = REVIEW_PANEL_WIDTH,
        height: int = REVIEW_PANEL_HEIGHT,
        fps: int = REVIEW_VIDEO_FPS,
        max_bytes: int = DEFAULT_MAX_PANEL_BYTES,
    ) -> None:
        if any(type(value) is not int or value <= 0 for value in (width, height, fps)):
            raise ValueError("width, height, and fps must be positive integers")
        if width % 2 or height % 2:
            raise ValueError("H.264 yuv420p dimensions must be even")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.output_path = Path(output_path)
        if self.output_path.suffix.lower() != ".mp4":
            raise ValueError("review video output must end in .mp4")
        parent = self.output_path.parent
        if not parent.is_dir() or parent.is_symlink():
            raise NotADirectoryError("review video parent must be a real directory")
        self.partial_path = parent / f".{self.output_path.stem}.partial.mp4"
        if os.path.lexists(self.output_path) or os.path.lexists(self.partial_path):
            raise FileExistsError("review video output or partial path already exists")
        self.width = width
        self.height = height
        self.fps = fps
        self.max_bytes = max_bytes
        self._writer = _open_imageio_writer(self.partial_path, fps=fps)
        self._frame_count = 0
        self._closed = False

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def append(self, frame: object, metadata: ReviewFrameMetadata) -> None:
        if self._closed:
            raise RuntimeError("review video writer is closed")
        if metadata.frame_index != self._frame_count:
            raise AssertionError("video frame index must equal trace sample index")
        normalized = normalize_rgb_frame(frame, width=self.width, height=self.height)
        self._writer.append_data(overlay_review_metadata(normalized, metadata))
        self._frame_count += 1
        if self.partial_path.exists() and self.partial_path.stat().st_size > self.max_bytes:
            raise ValueError("partial review video exceeds max_bytes")

    def close(self) -> Path:
        if self._closed:
            raise RuntimeError("review video writer is already closed")
        try:
            self._writer.close()
            status = self.partial_path.stat()
            if not stat.S_ISREG(status.st_mode) or status.st_size <= 0:
                raise ValueError("video encoder did not produce a regular non-empty file")
            if status.st_size > self.max_bytes:
                raise ValueError("review video exceeds max_bytes")
            os.link(self.partial_path, self.output_path, follow_symlinks=False)
            directory_fd = os.open(self.output_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self.partial_path.unlink()
            self._closed = True
            return self.output_path
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        if self._closed:
            return
        try:
            self._writer.close()
        finally:
            self.partial_path.unlink(missing_ok=True)
            self._closed = True

    def __enter__(self) -> "AtomicReviewVideoWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False
