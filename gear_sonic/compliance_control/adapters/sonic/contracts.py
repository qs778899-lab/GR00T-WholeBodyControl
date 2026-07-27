"""Pinned SONIC release contracts kept outside the tracker-neutral core."""

from __future__ import annotations

from collections.abc import Sequence


SONIC_RELEASE_TRACKING_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


def require_sonic_release_tracking_body_names(
    body_names: Sequence[str],
) -> tuple[str, ...]:
    """Return the ordered names or reject a shortened/reordered release contract."""

    if isinstance(body_names, (str, bytes)):
        raise TypeError("body_names must be a sequence of names")
    resolved = tuple(body_names)
    if resolved != SONIC_RELEASE_TRACKING_BODY_NAMES:
        raise AssertionError(
            "SONIC tracking bodies must exactly match the ordered release contract"
        )
    return resolved
