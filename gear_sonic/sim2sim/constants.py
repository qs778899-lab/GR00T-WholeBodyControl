"""Shared constants for sim2sim MuJoCo evaluation."""

import numpy as np

REFERENCE_NAME_PREFIX = "ref_"
PACKED_ZMQ_HEADER_SIZE = 1280
G1_ISAACLAB_TO_MUJOCO_DOF = np.array(
    [
        0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
        11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
    ],
    dtype=np.int32,
)

SIM2SIM_BODY_FRAMES = [
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
]
