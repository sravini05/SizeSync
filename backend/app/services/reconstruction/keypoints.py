"""
Stage 5.4 -- Pose / keypoint extraction.

Runs MediaPipe Pose Landmarker (the same model used for Stage 1 photo
validation, via app/services/pose_model.py) on each of the three views and
stores 2D pixel-space keypoints + per-point visibility/confidence.

Also defines the MediaPipe(33-point BlazePose) -> SMPL-X joint index
correspondence used by the fitting stage's reprojection loss. This is an
approximate mapping -- MediaPipe and SMPL-X don't share a joint definition,
so only joints with a reasonable anatomical correspondence are mapped;
everything else (spine chain, individual fingers, etc.) is left to the
pose prior (VPoser) rather than guessed at.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.services.pose_model import get_pose_landmarker, POSE_MODEL_REQUIREMENT_HINT

logger = logging.getLogger("sizesync.reconstruction.keypoints")


class PoseModelMissingError(Exception):
    pass


class PoseNotDetectedError(Exception):
    pass


@dataclass
class ViewKeypoints:
    view: str
    points_px: np.ndarray       # (33, 2) float32, pixel coordinates in the preprocessed image
    visibility: np.ndarray      # (33,) float32, 0..1
    image_width: int
    image_height: int


def extract_keypoints(rgb: np.ndarray, view: str) -> ViewKeypoints:
    landmarker = get_pose_landmarker()
    if landmarker is None:
        raise PoseModelMissingError(POSE_MODEL_REQUIREMENT_HINT)

    import mediapipe as mp

    h, w = rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        raise PoseNotDetectedError(
            f"No body pose detected in the {view} photo during reconstruction "
            "(it passed initial validation, but re-detection failed -- please retry)."
        )

    landmarks = result.pose_landmarks[0]
    points = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
    visibility = np.array([getattr(lm, "visibility", 1.0) for lm in landmarks], dtype=np.float32)

    return ViewKeypoints(view=view, points_px=points, visibility=visibility, image_width=w, image_height=h)


def extract_all(rgb_by_view: dict) -> dict:
    return {view: extract_keypoints(rgb, view) for view, rgb in rgb_by_view.items()}


# --- MediaPipe (33-pt BlazePose) -> SMPL-X joint index correspondence ---
# SMPL-X main body joints (first 22, from the standard SMPL/SMPL-X joint
# ordering used by the `smplx` package's `.joints` output):
#   0 pelvis, 1 left_hip, 2 right_hip, 3 spine1, 4 left_knee, 5 right_knee,
#   6 spine2, 7 left_ankle, 8 right_ankle, 9 spine3, 10 left_foot, 11 right_foot,
#   12 neck, 13 left_collar, 14 right_collar, 15 head, 16 left_shoulder,
#   17 right_shoulder, 18 left_elbow, 19 right_elbow, 20 left_wrist, 21 right_wrist
#
# MediaPipe Pose landmark indices (0-32): 0 nose, 11/12 shoulders, 13/14 elbows,
# 15/16 wrists, 23/24 hips, 25/26 knees, 27/28 ankles, 31/32 foot index.
#
# Only joints with a direct, unambiguous anatomical match are included.
# Pelvis/spine/collar/neck are intentionally left unmapped -- MediaPipe has
# no direct equivalent, and guessing a synthetic target would inject noise
# into the fit. VPoser's pose prior keeps the unconstrained joints plausible.
#
# Mapping: MediaPipe landmark index -> SMPL-X joint index.
MEDIAPIPE_TO_SMPLX_JOINTS: dict[int, int] = {
    11: 16,  # left_shoulder   -> left_shoulder
    12: 17,  # right_shoulder  -> right_shoulder
    13: 18,  # left_elbow      -> left_elbow
    14: 19,  # right_elbow     -> right_elbow
    15: 20,  # left_wrist      -> left_wrist
    16: 21,  # right_wrist     -> right_wrist
    23: 1,   # left_hip        -> left_hip
    24: 2,   # right_hip       -> right_hip
    25: 4,   # left_knee       -> left_knee
    26: 5,   # right_knee      -> right_knee
    27: 7,   # left_ankle      -> left_ankle
    28: 8,   # right_ankle     -> right_ankle
    31: 10,  # left_foot_index -> left_foot
    32: 11,  # right_foot_index-> right_foot
    0: 15,   # nose            -> head (approximate; head joint is skull base, nose is offset)
}
