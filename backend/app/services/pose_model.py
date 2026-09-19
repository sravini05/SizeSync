"""
Shared MediaPipe Pose Landmarker loader.

Both app/services/validation.py (Stage 1 photo checks) and
app/services/reconstruction/keypoints.py (Stage 2 fitting) need the same
33-point BlazePose landmarker. This module loads it once and hands back the
same instance, and never raises -- callers get None if the model file isn't
present and decide for themselves whether that's fatal.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger("sizesync.pose_model")

_landmarker = None
_load_attempted = False


def get_pose_landmarker():
    """Returns a shared mediapipe.tasks.vision.PoseLandmarker (IMAGE mode),
    or None if the model file isn't present / failed to load.
    """
    global _landmarker, _load_attempted
    if _load_attempted:
        return _landmarker
    _load_attempted = True

    model_path = Path(settings.POSE_MODEL_PATH)
    if not model_path.exists():
        logger.warning("Pose model not found at %s.", model_path)
        return None

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
        )
        _landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        logger.info("MediaPipe Pose Landmarker loaded from %s", model_path)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Failed to load Pose Landmarker (%s).", exc)
        _landmarker = None

    return _landmarker


POSE_MODEL_REQUIREMENT_HINT = (
    f"Missing MediaPipe Pose Landmarker model at '{settings.POSE_MODEL_PATH}'. "
    "Download it with:\n"
    "  curl -L -o "
    f"{settings.POSE_MODEL_PATH} "
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
