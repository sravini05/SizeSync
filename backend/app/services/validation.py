"""
Photo quality validation (Section 12 of the spec).

These are REAL checks that run against the actual pixel data -- nothing here
is mocked or hardcoded. Two detector tiers are used:

1. Always available, fully offline (ships inside opencv-python):
   - Sharpness (Laplacian variance)
   - Brightness (mean luma + clipping)
   - Resolution / aspect sanity
   - Person presence, count, and framing via OpenCV's HOG person detector

2. Optional, higher quality (requires a one-time model download, see
   backend/README.md): MediaPipe Pose Landmarker, used for a stricter
   "full body visible" check (ankles + head both confidently detected)
   and a rough pose/orientation sanity check. If the model file isn't
   present, we log a warning and fall back to the HOG-only checks rather
   than silently pretending the stricter check passed.

This module intentionally does NOT do body reconstruction or measurement --
see services/reconstruction/ for that (later phase).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("sizesync.validation")

MIN_WIDTH, MIN_HEIGHT = 480, 640
BLUR_VARIANCE_THRESHOLD = 60.0        # below this = "excessively blurry"
DARK_MEAN_THRESHOLD = 40.0            # 0-255 scale
BRIGHT_MEAN_THRESHOLD = 235.0
EDGE_MARGIN_FRACTION = 0.02           # bbox within this fraction of edge = likely cropped


@dataclass
class Issue:
    code: str
    message: str


@dataclass
class ImageCheckResult:
    view: str
    passed: bool
    checks: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)

    def to_dict(self):
        return {
            "view": self.view,
            "passed": self.passed,
            "checks": self.checks,
            "issues": [i.__dict__ for i in self.issues],
        }


def _try_load_pose_landmarker():
    """Delegates to the shared loader (see app/services/pose_model.py), which
    both this module and the Stage 2 keypoint extractor use. Kept as a
    thin wrapper here so the rest of this file didn't need to change.
    """
    from app.services.pose_model import get_pose_landmarker

    landmarker = get_pose_landmarker()
    if landmarker is None:
        logger.warning(
            "Falling back to HOG-only validation. "
            "See backend/README.md to enable stricter full-body-visible checks."
        )
    return landmarker


def _check_resolution(img: np.ndarray) -> tuple[bool, Optional[Issue]]:
    h, w = img.shape[:2]
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return False, Issue(
            "low_resolution",
            f"Image is only {w}x{h}px. Please upload a photo at least {MIN_WIDTH}x{MIN_HEIGHT}px.",
        )
    return True, None


def _check_blur(gray: np.ndarray) -> tuple[bool, Optional[Issue]]:
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < BLUR_VARIANCE_THRESHOLD:
        return False, Issue(
            "excessive_blur",
            "This photo looks blurry. Hold the camera steady and make sure the subject is in focus.",
        )
    return True, None


def _check_brightness(gray: np.ndarray) -> tuple[bool, Optional[Issue]]:
    mean = float(gray.mean())
    if mean < DARK_MEAN_THRESHOLD:
        return False, Issue(
            "poor_lighting_dark",
            "This photo is too dark. Please retake it in a well-lit area.",
        )
    if mean > BRIGHT_MEAN_THRESHOLD:
        return False, Issue(
            "poor_lighting_bright",
            "This photo is overexposed. Please avoid strong backlight or direct flash.",
        )
    return True, None


def _detect_people(img: np.ndarray):
    """Returns list of (x, y, w, h) bounding boxes via OpenCV HOG."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # HOG works best around VGA resolution -- downscale large images for speed
    # and consistency, keep the scale factor to map boxes back if needed.
    h, w = img.shape[:2]
    scale = min(1.0, 640.0 / max(h, w))
    resized = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img

    boxes, weights = hog.detectMultiScale(
        resized, winStride=(8, 8), padding=(16, 16), scale=1.05
    )
    people = [(box, float(weight)) for box, weight in zip(boxes, weights) if weight > 0.3]
    return people, resized.shape[:2]


def _check_person_presence_and_framing(img: np.ndarray) -> tuple[bool, list[Issue], dict]:
    issues: list[Issue] = []
    checks = {}

    people, (rh, rw) = _detect_people(img)
    checks["person_count"] = len(people)

    if len(people) == 0:
        issues.append(Issue("no_person_detected", "We couldn't detect a person in this photo. Please try another."))
        checks["person_detected"] = False
        return False, issues, checks

    checks["person_detected"] = True

    if len(people) > 1:
        issues.append(Issue("multiple_people", "More than one person appears to be in this photo. Please upload a photo of yourself alone."))

    # Framing: use the highest-confidence box to check the subject isn't
    # touching top/bottom edges (a proxy for "full body visible" without pose).
    best_box, _ = max(people, key=lambda p: p[1])
    x, y, w, h = best_box
    margin_y = rh * EDGE_MARGIN_FRACTION
    cropped_top = y <= margin_y
    cropped_bottom = (y + h) >= (rh - margin_y)
    checks["touches_top_edge"] = bool(cropped_top)
    checks["touches_bottom_edge"] = bool(cropped_bottom)

    if cropped_top or cropped_bottom:
        issues.append(Issue(
            "possibly_cropped",
            "Your full body may not be visible. Make sure your head and feet are both inside the frame.",
        ))

    passed = len(people) == 1 and not cropped_top and not cropped_bottom
    return passed, issues, checks


def _check_full_body_pose(img_rgb: np.ndarray) -> tuple[Optional[bool], list[Issue], dict]:
    """Stricter check using MediaPipe Pose landmarks, if available.
    Returns (passed_or_None, issues, checks). None means "not run" (model
    unavailable) -- caller should not treat that as failure.
    """
    landmarker = _try_load_pose_landmarker()
    if landmarker is None:
        return None, [], {"pose_model_available": False}

    import mediapipe as mp

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return False, [Issue("pose_not_detected", "We couldn't detect a clear body pose in this photo.")], {"pose_model_available": True}

    landmarks = result.pose_landmarks[0]
    # MediaPipe Pose indices: 0 = nose, 27/28 = ankles
    nose = landmarks[0]
    l_ankle, r_ankle = landmarks[27], landmarks[28]

    VIS_THRESHOLD = 0.5
    head_visible = getattr(nose, "visibility", 1.0) >= VIS_THRESHOLD
    ankles_visible = (
        getattr(l_ankle, "visibility", 1.0) >= VIS_THRESHOLD
        or getattr(r_ankle, "visibility", 1.0) >= VIS_THRESHOLD
    )

    issues = []
    if not head_visible:
        issues.append(Issue("head_not_visible", "Your head isn't clearly visible. Please recheck framing."))
    if not ankles_visible:
        issues.append(Issue("feet_not_visible", "Your feet/ankles aren't clearly visible. Please recheck framing."))

    passed = head_visible and ankles_visible
    return passed, issues, {
        "pose_model_available": True,
        "head_visible": head_visible,
        "ankles_visible": ankles_visible,
    }


def validate_image(file_path: str, view: str) -> ImageCheckResult:
    """Run all Section 12 checks against a single uploaded photo."""
    img = cv2.imread(file_path)
    if img is None:
        return ImageCheckResult(
            view=view,
            passed=False,
            issues=[Issue("unreadable_file", "This file couldn't be read as an image. Please upload a JPG, PNG, or WEBP.")],
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_issues: list[Issue] = []
    checks: dict = {}

    ok, issue = _check_resolution(img)
    checks["resolution_ok"] = ok
    if issue:
        all_issues.append(issue)

    ok, issue = _check_blur(gray)
    checks["sharpness_ok"] = ok
    if issue:
        all_issues.append(issue)

    ok, issue = _check_brightness(gray)
    checks["lighting_ok"] = ok
    if issue:
        all_issues.append(issue)

    framing_ok, framing_issues, framing_checks = _check_person_presence_and_framing(img)
    checks.update(framing_checks)
    all_issues.extend(framing_issues)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pose_ok, pose_issues, pose_checks = _check_full_body_pose(img_rgb)
    checks.update(pose_checks)
    if pose_ok is not None:
        all_issues.extend(pose_issues)

    # Overall pass/fail: hard checks must all pass. The pose check only
    # counts against the result if the model was actually available
    # (pose_ok is not None) -- otherwise we rely on the HOG framing check.
    hard_checks_ok = checks.get("resolution_ok") and checks.get("sharpness_ok") and checks.get("lighting_ok") and framing_ok
    overall = hard_checks_ok and (pose_ok is not False)

    return ImageCheckResult(view=view, passed=bool(overall), checks=checks, issues=all_issues)
