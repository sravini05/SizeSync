"""
Stage 5.3 -- Segmentation.

Uses MediaPipe's Image Segmenter to get a binary person silhouette mask per
view. Also precomputes a signed-distance field from each mask: this is what
the fitting stage (5.5) uses for the silhouette loss term, sampled at each
projected mesh vertex via bilinear interpolation -- a lightweight
alternative to a full differentiable renderer (which would pull in
PyTorch3D / nvdiffrast and their notoriously painful Windows installs).

Model choice (fixed 2024 bug): this originally used `selfie_segmenter.tflite`,
which is a PORTRAIT/SELFIE model -- per MediaPipe's own docs it's trained on
close-up, head-and-shoulders, video-call-style framing at a small square
input (256x256 / 144x256). Our app requires full-body, head-to-feet photos
(Stage 1 validation enforces this), which is exactly the framing that model
is not trained for -- it would frequently classify most of a legitimate
full-body photo as background, producing near-zero person pixels even on a
perfectly good photo.

Fixed by switching to MediaPipe's DeepLab v3 model, which is a
general-purpose semantic segmenter (Apache 2.0, PASCAL VOC 21 categories,
"person" among them) not restricted to close-up portrait framing, and is
the model MediaPipe's own docs recommend for anything other than selfie
segmentation. See MEDIAPIPE_SEGMENTER_MIGRATION note below for the exact
change and why.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger("sizesync.reconstruction.segmentation")

_segmenter = None
_load_attempted = False
_person_category_index: Optional[int] = None

# Standard PASCAL VOC 2012 class ordering used by DeepLab v3. Used only as a
# fallback if the model file doesn't embed its own label map (MediaPipe's
# ImageSegmenter.labels returns an empty list in that case, per their docs) --
# whenever the label map IS present, we resolve "person" from it directly
# instead of trusting this hardcoded index.
PASCAL_VOC_PERSON_INDEX = 15

# Minimum person-pixel coverage required to accept a segmentation, expressed
# as a FRACTION of total image pixels rather than a fixed pixel count -- a
# fixed count (the previous implementation used 500px flat) doesn't scale
# with the preprocessed image's resolution. 0.5% is intentionally low; this
# check exists to catch true failures (wrong model output, near-empty mask),
# not to second-guess framing that Stage 1 validation already approved.
MIN_PERSON_COVERAGE_FRACTION = 0.005


class SegmentationModelMissingError(Exception):
    """Raised when the .tflite model file itself isn't present."""


class SegmentationFailedError(Exception):
    """Raised when the model ran but produced an unusable result for a given photo."""


def _segmenter_model_path() -> Path:
    # Reuses the same models/ dir as the pose landmarker for convenience.
    return Path(settings.POSE_MODEL_PATH).parent / "deeplabv3.tflite"


def _get_segmenter():
    global _segmenter, _load_attempted, _person_category_index
    if _load_attempted:
        return _segmenter
    _load_attempted = True

    model_path = _segmenter_model_path()
    if not model_path.exists():
        logger.warning("Segmentation model not found at %s", model_path)
        return None

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        _segmenter = mp_vision.ImageSegmenter.create_from_options(options)

        # Resolve the "person" category index from the model's own label map
        # rather than assuming an index -- falls back to the standard PASCAL
        # VOC ordering only if the model has no embedded label map.
        labels = list(getattr(_segmenter, "labels", []) or [])
        if "person" in labels:
            _person_category_index = labels.index("person")
        else:
            if labels:
                logger.warning(
                    "Segmentation model's label map has no 'person' category (%s); "
                    "falling back to PASCAL VOC index %d.", labels, PASCAL_VOC_PERSON_INDEX,
                )
            _person_category_index = PASCAL_VOC_PERSON_INDEX
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Failed to load Image Segmenter (%s).", exc)
        _segmenter = None

    return _segmenter


SEGMENTER_REQUIREMENT_HINT = (
    "Missing MediaPipe Image Segmenter model. Download it with:\n"
    f"  curl -L -o {_segmenter_model_path()} "
    '"https://storage.googleapis.com/mediapipe-assets/deeplabv3.tflite?generation=1661875711618421"\n'
    "(Note: this is DeepLab v3, a general-purpose segmenter -- NOT selfie_segmenter.tflite, "
    "which is a close-up-portrait-only model that misclassifies full-body photos as background.)"
)


@dataclass
class SegmentationResult:
    view: str
    mask: np.ndarray            # bool, HxW, True = person
    distance_field: np.ndarray  # float32, HxW; 0 inside mask, positive distance outside
    coverage_fraction: float    # sanity metric: fraction of pixels classified as person


def segment_view(rgb: np.ndarray, view: str, work_dir: Optional[Path] = None) -> SegmentationResult:
    segmenter = _get_segmenter()
    if segmenter is None:
        raise SegmentationModelMissingError(SEGMENTER_REQUIREMENT_HINT)

    import mediapipe as mp

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = segmenter.segment(mp_image)

    category_mask = result.category_mask.numpy_view()
    mask = category_mask == _person_category_index

    coverage = float(mask.sum()) / mask.size
    if coverage < MIN_PERSON_COVERAGE_FRACTION:
        raise SegmentationFailedError(
            f"Segmentation found almost no person pixels in the {view} photo "
            f"({coverage * 100:.2f}% of the image, category index {_person_category_index}). "
            "The photo may not match the validated original -- please retry."
        )

    outside = (~mask).astype(np.uint8)
    distance_field = cv2.distanceTransform(outside, cv2.DIST_L2, 5).astype(np.float32)

    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(work_dir / f"{view}_mask.png"), (mask * 255).astype(np.uint8))

    return SegmentationResult(view=view, mask=mask, distance_field=distance_field, coverage_fraction=coverage)


def segment_all(rgb_by_view: dict, work_dir: Optional[Path] = None) -> dict:
    return {view: segment_view(rgb, view, work_dir) for view, rgb in rgb_by_view.items()}
