"""
Stage 5.2 -- Preprocessing.

Corrects EXIF orientation, resizes to a consistent working resolution, and
writes normalized working copies into a per-job scratch directory. This is
the only stage that touches the raw uploaded files; everything downstream
(segmentation, keypoints, fitting) reads from the preprocessed copies so
results are reproducible regardless of the original photo's orientation/size.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

TARGET_LONG_EDGE = 1024  # px -- enough detail for keypoints/segmentation, fast to process


@dataclass
class PreprocessedImage:
    view: str
    path: Path          # working copy on disk (RGB, EXIF-corrected, resized)
    width: int
    height: int


def _load_exif_corrected_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # rotates/flips per EXIF orientation tag
    return img.convert("RGB")


def _resize_long_edge(img: Image.Image, target: int = TARGET_LONG_EDGE) -> Image.Image:
    w, h = img.size
    scale = target / max(w, h)
    if scale >= 1.0:
        return img  # never upscale
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def preprocess_view(source_path: str, view: str, work_dir: Path) -> PreprocessedImage:
    """Load, EXIF-correct, resize, and write a working copy for one view."""
    img = _load_exif_corrected_rgb(source_path)
    img = _resize_long_edge(img)

    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / f"{view}_preprocessed.png"
    img.save(out_path)

    return PreprocessedImage(view=view, path=out_path, width=img.width, height=img.height)


def preprocess_all(paths: dict, work_dir: Path) -> dict:
    """paths: {"front": "...", "side": "...", "back": "..."} -> {view: PreprocessedImage}"""
    return {view: preprocess_view(p, view, work_dir) for view, p in paths.items()}


def load_normalized_array(pre: PreprocessedImage) -> np.ndarray:
    """RGB uint8 numpy array, for feeding into MediaPipe / OpenCV steps."""
    img = cv2.imread(str(pre.path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
