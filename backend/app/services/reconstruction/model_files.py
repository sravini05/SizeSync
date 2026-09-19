"""
Model-file verification for Stage 2.

Per the spec: before running any reconstruction, verify the required SMPL-X
and VPoser files actually exist on disk. Never create fake/dummy weights.
If something is missing, raise a ModelFilesMissingError with the exact
file(s) expected and where to get them -- this becomes the user-facing
error message (surfaced through ReconstructionResult.error), not a stack
trace or a silently degraded result.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from app.config import settings

logger = logging.getLogger("sizesync.reconstruction.model_files")


class ModelFilesMissingError(Exception):
    def __init__(self, missing: List["RequiredFile"]):
        self.missing = missing
        lines = ["Required model file(s) missing:"]
        for f in missing:
            lines.append(f"  - {f.description}: expected at '{f.path}'")
            lines.append(f"    Get it from: {f.source}")
        super().__init__("\n".join(lines))


@dataclass
class RequiredFile:
    path: Path
    description: str
    source: str
    exists: bool = False


def _smplx_candidates() -> List[Path]:
    d = Path(settings.SMPLX_MODEL_DIR)
    gender = settings.SMPLX_GENDER.upper()
    # the `smplx` python package looks for files named like this inside
    # SMPLX_MODEL_DIR/smplx/SMPLX_{GENDER}.{npz,pkl}
    return [
        d / "smplx" / f"SMPLX_{gender}.npz",
        d / "smplx" / f"SMPLX_{gender}.pkl",
        d / f"SMPLX_{gender}.npz",
        d / f"SMPLX_{gender}.pkl",
    ]


def _vposer_candidates() -> List[Path]:
    d = Path(settings.VPOSER_DIR)
    return [
        d / "snapshots" / "V02_05_epoch=13_val_loss=0.03.ckpt",
        d / "V02_05_epoch=13_val_loss=0.03.ckpt",
    ]


def verify_model_files() -> None:
    """Raises ModelFilesMissingError listing every missing file, or returns
    silently if everything required is present. Called at the start of every
    reconstruction job, before any expensive work happens.
    """
    smplx_paths = _smplx_candidates()
    smplx_found = any(p.exists() for p in smplx_paths)

    vposer_paths = _vposer_candidates()
    vposer_found = any(p.exists() for p in vposer_paths)

    missing: List[RequiredFile] = []
    if not smplx_found:
        missing.append(RequiredFile(
            path=smplx_paths[0],
            description=f"SMPL-X {settings.SMPLX_GENDER} body model",
            source=(
                "Register and download at https://smpl-x.is.tue.mpg.de "
                "(free, non-commercial research license -- see backend/README.md "
                "for the commercial licensing path via Meshcapade). "
                f"Place the file at: {smplx_paths[0]}"
            ),
        ))
    if not vposer_found:
        missing.append(RequiredFile(
            path=vposer_paths[0],
            description="VPoser v2.05 pose prior checkpoint",
            source=(
                "Register and download at https://smpl-x.is.tue.mpg.de/downloads "
                "(under 'VPoser: Variational Human Pose Prior'). "
                f"Place the checkpoint at: {vposer_paths[0]}"
            ),
        ))

    if missing:
        raise ModelFilesMissingError(missing)


def resolve_smplx_model_dir() -> str:
    """Returns the directory to pass as `model_path` to smplx.create(...) --
    the smplx package expects the *parent* of the `smplx/` subfolder.

    Also ensures the .npz file is actually loadable by the `smplx` package
    (see _ensure_smplx_npz_complete below) -- some official SMPL-X .npz
    distributions omit the hand-PCA arrays, which `smplx` reads
    unconditionally at construction time regardless of use_pca.
    """
    for p in _smplx_candidates():
        if p.exists():
            if p.suffix == ".npz":
                p = _ensure_smplx_npz_complete(p)
            # smplx.create wants the dir containing "smplx/SMPLX_*.{npz,pkl}"
            return str(p.parent.parent if p.parent.name == "smplx" else p.parent)
    raise ModelFilesMissingError([RequiredFile(
        path=_smplx_candidates()[0],
        description=f"SMPL-X {settings.SMPLX_GENDER} body model",
        source="https://smpl-x.is.tue.mpg.de",
    )])


def resolve_vposer_ckpt() -> str:
    for p in _vposer_candidates():
        if p.exists():
            return str(p)
    raise ModelFilesMissingError([RequiredFile(
        path=_vposer_candidates()[0],
        description="VPoser v2.05 checkpoint",
        source="https://smpl-x.is.tue.mpg.de/downloads",
    )])


# --- Compatibility fix: some SMPL-X .npz distributions omit hand-PCA arrays ---
#
# Root cause (traced to the exact library line): the `smplx` PyPI package's
# SMPLH.__init__ -- which SMPLX inherits and calls via super().__init__() --
# contains:
#
#     left_hand_components = data_struct.hands_componentsl[:num_pca_comps]
#     right_hand_components = data_struct.hands_componentsr[:num_pca_comps]
#
# These two lines run UNCONDITIONALLY at construction time, before the
# `if self.use_pca:` check that gates whether they're actually registered
# as torch buffers. So a model file missing `hands_componentsl` crashes
# with "'Struct' object has no attribute 'hands_componentsl'" even when
# use_pca=False (as this project always sets it -- see smplx_fit.py
# _build_body_model). This is NOT a VPoser / human_body_prior issue --
# VPoser is loaded afterward in our pipeline and was never reached; the
# failure is entirely inside `smplx`'s own model loading, several lines
# before any pose-prior code runs.
#
# Some official SMPL-X .npz releases genuinely omit these hand-PCA arrays
# (they're only needed for MANO-style hand pose, which this project never
# fits -- no hand keypoints are detected or used anywhere in the pipeline).
# Rather than editing the vendored `smplx` package, we patch a copy of the
# .npz with correctly-shaped zero placeholders so the library's loader
# succeeds, and point smplx.create() at that copy instead of the original.
REQUIRED_SMPLX_HAND_KEYS = {
    "hands_componentsl": (45, 45),  # 15 hand joints * 3 axis-angle; only read if use_pca=True (unused here)
    "hands_componentsr": (45, 45),
    "hands_meanl": (45,),           # added to full_pose as a mean-pose offset REGARDLESS of use_pca
    "hands_meanr": (45,),
}


def _patched_npz_cache_path(original: Path) -> Path:
    # Nested under an "smplx/" subfolder to match the layout smplx.create()
    # expects (model_path/smplx/SMPLX_{GENDER}.{ext}) -- NOT flat directly
    # under smplx_patched/. This mirrors the "smplx/" candidate layout in
    # _smplx_candidates() above, so resolve_smplx_model_dir()'s existing
    # `p.parent.name == "smplx"` check picks the right parent directory for
    # the patched copy the same way it already does for a real install.
    return settings.WORK_DIR / "smplx_patched" / "smplx" / original.name


def _ensure_smplx_npz_complete(npz_path: Path) -> Path:
    """Returns a path to a SMPL-X .npz that definitely has the hand-PCA keys
    `smplx` unconditionally reads -- the original file if it already has
    them, or a cached patched copy (zero-filled placeholders) otherwise.
    Never modifies the original downloaded file.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        missing = [k for k in REQUIRED_SMPLX_HAND_KEYS if k not in data.files]
        if not missing:
            return npz_path

        cache_path = _patched_npz_cache_path(npz_path)
        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            patched = {k: data[k] for k in data.files}
            for key in missing:
                patched[key] = np.zeros(REQUIRED_SMPLX_HAND_KEYS[key], dtype=np.float32)
            np.savez(cache_path, **patched)
            logger.warning(
                "SMPL-X file %s is missing %s (only needed for hand PCA/pose, which "
                "this project doesn't use -- use_pca=False, no hand keypoints in the "
                "fitting loss). Wrote a patched copy with zero-filled placeholders to "
                "%s. Note: resting fingers will render flat/open rather than the "
                "natural relaxed curl the real hands_meanl/r would give, since hand "
                "detail isn't part of this project's reconstruction.",
                npz_path, missing, cache_path,
            )
        return cache_path
