"""
Central configuration for the SizeSync backend.
All values are overridable via environment variables (see .env.example).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    # Storage
    STORAGE_DIR: Path = Path(os.getenv("SIZESYNC_STORAGE_DIR", BASE_DIR / "storage"))
    UPLOADS_DIR: Path = STORAGE_DIR / "uploads"
    MESHES_DIR: Path = STORAGE_DIR / "meshes"

    # Database
    DATABASE_URL: str = os.getenv(
        "SIZESYNC_DATABASE_URL", f"sqlite:///{BASE_DIR / 'sizesync.db'}"
    )

    # Uploads
    MAX_UPLOAD_MB: int = int(os.getenv("SIZESYNC_MAX_UPLOAD_MB", "15"))
    ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

    # Pose/validation model
    # Path to the MediaPipe Pose Landmarker .task model file. Must be downloaded
    # separately (see backend/README.md) because it is not bundled with the
    # mediapipe pip package and this environment has no outbound network access.
    POSE_MODEL_PATH: str = os.getenv(
        "SIZESYNC_POSE_MODEL_PATH", str(BASE_DIR / "models" / "pose_landmarker_full.task")
    )

    # Reconstruction backend selection. "prototype" = honest not-implemented
    # stub. "smplx-multiview" = the real Stage 2 pipeline (see
    # services/reconstruction/multiview_smplx.py).
    RECONSTRUCTION_BACKEND: str = os.getenv("SIZESYNC_RECONSTRUCTION_BACKEND", "prototype")

    # -- Stage 2: SMPL-X reconstruction --
    # Directory containing the SMPL-X body model files, downloaded manually
    # by the developer from https://smpl-x.is.tue.mpg.de (gated, requires
    # registration -- see backend/README.md). Expected filename inside:
    # SMPLX_NEUTRAL.npz (the smplx python package also accepts .pkl).
    SMPLX_MODEL_DIR: str = os.getenv("SIZESYNC_SMPLX_MODEL_DIR", str(BASE_DIR / "models" / "smplx"))
    SMPLX_GENDER: str = os.getenv("SIZESYNC_SMPLX_GENDER", "neutral")

    # VPoser checkpoint directory (human_body_prior), also gated/manual download.
    VPOSER_DIR: str = os.getenv("SIZESYNC_VPOSER_DIR", str(BASE_DIR / "models" / "vposer_v2_05"))

    # "cuda" if a GPU is available and torch was built with CUDA, else "cpu".
    # Never silently falls back to a weaker/fake method -- just runs slower.
    RECONSTRUCTION_DEVICE: str = os.getenv("SIZESYNC_RECONSTRUCTION_DEVICE", "auto")

    # Number of optimization iterations per fitting stage (coarse->fine).
    # Lower = faster/rougher, higher = slower/more accurate. Tune down for
    # CPU-only machines.
    SMPLX_FIT_ITERS_STAGE1: int = int(os.getenv("SIZESYNC_FIT_ITERS_STAGE1", "60"))   # orient+translation
    SMPLX_FIT_ITERS_STAGE2: int = int(os.getenv("SIZESYNC_FIT_ITERS_STAGE2", "120"))  # pose
    SMPLX_FIT_ITERS_STAGE3: int = int(os.getenv("SIZESYNC_FIT_ITERS_STAGE3", "120"))  # shape + refine

    # Scratch space for intermediate per-job files (masks, keypoint json, etc).
    WORK_DIR: Path = STORAGE_DIR / "work"

    # CORS
    CORS_ORIGINS = os.getenv("SIZESYNC_CORS_ORIGINS", "http://localhost:5173").split(",")


settings = Settings()

for d in (settings.STORAGE_DIR, settings.UPLOADS_DIR, settings.MESHES_DIR, settings.WORK_DIR):
    d.mkdir(parents=True, exist_ok=True)
