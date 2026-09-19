import uuid
import enum
from datetime import datetime

from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON, Enum as SAEnum
from sqlalchemy.orm import relationship

from app.database import Base


def gen_id() -> str:
    return str(uuid.uuid4())


class ProfileStatus(str, enum.Enum):
    # --- Stage 1 (unchanged) ---
    UPLOADED = "uploaded"                     # files received, not yet validated
    VALIDATING = "validating"                  # validation running
    VALIDATION_FAILED = "validation_failed"     # one or more photos rejected
    READY_FOR_RECONSTRUCTION = "ready_for_reconstruction"  # validated, waiting for /reconstruct

    # --- Stage 2: AI body reconstruction pipeline ---
    PREPROCESSING = "preprocessing"             # EXIF fix, resize, normalize
    SEGMENTING = "segmenting"                    # MediaPipe silhouette masks
    DETECTING_POSE = "detecting_pose"             # MediaPipe 2D keypoints
    RECONSTRUCTING = "reconstructing"              # per-view camera init from keypoints
    FITTING_SMPLX = "fitting_smplx"                 # multi-view SMPL-X optimization
    GENERATING_MESH = "generating_mesh"              # forward pass + .glb export + height calibration
    COMPLETED = "completed"                           # mesh ready

    FAILED = "failed"                                  # any stage failed; reason in reconstruction_meta


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=gen_id)
    # No auth system in the MVP. A client-generated device/session id is
    # enough to let a browser find its own profile again.
    client_ref = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    profiles = relationship("BodyProfile", back_populates="user", cascade="all, delete-orphan")


class BodyProfile(Base):
    __tablename__ = "body_profiles"

    id = Column(String, primary_key=True, default=gen_id)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)

    height_cm = Column(Float, nullable=False)

    front_image_path = Column(String, nullable=True)
    side_image_path = Column(String, nullable=True)
    back_image_path = Column(String, nullable=True)

    status = Column(SAEnum(ProfileStatus), default=ProfileStatus.UPLOADED, nullable=False)

    # Populated by services/validation.py -- per-image pass/fail + reasons.
    validation_report = Column(JSON, nullable=True)

    # Populated in later phases by BodyReconstructionService. Left null here
    # on purpose -- this stage does not compute measurements.
    shoulder_cm = Column(Float, nullable=True)
    chest_cm = Column(Float, nullable=True)
    waist_cm = Column(Float, nullable=True)
    hip_cm = Column(Float, nullable=True)
    arm_length_cm = Column(Float, nullable=True)
    inseam_cm = Column(Float, nullable=True)
    thigh_cm = Column(Float, nullable=True)

    mesh_url = Column(String, nullable=True)
    confidence_scores = Column(JSON, nullable=True)  # {"chest": "high", "hip": "medium", ...}
    reconstruction_meta = Column(JSON, nullable=True)  # method name, prototype flag, timings, errors

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="profiles")
