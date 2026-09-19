import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal, get_db
from app.models import BodyProfile, ProfileStatus, User
from app.schemas import BodyProfileOut, ImageValidationResult, MeasurementsOut, ValidationReport
from app.services.validation import validate_image
from app.services.reconstruction import get_reconstruction_service
from app.services.reconstruction.base import ReconstructionInput

logger = logging.getLogger("sizesync.routers.profile")
router = APIRouter(prefix="/api/profiles", tags=["profiles"])

# Statuses from which a (re)reconstruction can be started.
RECONSTRUCTABLE_STATUSES = {
    ProfileStatus.READY_FOR_RECONSTRUCTION,
    ProfileStatus.FAILED,
}
# Statuses that mean a job is actively running -- don't start a second one.
IN_FLIGHT_STATUSES = {
    ProfileStatus.PREPROCESSING, ProfileStatus.SEGMENTING, ProfileStatus.DETECTING_POSE,
    ProfileStatus.RECONSTRUCTING, ProfileStatus.FITTING_SMPLX, ProfileStatus.GENERATING_MESH,
}

VIEWS = ("front", "side", "back")


def _get_or_create_user(db: Session, client_ref: str) -> User:
    user = db.query(User).filter(User.client_ref == client_ref).first()
    if user:
        return user
    user = User(client_ref=client_ref)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _save_upload(upload: UploadFile, profile_id: str, view: str) -> str:
    if upload.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(400, f"Unsupported file type for {view} photo: {upload.content_type}")

    suffix = Path(upload.filename or "").suffix or ".jpg"
    dest_dir = settings.UPLOADS_DIR / profile_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{view}{suffix}"

    with dest_path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)

    size_mb = dest_path.stat().st_size / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_MB:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(400, f"{view} photo exceeds {settings.MAX_UPLOAD_MB}MB limit")

    return str(dest_path)


def _to_out(profile: BodyProfile) -> BodyProfileOut:
    validation_report = None
    if profile.validation_report:
        validation_report = ValidationReport(
            all_passed=profile.validation_report["all_passed"],
            results=[ImageValidationResult(**r) for r in profile.validation_report["results"]],
        )

    return BodyProfileOut(
        id=profile.id,
        status=profile.status.value if hasattr(profile.status, "value") else profile.status,
        height_cm=profile.height_cm,
        validation_report=validation_report,
        measurements=MeasurementsOut(
            height_cm=profile.height_cm,
            shoulder_cm=profile.shoulder_cm,
            chest_cm=profile.chest_cm,
            waist_cm=profile.waist_cm,
            hip_cm=profile.hip_cm,
            arm_length_cm=profile.arm_length_cm,
            inseam_cm=profile.inseam_cm,
            thigh_cm=profile.thigh_cm,
        ),
        confidence_scores=profile.confidence_scores,
        mesh_url=profile.mesh_url,  # already a public /media/... URL, see _run_reconstruction_job
        reconstruction_meta=profile.reconstruction_meta,
    )


@router.post("", response_model=BodyProfileOut)
async def create_profile(
    client_ref: str = Form(...),
    height_cm: float = Form(...),
    front: UploadFile = File(...),
    side: UploadFile = File(...),
    back: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not (100 <= height_cm <= 250):
        raise HTTPException(400, "height_cm must be between 100 and 250")

    user = _get_or_create_user(db, client_ref)

    profile = BodyProfile(user_id=user.id, height_cm=height_cm, status=ProfileStatus.UPLOADED)
    db.add(profile)
    db.commit()
    db.refresh(profile)

    try:
        paths = {}
        for view, upload in zip(VIEWS, (front, side, back)):
            paths[view] = _save_upload(upload, profile.id, view)

        profile.front_image_path = paths["front"]
        profile.side_image_path = paths["side"]
        profile.back_image_path = paths["back"]
        profile.status = ProfileStatus.VALIDATING
        db.commit()

        results = [validate_image(paths[view], view) for view in VIEWS]
        all_passed = all(r.passed for r in results)

        profile.validation_report = {
            "all_passed": all_passed,
            "results": [r.to_dict() for r in results],
        }
        profile.status = (
            ProfileStatus.READY_FOR_RECONSTRUCTION if all_passed else ProfileStatus.VALIDATION_FAILED
        )
        db.commit()
        db.refresh(profile)

    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Validation failed unexpectedly for profile %s", profile.id)
        profile.status = ProfileStatus.VALIDATION_FAILED
        profile.validation_report = {
            "all_passed": False,
            "results": [{
                "view": "unknown", "passed": False,
                "issues": [{"code": "internal_error", "message": str(exc)}],
                "checks": {},
            }],
        }
        db.commit()

    return _to_out(profile)


@router.get("/{profile_id}", response_model=BodyProfileOut)
def get_profile(profile_id: str, db: Session = Depends(get_db)):
    profile = db.query(BodyProfile).filter(BodyProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(404, "Profile not found")
    return _to_out(profile)


def _run_reconstruction_job(profile_id: str):
    """Runs in a background task -- opens its own DB session since the
    request-scoped one is gone by the time this executes. Updates the
    profile's status live as the pipeline advances through named stages,
    so GET /api/profiles/{id} polling reflects real progress.
    """
    db = SessionLocal()
    try:
        profile = db.query(BodyProfile).filter(BodyProfile.id == profile_id).first()
        if not profile:
            logger.error("Reconstruction job started for missing profile %s", profile_id)
            return

        def on_stage(stage_name: str):
            try:
                profile.status = ProfileStatus(stage_name)
                db.commit()
            except ValueError:
                logger.warning("Unknown reconstruction stage name %s", stage_name)

        try:
            service = get_reconstruction_service()
        except Exception as exc:  # noqa: BLE001 -- e.g. missing torch/smplx install
            profile.status = ProfileStatus.FAILED
            profile.reconstruction_meta = {"error": f"Reconstruction backend unavailable: {exc}"}
            db.commit()
            return

        result = service.reconstruct(
            ReconstructionInput(
                front_image_path=profile.front_image_path,
                side_image_path=profile.side_image_path,
                back_image_path=profile.back_image_path,
                height_cm=profile.height_cm,
            ),
            on_stage=on_stage,
        )

        if result.success:
            mesh_filename = Path(result.mesh_path).name
            profile.mesh_url = f"/media/meshes/{mesh_filename}"
            profile.confidence_scores = result.confidence_scores or None
            profile.reconstruction_meta = {
                "method": result.method,
                "is_prototype": result.is_prototype,
            }
            profile.status = ProfileStatus.COMPLETED
        else:
            profile.status = ProfileStatus.FAILED
            profile.reconstruction_meta = {"error": result.error, "method": result.method}
        db.commit()

    except Exception as exc:  # noqa: BLE001 -- last-resort catch so a job never hangs a profile forever
        logger.exception("Unhandled error in reconstruction job for profile %s", profile_id)
        try:
            profile = db.query(BodyProfile).filter(BodyProfile.id == profile_id).first()
            if profile:
                profile.status = ProfileStatus.FAILED
                profile.reconstruction_meta = {"error": f"Unexpected error: {exc}"}
                db.commit()
        except Exception:
            logger.exception("Also failed to record the failure for profile %s", profile_id)
    finally:
        db.close()


@router.post("/{profile_id}/reconstruct", response_model=BodyProfileOut)
def start_reconstruction(profile_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    profile = db.query(BodyProfile).filter(BodyProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(404, "Profile not found")

    if profile.status in IN_FLIGHT_STATUSES:
        return _to_out(profile)  # already running -- idempotent, just report current state

    if profile.status not in RECONSTRUCTABLE_STATUSES:
        raise HTTPException(
            400,
            f"Cannot start reconstruction from status '{profile.status.value}'. "
            "Photos must pass validation first.",
        )

    profile.status = ProfileStatus.PREPROCESSING
    profile.reconstruction_meta = None
    db.commit()
    db.refresh(profile)

    background_tasks.add_task(_run_reconstruction_job, profile_id)

    return _to_out(profile)
