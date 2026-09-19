"""
MultiViewSmplxReconstructionService -- the real Stage 2 implementation of
BodyReconstructionService.

Orchestrates: model-file verification -> preprocessing -> segmentation ->
keypoint detection -> multi-view SMPL-X fitting -> mesh generation -> height
calibration -> .glb export. Reports progress via the optional `on_stage`
callback at each named stage so the caller (the /reconstruct endpoint) can
persist live status for the frontend to poll.

Never returns success=True without an actual exported mesh file on disk.
Any failure at any stage returns success=False with the specific reason --
no fallback to a placeholder mesh, no fabricated measurements (measurements
aren't computed at all yet -- that's Stage 3).
"""
from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from app.config import settings
from app.services.reconstruction.base import (
    BodyReconstructionService, ReconstructionInput, ReconstructionResult,
)
from app.services.reconstruction.model_files import verify_model_files, ModelFilesMissingError
from app.services.reconstruction.preprocessing import preprocess_all, load_normalized_array
from app.services.reconstruction.segmentation import (
    segment_all, SegmentationModelMissingError, SegmentationFailedError,
)
from app.services.reconstruction.keypoints import extract_all, PoseModelMissingError, PoseNotDetectedError
from app.services.reconstruction.mesh_export import compute_scale_factor, export_glb

logger = logging.getLogger("sizesync.reconstruction.multiview_smplx")

METHOD_NAME = "smplx-multiview-optimization-v1"


class MultiViewSmplxReconstructionService(BodyReconstructionService):

    def reconstruct(
        self,
        data: ReconstructionInput,
        on_stage: Optional[callable] = None,
    ) -> ReconstructionResult:
        def stage(name: str):
            logger.info("Reconstruction stage: %s", name)
            if on_stage:
                on_stage(name)

        job_id = uuid.uuid4().hex[:12]
        work_dir = settings.WORK_DIR / job_id
        started = time.time()

        try:
            # --- Precondition: verify SMPL-X / VPoser files exist. Never
            # fabricate them, never proceed without them. ---
            verify_model_files()

            # --- 5.2 Preprocessing ---
            stage("preprocessing")
            source_paths = {
                "front": data.front_image_path,
                "side": data.side_image_path,
                "back": data.back_image_path,
            }
            preprocessed = preprocess_all(source_paths, work_dir)
            rgb_by_view = {view: load_normalized_array(p) for view, p in preprocessed.items()}
            image_sizes = {view: (p.width, p.height) for view, p in preprocessed.items()}

            # --- 5.3 Segmentation ---
            stage("segmenting")
            segmentations = segment_all(rgb_by_view, work_dir)

            # --- 5.4 Pose / keypoints ---
            stage("detecting_pose")
            keypoints = extract_all(rgb_by_view)

            # --- 5.5 Multi-view SMPL-X fitting (camera/orientation init counts
            # as "reconstructing"; the optimization itself as "fitting_smplx") ---
            stage("reconstructing")
            from app.services.reconstruction.smplx_fit import fit_multiview_smplx

            def fit_progress(sub_stage: str, frac: float):
                if sub_stage in ("pose", "shape_and_refine"):
                    stage("fitting_smplx")

            fit_result = fit_multiview_smplx(
                keypoints=keypoints,
                segmentations=segmentations,
                image_sizes=image_sizes,
                on_progress=fit_progress,
            )

            # --- 5.6/5.7 Mesh generation + height calibration + .glb export ---
            stage("generating_mesh")
            scale_factor = compute_scale_factor(fit_result.joints, data.height_cm)

            mesh_filename = f"{job_id}.glb"
            mesh_path = settings.MESHES_DIR / mesh_filename
            export_glb(fit_result.vertices, fit_result.faces, scale_factor, mesh_path)

            if not mesh_path.exists() or mesh_path.stat().st_size == 0:
                # Belt-and-braces: never report success without a real file on disk.
                raise RuntimeError("Mesh export reported success but no file was written.")

            elapsed = time.time() - started

            confidence_scores = _estimate_confidence(fit_result.per_view_reprojection_error_px)

            return ReconstructionResult(
                success=True,
                mesh_path=str(mesh_path),
                measurements=None,  # Stage 3 -- intentionally not computed here
                confidence_scores=confidence_scores,
                method=METHOD_NAME,
                is_prototype=True,  # still true until real multi-camera calibration replaces the 0/90/180 assumption
                error=None,
            )

        except ModelFilesMissingError as exc:
            return ReconstructionResult(
                success=False, method=METHOD_NAME, is_prototype=True,
                error=f"Setup incomplete -- {exc}",
            )
        except (SegmentationModelMissingError, PoseModelMissingError) as exc:
            return ReconstructionResult(
                success=False, method=METHOD_NAME, is_prototype=True,
                error=f"Setup incomplete -- {exc}",
            )
        except (SegmentationFailedError, PoseNotDetectedError) as exc:
            return ReconstructionResult(
                success=False, method=METHOD_NAME, is_prototype=True,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Reconstruction job %s failed unexpectedly", job_id)
            return ReconstructionResult(
                success=False, method=METHOD_NAME, is_prototype=True,
                error=f"Reconstruction failed: {exc}",
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


def _estimate_confidence(per_view_error_px: dict) -> dict:
    """Rough confidence heuristic from mean keypoint reprojection error per
    view -- lower error = the fitted body explains the detected keypoints
    well in that view. This scores overall fit quality; Stage 3 will refine
    this into per-measurement confidence once actual measurements exist.
    """
    if not per_view_error_px:
        return {}
    avg_err = sum(per_view_error_px.values()) / len(per_view_error_px)
    if avg_err < 15:
        level = "high"
    elif avg_err < 35:
        level = "medium"
    else:
        level = "low"
    return {"overall_fit": level, **{f"{v}_view_px_error": round(e, 1) for v, e in per_view_error_px.items()}}
