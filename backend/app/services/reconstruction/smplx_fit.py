"""
Stage 5.5 -- Multi-view SMPL-X fitting.

Fits ONE shared SMPL-X body (shape beta, pose theta, global orientation,
translation) against all three views (front/side/back) simultaneously,
using per-view cameras (see camera.py) rather than fitting three
independent bodies and averaging afterward -- this is what lets the depth/
girth information from the side view actually correct the front view's
shape ambiguity.

Losses combined every iteration:
  - keypoint reprojection loss (SMPL-X joints -> MediaPipe keypoints, per view)
  - silhouette loss (projected mesh vertices -> per-view distance-transform
    field from segmentation.py; a lightweight non-differentiable-renderer
    approximation, see segmentation.py docstring)
  - VPoser pose prior (keeps the pose plausible / penalizes unlikely poses)
  - shape regularization (keeps beta near the population mean; avoids the
    optimizer exploiting shape params to "cheat" the other losses)

Runs in three coarse-to-fine stages (camera+orientation -> pose -> shape/
joint refine), iteration counts configurable via Settings for CPU vs GPU.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from app.config import settings
from app.services.reconstruction.camera import (
    ViewCamera, init_view_cameras, project_points, camera_parameters,
)
from app.services.reconstruction.keypoints import ViewKeypoints, MEDIAPIPE_TO_SMPLX_JOINTS
from app.services.reconstruction.segmentation import SegmentationResult
from app.services.reconstruction.model_files import resolve_smplx_model_dir, resolve_vposer_ckpt

logger = logging.getLogger("sizesync.reconstruction.smplx_fit")


def resolve_device() -> str:
    if settings.RECONSTRUCTION_DEVICE != "auto":
        return settings.RECONSTRUCTION_DEVICE
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class FitResult:
    vertices: np.ndarray          # (10475, 3) float32, meters, canonical (unscaled) SMPL-X frame
    faces: np.ndarray             # (F, 3) int32
    joints: np.ndarray            # (J, 3) float32
    betas: np.ndarray
    body_pose: np.ndarray
    global_orient: np.ndarray
    final_loss: float
    per_view_reprojection_error_px: dict  # {"front": float, ...} mean px error, for confidence scoring


def _build_body_model(device: str):
    import smplx
    model_dir = resolve_smplx_model_dir()
    model = smplx.create(
        model_path=model_dir,
        model_type="smplx",
        gender=settings.SMPLX_GENDER,
        use_pca=False,           # explicit finger pose params (unused here, but avoids PCA ambiguity)
        num_betas=10,
        batch_size=1,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # body model weights are fixed; we only optimize its inputs
    return model


def _build_vposer(device: str):
    from human_body_prior.tools.model_loader import load_model
    from human_body_prior.models.vposer_model import VPoser

    ckpt = resolve_vposer_ckpt()
    vposer, _ = load_model(str(__import__("pathlib").Path(ckpt).parent.parent), model_code=VPoser,
                            remove_words_in_model_weights="vp_model.", disable_grad=True)
    return vposer.to(device)


def _keypoint_loss(joints_3d: torch.Tensor, cameras: dict, keypoints: dict, device: str) -> torch.Tensor:
    total = torch.tensor(0.0, device=device)
    count = 0
    for view, kp in keypoints.items():
        cam = cameras[view]
        target_px = torch.tensor(kp.points_px, device=device, dtype=torch.float32)
        vis = torch.tensor(kp.visibility, device=device, dtype=torch.float32)

        smplx_idx = list(MEDIAPIPE_TO_SMPLX_JOINTS.values())
        mp_idx = list(MEDIAPIPE_TO_SMPLX_JOINTS.keys())

        pred_px = project_points(joints_3d[smplx_idx], cam, device)
        target = target_px[mp_idx]
        w = vis[mp_idx].unsqueeze(-1)

        err = ((pred_px - target) ** 2 * w).sum()
        total = total + err
        count += len(mp_idx)
    return total / max(count, 1)


def _silhouette_loss(vertices_3d: torch.Tensor, cameras: dict, seg: dict, device: str,
                      sample_stride: int = 5) -> torch.Tensor:
    sampled = vertices_3d[::sample_stride]
    total = torch.tensor(0.0, device=device)
    for view, s in seg.items():
        cam = cameras[view]
        pred_px = project_points(sampled, cam, device)

        h, w = s.distance_field.shape
        field = torch.tensor(s.distance_field, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # normalize pixel coords to [-1, 1] for grid_sample
        gx = (pred_px[:, 0] / max(w - 1, 1)) * 2 - 1
        gy = (pred_px[:, 1] / max(h - 1, 1)) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)

        sampled_dist = torch.nn.functional.grid_sample(
            field, grid, mode="bilinear", padding_mode="border", align_corners=True
        ).view(-1)
        total = total + (sampled_dist ** 2).mean()
    return total / max(len(seg), 1)


def fit_multiview_smplx(
    keypoints: dict,               # {view: ViewKeypoints}
    segmentations: dict,            # {view: SegmentationResult}
    image_sizes: dict,               # {view: (w, h)}
    on_progress=None,                 # optional callable(stage_name: str, frac: float)
) -> FitResult:
    device = resolve_device()
    logger.info("Fitting on device=%s", device)

    body_model = _build_body_model(device)
    vposer = _build_vposer(device)

    views = list(keypoints.keys())
    cameras = init_view_cameras(views, image_sizes, device)

    betas = torch.zeros(1, 10, device=device, requires_grad=True)
    global_orient = torch.zeros(1, 3, device=device, requires_grad=True)
    transl = torch.zeros(1, 3, device=device, requires_grad=True)
    pose_latent = torch.zeros(1, 32, device=device, requires_grad=True)  # VPoser latent space

    def current_joints_and_vertices():
        body_pose = vposer.decode(pose_latent)["pose_body"].reshape(1, -1)
        out = body_model(
            betas=betas, global_orient=global_orient, transl=transl, body_pose=body_pose,
        )
        return out.joints[0], out.vertices[0], body_pose

    # --- Stage A: fit camera scale/offset + global orientation/translation ---
    # (shape and detailed pose still at rest -- just get the body roughly
    # in-frame and facing the right way in every view before refining detail)
    params_a = camera_parameters(cameras) + [global_orient, transl]
    opt_a = torch.optim.Adam(params_a, lr=0.05)
    for i in range(settings.SMPLX_FIT_ITERS_STAGE1):
        opt_a.zero_grad()
        joints, vertices, _ = current_joints_and_vertices()
        loss = _keypoint_loss(joints, cameras, keypoints, device)
        loss.backward()
        opt_a.step()
    if on_progress:
        on_progress("camera_and_orientation", 1.0)

    # --- Stage B: fit detailed pose (VPoser latent) ---
    params_b = [pose_latent, global_orient, transl] + camera_parameters(cameras)
    opt_b = torch.optim.Adam(params_b, lr=0.03)
    for i in range(settings.SMPLX_FIT_ITERS_STAGE2):
        opt_b.zero_grad()
        joints, vertices, _ = current_joints_and_vertices()
        kp_loss = _keypoint_loss(joints, cameras, keypoints, device)
        pose_prior_loss = (pose_latent ** 2).mean()  # VPoser latent ~ N(0, I); penalize departure
        loss = kp_loss + 0.02 * pose_prior_loss
        loss.backward()
        opt_b.step()
    if on_progress:
        on_progress("pose", 1.0)

    # --- Stage C: fit shape (betas) + silhouette refinement, everything jointly ---
    params_c = [betas, pose_latent, global_orient, transl] + camera_parameters(cameras)
    opt_c = torch.optim.Adam(params_c, lr=0.02)
    final_loss_value = 0.0
    for i in range(settings.SMPLX_FIT_ITERS_STAGE3):
        opt_c.zero_grad()
        joints, vertices, _ = current_joints_and_vertices()
        kp_loss = _keypoint_loss(joints, cameras, keypoints, device)
        sil_loss = _silhouette_loss(vertices, cameras, segmentations, device)
        pose_prior_loss = (pose_latent ** 2).mean()
        shape_prior_loss = (betas ** 2).mean()
        loss = kp_loss + 0.001 * sil_loss + 0.02 * pose_prior_loss + 0.01 * shape_prior_loss
        loss.backward()
        opt_c.step()
        final_loss_value = float(loss.item())
    if on_progress:
        on_progress("shape_and_refine", 1.0)

    with torch.no_grad():
        joints, vertices, body_pose = current_joints_and_vertices()

        per_view_error = {}
        for view, kp in keypoints.items():
            cam = cameras[view]
            smplx_idx = list(MEDIAPIPE_TO_SMPLX_JOINTS.values())
            mp_idx = list(MEDIAPIPE_TO_SMPLX_JOINTS.keys())
            pred_px = project_points(joints[smplx_idx], cam, device)
            target_px = torch.tensor(kp.points_px[mp_idx], device=device, dtype=torch.float32)
            err = torch.sqrt(((pred_px - target_px) ** 2).sum(-1)).mean().item()
            per_view_error[view] = err

        return FitResult(
            vertices=vertices.cpu().numpy(),
            faces=body_model.faces.astype(np.int32),
            joints=joints.cpu().numpy(),
            betas=betas.detach().cpu().numpy()[0],
            body_pose=body_pose.detach().cpu().numpy()[0],
            global_orient=global_orient.detach().cpu().numpy()[0],
            final_loss=final_loss_value,
            per_view_reprojection_error_px=per_view_error,
        )
