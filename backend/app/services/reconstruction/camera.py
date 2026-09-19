"""
Per-view camera model for the front/side/back multi-view fit.

We do NOT have calibrated cameras -- three phone photos aren't a synced
multi-camera rig. Instead we make an explicit, documented approximation:
the subject stood in roughly the same spot and rotated in place through
0 degrees (front), 90 degrees (side), and 180 degrees (back). Each view
therefore gets a FIXED rotation (not learned -- learning it from three
single images is underconstrained and would likely drift to a degenerate
solution) plus a small per-view LEARNED weak-perspective scale + 2D offset,
which absorbs the fact the phone wasn't held at exactly the same distance
or height for every shot.

This is a documented limitation (see Stage 2 README "Known limitations"),
not a hidden simplification -- multi-view still substantially constrains
shape compared to any single view, it just isn't lab-grade photogrammetry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Fixed yaw rotation (radians) applied to the shared 3D body for each view,
# rotating the body rather than moving the camera -- equivalent, simpler to
# reason about since the body's own global_orient is already a free variable.
VIEW_YAW_RADIANS = {
    "front": 0.0,
    "side": math.pi / 2,
    "back": math.pi,
}


@dataclass
class ViewCamera:
    view: str
    yaw: float               # fixed, radians
    scale: torch.Tensor      # learned, scalar
    offset: torch.Tensor     # learned, (2,) -- pixel-space xy offset


def init_view_cameras(views, image_sizes: dict, device: str) -> dict:
    """image_sizes: {view: (width, height)} of the preprocessed images.
    Returns {view: ViewCamera} with learnable scale/offset parameters
    initialized to roughly center+fit the body in frame; the optimizer
    refines these in fitting stage 1 (orientation+translation).
    """
    cameras = {}
    for view in views:
        w, h = image_sizes[view]
        scale = torch.tensor(float(h) / 2.2, device=device, requires_grad=True)  # rough initial guess
        offset = torch.tensor([w / 2.0, h / 2.0], device=device, requires_grad=True)
        cameras[view] = ViewCamera(view=view, yaw=VIEW_YAW_RADIANS[view], scale=scale, offset=offset)
    return cameras


def yaw_rotation_matrix(yaw: float, device: str) -> torch.Tensor:
    c, s = math.cos(yaw), math.sin(yaw)
    return torch.tensor([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], device=device, dtype=torch.float32)


def project_points(points_3d: torch.Tensor, camera: ViewCamera, device: str) -> torch.Tensor:
    """Weak-perspective projection: rotate by the view's fixed yaw, then
    orthographically project (x, y) scaled + offset into pixel space.
    points_3d: (N, 3) in the shared body/world frame.
    Returns: (N, 2) pixel coordinates.
    """
    R = yaw_rotation_matrix(camera.yaw, device)
    rotated = points_3d @ R.T
    xy = rotated[:, :2]
    # Flip y: image coordinates increase downward, body "up" (+Y) should map
    # to smaller pixel y.
    xy = xy * torch.tensor([1.0, -1.0], device=device)
    projected = xy * camera.scale + camera.offset
    return projected


def camera_parameters(cameras: dict):
    """All learnable tensors across every view, for the optimizer."""
    params = []
    for cam in cameras.values():
        params.extend([cam.scale, cam.offset])
    return params
