"""
Stage 5.6/5.7 -- Height calibration + .glb export.

SMPL-X outputs vertices in its own canonical unit scale (meters, roughly --
the exact height depends on the fitted beta/pose). We calibrate to the
user's real height by measuring the fitted mesh's own vertical extent
(head to heel, using SMPL-X's head and foot joints) and applying a single
uniform scale factor so that measured extent equals the entered height.
This is what makes every downstream measurement (Stage 3) meaningful in
real-world centimeters rather than an arbitrary model unit.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

logger = logging.getLogger("sizesync.reconstruction.mesh_export")

# SMPL-X joint indices (see keypoints.py docstring for the full table)
HEAD_JOINT = 15
LEFT_FOOT_JOINT = 10
RIGHT_FOOT_JOINT = 11


def compute_scale_factor(joints: np.ndarray, height_cm: float) -> float:
    """Returns the multiplier to apply to vertices/joints so the mesh's own
    head-to-heel extent matches the user's entered height."""
    head_y = joints[HEAD_JOINT][1]
    foot_y = min(joints[LEFT_FOOT_JOINT][1], joints[RIGHT_FOOT_JOINT][1])
    mesh_height_units = float(head_y - foot_y)

    if mesh_height_units <= 1e-4:
        raise ValueError(
            f"Fitted mesh has an implausible vertical extent ({mesh_height_units:.4f} units) "
            "-- reconstruction likely failed to converge."
        )

    target_m = height_cm / 100.0
    return target_m / mesh_height_units


def export_glb(vertices: np.ndarray, faces: np.ndarray, scale_factor: float, out_path: Path) -> Path:
    """Applies the height-calibration scale, re-centers the mesh at the
    origin (feet at y=0), and writes a .glb file."""
    scaled = vertices * scale_factor

    min_y = scaled[:, 1].min()
    center_x = (scaled[:, 0].max() + scaled[:, 0].min()) / 2.0
    center_z = (scaled[:, 2].max() + scaled[:, 2].min()) / 2.0
    scaled = scaled - np.array([center_x, min_y, center_z])

    mesh = trimesh.Trimesh(vertices=scaled, faces=faces, process=False)
    mesh.visual.vertex_colors = np.tile([220, 200, 180, 255], (scaled.shape[0], 1))  # neutral skin-tone-ish grey

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path), file_type="glb")
    logger.info("Exported mesh to %s (%d vertices)", out_path, len(scaled))
    return out_path
