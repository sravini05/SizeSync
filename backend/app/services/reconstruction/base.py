"""
BodyReconstructionService -- the replaceable AI reconstruction abstraction
required by Section 6 of the spec.

This file defines the CONTRACT only. It is wired up and implemented starting
in Phase 5/6 of the build (SMPL-X-compatible multi-view fitting). It exists
now so the rest of the system (DB schema, API routes, frontend types) can be
built against a stable interface without depending on the reconstruction
internals.

Do not add a fake/stub implementation that returns plausible-looking
measurements here -- see prototype.py for why, and how "not implemented yet"
is surfaced honestly to the API layer instead.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ReconstructionInput:
    front_image_path: str
    side_image_path: str
    back_image_path: str
    height_cm: float


@dataclass
class Measurements:
    shoulder_cm: Optional[float] = None
    chest_cm: Optional[float] = None
    waist_cm: Optional[float] = None
    hip_cm: Optional[float] = None
    arm_length_cm: Optional[float] = None
    inseam_cm: Optional[float] = None
    thigh_cm: Optional[float] = None


@dataclass
class ReconstructionResult:
    success: bool
    mesh_path: Optional[str] = None                 # local path to a .glb file
    measurements: Optional[Measurements] = None
    confidence_scores: dict = field(default_factory=dict)  # {"chest": ConfidenceLevel, ...}
    method: str = "unknown"                          # e.g. "smplx-multiview-v0"
    is_prototype: bool = True                         # must stay True until a real
                                                        # multi-view SMPL-X fit ships
    error: Optional[str] = None


class BodyReconstructionService(ABC):
    """Contract every reconstruction backend must implement.

    Swapping in a stronger model later (e.g. a real multi-view SMPL-X fitter)
    means writing one new class here -- nothing above this layer (API routes,
    DB models, frontend) should need to change.
    """

    @abstractmethod
    def reconstruct(
        self,
        data: ReconstructionInput,
        on_stage: Optional[callable] = None,
    ) -> ReconstructionResult:
        """Run the full pipeline: preprocessing -> segmentation -> pose/keypoints
        -> multi-view fitting -> SMPL-X parameter estimation -> mesh export ->
        real-world scale calibration -> measurement extraction.

        `on_stage`, if given, is called as on_stage("stage_name") each time
        the pipeline advances to a new named stage (matching ProfileStatus
        values), so the caller can persist live progress for polling clients.
        This is additive/optional -- existing callers that don't pass it are
        unaffected.

        Must never return fabricated measurements. If the backend cannot
        produce a real result, return ReconstructionResult(success=False,
        error=...) instead of guessing.
        """
        raise NotImplementedError
