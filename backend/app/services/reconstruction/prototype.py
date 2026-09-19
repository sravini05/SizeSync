"""
NotYetImplementedReconstructionService

Per Section 6 / Section 18 of the spec: we do NOT hardcode fake measurements
and present them as AI-generated. The actual SMPL-X-compatible multi-view
fitting pipeline is Phase 5/6 of the build and is not implemented yet.

This class exists so that:
  - the API layer has something to call against BodyReconstructionService
    right now, and
  - the failure mode is explicit and visible to the user (clearly marked
    "not implemented" error) rather than a silently fabricated result.

When Phase 5/6 lands, a new class (e.g. SmplxMultiViewReconstructionService)
implementing the same BodyReconstructionService interface will replace the
one wired up in app/services/reconstruction/__init__.py.
"""
from __future__ import annotations

from typing import Optional

from app.services.reconstruction.base import (
    BodyReconstructionService,
    ReconstructionInput,
    ReconstructionResult,
)


class NotYetImplementedReconstructionService(BodyReconstructionService):
    def reconstruct(self, data: ReconstructionInput, on_stage: Optional[callable] = None) -> ReconstructionResult:
        return ReconstructionResult(
            success=False,
            is_prototype=True,
            method="not-implemented",
            error=(
                "Body reconstruction is not implemented yet in this build. "
                "This stage only covers photo upload and validation "
                "(Phases 1-4). Reconstruction ships in the next stage."
            ),
        )
