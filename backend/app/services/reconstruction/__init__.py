from app.config import settings
from app.services.reconstruction.base import BodyReconstructionService
from app.services.reconstruction.prototype import NotYetImplementedReconstructionService


def get_reconstruction_service() -> BodyReconstructionService:
    """Factory so the rest of the app never imports a concrete backend
    directly -- swap RECONSTRUCTION_BACKEND (env var) to change engines.
    """
    backend = settings.RECONSTRUCTION_BACKEND
    if backend == "prototype":
        return NotYetImplementedReconstructionService()
    if backend == "smplx-multiview":
        # Imported lazily so `torch`/`smplx`/`human_body_prior` are only
        # required when this backend is actually selected -- the
        # "prototype" backend (and Stage 1 validation) still work without
        # them installed.
        from app.services.reconstruction.multiview_smplx import MultiViewSmplxReconstructionService
        return MultiViewSmplxReconstructionService()
    raise ValueError(f"Unknown SIZESYNC_RECONSTRUCTION_BACKEND: {backend}")
