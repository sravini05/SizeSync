from typing import Optional, Dict, List
from pydantic import BaseModel, Field


class ImageValidationIssue(BaseModel):
    code: str
    message: str


class ImageValidationResult(BaseModel):
    view: str  # "front" | "side" | "back"
    passed: bool
    issues: List[ImageValidationIssue] = Field(default_factory=list)
    checks: Dict[str, bool] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    all_passed: bool
    results: List[ImageValidationResult]


class MeasurementsOut(BaseModel):
    height_cm: Optional[float] = None
    shoulder_cm: Optional[float] = None
    chest_cm: Optional[float] = None
    waist_cm: Optional[float] = None
    hip_cm: Optional[float] = None
    arm_length_cm: Optional[float] = None
    inseam_cm: Optional[float] = None
    thigh_cm: Optional[float] = None


class BodyProfileOut(BaseModel):
    id: str
    status: str
    height_cm: float
    validation_report: Optional[ValidationReport] = None
    measurements: MeasurementsOut
    confidence_scores: Optional[Dict[str, float | str]] = None
    mesh_url: Optional[str] = None
    reconstruction_meta: Optional[dict] = None

    class Config:
        from_attributes = True
