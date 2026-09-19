export type ProfileStatus =
  | "uploaded"
  | "validating"
  | "validation_failed"
  | "ready_for_reconstruction"
  | "preprocessing"
  | "segmenting"
  | "detecting_pose"
  | "reconstructing"
  | "fitting_smplx"
  | "generating_mesh"
  | "completed"
  | "failed";

export interface ValidationIssue {
  code: string;
  message: string;
}

export interface ImageValidationResult {
  view: "front" | "side" | "back";
  passed: boolean;
  issues: ValidationIssue[];
  checks: Record<string, boolean | number>;
}

export interface ValidationReport {
  all_passed: boolean;
  results: ImageValidationResult[];
}

export interface Measurements {
  height_cm: number | null;
  shoulder_cm: number | null;
  chest_cm: number | null;
  waist_cm: number | null;
  hip_cm: number | null;
  arm_length_cm: number | null;
  inseam_cm: number | null;
  thigh_cm: number | null;
}

export interface BodyProfile {
  id: string;
  status: ProfileStatus;
  height_cm: number;
  validation_report: ValidationReport | null;
  measurements: Measurements;
  confidence_scores: Record<string, string> | null;
  mesh_url: string | null;
  reconstruction_meta: Record<string, unknown> | null;
}
