# SizeSync — Stage 1 Prototype

AI-powered clothing size platform. This is the **first build stage only**:

```
Landing → Create Profile (front/side/back photos + height) → Upload → Validation → Status
```

Body reconstruction, the 3D mesh, measurement extraction, confidence scores,
and the size profile dashboard are **not built yet** — those are Phases 5-11,
the next stage. Building them now would mean either faking measurements or
shipping a reconstruction pipeline nobody's reviewed yet; the spec explicitly
asks to avoid both, so this stage stops right after validation, with an
honest "not implemented yet" placeholder wired into the architecture where
reconstruction will plug in.

## What's actually implemented

- **Frontend**: landing page, photo+height upload form, live validation results screen (React/TS/Tailwind).
- **Backend**: FastAPI service, SQLite-backed `User`/`BodyProfile` models, `POST /api/profiles` + `GET /api/profiles/{id}`.
- **Validation (real, not mocked)**: blur detection, brightness/exposure, resolution, person detection, multiple-people detection, and framing/cropping checks — all running against actual pixel data via OpenCV, with an optional stricter MediaPipe Pose check.
- **Reconstruction architecture (interface only)**: `BodyReconstructionService` abstract class + a placeholder backend that returns an explicit "not implemented" result rather than fabricated measurements, so Phase 5/6 can plug in a real SMPL-X-compatible fitter later without changing anything above it.

## Run it

Two terminals:

```bash
# Terminal 1
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000

# Terminal 2
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173.

See `backend/README.md` and `frontend/README.md` for details, including the
optional MediaPipe Pose model download for stricter validation.

## A note on this sandbox

This code was written and syntax-checked in a sandboxed environment without
outbound network access, so `pip install` / `npm install` couldn't be run
here to fully compile/execute the apps end-to-end. What I *could* verify
here:
- All Python files compile (`python3 -m py_compile`) with no syntax errors.
- The validation module was actually executed offline against a synthetic
  test image and correctly rejected it (no person detected) — the checks are
  real, not stubbed.
- All frontend files pass a bracket/paren balance check; `npm install` itself
  needs to happen on your machine.

Please run through the "what to test" checklists in each README on your
machine to confirm everything works end-to-end — flag anything broken and
I'll fix it before we move to Phase 5 (reconstruction).

## Next stage (Phase 5-6)

Once this stage checks out, next is implementing an actual
`BodyReconstructionService` backend: image preprocessing → segmentation →
pose/keypoint detection → multi-view body reconstruction → SMPL-X parameter
fitting → `.glb` mesh export → real-world scale calibration using the
user-entered height. I'll flag up front what reconstruction approach is
realistically runnable in this environment vs. what needs a GPU/cloud
service, rather than quietly downgrading the result.
