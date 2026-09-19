# SizeSync Backend (Stage 1: upload + validation)

FastAPI service covering Phases 1-4 of the build order: receive front/side/back
photos + height, run real photo-quality validation, and store the result.
**Body reconstruction (Phase 5+) is not implemented yet** — see
`app/services/reconstruction/prototype.py` for why, and how that's surfaced
honestly instead of faking measurements.

## Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Optional: enable the stricter pose-based validation check

The base checks (blur, brightness, resolution, person detection/framing,
multiple-people) work out of the box using OpenCV only. For the stricter
"head and feet both clearly visible" check, download the MediaPipe Pose
Landmarker model once:

```bash
mkdir -p models
curl -L -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

If you skip this, validation still works — it just relies on the OpenCV
framing heuristic alone, and logs a warning saying so.

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs
Health check: http://localhost:8000/api/health

## API

- `POST /api/profiles` — multipart form: `client_ref`, `height_cm`, `front`, `side`, `back` files. Runs validation synchronously and returns the profile with a `validation_report`.
- `GET /api/profiles/{id}` — fetch current profile/status.

## What to test

1. Start the backend, hit `/api/health` — should return `{"status": "ok", ...}`.
2. Open `/docs` and POST to `/api/profiles` with three real photos of a person and a height — check `validation_report` per-photo pass/fail and messages make sense.
3. Try a photo with no person in it (e.g. a landscape) — should fail with `no_person_detected`.
4. Try a very dark or very blurry photo — should fail with the matching issue code.
5. Try a heavily cropped photo (person's feet cut off) — should fail `possibly_cropped` (HOG heuristic) or `feet_not_visible` (if the pose model is installed).
6. Confirm `GET /api/profiles/{id}` returns the same data afterward.

## Project layout

```
app/
  main.py            FastAPI app, CORS, router registration
  config.py           env-based settings
  database.py          SQLAlchemy engine/session
  models.py             User, BodyProfile ORM models
  schemas.py             Pydantic request/response models
  routers/profile.py      POST/GET /api/profiles
  services/
    validation.py           Section 12 photo quality checks (real, not mocked)
    reconstruction/
      base.py                 BodyReconstructionService interface (Section 6)
      prototype.py             honest "not implemented yet" backend
      __init__.py                factory / backend selection
```
