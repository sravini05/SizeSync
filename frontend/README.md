# SizeSync Frontend (Stage 1: upload + validation UI)

React + TypeScript + Vite + Tailwind. Covers the landing page, body profile
creation flow (photo upload + height), and a processing screen that shows
per-photo validation results. The 3D viewer and measurement dashboard land in
a later stage once backend reconstruction (Phase 5+) exists.

## Setup

```bash
cd frontend
npm install
```

## Run

```bash
npm run dev
```

Visit http://localhost:5173. The Vite dev server proxies `/api` to
`http://localhost:8000`, so start the backend first (see `../backend/README.md`).

## What to test

1. Landing page loads at `/` with the "Create My Size Profile" CTA.
2. `/create` — upload three photos and a height, submit. Button is disabled until all four fields are filled.
3. On submit, you're routed to `/profiles/:id` showing per-photo validation results (pass/fail + reasons).
4. Upload a bad photo (e.g. no person, or very dark) — confirm the failure reason is specific and there's a "Retake photos" button back to `/create`.
5. Upload three good, full-body photos — confirm all three show "Passed" and the status explains reconstruction is the next build stage.

## Project layout

```
src/
  main.tsx / App.tsx      entry point + routes
  pages/
    Landing.tsx              Section 3 landing page
    CreateProfile.tsx          Section 4 upload + height form
    Processing.tsx              Section 12 validation results / status
  components/
    Header.tsx, MeasureRule.tsx, PhotoUploadCard.tsx, HeightInput.tsx
  api/client.ts             typed fetch wrapper for the backend API
  types.ts                    types mirroring backend Pydantic schemas
```
