import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import Header from "../components/Header";
import MeshViewer from "../components/MeshViewer";
import { getProfile, startReconstruction, ApiError } from "../api/client";
import type { BodyProfile, ProfileStatus } from "../types";

const IN_PROGRESS_STATUSES = new Set<ProfileStatus>([
  "uploaded",
  "validating",
  "preprocessing",
  "segmenting",
  "detecting_pose",
  "reconstructing",
  "fitting_smplx",
  "generating_mesh",
]);

const VIEW_LABELS: Record<string, string> = { front: "Front", side: "Side", back: "Back" };

// The six user-facing phases from the spec, each mapped from one or more
// backend statuses (segmenting + detecting_pose both read as "Understanding
// body shape" -- the user doesn't need the sub-stage distinction).
const PIPELINE_STEPS: { label: string; statuses: ProfileStatus[] }[] = [
  { label: "Preparing photos", statuses: ["preprocessing"] },
  { label: "Understanding body shape", statuses: ["segmenting", "detecting_pose"] },
  { label: "Building 3D body", statuses: ["reconstructing"] },
  { label: "Fitting body model", statuses: ["fitting_smplx"] },
  { label: "Generating 3D model", statuses: ["generating_mesh"] },
  { label: "Complete", statuses: ["completed"] },
];

function currentStepIndex(status: ProfileStatus): number {
  return PIPELINE_STEPS.findIndex((s) => s.statuses.includes(status)); // -1 if not a pipeline step
}

export default function Processing() {
  const { id } = useParams<{ id: string }>();
  const [profile, setProfile] = useState<BodyProfile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retrying, setRetrying] = useState(false);
  const triggeredRef = useRef(false);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    async function poll() {
      try {
        const p = await getProfile(id!);
        if (cancelled) return;
        setProfile(p);

        // Auto-start reconstruction the first time we see a validated,
        // not-yet-started profile -- the user already asked for this by
        // completing upload; no extra click needed.
        if (p.status === "ready_for_reconstruction" && !triggeredRef.current) {
          triggeredRef.current = true;
          try {
            await startReconstruction(id!);
          } catch {
            // If kicking it off fails, keep polling -- status still
            // reflects reality and the user can retry manually.
            triggeredRef.current = false;
          }
        }

        const stillGoing = IN_PROGRESS_STATUSES.has(p.status) || p.status === "ready_for_reconstruction";
        if (stillGoing) {
          timer = setTimeout(poll, 1500);
        }
      } catch {
        if (!cancelled) setError("Couldn't load this profile.");
      }
    }
    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [id]);

  async function handleRetry() {
    if (!id) return;
    setRetrying(true);
    try {
      const p = await startReconstruction(id);
      setProfile(p);
      triggeredRef.current = true;
      setTimeout(async () => {
        const fresh = await getProfile(id);
        setProfile(fresh);
      }, 1500);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Couldn't restart reconstruction.");
    } finally {
      setRetrying(false);
    }
  }

  return (
    <div className="min-h-screen bg-mist">
      <Header />
      <main className="mx-auto max-w-3xl px-6 py-14">
        {error && <p className="text-thread">{error}</p>}

        {!profile && !error && <p className="font-mono text-sm text-ink/50">Loading…</p>}

        {profile && (
          <>
            <StatusBanner profile={profile} />

            {profile.validation_report && (
              <section className="mt-8 space-y-4">
                <h2 className="font-mono text-xs uppercase tracking-widest text-ink/50">
                  Photo validation
                </h2>
                {profile.validation_report.results.map((r) => (
                  <div
                    key={r.view}
                    className={`rounded-2xl border p-5 ${
                      r.passed ? "border-sage/30 bg-sage/5" : "border-thread/30 bg-thread/5"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-display text-lg">{VIEW_LABELS[r.view] ?? r.view}</span>
                      <span
                        className={`font-mono text-xs uppercase tracking-widest ${
                          r.passed ? "text-sage" : "text-thread"
                        }`}
                      >
                        {r.passed ? "Passed" : "Needs a retake"}
                      </span>
                    </div>
                    {r.issues.length > 0 && (
                      <ul className="mt-3 space-y-1 text-sm text-ink/70">
                        {r.issues.map((issue) => (
                          <li key={issue.code}>· {issue.message}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
              </section>
            )}

            {profile.status === "validation_failed" && (
              <div className="mt-8">
                <Link
                  to="/create"
                  className="inline-block rounded-full bg-ink px-7 py-3 text-sm font-medium text-mist hover:bg-thread"
                >
                  Retake photos
                </Link>
              </div>
            )}

            {(IN_PROGRESS_STATUSES.has(profile.status) || profile.status === "ready_for_reconstruction") && (
              <PipelineStepper status={profile.status} />
            )}

            {profile.status === "failed" && (
              <div className="mt-8 space-y-4">
                <div className="rounded-2xl border border-thread/30 bg-thread/5 p-5 text-sm text-ink/70">
                  {String(profile.reconstruction_meta?.error ?? "Something went wrong during reconstruction.")}
                </div>
                <button
                  type="button"
                  disabled={retrying}
                  onClick={handleRetry}
                  className="rounded-full bg-ink px-7 py-3 text-sm font-medium text-mist transition-colors enabled:hover:bg-thread disabled:opacity-40"
                >
                  {retrying ? "Retrying…" : "Try reconstruction again"}
                </button>
              </div>
            )}

            {profile.status === "completed" && profile.mesh_url && (
              <section className="mt-8">
                <h2 className="mb-3 font-mono text-xs uppercase tracking-widest text-ink/50">
                  Your 3D body
                </h2>
                <MeshViewer meshUrl={profile.mesh_url} />
                {profile.confidence_scores?.overall_fit && (
                  <p className="mt-3 font-mono text-xs text-ink/40">
                    Fit confidence: {profile.confidence_scores.overall_fit}
                  </p>
                )}
                <p className="mt-4 text-sm text-ink/50">
                  Measurements aren't extracted yet — that's the next build stage.
                </p>
              </section>
            )}
          </>
        )}
      </main>
    </div>
  );
}

function PipelineStepper({ status }: { status: ProfileStatus }) {
  const activeIdx = currentStepIndex(status);
  return (
    <section className="mt-10">
      <ol className="space-y-3">
        {PIPELINE_STEPS.map((step, i) => {
          const done = activeIdx !== -1 && activeIdx > i;
          const active = i === activeIdx;
          return (
            <li key={step.label} className="flex items-center gap-3">
              <span
                className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full font-mono text-[11px] ${
                  done
                    ? "bg-sage text-mist"
                    : active
                      ? "bg-thread text-mist"
                      : "bg-ink/10 text-ink/40"
                }`}
              >
                {done ? "✓" : i + 1}
              </span>
              <span className={active ? "font-medium text-ink" : done ? "text-ink/60" : "text-ink/40"}>
                {step.label}
              </span>
              {active && (
                <span className="ml-1 h-1.5 w-1.5 animate-pulse rounded-full bg-thread" aria-hidden />
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function StatusBanner({ profile }: { profile: BodyProfile }) {
  const copy: Record<string, { title: string; body: string }> = {
    uploaded: { title: "Photos received", body: "Preparing to validate your photos…" },
    validating: { title: "Validating your photos", body: "Checking framing, lighting, and focus…" },
    validation_failed: {
      title: "A couple of photos need a retake",
      body: "Fix the issues below and upload again for the best results.",
    },
    ready_for_reconstruction: {
      title: "Photos look good",
      body: "Starting 3D body reconstruction…",
    },
    preprocessing: { title: "Building your 3D body model", body: "Preparing your photos…" },
    segmenting: { title: "Building your 3D body model", body: "Understanding your body shape…" },
    detecting_pose: { title: "Building your 3D body model", body: "Understanding your body shape…" },
    reconstructing: { title: "Building your 3D body model", body: "Combining your three photos…" },
    fitting_smplx: { title: "Building your 3D body model", body: "Fitting your body model…" },
    generating_mesh: { title: "Building your 3D body model", body: "Generating your 3D model…" },
    completed: { title: "Your 3D body is ready", body: "Rotate, zoom, and pan below." },
    failed: { title: "Reconstruction didn't finish", body: "" },
  };
  const c = copy[profile.status] ?? { title: profile.status, body: "" };

  return (
    <div>
      <p className="font-mono text-xs uppercase tracking-widest text-thread">
        Height: {profile.height_cm} cm
      </p>
      <h1 className="mt-2 font-display text-3xl tracking-tight">{c.title}</h1>
      {c.body && <p className="mt-2 max-w-xl text-ink/60">{c.body}</p>}
    </div>
  );
}
