import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Header from "../components/Header";
import PhotoUploadCard from "../components/PhotoUploadCard";
import HeightInput from "../components/HeightInput";
import { createProfile, ApiError } from "../api/client";

const GUIDELINES = [
  "Full body visible, head to feet",
  "Simple, uncluttered background",
  "Stand naturally, arms relaxed at your sides",
  "Wear tight or normal-fit clothing",
  "Camera at roughly waist/chest height",
  "Good, even lighting — avoid strong backlight",
];

export default function CreateProfile() {
  const navigate = useNavigate();
  const [front, setFront] = useState<File | null>(null);
  const [side, setSide] = useState<File | null>(null);
  const [back, setBack] = useState<File | null>(null);
  const [height, setHeight] = useState<number | "">("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = front && side && back && height !== "" && !submitting;

  async function handleSubmit() {
    if (!canSubmit || !front || !side || !back || height === "") return;
    setSubmitting(true);
    setError(null);
    try {
      const profile = await createProfile({ heightCm: height, front, side, back });
      navigate(`/profiles/${profile.id}`);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Please try again.");
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen bg-mist">
      <Header />
      <main className="mx-auto max-w-4xl px-6 py-14">
        <h1 className="font-display text-3xl tracking-tight">Create your body profile</h1>
        <p className="mt-2 text-ink/60">
          Three photos and your height — that's all we need. No manual
          measurements.
        </p>

        <section className="mt-8 rounded-2xl border border-ink/10 bg-white p-6">
          <h2 className="font-mono text-xs uppercase tracking-widest text-ink/50">
            Before you start
          </h2>
          <ul className="mt-3 grid grid-cols-1 gap-x-8 gap-y-2 text-sm text-ink/70 sm:grid-cols-2">
            {GUIDELINES.map((g) => (
              <li key={g} className="flex gap-2">
                <span className="text-thread">·</span>
                {g}
              </li>
            ))}
          </ul>
        </section>

        <section className="mt-8 grid grid-cols-1 gap-6 sm:grid-cols-3">
          <PhotoUploadCard view="front" label="Front" hint="Face the camera directly." file={front} onChange={setFront} />
          <PhotoUploadCard view="side" label="Side" hint="Turn 90°, arms relaxed." file={side} onChange={setSide} />
          <PhotoUploadCard view="back" label="Back" hint="Face away from the camera." file={back} onChange={setBack} />
        </section>

        <section className="mt-6 max-w-xs">
          <HeightInput valueCm={height} onChange={setHeight} />
        </section>

        {error && (
          <p className="mt-6 rounded-lg bg-thread/10 px-4 py-3 text-sm text-thread">{error}</p>
        )}

        <div className="mt-10">
          <button
            type="button"
            disabled={!canSubmit}
            onClick={handleSubmit}
            className="rounded-full bg-ink px-8 py-4 text-sm font-medium text-mist transition-colors enabled:hover:bg-thread disabled:cursor-not-allowed disabled:opacity-30"
          >
            {submitting ? "Uploading & validating…" : "Build My Profile"}
          </button>
        </div>
      </main>
    </div>
  );
}
