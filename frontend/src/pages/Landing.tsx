import { Link } from "react-router-dom";
import Header from "../components/Header";
import MeasureRule from "../components/MeasureRule";

export default function Landing() {
  return (
    <div className="min-h-screen bg-mist">
      <Header />

      <main className="mx-auto grid max-w-6xl grid-cols-1 items-center gap-12 px-6 py-20 md:grid-cols-[80px_1fr]">
        <MeasureRule className="hidden h-[420px] w-[60px] text-ink md:block" />

        <div className="max-w-2xl">
          <p className="mb-4 font-mono text-xs uppercase tracking-[0.2em] text-thread">
            AI Body Measurement
          </p>
          <h1 className="font-display text-5xl leading-[1.05] tracking-tight text-ink md:text-6xl">
            Know your body.
            <br />
            <span className="italic text-ink/60">Know your size.</span>
          </h1>
          <p className="mt-6 max-w-lg text-lg leading-relaxed text-ink/70">
            Create your personalized body profile and find the right clothing
            size across brands — from three photos and your height, nothing
            else to measure by hand.
          </p>

          <div className="mt-10 flex items-center gap-4">
            <Link
              to="/create"
              className="group inline-flex items-center gap-3 rounded-full bg-ink px-7 py-4 font-body text-sm font-medium text-mist transition-colors hover:bg-thread"
            >
              Create My Size Profile
              <span className="transition-transform group-hover:translate-x-1">→</span>
            </Link>
            <span className="font-mono text-xs text-ink/40">
              takes about 2 minutes
            </span>
          </div>

          <dl className="mt-16 grid grid-cols-3 gap-6 border-t border-ink/10 pt-8">
            {[
              ["3", "photos needed"],
              ["8", "measurements extracted"],
              ["1", "body profile, every brand"],
            ].map(([n, label]) => (
              <div key={label}>
                <dt className="font-mono text-2xl text-ink">{n}</dt>
                <dd className="mt-1 text-xs text-ink/50">{label}</dd>
              </div>
            ))}
          </dl>
        </div>
      </main>
    </div>
  );
}
