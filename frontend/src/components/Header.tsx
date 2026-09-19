import { Link } from "react-router-dom";

export default function Header() {
  return (
    <header className="border-b border-ink/10">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
        <Link to="/" className="flex items-center gap-2">
          <span className="font-display text-xl tracking-tight">SizeSync</span>
          <span className="h-1.5 w-1.5 rounded-full bg-thread" aria-hidden />
        </Link>
        <nav className="font-mono text-xs uppercase tracking-widest text-ink/50">
          <span>01 · Body Profile</span>
        </nav>
      </div>
    </header>
  );
}
