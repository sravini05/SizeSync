import { useRef, useState } from "react";

interface Props {
  view: "front" | "side" | "back";
  label: string;
  hint: string;
  file: File | null;
  onChange: (file: File | null) => void;
}

export default function PhotoUploadCard({ view, label, hint, file, onChange }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);

  function handleFile(f: File | null) {
    onChange(f);
    if (f) setPreview(URL.createObjectURL(f));
    else setPreview(null);
  }

  return (
    <div className="flex flex-col">
      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
        className={`relative flex aspect-[3/4] cursor-pointer flex-col items-center justify-center overflow-hidden rounded-2xl border-2 border-dashed transition-colors ${
          file ? "border-sage bg-sage/5" : "border-ink/20 bg-white hover:border-ink/40"
        }`}
      >
        {preview ? (
          <img src={preview} alt={`${label} preview`} className="h-full w-full object-cover" />
        ) : (
          <div className="flex flex-col items-center gap-2 px-4 text-center">
            <span className="font-mono text-2xl text-ink/30">+</span>
            <span className="font-mono text-xs uppercase tracking-widest text-ink/40">
              {label}
            </span>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          className="hidden"
          data-testid={`upload-${view}`}
          onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
        />
        {file && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              handleFile(null);
              if (inputRef.current) inputRef.current.value = "";
            }}
            className="absolute right-2 top-2 rounded-full bg-ink/80 px-2 py-1 font-mono text-[10px] text-mist"
          >
            remove
          </button>
        )}
      </div>
      <p className="mt-2 text-xs text-ink/50">{hint}</p>
    </div>
  );
}
