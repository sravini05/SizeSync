interface Props {
  valueCm: number | "";
  onChange: (v: number | "") => void;
}

export default function HeightInput({ valueCm, onChange }: Props) {
  const inches = typeof valueCm === "number" ? valueCm / 2.54 : null;
  const ft = inches !== null ? Math.floor(inches / 12) : null;
  const inRem = inches !== null ? Math.round(inches % 12) : null;

  return (
    <div className="rounded-2xl border border-ink/10 bg-white p-6">
      <label htmlFor="height" className="font-mono text-xs uppercase tracking-widest text-ink/50">
        Height
      </label>
      <div className="mt-2 flex items-baseline gap-2">
        <input
          id="height"
          type="number"
          inputMode="decimal"
          min={100}
          max={250}
          placeholder="172"
          value={valueCm}
          onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))}
          className="w-28 border-b-2 border-ink/20 bg-transparent font-display text-4xl tabular-nums outline-none focus:border-thread"
        />
        <span className="font-mono text-sm text-ink/40">cm</span>
      </div>
      {ft !== null && (
        <p className="mt-2 font-mono text-xs text-ink/40">
          ≈ {ft}′{inRem}″
        </p>
      )}
    </div>
  );
}
