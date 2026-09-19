interface Props {
  className?: string;
  labelEvery?: number;
}

/** A vertical tape-measure rule -- the page's signature element. Ticks get
 * taller every 5 units and carry a cm label every `labelEvery` units, the
 * way an actual dressmaker's tape does. */
export default function MeasureRule({ className = "", labelEvery = 10 }: Props) {
  const ticks = Array.from({ length: 21 }, (_, i) => i);

  return (
    <svg
      viewBox="0 0 60 400"
      className={className}
      role="img"
      aria-label="Decorative measuring tape"
    >
      <line x1="30" y1="0" x2="30" y2="400" stroke="currentColor" strokeOpacity="0.25" strokeWidth="1" />
      {ticks.map((i) => {
        const y = i * 20;
        const isMajor = i % 5 === 0;
        const isLabeled = i % (labelEvery / 2) === 0;
        const len = isMajor ? 18 : 9;
        return (
          <g key={i}>
            <line
              x1={30 - len / 2}
              x2={30 + len / 2}
              y1={y}
              y2={y}
              stroke="currentColor"
              strokeOpacity={isMajor ? 0.8 : 0.35}
              strokeWidth={isMajor ? 1.5 : 1}
            />
            {isMajor && isLabeled && (
              <text
                x={44}
                y={y + 3}
                fontSize="9"
                fontFamily="'IBM Plex Mono', monospace"
                fill="currentColor"
                opacity="0.5"
              >
                {i * 5}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
