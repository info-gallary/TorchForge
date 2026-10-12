const WIDTH = 480;
const HEIGHT = 140;
const PAD = 26;

function buildPath(values, minV, maxV) {
  if (values.length === 0) return "";
  const range = maxV - minV || 1;
  const stepX = values.length > 1 ? (WIDTH - 2 * PAD) / (values.length - 1) : 0;
  return values
    .map((v, i) => {
      const x = PAD + i * stepX;
      const y = HEIGHT - PAD - ((v - minV) / range) * (HEIGHT - 2 * PAD);
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

/** A minimal inline-SVG line chart -- no charting library needed for one
 * or two series of a few dozen points. */
export default function LiveChart({ title, series, color = "var(--accent)" }) {
  const values = series.map((p) => p.value);
  const minV = values.length ? Math.min(...values) : 0;
  const maxV = values.length ? Math.max(...values) : 1;
  const path = buildPath(values, minV, maxV);
  const areaPath = path ? `${path} L${WIDTH - PAD},${HEIGHT - PAD} L${PAD},${HEIGHT - PAD} Z` : "";
  const gradId = `grad-${title.replace(/\s+/g, "-")}`;

  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 6 }}>
        {title} {values.length > 0 && (
          <span className="mono" style={{ color: "var(--text-faint)" }}>(latest: {values[values.length - 1].toFixed(4)})</span>
        )}
      </div>
      <svg width="100%" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} style={{ background: "var(--panel-solid)", border: "1px solid var(--border)", borderRadius: 9, display: "block" }}>
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.35" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        <line x1={PAD} y1={HEIGHT - PAD} x2={WIDTH - PAD} y2={HEIGHT - PAD} stroke="var(--border)" />
        <line x1={PAD} y1={PAD} x2={PAD} y2={HEIGHT - PAD} stroke="var(--border)" />
        {areaPath && <path d={areaPath} fill={`url(#${gradId})`} stroke="none" />}
        {path && <path d={path} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />}
        <text x={PAD} y={14} fontSize={10} fill="var(--text-faint)">{maxV.toFixed(3)}</text>
        <text x={PAD} y={HEIGHT - PAD + 13} fontSize={10} fill="var(--text-faint)">{minV.toFixed(3)}</text>
      </svg>
    </div>
  );
}
