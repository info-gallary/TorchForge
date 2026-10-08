const WIDTH = 480;
const HEIGHT = 160;
const PAD = 28;

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
export default function LiveChart({ title, series, color = "#2563eb" }) {
  const values = series.map((p) => p.value);
  const minV = values.length ? Math.min(...values) : 0;
  const maxV = values.length ? Math.max(...values) : 1;
  const path = buildPath(values, minV, maxV);

  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 12, color: "#555", marginBottom: 4 }}>
        {title} {values.length > 0 && <span style={{ color: "#999" }}>(latest: {values[values.length - 1].toFixed(4)})</span>}
      </div>
      <svg width={WIDTH} height={HEIGHT} style={{ background: "#fafafa", border: "1px solid #e5e7eb", borderRadius: 6 }}>
        <line x1={PAD} y1={HEIGHT - PAD} x2={WIDTH - PAD} y2={HEIGHT - PAD} stroke="#ccc" />
        <line x1={PAD} y1={PAD} x2={PAD} y2={HEIGHT - PAD} stroke="#ccc" />
        {path && <path d={path} fill="none" stroke={color} strokeWidth={2} />}
        <text x={PAD} y={14} fontSize={10} fill="#999">{maxV.toFixed(3)}</text>
        <text x={PAD} y={HEIGHT - PAD + 12} fontSize={10} fill="#999">{minV.toFixed(3)}</text>
      </svg>
    </div>
  );
}
