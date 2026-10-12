import { categoryColor } from "../categoryColors";

const CATEGORY_LABELS = {
  conv: "Convolution",
  pool: "Pooling",
  norm: "Normalization",
  activation: "Activation",
  regularization: "Regularization",
  shape: "Shape",
  dense: "Dense",
  output: "Output",
  merge: "Merge (branches)",
};

function groupByCategory(layerSpecs) {
  const groups = {};
  for (const [type, spec] of Object.entries(layerSpecs)) {
    const cat = spec.category || "other";
    if (!groups[cat]) groups[cat] = [];
    groups[cat].push({ type, ...spec });
  }
  return groups;
}

export default function NodePalette({ layerSpecs }) {
  const groups = groupByCategory(layerSpecs);

  const onDragStart = (event, layerType) => {
    event.dataTransfer.setData("application/vt-layer-type", layerType);
    event.dataTransfer.effectAllowed = "move";
  };

  return (
    <div style={{ padding: 16, overflowY: "auto", height: "100%" }}>
      <h3 style={{ margin: "0 0 2px", fontSize: 14, fontWeight: 700 }}>Layers</h3>
      <p style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 4, marginBottom: 18 }}>
        Drag a layer onto the canvas.
      </p>
      {Object.entries(groups).map(([cat, items]) => (
        <div key={cat} style={{ marginBottom: 16 }}>
          <div style={{
            fontSize: 10.5, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em",
            color: "var(--text-faint)", marginBottom: 7,
          }}>
            {CATEGORY_LABELS[cat] || cat}
          </div>
          {items.map((item) => {
            const color = categoryColor(item.category);
            return (
              <div
                key={item.type}
                draggable
                onDragStart={(e) => onDragStart(e, item.type)}
                style={{
                  display: "flex", alignItems: "center", gap: 8,
                  padding: "8px 10px", marginBottom: 6,
                  background: "var(--surface)", border: "1px solid var(--border)",
                  borderLeft: `3px solid ${color}`, borderRadius: 7,
                  cursor: "grab", fontSize: 13, color: "var(--text)",
                  transition: "border-color 0.12s, transform 0.12s",
                }}
                onMouseEnter={(e) => { e.currentTarget.style.borderColor = color; }}
                onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--border)"; e.currentTarget.style.borderLeftColor = color; }}
              >
                {item.label}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}
