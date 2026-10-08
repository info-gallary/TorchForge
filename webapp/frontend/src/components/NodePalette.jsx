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
    <div style={{ padding: 12, overflowY: "auto", height: "100%" }}>
      <h3 style={{ marginTop: 0 }}>Layers</h3>
      <p style={{ fontSize: 12, color: "#666" }}>Drag a layer onto the canvas.</p>
      {Object.entries(groups).map(([cat, items]) => (
        <div key={cat} style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 11, textTransform: "uppercase", color: "#888", marginBottom: 6 }}>
            {CATEGORY_LABELS[cat] || cat}
          </div>
          {items.map((item) => (
            <div
              key={item.type}
              draggable
              onDragStart={(e) => onDragStart(e, item.type)}
              style={{
                padding: "6px 10px",
                marginBottom: 6,
                background: "#f1f5f9",
                border: "1px solid #cbd5e1",
                borderRadius: 6,
                cursor: "grab",
                fontSize: 13,
              }}
            >
              {item.label}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
