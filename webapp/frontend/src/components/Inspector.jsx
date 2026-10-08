export default function Inspector({ selectedNode, layerSpecs, onChangeParam, onDelete }) {
  if (!selectedNode) {
    return (
      <div style={{ padding: 12, color: "#888", fontSize: 13 }}>
        Click a layer on the canvas to edit its settings.
      </div>
    );
  }

  const spec = layerSpecs[selectedNode.data.layerType];
  const paramDefs = Object.entries(spec?.params || {});

  return (
    <div style={{ padding: 12 }}>
      <h3 style={{ marginTop: 0 }}>{selectedNode.data.label}</h3>
      {paramDefs.length === 0 && (
        <p style={{ fontSize: 12, color: "#888" }}>This layer has no settings.</p>
      )}
      {paramDefs.map(([key, def]) => (
        <label key={key} style={{ display: "block", marginBottom: 10, fontSize: 13 }}>
          {key}
          <input
            type="number"
            step={def.type === "float" ? "0.001" : "1"}
            min={def.min}
            max={def.max}
            value={selectedNode.data.params[key] ?? def.default}
            onChange={(e) => {
              const raw = e.target.value;
              const value = def.type === "float" ? parseFloat(raw) : parseInt(raw, 10);
              onChangeParam(selectedNode.id, key, Number.isNaN(value) ? def.default : value);
            }}
            style={{ display: "block", width: "100%", marginTop: 4, padding: 4 }}
          />
        </label>
      ))}
      <button
        onClick={() => onDelete(selectedNode.id)}
        style={{
          marginTop: 8,
          padding: "6px 12px",
          background: "#fee2e2",
          border: "1px solid #fca5a5",
          borderRadius: 6,
          cursor: "pointer",
        }}
      >
        Delete layer
      </button>
    </div>
  );
}
