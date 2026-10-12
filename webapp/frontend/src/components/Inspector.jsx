import { categoryColor } from "../categoryColors";

export default function Inspector({ selectedNode, layerSpecs, onChangeParam, onDelete }) {
  if (!selectedNode) {
    return (
      <div style={{ padding: 16, color: "var(--text-faint)", fontSize: 13 }}>
        Click a layer on the canvas to edit its settings.
      </div>
    );
  }

  const spec = layerSpecs[selectedNode.data.layerType];
  const paramDefs = Object.entries(spec?.params || {});
  const color = categoryColor(selectedNode.data.category);

  return (
    <div style={{ padding: 16 }}>
      <h3 style={{ margin: "0 0 14px", fontSize: 14, fontWeight: 700, display: "flex", alignItems: "center", gap: 7 }}>
        <span style={{ width: 8, height: 8, borderRadius: "50%", background: color }} />
        {selectedNode.data.label}
      </h3>
      {paramDefs.length === 0 && (
        <p style={{ fontSize: 12.5, color: "var(--text-faint)" }}>This layer has no settings.</p>
      )}
      {paramDefs.map(([key, def]) => (
        <label key={key} className="field-label">
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
          />
        </label>
      ))}
      <button
        className="btn"
        onClick={() => onDelete(selectedNode.id)}
        style={{ marginTop: 8, color: "var(--danger)", borderColor: "rgba(248,113,113,0.3)" }}
      >
        Delete layer
      </button>
    </div>
  );
}
