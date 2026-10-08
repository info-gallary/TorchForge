import { Handle, Position } from "@xyflow/react";

const CATEGORY_COLORS = {
  conv: "#3b82f6",
  pool: "#8b5cf6",
  norm: "#06b6d4",
  activation: "#f59e0b",
  regularization: "#ef4444",
  shape: "#64748b",
  dense: "#10b981",
  output: "#ec4899",
  merge: "#9333ea",
};

const MERGE_TYPES = new Set(["add_merge", "concat_merge"]);

function summarizeParams(params) {
  const entries = Object.entries(params || {});
  if (entries.length === 0) return null;
  return entries.map(([k, v]) => `${k}=${v}`).join(", ");
}

export default function LayerNode({ data, selected }) {
  const color = CATEGORY_COLORS[data.category] || "#64748b";
  const paramSummary = summarizeParams(data.params);
  const isMerge = MERGE_TYPES.has(data.layerType);

  return (
    <div
      style={{
        border: `2px solid ${selected ? "#1d4ed8" : color}`,
        borderRadius: 8,
        background: "#fff",
        minWidth: 160,
        boxShadow: selected ? "0 0 0 2px rgba(29,78,216,0.3)" : "0 1px 3px rgba(0,0,0,0.15)",
        fontSize: 13,
      }}
    >
      {isMerge ? (
        <>
          <Handle
            type="target"
            position={Position.Left}
            id="in0"
            style={{ background: color, top: "35%" }}
          />
          <Handle
            type="target"
            position={Position.Left}
            id="in1"
            style={{ background: color, top: "65%" }}
          />
        </>
      ) : (
        <Handle type="target" position={Position.Left} style={{ background: color }} />
      )}
      <div
        style={{
          background: color,
          color: "#fff",
          padding: "4px 10px",
          borderRadius: "6px 6px 0 0",
          fontWeight: 600,
        }}
      >
        {data.label}
      </div>
      <div style={{ padding: "6px 10px" }}>
        {paramSummary && (
          <div style={{ color: "#444", marginBottom: 4 }}>{paramSummary}</div>
        )}
        {isMerge && (
          <div style={{ color: "#888", fontSize: 11 }}>in0 (top), in1 (bottom)</div>
        )}
        {data.shapeText ? (
          <div style={{ color: "#16a34a", fontFamily: "monospace" }}>&rarr; {data.shapeText}</div>
        ) : (
          <div style={{ color: "#aaa", fontStyle: "italic" }}>shape unknown</div>
        )}
      </div>
      <Handle type="source" position={Position.Right} style={{ background: color }} />
    </div>
  );
}
