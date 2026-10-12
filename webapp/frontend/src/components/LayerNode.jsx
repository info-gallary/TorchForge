import { Handle, Position } from "@xyflow/react";
import { categoryColor } from "../categoryColors";

const MERGE_TYPES = new Set(["add_merge", "concat_merge"]);

function summarizeParams(params) {
  const entries = Object.entries(params || {});
  if (entries.length === 0) return null;
  return entries.map(([k, v]) => `${k}=${v}`).join(", ");
}

export default function LayerNode({ data, selected }) {
  const color = categoryColor(data.category);
  const paramSummary = summarizeParams(data.params);
  const isMerge = MERGE_TYPES.has(data.layerType);

  return (
    <div
      style={{
        border: `1.5px solid ${selected ? color : "var(--border)"}`,
        borderRadius: 10,
        background: "var(--surface)",
        minWidth: 170,
        boxShadow: selected ? `0 0 0 3px ${color}33, var(--shadow-md)` : "var(--shadow-sm)",
        fontSize: 13,
        overflow: "hidden",
        transition: "box-shadow 0.15s, border-color 0.15s",
      }}
    >
      {isMerge ? (
        <>
          <Handle type="target" position={Position.Left} id="in0" style={{ background: color, top: "35%", width: 9, height: 9, border: "2px solid var(--surface)" }} />
          <Handle type="target" position={Position.Left} id="in1" style={{ background: color, top: "65%", width: 9, height: 9, border: "2px solid var(--surface)" }} />
        </>
      ) : (
        <Handle type="target" position={Position.Left} style={{ background: color, width: 9, height: 9, border: "2px solid var(--surface)" }} />
      )}
      <div
        style={{
          background: `linear-gradient(135deg, ${color}2e, ${color}14)`,
          borderBottom: `1px solid ${color}40`,
          color: "var(--text)",
          padding: "6px 11px",
          fontWeight: 700,
          fontSize: 12.5,
          display: "flex", alignItems: "center", gap: 6,
        }}
      >
        <span style={{ width: 7, height: 7, borderRadius: "50%", background: color, flexShrink: 0 }} />
        {data.label}
      </div>
      <div style={{ padding: "7px 11px" }}>
        {paramSummary && (
          <div className="mono" style={{ color: "var(--text-dim)", marginBottom: 4, fontSize: 11.5 }}>{paramSummary}</div>
        )}
        {isMerge && (
          <div style={{ color: "var(--text-faint)", fontSize: 10.5, marginBottom: 4 }}>in0 (top), in1 (bottom)</div>
        )}
        {data.shapeText ? (
          <div className="mono" style={{ color: "var(--success)", fontSize: 12 }}>&rarr; {data.shapeText}</div>
        ) : (
          <div style={{ color: "var(--text-faint)", fontStyle: "italic", fontSize: 11.5 }}>shape unknown</div>
        )}
      </div>
      <Handle type="source" position={Position.Right} style={{ background: color, width: 9, height: 9, border: "2px solid var(--surface)" }} />
    </div>
  );
}
