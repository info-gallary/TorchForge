import LiveChart from "./LiveChart";

export default function TrainPanel({
  datasets, dataset, onDatasetChange,
  config, onConfigChange,
  status, log, lossSeries, accSeries,
  onTrain, onExportCode, exportedCode,
  onViewDiagram, diagramUrl, diagramError,
  neurocudaResult,
  shapeError,
}) {
  return (
    <div style={{ padding: 12, height: "100%", overflowY: "auto" }}>
      <h3 style={{ marginTop: 0 }}>Dataset &amp; training</h3>

      <label style={{ display: "block", marginBottom: 10, fontSize: 13 }}>
        Dataset
        <select
          value={dataset}
          onChange={(e) => onDatasetChange(e.target.value)}
          style={{ display: "block", width: "100%", marginTop: 4, padding: 4 }}
        >
          {datasets.map((d) => (
            <option key={d.key} value={d.key}>
              {d.label} ({d.input_shape.join("x")}, {d.num_classes} classes)
            </option>
          ))}
        </select>
      </label>

      {[
        ["epochs", "Epochs", 1],
        ["batch_size", "Batch size", 1],
        ["lr", "Learning rate", 0.0001],
      ].map(([key, label, step]) => (
        <label key={key} style={{ display: "block", marginBottom: 10, fontSize: 13 }}>
          {label}
          <input
            type="number"
            step={step}
            value={config[key]}
            onChange={(e) => onConfigChange(key, parseFloat(e.target.value))}
            style={{ display: "block", width: "100%", marginTop: 4, padding: 4 }}
          />
        </label>
      ))}

      <label style={{ display: "block", marginBottom: 10, fontSize: 13 }}>
        Optimizer
        <select
          value={config.optimizer}
          onChange={(e) => onConfigChange("optimizer", e.target.value)}
          style={{ display: "block", width: "100%", marginTop: 4, padding: 4 }}
        >
          <option value="adam">Adam</option>
          <option value="sgd">SGD (momentum 0.9)</option>
        </select>
      </label>

      {shapeError && (
        <div style={{ background: "#fee2e2", border: "1px solid #fca5a5", borderRadius: 6, padding: 8, fontSize: 12, marginBottom: 10 }}>
          {shapeError}
        </div>
      )}

      <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
        <button
          onClick={onTrain}
          disabled={status === "training" || !!shapeError}
          style={{
            flex: 1, padding: "8px 12px", borderRadius: 6, border: "none",
            background: status === "training" ? "#94a3b8" : "#16a34a", color: "#fff",
            cursor: status === "training" ? "not-allowed" : "pointer", fontWeight: 600,
          }}
        >
          {status === "training" ? "Training..." : "Train"}
        </button>
        <button
          onClick={onExportCode}
          disabled={!!shapeError}
          style={{ padding: "8px 12px", borderRadius: 6, border: "1px solid #cbd5e1", background: "#fff", cursor: "pointer" }}
        >
          Export code
        </button>
        <button
          onClick={onViewDiagram}
          disabled={!!shapeError}
          style={{ padding: "8px 12px", borderRadius: 6, border: "1px solid #cbd5e1", background: "#fff", cursor: "pointer" }}
        >
          Diagram
        </button>
      </div>

      {diagramError && (
        <div style={{ background: "#fee2e2", border: "1px solid #fca5a5", borderRadius: 6, padding: 8, fontSize: 12, marginBottom: 10 }}>
          {diagramError}
        </div>
      )}
      {diagramUrl && (
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 12, color: "#555", marginBottom: 4 }}>
            Architecture diagram (via <code>visualtorch</code>)
          </div>
          <img
            src={diagramUrl}
            alt="Architecture diagram"
            style={{ maxWidth: "100%", border: "1px solid #e5e7eb", borderRadius: 6, background: "#fff" }}
          />
        </div>
      )}

      <LiveChart title="Training loss" series={lossSeries} color="#2563eb" />
      <LiveChart title="Validation accuracy" series={accSeries} color="#16a34a" />

      <div style={{ marginTop: 8 }}>
        <div style={{ fontSize: 12, color: "#555", marginBottom: 4 }}>Log</div>
        <div
          style={{
            background: "#0f172a", color: "#d1d5db", borderRadius: 6, padding: 8,
            fontFamily: "monospace", fontSize: 11, height: 140, overflowY: "auto",
          }}
        >
          {log.length === 0 && <div style={{ color: "#64748b" }}>No activity yet.</div>}
          {log.map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </div>
      </div>

      {neurocudaResult && (
        <div style={{ marginTop: 14 }}>
          <div style={{ fontSize: 12, color: "#555", marginBottom: 4 }}>NeuroCUDA acceleration</div>
          <div
            style={{
              background: neurocudaResult.success === false ? "#fee2e2" : "#ecfdf5",
              border: `1px solid ${neurocudaResult.success === false ? "#fca5a5" : "#6ee7b7"}`,
              borderRadius: 6, padding: 10, fontSize: 12,
            }}
          >
            {!neurocudaResult.attempted && <div>{neurocudaResult.reason}</div>}
            {neurocudaResult.attempted && neurocudaResult.success === false && (
              <div>
                NeuroCUDA acceleration failed:
                <pre style={{ whiteSpace: "pre-wrap", fontSize: 11 }}>{neurocudaResult.error}</pre>
              </div>
            )}
            {neurocudaResult.attempted && neurocudaResult.success && (
              <div>
                <div>
                  Compiled and ran <strong>{neurocudaResult.layers_accelerated}</strong> trained
                  layer(s) ({neurocudaResult.layer_types.join(" -> ")}) for real on the GPU via
                  NeuroCUDA, using kernels: {neurocudaResult.kernels_used.join(", ") || "(none)"}.
                </div>
                <div style={{ marginTop: 4 }}>
                  Matches PyTorch's own output: <strong>{String(neurocudaResult.matches_pytorch)}</strong>
                  {" "}(max abs diff {neurocudaResult.max_abs_diff.toExponential(2)})
                </div>
                {neurocudaResult.note && (
                  <div style={{ marginTop: 4, color: "#555" }}>{neurocudaResult.note}</div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {exportedCode && (
        <div style={{ marginTop: 14 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ fontSize: 12, color: "#555" }}>Minimal PyTorch code</div>
            <button
              onClick={() => navigator.clipboard.writeText(exportedCode)}
              style={{ fontSize: 11, padding: "2px 8px", cursor: "pointer" }}
            >
              Copy
            </button>
          </div>
          <pre
            style={{
              background: "#0f172a", color: "#e2e8f0", padding: 10, borderRadius: 6,
              fontSize: 11, maxHeight: 260, overflow: "auto", marginTop: 4,
            }}
          >
            {exportedCode}
          </pre>
        </div>
      )}
    </div>
  );
}
