import LiveChart from "./LiveChart";

function ErrorBanner({ children }) {
  return (
    <div style={{
      background: "rgba(248,113,113,0.1)", border: "1px solid rgba(248,113,113,0.35)",
      color: "#fecaca", borderRadius: 8, padding: "9px 12px", fontSize: 12.5, marginBottom: 12,
    }}>
      {children}
    </div>
  );
}

function NeuroCudaCard({ result }) {
  const ok = result.attempted && result.success;
  const failed = result.attempted && result.success === false;

  return (
    <div
      className="card"
      style={{
        marginTop: 16, padding: "14px 16px", overflow: "hidden",
        borderColor: ok ? "rgba(118,185,0,0.35)" : failed ? "rgba(248,113,113,0.35)" : "var(--border)",
        boxShadow: ok ? "var(--shadow-glow-neuro)" : "var(--shadow-sm)",
        background: ok
          ? "linear-gradient(165deg, rgba(118,185,0,0.08), var(--surface) 55%)"
          : "var(--surface)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <span style={{
          width: 22, height: 22, borderRadius: 6, flexShrink: 0,
          background: "linear-gradient(135deg, var(--neuro-a), var(--neuro-b))",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 12, fontWeight: 800, color: "#06240a",
        }}>N</span>
        <span className="neuro-wordmark" style={{ fontSize: 14 }}>NeuroCUDA</span>
        <span style={{ fontSize: 11.5, color: "var(--text-faint)" }}>compiler acceleration</span>
        <span
          className="pill"
          style={{
            marginLeft: "auto",
            color: ok ? "var(--success)" : failed ? "var(--danger)" : "var(--text-faint)",
            borderColor: ok ? "rgba(52,211,153,0.4)" : failed ? "rgba(248,113,113,0.4)" : "var(--border)",
            background: ok ? "rgba(52,211,153,0.08)" : failed ? "rgba(248,113,113,0.08)" : "transparent",
          }}
        >
          {ok ? "✓ verified" : failed ? "✕ failed" : "not applicable"}
        </span>
      </div>

      {!result.attempted && (
        <div style={{ fontSize: 12.5, color: "var(--text-dim)" }}>{result.reason}</div>
      )}

      {failed && (
        <pre style={{
          whiteSpace: "pre-wrap", fontSize: 11, color: "var(--danger)", margin: 0,
          background: "var(--panel-solid)", borderRadius: 7, padding: 10, maxHeight: 160, overflow: "auto",
        }}>{result.error}</pre>
      )}

      {ok && (
        <>
          <div style={{ display: "flex", gap: 20, marginBottom: 10 }}>
            <div>
              <div style={{ fontSize: 22, fontWeight: 800, lineHeight: 1 }}>{result.layers_accelerated}</div>
              <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 3 }}>layers on real CUDA kernels</div>
            </div>
            <div>
              <div className="mono" style={{ fontSize: 15, fontWeight: 700, lineHeight: 1.3, color: "var(--success)" }}>
                {result.max_abs_diff.toExponential(2)}
              </div>
              <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 3 }}>max abs diff vs. PyTorch</div>
            </div>
          </div>

          <div style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 8 }}>
            {result.layer_types.join(" → ")}
          </div>

          <div style={{ marginBottom: result.note ? 8 : 0 }}>
            {result.kernels_used.map((k) => (
              <span key={k} className="mono" style={{
                display: "inline-block", fontSize: 11, fontWeight: 700, color: "var(--neuro-a)",
                background: "rgba(118,185,0,0.1)", border: "1px solid rgba(118,185,0,0.3)",
                borderRadius: 999, padding: "3px 10px", marginRight: 6, marginBottom: 4,
              }}>{k}</span>
            ))}
          </div>

          {result.note && (
            <div style={{ fontSize: 11.5, color: "var(--text-faint)", marginBottom: 4 }}>{result.note}</div>
          )}

          <details style={{ marginTop: 6 }}>
            <summary style={{ cursor: "pointer", fontSize: 11.5, color: "var(--text-faint)" }}>
              View generated NeuroLang source
            </summary>
            <pre className="mono" style={{
              fontSize: 11, color: "var(--text-dim)", background: "var(--panel-solid)",
              borderRadius: 7, padding: 10, marginTop: 6, maxHeight: 220, overflow: "auto",
            }}>{result.source}</pre>
          </details>
        </>
      )}
    </div>
  );
}

export default function TrainPanel({
  datasets, dataset, onDatasetChange,
  config, onConfigChange,
  status, log, lossSeries, accSeries,
  onTrain, onExportCode, exportedCode,
  onViewDiagram, diagramUrl, diagramError,
  diagramOptions, diagramConfig, onDiagramConfigChange,
  neurocudaResult,
  shapeError,
}) {
  return (
    <div style={{ padding: 16, height: "100%", overflowY: "auto" }}>
      <h3 style={{ margin: "0 0 14px", fontSize: 14, fontWeight: 700 }}>Dataset &amp; training</h3>

      <label className="field-label">
        Dataset
        <select value={dataset} onChange={(e) => onDatasetChange(e.target.value)}>
          {datasets.map((d) => (
            <option key={d.key} value={d.key}>
              {d.label} ({d.input_shape.join("x")}, {d.num_classes} classes)
            </option>
          ))}
        </select>
      </label>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        {[
          ["epochs", "Epochs", 1],
          ["batch_size", "Batch size", 1],
        ].map(([key, label, step]) => (
          <label key={key} className="field-label">
            {label}
            <input type="number" step={step} value={config[key]} onChange={(e) => onConfigChange(key, parseFloat(e.target.value))} />
          </label>
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <label className="field-label">
          Learning rate
          <input type="number" step={0.0001} value={config.lr} onChange={(e) => onConfigChange("lr", parseFloat(e.target.value))} />
        </label>
        <label className="field-label">
          Optimizer
          <select value={config.optimizer} onChange={(e) => onConfigChange("optimizer", e.target.value)}>
            <option value="adam">Adam</option>
            <option value="sgd">SGD (momentum 0.9)</option>
          </select>
        </label>
      </div>

      {shapeError && <ErrorBanner>{shapeError}</ErrorBanner>}

      <details style={{ marginBottom: 14, fontSize: 12 }}>
        <summary style={{ cursor: "pointer", color: "var(--text-dim)", fontWeight: 600 }}>Diagram options</summary>
        <div style={{ marginTop: 10, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          <label className="field-label">
            Style
            <select value={diagramConfig.style} onChange={(e) => onDiagramConfigChange("style", e.target.value)}>
              {diagramOptions.styles.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </label>
          <label className="field-label">
            Palette
            <select value={diagramConfig.palette} onChange={(e) => onDiagramConfigChange("palette", e.target.value)}>
              {diagramOptions.palettes.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </label>
        </div>
        <div style={{ marginTop: 4, display: "flex", flexDirection: "column", gap: 6, fontSize: 12.5, color: "var(--text-dim)" }}>
          <label><input type="checkbox" checked={diagramConfig.legend} onChange={(e) => onDiagramConfigChange("legend", e.target.checked)} /> Legend</label>
          <label><input type="checkbox" checked={diagramConfig.show_dimension} onChange={(e) => onDiagramConfigChange("show_dimension", e.target.checked)} /> Show output shapes</label>
          <label><input type="checkbox" checked={diagramConfig.draw_volume} onChange={(e) => onDiagramConfigChange("draw_volume", e.target.checked)} disabled={diagramConfig.style !== "flow"} /> 3D volumetric (flow style only)</label>
          <label><input type="checkbox" checked={diagramConfig.hide_activations} onChange={(e) => onDiagramConfigChange("hide_activations", e.target.checked)} /> Hide activation layers</label>
        </div>
      </details>

      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <button className="btn btn-primary" style={{ flex: 1 }} onClick={onTrain} disabled={status === "training" || !!shapeError}>
          {status === "training" ? "Training…" : "Train"}
        </button>
        <button className="btn btn-ghost" onClick={onExportCode} disabled={!!shapeError}>Export code</button>
        <button className="btn btn-ghost" onClick={onViewDiagram} disabled={!!shapeError}>Diagram</button>
      </div>

      {diagramError && <ErrorBanner>{diagramError}</ErrorBanner>}
      {diagramUrl && (
        <div style={{ marginBottom: 16 }}>
          <div className="section-label">Architecture diagram <span style={{ textTransform: "none", fontWeight: 400 }}>(via visualtorch)</span></div>
          <img src={diagramUrl} alt="Architecture diagram" style={{ maxWidth: "100%", border: "1px solid var(--border)", borderRadius: 9, background: "#fff" }} />
        </div>
      )}

      <LiveChart title="Training loss" series={lossSeries} color="var(--accent)" />
      <LiveChart title="Validation accuracy" series={accSeries} color="var(--success)" />

      <div style={{ marginTop: 10 }}>
        <div className="section-label">Log</div>
        <div className="mono" style={{
          background: "var(--panel-solid)", border: "1px solid var(--border)", color: "var(--text-dim)",
          borderRadius: 9, padding: 10, fontSize: 11, height: 140, overflowY: "auto",
        }}>
          {log.length === 0 && <div style={{ color: "var(--text-faint)" }}>No activity yet.</div>}
          {log.map((line, i) => <div key={i}>{line}</div>)}
        </div>
      </div>

      {neurocudaResult && <NeuroCudaCard result={neurocudaResult} />}

      {exportedCode && (
        <div style={{ marginTop: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
            <div className="section-label" style={{ marginBottom: 0 }}>Minimal PyTorch code</div>
            <button className="btn btn-ghost" style={{ fontSize: 11, padding: "4px 10px" }} onClick={() => navigator.clipboard.writeText(exportedCode)}>
              Copy
            </button>
          </div>
          <pre className="mono" style={{
            background: "var(--panel-solid)", border: "1px solid var(--border)", color: "var(--text-dim)",
            padding: 10, borderRadius: 9, fontSize: 11, maxHeight: 260, overflow: "auto",
          }}>{exportedCode}</pre>
        </div>
      )}
    </div>
  );
}
