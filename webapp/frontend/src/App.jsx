import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import Canvas from "./components/Canvas";
import NodePalette from "./components/NodePalette";
import Inspector from "./components/Inspector";
import TrainPanel from "./components/TrainPanel";
import {
  fetchDatasets, fetchLayers, fetchShapePreview, fetchCodeExport, fetchDiagram,
  fetchDiagramOptions, startTraining,
} from "./api";
import { EXAMPLES } from "./examples";

function toGraphPayload(nodes, edges) {
  return {
    nodes: nodes.map((n) => ({ id: n.id, type: n.data.layerType, params: n.data.params })),
    edges: edges.map((e) => ({
      source: e.source,
      target: e.target,
      targetHandle: e.targetHandle ?? null,
    })),
  };
}

export default function App() {
  const [datasets, setDatasets] = useState([]);
  const [layerSpecs, setLayerSpecs] = useState({});
  const [dataset, setDataset] = useState("cifar10");

  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [selectedNodeId, setSelectedNodeId] = useState(null);

  const [shapeError, setShapeError] = useState(null);

  const [config, setConfig] = useState({ epochs: 3, batch_size: 128, lr: 0.001, optimizer: "adam" });
  const [status, setStatus] = useState("idle"); // idle | training | done | error
  const [log, setLog] = useState([]);
  const [lossSeries, setLossSeries] = useState([]);
  const [accSeries, setAccSeries] = useState([]);
  const [exportedCode, setExportedCode] = useState(null);
  const [diagramUrl, setDiagramUrl] = useState(null);
  const [diagramError, setDiagramError] = useState(null);
  const [diagramOptions, setDiagramOptions] = useState({ styles: ["flow", "graph", "lenet"], palettes: ["okabe_ito"] });
  const [diagramConfig, setDiagramConfig] = useState({
    style: "flow", legend: false, show_dimension: false, palette: "okabe_ito",
    draw_volume: true, hide_activations: false,
  });
  const [neurocudaResult, setNeurocudaResult] = useState(null);
  const wsRef = useRef(null);
  const diagramUrlRef = useRef(null);

  useEffect(() => {
    fetchDatasets().then(setDatasets).catch((e) => setLog((l) => [...l, `error loading datasets: ${e}`]));
    fetchLayers().then(setLayerSpecs).catch((e) => setLog((l) => [...l, `error loading layers: ${e}`]));
    fetchDiagramOptions().then(setDiagramOptions).catch(() => {});
  }, []);

  const onDiagramConfigChange = (key, value) => setDiagramConfig((c) => ({ ...c, [key]: value }));

  const nodesKey = nodes.map((n) => `${n.id}:${JSON.stringify(n.data.params)}`).join(",");

  // Live shape preview: whenever the graph or dataset changes, ask the
  // backend to infer every node's output shape (debounced slightly so
  // dragging/typing doesn't spam requests).
  useEffect(() => {
    if (nodes.length === 0) {
      setShapeError(null);
      return;
    }
    const handle = setTimeout(async () => {
      const payload = toGraphPayload(nodes, edges);
      const result = await fetchShapePreview({ dataset, ...payload });
      if (result.error) {
        setShapeError(result.error);
        setNodes((nds) => nds.map((n) => ({ ...n, data: { ...n.data, shapeText: null } })));
        return;
      }
      setShapeError(null);
      setNodes((nds) =>
        nds.map((n) => {
          const shape = result.shapes[n.id];
          const shapeText = shape
            ? shape.mode === "image"
              ? `${shape.channels}x${shape.h}x${shape.w}`
              : `${shape.features}`
            : null;
          return { ...n, data: { ...n.data, shapeText } };
        }),
      );
    }, 300);
    return () => clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodesKey, edges, dataset]);

  const selectedNode = useMemo(
    () => nodes.find((n) => n.id === selectedNodeId) || null,
    [nodes, selectedNodeId],
  );

  const onChangeParam = useCallback((nodeId, key, value) => {
    setNodes((nds) =>
      nds.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, params: { ...n.data.params, [key]: value } } } : n,
      ),
    );
  }, []);

  const onDeleteNode = useCallback((nodeId) => {
    setNodes((nds) => nds.filter((n) => n.id !== nodeId));
    setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
    setSelectedNodeId(null);
  }, []);

  const onConfigChange = (key, value) => setConfig((c) => ({ ...c, [key]: value }));

  const onTrain = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    setStatus("training");
    setLog([]);
    setLossSeries([]);
    setAccSeries([]);
    setNeurocudaResult(null);
    const payload = toGraphPayload(nodes, edges);

    wsRef.current = startTraining({ dataset, ...payload, ...config }, (msg) => {
      if (msg.type === "status") {
        setLog((l) => [...l, msg.message]);
      } else if (msg.type === "step") {
        setLog((l) => [...l, `epoch ${msg.epoch + 1} step ${msg.step}/${msg.total_steps} loss=${msg.loss.toFixed(4)}`]);
        setLossSeries((s) => [...s, { value: msg.loss }]);
      } else if (msg.type === "epoch_end") {
        setLog((l) => [
          ...l,
          `epoch ${msg.epoch + 1} done: train_loss=${msg.train_loss.toFixed(4)} ` +
            `val_loss=${msg.val_loss.toFixed(4)} val_acc=${(msg.val_acc * 100).toFixed(2)}% ` +
            `(${msg.elapsed_s.toFixed(1)}s)`,
        ]);
        setAccSeries((s) => [...s, { value: msg.val_acc }]);
      } else if (msg.type === "neurocuda") {
        setNeurocudaResult(msg);
      } else if (msg.type === "done") {
        setLog((l) => [...l, "Training complete."]);
        setStatus("done");
      } else if (msg.type === "error") {
        setLog((l) => [...l, `ERROR: ${msg.message}`]);
        setStatus("error");
      }
    });
  };

  const onExportCode = async () => {
    const payload = toGraphPayload(nodes, edges);
    const result = await fetchCodeExport({ dataset, ...payload, ...config });
    if (result.error) {
      setLog((l) => [...l, `export error: ${result.error}`]);
      return;
    }
    setExportedCode(result.code);
  };

  const onLoadExample = (key) => {
    const example = EXAMPLES[key];
    if (!example) return;

    wsRef.current?.close();
    if (diagramUrlRef.current) {
      URL.revokeObjectURL(diagramUrlRef.current);
      diagramUrlRef.current = null;
    }

    setDataset(example.dataset);
    setNodes(
      example.nodes.map((n) => {
        const spec = layerSpecs[n.type] || {};
        const defaultParams = Object.fromEntries(
          Object.entries(spec.params || {}).map(([k, d]) => [k, d.default]),
        );
        return {
          id: n.id,
          type: "layerNode",
          position: n.position,
          data: {
            layerType: n.type,
            label: spec.label || n.type,
            category: spec.category || "other",
            params: { ...defaultParams, ...n.params },
            shapeText: null,
          },
        };
      }),
    );
    setEdges(
      example.edges.map((e, i) => ({
        id: `example-${key}-${i}`,
        source: e.source,
        target: e.target,
        targetHandle: e.targetHandle,
      })),
    );
    setSelectedNodeId(null);
    setShapeError(null);
    setExportedCode(null);
    setDiagramUrl(null);
    setDiagramError(null);
    setNeurocudaResult(null);
    setLog([]);
    setLossSeries([]);
    setAccSeries([]);
    setStatus("idle");
  };

  const onViewDiagram = async () => {
    const payload = toGraphPayload(nodes, edges);
    try {
      const url = await fetchDiagram({ dataset, ...payload, ...diagramConfig });
      if (diagramUrlRef.current) {
        URL.revokeObjectURL(diagramUrlRef.current);
      }
      diagramUrlRef.current = url;
      setDiagramUrl(url);
      setDiagramError(null);
    } catch (e) {
      setDiagramError(String(e.message || e));
    }
  };

  return (
    <div style={{ display: "grid", gridTemplateRows: "48px 1fr", height: "100vh" }}>
      <div
        style={{
          display: "flex", alignItems: "center", gap: 10, padding: "0 16px",
          borderBottom: "1px solid #e5e7eb", background: "#0f172a", color: "#fff",
        }}
      >
        <strong style={{ fontSize: 16 }}>TorchForge</strong>
        <span style={{ fontSize: 12, color: "#94a3b8" }}>
          drag-and-drop PyTorch architectures &middot; trailing Linear/ReLU layers run for real on NeuroCUDA
        </span>
        <div style={{ marginLeft: "auto" }}>
          {Object.entries(EXAMPLES).map(([key, ex]) => (
            <button
              key={key}
              onClick={() => onLoadExample(key)}
              style={{
                fontSize: 12, padding: "5px 10px", borderRadius: 6,
                border: "1px solid #334155", background: "#1e293b", color: "#e2e8f0", cursor: "pointer",
              }}
            >
              Load example: {ex.label}
            </button>
          ))}
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "220px 1fr 260px 340px", minHeight: 0 }}>
      <div style={{ borderRight: "1px solid #e5e7eb" }}>
        <NodePalette layerSpecs={layerSpecs} />
      </div>
      <div>
        <ReactFlowProvider>
          <Canvas
            nodes={nodes}
            edges={edges}
            setNodes={setNodes}
            setEdges={setEdges}
            layerSpecs={layerSpecs}
            onSelectNode={setSelectedNodeId}
          />
        </ReactFlowProvider>
      </div>
      <div style={{ borderLeft: "1px solid #e5e7eb", borderRight: "1px solid #e5e7eb" }}>
        <Inspector
          selectedNode={selectedNode}
          layerSpecs={layerSpecs}
          onChangeParam={onChangeParam}
          onDelete={onDeleteNode}
        />
      </div>
      <div>
        <TrainPanel
          datasets={datasets}
          dataset={dataset}
          onDatasetChange={setDataset}
          config={config}
          onConfigChange={onConfigChange}
          status={status}
          log={log}
          lossSeries={lossSeries}
          accSeries={accSeries}
          onTrain={onTrain}
          onExportCode={onExportCode}
          exportedCode={exportedCode}
          onViewDiagram={onViewDiagram}
          diagramUrl={diagramUrl}
          diagramError={diagramError}
          diagramOptions={diagramOptions}
          diagramConfig={diagramConfig}
          onDiagramConfigChange={onDiagramConfigChange}
          neurocudaResult={neurocudaResult}
          shapeError={shapeError}
        />
      </div>
      </div>
    </div>
  );
}
