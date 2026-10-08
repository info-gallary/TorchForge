const API_BASE = "http://127.0.0.1:8000";
const WS_BASE = "ws://127.0.0.1:8000";

export async function fetchDatasets() {
  const res = await fetch(`${API_BASE}/api/datasets`);
  const data = await res.json();
  return data.datasets;
}

export async function fetchLayers() {
  const res = await fetch(`${API_BASE}/api/layers`);
  const data = await res.json();
  return data.layers;
}

export async function fetchShapePreview({ dataset, nodes, edges }) {
  const res = await fetch(`${API_BASE}/api/shape_preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset, nodes, edges }),
  });
  return res.json();
}

export async function fetchCodeExport({ dataset, nodes, edges, epochs, batch_size, lr, optimizer }) {
  const res = await fetch(`${API_BASE}/api/code_export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset, nodes, edges, epochs, batch_size, lr, optimizer }),
  });
  return res.json();
}

/** Returns an object URL for the rendered diagram PNG, or throws with the
 * backend's error text on an invalid graph. Caller should revokeObjectURL
 * the previous one before requesting a new diagram to avoid leaking memory. */
export async function fetchDiagram({ dataset, nodes, edges, style = "flow" }) {
  const res = await fetch(`${API_BASE}/api/diagram`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset, nodes, edges, style }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `diagram request failed (${res.status})`);
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

/**
 * Opens the training WebSocket, sends the graph + config, and invokes
 * `onMessage` for every progress message the backend streams back
 * (status / step / epoch_end / done / error). Returns the WebSocket so
 * the caller can close it early if needed.
 */
export function startTraining({ dataset, nodes, edges, epochs, batch_size, lr, optimizer }, onMessage) {
  const ws = new WebSocket(`${WS_BASE}/ws/train`);
  ws.onopen = () => {
    ws.send(JSON.stringify({ dataset, nodes, edges, epochs, batch_size, lr, optimizer }));
  };
  ws.onmessage = (event) => {
    onMessage(JSON.parse(event.data));
  };
  ws.onerror = () => {
    onMessage({ type: "error", message: "WebSocket connection error -- is the backend running?" });
  };
  return ws;
}
