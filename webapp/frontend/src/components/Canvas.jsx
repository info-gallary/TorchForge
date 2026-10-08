import { useCallback, useRef } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
  useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import LayerNode from "./LayerNode";

const nodeTypes = { layerNode: LayerNode };

let nodeCounter = 0;
function nextNodeId() {
  nodeCounter += 1;
  return `node_${nodeCounter}_${Date.now()}`;
}

export default function Canvas({ nodes, edges, setNodes, setEdges, layerSpecs, onSelectNode }) {
  const wrapperRef = useRef(null);
  const { screenToFlowPosition } = useReactFlow();

  const onNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    [setNodes],
  );
  const onEdgesChange = useCallback(
    (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    [setEdges],
  );
  const onConnect = useCallback(
    (connection) => setEdges((eds) => addEdge({ ...connection, animated: false }, eds)),
    [setEdges],
  );

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();
      const layerType = event.dataTransfer.getData("application/vt-layer-type");
      if (!layerType || !layerSpecs[layerType]) return;

      const position = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      const spec = layerSpecs[layerType];
      const defaultParams = Object.fromEntries(
        Object.entries(spec.params || {}).map(([k, d]) => [k, d.default]),
      );

      const id = nextNodeId();
      setNodes((nds) => [
        ...nds,
        {
          id,
          type: "layerNode",
          position,
          data: {
            layerType,
            label: spec.label,
            category: spec.category,
            params: defaultParams,
            shapeText: null,
          },
        },
      ]);
    },
    [layerSpecs, screenToFlowPosition, setNodes],
  );

  return (
    <div ref={wrapperRef} style={{ height: "100%" }} onDragOver={onDragOver} onDrop={onDrop}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={(_, node) => onSelectNode(node.id)}
        onPaneClick={() => onSelectNode(null)}
      >
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
}
