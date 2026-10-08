"""TorchForge webapp backend: FastAPI app exposing the layer palette,
dataset list, shape-preview, code-export, diagram-export, and a
WebSocket training stream (which also tries real NeuroCUDA acceleration
of the model's trailing Linear/ReLU suffix -- see neurocuda_bridge.py).

Run with:
    cd backend
    uvicorn app.main:app --reload --port 8000
"""

import asyncio
import queue

from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.codegen import generate_script
from app.datasets import get_dataset_spec, list_datasets
from app.diagram import PALETTES as DIAGRAM_PALETTES
from app.diagram import STYLES as DIAGRAM_STYLES
from app.diagram import render_diagram
from app.graph import GraphError, resolve_dag
from app.layers import LAYER_SPECS, build_dag
from app.training import start_training

app = FastAPI(title="TorchForge backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev tool only; not deployed publicly
    allow_methods=["*"],
    allow_headers=["*"],
)


class NodeSpec(BaseModel):
    id: str
    type: str
    params: dict = {}


class EdgeSpec(BaseModel):
    source: str
    target: str
    targetHandle: str | None = None


class GraphRequest(BaseModel):
    dataset: str
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]


class CodeExportRequest(GraphRequest):
    epochs: int = 3
    batch_size: int = 128
    lr: float = 1e-3
    optimizer: str = "adam"


class DiagramRequest(GraphRequest):
    style: str = "flow"  # "flow" | "graph" | "lenet" (visualtorch's styles)
    legend: bool = False
    show_dimension: bool = False
    palette: str = "okabe_ito"
    draw_volume: bool = True
    hide_activations: bool = False


def _dump(req: GraphRequest):
    nodes = [n.model_dump() for n in req.nodes]
    edges = [e.model_dump() for e in req.edges]
    return nodes, edges


@app.get("/api/datasets")
def api_list_datasets():
    return {"datasets": list_datasets()}


@app.get("/api/layers")
def api_list_layers():
    return {"layers": LAYER_SPECS}


@app.post("/api/shape_preview")
def api_shape_preview(req: GraphRequest):
    try:
        spec = get_dataset_spec(req.dataset)
        nodes, edges = _dump(req)
        order_ids, nodes_by_id, incoming, outgoing, source_id, sink_id = resolve_dag(nodes, edges)
        _model, shapes_by_node, warnings, ends_with_logsoftmax = build_dag(
            order_ids, nodes_by_id, incoming, outgoing, source_id, sink_id,
            tuple(spec["input_shape"]),
        )
        return {
            "order": order_ids,
            "shapes": shapes_by_node,
            "warnings": warnings,
            "ends_with_logsoftmax": ends_with_logsoftmax,
            "error": None,
        }
    except (GraphError, ValueError) as e:
        return {"order": [], "shapes": {}, "warnings": [], "error": str(e)}


@app.post("/api/code_export")
def api_code_export(req: CodeExportRequest):
    try:
        spec = get_dataset_spec(req.dataset)
        nodes, edges = _dump(req)
        order_ids, nodes_by_id, incoming, outgoing, source_id, sink_id = resolve_dag(nodes, edges)
        model, shapes_by_node, warnings, ends_with_logsoftmax = build_dag(
            order_ids, nodes_by_id, incoming, outgoing, source_id, sink_id,
            tuple(spec["input_shape"]),
        )
        code = generate_script(
            dataset_key=req.dataset,
            order_ids=order_ids,
            nodes_by_id=nodes_by_id,
            incoming=incoming,
            merge_ops=model.merge_ops,
            source_id=source_id,
            sink_id=sink_id,
            shapes_by_node=shapes_by_node,
            warnings=warnings,
            ends_with_logsoftmax=ends_with_logsoftmax,
            epochs=req.epochs,
            batch_size=req.batch_size,
            lr=req.lr,
            optimizer_name=req.optimizer,
        )
        return {"code": code, "error": None}
    except (GraphError, ValueError) as e:
        return {"code": None, "error": str(e)}


@app.get("/api/diagram_options")
def api_diagram_options():
    return {"styles": list(DIAGRAM_STYLES), "palettes": DIAGRAM_PALETTES}


@app.post("/api/diagram")
def api_diagram(req: DiagramRequest):
    try:
        spec = get_dataset_spec(req.dataset)
        nodes, edges = _dump(req)
        order_ids, nodes_by_id, incoming, outgoing, source_id, sink_id = resolve_dag(nodes, edges)
        png_bytes = render_diagram(
            order_ids, nodes_by_id, incoming, outgoing, source_id, sink_id,
            tuple(spec["input_shape"]), style=req.style, legend=req.legend,
            show_dimension=req.show_dimension, palette=req.palette,
            draw_volume=req.draw_volume, hide_activations=req.hide_activations,
        )
        return Response(content=png_bytes, media_type="image/png")
    except (GraphError, ValueError) as e:
        return Response(content=str(e), media_type="text/plain", status_code=400)


@app.websocket("/ws/train")
async def ws_train(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    try:
        payload = await websocket.receive_json()
        nodes = payload.get("nodes", [])
        edges = payload.get("edges", [])
        config = {
            "dataset": payload["dataset"],
            "epochs": payload.get("epochs", 3),
            "batch_size": payload.get("batch_size", 128),
            "lr": payload.get("lr", 1e-3),
            "optimizer": payload.get("optimizer", "adam"),
        }

        q: "queue.Queue" = start_training(nodes, edges, config)

        while True:
            message = await loop.run_in_executor(None, q.get)
            await websocket.send_json(message)
            if message["type"] in ("done", "error"):
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001 - last-resort guard so the socket closes cleanly
        try:
            await websocket.send_json({"type": "error", "message": f"{type(e).__name__}: {e}"})
        except Exception:
            pass
    finally:
        await websocket.close()
