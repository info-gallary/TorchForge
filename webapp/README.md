# TorchForge webapp

The drag-and-drop canvas + training UI half of [TorchForge](../README.md).
See the top-level README for the big picture (including how this talks
to the real NeuroCUDA compiler in `../compiler/`); this file covers the
implementation layout.

## Running

```bash
# backend, from backend/
python -m uvicorn app.main:app --reload --port 8000

# frontend, from frontend/
npm install   # first time only
npm run dev
```

## Backend (`backend/app/`)

- `main.py` — FastAPI app: REST endpoints (`/api/datasets`,
  `/api/layers`, `/api/shape_preview`, `/api/code_export`,
  `/api/diagram`) and the `/ws/train` WebSocket.
- `graph.py` — resolves the canvas (nodes + edges) into a topologically
  ordered DAG. Supports real branching: any layer may fan out to
  multiple downstream layers; `add_merge`/`concat_merge` are the only
  layer types allowed to fan two branches back in (exactly 2 inputs
  each, disambiguated by React Flow's `targetHandle`: `in0`/`in1`).
- `layers.py` — the layer palette (`LAYER_SPECS`) and `build_dag()`,
  which walks the resolved DAG in topological order doing real shape
  inference (auto-inserting a `Flatten` before a `Linear` that still
  sees image-shaped input) and builds a `DAGModel` — a real `nn.Module`
  whose `forward()` executes the graph (merge nodes do `a + b` or
  `torch.cat([a, b], dim=1)`; everything else calls its own
  `nn.Module`). This is a genuine, trainable PyTorch model, not a
  simulation.
- `datasets.py` — the torchvision dataset registry (CIFAR-10,
  CIFAR-100, MNIST, Fashion-MNIST), downloaded to `../data/` on first
  use.
- `training.py` — runs training on a background thread (it's
  genuinely blocking), streaming progress messages through a
  `queue.Queue` that `main.py`'s WebSocket handler drains via
  `run_in_executor` so the asyncio event loop is never blocked. After
  training, calls `neurocuda_bridge.try_accelerate(...)`.
- `neurocuda_bridge.py` — the real NeuroCUDA integration: finds the
  model's trailing Linear/ReLU suffix (if any), dumps its trained
  weights + one batch of real activations to `.nct` files, generates
  NeuroLang source using `load(...)`, runs it through NeuroCUDA's own
  compiler pipeline (imported directly from `../compiler/src/`), and
  compares the GPU output against PyTorch's own forward pass. Reports
  honestly when nothing is eligible (e.g. the model ends in Conv2d)
  rather than fabricating a result.
- `codegen.py` — exports the architecture as a minimal, standalone
  PyTorch script: a small custom `nn.Module` subclass with an explicit
  `forward()` (branches can't be expressed as `nn.Sequential`), built
  from the exact same per-node shapes `build_dag()` computed, so the
  exported code and the live model can never drift apart.
- `diagram.py` — renders an architecture diagram via the real
  `visualtorch` PyPI package (an *untrained* copy of the model is
  built — only shapes matter for a diagram).

## Frontend (`frontend/src/`)

- `App.jsx` — top-level state (nodes/edges, training status, results)
  and the four-panel layout.
- `components/Canvas.jsx` — the React Flow canvas: drag-from-palette
  wiring (`onDrop`/`onDragOver`), node dragging/connecting (built in).
- `components/LayerNode.jsx` — custom node renderer; merge-type nodes
  get two distinct target handles (`in0` top, `in1` bottom) instead of
  one.
- `components/NodePalette.jsx`, `Inspector.jsx`, `TrainPanel.jsx`,
  `LiveChart.jsx` (a small inline-SVG chart, no charting library
  needed for a couple of series).
- `api.js` — REST + WebSocket client.
- `examples.js` — one-click demo architectures that pre-populate the
  canvas via the header's "Load example" buttons: a plain sequential
  CNN and a residual Add-merge skip-connection model, both for MNIST.

## Tests

```bash
cd backend
python -m unittest discover -s tests   # 25 tests
```

Covers DAG resolution (chains, branches, invalid graphs), shape
inference (including merge-shape rules and the implicit-Flatten
insertion), and the NeuroCUDA bridge's suffix-detection logic, plus a
real end-to-end GPU test of the acceleration bridge (skipped
automatically if `nvcc` isn't on PATH).
