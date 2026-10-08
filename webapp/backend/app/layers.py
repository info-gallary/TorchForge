"""The layer palette: what the canvas can offer, and how a DAG of layer
specs (type + user-set params + connections) becomes a real,
executable torch.nn.Module.

The whole point of this module is "least lines, no shape bookkeeping":
the user only ever sets the params they actually care about
(out_channels, kernel_size, out_features, dropout p, ...). Everything
shape-dependent (in_channels, in_features, num_features for BatchNorm)
is inferred here by walking the graph in topological order and
tracking each node's output shape -- and if the user forgets a Flatten
before their first Linear, one is inserted automatically rather than
erroring, since that's the single most common beginner mistake this
tool exists to remove.

Branches are supported via two merge layer types, `add_merge` (requires
both input shapes to match exactly, like NeuroCUDA's own `add`) and
`concat_merge` (concatenates channels, or features if both inputs are
already flat). A DAGModel ties the per-node nn.Modules together into one
real, trainable model whose forward() walks the same topological order.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from app.graph import MERGE_TYPES, GraphError

LAYER_SPECS = {
    "conv2d": {
        "label": "Conv2D",
        "category": "conv",
        "params": {
            "out_channels": {"type": "int", "default": 16, "min": 1},
            "kernel_size": {"type": "int", "default": 3, "min": 1},
            "stride": {"type": "int", "default": 1, "min": 1},
            "padding": {"type": "int", "default": 1, "min": 0},
        },
    },
    "maxpool2d": {
        "label": "MaxPool2D",
        "category": "pool",
        "params": {
            "kernel_size": {"type": "int", "default": 2, "min": 1},
            "stride": {"type": "int", "default": 2, "min": 1},
        },
    },
    "avgpool2d": {
        "label": "AvgPool2D",
        "category": "pool",
        "params": {
            "kernel_size": {"type": "int", "default": 2, "min": 1},
            "stride": {"type": "int", "default": 2, "min": 1},
        },
    },
    "batchnorm2d": {"label": "BatchNorm2D", "category": "norm", "params": {}},
    "relu": {"label": "ReLU", "category": "activation", "params": {}},
    "leakyrelu": {
        "label": "LeakyReLU",
        "category": "activation",
        "params": {"negative_slope": {"type": "float", "default": 0.01, "min": 0.0}},
    },
    "dropout": {
        "label": "Dropout",
        "category": "regularization",
        "params": {"p": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}},
    },
    "flatten": {"label": "Flatten", "category": "shape", "params": {}},
    "linear": {
        "label": "Linear",
        "category": "dense",
        "params": {"out_features": {"type": "int", "default": 128, "min": 1}},
    },
    "logsoftmax": {"label": "LogSoftmax", "category": "output", "params": {}},
    "add_merge": {
        "label": "Add (merge)",
        "category": "merge",
        "params": {},
        "inputs": 2,
    },
    "concat_merge": {
        "label": "Concat (merge)",
        "category": "merge",
        "params": {},
        "inputs": 2,
    },
}


@dataclass
class ShapeState:
    mode: str  # "image" or "flat"
    channels: int = 0
    h: int = 0
    w: int = 0
    features: int = 0

    def describe(self) -> str:
        if self.mode == "image":
            return f"{self.channels}x{self.h}x{self.w}"
        return f"{self.features}"

    def as_dict(self) -> dict:
        if self.mode == "image":
            return {"mode": "image", "channels": self.channels, "h": self.h, "w": self.w}
        return {"mode": "flat", "features": self.features}


def _conv_out(size: int, kernel_size: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel_size) // stride + 1


class DAGModel(nn.Module):
    """A real nn.Module whose forward() executes the canvas graph in
    topological order, routing each node's output to whichever
    downstream node(s) it's connected to. nn.Sequential can't express
    branches, so this exists instead -- it's still an ordinary PyTorch
    model (trains with any optimizer, autograd works normally)."""

    def __init__(self, modules_by_id: dict, order_ids: list, incoming: dict,
                 merge_ops: dict, source_id: str, sink_id: str):
        super().__init__()
        self.mod_dict = nn.ModuleDict(modules_by_id)
        self.order_ids = order_ids
        self.incoming = incoming
        self.merge_ops = merge_ops
        self.source_id = source_id
        self.sink_id = sink_id

    def forward(self, x):
        values = {}
        for nid in self.order_ids:
            if nid in self.merge_ops:
                a_id = self.incoming[nid][0]["source"]
                b_id = self.incoming[nid][1]["source"]
                a, b = values[a_id], values[b_id]
                values[nid] = a + b if self.merge_ops[nid] == "add" else torch.cat([a, b], dim=1)
            elif nid == self.source_id:
                values[nid] = self.mod_dict[nid](x)
            else:
                (pred,) = self.incoming[nid]
                values[nid] = self.mod_dict[nid](values[pred["source"]])
        return values[self.sink_id]


def build_dag(order_ids: list, nodes_by_id: dict, incoming: dict, outgoing: dict,
              source_id: str, sink_id: str, input_shape: tuple):
    """Builds a DAGModel from a resolved graph (src/app/graph.py's
    resolve_dag). input_shape: (channels, h, w) of the dataset's input
    images.

    Returns (model: DAGModel, shapes_by_node: dict, warnings: list,
    ends_with_logsoftmax: bool). Raises GraphError (a user-facing
    message) if any layer would produce an invalid shape.
    """
    channels, h, w = input_shape
    modules_by_id: dict = {}
    merge_ops: dict = {}
    shapes_by_node: dict = {}
    warnings: list = []

    for nid in order_ids:
        node = nodes_by_id[nid]
        ltype = node["type"]
        params = node.get("params") or {}

        if ltype in MERGE_TYPES:
            a_shape = shapes_by_node[incoming[nid][0]["source"]]
            b_shape = shapes_by_node[incoming[nid][1]["source"]]
            state = _merge_shapes(ltype, a_shape, b_shape, nid)
            merge_ops[nid] = "add" if ltype == "add_merge" else "concat"
        else:
            if nid == source_id:
                in_state = ShapeState(mode="image", channels=channels, h=h, w=w)
            else:
                (pred,) = incoming[nid]
                in_state = _state_from_dict(shapes_by_node[pred["source"]])
            module, state, warning = _build_single_layer(ltype, params, in_state, nid)
            modules_by_id[nid] = module
            if warning:
                warnings.append(warning)

        shapes_by_node[nid] = state.as_dict()

    ends_with_logsoftmax = nodes_by_id[sink_id]["type"] == "logsoftmax"
    model = DAGModel(modules_by_id, order_ids, incoming, merge_ops, source_id, sink_id)
    return model, shapes_by_node, warnings, ends_with_logsoftmax


def _state_from_dict(d: dict) -> ShapeState:
    if d["mode"] == "image":
        return ShapeState(mode="image", channels=d["channels"], h=d["h"], w=d["w"])
    return ShapeState(mode="flat", features=d["features"])


def _merge_shapes(ltype: str, a: dict, b: dict, node_id: str) -> ShapeState:
    label = LAYER_SPECS[ltype]["label"]
    if ltype == "add_merge":
        if a != b:
            raise GraphError(
                f"'{label}' layer '{node_id}': both branches must have the exact same shape "
                f"to add them, got {_describe_dict(a)} and {_describe_dict(b)}."
            )
        return _state_from_dict(a)

    # concat_merge
    if a["mode"] != b["mode"]:
        raise GraphError(
            f"'{label}' layer '{node_id}': can't concatenate an image-shaped branch with a "
            f"flat one ({_describe_dict(a)} vs {_describe_dict(b)}). Add a Flatten to match them."
        )
    if a["mode"] == "image":
        if a["h"] != b["h"] or a["w"] != b["w"]:
            raise GraphError(
                f"'{label}' layer '{node_id}': both branches must have the same height/width "
                f"to concatenate channels, got {_describe_dict(a)} and {_describe_dict(b)}."
            )
        return ShapeState(mode="image", channels=a["channels"] + b["channels"], h=a["h"], w=a["w"])
    return ShapeState(mode="flat", features=a["features"] + b["features"])


def _describe_dict(d: dict) -> str:
    if d["mode"] == "image":
        return f"{d['channels']}x{d['h']}x{d['w']}"
    return str(d["features"])


def _build_single_layer(ltype: str, params: dict, state: ShapeState, node_id: str):
    """Builds the nn.Module for one non-merge node. Returns (module,
    new_state, warning_or_None)."""
    spec = LAYER_SPECS.get(ltype)
    if spec is None:
        raise GraphError(f"Unknown layer type '{ltype}'.")

    if ltype == "conv2d":
        _require_mode(state, "image", "Conv2D", node_id)
        out_channels = int(params.get("out_channels", 16))
        kernel_size = int(params.get("kernel_size", 3))
        stride = int(params.get("stride", 1))
        padding = int(params.get("padding", 1))
        new_h = _conv_out(state.h, kernel_size, stride, padding)
        new_w = _conv_out(state.w, kernel_size, stride, padding)
        _require_positive_spatial(new_h, new_w, "Conv2D", node_id)
        module = nn.Conv2d(state.channels, out_channels, kernel_size, stride=stride, padding=padding)
        return module, ShapeState(mode="image", channels=out_channels, h=new_h, w=new_w), None

    if ltype in ("maxpool2d", "avgpool2d"):
        _require_mode(state, "image", spec["label"], node_id)
        kernel_size = int(params.get("kernel_size", 2))
        stride = int(params.get("stride", kernel_size))
        new_h = _conv_out(state.h, kernel_size, stride, 0)
        new_w = _conv_out(state.w, kernel_size, stride, 0)
        _require_positive_spatial(new_h, new_w, spec["label"], node_id)
        pool_cls = nn.MaxPool2d if ltype == "maxpool2d" else nn.AvgPool2d
        module = pool_cls(kernel_size, stride=stride)
        return module, ShapeState(mode="image", channels=state.channels, h=new_h, w=new_w), None

    if ltype == "batchnorm2d":
        _require_mode(state, "image", "BatchNorm2D", node_id)
        return nn.BatchNorm2d(state.channels), state, None

    if ltype == "relu":
        return nn.ReLU(), state, None

    if ltype == "leakyrelu":
        return nn.LeakyReLU(negative_slope=float(params.get("negative_slope", 0.01))), state, None

    if ltype == "dropout":
        return nn.Dropout(p=float(params.get("p", 0.5))), state, None

    if ltype == "flatten":
        if state.mode == "image":
            new_state = ShapeState(mode="flat", features=state.channels * state.h * state.w)
            return nn.Flatten(), new_state, None
        return nn.Identity(), state, None

    if ltype == "linear":
        out_features = int(params.get("out_features", 128))
        if state.mode == "image":
            warning = (
                f"Inserted an implicit Flatten before layer '{node_id}' (Linear) since "
                f"the data was still image-shaped ({state.describe()})."
            )
            in_features = state.channels * state.h * state.w
            module = nn.Sequential(nn.Flatten(), nn.Linear(in_features, out_features))
            return module, ShapeState(mode="flat", features=out_features), warning
        module = nn.Linear(state.features, out_features)
        return module, ShapeState(mode="flat", features=out_features), None

    if ltype == "logsoftmax":
        return nn.LogSoftmax(dim=1), state, None

    raise GraphError(f"Layer type '{ltype}' is not implemented.")  # pragma: no cover


def _require_mode(state: ShapeState, mode: str, label: str, node_id: str) -> None:
    if state.mode != mode:
        raise GraphError(
            f"Layer '{label}' ({node_id}) needs image-shaped input (channels x height x "
            f"width), but the data here is already flat ({state.describe()}). Move this "
            f"layer before any Flatten/Linear layer."
        )


def _require_positive_spatial(h: int, w: int, label: str, node_id: str) -> None:
    if h < 1 or w < 1:
        raise GraphError(
            f"Layer '{label}' ({node_id}) would shrink the spatial size to {h}x{w}, which "
            f"isn't valid. Reduce its kernel size/stride, add padding, or remove an earlier "
            f"downsampling layer."
        )
