"""Renders the assembled architecture as a short, standalone,
copy-pasteable PyTorch script -- the "minimal lines" the canvas is
building towards.

A sequential chain could be exported as a bare `nn.Sequential(...)`,
but branches (Add/Concat merges) can't be expressed that way, so this
always generates a small custom `nn.Module` subclass with an explicit
`forward()` mirroring the graph's topological order -- which also works
perfectly fine for the non-branching case, so there's only one code
path to maintain.

Shape info (in_channels for Conv2d/BatchNorm2d, in_features for Linear)
is read from `shapes_by_node`, the exact same per-node shapes
app.layers.build_dag computed while constructing the live model -- so
the exported code and the model actually trained in the browser can
never drift apart.
"""

import re

from app.datasets import get_dataset_spec
from app.graph import MERGE_TYPES

_SHAPELESS_LAYER_REPR = {
    "maxpool2d": lambda p: (
        f"nn.MaxPool2d({int(p.get('kernel_size', 2))}, "
        f"stride={int(p.get('stride', p.get('kernel_size', 2)))})"
    ),
    "avgpool2d": lambda p: (
        f"nn.AvgPool2d({int(p.get('kernel_size', 2))}, "
        f"stride={int(p.get('stride', p.get('kernel_size', 2)))})"
    ),
    "relu": lambda p: "nn.ReLU()",
    "leakyrelu": lambda p: f"nn.LeakyReLU(negative_slope={float(p.get('negative_slope', 0.01))})",
    "dropout": lambda p: f"nn.Dropout(p={float(p.get('p', 0.5))})",
    "flatten": lambda p: "nn.Flatten()",
    "logsoftmax": lambda p: "nn.LogSoftmax(dim=1)",
}


def _attr_name(node_id: str) -> str:
    return "layer_" + re.sub(r"\W", "_", node_id)


def _var_name(node_id: str) -> str:
    return "v_" + re.sub(r"\W", "_", node_id)


def generate_script(
    *, dataset_key: str, order_ids: list, nodes_by_id: dict, incoming: dict,
    merge_ops: dict, source_id: str, sink_id: str, shapes_by_node: dict,
    warnings: list, ends_with_logsoftmax: bool,
    epochs: int, batch_size: int, lr: float, optimizer_name: str,
) -> str:
    spec = get_dataset_spec(dataset_key)
    dataset_cls = spec["cls"].__name__
    in_channels, in_h, in_w = spec["input_shape"]

    def shape_before(nid: str) -> dict:
        if nid == source_id:
            return {"mode": "image", "channels": in_channels, "h": in_h, "w": in_w}
        (pred,) = incoming[nid]
        return shapes_by_node[pred["source"]]

    init_lines = []
    forward_lines = []

    for nid in order_ids:
        node = nodes_by_id[nid]
        ltype = node["type"]
        params = node.get("params") or {}
        attr = _attr_name(nid)
        var = _var_name(nid)

        if ltype in MERGE_TYPES:
            a_id = incoming[nid][0]["source"]
            b_id = incoming[nid][1]["source"]
            a_var, b_var = _var_name(a_id), _var_name(b_id)
            if merge_ops[nid] == "add":
                forward_lines.append(f"        {var} = {a_var} + {b_var}")
            else:
                forward_lines.append(f"        {var} = torch.cat([{a_var}, {b_var}], dim=1)")
            continue

        prev_shape = shape_before(nid)
        input_var = "x" if nid == source_id else _var_name(incoming[nid][0]["source"])

        if ltype == "conv2d":
            init_lines.append(
                f"        self.{attr} = nn.Conv2d({prev_shape['channels']}, "
                f"{int(params.get('out_channels', 16))}, "
                f"kernel_size={int(params.get('kernel_size', 3))}, "
                f"stride={int(params.get('stride', 1))}, "
                f"padding={int(params.get('padding', 1))})"
            )
        elif ltype == "batchnorm2d":
            init_lines.append(f"        self.{attr} = nn.BatchNorm2d({prev_shape['channels']})")
        elif ltype == "linear":
            if prev_shape["mode"] == "image":
                in_features = prev_shape["channels"] * prev_shape["h"] * prev_shape["w"]
                init_lines.append(
                    f"        self.{attr} = nn.Sequential(nn.Flatten(), "
                    f"nn.Linear({in_features}, {int(params.get('out_features', 128))}))"
                )
            else:
                init_lines.append(
                    f"        self.{attr} = nn.Linear({prev_shape['features']}, "
                    f"{int(params.get('out_features', 128))})"
                )
        elif ltype in _SHAPELESS_LAYER_REPR:
            init_lines.append(f"        self.{attr} = {_SHAPELESS_LAYER_REPR[ltype](params)}")

        forward_lines.append(f"        {var} = self.{attr}({input_var})")

    sink_var = _var_name(sink_id)
    model_lines = "\n".join(init_lines)
    forward_body = "\n".join(forward_lines)
    loss_fn = "nn.NLLLoss()" if ends_with_logsoftmax else "nn.CrossEntropyLoss()"
    opt_ctor = "torch.optim.Adam" if optimizer_name == "adam" else "torch.optim.SGD"
    opt_extra = "" if optimizer_name == "adam" else ", momentum=0.9"

    warning_comment = ""
    if warnings:
        warning_comment = "\n".join(f"# note: {w}" for w in warnings) + "\n"

    return f'''\
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

transform = transforms.ToTensor()
train_set = torchvision.datasets.{dataset_cls}(root="./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.{dataset_cls}(root="./data", train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size={batch_size}, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size={batch_size})

{warning_comment}class Net(nn.Module):
    def __init__(self):
        super().__init__()
{model_lines}

    def forward(self, x):
{forward_body}
        return {sink_var}

model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = {opt_ctor}(model.parameters(), lr={lr}{opt_extra})
criterion = {loss_fn}

for epoch in range({epochs}):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    print(f"epoch {{epoch + 1}} done, last batch loss={{loss.item():.4f}}")

# quick test-set accuracy
model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f"test accuracy: {{correct / total:.4f}}")
'''
