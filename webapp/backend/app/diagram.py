"""Architecture diagram export via the real `visualtorch` package
(github.com/willyfh/visualtorch, PyPI: visualtorch) -- the library the
user's reference post was about. Builds an (untrained -- random-init
weights are fine, only shapes matter for a diagram) DAGModel from the
canvas graph and renders it as a PNG.

Exposes a curated slice of visualtorch's own options (style, legend,
dimension labels, palette, flat-vs-3D, hiding activation layers) rather
than its full kwarg surface, since the frontend offers these as plain
toggles/dropdowns, not a raw Python kwargs editor.
"""

import io

import torch.nn as nn
import visualtorch

from app.layers import build_dag

PALETTES = list(visualtorch.PALETTES.keys())
STYLES = ("flow", "graph", "lenet")

_ACTIVATION_TYPES = (nn.ReLU, nn.LeakyReLU)


def render_diagram(
    order_ids, nodes_by_id, incoming, outgoing, source_id, sink_id, input_shape: tuple,
    *, style: str = "flow", legend: bool = False, show_dimension: bool = False,
    palette: str = "okabe_ito", draw_volume: bool = True, hide_activations: bool = False,
) -> bytes:
    model, _shapes, _warnings, _ends = build_dag(
        order_ids, nodes_by_id, incoming, outgoing, source_id, sink_id, input_shape
    )
    model.eval()

    channels, h, w = input_shape
    kwargs = {
        "legend": legend,
        "show_dimension": show_dimension,
        "palette": palette if palette in PALETTES else "okabe_ito",
    }
    if style == "flow":
        kwargs["draw_volume"] = draw_volume
    if hide_activations:
        kwargs["type_ignore"] = list(_ACTIVATION_TYPES)

    img = visualtorch.render(model, input_shape=(1, channels, h, w), style=style, **kwargs)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
