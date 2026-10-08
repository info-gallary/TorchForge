"""Architecture diagram export via the real `visualtorch` package
(github.com/willyfh/visualtorch, PyPI: visualtorch) -- the library the
user's reference post was about. Builds an (untrained -- random-init
weights are fine, only shapes matter for a diagram) DAGModel from the
canvas graph and renders it as a PNG.
"""

import io

import visualtorch

from app.layers import build_dag


def render_diagram(order_ids, nodes_by_id, incoming, outgoing, source_id, sink_id,
                    input_shape: tuple, style: str = "flow") -> bytes:
    model, _shapes, _warnings, _ends = build_dag(
        order_ids, nodes_by_id, incoming, outgoing, source_id, sink_id, input_shape
    )
    model.eval()

    channels, h, w = input_shape
    img = visualtorch.render(model, input_shape=(1, channels, h, w), style=style)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
