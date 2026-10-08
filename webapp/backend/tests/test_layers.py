import unittest

import torch

from app.graph import GraphError, resolve_dag
from app.layers import build_dag


def _build(nodes, edges, input_shape=(3, 32, 32)):
    order, nodes_by_id, incoming, outgoing, source_id, sink_id = resolve_dag(nodes, edges)
    return build_dag(order, nodes_by_id, incoming, outgoing, source_id, sink_id, input_shape)


class TestShapeInferenceSequential(unittest.TestCase):
    def test_conv_preserves_spatial_with_padding(self):
        nodes = [{"id": "a", "type": "conv2d", "params": {"out_channels": 8, "kernel_size": 3, "padding": 1}}]
        model, shapes, warnings, ends_logsoftmax = _build(nodes, [])
        self.assertEqual(shapes["a"], {"mode": "image", "channels": 8, "h": 32, "w": 32})
        self.assertEqual(warnings, [])
        self.assertFalse(ends_logsoftmax)

    def test_maxpool_halves_spatial(self):
        nodes = [
            {"id": "a", "type": "conv2d", "params": {"out_channels": 4, "kernel_size": 3, "padding": 1}},
            {"id": "b", "type": "maxpool2d", "params": {"kernel_size": 2, "stride": 2}},
        ]
        edges = [{"source": "a", "target": "b"}]
        _model, shapes, _w, _e = _build(nodes, edges)
        self.assertEqual(shapes["b"], {"mode": "image", "channels": 4, "h": 16, "w": 16})

    def test_implicit_flatten_before_linear(self):
        nodes = [
            {"id": "a", "type": "conv2d", "params": {"out_channels": 4, "kernel_size": 3, "padding": 1}},
            {"id": "b", "type": "linear", "params": {"out_features": 10}},
        ]
        edges = [{"source": "a", "target": "b"}]
        _model, shapes, warnings, _e = _build(nodes, edges)
        self.assertEqual(shapes["b"], {"mode": "flat", "features": 10})
        self.assertTrue(any("implicit Flatten" in w for w in warnings))

    def test_pool_shrinking_to_zero_raises(self):
        nodes = [{"id": "a", "type": "maxpool2d", "params": {"kernel_size": 40, "stride": 40}}]
        with self.assertRaises(GraphError):
            _build(nodes, [], input_shape=(3, 32, 32))

    def test_ends_with_logsoftmax_detected(self):
        nodes = [
            {"id": "a", "type": "linear", "params": {"out_features": 10}},
            {"id": "b", "type": "logsoftmax", "params": {}},
        ]
        edges = [{"source": "a", "target": "b"}]
        _model, _s, _w, ends_logsoftmax = _build(nodes, edges)
        self.assertTrue(ends_logsoftmax)


class TestShapeInferenceMerge(unittest.TestCase):
    def test_add_merge_requires_equal_shapes(self):
        nodes = [
            {"id": "a", "type": "conv2d", "params": {"out_channels": 4, "kernel_size": 3, "padding": 1}},
            {"id": "b", "type": "conv2d", "params": {"out_channels": 8, "kernel_size": 3, "padding": 1}},
            {"id": "m", "type": "add_merge", "params": {}},
        ]
        edges = [
            {"source": "a", "target": "m", "targetHandle": "in0"},
            {"source": "b", "target": "m", "targetHandle": "in1"},
        ]
        with self.assertRaises(GraphError):
            _build(nodes, edges)

    def test_add_merge_with_equal_shapes_succeeds(self):
        nodes = [
            {"id": "src", "type": "conv2d", "params": {"out_channels": 4, "kernel_size": 3, "padding": 1}},
            {"id": "a", "type": "conv2d", "params": {"out_channels": 8, "kernel_size": 3, "padding": 1}},
            {"id": "b", "type": "conv2d", "params": {"out_channels": 8, "kernel_size": 3, "padding": 1}},
            {"id": "m", "type": "add_merge", "params": {}},
        ]
        edges = [
            {"source": "src", "target": "a"},
            {"source": "src", "target": "b"},
            {"source": "a", "target": "m", "targetHandle": "in0"},
            {"source": "b", "target": "m", "targetHandle": "in1"},
        ]
        _model, shapes, _w, _e = _build(nodes, edges)
        self.assertEqual(shapes["m"], {"mode": "image", "channels": 8, "h": 32, "w": 32})

    def test_concat_merge_sums_channels(self):
        nodes = [
            {"id": "src", "type": "conv2d", "params": {"out_channels": 4, "kernel_size": 3, "padding": 1}},
            {"id": "a", "type": "conv2d", "params": {"out_channels": 8, "kernel_size": 3, "padding": 1}},
            {"id": "b", "type": "conv2d", "params": {"out_channels": 5, "kernel_size": 3, "padding": 1}},
            {"id": "m", "type": "concat_merge", "params": {}},
        ]
        edges = [
            {"source": "src", "target": "a"},
            {"source": "src", "target": "b"},
            {"source": "a", "target": "m", "targetHandle": "in0"},
            {"source": "b", "target": "m", "targetHandle": "in1"},
        ]
        _model, shapes, _w, _e = _build(nodes, edges)
        self.assertEqual(shapes["m"], {"mode": "image", "channels": 13, "h": 32, "w": 32})

    def test_dag_model_forward_runs_and_matches_expected_shape(self):
        nodes = [
            {"id": "src", "type": "conv2d", "params": {"out_channels": 4, "kernel_size": 3, "padding": 1}},
            {"id": "a", "type": "conv2d", "params": {"out_channels": 6, "kernel_size": 3, "padding": 1}},
            {"id": "b", "type": "conv2d", "params": {"out_channels": 6, "kernel_size": 3, "padding": 1}},
            {"id": "m", "type": "add_merge", "params": {}},
            {"id": "fc", "type": "linear", "params": {"out_features": 10}},
        ]
        edges = [
            {"source": "src", "target": "a"},
            {"source": "src", "target": "b"},
            {"source": "a", "target": "m", "targetHandle": "in0"},
            {"source": "b", "target": "m", "targetHandle": "in1"},
            {"source": "m", "target": "fc"},
        ]
        model, _shapes, _w, _e = _build(nodes, edges, input_shape=(3, 16, 16))
        x = torch.randn(2, 3, 16, 16)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))


if __name__ == "__main__":
    unittest.main()
