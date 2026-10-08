"""Tests for the trailing-Linear/ReLU-suffix detection (pure logic, no
GPU) and, when nvcc is available, a real end-to-end acceleration run.
"""

import shutil
import unittest
from pathlib import Path

from app.graph import resolve_dag
from app.layers import build_dag
from app.neurocuda_bridge import find_trailing_linear_chain, try_accelerate


class TestFindTrailingLinearChain(unittest.TestCase):
    def _resolve(self, nodes, edges):
        order, nodes_by_id, incoming, outgoing, source_id, sink_id = resolve_dag(nodes, edges)
        return nodes_by_id, incoming, sink_id

    def test_pure_linear_relu_chain_fully_eligible(self):
        nodes = [
            {"id": "a", "type": "conv2d", "params": {}},
            {"id": "fc1", "type": "linear", "params": {}},
            {"id": "r1", "type": "relu", "params": {}},
            {"id": "fc2", "type": "linear", "params": {}},
        ]
        edges = [
            {"source": "a", "target": "fc1"},
            {"source": "fc1", "target": "r1"},
            {"source": "r1", "target": "fc2"},
        ]
        nodes_by_id, incoming, sink_id = self._resolve(nodes, edges)
        entry_id, suffix, skipped = find_trailing_linear_chain(nodes_by_id, incoming, sink_id)
        self.assertEqual(entry_id, "a")
        self.assertEqual(suffix, ["fc1", "r1", "fc2"])
        self.assertIsNone(skipped)

    def test_trailing_logsoftmax_is_skipped_not_disqualifying(self):
        nodes = [
            {"id": "a", "type": "conv2d", "params": {}},
            {"id": "fc1", "type": "linear", "params": {}},
            {"id": "out", "type": "logsoftmax", "params": {}},
        ]
        edges = [{"source": "a", "target": "fc1"}, {"source": "fc1", "target": "out"}]
        nodes_by_id, incoming, sink_id = self._resolve(nodes, edges)
        entry_id, suffix, skipped = find_trailing_linear_chain(nodes_by_id, incoming, sink_id)
        self.assertEqual(entry_id, "a")
        self.assertEqual(suffix, ["fc1"])
        self.assertEqual(skipped, "out")

    def test_conv_only_model_has_nothing_eligible(self):
        nodes = [{"id": "a", "type": "conv2d", "params": {}}, {"id": "b", "type": "relu", "params": {}}]
        edges = [{"source": "a", "target": "b"}]
        nodes_by_id, incoming, sink_id = self._resolve(nodes, edges)
        entry_id, suffix, skipped = find_trailing_linear_chain(nodes_by_id, incoming, sink_id)
        self.assertIsNone(entry_id)
        self.assertEqual(suffix, [])

    def test_relu_only_suffix_without_any_linear_is_not_eligible(self):
        # A trailing ReLU with no Linear anywhere in the suffix isn't worth
        # accelerating (nothing for NeuroCUDA's matmul to do).
        nodes = [{"id": "a", "type": "conv2d", "params": {}}, {"id": "b", "type": "relu", "params": {}}]
        edges = [{"source": "a", "target": "b"}]
        nodes_by_id, incoming, sink_id = self._resolve(nodes, edges)
        entry_id, suffix, _skipped = find_trailing_linear_chain(nodes_by_id, incoming, sink_id)
        self.assertIsNone(entry_id)
        self.assertEqual(suffix, [])

    def test_branching_entry_point_is_fine(self):
        # The suffix's entry point may itself be a merge node, or a node
        # that also fans out elsewhere -- only the suffix nodes themselves
        # need exactly one incoming edge.
        nodes = [
            {"id": "a", "type": "conv2d", "params": {}},
            {"id": "b", "type": "conv2d", "params": {}},
            {"id": "m", "type": "add_merge", "params": {}},
            {"id": "fc1", "type": "linear", "params": {}},
        ]
        edges = [
            {"source": "a", "target": "b"},  # arbitrary predecessor chain
            {"source": "a", "target": "m", "targetHandle": "in0"},
            {"source": "b", "target": "m", "targetHandle": "in1"},
            {"source": "m", "target": "fc1"},
        ]
        nodes_by_id, incoming, sink_id = self._resolve(nodes, edges)
        entry_id, suffix, _skipped = find_trailing_linear_chain(nodes_by_id, incoming, sink_id)
        self.assertEqual(entry_id, "m")
        self.assertEqual(suffix, ["fc1"])


@unittest.skipUnless(shutil.which("nvcc"), "nvcc not found on PATH; skipping GPU bridge test")
class TestTryAccelerateGPU(unittest.TestCase):
    def setUp(self):
        self.workdir = Path(__file__).resolve().parent.parent.parent / "data" / "_test_bridge_gpu"

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_accelerates_and_matches_pytorch(self):
        import torch

        nodes = [
            {"id": "a", "type": "conv2d", "params": {"out_channels": 4, "kernel_size": 3, "padding": 1}},
            {"id": "fc1", "type": "linear", "params": {"out_features": 16}},
            {"id": "r1", "type": "relu", "params": {}},
            {"id": "fc2", "type": "linear", "params": {"out_features": 10}},
            {"id": "out", "type": "logsoftmax", "params": {}},
        ]
        edges = [
            {"source": "a", "target": "fc1"},
            {"source": "fc1", "target": "r1"},
            {"source": "r1", "target": "fc2"},
            {"source": "fc2", "target": "out"},
        ]
        order, nodes_by_id, incoming, outgoing, source_id, sink_id = resolve_dag(nodes, edges)
        model, _shapes, _w, _e = build_dag(
            order, nodes_by_id, incoming, outgoing, source_id, sink_id, (1, 8, 8)
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        images = torch.randn(5, 1, 8, 8)
        labels = torch.zeros(5, dtype=torch.long)

        result = try_accelerate(
            model, order, nodes_by_id, incoming, sink_id, (images, labels), device, self.workdir
        )
        self.assertTrue(result["attempted"])
        self.assertTrue(result.get("success"), result.get("error"))
        self.assertEqual(result["layers_accelerated"], 3)
        self.assertTrue(result["matches_pytorch"])
        self.assertLess(result["max_abs_diff"], 1e-2)
        self.assertEqual(result["note"] and "LogSoftmax" in result["note"], True)


if __name__ == "__main__":
    unittest.main()
