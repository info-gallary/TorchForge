import unittest

from app.graph import GraphError, resolve_dag


class TestResolveDAGSequential(unittest.TestCase):
    def test_simple_chain(self):
        nodes = [{"id": "a", "type": "conv2d"}, {"id": "b", "type": "relu"}]
        edges = [{"source": "a", "target": "b"}]
        order, nodes_by_id, incoming, outgoing, source_id, sink_id = resolve_dag(nodes, edges)
        self.assertEqual(order, ["a", "b"])
        self.assertEqual(source_id, "a")
        self.assertEqual(sink_id, "b")

    def test_empty_graph_raises(self):
        with self.assertRaises(GraphError):
            resolve_dag([], [])

    def test_disconnected_node_raises(self):
        nodes = [{"id": "a", "type": "conv2d"}, {"id": "b", "type": "relu"}]
        with self.assertRaises(GraphError):
            resolve_dag(nodes, [])  # two separate sources, no edges

    def test_cycle_raises(self):
        nodes = [{"id": "a", "type": "relu"}, {"id": "b", "type": "relu"}]
        edges = [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}]
        with self.assertRaises(GraphError):
            resolve_dag(nodes, edges)

    def test_dangling_edge_raises(self):
        nodes = [{"id": "a", "type": "relu"}]
        edges = [{"source": "a", "target": "ghost"}]
        with self.assertRaises(GraphError):
            resolve_dag(nodes, edges)


class TestResolveDAGBranching(unittest.TestCase):
    def _branching_nodes_edges(self):
        nodes = [
            {"id": "a", "type": "conv2d"},
            {"id": "b1", "type": "relu"},
            {"id": "b2", "type": "relu"},
            {"id": "merge", "type": "add_merge"},
        ]
        edges = [
            {"source": "a", "target": "b1"},
            {"source": "a", "target": "b2"},
            {"source": "b1", "target": "merge", "targetHandle": "in0"},
            {"source": "b2", "target": "merge", "targetHandle": "in1"},
        ]
        return nodes, edges

    def test_fan_out_then_merge(self):
        nodes, edges = self._branching_nodes_edges()
        order, nodes_by_id, incoming, outgoing, source_id, sink_id = resolve_dag(nodes, edges)
        self.assertEqual(source_id, "a")
        self.assertEqual(sink_id, "merge")
        # topological order must place both branches before the merge
        self.assertLess(order.index("b1"), order.index("merge"))
        self.assertLess(order.index("b2"), order.index("merge"))

    def test_merge_predecessors_ordered_by_handle(self):
        nodes, edges = self._branching_nodes_edges()
        _order, _nodes_by_id, incoming, _outgoing, _source_id, _sink_id = resolve_dag(nodes, edges)
        self.assertEqual(incoming["merge"][0]["source"], "b1")
        self.assertEqual(incoming["merge"][1]["source"], "b2")

    def test_merge_with_wrong_input_count_raises(self):
        nodes = [
            {"id": "a", "type": "conv2d"},
            {"id": "merge", "type": "add_merge"},
        ]
        edges = [{"source": "a", "target": "merge", "targetHandle": "in0"}]
        with self.assertRaises(GraphError):
            resolve_dag(nodes, edges)

    def test_regular_layer_with_two_inputs_raises(self):
        nodes = [
            {"id": "a", "type": "conv2d"},
            {"id": "b", "type": "conv2d"},
            {"id": "c", "type": "relu"},
        ]
        edges = [{"source": "a", "target": "c"}, {"source": "b", "target": "c"}]
        with self.assertRaises(GraphError):
            resolve_dag(nodes, edges)

    def test_two_sinks_raises(self):
        nodes = [
            {"id": "a", "type": "conv2d"},
            {"id": "b1", "type": "relu"},
            {"id": "b2", "type": "relu"},
        ]
        edges = [{"source": "a", "target": "b1"}, {"source": "a", "target": "b2"}]
        with self.assertRaises(GraphError):
            resolve_dag(nodes, edges)


if __name__ == "__main__":
    unittest.main()
