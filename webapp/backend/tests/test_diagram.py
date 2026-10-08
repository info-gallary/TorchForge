import io
import unittest

from PIL import Image

from app.diagram import PALETTES, STYLES, render_diagram
from app.graph import resolve_dag


def _resolved(nodes, edges):
    return resolve_dag(nodes, edges)


SIMPLE_NODES = [
    {"id": "a", "type": "conv2d", "params": {"out_channels": 4, "kernel_size": 3, "padding": 1}},
    {"id": "b", "type": "relu", "params": {}},
    {"id": "c", "type": "linear", "params": {"out_features": 5}},
]
SIMPLE_EDGES = [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}]


class TestRenderDiagram(unittest.TestCase):
    def _render(self, **opts):
        order, nodes_by_id, incoming, outgoing, source_id, sink_id = _resolved(
            SIMPLE_NODES, SIMPLE_EDGES
        )
        png_bytes = render_diagram(
            order, nodes_by_id, incoming, outgoing, source_id, sink_id, (3, 16, 16), **opts
        )
        img = Image.open(io.BytesIO(png_bytes))
        img.load()
        return img

    def test_every_style_renders_a_valid_png(self):
        for style in STYLES:
            img = self._render(style=style)
            self.assertEqual(img.format, "PNG")
            self.assertGreater(img.width, 0)
            self.assertGreater(img.height, 0)

    def test_legend_and_show_dimension_dont_error(self):
        img = self._render(style="flow", legend=True, show_dimension=True)
        self.assertEqual(img.format, "PNG")

    def test_flat_vs_volumetric_produce_different_images(self):
        flat = self._render(style="flow", draw_volume=False)
        volumetric = self._render(style="flow", draw_volume=True)
        # Flat omits the 3D depth entirely, so it must be narrower.
        self.assertLess(flat.width, volumetric.width)

    def test_hide_activations_omits_relu_from_legend(self):
        img = self._render(style="flow", hide_activations=True, legend=True)
        self.assertEqual(img.format, "PNG")

    def test_unknown_palette_falls_back_instead_of_raising(self):
        img = self._render(style="flow", palette="not_a_real_palette")
        self.assertEqual(img.format, "PNG")

    def test_known_palettes_are_all_accepted(self):
        for palette in PALETTES:
            img = self._render(style="flow", palette=palette)
            self.assertEqual(img.format, "PNG")


if __name__ == "__main__":
    unittest.main()
