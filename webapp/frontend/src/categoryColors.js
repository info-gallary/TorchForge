// Shared between NodePalette (the drag source) and LayerNode (the canvas
// card) so a layer's color is the same wherever it appears.
export const CATEGORY_COLORS = {
  conv: "#60a5fa",
  pool: "#a78bfa",
  norm: "#22d3ee",
  activation: "#fbbf24",
  regularization: "#fb7185",
  shape: "#94a3b8",
  dense: "#34d399",
  output: "#f472b6",
  merge: "#c084fc",
};

export function categoryColor(category) {
  return CATEGORY_COLORS[category] || "#94a3b8";
}
