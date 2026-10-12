// One-click demo architectures, each laid out with explicit positions
// so it reads clearly the moment it's loaded -- pick "Load example"
// in the header instead of building something from scratch first.
export const EXAMPLES = {
  // A plain, fully sequential CNN -- no branching, the classic
  // conv/relu/pool x2 -> flatten -> dense head shape. The easiest one
  // to explain in a demo: "just a stack of layers."
  simple_cnn_mnist: {
    label: "Simple CNN (MNIST)",
    dataset: "mnist",
    nodes: [
      { id: "conv1", type: "conv2d", params: { out_channels: 8, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 40, y: 220 } },
      { id: "relu1", type: "relu", params: {}, position: { x: 280, y: 220 } },
      { id: "pool1", type: "maxpool2d", params: { kernel_size: 2, stride: 2 }, position: { x: 520, y: 220 } },
      { id: "conv2", type: "conv2d", params: { out_channels: 16, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 760, y: 220 } },
      { id: "relu2", type: "relu", params: {}, position: { x: 1000, y: 220 } },
      { id: "pool2", type: "maxpool2d", params: { kernel_size: 2, stride: 2 }, position: { x: 1240, y: 220 } },
      { id: "fc1", type: "linear", params: { out_features: 64 }, position: { x: 1480, y: 220 } },
      { id: "relu3", type: "relu", params: {}, position: { x: 1720, y: 220 } },
      { id: "fc2", type: "linear", params: { out_features: 10 }, position: { x: 1960, y: 220 } },
      { id: "out", type: "logsoftmax", params: {}, position: { x: 2200, y: 220 } },
    ],
    edges: [
      { source: "conv1", target: "relu1" },
      { source: "relu1", target: "pool1" },
      { source: "pool1", target: "conv2" },
      { source: "conv2", target: "relu2" },
      { source: "relu2", target: "pool2" },
      { source: "pool2", target: "fc1" },
      { source: "fc1", target: "relu3" },
      { source: "relu3", target: "fc2" },
      { source: "fc2", target: "out" },
    ],
  },
  // A residual-style skip connection: Conv2D branches into two
  // parallel Conv2D+ReLU paths, merged back with Add, then a small
  // Linear/ReLU/Linear head. The exact architecture used to validate
  // the NeuroCUDA bridge during development: trained to 92.4% on MNIST
  // in one epoch, with its 3-layer head compiled and run through real
  // NeuroCUDA CUDA kernels, matching PyTorch's output to within 3.2e-5.
  residual_mnist: {
    label: "Residual branch (MNIST)",
    dataset: "mnist",
    nodes: [
      { id: "conv1", type: "conv2d", params: { out_channels: 8, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 40, y: 260 } },
      { id: "conv2a", type: "conv2d", params: { out_channels: 8, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 320, y: 80 } },
      { id: "relu_a", type: "relu", params: {}, position: { x: 600, y: 80 } },
      { id: "conv2b", type: "conv2d", params: { out_channels: 8, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 320, y: 440 } },
      { id: "relu_b", type: "relu", params: {}, position: { x: 600, y: 440 } },
      { id: "merge", type: "add_merge", params: {}, position: { x: 880, y: 260 } },
      { id: "fc1", type: "linear", params: { out_features: 32 }, position: { x: 1160, y: 260 } },
      { id: "relu_fc", type: "relu", params: {}, position: { x: 1440, y: 260 } },
      { id: "fc2", type: "linear", params: { out_features: 10 }, position: { x: 1720, y: 260 } },
      { id: "out", type: "logsoftmax", params: {}, position: { x: 2000, y: 260 } },
    ],
    edges: [
      { source: "conv1", target: "conv2a" },
      { source: "conv1", target: "conv2b" },
      { source: "conv2a", target: "relu_a" },
      { source: "conv2b", target: "relu_b" },
      { source: "relu_a", target: "merge", targetHandle: "in0" },
      { source: "relu_b", target: "merge", targetHandle: "in1" },
      { source: "merge", target: "fc1" },
      { source: "fc1", target: "relu_fc" },
      { source: "relu_fc", target: "fc2" },
      { source: "fc2", target: "out" },
    ],
  },
};
