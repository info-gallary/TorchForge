"""Reader/writer for the .nct binary tensor format shared with generated
CUDA programs (see runtime/tensor_io.hpp): magic b"NCT1", int32 rows,
int32 cols, then rows*cols float32 values in row-major order.

`load_nct` lets benchmarks and correctness-check tests load the exact
input/output tensors a compiled NeuroCUDA program used, to compare
against a NumPy/PyTorch reference computed on the same data.
`save_nct` is the other direction: it lets external data -- e.g. a
trained PyTorch model's weight matrices -- be written in this format so
a NeuroLang `load("path.nct")` can read it into a real compiled CUDA
program (see webapp/backend/app/neurocuda_bridge.py).
"""

import struct

import numpy as np

MAGIC = b"NCT1"


def load_nct(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"{path}: not a .nct tensor file (bad magic {magic!r})")
        rows, cols = struct.unpack("<ii", f.read(8))
        data = f.read(rows * cols * 4)
        if len(data) != rows * cols * 4:
            raise ValueError(f"{path}: truncated tensor data")
        arr = np.frombuffer(data, dtype="<f4").reshape(rows, cols)
        return arr.copy()


def save_nct(path: str, array) -> None:
    arr = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
    if arr.ndim != 2:
        raise ValueError(f"save_nct expects a 2D array, got shape {arr.shape}")
    rows, cols = arr.shape
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<ii", rows, cols))
        f.write(arr.tobytes())
