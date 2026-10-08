"""Reads just the header of a .nct binary tensor file (see
runtime/tensor_io.hpp for the full format) -- magic + rows + cols --
without loading the (potentially large) float32 payload.

Used only by semantic analysis, so that `tensor W = load("w.nct")` can
infer W's shape from the file itself, the same way `random(rows, cols)`
and `zeros(rows, cols)` get their shape from literal arguments. Kept
dependency-free (pure stdlib `struct`), unlike benchmarks/tensor_io.py,
which needs NumPy and is not imported by the core compiler.
"""

import struct

MAGIC = b"NCT1"


def read_nct_shape(path: str) -> tuple[int, int]:
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != MAGIC:
                raise ValueError(f"'{path}' is not a valid .nct tensor file (bad magic {magic!r})")
            rows, cols = struct.unpack("<ii", f.read(8))
            return rows, cols
    except FileNotFoundError:
        raise ValueError(f"'{path}' does not exist") from None
    except OSError as e:
        raise ValueError(f"could not read '{path}': {e}") from None
