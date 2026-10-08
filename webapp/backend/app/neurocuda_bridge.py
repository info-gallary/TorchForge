"""Bridges the webapp to the REAL NeuroCUDA compiler (../../compiler),
for the one part of a trained model NeuroLang can actually express: a
trailing chain of Linear (+ ReLU) layers with no branching.

This is an intentionally honest, bounded integration. NeuroCUDA v1 has
no convolution kernels and no autograd, so it cannot run an entire
CNN's forward pass, let alone its training -- and this module does not
pretend otherwise. What it genuinely does:

  1. Finds the maximal trailing Linear/ReLU suffix of the trained model
     (if any -- a model ending in Conv2d or a merge layer has none).
  2. Dumps that suffix's already-trained weights/biases and one real
     batch of entry activations to .nct files (benchmarks/tensor_io.py's
     format).
  3. Generates real NeuroLang source referencing them via `load(...)`,
     and runs it through the exact same lexer/parser/semantic/IR/
     optimizer/codegen pipeline `python -m src.cli run` uses.
  4. Compiles with nvcc and executes the resulting kernels on the GPU.
  5. Compares the output against PyTorch's own forward pass for that
     same suffix, on that same data, within a stated tolerance.

If step 1 finds nothing eligible, or nvcc isn't available, this reports
that honestly (`attempted: False`) instead of fabricating a result.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

COMPILER_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "compiler"
if str(COMPILER_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPILER_ROOT))

from benchmarks.tensor_io import load_nct, save_nct  # noqa: E402
from src.cli import RUNTIME_DIR, find_msvc_bin_dir, run_pipeline  # noqa: E402
from src.errors import NeuroLangError  # noqa: E402

ELIGIBLE_TYPES = {"linear", "relu"}


def find_trailing_linear_chain(nodes_by_id, incoming, sink_id):
    """Walks backward from the sink. A node extends the eligible suffix
    if it's Linear/ReLU and has exactly one incoming edge (so it isn't
    the graph's single source and isn't a merge layer). Whether its
    predecessor also fans out to other branches elsewhere is irrelevant
    -- this only needs that predecessor's own output tensor, which is
    captured regardless of who else also uses it.

    A trailing LogSoftmax (the standard final layer of a classifier) is
    skipped over rather than treated as disqualifying: NeuroCUDA has no
    softmax kernel, so LogSoftmax itself always keeps running on
    PyTorch, but that shouldn't block accelerating the real Linear/ReLU
    trunk feeding it -- otherwise this would almost never find anything
    in practice, since LogSoftmax is nearly always the very last layer.

    Returns (entry_node_id, suffix_ids_in_forward_order, skipped_final_activation),
    where skipped_final_activation is the LogSoftmax node id if one was
    skipped, else None. Returns (None, [], None) if nothing is eligible.
    """
    nid = sink_id
    skipped_final_activation = None
    if nodes_by_id[nid]["type"] == "logsoftmax" and len(incoming[nid]) == 1:
        skipped_final_activation = nid
        nid = incoming[nid][0]["source"]

    suffix = []
    while nodes_by_id[nid]["type"] in ELIGIBLE_TYPES and len(incoming[nid]) == 1:
        suffix.append(nid)
        nid = incoming[nid][0]["source"]

    suffix.reverse()
    if not suffix or not any(nodes_by_id[n]["type"] == "linear" for n in suffix):
        return None, [], None
    return nid, suffix, skipped_final_activation


def _nvcc_env():
    env = os.environ.copy()
    if sys.platform == "win32" and not shutil.which("cl"):
        cl_dir = find_msvc_bin_dir()
        if cl_dir:
            env["PATH"] = cl_dir + os.pathsep + env.get("PATH", "")
    return env


def try_accelerate(model, order_ids, nodes_by_id, incoming, sink_id,
                    sample_batch, device, workdir: Path) -> dict:
    """Never raises for "NeuroCUDA doesn't apply here" -- only reports it.
    `sample_batch` is one batch of (images, labels) from a DataLoader.
    """
    entry_id, suffix_ids, skipped_final_activation = find_trailing_linear_chain(
        nodes_by_id, incoming, sink_id
    )
    if not suffix_ids:
        return {
            "attempted": False,
            "reason": (
                "This architecture's final layer(s) aren't a plain Linear/ReLU chain "
                "(NeuroCUDA v1 has no convolution or branching support), so 0 layers "
                "were accelerated with real NeuroCUDA this run."
            ),
        }

    if not shutil.which("nvcc"):
        return {"attempted": False, "reason": "nvcc not found on PATH -- skipped."}

    workdir.mkdir(parents=True, exist_ok=True)
    images, _labels = sample_batch
    images = images.to(device)

    model.eval()
    with torch.no_grad():
        values = {}
        for nid in order_ids:
            if nid in model.merge_ops:
                a = values[model.incoming[nid][0]["source"]]
                b = values[model.incoming[nid][1]["source"]]
                values[nid] = a + b if model.merge_ops[nid] == "add" else torch.cat([a, b], dim=1)
            elif nid == model.source_id:
                values[nid] = model.mod_dict[nid](images)
            else:
                (pred,) = model.incoming[nid]
                values[nid] = model.mod_dict[nid](values[pred["source"]])
            if nid == entry_id:
                break

        entry_activation = values[entry_id]
        for nid in suffix_ids:
            (pred,) = model.incoming[nid]
            values[nid] = model.mod_dict[nid](values[pred["source"]])
        pytorch_output = values[suffix_ids[-1]].cpu().numpy()

    entry_np = (
        entry_activation.reshape(entry_activation.shape[0], -1)
        .cpu().numpy().astype(np.float32)
    )
    batch_size = entry_np.shape[0]

    lines = []
    x_path = workdir / "X.nct"
    save_nct(str(x_path), entry_np)
    cur_name = "X"
    lines.append(f'tensor X = load("{x_path.as_posix()}")')

    for i, nid in enumerate(suffix_ids):
        node = nodes_by_id[nid]
        module = model.mod_dict[nid]
        if node["type"] == "linear":
            lin = module[-1] if isinstance(module, nn.Sequential) else module
            weight_t = lin.weight.detach().cpu().numpy().T.astype(np.float32)  # (in, out)
            w_path = workdir / f"W{i}.nct"
            save_nct(str(w_path), weight_t)
            w_name = f"W{i}"
            lines.append(f'tensor {w_name} = load("{w_path.as_posix()}")')
            mm_name = f"T{i}_mm"
            lines.append(f"{mm_name} = matmul({cur_name}, {w_name})")
            cur_name = mm_name
            if lin.bias is not None:
                bias = lin.bias.detach().cpu().numpy().astype(np.float32)
                bias_mat = np.tile(bias, (batch_size, 1))
                b_path = workdir / f"B{i}.nct"
                save_nct(str(b_path), bias_mat)
                b_name = f"B{i}"
                lines.append(f'tensor {b_name} = load("{b_path.as_posix()}")')
                add_name = f"T{i}_add"
                lines.append(f"{add_name} = add({cur_name}, {b_name})")
                cur_name = add_name
        else:  # relu
            relu_name = f"T{i}_relu"
            lines.append(f"{relu_name} = relu({cur_name})")
            cur_name = relu_name

    lines.append(f"print {cur_name}")
    source = "\n".join(lines) + "\n"

    try:
        result = run_pipeline(source, "<neurocuda-bridge>", optimize_ir=True)
    except NeuroLangError as e:
        return {"attempted": True, "success": False, "error": str(e), "source": source}

    cu_path = workdir / "accel.cu"
    exe_path = workdir / ("accel.exe" if sys.platform == "win32" else "accel")
    cu_path.write_text(result.codegen.source, encoding="utf-8")

    compile_cmd = ["nvcc", "-O3", "-I", str(RUNTIME_DIR), "-o", str(exe_path), str(cu_path)]
    proc = subprocess.run(
        compile_cmd, cwd=workdir, env=_nvcc_env(), capture_output=True, text=True, timeout=180
    )
    if proc.returncode != 0:
        return {
            "attempted": True, "success": False,
            "error": "nvcc failed:\n" + proc.stdout + proc.stderr, "source": source,
        }

    run_proc = subprocess.run(
        [str(exe_path)], cwd=workdir, capture_output=True, text=True, timeout=60
    )
    if run_proc.returncode != 0:
        return {
            "attempted": True, "success": False,
            "error": "program failed:\n" + run_proc.stdout + run_proc.stderr, "source": source,
        }

    neurocuda_output = load_nct(str(workdir / f"{cur_name}.nct"))
    max_abs_diff = float(np.max(np.abs(neurocuda_output - pytorch_output)))
    matches = bool(np.allclose(neurocuda_output, pytorch_output, atol=1e-2, rtol=1e-3))

    note = None
    if skipped_final_activation:
        note = (
            f"The final LogSoftmax layer ('{skipped_final_activation}') still runs on "
            f"PyTorch -- NeuroCUDA v1 has no softmax kernel. Everything before it in this "
            f"chain ran on NeuroCUDA."
        )

    return {
        "attempted": True,
        "success": True,
        "layers_accelerated": len(suffix_ids),
        "layer_types": [nodes_by_id[n]["type"] for n in suffix_ids],
        "matches_pytorch": matches,
        "max_abs_diff": max_abs_diff,
        "kernels_used": result.codegen.kernels_used,
        "source": source,
        "stdout": run_proc.stdout,
        "note": note,
    }
