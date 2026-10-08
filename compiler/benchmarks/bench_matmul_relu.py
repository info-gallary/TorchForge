"""Benchmark: NeuroCUDA's matmul+relu fusion vs. unfused vs. PyTorch.

    python -m benchmarks.bench_matmul_relu [--size 1024] [--repeat 20] [--warmup 5]

Compares three things on the exact same input data:
  1. NeuroCUDA, optimizer ON  -- a single fused `matmul_relu` kernel
  2. NeuroCUDA, optimizer OFF -- separate `matmul` then `relu` kernels
  3. PyTorch                  -- torch.relu(torch.matmul(A, B)) on the GPU

Correctness: all three are checked against a NumPy reference
(np.maximum(A @ B, 0)) within a stated tolerance -- naive/tiled CUDA
and cuBLAS/NumPy sum in different orders, so exact equality isn't
expected, but the values must agree closely.

Timing: NeuroCUDA's own binary does the warm-up and repeated,
synchronized CUDA-event timing internally (NEUROCUDA_WARMUP /
NEUROCUDA_REPEAT -- see src/cuda_codegen.py) and reports mean/min/max.
PyTorch is timed the same way in Python: warm-up iterations, then
repeated timed runs with torch.cuda.synchronize() around each one.
Compilation time (nvcc) is reported separately from GPU execution
time, per CLAUDE.md.

This script only ever prints numbers it actually measured on this
machine. NeuroCUDA's kernels are a teaching-grade naive/tiled
implementation; PyTorch's matmul calls into cuBLAS, a vendor-tuned
library used in production. PyTorch is very likely faster here, and if
it is, this script says so rather than hiding or spinning it --
CLAUDE.md is explicit that this project must never claim to outperform
or replace PyTorch generally.
"""

import argparse
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.tensor_io import load_nct  # noqa: E402
from src.cli import RUNTIME_DIR, find_msvc_bin_dir, run_pipeline  # noqa: E402

ATOL, RTOL = 1e-2, 1e-3


def _nvcc_env():
    env = os.environ.copy()
    if sys.platform == "win32" and not shutil.which("cl"):
        cl_dir = find_msvc_bin_dir()
        if cl_dir:
            env["PATH"] = cl_dir + os.pathsep + env.get("PATH", "")
    return env


def build_and_run_neurocuda(size: int, *, optimize_ir: bool, warmup: int, repeat: int,
                             seed: int, workdir: Path):
    source = (
        f"tensor A = random({size}, {size})\n"
        f"tensor B = random({size}, {size})\n"
        f"C = matmul(A, B)\n"
        f"D = relu(C)\n"
        f"print D\n"
    )
    result = run_pipeline(source, "<benchmark>", optimize_ir=optimize_ir)

    workdir.mkdir(parents=True, exist_ok=True)
    cu_path = workdir / "bench.cu"
    exe_path = workdir / ("bench.exe" if sys.platform == "win32" else "bench")
    cu_path.write_text(result.codegen.source, encoding="utf-8")

    compile_cmd = ["nvcc", "-O3", "-I", str(RUNTIME_DIR), "-o", str(exe_path), str(cu_path)]
    proc = subprocess.run(compile_cmd, cwd=workdir, env=_nvcc_env(), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("nvcc failed to compile:\n" + proc.stdout + proc.stderr)

    run_env = os.environ.copy()
    run_env["NEUROCUDA_SEED"] = str(seed)
    run_env["NEUROCUDA_WARMUP"] = str(warmup)
    run_env["NEUROCUDA_REPEAT"] = str(repeat)
    run_proc = subprocess.run(
        [str(exe_path)], cwd=workdir, env=run_env, capture_output=True, text=True
    )
    if run_proc.returncode != 0:
        raise RuntimeError("generated program failed:\n" + run_proc.stdout + run_proc.stderr)

    timings = {}
    for line in run_proc.stdout.splitlines():
        if line.startswith("TIMING "):
            key, val = line[len("TIMING "):].split("=")
            timings[key] = float(val)

    D = load_nct(str(workdir / "D.nct"))
    return D, timings, result.codegen.kernels_used


def bench_pytorch(A: np.ndarray, B: np.ndarray, *, warmup: int, repeat: int):
    import torch

    if not torch.cuda.is_available():
        return None, None

    device = torch.device("cuda")
    tA = torch.from_numpy(A).to(device)
    tB = torch.from_numpy(B).to(device)

    out = None
    for _ in range(warmup):
        out = torch.relu(torch.matmul(tA, tB))
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = torch.relu(torch.matmul(tA, tB))
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return out.detach().cpu().numpy(), times_ms


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=1024, help="square matrix size (default: 1024)")
    parser.add_argument("--repeat", type=int, default=20, help="measured repeats after warm-up")
    parser.add_argument("--warmup", type=int, default=5, help="untimed warm-up iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workdir", default=str(REPO_ROOT / "generated" / "_bench"))
    args = parser.parse_args(argv)

    if not shutil.which("nvcc"):
        print("error: nvcc not found on PATH; this benchmark needs the CUDA toolkit and a GPU.")
        return 1

    workdir = Path(args.workdir)
    n = args.size
    print(f"NeuroCUDA matmul+relu benchmark -- {n}x{n}, {args.warmup} warm-up + "
          f"{args.repeat} measured repeats, seed={args.seed}")

    print("\n[1/3] NeuroCUDA, optimizer ON (fused matmul_relu)...")
    D_fused, t_fused, k_fused = build_and_run_neurocuda(
        n, optimize_ir=True, warmup=args.warmup, repeat=args.repeat,
        seed=args.seed, workdir=workdir / "fused",
    )
    print(f"      kernels used: {k_fused}")

    print("[2/3] NeuroCUDA, optimizer OFF (separate matmul + relu)...")
    D_unfused, t_unfused, k_unfused = build_and_run_neurocuda(
        n, optimize_ir=False, warmup=args.warmup, repeat=args.repeat,
        seed=args.seed, workdir=workdir / "unfused",
    )
    print(f"      kernels used: {k_unfused}")

    A = load_nct(str(workdir / "fused" / "A.nct"))
    B = load_nct(str(workdir / "fused" / "B.nct"))
    ref = np.maximum(A @ B, 0.0)

    print("[3/3] PyTorch (torch.relu(torch.matmul(A, B))) on the GPU...")
    D_torch, t_torch = bench_pytorch(A, B, warmup=args.warmup, repeat=args.repeat)

    def max_diff(x):
        return float(np.max(np.abs(ref - x)))

    print(f"\nCorrectness vs. NumPy reference (atol={ATOL}, rtol={RTOL}):")
    print(f"  fused:    allclose={np.allclose(ref, D_fused, atol=ATOL, rtol=RTOL)!s:5}  max_abs_diff={max_diff(D_fused):.6f}")
    print(f"  unfused:  allclose={np.allclose(ref, D_unfused, atol=ATOL, rtol=RTOL)!s:5}  max_abs_diff={max_diff(D_unfused):.6f}")
    if D_torch is not None:
        print(f"  pytorch:  allclose={np.allclose(ref, D_torch, atol=ATOL, rtol=RTOL)!s:5}  max_abs_diff={max_diff(D_torch):.6f}")
    else:
        print("  pytorch:  torch.cuda.is_available() is False -- skipped")

    fused_ms = t_fused["D_mean_ms"]
    unfused_ms = t_unfused["C_mean_ms"] + t_unfused["D_mean_ms"]
    print(f"\nTiming (mean of {args.repeat} measured runs after {args.warmup} warm-up iterations, ms):")
    print(f"  NeuroCUDA fused   (matmul_relu):   {fused_ms:8.4f}  "
          f"(min {t_fused['D_min_ms']:.4f}, max {t_fused['D_max_ms']:.4f})")
    print(f"  NeuroCUDA unfused (matmul + relu): {unfused_ms:8.4f}  "
          f"(matmul {t_unfused['C_mean_ms']:.4f} + relu {t_unfused['D_mean_ms']:.4f})")
    if t_torch is not None:
        torch_mean = statistics.mean(t_torch)
        print(f"  PyTorch (cuBLAS matmul + relu):    {torch_mean:8.4f}  "
              f"(min {min(t_torch):.4f}, max {max(t_torch):.4f})")
    else:
        torch_mean = None

    print()
    if fused_ms < unfused_ms:
        saved = unfused_ms - fused_ms
        print(
            f"Fusion saved {saved:.4f} ms ({saved / unfused_ms * 100:.1f}%) vs. the unfused path "
            f"at this size: one fewer kernel launch, and the {n}x{n} intermediate no longer makes "
            f"a full round trip through global memory."
        )
    else:
        print(
            "Fusion did not win at this size on this run. Reporting the measured numbers as-is "
            "rather than claiming a speedup that didn't happen."
        )

    if torch_mean is not None:
        if torch_mean < fused_ms:
            print(
                f"PyTorch was faster than NeuroCUDA's fused kernel ({torch_mean:.4f} ms vs "
                f"{fused_ms:.4f} ms). Expected: PyTorch's matmul calls into cuBLAS, a vendor-tuned "
                f"library; NeuroCUDA's tiled kernel is a teaching implementation. NeuroCUDA does "
                f"not claim to outperform PyTorch in general."
            )
        else:
            print(
                f"NeuroCUDA's fused kernel was faster than PyTorch on this run ({fused_ms:.4f} ms "
                f"vs {torch_mean:.4f} ms) at size {n}x{n}. Unusual -- worth re-checking at other "
                f"sizes rather than assuming it generalizes."
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
