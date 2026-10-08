"""End-to-end integration test matching CLAUDE.md's Definition of Done:
"A user can write a .nl program, inspect compiler stages, compile it
to CUDA, execute it on GPU, and verify at least one optimization
against a correct baseline." This drives examples/demo.nl itself
through the exact same code paths `python -m src.cli ...` uses, rather
than a hand-built program like tests/test_codegen.py does, and checks
the optimized (fused) result against the unoptimized (baseline) one
and against NumPy.

Skipped automatically when nvcc is unavailable.
"""

import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from benchmarks.tensor_io import load_nct
from src.cli import RUNTIME_DIR, find_msvc_bin_dir, run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_PATH = REPO_ROOT / "examples" / "demo.nl"


def _nvcc_env():
    import os

    env = os.environ.copy()
    if sys.platform == "win32" and not shutil.which("cl"):
        cl_dir = find_msvc_bin_dir()
        if cl_dir:
            env["PATH"] = cl_dir + os.pathsep + env.get("PATH", "")
    return env


def _build_and_run(workdir: Path, *, optimize_ir: bool):
    source = DEMO_PATH.read_text(encoding="utf-8")
    result = run_pipeline(source, str(DEMO_PATH), optimize_ir=optimize_ir)

    workdir.mkdir(parents=True, exist_ok=True)
    cu_path = workdir / "demo.cu"
    exe_path = workdir / ("demo.exe" if sys.platform == "win32" else "demo")
    cu_path.write_text(result.codegen.source, encoding="utf-8")

    proc = subprocess.run(
        ["nvcc", "-O3", "-I", str(RUNTIME_DIR), "-o", str(exe_path), str(cu_path)],
        cwd=workdir,
        env=_nvcc_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    run_proc = subprocess.run(
        [str(exe_path)], cwd=workdir, capture_output=True, text=True, timeout=60
    )
    assert run_proc.returncode == 0, run_proc.stdout + run_proc.stderr
    return result, run_proc.stdout


@unittest.skipUnless(shutil.which("nvcc"), "nvcc not found on PATH; skipping GPU integration test")
class TestDemoEndToEnd(unittest.TestCase):
    def setUp(self):
        self.workdir = REPO_ROOT / "generated" / "_test_integration"

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_demo_compiles_runs_and_matches_baseline_and_numpy(self):
        fused_result, fused_stdout = _build_and_run(self.workdir / "fused", optimize_ir=True)
        self.assertEqual(fused_result.codegen.kernels_used, ["matmul_relu"])
        self.assertIn("TIMING D_mean_ms=", fused_stdout)

        baseline_result, _ = _build_and_run(self.workdir / "baseline", optimize_ir=False)
        self.assertEqual(baseline_result.codegen.kernels_used, ["matmul", "relu"])

        A = load_nct(str(self.workdir / "fused" / "A.nct"))
        B = load_nct(str(self.workdir / "fused" / "B.nct"))
        D_fused = load_nct(str(self.workdir / "fused" / "D.nct"))
        D_baseline = load_nct(str(self.workdir / "baseline" / "D.nct"))

        reference = np.maximum(A @ B, 0.0)
        self.assertTrue(np.allclose(reference, D_fused, atol=1e-2, rtol=1e-3))
        self.assertTrue(np.allclose(D_fused, D_baseline, atol=1e-2, rtol=1e-3))


if __name__ == "__main__":
    unittest.main()
