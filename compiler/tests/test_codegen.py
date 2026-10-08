"""Codegen tests.

TestCudaCodegenStructure checks the generated source textually and
runs everywhere (no GPU/nvcc needed). TestCudaCodegenGPU actually
invokes nvcc and runs the resulting binary on a real GPU; it is
skipped automatically when nvcc (and, on Windows, a usable MSVC host
compiler) is not available, per CLAUDE.md's "provide helpful errors
when CUDA is unavailable" -- here that just means "skip, don't fail".
"""

import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from src.cuda_codegen import generate_cuda
from src.ir import build_ir
from src.optimizer import optimize
from src.parser import parse
from src.semantic import analyze

REPO_ROOT = Path(__file__).resolve().parent.parent


def compile_pipeline(source: str, *, run_optimizer: bool = True):
    program = parse(source)
    sem = analyze(program)
    ir = build_ir(program, sem)
    if run_optimizer:
        ir = optimize(ir).program
    return generate_cuda(ir, source_name="<test>")


class TestCudaCodegenStructure(unittest.TestCase):
    def test_demo_program_emits_fused_kernel_only(self):
        source = (
            "tensor A = random(8, 8)\n"
            "tensor B = random(8, 8)\n"
            "C = matmul(A, B)\n"
            "D = relu(C)\n"
            "print D\n"
        )
        result = compile_pipeline(source)
        self.assertEqual(result.kernels_used, ["matmul_relu"])
        self.assertIn("__global__ void k_matmul_relu(", result.source)
        self.assertNotIn("__global__ void k_matmul(", result.source)
        self.assertNotIn("__global__ void k_relu(", result.source)
        self.assertIn('nc_write_tensor_file("A.nct"', result.source)
        self.assertIn('nc_write_tensor_file("D.nct"', result.source)
        self.assertIn("TIMING D_mean_ms=", result.source)
        self.assertIn("cudaFree(d_A)", result.source)
        self.assertIn("cudaFree(d_B)", result.source)
        self.assertIn("cudaFree(d_D)", result.source)

    def test_unoptimized_emits_both_separate_kernels(self):
        source = (
            "tensor A = random(8, 8)\n"
            "tensor B = random(8, 8)\n"
            "C = matmul(A, B)\n"
            "D = relu(C)\n"
            "print D\n"
        )
        result = compile_pipeline(source, run_optimizer=False)
        self.assertEqual(result.kernels_used, ["matmul", "relu"])

    def test_zeros_only_program_emits_no_kernels(self):
        result = compile_pipeline("tensor Z = zeros(3, 3)\nprint Z\n")
        self.assertEqual(result.kernels_used, [])
        self.assertNotIn("__global__", result.source)

    def test_only_used_kernels_are_emitted_for_add(self):
        source = "tensor A = random(2, 2)\ntensor B = random(2, 2)\nC = add(A, B)\nprint C\n"
        result = compile_pipeline(source)
        self.assertEqual(result.kernels_used, ["add"])
        self.assertIn("__global__ void k_add(", result.source)
        self.assertNotIn("__global__ void k_matmul", result.source)


@unittest.skipUnless(
    shutil.which("nvcc"), "nvcc not found on PATH; skipping real GPU compile+run test"
)
class TestCudaCodegenGPU(unittest.TestCase):
    """Actually compiles and runs generated CUDA on the GPU present in this
    environment, and checks the result against a NumPy reference."""

    def setUp(self):
        import numpy as np  # local import: only needed for this GPU test

        self.np = np
        self.workdir = REPO_ROOT / "generated" / "_test_gpu_codegen"
        self.workdir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _nvcc_env(self):
        import os

        from src.cli import find_msvc_bin_dir

        env = os.environ.copy()
        if sys.platform == "win32" and not shutil.which("cl"):
            cl_dir = find_msvc_bin_dir()
            if cl_dir:
                env["PATH"] = cl_dir + os.pathsep + env.get("PATH", "")
        return env

    def test_matmul_relu_matches_numpy(self):
        np = self.np
        source = (
            "tensor A = random(64, 48)\n"
            "tensor B = random(48, 32)\n"
            "C = matmul(A, B)\n"
            "D = relu(C)\n"
            "print D\n"
        )
        result = compile_pipeline(source)
        cu_path = self.workdir / "prog.cu"
        cu_path.write_text(result.source, encoding="utf-8")

        exe_path = self.workdir / "prog.exe"
        compile_proc = subprocess.run(
            [
                "nvcc",
                "-O3",
                "-I",
                str(REPO_ROOT / "runtime"),
                "-o",
                str(exe_path),
                str(cu_path),
            ],
            cwd=self.workdir,
            env=self._nvcc_env(),
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(compile_proc.returncode, 0, msg=compile_proc.stdout + compile_proc.stderr)

        run_proc = subprocess.run(
            [str(exe_path)], cwd=self.workdir, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(run_proc.returncode, 0, msg=run_proc.stdout + run_proc.stderr)
        self.assertIn("TIMING D_mean_ms=", run_proc.stdout)

        from benchmarks.tensor_io import load_nct

        A = load_nct(str(self.workdir / "A.nct"))
        B = load_nct(str(self.workdir / "B.nct"))
        D = load_nct(str(self.workdir / "D.nct"))
        ref = np.maximum(A @ B, 0.0)
        self.assertTrue(np.allclose(ref, D, atol=1e-2, rtol=1e-3))


if __name__ == "__main__":
    unittest.main()
