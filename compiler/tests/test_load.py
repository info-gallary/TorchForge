"""Tests for the `load("path.nct")` creation op: reads real external data
into a tensor (e.g. a trained model's weights), instead of generating it
with random/zeros. Shape is inferred from the file's own header, the
same way random/zeros are sized by literal arguments.
"""

import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from benchmarks.tensor_io import load_nct, save_nct
from src.cuda_codegen import generate_cuda
from src.errors import LexError, SemanticError
from src.ir import build_ir
from src.lexer import TokenType, tokenize
from src.optimizer import optimize
from src.parser import parse
from src.semantic import Shape, analyze

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestLoadLexerParser(unittest.TestCase):
    def test_string_literal_tokenizes(self):
        tokens = tokenize('tensor W = load("weights.nct")\n')
        string_tok = [t for t in tokens if t.type == TokenType.STRING][0]
        self.assertEqual(string_tok.value, "weights.nct")

    def test_unterminated_string_raises(self):
        with self.assertRaises(LexError):
            tokenize('tensor W = load("weights.nct\n')

    def test_parses_load_call(self):
        program = parse('tensor W = load("weights.nct")\n')
        decl = program.statements[0]
        self.assertEqual(decl.expr.func, "load")
        self.assertEqual(decl.expr.args[0].value, "weights.nct")

    def test_string_argument_parses_but_is_a_semantic_error_for_random(self):
        # A string literal where random() expects a number literal parses
        # fine grammatically (STRING is a valid `arg`); semantic analysis
        # is what actually rejects it -- see TestLoadSemantic below.
        program = parse('tensor A = random("x", 2)\n')
        with self.assertRaises(SemanticError):
            analyze(program)


class TestLoadSemantic(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = REPO_ROOT / "generated" / "_test_load_semantic"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.w_path = self.tmp_dir / "w.nct"
        save_nct(str(self.w_path), np.zeros((4, 7), dtype=np.float32))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_shape_inferred_from_file(self):
        source = f'tensor W = load("{self.w_path.as_posix()}")\nprint W\n'
        result = analyze(parse(source))
        self.assertEqual(result.symbols["W"], Shape(4, 7))

    def test_missing_file_raises(self):
        source = 'tensor W = load("generated/_test_load_semantic/does_not_exist.nct")\n'
        with self.assertRaises(SemanticError):
            analyze(parse(source))

    def test_non_string_argument_rejected(self):
        with self.assertRaises(SemanticError):
            analyze(parse("tensor W = load(5)\n"))

    def test_wrong_arg_count_rejected(self):
        with self.assertRaises(SemanticError):
            analyze(parse(f'tensor W = load("{self.w_path.as_posix()}", 1)\n'))

    def test_load_usable_in_matmul(self):
        source = (
            f'tensor W = load("{self.w_path.as_posix()}")\n'
            f'tensor X = random(3, 4)\n'
            f'Y = matmul(X, W)\n'
            f'print Y\n'
        )
        result = analyze(parse(source))
        self.assertEqual(result.symbols["Y"], Shape(3, 7))

    def test_load_not_allowed_as_plain_assign(self):
        source = f'W = load("{self.w_path.as_posix()}")\n'
        with self.assertRaises(SemanticError):
            analyze(parse(source))


class TestLoadIRAndCodegen(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = REPO_ROOT / "generated" / "_test_load_codegen"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.w_path = self.tmp_dir / "w.nct"
        save_nct(str(self.w_path), np.ones((2, 2), dtype=np.float32))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_ir_renders_path(self):
        source = f'tensor W = load("{self.w_path.as_posix()}")\nprint W\n'
        program = parse(source)
        sem = analyze(program)
        ir = build_ir(program, sem)
        text = str(ir)
        self.assertIn("load(", text)
        self.assertIn(self.w_path.as_posix(), text)

    def test_codegen_emits_read_tensor_file_call(self):
        source = f'tensor W = load("{self.w_path.as_posix()}")\nprint W\n'
        program = parse(source)
        sem = analyze(program)
        ir = build_ir(program, sem)
        result = generate_cuda(optimize(ir).program, source_name="<test>")
        self.assertIn("nc_read_tensor_file(", result.source)
        # No kernel needed for a pure load+print program.
        self.assertEqual(result.kernels_used, [])


def _nvcc_env():
    import os

    env = os.environ.copy()
    if sys.platform == "win32" and not shutil.which("cl"):
        from src.cli import find_msvc_bin_dir

        cl_dir = find_msvc_bin_dir()
        if cl_dir:
            env["PATH"] = cl_dir + os.pathsep + env.get("PATH", "")
    return env


@unittest.skipUnless(shutil.which("nvcc"), "nvcc not found on PATH; skipping GPU load() test")
class TestLoadGPU(unittest.TestCase):
    def setUp(self):
        self.workdir = REPO_ROOT / "generated" / "_test_load_gpu"
        self.workdir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_loaded_weights_matmul_matches_numpy(self):
        from src.cli import RUNTIME_DIR

        rng = np.random.default_rng(0)
        X = rng.random((5, 3), dtype=np.float64).astype(np.float32)
        W = rng.random((3, 4), dtype=np.float64).astype(np.float32)
        x_path = self.workdir / "X.nct"
        w_path = self.workdir / "W.nct"
        save_nct(str(x_path), X)
        save_nct(str(w_path), W)

        source = (
            f'tensor X = load("{x_path.as_posix()}")\n'
            f'tensor W = load("{w_path.as_posix()}")\n'
            f'Y = matmul(X, W)\n'
            f'print Y\n'
        )
        program = parse(source)
        sem = analyze(program)
        ir = build_ir(program, sem)
        result = generate_cuda(optimize(ir).program, source_name="<test>")

        cu_path = self.workdir / "prog.cu"
        cu_path.write_text(result.source, encoding="utf-8")
        exe_path = self.workdir / "prog.exe"
        proc = subprocess.run(
            ["nvcc", "-O3", "-I", str(RUNTIME_DIR), "-o", str(exe_path), str(cu_path)],
            cwd=self.workdir, env=_nvcc_env(), capture_output=True, text=True, timeout=180,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

        run_proc = subprocess.run(
            [str(exe_path)], cwd=self.workdir, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(run_proc.returncode, 0, run_proc.stdout + run_proc.stderr)

        Y = load_nct(str(self.workdir / "Y.nct"))
        expected = X @ W
        self.assertTrue(np.allclose(Y, expected, atol=1e-3, rtol=1e-3))


if __name__ == "__main__":
    unittest.main()
