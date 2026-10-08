"""NeuroCUDA command-line interface.

    python -m src.cli compile examples/demo.nl --out generated/demo.cu
    python -m src.cli run examples/demo.nl
    python -m src.cli explain examples/demo.nl
    python -m src.cli ask "Multiply A and B, then apply ReLU"

`compile` and `explain` only need Python (no GPU required). `run`
additionally needs the NVIDIA CUDA Toolkit (`nvcc`) and a GPU; if
either is missing, it prints a clear explanation instead of a
traceback, per CLAUDE.md's "provide helpful errors when CUDA is
unavailable."
"""

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from src.errors import NeuroLangError
from src.ir import build_ir
from src.lexer import TokenType, tokenize
from src.optimizer import optimize
from src.parser import parse
from src.semantic import analyze
from src.cuda_codegen import generate_cuda

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = REPO_ROOT / "runtime"


class PipelineResult:
    def __init__(self, program, semantic_result, ir_before, ir_after, opt_notes, codegen):
        self.program = program
        self.semantic = semantic_result
        self.ir_before = ir_before
        self.ir_after = ir_after
        self.opt_notes = opt_notes
        self.codegen = codegen


def run_pipeline(source: str, source_name: str, *, optimize_ir: bool = True, do_codegen: bool = True):
    """Run every deterministic compiler stage and return the result of each
    one. Raises src.errors.NeuroLangError (LexError/ParseError/SemanticError)
    on invalid input -- callers are expected to catch it and print a clean
    message rather than a traceback.
    """
    program = parse(source)
    semantic_result = analyze(program)
    ir_before = build_ir(program, semantic_result)

    if optimize_ir:
        opt = optimize(ir_before)
        ir_after, notes = opt.program, opt.notes
    else:
        ir_after, notes = ir_before, []

    codegen_result = generate_cuda(ir_after, source_name=source_name) if do_codegen else None
    return PipelineResult(program, semantic_result, ir_before, ir_after, notes, codegen_result)


def read_source_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        sys.exit(1)
    return p.read_text(encoding="utf-8")


def default_cu_path(nl_path: str) -> Path:
    return REPO_ROOT / "generated" / (Path(nl_path).stem + ".cu")


def find_msvc_bin_dir() -> str | None:
    """Best-effort discovery of MSVC's cl.exe directory via vswhere, for
    machines (common on Windows student laptops) where the CUDA Toolkit and
    Visual Studio Build Tools are installed but `cl.exe` isn't on PATH
    because no 'x64 Native Tools Command Prompt' was used to launch this
    shell. nvcc needs cl.exe as its host compiler on Windows.
    """
    if sys.platform != "win32":
        return None
    vswhere = Path("C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe")
    if not vswhere.exists():
        return None
    try:
        proc = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError:
        return None
    install_path = proc.stdout.strip()
    if not install_path:
        return None
    msvc_root = Path(install_path) / "VC" / "Tools" / "MSVC"
    if not msvc_root.exists():
        return None
    for version_dir in sorted(msvc_root.glob("*"), reverse=True):
        cl_dir = version_dir / "bin" / "Hostx64" / "x64"
        if (cl_dir / "cl.exe").exists():
            return str(cl_dir)
    return None


# -- commands ----------------------------------------------------------------


def cmd_compile(args) -> int:
    source = read_source_file(args.file)
    try:
        result = run_pipeline(source, args.file, optimize_ir=not args.no_optimize)
    except NeuroLangError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else default_cu_path(args.file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.codegen.source, encoding="utf-8")

    print(f"Compiled {args.file} -> {out_path}")
    symbols = ", ".join(f"{name}{shape}" for name, shape in result.semantic.symbols.items())
    print(f"Symbols: {symbols}")
    if result.opt_notes:
        print("Optimizations applied:")
        for note in result.opt_notes:
            print(f"  - {note}")
    else:
        print("Optimizations applied: none")
    kernels = ", ".join(result.codegen.kernels_used) or "(none -- fully constant-folded)"
    print(f"CUDA kernels used: {kernels}")
    return 0


def cmd_run(args) -> int:
    source = read_source_file(args.file)
    try:
        result = run_pipeline(source, args.file, optimize_ir=not args.no_optimize)
    except NeuroLangError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    nvcc = shutil.which("nvcc")
    if not nvcc:
        print(
            textwrap.dedent(
                """\
                error: nvcc (the NVIDIA CUDA compiler) was not found on PATH.

                NeuroCUDA's `run` command needs the NVIDIA CUDA Toolkit
                installed and an NVIDIA GPU to actually execute a program.
                You can still inspect every other compiler stage without a
                GPU:
                    python -m src.cli compile {file}   # writes the .cu file
                    python -m src.cli explain {file}   # tokens / AST / IR
                """
            ).format(file=args.file),
            file=sys.stderr,
        )
        return 1

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.file).stem
    cu_path = workdir / f"{stem}.cu"
    exe_path = workdir / (f"{stem}.exe" if sys.platform == "win32" else stem)
    cu_path.write_text(result.codegen.source, encoding="utf-8")

    env = os.environ.copy()
    if sys.platform == "win32" and not shutil.which("cl"):
        cl_dir = find_msvc_bin_dir()
        if cl_dir:
            env["PATH"] = cl_dir + os.pathsep + env.get("PATH", "")
        else:
            print(
                "warning: no MSVC 'cl.exe' found on PATH, and none could be "
                "auto-detected. nvcc needs a host C++ compiler on Windows; "
                "install the 'Desktop development with C++' workload in "
                "Visual Studio, or run this command from an 'x64 Native "
                "Tools Command Prompt'.",
                file=sys.stderr,
            )

    compile_cmd = ["nvcc", "-O3", "-I", str(RUNTIME_DIR), "-o", str(exe_path), str(cu_path)]
    print(f"Compiling: {' '.join(compile_cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(compile_cmd, cwd=workdir, env=env, capture_output=True, text=True)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    if proc.returncode != 0:
        print("error: nvcc failed to compile the generated CUDA code:", file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return 1
    print(f"Compilation time: {compile_ms:.1f} ms (host-side, nvcc; separate from GPU execution time below)")

    run_env = os.environ.copy()
    if args.seed is not None:
        run_env["NEUROCUDA_SEED"] = str(args.seed)
    if args.repeat is not None:
        run_env["NEUROCUDA_REPEAT"] = str(args.repeat)
    if args.warmup is not None:
        run_env["NEUROCUDA_WARMUP"] = str(args.warmup)

    run_proc = subprocess.run(
        [str(exe_path)], cwd=workdir, env=run_env, capture_output=True, text=True
    )
    sys.stdout.write(run_proc.stdout)
    if run_proc.returncode != 0:
        print("error: the generated program exited with a non-zero status:", file=sys.stderr)
        print(run_proc.stderr, file=sys.stderr)
        return 1
    return 0


def cmd_explain(args) -> int:
    source = read_source_file(args.file)

    print("=== Tokens ===")
    try:
        tokens = tokenize(source)
    except NeuroLangError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for tok in tokens:
        if tok.type in (TokenType.NEWLINE, TokenType.EOF):
            continue
        print(f"  {tok!r}")

    try:
        result = run_pipeline(source, args.file, optimize_ir=not args.no_optimize)
    except NeuroLangError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print("\n=== AST ===")
    for stmt in result.program.statements:
        print(f"  {stmt}")

    print("\n=== Symbol table ===")
    for name, shape in result.semantic.symbols.items():
        print(f"  {name}: {shape}")

    print("\n=== IR (before optimization) ===")
    for instr in result.ir_before.instructions:
        print(f"  {instr}")

    print("\n=== Optimizations applied ===")
    if result.opt_notes:
        for note in result.opt_notes:
            print(f"  - {note}")
    else:
        print("  (none)")

    print("\n=== IR (after optimization) ===")
    for instr in result.ir_after.instructions:
        print(f"  {instr}")

    print("\n=== CUDA kernels used ===")
    kernels = ", ".join(result.codegen.kernels_used) or "(none -- fully constant-folded)"
    print(f"  {kernels}")
    return 0


def cmd_ask(args) -> int:
    from src.agent import handle_request

    outcome = handle_request(args.text, run_after_validate=not args.no_run)
    print(outcome.render())
    return 0 if outcome.status == "ok" else 1


# -- argument parsing ----------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="NeuroCUDA: compile NeuroLang tensor programs to CUDA C++.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_compile = sub.add_parser("compile", help="Compile a .nl file to a .cu file")
    p_compile.add_argument("file")
    p_compile.add_argument("--out", help="Output .cu path (default: generated/<name>.cu)")
    p_compile.add_argument("--no-optimize", action="store_true", help="Skip optimizer passes")
    p_compile.set_defaults(func=cmd_compile)

    p_run = sub.add_parser("run", help="Compile, build with nvcc, and execute on the GPU")
    p_run.add_argument("file")
    p_run.add_argument(
        "--workdir", default=str(REPO_ROOT / "generated"),
        help="Directory for the .cu/.exe and dumped .nct tensors (default: generated/)",
    )
    p_run.add_argument("--no-optimize", action="store_true", help="Skip optimizer passes")
    p_run.add_argument("--seed", type=int, default=None, help="Sets NEUROCUDA_SEED")
    p_run.add_argument(
        "--repeat", type=int, default=None,
        help="Sets NEUROCUDA_REPEAT: measured kernel-timing repeats (mean/min/max reported)",
    )
    p_run.add_argument(
        "--warmup", type=int, default=None,
        help="Sets NEUROCUDA_WARMUP: untimed kernel launches run before the measured repeats",
    )
    p_run.set_defaults(func=cmd_run)

    p_explain = sub.add_parser("explain", help="Show tokens, AST, IR and optimized IR")
    p_explain.add_argument("file")
    p_explain.add_argument("--no-optimize", action="store_true", help="Skip optimizer passes")
    p_explain.set_defaults(func=cmd_explain)

    p_ask = sub.add_parser(
        "ask", help="Translate an English request into NeuroLang (rule-based) and run it"
    )
    p_ask.add_argument("text")
    p_ask.add_argument(
        "--no-run", action="store_true", help="Only translate + validate, do not compile/run"
    )
    p_ask.set_defaults(func=cmd_ask)

    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
