"""Agent orchestration: understand -> generate -> validate -> repair (at
most once) -> compile/run -> summarize.

This is the loop CLAUDE.md describes for the optional NLP/agent layer.
Safety is structural, not a policy we have to remember to enforce: the
only NeuroLang source this module ever compiles is text produced by
src.nlp_frontend and then checked by the exact same
lexer/parser/semantic pipeline every hand-written .nl file goes
through (src.cli.run_pipeline). The only subprocess this module ever
runs is `nvcc` compiling that generated .cu file, followed by running
the resulting trusted binary -- never an arbitrary shell command, and
never unvalidated code.
"""

import difflib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

from src.errors import NeuroLangError
from src.nlp_frontend import Clarification, NLPFrontend, Translation

_ASSIGN_RE = re.compile(r"^\s*(?:tensor\s+)?([A-Za-z_]\w*)\s*=", re.MULTILINE)


@dataclass
class AgentOutcome:
    status: str  # 'needs_clarification' | 'invalid' | 'ok' | 'error'
    message: str = ""
    source: str | None = None
    notes: list = field(default_factory=list)
    stdout: str = ""

    def render(self) -> str:
        if self.status == "needs_clarification":
            return f"I need more information before I can translate this:\n  {self.message}"

        if self.status == "invalid":
            lines = [
                "I translated the request to NeuroLang, but the compiler rejected it:",
                f"  {self.message}",
                "",
                "Generated NeuroLang:",
            ]
            lines += [f"  {l}" for l in (self.source or "").splitlines()]
            return "\n".join(lines)

        if self.status == "error":
            return f"error: {self.message}"

        lines = ["Generated NeuroLang:"]
        lines += [f"  {l}" for l in (self.source or "").splitlines()]
        if self.notes:
            lines += ["", "Notes:"] + [f"  - {n}" for n in self.notes]
        if self.stdout:
            lines += ["", "Program output:", self.stdout.rstrip()]
        return "\n".join(lines)


def _try_validate(source: str, source_name: str) -> NeuroLangError | None:
    from src.cli import run_pipeline

    try:
        run_pipeline(source, source_name, optimize_ir=True, do_codegen=False)
        return None
    except NeuroLangError as e:
        return e


def _attempt_repair(source: str, error: NeuroLangError) -> str | None:
    """The one and only repair heuristic: if the error is an undefined-
    tensor name that's a near-miss typo of a name already defined
    elsewhere in the program (e.g. the NLP frontend and the user meant
    the same tensor but spelled it differently), rewrite that single
    identifier and let the caller retry validation exactly once. Returns
    None if no repair applies -- the caller does not loop or retry further.
    """
    m = re.search(r"undefined tensor '([^']+)'", error.message)
    if not m:
        return None
    bad_name = m.group(1)

    defined = set(_ASSIGN_RE.findall(source))
    candidates = difflib.get_close_matches(bad_name, defined, n=1, cutoff=0.6)
    if not candidates:
        return None

    fixed_name = candidates[0]
    repaired = re.sub(rf"\b{re.escape(bad_name)}\b", fixed_name, source)
    return repaired if repaired != source else None


def handle_request(text: str, *, run_after_validate: bool = True) -> AgentOutcome:
    parsed = NLPFrontend().parse(text)

    if isinstance(parsed, Clarification):
        return AgentOutcome(status="needs_clarification", message=parsed.question)

    assert isinstance(parsed, Translation)
    source = parsed.source
    notes = list(parsed.notes)
    source_name = "<ask>"

    error = _try_validate(source, source_name)
    if error is not None:
        repaired = _attempt_repair(source, error)
        if repaired is not None:
            retry_error = _try_validate(repaired, source_name)
            if retry_error is None:
                notes.append(f"auto-repaired one issue and revalidated: {error.message}")
                source = repaired
                error = None
            else:
                error = retry_error

    if error is not None:
        return AgentOutcome(status="invalid", message=str(error), source=source, notes=notes)

    if not run_after_validate:
        return AgentOutcome(status="ok", source=source, notes=notes)

    return _compile_and_run(source, source_name, notes)


def _compile_and_run(source: str, source_name: str, notes: list) -> AgentOutcome:
    from src.cli import RUNTIME_DIR, REPO_ROOT, find_msvc_bin_dir, run_pipeline

    try:
        result = run_pipeline(source, source_name, optimize_ir=True)
    except NeuroLangError as e:  # pragma: no cover - already validated by the caller
        return AgentOutcome(status="error", message=str(e), source=source, notes=notes)

    nvcc = shutil.which("nvcc")
    if not nvcc:
        notes.append(
            "nvcc was not found on PATH, so the translated program was validated but not "
            "run -- showing the generated NeuroLang only"
        )
        return AgentOutcome(status="ok", source=source, notes=notes)

    workdir = REPO_ROOT / "generated" / "_ask"
    workdir.mkdir(parents=True, exist_ok=True)
    cu_path = workdir / "ask.cu"
    exe_path = workdir / ("ask.exe" if sys.platform == "win32" else "ask")
    cu_path.write_text(result.codegen.source, encoding="utf-8")

    env = os.environ.copy()
    if sys.platform == "win32" and not shutil.which("cl"):
        cl_dir = find_msvc_bin_dir()
        if cl_dir:
            env["PATH"] = cl_dir + os.pathsep + env.get("PATH", "")

    compile_cmd = ["nvcc", "-O3", "-I", str(RUNTIME_DIR), "-o", str(exe_path), str(cu_path)]
    proc = subprocess.run(compile_cmd, cwd=workdir, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        return AgentOutcome(
            status="error",
            message="nvcc failed to compile the generated CUDA code:\n" + proc.stdout + proc.stderr,
            source=source,
            notes=notes,
        )

    run_proc = subprocess.run([str(exe_path)], cwd=workdir, capture_output=True, text=True)
    if run_proc.returncode != 0:
        return AgentOutcome(
            status="error",
            message="the generated program exited with a non-zero status:\n" + run_proc.stderr,
            source=source,
            notes=notes,
        )

    return AgentOutcome(status="ok", source=source, notes=notes, stdout=run_proc.stdout)
