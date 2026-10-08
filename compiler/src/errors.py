"""Shared compiler diagnostics for every NeuroCUDA pipeline stage.

Each error carries the source line it came from so the CLI can print
"file:line: message" style diagnostics like a real compiler.
"""


class NeuroLangError(Exception):
    """Base class for all NeuroCUDA compiler diagnostics."""

    stage = "compiler"

    def __init__(self, message: str, line: int | None = None):
        self.message = message
        self.line = line
        super().__init__(str(self))

    def __str__(self) -> str:
        if self.line is not None:
            return f"[{self.stage}] line {self.line}: {self.message}"
        return f"[{self.stage}] {self.message}"


class LexError(NeuroLangError):
    stage = "lexer"


class ParseError(NeuroLangError):
    stage = "parser"


class SemanticError(NeuroLangError):
    stage = "semantic"


class CodegenError(NeuroLangError):
    stage = "codegen"
