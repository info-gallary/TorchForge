"""Semantic analysis for NeuroLang.

Walks the AST once, in program order (no forward references), and:
  * maintains a symbol table of name -> Shape for every defined tensor
  * enforces the MVP type/shape rules: 2D float32 tensors, exact shape
    compatibility, no implicit broadcasting
  * enforces the allocation/computation split implied by the grammar:
    `tensor NAME = ...` must build a fresh tensor (random/zeros);
    plain `NAME = ...` must compute one from existing tensors
    (add/matmul/relu)

Raises src.errors.SemanticError with a line number on the first problem
found. On success, returns a SemanticResult with the ordered symbol
table and the shape computed for every Call node (keyed by id(node)),
which src/ir.py reuses instead of recomputing shapes.
"""

from dataclasses import dataclass

from src.ast_nodes import (
    Assign,
    Call,
    Ident,
    NumberLiteral,
    Print,
    Program,
    StringLiteral,
    TensorDecl,
)
from src.errors import SemanticError
from src.tensor_file import read_nct_shape

CREATION_OPS = {"random", "zeros", "load"}
COMPUTE_OPS = {"add", "matmul", "relu"}
ALL_OPS = CREATION_OPS | COMPUTE_OPS

DTYPE = "float32"


@dataclass(frozen=True)
class Shape:
    rows: int
    cols: int

    def __str__(self) -> str:
        return f"({self.rows}, {self.cols})"


@dataclass
class SemanticResult:
    symbols: dict  # name -> Shape, in definition order
    shapes_by_call: dict  # id(Call node) -> Shape


class _Analyzer:
    def __init__(self):
        self.symbols: dict[str, Shape] = {}
        self.shapes_by_call: dict[int, Shape] = {}

    def analyze(self, program: Program) -> SemanticResult:
        for stmt in program.statements:
            self.visit_statement(stmt)
        return SemanticResult(symbols=self.symbols, shapes_by_call=self.shapes_by_call)

    def visit_statement(self, stmt) -> None:
        if isinstance(stmt, TensorDecl):
            self._check_not_redefined(stmt.name, stmt.line)
            if stmt.expr.func not in CREATION_OPS:
                raise SemanticError(
                    f"'tensor {stmt.name} = {stmt.expr.func}(...)' is not allowed: "
                    f"a 'tensor' declaration must allocate with random(rows, cols), "
                    f"zeros(rows, cols), or load(\"path.nct\"). Use plain "
                    f"'{stmt.name} = {stmt.expr.func}(...)' for computed values.",
                    stmt.line,
                )
            shape = self.visit_call(stmt.expr)
            self.symbols[stmt.name] = shape
            return

        if isinstance(stmt, Assign):
            self._check_not_redefined(stmt.name, stmt.line)
            if stmt.expr.func not in COMPUTE_OPS:
                raise SemanticError(
                    f"'{stmt.name} = {stmt.expr.func}(...)' is not allowed: "
                    f"a plain assignment must compute a value with add(...), "
                    f"matmul(...) or relu(...). Use 'tensor {stmt.name} = "
                    f"{stmt.expr.func}(...)' to allocate a new tensor instead.",
                    stmt.line,
                )
            shape = self.visit_call(stmt.expr)
            self.symbols[stmt.name] = shape
            return

        if isinstance(stmt, Print):
            if stmt.name not in self.symbols:
                raise SemanticError(f"undefined tensor '{stmt.name}'", stmt.line)
            return

        raise SemanticError(f"unsupported statement {stmt!r}", stmt.line)

    def _check_not_redefined(self, name: str, line: int) -> None:
        if name in self.symbols:
            raise SemanticError(
                f"tensor '{name}' is already defined; NeuroLang v1 does not "
                f"support reassignment",
                line,
            )

    def visit_call(self, call: Call) -> Shape:
        if call.func not in ALL_OPS:
            raise SemanticError(
                f"unknown operation '{call.func}'; supported operations are: "
                + ", ".join(sorted(ALL_OPS)),
                call.line,
            )

        if call.func in ("random", "zeros"):
            shape = self._check_creation(call)
        elif call.func == "load":
            shape = self._check_load(call)
        elif call.func == "add":
            shape = self._check_add(call)
        elif call.func == "matmul":
            shape = self._check_matmul(call)
        elif call.func == "relu":
            shape = self._check_relu(call)
        else:  # pragma: no cover - guarded by ALL_OPS membership above
            raise SemanticError(f"unknown operation '{call.func}'", call.line)

        self.shapes_by_call[id(call)] = shape
        return shape

    def _check_creation(self, call: Call) -> Shape:
        if len(call.args) != 2:
            raise SemanticError(
                f"{call.func}(rows, cols) expects exactly 2 arguments, "
                f"got {len(call.args)}",
                call.line,
            )
        rows_arg, cols_arg = call.args
        rows = self._require_positive_int_literal(rows_arg, call.func, "rows")
        cols = self._require_positive_int_literal(cols_arg, call.func, "cols")
        return Shape(rows, cols)

    def _check_load(self, call: Call) -> Shape:
        if len(call.args) != 1:
            raise SemanticError(
                f"load(path) expects exactly 1 argument, got {len(call.args)}", call.line
            )
        (path_arg,) = call.args
        if not isinstance(path_arg, StringLiteral):
            raise SemanticError(
                "load(...): the argument must be a string literal file path, e.g. "
                'load("weights.nct")',
                path_arg.line,
            )
        try:
            rows, cols = read_nct_shape(path_arg.value)
        except ValueError as e:
            raise SemanticError(f"load({path_arg.value!r}): {e}", call.line) from None
        return Shape(rows, cols)

    def _require_positive_int_literal(self, arg, func: str, which: str) -> int:
        if not isinstance(arg, NumberLiteral):
            raise SemanticError(
                f"{func}(...): the {which} argument must be an integer literal, "
                f"not a tensor name (no dynamic shapes in NeuroLang v1)",
                arg.line,
            )
        if arg.value <= 0:
            raise SemanticError(
                f"{func}(...): the {which} argument must be a positive integer, "
                f"got {arg.value}",
                arg.line,
            )
        return arg.value

    def _resolve_tensor_arg(self, arg, func: str) -> Shape:
        if not isinstance(arg, Ident):
            raise SemanticError(
                f"{func}(...): arguments must be tensor names, not literal numbers",
                arg.line,
            )
        if arg.name not in self.symbols:
            raise SemanticError(f"undefined tensor '{arg.name}'", arg.line)
        return self.symbols[arg.name]

    def _check_add(self, call: Call) -> Shape:
        if len(call.args) != 2:
            raise SemanticError(
                f"add(A, B) expects exactly 2 arguments, got {len(call.args)}", call.line
            )
        a_shape = self._resolve_tensor_arg(call.args[0], "add")
        b_shape = self._resolve_tensor_arg(call.args[1], "add")
        if a_shape != b_shape:
            raise SemanticError(
                f"add(A, B): shape mismatch, A is {a_shape} but B is {b_shape} "
                f"(NeuroLang v1 has no implicit broadcasting; shapes must match exactly)",
                call.line,
            )
        return a_shape

    def _check_matmul(self, call: Call) -> Shape:
        if len(call.args) != 2:
            raise SemanticError(
                f"matmul(A, B) expects exactly 2 arguments, got {len(call.args)}", call.line
            )
        a_shape = self._resolve_tensor_arg(call.args[0], "matmul")
        b_shape = self._resolve_tensor_arg(call.args[1], "matmul")
        if a_shape.cols != b_shape.rows:
            raise SemanticError(
                f"matmul(A, B): incompatible shapes, A is {a_shape} and B is {b_shape} "
                f"(A's columns [{a_shape.cols}] must equal B's rows [{b_shape.rows}])",
                call.line,
            )
        return Shape(a_shape.rows, b_shape.cols)

    def _check_relu(self, call: Call) -> Shape:
        if len(call.args) != 1:
            raise SemanticError(
                f"relu(A) expects exactly 1 argument, got {len(call.args)}", call.line
            )
        return self._resolve_tensor_arg(call.args[0], "relu")


def analyze(program: Program) -> SemanticResult:
    return _Analyzer().analyze(program)
