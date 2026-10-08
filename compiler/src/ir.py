"""Intermediate representation for NeuroLang.

The IR is a flat list of three-address-style instructions, independent
of CUDA: each instruction names its result, its operand tensors (by
name) and, for the two allocation ops, the literal dimensions. Because
the grammar forbids nested call expressions, there is a 1:1
correspondence between source statements and IR instructions, so no
synthetic temporaries are needed here -- building the IR is a direct
lowering pass over the already-validated AST.

The optimizer (src/optimizer.py) consumes and produces this same
IRProgram type; the codegen (src/cuda_codegen.py) only ever reads it,
never the AST.
"""

from dataclasses import dataclass, field

from src.ast_nodes import Assign, Print, Program, StringLiteral, TensorDecl
from src.semantic import SemanticResult, Shape

CREATION_OPS = {"random", "zeros", "load"}
DIM_CREATION_OPS = {"random", "zeros"}  # sized by literal (rows, cols), unlike 'load'


@dataclass
class IRInstr:
    op: str  # 'random'|'zeros'|'load'|'add'|'matmul'|'relu'|'add_relu'|'matmul_relu'|'print'
    result: str | None  # output tensor name; None for 'print'
    args: list  # list[str] of operand tensor names
    shape: Shape | None  # shape of the result; None for 'print'
    line: int
    dims: tuple | None = None  # (rows, cols) literal dims, only set for 'random'/'zeros'
    path: str | None = None  # file path, only set for 'load'

    def __str__(self) -> str:
        if self.op == "print":
            return f"print {self.args[0]}"
        if self.op in DIM_CREATION_OPS:
            rows, cols = self.dims
            return f"{self.result}:{self.shape} = {self.op}({rows}, {cols})"
        if self.op == "load":
            return f'{self.result}:{self.shape} = load("{self.path}")'
        args = ", ".join(self.args)
        return f"{self.result}:{self.shape} = {self.op}({args})"


@dataclass
class IRProgram:
    instructions: list = field(default_factory=list)

    def __str__(self) -> str:
        return "\n".join(str(instr) for instr in self.instructions)


def build_ir(program: Program, semantic_result: SemanticResult) -> IRProgram:
    """Lower a validated AST into IR. `program` must already have passed
    src.semantic.analyze() -- this function trusts shapes without re-checking.
    """
    instructions = []
    for stmt in program.statements:
        if isinstance(stmt, TensorDecl):
            call = stmt.expr
            shape = semantic_result.shapes_by_call[id(call)]
            if call.func == "load":
                path_arg = call.args[0]
                assert isinstance(path_arg, StringLiteral)
                instructions.append(
                    IRInstr(
                        op=call.func,
                        result=stmt.name,
                        args=[],
                        shape=shape,
                        line=stmt.line,
                        path=path_arg.value,
                    )
                )
            else:
                dims = (call.args[0].value, call.args[1].value)
                instructions.append(
                    IRInstr(
                        op=call.func,
                        result=stmt.name,
                        args=[],
                        shape=shape,
                        line=stmt.line,
                        dims=dims,
                    )
                )
        elif isinstance(stmt, Assign):
            call = stmt.expr
            shape = semantic_result.shapes_by_call[id(call)]
            arg_names = [arg.name for arg in call.args]
            instructions.append(
                IRInstr(
                    op=call.func,
                    result=stmt.name,
                    args=arg_names,
                    shape=shape,
                    line=stmt.line,
                )
            )
        elif isinstance(stmt, Print):
            instructions.append(
                IRInstr(op="print", result=None, args=[stmt.name], shape=None, line=stmt.line)
            )
        else:  # pragma: no cover - semantic analysis already rejects anything else
            raise TypeError(f"cannot lower statement {stmt!r} to IR")

    return IRProgram(instructions=instructions)
