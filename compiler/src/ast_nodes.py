"""AST node definitions for NeuroLang.

Grammar (EBNF):

    program    := statement*
    statement  := decl_stmt | print_stmt
    decl_stmt  := ['tensor'] IDENT '=' call NEWLINE
    call       := IDENT '(' arg (',' arg)* ')'
    arg        := NUMBER | IDENT | STRING
    print_stmt := 'print' IDENT NEWLINE

There is deliberately no expression nesting: a call's arguments are
always a bare number, identifier, or string literal, never another
call. This mirrors the assignment's own example (`C = matmul(A, B)`
then `D = relu(C)`) and keeps every later stage -- semantic analysis,
IR, optimization, codegen -- operating on flat, already-named values.

String literals exist only as the argument to `load("path.nct")`, the
one creation op (alongside `random`/`zeros`) that reads real external
data into a tensor instead of generating it.
"""

from dataclasses import dataclass


@dataclass
class Node:
    line: int


@dataclass
class NumberLiteral(Node):
    value: int


@dataclass
class StringLiteral(Node):
    value: str


@dataclass
class Ident(Node):
    name: str


@dataclass
class Call(Node):
    func: str
    args: list  # list[NumberLiteral | Ident | StringLiteral]


@dataclass
class TensorDecl(Node):
    """`tensor NAME = random(...)` or `tensor NAME = zeros(...)` -- allocates a new tensor."""

    name: str
    expr: Call


@dataclass
class Assign(Node):
    """`NAME = add(...)` / `matmul(...)` / `relu(...)` -- a derived, computed tensor."""

    name: str
    expr: Call


@dataclass
class Print(Node):
    name: str


@dataclass
class Program(Node):
    statements: list  # list[TensorDecl | Assign | Print]
