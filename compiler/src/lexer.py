"""Lexer for NeuroLang.

Tokenizes identifiers, keywords (`tensor`, `print`), integer literals,
double-quoted string literals (file paths for `load(...)`), `=`, `(`,
`)`, `,`, newlines (statement separators) and `#` line comments.
Function names such as `random`, `zeros`, `load`, `add`, `matmul`,
`relu` are NOT keywords here -- they are plain identifiers resolved
later by the semantic analyzer. This keeps the lexer small and keeps
the set of supported operations a semantic concept, not a lexical one.
"""

from dataclasses import dataclass
from enum import Enum, auto

from src.errors import LexError

KEYWORDS = {"tensor", "print"}


class TokenType(Enum):
    TENSOR = auto()
    PRINT = auto()
    IDENT = auto()
    NUMBER = auto()
    STRING = auto()
    EQUALS = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    NEWLINE = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, line={self.line}, col={self.col})"


_SINGLE_CHAR = {
    "=": TokenType.EQUALS,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    ",": TokenType.COMMA,
}


def tokenize(source: str) -> list[Token]:
    """Convert NeuroLang source text into a flat list of tokens, ending in EOF."""
    tokens: list[Token] = []
    line = 1
    col = 1
    i = 0
    n = len(source)

    def peek(offset: int = 0) -> str:
        j = i + offset
        return source[j] if j < n else ""

    while i < n:
        ch = source[i]

        if ch == "\n":
            tokens.append(Token(TokenType.NEWLINE, "\\n", line, col))
            i += 1
            line += 1
            col = 1
            continue

        if ch in " \t\r":
            i += 1
            col += 1
            continue

        if ch == "#":
            while i < n and source[i] != "\n":
                i += 1
                col += 1
            continue

        if ch == '"':
            start_col = col
            i += 1
            col += 1
            start = i
            while i < n and source[i] not in ('"', "\n"):
                i += 1
                col += 1
            if i >= n or source[i] != '"':
                raise LexError("unterminated string literal", line)
            text = source[start:i]
            i += 1
            col += 1
            tokens.append(Token(TokenType.STRING, text, line, start_col))
            continue

        if ch.isdigit():
            start = i
            start_col = col
            while i < n and source[i].isdigit():
                i += 1
                col += 1
            if i < n and source[i] == ".":
                raise LexError(
                    f"decimal numbers are not supported in NeuroLang v1 "
                    f"(got {source[start:i + 1]!r}); use integer dimensions only",
                    line,
                )
            tokens.append(Token(TokenType.NUMBER, source[start:i], line, start_col))
            continue

        if ch.isalpha() or ch == "_":
            start = i
            start_col = col
            while i < n and (source[i].isalnum() or source[i] == "_"):
                i += 1
                col += 1
            text = source[start:i]
            if text in KEYWORDS:
                ttype = TokenType.TENSOR if text == "tensor" else TokenType.PRINT
            else:
                ttype = TokenType.IDENT
            tokens.append(Token(ttype, text, line, start_col))
            continue

        if ch in _SINGLE_CHAR:
            tokens.append(Token(_SINGLE_CHAR[ch], ch, line, col))
            i += 1
            col += 1
            continue

        raise LexError(f"unexpected character {ch!r}", line)

    tokens.append(Token(TokenType.EOF, "", line, col))
    return tokens
