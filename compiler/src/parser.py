"""Recursive-descent parser for NeuroLang.

Builds the AST defined in src/ast_nodes.py from the token stream
produced by src/lexer.py, raising src.errors.ParseError with a line
number on malformed input.
"""

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
from src.errors import ParseError
from src.lexer import Token, TokenType, tokenize


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    # -- token stream helpers -------------------------------------------------

    @property
    def current(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.type != TokenType.EOF:
            self.pos += 1
        return tok

    def expect(self, ttype: TokenType, what: str) -> Token:
        tok = self.current
        if tok.type != ttype:
            raise ParseError(
                f"expected {what}, found {_describe(tok)}", tok.line
            )
        return self.advance()

    def at(self, ttype: TokenType) -> bool:
        return self.current.type == ttype

    def skip_newlines(self) -> None:
        while self.at(TokenType.NEWLINE):
            self.advance()

    # -- grammar ----------------------------------------------------------------

    def parse_program(self) -> Program:
        start_line = self.current.line
        statements = []
        self.skip_newlines()
        while not self.at(TokenType.EOF):
            statements.append(self.parse_statement())
            if self.at(TokenType.EOF):
                break
            if self.at(TokenType.NEWLINE):
                self.skip_newlines()
            else:
                tok = self.current
                raise ParseError(
                    f"expected end of statement (newline), found {_describe(tok)}",
                    tok.line,
                )
        return Program(line=start_line, statements=statements)

    def parse_statement(self):
        tok = self.current
        if tok.type == TokenType.TENSOR:
            return self.parse_decl(explicit_tensor=True)
        if tok.type == TokenType.PRINT:
            return self.parse_print()
        if tok.type == TokenType.IDENT:
            return self.parse_decl(explicit_tensor=False)
        raise ParseError(
            f"expected a statement ('tensor ...', 'NAME = ...', or 'print NAME'), "
            f"found {_describe(tok)}",
            tok.line,
        )

    def parse_decl(self, *, explicit_tensor: bool):
        line = self.current.line
        if explicit_tensor:
            self.expect(TokenType.TENSOR, "'tensor'")
        name_tok = self.expect(TokenType.IDENT, "a tensor name")
        self.expect(TokenType.EQUALS, "'='")
        expr = self.parse_call()
        if explicit_tensor:
            return TensorDecl(line=line, name=name_tok.value, expr=expr)
        return Assign(line=line, name=name_tok.value, expr=expr)

    def parse_print(self):
        line = self.current.line
        self.expect(TokenType.PRINT, "'print'")
        name_tok = self.expect(TokenType.IDENT, "a tensor name")
        return Print(line=line, name=name_tok.value)

    def parse_call(self) -> Call:
        line = self.current.line
        func_tok = self.expect(TokenType.IDENT, "an operation name")
        self.expect(TokenType.LPAREN, "'('")
        args = []
        if not self.at(TokenType.RPAREN):
            args.append(self.parse_arg())
            while self.at(TokenType.COMMA):
                self.advance()
                args.append(self.parse_arg())
        self.expect(TokenType.RPAREN, "')'")
        return Call(line=line, func=func_tok.value, args=args)

    def parse_arg(self):
        tok = self.current
        if tok.type == TokenType.NUMBER:
            self.advance()
            return NumberLiteral(line=tok.line, value=int(tok.value))
        if tok.type == TokenType.IDENT:
            self.advance()
            return Ident(line=tok.line, name=tok.value)
        if tok.type == TokenType.STRING:
            self.advance()
            return StringLiteral(line=tok.line, value=tok.value)
        raise ParseError(
            f"expected a number, a tensor name, or a string literal as an argument, "
            f"found {_describe(tok)}",
            tok.line,
        )


def _describe(tok: Token) -> str:
    if tok.type == TokenType.EOF:
        return "end of file"
    if tok.type == TokenType.NEWLINE:
        return "end of line"
    return f"{tok.type.name.lower()} {tok.value!r}"


def parse(source: str) -> Program:
    """Tokenize and parse NeuroLang source text into a Program AST."""
    return Parser(tokenize(source)).parse_program()
