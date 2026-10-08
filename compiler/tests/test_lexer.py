import unittest

from src.errors import LexError
from src.lexer import TokenType, tokenize


class TestLexer(unittest.TestCase):
    def test_simple_program(self):
        tokens = tokenize("tensor A = random(2, 2)\nprint A\n")
        types = [t.type for t in tokens]
        self.assertEqual(
            types,
            [
                TokenType.TENSOR,
                TokenType.IDENT,
                TokenType.EQUALS,
                TokenType.IDENT,
                TokenType.LPAREN,
                TokenType.NUMBER,
                TokenType.COMMA,
                TokenType.NUMBER,
                TokenType.RPAREN,
                TokenType.NEWLINE,
                TokenType.PRINT,
                TokenType.IDENT,
                TokenType.NEWLINE,
                TokenType.EOF,
            ],
        )

    def test_comments_are_ignored(self):
        tokens = tokenize("# a comment\ntensor A = zeros(1, 1) # trailing comment\n")
        types = [t.type for t in tokens]
        self.assertEqual(
            types,
            [
                TokenType.NEWLINE,
                TokenType.TENSOR,
                TokenType.IDENT,
                TokenType.EQUALS,
                TokenType.IDENT,
                TokenType.LPAREN,
                TokenType.NUMBER,
                TokenType.COMMA,
                TokenType.NUMBER,
                TokenType.RPAREN,
                TokenType.NEWLINE,
                TokenType.EOF,
            ],
        )

    def test_line_and_col_tracking(self):
        tokens = tokenize("tensor A = zeros(1, 1)\nprint A\n")
        print_tok = [t for t in tokens if t.type == TokenType.PRINT][0]
        self.assertEqual(print_tok.line, 2)
        self.assertEqual(print_tok.col, 1)

    def test_unexpected_character_raises(self):
        with self.assertRaises(LexError):
            tokenize("tensor A = zeros(1, 1) @\n")

    def test_decimal_number_rejected(self):
        with self.assertRaises(LexError):
            tokenize("tensor A = zeros(1.5, 1)\n")

    def test_keywords_vs_identifiers(self):
        tokens = tokenize("tensor tensorX = random(1, 1)\n")
        self.assertEqual(tokens[0].type, TokenType.TENSOR)
        self.assertEqual(tokens[1].type, TokenType.IDENT)
        self.assertEqual(tokens[1].value, "tensorX")


if __name__ == "__main__":
    unittest.main()
