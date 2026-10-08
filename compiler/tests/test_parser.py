import unittest

from src.ast_nodes import Assign, Call, Ident, NumberLiteral, Print, TensorDecl
from src.errors import ParseError
from src.parser import parse


class TestParser(unittest.TestCase):
    def test_demo_program(self):
        source = (
            "tensor A = random(1024, 1024)\n"
            "tensor B = random(1024, 1024)\n"
            "C = matmul(A, B)\n"
            "D = relu(C)\n"
            "print D\n"
        )
        program = parse(source)
        self.assertEqual(len(program.statements), 5)

        a = program.statements[0]
        self.assertIsInstance(a, TensorDecl)
        self.assertEqual(a.name, "A")
        self.assertEqual(a.expr.func, "random")
        self.assertEqual([arg.value for arg in a.expr.args], [1024, 1024])

        c = program.statements[2]
        self.assertIsInstance(c, Assign)
        self.assertEqual(c.name, "C")
        self.assertEqual(c.expr.func, "matmul")
        self.assertEqual([arg.name for arg in c.expr.args], ["A", "B"])

        p = program.statements[4]
        self.assertIsInstance(p, Print)
        self.assertEqual(p.name, "D")

    def test_no_tensor_keyword_is_still_a_decl(self):
        program = parse("A = zeros(2, 2)\n")
        self.assertIsInstance(program.statements[0], Assign)

    def test_missing_newline_between_statements_raises(self):
        with self.assertRaises(ParseError):
            parse("tensor A = zeros(2, 2) tensor B = zeros(2, 2)\n")

    def test_nested_call_is_rejected(self):
        # The grammar intentionally has no expression nesting.
        with self.assertRaises(ParseError):
            parse("D = relu(matmul(A, B))\n")

    def test_unclosed_paren_raises(self):
        with self.assertRaises(ParseError):
            parse("tensor A = zeros(2, 2\n")

    def test_trailing_comma_raises(self):
        with self.assertRaises(ParseError):
            parse("tensor A = zeros(2, 2,)\n")

    def test_empty_source_is_empty_program(self):
        program = parse("")
        self.assertEqual(program.statements, [])

    def test_blank_lines_are_skipped(self):
        program = parse("\n\ntensor A = zeros(1, 1)\n\n\nprint A\n")
        self.assertEqual(len(program.statements), 2)


if __name__ == "__main__":
    unittest.main()
