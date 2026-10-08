import unittest

from src.errors import SemanticError
from src.parser import parse
from src.semantic import Shape, analyze


class TestSemantic(unittest.TestCase):
    def test_demo_program_shapes(self):
        source = (
            "tensor A = random(1024, 1024)\n"
            "tensor B = random(1024, 1024)\n"
            "C = matmul(A, B)\n"
            "D = relu(C)\n"
            "print D\n"
        )
        result = analyze(parse(source))
        self.assertEqual(result.symbols["A"], Shape(1024, 1024))
        self.assertEqual(result.symbols["C"], Shape(1024, 1024))
        self.assertEqual(result.symbols["D"], Shape(1024, 1024))

    def test_matmul_shape_rules(self):
        source = "tensor A = random(2, 3)\ntensor B = random(3, 4)\nC = matmul(A, B)\n"
        result = analyze(parse(source))
        self.assertEqual(result.symbols["C"], Shape(2, 4))

    def test_matmul_incompatible_shapes_raises(self):
        source = "tensor A = random(2, 3)\ntensor B = random(5, 4)\nC = matmul(A, B)\n"
        with self.assertRaises(SemanticError):
            analyze(parse(source))

    def test_add_shape_mismatch_raises(self):
        source = "tensor A = random(2, 3)\ntensor B = random(2, 4)\nC = add(A, B)\n"
        with self.assertRaises(SemanticError):
            analyze(parse(source))

    def test_undefined_tensor_raises(self):
        with self.assertRaises(SemanticError):
            analyze(parse("C = relu(A)\n"))

    def test_print_undefined_raises(self):
        with self.assertRaises(SemanticError):
            analyze(parse("tensor A = zeros(1, 1)\nprint B\n"))

    def test_redefinition_raises(self):
        source = "tensor A = zeros(1, 1)\ntensor A = zeros(2, 2)\n"
        with self.assertRaises(SemanticError):
            analyze(parse(source))

    def test_tensor_keyword_requires_creation_op(self):
        source = "tensor A = zeros(1, 1)\ntensor B = relu(A)\n"
        with self.assertRaises(SemanticError):
            analyze(parse(source))

    def test_plain_assign_requires_compute_op(self):
        source = "A = zeros(1, 1)\n"
        with self.assertRaises(SemanticError):
            analyze(parse(source))

    def test_non_positive_dimension_raises(self):
        with self.assertRaises(SemanticError):
            analyze(parse("tensor A = zeros(0, 4)\n"))

    def test_dimension_must_be_literal_not_tensor(self):
        source = "tensor N = zeros(1, 1)\ntensor A = zeros(N, 4)\n"
        with self.assertRaises(SemanticError):
            analyze(parse(source))

    def test_unknown_operation_raises(self):
        with self.assertRaises(SemanticError):
            analyze(parse("tensor A = sigmoid(2, 2)\n"))


if __name__ == "__main__":
    unittest.main()
