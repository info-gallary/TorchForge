import unittest

from src.nlp_frontend import Clarification, NLPFrontend, Translation
from src.parser import parse
from src.semantic import analyze


class TestNLPFrontend(unittest.TestCase):
    def setUp(self):
        self.frontend = NLPFrontend()

    def test_claude_md_example_is_ambiguous_without_shapes(self):
        # This is literally CLAUDE.md's own English example. Without any
        # prior sizing, A and B's shapes are unknown -- must ask, not guess.
        result = self.frontend.parse(
            "Multiply matrices A and B, apply ReLU, and return the result."
        )
        self.assertIsInstance(result, Clarification)
        self.assertIn("A", result.question)
        self.assertIn("B", result.question)

    def test_fully_specified_request_translates_and_compiles(self):
        text = (
            "Create a tensor A of size 4x4 with random values. "
            "Create a tensor B of size 4x4 with random values. "
            "Multiply A and B, apply ReLU, and return the result."
        )
        result = self.frontend.parse(text)
        self.assertIsInstance(result, Translation)
        self.assertIn("tensor A = random(4, 4)", result.source)
        self.assertIn("tensor B = random(4, 4)", result.source)
        self.assertIn("= matmul(A, B)", result.source)
        self.assertIn("= relu(", result.source)
        self.assertTrue(result.source.strip().splitlines()[-1].startswith("print "))

        # And it must actually be valid NeuroLang.
        analyze(parse(result.source))

    def test_zeros_creation(self):
        result = self.frontend.parse("Create a tensor Z of size 3x3 of zeros. Print Z.")
        self.assertIsInstance(result, Translation)
        self.assertIn("tensor Z = zeros(3, 3)", result.source)

    def test_add_two_known_tensors(self):
        text = (
            "Make a tensor A of size 2x2 with random values. "
            "Make a tensor B of size 2x2 with random values. "
            "Add A and B. Print the result."
        )
        # "Print the result" isn't a supported phrasing on its own (only
        # "return the result" is) -- this exercises the unsupported-clause path.
        result = self.frontend.parse(text)
        self.assertIsInstance(result, Clarification)

    def test_add_with_explicit_names(self):
        text = (
            "Make a tensor A of size 2x2 with random values. "
            "Make a tensor B of size 2x2 with random values. "
            "Add A and B, call it C. Print C."
        )
        result = self.frontend.parse(text)
        self.assertIsInstance(result, Translation)
        self.assertIn("C = add(A, B)", result.source)
        self.assertIn("print C", result.source)
        analyze(parse(result.source))

    def test_relu_without_target_uses_last_result(self):
        text = "Create a tensor A of size 2x2 with random values. Apply relu. Return the result."
        result = self.frontend.parse(text)
        self.assertIsInstance(result, Translation)
        self.assertIn("= relu(A)", result.source)

    def test_undefined_operand_is_flagged(self):
        result = self.frontend.parse("Multiply A and B. Return the result.")
        self.assertIsInstance(result, Clarification)

    def test_implicit_print_added_when_missing(self):
        result = self.frontend.parse("Create a tensor A of size 2x2 with random values.")
        self.assertIsInstance(result, Translation)
        self.assertIn("print A", result.source)
        self.assertTrue(result.notes)

    def test_unrecognized_clause_asks_for_clarification(self):
        result = self.frontend.parse("Please teleport the tensor to the cloud.")
        self.assertIsInstance(result, Clarification)

    def test_empty_request(self):
        result = self.frontend.parse("   ")
        self.assertIsInstance(result, Clarification)


if __name__ == "__main__":
    unittest.main()
