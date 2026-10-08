import unittest
from unittest import mock

from src.agent import AgentOutcome, handle_request


class TestAgent(unittest.TestCase):
    def test_ambiguous_request_needs_clarification(self):
        outcome = handle_request(
            "Multiply matrices A and B, apply ReLU, and return the result.",
            run_after_validate=False,
        )
        self.assertEqual(outcome.status, "needs_clarification")
        self.assertIn("A", outcome.message)

    def test_valid_request_without_running(self):
        text = (
            "Create a tensor A of size 4x4 with random values. "
            "Create a tensor B of size 4x4 with random values. "
            "Multiply A and B, apply ReLU, and return the result."
        )
        outcome = handle_request(text, run_after_validate=False)
        self.assertEqual(outcome.status, "ok")
        self.assertIn("matmul(A, B)", outcome.source)
        self.assertIn("print", outcome.source)

    def test_render_for_clarification(self):
        outcome = AgentOutcome(status="needs_clarification", message="what size is A?")
        rendered = outcome.render()
        self.assertIn("need more information", rendered)
        self.assertIn("what size is A?", rendered)

    def test_render_for_ok_with_notes_and_output(self):
        outcome = AgentOutcome(
            status="ok",
            source="tensor A = zeros(1, 1)\nprint A\n",
            notes=["a note"],
            stdout="mean=0.0\n",
        )
        rendered = outcome.render()
        self.assertIn("Generated NeuroLang", rendered)
        self.assertIn("a note", rendered)
        self.assertIn("mean=0.0", rendered)

    @mock.patch("src.agent.shutil.which", return_value=None)
    def test_run_after_validate_without_nvcc_falls_back_gracefully(self, _mock_which):
        text = "Create a tensor A of size 2x2 with random values. Return the result."
        outcome = handle_request(text, run_after_validate=True)
        self.assertEqual(outcome.status, "ok")
        self.assertTrue(any("nvcc" in n for n in outcome.notes))

    def test_repair_fixes_a_typo_in_an_undefined_tensor_reference(self):
        # Simulate what the NLP frontend could produce if the user typo'd a
        # tensor name that's a near-miss of one actually defined -- the
        # agent's one repair heuristic should fix exactly this.
        from src.agent import _attempt_repair
        from src.errors import SemanticError

        source = "tensor Alpha = zeros(2, 2)\nprint Alph\n"
        error = SemanticError("undefined tensor 'Alph'", line=2)
        repaired = _attempt_repair(source, error)
        self.assertIsNotNone(repaired)
        self.assertIn("print Alpha", repaired)

    def test_repair_returns_none_when_no_close_match(self):
        from src.agent import _attempt_repair
        from src.errors import SemanticError

        source = "tensor Zeta = zeros(2, 2)\nprint Qwerty\n"
        error = SemanticError("undefined tensor 'Qwerty'", line=2)
        self.assertIsNone(_attempt_repair(source, error))


if __name__ == "__main__":
    unittest.main()
