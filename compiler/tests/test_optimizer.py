import unittest

from src.ir import build_ir
from src.optimizer import optimize
from src.parser import parse
from src.semantic import analyze


def compile_ir(source: str):
    program = parse(source)
    result = analyze(program)
    return build_ir(program, result)


class TestOptimizer(unittest.TestCase):
    def test_matmul_relu_fusion(self):
        source = (
            "tensor A = random(4, 4)\n"
            "tensor B = random(4, 4)\n"
            "C = matmul(A, B)\n"
            "D = relu(C)\n"
            "print D\n"
        )
        result = optimize(compile_ir(source))
        ops = [i.op for i in result.program.instructions]
        self.assertEqual(ops, ["random", "random", "matmul_relu", "print"])
        fused = result.program.instructions[2]
        self.assertEqual(fused.result, "D")
        self.assertEqual(fused.args, ["A", "B"])
        self.assertTrue(any("fused matmul" in note for note in result.notes))

    def test_add_relu_fusion(self):
        source = (
            "tensor A = random(4, 4)\n"
            "tensor B = random(4, 4)\n"
            "C = add(A, B)\n"
            "D = relu(C)\n"
            "print D\n"
        )
        result = optimize(compile_ir(source))
        ops = [i.op for i in result.program.instructions]
        self.assertEqual(ops, ["random", "random", "add_relu", "print"])

    def test_no_fusion_when_intermediate_is_printed(self):
        source = (
            "tensor A = random(4, 4)\n"
            "tensor B = random(4, 4)\n"
            "C = matmul(A, B)\n"
            "D = relu(C)\n"
            "print C\n"
            "print D\n"
        )
        result = optimize(compile_ir(source))
        ops = [i.op for i in result.program.instructions]
        self.assertEqual(ops, ["random", "random", "matmul", "relu", "print", "print"])

    def test_no_fusion_when_intermediate_has_multiple_uses(self):
        source = (
            "tensor A = random(4, 4)\n"
            "tensor B = random(4, 4)\n"
            "C = matmul(A, B)\n"
            "D = relu(C)\n"
            "E = add(C, D)\n"
            "print E\n"
        )
        result = optimize(compile_ir(source))
        ops = [i.op for i in result.program.instructions]
        self.assertEqual(ops, ["random", "random", "matmul", "relu", "add", "print"])

    def test_fold_add_identity_with_zeros(self):
        source = (
            "tensor A = random(3, 3)\n"
            "tensor Z = zeros(3, 3)\n"
            "C = add(A, Z)\n"
            "print C\n"
        )
        result = optimize(compile_ir(source))
        ops = [i.op for i in result.program.instructions]
        # Z and the add are both folded away; print now reads A directly.
        self.assertEqual(ops, ["random", "print"])
        self.assertEqual(result.program.instructions[-1].args, ["A"])

    def test_fold_relu_of_zeros(self):
        source = "tensor Z = zeros(2, 2)\nD = relu(Z)\nprint D\n"
        result = optimize(compile_ir(source))
        ops = [i.op for i in result.program.instructions]
        self.assertEqual(ops, ["zeros", "print"])
        self.assertEqual(result.program.instructions[-1].args, ["Z"])

    def test_fold_matmul_with_zeros_operand(self):
        source = (
            "tensor A = random(2, 3)\n"
            "tensor Z = zeros(3, 4)\n"
            "C = matmul(A, Z)\n"
            "print C\n"
        )
        result = optimize(compile_ir(source))
        ops = [i.op for i in result.program.instructions]
        # Neither A's nor Z's actual values matter any more, only the
        # output shape (already captured in `dims`), so both are dead.
        self.assertEqual(ops, ["zeros", "print"])
        folded = result.program.instructions[0]
        self.assertEqual(folded.dims, (2, 4))

    def test_dead_code_elimination_drops_unused_zeros(self):
        source = (
            "tensor A = random(2, 2)\n"
            "tensor Z = zeros(2, 2)\n"
            "C = add(A, Z)\n"
            "print C\n"
        )
        result = optimize(compile_ir(source))
        results = [i.result for i in result.program.instructions if i.result]
        self.assertNotIn("Z", results)

    def test_unrelated_program_is_unchanged(self):
        source = "tensor A = random(2, 2)\nprint A\n"
        result = optimize(compile_ir(source))
        ops = [i.op for i in result.program.instructions]
        self.assertEqual(ops, ["random", "print"])
        self.assertEqual(result.notes, [])


if __name__ == "__main__":
    unittest.main()
