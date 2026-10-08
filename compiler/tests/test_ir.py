import unittest

from src.ir import build_ir
from src.parser import parse
from src.semantic import Shape, analyze


def lower(source: str):
    program = parse(source)
    result = analyze(program)
    return build_ir(program, result)


class TestIR(unittest.TestCase):
    def test_demo_program_lowers_one_instr_per_statement(self):
        source = (
            "tensor A = random(1024, 1024)\n"
            "tensor B = random(1024, 1024)\n"
            "C = matmul(A, B)\n"
            "D = relu(C)\n"
            "print D\n"
        )
        ir = lower(source)
        ops = [i.op for i in ir.instructions]
        self.assertEqual(ops, ["random", "random", "matmul", "relu", "print"])

        a = ir.instructions[0]
        self.assertEqual(a.result, "A")
        self.assertEqual(a.dims, (1024, 1024))
        self.assertEqual(a.shape, Shape(1024, 1024))

        c = ir.instructions[2]
        self.assertEqual(c.result, "C")
        self.assertEqual(c.args, ["A", "B"])
        self.assertEqual(c.shape, Shape(1024, 1024))

        p = ir.instructions[4]
        self.assertIsNone(p.result)
        self.assertEqual(p.args, ["D"])

    def test_matmul_result_shape_is_rectangular(self):
        source = "tensor A = random(2, 3)\ntensor B = random(3, 4)\nC = matmul(A, B)\n"
        ir = lower(source)
        c = ir.instructions[-1]
        self.assertEqual(c.shape, Shape(2, 4))

    def test_ir_text_rendering(self):
        ir = lower("tensor A = zeros(2, 2)\nprint A\n")
        text = str(ir)
        self.assertIn("A:(2, 2) = zeros(2, 2)", text)
        self.assertIn("print A", text)


if __name__ == "__main__":
    unittest.main()
