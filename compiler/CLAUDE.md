# NeuroCUDA — Principles of Compiler Design Project

## Goal
Build a small, working compiler that translates a beginner-friendly tensor language (NeuroLang) into CUDA C++ kernels. Add optional natural-language input using a simple NLP + agentic AI layer.

**Project claim:** A simpler interface than PyTorch for a limited set of tensor operations, with potential speedups on selected workloads through kernel fusion. Do NOT claim to outperform or replace PyTorch generally. Benchmark fairly.

## Example
NeuroLang input (`demo.nl`):
```text
tensor A = random(1024, 1024)
tensor B = random(1024, 1024)
C = matmul(A, B)
D = relu(C)
print D
```

Optional English input:
```text
Multiply matrices A and B, apply ReLU, and return the result.
```

## Required Compiler Pipeline
1. **Lexer:** tokenize identifiers, numbers, keywords, operators, parentheses and commas.
2. **Parser:** build an Abstract Syntax Tree (AST) with clear syntax errors.
3. **Semantic analyzer:** maintain a symbol table; validate definitions, tensor shapes, supported data types and operations.
4. **Intermediate Representation (IR):** represent operations independently from CUDA code.
5. **Optimizer:** start with constant folding (where applicable) and fuse consecutive elementwise operations; optional matmul + ReLU fusion.
6. **Code generator:** produce valid `.cu` files using CUDA kernels and host-side launch code.
7. **Runtime:** compile with `nvcc`, execute on an NVIDIA GPU, and report output and timing.

**Important:** The deterministic lexer, parser, semantic checks, IR and code generator must work without any LLM. AI is only an optional frontend/helper.

## NLP + Agentic AI
- Use a lightweight rule-based intent parser first; optionally use a local or API LLM to translate English into NeuroLang.
- An agent may follow: understand request → generate NeuroLang → validate with compiler → repair one error → compile/run → summarize output.
- Never run untrusted arbitrary generated shell commands or unrestricted CUDA code. Only compile validated code generated from the supported DSL.
- If the request is ambiguous, ask for clarification rather than guessing tensor shapes.

## MVP Language Scope
- `tensor`, variable assignment, `print`.
- 2D `float32` tensors only; explicit dimensions.
- `random(rows, cols)`, `zeros(rows, cols)`, `add(A, B)`, `matmul(A, B)`, `relu(A)`.
- Validate exact shape compatibility; no implicit broadcasting.
- Implement basic CUDA kernels first, then one fused optimization.
- No autograd, neural-network training, distributed GPU execution or full PyTorch API in v1.

## Suggested Stack
- Python 3.11+ for lexer/parser/semantic analysis/IR and command-line interface.
- Standard Python libraries initially; optional `lark` for parsing if needed.
- CUDA C++ and `nvcc` for generated kernels.
- NumPy and PyTorch only for checking correctness and benchmarking (not for executing the compiler's generated operations).
- NVIDIA GPU with a working CUDA toolkit; provide helpful errors when CUDA is unavailable.

## Repository Structure
```text
neurocuda/
  CLAUDE.md
  README.md
  examples/demo.nl
  src/lexer.py
  src/parser.py
  src/ast_nodes.py
  src/semantic.py
  src/ir.py
  src/optimizer.py
  src/cuda_codegen.py
  src/cli.py
  src/nlp_frontend.py
  src/agent.py
  runtime/                 # CUDA runtime helpers
  tests/
  benchmarks/
  generated/               # ignored generated .cu and binaries
```

## CLI Targets
```bash
python -m src.cli compile examples/demo.nl --out generated/demo.cu
python -m src.cli run examples/demo.nl
python -m src.cli explain examples/demo.nl  # tokens, AST, IR, optimized IR
python -m src.cli ask "Multiply A and B, then apply ReLU"  # optional NLP frontend
```

## Development Order
1. Get lexer, parser, AST and symbol table working with unit tests.
2. Add shape/type checking and an inspectable IR.
3. Generate and execute CUDA for `add` and `relu`.
4. Add basic `matmul` and fused `matmul + relu` or an elementwise fusion.
5. Add optional NLP + agent validation/repair loop.
6. Compare against NumPy/PyTorch, document results, and prepare a short compiler pipeline demo.

## Quality Rules
- Build a real compiler; do not bypass stages by letting an LLM write the entire `.cu` output.
- Favor readable, modular, commented code over unnecessary frameworks.
- Show meaningful syntax, type and tensor-dimension errors.
- Compare values against a PyTorch reference using a stated numeric tolerance.
- Benchmark with warm-up, repeated runs, synchronization, and clear separation of compilation time and GPU execution time. Compare equivalent data transfers and workloads.
- Report when PyTorch is faster. Never fabricate benchmark numbers.
- Keep the assignment runnable, explainable and feasible on a student laptop.

## Definition of Done
A user can write a `.nl` program, inspect compiler stages, compile it to CUDA, execute it on GPU, and verify at least one optimization against a correct baseline. The NLP/agent layer, if enabled, produces the same validated DSL before compilation.

## Instructions for Claude Code
Start with a minimal working vertical slice (`tensor A = zeros(2, 2)` → AST → IR → generated CUDA → execution). Implement and test incrementally. After each stage, summarize completed files, how to run them, and the next small milestone. Do not silently expand the scope.
