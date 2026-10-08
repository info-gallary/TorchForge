# NeuroCUDA

A small, real compiler — built for a Principles of Compiler Design
course — that translates **NeuroLang**, a tiny beginner-friendly
tensor language, into CUDA C++ kernels, compiles them with `nvcc`, and
runs them on an NVIDIA GPU. An optional rule-based NLP layer can
translate simple English requests into NeuroLang.

**Project claim:** NeuroCUDA offers a simpler interface than PyTorch
for a *limited* set of tensor operations, with a real (if modest)
speedup on one specific pattern — fusing `matmul` + `relu` into a
single kernel. It does **not** claim to outperform or replace PyTorch
generally: PyTorch's `matmul` calls into cuBLAS, a vendor-tuned
library; NeuroCUDA's kernels are teaching-grade. See
[Benchmarks](#benchmarks) for real, measured numbers, including the
cases where PyTorch wins.

Every pipeline stage below the optional NLP layer — lexer, parser,
semantic analysis, IR, optimizer, code generator — is deterministic
Python with **no LLM involved**. The NLP/agent layer is a thin,
optional front end that produces NeuroLang source text, which is then
validated by the exact same compiler every hand-written `.nl` file
goes through before anything is ever compiled or run.

## Requirements

- Python 3.11+ (no third-party packages needed for the compiler itself)
- To actually run compiled programs (`run`/`ask`/benchmarks): the
  NVIDIA CUDA Toolkit (`nvcc`) and an NVIDIA GPU
- On Windows, `nvcc` also needs an MSVC host compiler (`cl.exe`). If
  Visual Studio Build Tools are installed but you're not running from
  an "x64 Native Tools Command Prompt", NeuroCUDA's CLI auto-detects
  `cl.exe` via `vswhere` and adds it to `PATH` for you.
- `numpy` and `torch` only for correctness checks and benchmarking —
  never for executing NeuroLang programs themselves

If `nvcc` isn't available, `compile` and `explain` still work fully
(no GPU needed); `run`/`ask` print a clear explanation instead of a
traceback.

## The NeuroLang language (v1 / MVP)

```text
tensor A = random(1024, 1024)
tensor B = random(1024, 1024)
C = matmul(A, B)
D = relu(C)
print D
```

Grammar:

```text
program    := statement*
statement  := decl_stmt | print_stmt
decl_stmt  := ['tensor'] IDENT '=' call
call       := IDENT '(' arg (',' arg)* ')'
arg        := NUMBER | IDENT | STRING
print_stmt := 'print' IDENT
```

Rules:

- 2D `float32` tensors only, with explicit integer dimensions. No
  implicit broadcasting — shapes must match exactly.
- `tensor NAME = ...` must allocate with `random(rows, cols)`,
  `zeros(rows, cols)`, or `load("path.nct")`. Plain `NAME = ...` must
  compute with `add(A, B)`, `matmul(A, B)`, or `relu(A)`. This split
  (mirroring the example above) gives clearer error messages and keeps
  "allocate" vs. "compute" explicit.
- No expression nesting — `D = relu(matmul(A, B))` is rejected; write
  it as two statements. This keeps every stage flat (one IR
  instruction per statement) with no synthetic temporaries needed.
- Names are defined once; NeuroLang v1 has no reassignment.
- `#` starts a line comment.

**`load("path.nct")`** — added for the TorchForge integration (see
`../webapp/backend/app/neurocuda_bridge.py`): reads a real tensor from
a `.nct` file (the same binary format `print` dumps and
`benchmarks/tensor_io.py` reads/writes) instead of generating one.
Shape is inferred from the file's own header at compile time, exactly
like `random`/`zeros` are sized by their literal arguments — so a
trained model's actual weight matrices can be loaded and run through
real, compiled CUDA kernels, not just random or zero data.

## Pipeline

```
.nl source
   │  src/lexer.py       tokenize (keywords, idents, numbers, punctuation)
   ▼
tokens
   │  src/parser.py      recursive descent -> AST (src/ast_nodes.py)
   ▼
AST
   │  src/semantic.py    symbol table, shape/type checking
   ▼
AST + symbol table
   │  src/ir.py           lower to flat 3-address IR (CUDA-independent)
   ▼
IR
   │  src/optimizer.py    constant folding (zeros), dead-code elim., fusion
   ▼
optimized IR
   │  src/cuda_codegen.py emit kernels + host launch code
   ▼
.cu file
   │  nvcc (invoked by src/cli.py)
   ▼
.exe  -->  run on GPU, report timing + output
```

Inspect every stage for a given program with:

```bash
python -m src.cli explain examples/demo.nl
```

which prints tokens, the AST, the symbol table, the IR before and
after optimization, the list of optimizations actually applied, and
which CUDA kernels will be generated.

### Semantic analysis

Single pass, program order, no forward references. Maintains a symbol
table (`name -> Shape(rows, cols)`), enforces the allocate/compute
split above, checks `add` shape equality, `matmul` inner-dimension
compatibility, and that every referenced tensor is already defined.
Errors carry a line number, e.g.:

```
error: [semantic] line 3: matmul(A, B): incompatible shapes, A is (2, 3) and B is (5, 4) (A's columns [3] must equal B's rows [5])
```

### Optimizer (`src/optimizer.py`)

Three passes:

1. **Constant folding around `zeros`.** `zeros(...)` is the only
   compile-time-known tensor value in the MVP language, so folding
   means simplifying arithmetic around it: `add(X, zeros) -> X`,
   `relu(zeros) -> zeros`, `matmul(zeros, X) -> zeros` (of the matmul's
   output shape). Followed by dead-code elimination, so anything that
   becomes unreachable (e.g. an input only consumed by a folded-away
   op) is dropped. A program that's *entirely* zero-folded compiles to
   **zero kernel launches** — verified in
   `tests/test_optimizer.py`.
2. **Elementwise fusion:** `T = add(A, B)` immediately followed by
   `D = relu(T)`, with `T` used nowhere else, becomes one
   `add_relu` kernel instead of two.
3. **Matmul+ReLU fusion:** `T = matmul(A, B)` followed by `D = relu(T)`
   (same single-use condition) becomes one `matmul_relu` kernel — ReLU
   is applied in the matmul kernel's epilogue, saving a kernel launch
   and a full read/write of the output matrix through global memory.

Both fusions only fire when the intermediate is used exactly once and
isn't itself printed — otherwise its standalone value is observable
and must still be computed.

### Code generator (`src/cuda_codegen.py`)

Only the kernels a given program actually needs are emitted (a
zeros-only program emits none at all). Elementwise kernels use a
16×16 thread block; `matmul`/`matmul_relu` use a classic 32×32
shared-memory-tiled kernel (`runtime/common.cuh` defines the tile
width). Every kernel launch is timed with CUDA events; two environment
variables the generated binary reads, `NEUROCUDA_WARMUP` and
`NEUROCUDA_REPEAT`, control untimed warm-up launches and measured,
synchronized repeats, after which mean/min/max kernel time is printed
as `TIMING <name>_mean_ms=...` (etc.) — this is what lets
`benchmarks/bench_matmul_relu.py` report a steady-state number rather
than a cold first-launch one. Every `random(...)` input and every
`print`ed tensor is also dumped to a small binary file
(`<name>.nct` — see `runtime/tensor_io.hpp`) so the exact same data can
be loaded into NumPy/PyTorch for a fair comparison
(`benchmarks/tensor_io.py`). `print` additionally writes a bounded
preview to stdout (shape, mean, top-left corner) since dumping a full
1024×1024 matrix to a terminal isn't useful.

## CLI

```bash
python -m src.cli compile examples/demo.nl --out generated/demo.cu   # no GPU needed
python -m src.cli run examples/demo.nl                                # compile + nvcc + execute
python -m src.cli explain examples/demo.nl                            # tokens, AST, IR, optimized IR
python -m src.cli ask "Multiply A and B, then apply ReLU"             # rule-based NLP + agent loop
```

`run` accepts `--no-optimize` (compare against the unfused baseline),
`--seed`, `--repeat`, `--warmup`, and `--workdir`.

Sample `run` output on this machine (RTX 3050, CUDA 12.6):

```
Compiling: nvcc -O3 -I .../runtime -o .../generated/demo.exe .../generated/demo.cu
Compilation time: 12025.5 ms (host-side, nvcc; separate from GPU execution time below)
TIMING D_mean_ms=4.118600
TIMING D_min_ms=4.059100
TIMING D_max_ms=4.202500
== D : shape (1024, 1024) ==
mean=255.857716
preview (top-left 4x4):
262.5386 252.2688 257.6550 258.7973
265.4305 249.5046 261.3244 256.1869
264.9323 254.8085 261.1365 262.4613
260.6562 243.4414 252.9718 248.7740
TIMING total_wall_ms=1919.242400
```

(`mean≈255.86` matches theory: for two independent `U(0,1)` 1024×1024
matrices, `E[matmul] = 1024 × E[X]·E[Y] = 1024 × 0.25 = 256`.)

## NLP + agent layer (`src/nlp_frontend.py`, `src/agent.py`)

A rule-based, regex-driven intent parser — no LLM call is wired up
(that's left as a documented, unimplemented extension point, since
CLAUDE.md asks for a rule-based parser *first* and wiring up a real
external LLM would mean managing API keys nobody has asked for).
Recognizes phrasings like "create a tensor A of size 4x4 with random
values", "multiply A and B", "apply relu to C", "print D" /
"return the result", and "call it NAME".

The agent loop is: **understand** (NLP frontend) → **generate**
(NeuroLang source) → **validate** (same lexer/parser/semantic pipeline
as any `.nl` file) → **repair** (at most one heuristic: fix a likely
typo'd tensor name via `difflib`, then revalidate once — no loops) →
**compile/run** (same `nvcc` path as `python -m src.cli run`) →
**summarize**.

Safety is structural: the only subprocess ever run is `nvcc` compiling
NeuroCUDA's own generated `.cu` file, never an arbitrary shell command
or unvalidated code. If a request is ambiguous — most importantly, if
it references a tensor with no declared size — NeuroCUDA asks rather
than guessing a shape:

```
$ python -m src.cli ask "Multiply matrices A and B, apply ReLU, and return the result."
I need more information before I can translate this:
  Tensors A, B are used but never created with an explicit size in this
  request, and NeuroLang has no implicit or dynamic shapes. What size
  (rows x cols) should A, B be?
```

A fully-specified request compiles and runs for real:

```
$ python -m src.cli ask "Create a tensor A of size 8x8 with random values. \
  Create a tensor B of size 8x8 with random values. \
  Multiply A and B, apply ReLU, and return the result."
Generated NeuroLang:
  tensor A = random(8, 8)
  tensor B = random(8, 8)
  T1 = matmul(A, B)
  T2 = relu(T1)
  print T2

Program output:
TIMING T2_mean_ms=...
== T2 : shape (8, 8) ==
...
```

## Benchmarks

```bash
python -m benchmarks.bench_matmul_relu --size 1024 --repeat 20 --warmup 5
```

Measured on this machine (RTX 3050, CUDA 12.6, driver 581.86), 1024×1024,
5 warm-up + 20 measured, synchronized repeats, seed 42:

| Path                              | mean (ms) | min (ms) | max (ms) |
|------------------------------------|----------:|---------:|---------:|
| NeuroCUDA, fused `matmul_relu`     |    4.1186 |   4.0591 |   4.2025 |
| NeuroCUDA, unfused `matmul`+`relu` |    4.5783 |        — |        — |
| PyTorch (cuBLAS matmul + relu)     |    1.0654 |   0.7852 |   2.6307 |

- Correctness: all three match a NumPy `np.maximum(A @ B, 0)` reference
  within `atol=1e-2, rtol=1e-3` (max abs diff ≈ 5–6e-4 for the CUDA
  kernels; different summation order than NumPy's BLAS, as expected
  for floating point).
- Fusion saved **10.0%** here (one fewer kernel launch, no full
  read/write of the 1024×1024 intermediate through global memory).
- **PyTorch was ~3.9× faster than NeuroCUDA's fused kernel.** Expected
  and reported honestly: PyTorch's `matmul` calls into cuBLAS, a
  vendor-tuned library; NeuroCUDA's tiled kernel is a teaching
  implementation. This project does not claim to outperform PyTorch in
  general — see the project claim at the top of this document. The
  benchmark script itself always prints whichever side actually won,
  it never assumes NeuroCUDA wins.

## Tests

```bash
python -m unittest discover -s tests -v
```

61 tests: unit tests for every deterministic stage (lexer, parser,
semantic analysis, IR, optimizer, NLP frontend, agent) that need no
GPU, plus GPU integration tests (`test_codegen.py`,
`test_integration.py`) that actually invoke `nvcc` and run on a real
GPU, checked against NumPy — these auto-skip if `nvcc` isn't on PATH.

## Repository layout

```text
CLAUDE.md
README.md
examples/demo.nl
src/
  lexer.py          tokenizer
  ast_nodes.py       AST node definitions
  parser.py          recursive-descent parser
  semantic.py        symbol table + shape/type checking
  ir.py              flat IR + AST-to-IR lowering
  optimizer.py       constant folding, DCE, fusion passes
  cuda_codegen.py     CUDA kernel + host code generation
  cli.py             compile / run / explain / ask
  nlp_frontend.py     rule-based English -> NeuroLang
  agent.py           understand/generate/validate/repair/run/summarize
  errors.py          shared diagnostic exception types
runtime/
  common.cuh          CUDA_CHECK macro, tile/block-size constants
  tensor_io.hpp        .nct binary tensor file writer (C++ side)
tests/                 unit + GPU integration tests
benchmarks/
  tensor_io.py         .nct binary tensor file reader (Python side)
  bench_matmul_relu.py  fused vs. unfused vs. PyTorch, with correctness checks
generated/             compiler output (.cu/.exe/.nct) -- gitignored
```

## Known limitations (by design, not oversight)

Per CLAUDE.md's MVP scope: no autograd, no neural-network training, no
distributed GPU execution, no attempt at the full PyTorch API. 2D
`float32` tensors only, with explicit dimensions and no implicit
broadcasting. These are intentional boundaries of a Principles of
Compiler Design project, not missing features.
