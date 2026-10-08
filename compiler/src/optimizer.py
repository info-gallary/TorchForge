"""Optimizer passes over NeuroCUDA IR.

Three passes, run in this order by `optimize()`:

1. Constant folding around `zeros` tensors. `zeros(...)` is the only
   compile-time-known tensor value in the MVP language, so "constant
   folding (where applicable)" here means recognizing and simplifying
   arithmetic around it:
     - add(X, zeros) / add(zeros, X)  -> X
     - relu(zeros)                    -> zeros (of the same shape)
     - matmul(zeros, X) / matmul(X, zeros) -> a new zeros of the
       matmul's output shape
   This is implemented as a name-substitution (alias) pass followed by
   dead-code elimination, so instructions that become unreachable
   (e.g. a `zeros` only consumed by a folded-away `add`) are dropped.

2. Elementwise fusion: `T = add(A, B)` immediately followed by
   `D = relu(T)`, where T is used nowhere else, becomes a single
   `D = add_relu(A, B)` instruction -- one kernel launch instead of two,
   and one fewer full round trip of T through global memory.

3. Matmul+ReLU fusion: `T = matmul(A, B)` followed by `D = relu(T)`,
   under the same single-use condition, becomes `D = matmul_relu(A, B)`
   -- the ReLU is applied in the matmul kernel's epilogue instead of a
   separate kernel reading and rewriting the whole output matrix.

Both fusions only apply when the intermediate result is used exactly
once (by the very next consumer) and is not itself printed -- otherwise
its standalone value is observable and must still be computed.

optimize() returns the optimized IRProgram together with a list of
human-readable notes describing every transformation applied, so the
CLI's `explain` command can show students exactly what changed.
"""

from dataclasses import dataclass, replace

from src.ir import IRProgram


@dataclass
class OptimizationResult:
    program: IRProgram
    notes: list  # list[str], human-readable description of each applied rewrite


def optimize(ir: IRProgram) -> OptimizationResult:
    notes: list[str] = []

    program = _fold_zero_constants(ir, notes)
    program = _fuse(program, "add", "add_relu", notes)
    program = _fuse(program, "matmul", "matmul_relu", notes)

    return OptimizationResult(program=program, notes=notes)


# -- pass 1: constant folding around zero tensors ------------------------------


def _fold_zero_constants(ir: IRProgram, notes: list) -> IRProgram:
    instructions = list(ir.instructions)
    zero_results = {instr.result for instr in instructions if instr.op == "zeros"}
    alias: dict[str, str] = {}

    def resolve(name: str) -> str:
        seen = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    new_instructions = []
    for instr in instructions:
        if instr.op == "add" and len(instr.args) == 2:
            a, b = (resolve(x) for x in instr.args)
            if a in zero_results and b in zero_results:
                zero_results.add(instr.result)
                new_instructions.append(
                    replace(instr, op="zeros", args=[], dims=(instr.shape.rows, instr.shape.cols))
                )
                notes.append(
                    f"folded add({instr.args[0]}, {instr.args[1]}) -> zeros{instr.shape} "
                    f"(both operands are zeros)"
                )
                continue
            if a in zero_results:
                alias[instr.result] = b
                notes.append(f"folded add({instr.args[0]}, {instr.args[1]}) -> {b} (left operand is zeros)")
                continue
            if b in zero_results:
                alias[instr.result] = a
                notes.append(f"folded add({instr.args[0]}, {instr.args[1]}) -> {a} (right operand is zeros)")
                continue
            instr = replace(instr, args=[a, b])

        elif instr.op == "relu" and len(instr.args) == 1:
            x = resolve(instr.args[0])
            if x in zero_results:
                alias[instr.result] = x
                notes.append(f"folded relu({instr.args[0]}) -> {x} (input is zeros, relu(0) = 0)")
                continue
            instr = replace(instr, args=[x])

        elif instr.op == "matmul" and len(instr.args) == 2:
            a, b = (resolve(x) for x in instr.args)
            if a in zero_results or b in zero_results:
                # The result is a fresh zeros of the matmul's output shape --
                # it is not an alias of either input, since its shape differs
                # in general from both.
                zero_results.add(instr.result)
                new_instructions.append(
                    replace(instr, op="zeros", args=[], dims=(instr.shape.rows, instr.shape.cols))
                )
                notes.append(
                    f"folded matmul({instr.args[0]}, {instr.args[1]}) -> zeros{instr.shape} "
                    f"(one operand is zeros)"
                )
                continue
            instr = replace(instr, args=[a, b])

        new_instructions.append(instr)

    # Rewrite remaining references (including `print`) through the alias map.
    resolved_instructions = []
    for instr in new_instructions:
        if instr.args:
            instr = replace(instr, args=[resolve(a) for a in instr.args])
        resolved_instructions.append(instr)

    return _eliminate_dead_code(IRProgram(instructions=resolved_instructions), notes)


def _eliminate_dead_code(ir: IRProgram, notes: list) -> IRProgram:
    used = set()
    for instr in ir.instructions:
        used.update(instr.args)

    kept = []
    for instr in ir.instructions:
        if instr.op == "print":
            kept.append(instr)
            continue
        if instr.result in used:
            kept.append(instr)
        else:
            notes.append(f"removed dead instruction '{instr}' (result unused)")

    if len(kept) == len(ir.instructions):
        return ir
    return IRProgram(instructions=kept)


# -- passes 2 & 3: elementwise / matmul+relu fusion -----------------------------


def _fuse(ir: IRProgram, producer_op: str, fused_op: str, notes: list) -> IRProgram:
    instructions = ir.instructions
    use_count: dict[str, int] = {}
    for instr in instructions:
        for arg in instr.args:
            use_count[arg] = use_count.get(arg, 0) + 1

    printed = {instr.args[0] for instr in instructions if instr.op == "print"}

    producer_by_result = {
        instr.result: instr for instr in instructions if instr.op == producer_op
    }

    # Pre-scan: decide which producer -> relu pairs fuse, before emitting
    # anything. (The producer always appears earlier in program order than
    # the relu that consumes it, so this can't be decided in one forward pass.)
    fuse_at_relu: dict[int, object] = {}  # id(relu instr) -> producer instr
    fused_away: set[int] = set()  # id() of producer instructions consumed by a fusion
    for instr in instructions:
        if (
            instr.op == "relu"
            and len(instr.args) == 1
            and instr.args[0] in producer_by_result
        ):
            intermediate = instr.args[0]
            producer = producer_by_result[intermediate]
            if use_count.get(intermediate, 0) == 1 and intermediate not in printed:
                fuse_at_relu[id(instr)] = producer
                fused_away.add(id(producer))

    new_instructions = []
    for instr in instructions:
        if id(instr) in fused_away:
            continue
        if id(instr) in fuse_at_relu:
            producer = fuse_at_relu[id(instr)]
            fused = replace(instr, op=fused_op, args=list(producer.args), dims=None)
            new_instructions.append(fused)
            notes.append(
                f"fused {producer_op}({', '.join(producer.args)}) + relu "
                f"-> {instr.result} = {fused_op}({', '.join(producer.args)})"
            )
            continue
        new_instructions.append(instr)

    return IRProgram(instructions=new_instructions)
