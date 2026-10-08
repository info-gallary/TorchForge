"""A lightweight, rule-based intent parser that translates simple English
requests into NeuroLang source text.

This is intentionally NOT an LLM call: CLAUDE.md asks for "a lightweight
rule-based intent parser first", and an LLM-based translator is left as
a documented, unimplemented extension point (see the module docstring
note at the bottom) rather than something wired up without being asked
for -- wiring up a real external LLM call would mean managing API keys
and network access that nobody has requested.

Recognized phrasings (case-insensitive):
  - "create/make a tensor A of size 4x4 [with random values | of zeros]"
  - "multiply A and B" / "matrix multiply A and B" / "matmul A, B"
  - "add A and B"
  - "apply relu to X" / "apply a relu activation to X" / "relu X"
    (if no target is named, applies to the most recently computed value)
  - "print X" / "show X" / "output X" / "return the result"
  - "call it NAME" / "name it NAME" / "store it in NAME" to name a result

Per CLAUDE.md: "If the request is ambiguous, ask for clarification
rather than guessing tensor shapes." Any tensor used as an operand that
was never created with an explicit size *in this same request* makes
the whole request ambiguous -- parse() returns a Clarification instead
of silently inventing a shape.
"""

import re
from dataclasses import dataclass, field

NAME = r"[A-Za-z_]\w*"
NUMBER = r"\d+"

_CREATE_RE = re.compile(
    rf"(?:create|make|define|allocate)\s+(?:a\s+)?tensor\s+(?P<name>{NAME})"
    rf"(?:\s+(?:of|with)\s+size|\s+sized|\s+of\s+shape|\s+with\s+shape)?"
    rf"\s*(?P<rows>{NUMBER})\s*(?:x|by)\s*(?P<cols>{NUMBER})"
    rf"(?P<rest>.*)",
    re.IGNORECASE,
)

_MATMUL_RE = re.compile(
    rf"(?:matrix\s+)?multiply(?:\s+matrices)?\s+(?P<a>{NAME})\s+(?:and|with|by)\s+(?P<b>{NAME})",
    re.IGNORECASE,
)
_MATMUL_RE2 = re.compile(
    rf"matmul\s+(?P<a>{NAME})\s*(?:,|and)\s*(?P<b>{NAME})", re.IGNORECASE
)
_ADD_RE = re.compile(
    rf"add\s+(?P<a>{NAME})\s+(?:and|to|with)\s+(?P<b>{NAME})", re.IGNORECASE
)
_RELU_RE = re.compile(
    rf"(?:apply\s+(?:an?\s+)?)?relu(?:\s+activation)?(?:\s+(?:to|on)\s+(?P<x>{NAME}))?",
    re.IGNORECASE,
)
_PRINT_RE = re.compile(rf"(?:print|show|output|display)\s+(?P<x>{NAME})", re.IGNORECASE)
_RETURN_RESULT_RE = re.compile(r"return\s+(?:the\s+)?result", re.IGNORECASE)
_RENAME_RE = re.compile(
    rf"(?:call|name)\s+it\s+(?P<name_a>{NAME})|(?:store|assign)\s+(?:it\s+)?(?:in|to)\s+(?P<name_b>{NAME})",
    re.IGNORECASE,
)

_CLAUSE_SPLIT_RE = re.compile(r"[,;.]|\bthen\b", re.IGNORECASE)


@dataclass
class Clarification:
    question: str


@dataclass
class Translation:
    source: str  # NeuroLang source text, ready to feed to src.parser.parse
    notes: list = field(default_factory=list)


class NLPFrontend:
    """Stateless; parse() does all the work on a per-call basis."""

    def parse(self, text: str):
        clauses = [c.strip(" \t") for c in _CLAUSE_SPLIT_RE.split(text)]
        clauses = [c for c in clauses if c]
        if not clauses:
            return Clarification("I didn't find any tensor operations in that request.")

        lines: list[str] = []
        notes: list[str] = []
        defined: set[str] = set()
        needs_shapes: set[str] = set()
        last_result: str | None = None
        temp_counter = 0

        def new_temp() -> str:
            nonlocal temp_counter
            temp_counter += 1
            return f"T{temp_counter}"

        def require_defined(name: str) -> None:
            if name not in defined:
                needs_shapes.add(name)

        for clause in clauses:
            m = _CREATE_RE.search(clause)
            if m:
                name = m.group("name")
                rows, cols = int(m.group("rows")), int(m.group("cols"))
                kind = "zeros" if "zero" in (m.group("rest") or "").lower() else "random"
                lines.append(f"tensor {name} = {kind}({rows}, {cols})")
                defined.add(name)
                last_result = name
                continue

            m = _MATMUL_RE.search(clause) or _MATMUL_RE2.search(clause)
            if m:
                a, b = m.group("a"), m.group("b")
                require_defined(a)
                require_defined(b)
                result = new_temp()
                lines.append(f"{result} = matmul({a}, {b})")
                defined.add(result)
                last_result = result
                continue

            m = _ADD_RE.search(clause)
            if m:
                a, b = m.group("a"), m.group("b")
                require_defined(a)
                require_defined(b)
                result = new_temp()
                lines.append(f"{result} = add({a}, {b})")
                defined.add(result)
                last_result = result
                continue

            if "relu" in clause.lower():
                m = _RELU_RE.search(clause)
                x = (m.group("x") if m else None) or last_result
                if x is None:
                    return Clarification(
                        "The request applies ReLU, but nothing has been computed yet "
                        "to apply it to."
                    )
                require_defined(x)
                result = new_temp()
                lines.append(f"{result} = relu({x})")
                defined.add(result)
                last_result = result
                continue

            m = _RENAME_RE.search(clause)
            if m:
                new_name = m.group("name_a") or m.group("name_b")
                if last_result is not None and lines and lines[-1].split(" ", 1)[0] == last_result:
                    lines[-1] = new_name + lines[-1][len(last_result):]
                    defined.discard(last_result)
                    defined.add(new_name)
                    last_result = new_name
                continue

            m = _PRINT_RE.search(clause)
            if m:
                target = m.group("x")
                require_defined(target)
                lines.append(f"print {target}")
                continue

            if _RETURN_RESULT_RE.search(clause):
                if last_result is None:
                    return Clarification("There is nothing computed yet to return.")
                lines.append(f"print {last_result}")
                continue

            return Clarification(
                f"I couldn't understand this part of the request: {clause!r}. Try phrasing "
                f"like 'create a tensor A of size 4x4 with random values', 'multiply A and B', "
                f"'apply relu to C', or 'print D'."
            )

        if needs_shapes:
            names = ", ".join(sorted(needs_shapes))
            return Clarification(
                f"Tensors {names} are used but never created with an explicit size in this "
                f"request, and NeuroLang has no implicit or dynamic shapes. What size "
                f"(rows x cols) should {names} be?"
            )

        if not any(line.startswith("print ") for line in lines):
            if last_result is None:
                return Clarification("I didn't find any tensor operations in that request.")
            lines.append(f"print {last_result}")
            notes.append(
                f"the request never said to print/show/return a result, so "
                f"'print {last_result}' (the last computed value) was added automatically"
            )

        return Translation(source="\n".join(lines) + "\n", notes=notes)


# Extension point for later: a local or API LLM could replace/augment
# NLPFrontend.parse() for phrasings the rules above don't cover, as long
# as its output is still run through the exact same validated NeuroLang
# pipeline before anything is compiled or executed (see src/agent.py).
# Not implemented here -- no LLM call is wired up in this project.
