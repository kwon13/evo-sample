"""Utilities for handling LLM-generated problem-generator source code."""

from __future__ import annotations

import ast
import re

from .program import ProblemInstance


ALLOWED_IMPORT_ROOTS = {
    "collections",
    "fractions",
    "functools",
    "itertools",
    "math",
    "random",
    "sympy",
}

FORBIDDEN_SOURCE_PATTERNS = (
    "while true",
    "print(",
    "input(",
    "open(",
    "__import__",
    "eval(",
    "exec(",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "os.",
    "sys.",
)

def _has_generate_function(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.FunctionDef) and node.name == "generate"
        for node in ast.walk(tree)
    )


def _trim_to_parseable_generator(code: str) -> str | None:
    """Return the shortest useful parseable prefix that defines generate."""
    lines = code.splitlines()
    best: str | None = None
    for end in range(len(lines), 0, -1):
        snippet = "\n".join(lines[:end]).strip()
        if not snippet:
            continue
        try:
            tree = ast.parse(snippet)
        except SyntaxError:
            continue
        if _has_generate_function(tree):
            best = snippet
            break
    return best


def _candidate_blocks(text: str) -> list[tuple[int, str]]:
    text = text.replace("<|im_sep|>", "").replace("<|endoftext|>", "")
    blocks: list[tuple[int, str]] = []

    for match in re.finditer(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL):
        blocks.append((match.start(), match.group(1).strip()))

    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith(("import ", "from ", "def generate")):
            blocks.append((sum(len(x) + 1 for x in lines[:i]), "\n".join(lines[i:])))

    blocks.append((len(text), text.strip()))
    return blocks


_CONCEPT_TYPE_ASSIGN_RE = re.compile(
    r"""^(\s*CONCEPT_TYPE\s*(?::\s*[^=]+)?=\s*)(['"])([^'"]+)\2""",
    re.M,
)
_CONCEPT_GROUP_ASSIGN_RE = re.compile(
    r"""^(\s*CONCEPT_GROUP\s*(?::\s*[^=]+)?=\s*)(['"])([^'"]+)\2""",
    re.M,
)


def _rescue_concept_labels(source: str) -> str:
    """Repair near-miss CONCEPT_TYPE/GROUP literals via fuzzy match.

    Rewrites a single unknown CONCEPT_TYPE assignment to its nearest
    whitelisted neighbour (and re-aligns CONCEPT_GROUP if needed). Skip
    the rescue when multiple unknowns appear or no neighbour is close
    enough — better to fail loudly downstream than silently mislabel.
    """
    from .concepts import (
        CONCEPT_TYPES, CONCEPT_TYPE_TO_GROUP, nearest_concept_type,
    )

    type_match = _CONCEPT_TYPE_ASSIGN_RE.search(source)
    if type_match is None:
        return source
    declared_type = type_match.group(3).strip()
    if declared_type in CONCEPT_TYPES:
        return source
    rescued = nearest_concept_type(declared_type)
    if rescued is None:
        return source

    new_type_assign = (
        f"{type_match.group(1)}{type_match.group(2)}{rescued}{type_match.group(2)}"
    )
    source = (
        source[:type_match.start()] + new_type_assign + source[type_match.end():]
    )
    expected_group = CONCEPT_TYPE_TO_GROUP.get(rescued)
    if expected_group:
        group_match = _CONCEPT_GROUP_ASSIGN_RE.search(source)
        if group_match and group_match.group(3) != expected_group:
            new_group_assign = (
                f"{group_match.group(1)}{group_match.group(2)}{expected_group}{group_match.group(2)}"
            )
            source = (
                source[:group_match.start()]
                + new_group_assign
                + source[group_match.end():]
            )
    return source


def extract_generator_code(text: str) -> str | None:
    """Extract the best AST-valid Python source defining ``generate``.

    The mutator can echo examples, Markdown fences, or prose. This helper only
    accepts snippets that actually parse as Python and contain a real
    ``FunctionDef`` named ``generate``; commented-out examples do not count.
    """
    scored: list[tuple[int, int, str]] = []
    for pos, raw in _candidate_blocks(text):
        if "```" in raw:
            raw = raw.split("```", 1)[0]
        code = raw.strip()
        if not code or "def generate" not in code:
            continue
        parsed = _trim_to_parseable_generator(code)
        if parsed is None:
            continue

        context = text[max(0, pos - 240):pos].lower()
        score = 0
        if "for inspection" in context or "do not follow" in context:
            score -= 20
        if "pass" in parsed:
            score -= 10
        if re.search(r"^\s*(import|from)\s+", parsed, re.M):
            score += 5
        if re.search(r"^\s*def\s+generate\b", parsed, re.M):
            score += 3
        score += min(pos // 5000, 4)

        if "random." in parsed and not re.search(
            r"^\s*import\s+random\b", parsed, re.M
        ):
            parsed = "import random\n" + parsed
        parsed = _rescue_concept_labels(parsed)
        scored.append((score, pos, parsed))

    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def lint_generator_source(source_code: str) -> list[str]:
    """Cheap static checks before executing an LLM-generated program."""
    reasons: list[str] = []
    lowered = source_code.lower()
    for pattern in FORBIDDEN_SOURCE_PATTERNS:
        if pattern in lowered:
            reasons.append(f"forbidden source pattern: {pattern}")

    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]

    if not _has_generate_function(tree):
        reasons.append("missing real generate function")

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif node.module:
                modules = [node.module]
            for module in modules:
                root = module.split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    reasons.append(f"disallowed import: {module}")
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            continue
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        else:
            reasons.append(f"top-level executable statement: {type(node).__name__}")

    return reasons


_SYMBOLIC_HINT_TOKENS = (
    "sqrt", "pi", "exp", "log", "ln", "sin", "cos", "tan", "atan", "asin",
    "acos", "rational", "frac", "/", "i*", "*i", "**", "^",
)


def _answer_is_disallowed_float(answer: str) -> bool:
    """Detect raw decimal-literal answers (rejected by HARD_CONTRACT).

    Accepts:
      - integers ("17", "-42")
      - sympy Rational / Fraction ("2/3", "-7/8")
      - symbolic expressions containing sympy / radical / trig markers
        (these may legitimately render to text like "sqrt(154)" or
        "pi/3" and have NO decimal point, so they pass).
      - exact symbolic expressions that combine integers with a decimal
        ONLY when the decimal appears inside a function call / radical
        (e.g. "sqrt(2.0)" is unusual; we still reject pure float).

    Rejects:
      - strings containing a decimal point that parse as a finite float
        (e.g. "0.17", "12.4096", "-1.2374874348...").
      - scientific notation literals ("1.5e-3").
    """
    s = (answer or "").strip()
    if not s:
        return False
    # Bare integer (optionally signed)
    if re.fullmatch(r"[+-]?\d+", s):
        return False
    # Fraction / Rational form a/b
    if re.fullmatch(r"[+-]?\d+\s*/\s*[+-]?\d+", s):
        return False
    # Symbolic-expression heuristic: presence of a math function or radical
    # suggests an exact form even if a decimal slips in for a constant.
    s_lower = s.lower()
    has_symbolic = any(tok in s_lower for tok in _SYMBOLIC_HINT_TOKENS)
    # Has a literal decimal point?
    has_dot = "." in s
    if not has_dot:
        # No decimal point — accept (covers exact symbolic with no float).
        return False
    if has_symbolic:
        # Decimal inside an exact symbolic expression: accept conservatively.
        # The mutation prompt asks for exact forms, but rejecting all decimals
        # inside e.g. "Rational(1, 2)" via string-match would over-reject.
        return False
    # Decimal without symbolic marker — check if it parses as a finite float.
    try:
        float(s)
    except (ValueError, TypeError):
        return False
    return True


def lint_problem_instance(inst: ProblemInstance) -> list[str]:
    """Reject common verifier-surviving but training-poor problem patterns."""
    reasons: list[str] = []
    problem = (inst.problem or "").strip()
    answer = (inst.answer or "").strip()
    answer_l = answer.lower()

    if not problem or not answer:
        reasons.append("empty problem or answer")

    if answer_l in {"undefined", "none", "nan", "inf", "infinity"}:
        reasons.append("non-scalar answer")
    if "," in answer or ";" in answer or re.search(r"\s+and\s+", answer_l):
        reasons.append("multi-part answer")
    # Reject decimal-literal answers per HARD_CONTRACT. Float answers
    # cause math_verify mismatches (rounding noise) and teach the solver
    # a numerical-precision habit that hurts on benchmark problems where
    # expected answers are integer / fractional / symbolic.
    if _answer_is_disallowed_float(answer):
        reasons.append("raw float answer (decimal literal without symbolic form)")

    return reasons
