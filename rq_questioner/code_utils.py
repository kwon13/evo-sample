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
            # from-imports of non-deterministic RNG symbols (e.g.
            # `from sympy import randprime`, `from random import choice`).
            if isinstance(node, ast.ImportFrom):
                reasons.extend(_check_from_import_rng(node))
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            continue
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        else:
            reasons.append(f"top-level executable statement: {type(node).__name__}")

    reasons.extend(_check_nondeterministic_rng_calls(tree))
    reasons.extend(_check_infinite_while_loops(tree))

    return reasons


def _is_constant_truthy(expr: ast.expr) -> bool:
    """True iff ``expr`` is a compile-time constant that evaluates truthy
    (``True``, nonzero literal, non-empty string, etc.). Anything dynamic
    (variable, comparison, function call) returns False — we only flag
    statically-obvious infinite loops."""
    if isinstance(expr, ast.Constant):
        return bool(expr.value)
    return False


def _check_infinite_while_loops(tree: ast.AST) -> list[str]:
    """Flag ``while <truthy_constant>:`` loops whose body cannot exit.

    A loop is treated as exitable if its body contains at least one of
    ``break``, ``return``, or ``raise``. We use ``ast.walk`` so a
    terminator nested inside an ``if`` (the rejection-sampling pattern)
    counts. Loops with a dynamic condition (``while cond:``,
    ``while i < n:``) are trusted — we cannot prove termination
    statically, and they are the overwhelmingly common case."""
    reasons: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.While):
            continue
        if not _is_constant_truthy(node.test):
            continue
        has_exit = any(
            isinstance(sub, (ast.Break, ast.Return, ast.Raise))
            for sub in ast.walk(node)
        )
        if not has_exit:
            reasons.append(
                "infinite while loop: `while True:` with no "
                "`break` / `return` / `raise` in body"
            )
    return reasons


def _check_from_import_rng(node: ast.ImportFrom) -> list[str]:
    """Reject `from X import Y` forms that pull in seed-ignoring RNG entry
    points (e.g. ``from sympy import randprime``, ``from random import
    choice``). Only ``from random import Random`` is allowed under the
    ``random`` module."""
    reasons: list[str] = []
    mod = node.module or ""
    if mod == "random":
        for alias in node.names:
            if alias.name != "Random":
                reasons.append(
                    f"non-deterministic RNG import: from random import "
                    f"{alias.name} (use `rng = random.Random(seed)`)"
                )
    elif mod.startswith("sympy"):
        for alias in node.names:
            nm = alias.name
            if nm.startswith("rand") or nm.startswith("random"):
                reasons.append(
                    f"non-deterministic RNG import: from {mod} import "
                    f"{nm} (uses sympy global RNG, ignores seed)"
                )
    return reasons


def _check_nondeterministic_rng_calls(tree: ast.AST) -> list[str]:
    """AST-level check that flags real *calls* into non-deterministic RNG
    surfaces. We look only at ``Call`` nodes (so a docstring mentioning
    "sympy.randprime" is fine) and we walk the function attribute chain.

    Allowed:
        rng = random.Random(seed); rng.<anything>(...)
        sympy.<non-random>(...)
    Blocked:
        random.<anything except Random>(...)
        sympy.rand*  /  sympy.random*
        numpy.random.* / np.random.*
        secrets.*
        os.urandom / os.getrandom
    """
    reasons: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Pattern A: <Name>.<attr>(...)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            mod = func.value.id
            attr = func.attr
            if mod == "random" and attr != "Random":
                reasons.append(
                    f"non-deterministic RNG call: random.{attr}(...) "
                    "— construct `rng = random.Random(seed)` and call "
                    f"`rng.{attr}(...)` instead"
                )
            elif mod == "sympy" and (attr.startswith("rand")
                                     or attr.startswith("random")):
                reasons.append(
                    f"non-deterministic RNG call: sympy.{attr}(...) "
                    "uses sympy's module-global RNG and ignores `seed`"
                )
            elif mod == "secrets":
                reasons.append(
                    f"non-deterministic RNG call: secrets.{attr}(...) "
                    "— use `random.Random(seed)`"
                )
            elif mod == "os" and attr in ("urandom", "getrandom"):
                reasons.append(
                    f"non-deterministic RNG call: os.{attr}(...) "
                    "— use `random.Random(seed)`"
                )
        # Pattern B: <Name>.<mid>.<attr>(...) — e.g. np.random.choice(...)
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
        ):
            root = func.value.value.id
            mid = func.value.attr
            attr = func.attr
            if root in ("np", "numpy") and mid == "random":
                reasons.append(
                    f"non-deterministic RNG call: {root}.random.{attr}(...) "
                    "— use `random.Random(seed)`"
                )
    return reasons


_SYMBOLIC_HINT_TOKENS = (
    "sqrt", "pi", "exp", "log", "ln", "sin", "cos", "tan", "atan", "asin",
    "acos", "rational", "frac", "/", "i*", "*i", "**", "^",
)

# Maximum decimal digits past the point allowed for a bare-float answer.
# Past this threshold the value is almost certainly noise from a rounding-
# free numeric computation (e.g. 12.4096736459123987) rather than a clean
# answer like 0.5 or 13.0948. Calibrated to be lenient: math benchmark
# answers occasionally need 4-6 decimals; 8 is a comfortable upper bound
# and rejects only obviously over-precise outputs.
_MAX_BARE_FLOAT_DECIMAL_DIGITS = 8

# Phrases in the problem text that explicitly grant the solver license
# to round to a stated precision; when these appear, accept decimal
# answers even at higher digit counts up to a soft cap.
_ROUNDING_LICENSE_PHRASES = (
    "round to",
    "rounded to",
    "decimal places",
    "to the nearest",
)


def _decimal_digits_after_point(s: str) -> int:
    if "." not in s:
        return 0
    after = s.split(".", 1)[1]
    # Strip trailing zeros — "21.0" counts as 0 informative decimals.
    after = after.rstrip("0")
    # Trim non-digit tail (e.g. "1.5e-3" → "5")
    m = re.match(r"\d*", after)
    return len(m.group(0)) if m else 0


def _answer_is_disallowed_float(answer: str, problem: str | None = None) -> bool:
    """Reject only the float-answer shapes that cause real harm.

    Allowed:
      - bare integers and signed integers
      - sympy.Rational / Fraction strings
      - symbolic expressions (sqrt / pi / log / trig / fractions)
      - finite decimals up to ~8 significant fractional digits
        (e.g. 0.5, 13.0948, 1.25, 24.5 all pass)
      - longer decimals when the problem text licenses rounding
        ("round to 4 decimal places", "to the nearest hundredth", etc.)

    Rejected:
      - nan / inf / non-finite literals (caught elsewhere as "non-scalar")
      - decimals with > 8 informative fractional digits AND no rounding
        license in the problem text — these are the float-noise answers
        that confuse math_verify and signal numerical-computation habit.
      - scientific notation with absurd precision (1.5e-12 etc.)
    """
    s = (answer or "").strip()
    if not s:
        return False
    # Integer
    if re.fullmatch(r"[+-]?\d+", s):
        return False
    # Rational / Fraction
    if re.fullmatch(r"[+-]?\d+\s*/\s*[+-]?\d+", s):
        return False
    # Symbolic-expression heuristic: presence of a math function or radical
    # marker suggests an exact form even if a decimal slips in for a constant.
    s_lower = s.lower()
    if any(tok in s_lower for tok in _SYMBOLIC_HINT_TOKENS):
        return False
    if "." not in s:
        # No decimal point — accept.
        return False
    # Scientific notation: only reject when exponent makes the value
    # absurdly small (essentially unrepresentable precision).
    sci = re.fullmatch(r"[+-]?\d+(?:\.\d+)?[eE][+-]?\d+", s)
    if sci:
        try:
            v = float(s)
        except (ValueError, TypeError):
            return False
        # Reject sub-femto-scale magnitudes which only come from over-
        # precise numerical computations.
        if 0 < abs(v) < 1e-9:
            return True
        return False
    # Plain decimal-literal: check digit count.
    try:
        float(s)
    except (ValueError, TypeError):
        return False
    digits = _decimal_digits_after_point(s)
    if digits <= _MAX_BARE_FLOAT_DECIMAL_DIGITS:
        return False
    # > 8 digits — only allow if the problem text licenses rounding.
    if problem:
        p_lower = problem.lower()
        if any(phrase in p_lower for phrase in _ROUNDING_LICENSE_PHRASES):
            # Be lenient even for longer decimals when rounding is licensed,
            # but still cap at 16 digits to reject obvious noise.
            return digits > 16
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
    # Reject decimal answers with excessive precision (> 8 informative
    # fractional digits and no "round to ..." license in the problem text).
    # Finite, low-precision decimals like 0.5, 1.25, 13.0948 are accepted —
    # they are common, harmless, and grade correctly via math_verify.
    if _answer_is_disallowed_float(answer, problem=problem):
        reasons.append(
            "over-precise float answer (excessive decimal digits without "
            "rounding license in problem text)"
        )

    return reasons
