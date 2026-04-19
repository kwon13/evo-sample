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

FORBIDDEN_PROBLEM_PATTERNS = (
    "round to",
    "rounded to",
    "decimal place",
    "if it exists",
    "if the inverse does not exist",
    "additionally",
    "also,",
    " and find ",
    " and compute ",
    " then find ",
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


def lint_problem_instance(inst: ProblemInstance) -> list[str]:
    """Reject common verifier-surviving but training-poor problem patterns."""
    reasons: list[str] = []
    problem = (inst.problem or "").strip()
    answer = (inst.answer or "").strip()
    problem_l = problem.lower()
    answer_l = answer.lower()

    if not problem or not answer:
        reasons.append("empty problem or answer")

    if answer_l in {"undefined", "none", "nan", "inf", "infinity"}:
        reasons.append("non-scalar answer")
    if "," in answer or ";" in answer or re.search(r"\s+and\s+", answer_l):
        reasons.append("multi-part answer")
    if re.search(r"\d+\.\d+", answer):
        reasons.append("decimal answer")

    for pattern in FORBIDDEN_PROBLEM_PATTERNS:
        if pattern in problem_l:
            reasons.append(f"forbidden problem pattern: {pattern.strip()}")

    return reasons
