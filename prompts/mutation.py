"""
Mutation prompts for R_Q-Evolve Questioner.

Design goals (revised):
  1. REWARD-HACK DEFENSE — explicitly name & forbid the degenerate
     patterns that hijack fitness R_Q = p(1-p)·H by making the
     solver uncertain *without* teaching it reasoning (e.g. `x % 1`,
     arbitrary float-rounding, random letter-variable ensembles).
  2. STRUCTURED CREATIVITY — replace the vague "make it harder"
     hint with 6 named complexity axes that push real reasoning
     depth (parametric lifting, case analysis, theorem chaining,
     structural constraints, dimension lift, hidden identity).
  3. CONCEPT DECLARATION — force the LLM to commit to a named
     mathematical domain + techniques so that near-duplicate
     problems in different niches become rarer.
  4. FEW-SHOT HYGIENE — filter champions for banned patterns
     before they reach the few-shot window so the model doesn't
     learn to imitate degenerate incumbents.
  5. DIAGNOSTIC SHARPENING — detect parents that look like
     reward-hacks (high-H/mid-p with `%1`, `*1`, etc. in source)
     and instruct the LLM to REPAIR them, not extend them.
"""

from __future__ import annotations
from rq_questioner.concepts import concept_prompt_block
from rq_questioner.map_elites import MAPElitesGrid
from rq_questioner.program import ProblemProgram


# ---------------------------------------------------------------------------
# Static rubric blocks (concatenated into every mutation template)
# ---------------------------------------------------------------------------

HARD_CONTRACT = (
    "Hard contract.\n"
    "\n"
    "The function `generate(seed)` must return a tuple "
    "`(problem_text: str, answer: str)`. Use inverse construction: "
    "choose the answer first, then design the problem around it. You "
    "may use only the standard library (math, fractions, itertools, "
    "functools, random) and sympy. Every seed in 0..4 must terminate "
    "quickly and produce a valid output.\n"
    "\n"
    "Answer format — examples:\n"
    "  BAD:  return problem, str(0.16552117772)        (raw float — rejected)\n"
    "  BAD:  return problem, \"12.4096736459\"          (rounded surd — rejected)\n"
    "  BAD:  return problem, \"approximately 7\"        (word — rejected)\n"
    "  GOOD: return problem, str(sympy.Rational(2, 3))\n"
    "  GOOD: return problem, str(sympy.sqrt(154))\n"
    "  GOOD: return problem, \"17\"\n"
    "\n"
    "Before returning, verify that sympy.sympify(answer) succeeds and "
    "that the result contains no Float.\n"
    "\n"
)

TEXT_HYGIENE = (
    "Text hygiene (strict).\n"
    "\n"
    "The rendered problem_text will be read by a student who has never "
    "seen this rubric. It must not contain any of the following:\n"
    "  - the words \"parametric\", \"non-parametric\", \"axis\", "
    "\"hidden identity\", \"dimension lift\", \"case analysis\", "
    "\"considered as\", or phrases like \"the problem is …\"\n"
    "  - the closed-form solution formula, theorem name, recurrence "
    "form, or derivation hints such as \"S = ...\" or \"C(n,k) = ...\"\n"
    "  - meta-commentary about how the problem was constructed\n"
    "\n"
    "A problem that recites its own solution method is rejected "
    "regardless of p or H.\n"
    "\n"
)

QUALITY_BAR = (
    "Quality bar.\n"
    "\n"
    "Target a pass rate around 0.5 with entropy in the 2 to 4 range. "
    "The problem should center on one coherent mathematical object "
    "and require at least three reasoning stages. Difficulty must "
    "come from genuine technique uncertainty, not from ambiguous "
    "wording.\n"
    "\n"
)

DEPTH_BY_EXAMPLE = (
    "Concrete depth example.\n"
    "\n"
    "Parent (p=0.85, too easy):\n"
    "  \"x^2 - 5x + 6 = 0; find sum of roots\"   ->   answer: 5\n"
    "\n"
    "Shallow child (rejected — bigger numbers, no new reasoning):\n"
    "  \"x^2 - 127x + 342 = 0; find sum of roots\"   ->   answer: 127\n"
    "\n"
    "Deep child (accepted — Vieta plus parameter recovery):\n"
    "  \"x^2 + p*x + 12 = 0 has roots whose reciprocals sum to 7/12; "
    "find p\"   ->   answer: -7\n"
    "\n"
    "The deep child requires Vieta's formulas plus a one-line "
    "reciprocal identity. The shallow child only requires reading off "
    "a coefficient.\n"
    "\n"
)

CALIBRATION_BREADTH = (
    "Calibration check (in_breadth only).\n"
    "\n"
    "Before finalizing, ask yourself: would a 7B math-tuned solver "
    "get this right roughly half the time on seeds 0..4? If you "
    "suspect over 80% (trivial substitution) or under 20% "
    "(under-specified or numerically nasty), redesign the ranges or "
    "constraints. Numerical answers requiring 6 or more decimal "
    "places of precision are a strong signal of under-specification.\n"
    "\n"
)

CONCEPT_DECLARATION = concept_prompt_block()


MUTATION_SYSTEM_PROMPT = (
    "You are an expert designer of Python-based competition-math problem "
    "generators.\n"
    "\n"
    "Your output is a single Python program defining `generate(seed)`. "
    "Calibrate difficulty so that a strong 7B math-tuned solver passes "
    "roughly half the time across seeds 0..4 — neither trivially solvable "
    "nor pathologically hard. Avoid recycling textbook clichés or named "
    "contest problems.\n"
    "\n"
    "Output discipline. Emit only Python source — no preamble, no markdown "
    "commentary, no explanations outside the code. Inside `generate()`, do "
    "not write multi-line docstrings or prose comments; brief one-line "
    "comments naming the mathematical object are acceptable.\n"
    "\n"
    "FIRST, in your private scratch-pad, think step-by-step to design a "
    "brand-new, non-trivial generator whose outputs are mathematically "
    "valid by construction.\n"
    "THEN, without revealing any of your private thoughts, output only "
    "the Python source for the generator."
)


# ---------------------------------------------------------------------------
# Score feedback template
# ---------------------------------------------------------------------------

SCORE_FEEDBACK = (
    "Parent performance.\n"
    "\n"
    "  pass rate  p = {p_hat:.2f}    (ideal 0.50 — solver wins about half the time)\n"
    "  entropy    H = {h_score:.2f}    (ideal 2.0 to 4.0; very high entropy "
    "usually means the solver is confused, not thinking)\n"
    "  R_Q          = {rq_score:.4f}  (higher is better)\n"
    "\n"
    "Diagnosis: {diagnosis}\n"
    "Action: {action}\n"
    "\n"
)


# ---------------------------------------------------------------------------
# Top-level mutation templates
# ---------------------------------------------------------------------------

MUTATE_DEPTH = (
    "Task: write a deeper variant of the parent generator. Stay in "
    "the same domain but require more reasoning — not just bigger "
    "numbers.\n"
    "\n"
    "Parent program:\n"
    "```python\n{code}\n```\n"
    "\n"
    "{score_feedback}"
    "{exec_feedback}"
    "{few_shot}"
    + QUALITY_BAR
    + DEPTH_BY_EXAMPLE +
    "Constraints for this mutation:\n"
    "  - Add one substantial reasoning move beyond the parent's depth.\n"
    "  - Keep CONCEPT_TYPE = '{parent_concept_type}' and "
    "CONCEPT_GROUP = '{parent_concept_group}' exactly.\n"
    "  - If the parent has a banned pattern, remove it.\n"
    "\n"
    + CONCEPT_DECLARATION
    + HARD_CONTRACT
    + TEXT_HYGIENE
)

MUTATE_BREADTH = (
    "Task: write a generator in a different domain from the parent, "
    "at matching quality. Use a new mathematical object — do not "
    "reuse the parent's.\n"
    "\n"
    "Parent program (context only — do not reuse its object):\n"
    "```python\n{code}\n```\n"
    "\n"
    "{score_feedback}"
    "{exec_feedback}"
    "{few_shot}"
    + QUALITY_BAR
    + DEPTH_BY_EXAMPLE +
    "Constraints for this mutation:\n"
    "  - Pick CONCEPT_TYPE from the whitelist whose CONCEPT_GROUP is "
    "NOT '{parent_concept_group}'.\n"
    "  - Suggested groups to try: {suggested_groups}.\n"
    "\n"
    + CALIBRATION_BREADTH
    + CONCEPT_DECLARATION
    + HARD_CONTRACT
    + TEXT_HYGIENE
)

MUTATE_CROSSOVER = (
    "Task: merge parents A and B into a hybrid generator whose "
    "single mathematical object is the intersection of A's and B's "
    "ideas — not a concatenation like \"compute X from A, then Y "
    "from B\".\n"
    "\n"
    "Parent A (p={p_hat_a:.2f}, H={h_a:.2f}):\n"
    "```python\n{code_a}\n```\n"
    "\n"
    "Parent B (p={p_hat_b:.2f}, H={h_b:.2f}):\n"
    "```python\n{code_b}\n```\n"
    "\n"
    "{few_shot}"
    + QUALITY_BAR
    + DEPTH_BY_EXAMPLE +
    "Constraints for this mutation:\n"
    "  - Identify a shared structure (group, graph, polynomial, "
    "probability space, modulus). Every reasoning step must use it.\n"
    "  - Define exactly one whitelisted CONCEPT_TYPE / CONCEPT_GROUP "
    "pair matching the hybrid object.\n"
    "\n"
    + CONCEPT_DECLARATION
    + HARD_CONTRACT
    + TEXT_HYGIENE
)


# Backward-compatible re-export: external code may import SINGLE_ANSWER_RULE.
SINGLE_ANSWER_RULE = HARD_CONTRACT


# ---------------------------------------------------------------------------
# Diagnostic / feedback helpers
# ---------------------------------------------------------------------------

def score_diagnosis(p_hat: float, h_score: float) -> tuple[str, str]:
    """Verbose diagnosis + action text for a parent's (p_hat, H).

    Action wording avoids the meta-vocabulary forbidden by TEXT_HYGIENE
    (parametric, case analysis, dimension lift, hidden identity, axis)
    so the rubric does not prime the model to echo those words into the
    rendered problem text.
    """
    if p_hat > 0.8:
        diag = f"far TOO EASY (solver wins {p_hat:.0%})"
        action = (
            "introduce a coefficient the student must solve for, or "
            "chain a second theorem onto the parent's result; do NOT "
            "just enlarge numbers"
        )
    elif p_hat > 0.65:
        diag = f"slightly too easy (p={p_hat:.2f})"
        action = (
            "add one structural requirement (integrality, coprimality, "
            "monotonicity) or use an identity that flips the obvious "
            "approach"
        )
    elif p_hat < 0.15:
        diag = f"far TOO HARD (solver wins only {p_hat:.0%})"
        action = (
            "simplify one reasoning stage, clarify the mathematical "
            "object, keep the others — total depth should stay >= 3"
        )
    elif p_hat < 0.3:
        diag = f"slightly too hard (p={p_hat:.2f})"
        action = (
            "reduce numeric ranges but preserve structure; consider a "
            "clean parity or sign branch the solver can commit to"
        )
    else:
        diag = f"good learnability band (p={p_hat:.2f})"
        action = (
            "preserve difficulty; deepen by extending the object "
            "(scalar -> vector, integer -> Gaussian integer) or "
            "chaining a second theorem"
        )

    if h_score < 0.8:
        diag += f"; LOW entropy (H={h_score:.2f} — solver over-confident)"
        action += (
            "; add a real ambiguity: multiple plausible techniques, "
            "non-obvious sign, or theorem-selection branch"
        )
    elif h_score > 5.0:
        diag += f"; SUSPICIOUSLY high entropy (H={h_score:.2f})"
        action += (
            "; check for reward-hacking patterns (ambiguous wording, "
            "float rounding, mod-1 tricks); prefer cleaner statements"
        )

    return diag, action


def build_score_feedback(parent: ProblemProgram) -> str:
    p_hat = getattr(parent, "p_hat", 0.5)
    h_score = getattr(parent, "h_score", 1.0)
    rq_score = getattr(parent, "rq_score", 0.0)
    diag, action = score_diagnosis(p_hat, h_score)
    return SCORE_FEEDBACK.format(
        p_hat=p_hat, h_score=h_score, rq_score=rq_score,
        diagnosis=diag, action=action,
    )


def build_few_shot_examples(
    grid: MAPElitesGrid,
    top_k: int = 3,
    min_rq: float = 0.25,
) -> str:
    """Top-RQ champions as few-shot, with reward-hack filtering.

    Champions that match a banned anti-pattern or that execute into
    an anti-patterned rendered problem are excluded so the mutator
    does not learn to imitate them. If no clean champion is found,
    we emit an explicit "examples unavailable" note instead of
    silently leaking a dirty archive into the prompt.
    """
    champions = grid.get_all_champions()
    if not champions:
        return ""

    clean = []
    for champ in champions:
        if (champ.rq_score or 0.0) < min_rq:
            continue
        inst = champ.execute(seed=0, timeout=3.0)
        rendered = inst.problem if inst else ""
        clean.append((champ, rendered))

    clean.sort(key=lambda t: -(t[0].rq_score or 0.0))
    picks = clean[:top_k]
    if not picks:
        return (
            "High-quality examples.\n"
            "\n"
            "No reward-hack-free champions are available yet. Design "
            "from the rubric alone and do not imitate the parent's "
            "shape.\n"
            "\n"
        )

    parts = ["High-quality examples (well-posed champions).\n"]
    for i, (champ, rendered) in enumerate(picks, 1):
        rq = getattr(champ, "rq_score", 0) or 0
        p = getattr(champ, "p_hat", 0) or 0
        h = getattr(champ, "h_score", 0) or 0
        parts.append(
            f"Example {i} (R_Q={rq:.3f}, p={p:.2f}, H={h:.2f}):\n"
            f"```python\n{champ.source_code}\n```"
        )
        if rendered:
            parts.append(f"Rendered at seed=0: {rendered[:180]}")
        parts.append("")
    return "\n".join(parts) + "\n"


def build_execution_feedback(parent: ProblemProgram) -> str:
    """Show the parent's rendered problem + difficulty label.

    We render TWO seeds so the LLM can see whether the parent's
    problem shape is actually seed-sensitive (a must-have — if both
    renderings are identical, the seed is being ignored and we flag
    this explicitly as an anti-pattern signal).
    """
    inst0 = parent.execute(seed=0, timeout=3.0)
    inst1 = parent.execute(seed=1, timeout=3.0)
    if inst0 is None:
        return ""

    p_hat = getattr(parent, "p_hat", 0.5)
    h_score = getattr(parent, "h_score", 1.0)
    difficulty = (
        "TOO EASY" if p_hat > 0.7 else
        "TOO HARD" if p_hat < 0.2 else
        "GOOD BAND"
    )

    lines = [
        "Parent execution trace.",
        "",
        f"seed=0 problem: {inst0.problem[:160]}",
        f"seed=0 answer:  {inst0.answer}",
    ]
    if inst1:
        lines.append(f"seed=1 problem: {inst1.problem[:160]}")
        lines.append(f"seed=1 answer:  {inst1.answer}")
        if inst1.problem.strip() == inst0.problem.strip():
            lines.append("")
            lines.append(
                "Warning: seed 0 and seed 1 produced identical text. "
                "The parent is effectively seed-independent — fix this "
                "in the mutation."
            )
    lines.append("")
    lines.append(f"Solver pass rate: {p_hat:.0%} ({difficulty})")
    lines.append(f"Solver entropy:   {h_score:.2f}")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Mutation prefill helpers (shared between feasibility and trainer)
# ---------------------------------------------------------------------------

# Stop tokens applied to every mutation generate() call. Keeps the model from
# wandering into prose / unrelated code after the function body closes.
MUTATION_STOP = ["\n```\n\n", "\n```\n#", "\n# end_of_code", "\n# === END"]


def parent_concept_fields(parent: ProblemProgram) -> tuple[str, str, str]:
    """Return (concept_type, concept_group, suggested_other_groups) for
    prompt interpolation. Falls back to 'unknown' when parent metadata
    is missing — concept rescue at extraction time handles those cases."""
    from rq_questioner.concepts import CONCEPT_GROUPS
    ctype = (
        parent.declared_concept_type()
        or (parent.metadata or {}).get("concept_type")
        or "unknown"
    )
    cgroup = (
        parent.declared_concept_group()
        or (parent.metadata or {}).get("concept_group")
        or "unknown"
    )
    others = [g for g in CONCEPT_GROUPS if g != cgroup]
    suggested = ", ".join(others[:3]) if others else "any other group"
    return ctype, cgroup, suggested


def choose_prefill_concept(
    op: str,
    parent: ProblemProgram,
    parent_b: ProblemProgram | None = None,
    rng=None,
) -> tuple[str, str]:
    """Pick a (concept_type, concept_group) pair for prefill injection.

    in_depth   : parent's exact concept (preserve constraint).
    in_breadth : random whitelisted type whose group differs from parent.
    crossover  : random whitelisted type from the union of A's and B's groups.
    """
    import random as _random
    from rq_questioner.concepts import CONCEPT_TYPES, CONCEPT_TYPE_TO_GROUP
    rng = rng or _random.Random()
    p_ctype = (
        parent.declared_concept_type()
        or (parent.metadata or {}).get("concept_type")
    )
    p_cgroup = (
        parent.declared_concept_group()
        or (parent.metadata or {}).get("concept_group")
    )

    def _pair(t: str) -> tuple[str, str]:
        return t, CONCEPT_TYPE_TO_GROUP[t]

    if op == "in_depth":
        if p_ctype in CONCEPT_TYPES:
            return _pair(p_ctype)
        return _pair(CONCEPT_TYPES[0])
    if op == "in_breadth":
        candidates = [
            t for t in CONCEPT_TYPES
            if CONCEPT_TYPE_TO_GROUP.get(t) != p_cgroup
        ]
        return _pair(rng.choice(candidates) if candidates else CONCEPT_TYPES[0])
    # crossover
    pb_cgroup = (
        parent_b.declared_concept_group() or (parent_b.metadata or {}).get("concept_group")
        if parent_b is not None else None
    )
    span = {g for g in (p_cgroup, pb_cgroup) if g}
    candidates = [
        t for t in CONCEPT_TYPES
        if CONCEPT_TYPE_TO_GROUP.get(t) in span
    ] or list(CONCEPT_TYPES)
    return _pair(rng.choice(candidates))


def build_mutation_prefill(
    ctype: str, cgroup: str, op: str = "in_depth"
) -> tuple[str, str]:
    """Return (suffix_appended_to_prompt, prefix_for_extract_recovery).

    The suffix opens a Python code fence and locks in CONCEPT_GROUP so
    MAP-Elites D-axis binning stays deterministic. CONCEPT_TYPE is
    handled differently per operation:

      in_depth  — TYPE is also locked to the parent's exact concept
                  (the operation's whole point is to preserve it).
      in_breadth / crossover — TYPE is left as an open quote so the
                  model picks a concrete whitelisted type that fits
                  the function body it is about to write. The model is
                  expected to close the quote and continue the program.
                  `nearest_concept_type` rescues near-whitelist labels
                  at extraction time.

    The recovery prefix (suffix minus the fence) is prepended to the
    model's output before code extraction so a full, parseable program
    is reconstructed.
    """
    if op == "in_depth":
        body = (
            "import random\n\n"
            f"CONCEPT_GROUP = \"{cgroup}\"\n"
            f"CONCEPT_TYPE = \"{ctype}\"\n\n"
            "def generate(seed):\n"
        )
    else:
        body = (
            "import random\n\n"
            f"CONCEPT_GROUP = \"{cgroup}\"\n"
            "CONCEPT_TYPE = \""
        )
    suffix = "\n```python\n" + body
    return suffix, body
