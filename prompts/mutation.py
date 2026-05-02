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
    "# === HARD CONTRACT ===\n"
    "# - `generate(seed)` returns `(problem_text: str, answer: str)`.\n"
    "# - `answer` is ONE sympy-parseable scalar (int, '3/7', 'sqrt(2)').\n"
    "#   No lists/ranges, no words, no decimals needing rounding.\n"
    "# - Inverse construction: pick the answer first, then the problem.\n"
    "# - Stdlib only (math/fractions/itertools/functools/random; sympy ok).\n"
    "# - Every seed in 0..4 must terminate fast and be valid.\n"
)

QUALITY_BAR = (
    "# === QUALITY BAR ===\n"
    "# Target p~0.5, H in 2-4. One coherent mathematical object,\n"
    "# >=3 named-technique reasoning stages. Difficulty must come from\n"
    "# technique uncertainty, not ambiguous wording.\n"
)

COMPLEXITY_AXES = (
    "# === AXES (pick 1, optionally 1 more) ===\n"
    "# 1. Parametric lifting (solve aux equation for a constant)\n"
    "# 2. Case analysis (parity / congruence / sign branch)\n"
    "# 3. Theorem chaining (Vieta->Newton, CRT->modular inverse)\n"
    "# 4. Structural constraint (integrality, coprimality, monotone)\n"
    "# 5. Dimension lift (scalar->vector, Z->Z[i], 2D->3D)\n"
    "# 6. Hidden identity (Catalan/Stirling/binomial via bijection)\n"
)

CONCEPT_DECLARATION = concept_prompt_block()


# ---------------------------------------------------------------------------
# Score feedback template
# ---------------------------------------------------------------------------

SCORE_FEEDBACK = (
    "# === PARENT PERFORMANCE ===\n"
    "# pass_rate  p_hat = {p_hat:.2f}   (ideal 0.50 — solver wins ~half)\n"
    "# entropy    H     = {h_score:.2f}   (ideal 2.0 - 4.0; too high often\n"
    "#                                     means solver is confused, not thinking)\n"
    "# R_Q              = {rq_score:.4f} (higher is better)\n"
    "#\n"
    "# DIAGNOSIS: {diagnosis}\n"
    "# ACTION:    {action}\n"
)


# ---------------------------------------------------------------------------
# Top-level mutation templates
# ---------------------------------------------------------------------------

MUTATE_DEPTH = (
    "# TASK: Write a deeper variant of the parent generator.\n"
    "# Same domain, more reasoning. Not just bigger numbers.\n"
    "#\n"
    "# Parent program:\n"
    "```python\n{code}\n```\n"
    "#\n"
    "{score_feedback}"
    "{exec_feedback}"
    "{few_shot}"
    "#\n"
    + QUALITY_BAR
    + COMPLEXITY_AXES +
    "# - Add one substantial reasoning move; optionally a second axis.\n"
    "# - Keep CONCEPT_TYPE = '{parent_concept_type}' and\n"
    "#   CONCEPT_GROUP = '{parent_concept_group}' EXACTLY.\n"
    "# - If the parent has a banned pattern, remove it.\n"
    "#\n"
    + CONCEPT_DECLARATION
    + HARD_CONTRACT
)

MUTATE_BREADTH = (
    "# TASK: Write a generator in a DIFFERENT domain than the parent,\n"
    "# at matching quality. New mathematical object, no reuse.\n"
    "#\n"
    "# Parent program (context only — do NOT reuse its object):\n"
    "```python\n{code}\n```\n"
    "#\n"
    "{score_feedback}"
    "{exec_feedback}"
    "{few_shot}"
    "#\n"
    + QUALITY_BAR
    + COMPLEXITY_AXES +
    "# - Pick CONCEPT_TYPE from the whitelist whose CONCEPT_GROUP is\n"
    "#   NOT '{parent_concept_group}'.\n"
    "#   Suggested groups to try: {suggested_groups}.\n"
    "#\n"
    + CONCEPT_DECLARATION
    + HARD_CONTRACT
)

MUTATE_CROSSOVER = (
    "# TASK: Merge parents A and B into a hybrid generator whose single\n"
    "# mathematical object is the INTERSECTION of A's and B's ideas\n"
    "# (NOT a concatenation 'compute X from A, then Y from B').\n"
    "#\n"
    "# Parent A (p={p_hat_a:.2f}, H={h_a:.2f}):\n"
    "```python\n{code_a}\n```\n"
    "#\n"
    "# Parent B (p={p_hat_b:.2f}, H={h_b:.2f}):\n"
    "```python\n{code_b}\n```\n"
    "#\n"
    "{few_shot}"
    "#\n"
    + QUALITY_BAR
    + COMPLEXITY_AXES +
    "# - Identify a shared structure (group, graph, polynomial,\n"
    "#   probability space, modulus). Every step uses it.\n"
    "# - Define exactly one whitelisted CONCEPT_TYPE / CONCEPT_GROUP\n"
    "#   pair matching the hybrid object.\n"
    "#\n"
    + CONCEPT_DECLARATION
    + HARD_CONTRACT
)


# Backward-compatible re-export: external code may import SINGLE_ANSWER_RULE.
SINGLE_ANSWER_RULE = HARD_CONTRACT


# ---------------------------------------------------------------------------
# Diagnostic / feedback helpers
# ---------------------------------------------------------------------------

def score_diagnosis(p_hat: float, h_score: float) -> tuple[str, str]:
    """Verbose diagnosis + action text for a parent's (p_hat, H)."""
    # p_hat bucket
    if p_hat > 0.8:
        diag = f"far TOO EASY (solver wins {p_hat:.0%})"
        action = (
            "lift the difficulty via axes 1 (parametric), 2 (case "
            "analysis) or 3 (theorem chaining); DO NOT just enlarge "
            "numbers"
        )
    elif p_hat > 0.65:
        diag = f"slightly too easy (p={p_hat:.2f})"
        action = (
            "introduce one non-trivial structural constraint (axis 4) "
            "or a hidden identity (axis 6)"
        )
    elif p_hat < 0.15:
        diag = f"far TOO HARD (solver wins only {p_hat:.0%})"
        action = (
            "simplify one reasoning stage, clarify the mathematical "
            "OBJECT, keep the others — total depth should stay >= 3"
        )
    elif p_hat < 0.3:
        diag = f"slightly too hard (p={p_hat:.2f})"
        action = (
            "reduce numeric ranges but preserve structure; consider a "
            "case-analysis split (axis 2) so the solver can commit"
        )
    else:
        diag = f"good learnability band (p={p_hat:.2f})"
        action = (
            "preserve difficulty; push axis 5 (dimension lift) or 6 "
            "(hidden identity) to raise reasoning depth and diversity"
        )

    # Entropy bucket
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
            "# === HIGH-QUALITY EXAMPLES ===\n"
            "# (no reward-hack-free champions yet — design from the\n"
            "# rubric alone; do NOT imitate the parent's shape)\n"
            "# === END EXAMPLES ===\n"
        )

    parts = ["# === HIGH-QUALITY EXAMPLES (well-posed champions) ==="]
    for i, (champ, rendered) in enumerate(picks, 1):
        rq = getattr(champ, "rq_score", 0) or 0
        p = getattr(champ, "p_hat", 0) or 0
        h = getattr(champ, "h_score", 0) or 0
        parts.append(
            f"#\n# Example {i} (R_Q={rq:.3f}, p={p:.2f}, H={h:.2f}):\n"
            f"```python\n{champ.source_code}\n```"
        )
        if rendered:
            parts.append(f"# rendered(seed=0): {rendered[:180]}")
    parts.append("# === END EXAMPLES ===\n")
    return "\n".join(parts)


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
        "# === PARENT EXECUTION TRACE ===",
        f"# seed=0 problem: {inst0.problem[:160]}",
        f"# seed=0 answer : {inst0.answer}",
    ]
    if inst1:
        lines.append(f"# seed=1 problem: {inst1.problem[:160]}")
        lines.append(f"# seed=1 answer : {inst1.answer}")
        if inst1.problem.strip() == inst0.problem.strip():
            lines.append(
                "# WARNING: seed 0 and seed 1 produced identical text —\n"
                "#          the parent is effectively seed-independent.\n"
                "#          Fix this in the mutation."
            )
    lines.append(f"# solver pass rate: {p_hat:.0%} ({difficulty})")
    lines.append(f"# solver entropy  : {h_score:.2f}")
    lines.append("# === END PARENT TRACE ===")
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


def build_mutation_prefill(ctype: str, cgroup: str) -> tuple[str, str]:
    """Return (suffix_appended_to_prompt, prefix_for_extract_recovery).

    The suffix opens a Python code fence, declares the chosen
    CONCEPT_TYPE / CONCEPT_GROUP constants, and starts the
    `def generate(seed):` header so the model only needs to write the
    function body. The recovery prefix (suffix minus the fence) is
    prepended to the model's output before code extraction so a full,
    parseable program is reconstructed.
    """
    body = (
        "import random\n\n"
        f"CONCEPT_TYPE = \"{ctype}\"\n"
        f"CONCEPT_GROUP = \"{cgroup}\"\n\n"
        "def generate(seed):\n"
    )
    suffix = "\n```python\n" + body
    return suffix, body
