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
from rq_questioner.map_elites import MAPElitesGrid
from rq_questioner.program import ProblemProgram


# ---------------------------------------------------------------------------
# Static rubric blocks (concatenated into every mutation template)
# ---------------------------------------------------------------------------

SINGLE_ANSWER_RULE = (
    "# === HARD CONTRACT ===\n"
    "# 1. Function name MUST be `generate(seed)`.\n"
    "# 2. MUST return `(problem_text: str, answer: str)`.\n"
    "# 3. `answer` MUST be a SINGLE exact scalar value — integer,\n"
    "#    simple fraction like '3/7', surd like 'sqrt(2)', or a\n"
    "#    rational combination. NOT ranges, lists, inequalities,\n"
    "#    or decimals produced only by rounding a closed-form.\n"
    "# 4. Derive `answer` FIRST from a mathematical identity, then\n"
    "#    assemble the problem TEXT around that identity (inverse\n"
    "#    construction). Never: 'compute T; then answer = T + noise'.\n"
    "# 5. Use only stdlib + math + fractions + itertools + functools\n"
    "#    (sympy optional). No file I/O, no randomness beyond the\n"
    "#    seeded `random.Random(seed)`.\n"
    "# 6. Every seed in {{0,1,2,3,4}} MUST terminate fast (<1s) and\n"
    "#    produce a well-posed problem (no div-by-zero, no empty\n"
    "#    choices, no infinite loops).\n"
)

ANTI_REWARD_HACK_RULES = (
    "# === ANTI-REWARD-HACKING RUBRIC (STRICT) ===\n"
    "# Fitness is R_Q = p(1-p)·H. An easy way to maximise H is to\n"
    "# produce problems that make the solver *confused*, not ones\n"
    "# that make it *reason*. Such problems are worthless for RL.\n"
    "# The following constructions are BANNED and will be auto-\n"
    "# rejected by the verifier:\n"
    "#\n"
    "#   - `x % 1`, 'mod 1', 'remainder when X is divided by 1'\n"
    "#     (0 for ints, an uninterpretable fractional part for\n"
    "#      floats — no reasoning content).\n"
    "#   - Multiplying by 1, dividing by 1, adding 0, subtracting 0.\n"
    "#   - 'Round to N decimal places' when the exact answer is an\n"
    "#     integer or closed-form rational.\n"
    "#   - Chains of unrelated arithmetic ('compute A, then multiply\n"
    "#     by an unrelated variable e'); every step must follow\n"
    "#     from a NAMED mathematical technique.\n"
    "#   - Floating-point cascades whose answer depends on IEEE-754\n"
    "#     rounding or decimal-place formatting.\n"
    "#   - Assigning random letter-variables like `a=964, b=494, ...`\n"
    "#     that appear in the statement but have no role in the\n"
    "#     mathematical structure (word-problem noise).\n"
    "#   - Hiding an operation that collapses the answer to a\n"
    "#     trivial constant (e.g. `derangement(9) % 1 == 0`).\n"
    "#   - Problems whose answer doesn't change across the 5 test\n"
    "#     seeds (the seed is ignored).\n"
)

REAL_DIFFICULTY_RUBRIC = (
    "# === WHAT 'HARD' MEANS HERE ===\n"
    "# A good R_Q program produces problems with:\n"
    "#   (a) a single coherent mathematical OBJECT — an equation,\n"
    "#       a combinatorial configuration, a number-theoretic\n"
    "#       condition, a geometric figure, a recurrence, ...;\n"
    "#   (b) a reasoning trace of at least 3 stages, each applying\n"
    "#       a NAMED technique (Vieta, CRT, inclusion-exclusion,\n"
    "#       Fermat's little thm, generating functions, modular\n"
    "#       inverse, telescoping, Newton's identities, AM-GM, ...);\n"
    "#   (c) an exact closed-form answer derived from (a) by (b),\n"
    "#       not by ad-hoc concatenation of operations;\n"
    "#   (d) genuine solver uncertainty from *which technique to\n"
    "#       apply*, not from ambiguous wording.\n"
)

COMPLEXITY_AXES = (
    "# === COMPLEXITY AXES (pick 2 per mutation, not surface tweaks) ===\n"
    "# 1. Parametric lifting: replace a concrete constant with a\n"
    "#    parameter the solver must first DETERMINE (solve an\n"
    "#    auxiliary equation / read it off a divisibility cond.).\n"
    "# 2. Case analysis: introduce a discrete branch (parity,\n"
    "#    congruence class, sign, sub-interval) that forces the\n"
    "#    solver to enumerate cases and combine partial answers.\n"
    "# 3. Theorem chaining: the output of one theorem becomes the\n"
    "#    input of another (Vieta -> Newton's identity -> evaluate\n"
    "#    a symmetric polynomial; CRT -> modular inverse -> solve).\n"
    "# 4. Structural constraint: add a non-trivial condition that\n"
    "#    shrinks the feasible set (integrality, coprimality,\n"
    "#    monotonicity, convexity, divisibility).\n"
    "# 5. Dimension / domain lift: scalar -> vector, Z -> Z[i],\n"
    "#    real -> complex, 2D figure -> 3D figure, deterministic\n"
    "#    -> probabilistic.\n"
    "# 6. Hidden identity: the answer equals a well-known constant\n"
    "#    (Catalan, Stirling, binomial identity) via a bijection\n"
    "#    the solver must discover rather than a direct formula.\n"
)

CONCEPT_DECLARATION = (
    "# === CONCEPT DECLARATION (first lines of function body) ===\n"
    "# The first two comment lines inside `generate` MUST be:\n"
    "#   # CONCEPT: <primary_domain> / <secondary_domain_or_NONE>\n"
    "#   # TECHNIQUES: <comma-separated named techniques>\n"
    "# where primary/secondary domains are drawn from:\n"
    "#   algebra, number_theory, combinatorics, probability,\n"
    "#   geometry, trigonometry, analysis, recurrence, inequality,\n"
    "#   linear_algebra, logic.\n"
)

MUTATION_METHOD_RULE = (
    "# === MUTATION METHOD ===\n"
    "# 1. Read the parent program. Name its mathematical OBJECT,\n"
    "#    the techniques it uses, and its answer type.\n"
    "# 2. Check whether the parent contains any banned construction\n"
    "#    from the anti-hack rubric above. If it does, your mutation\n"
    "#    MUST REMOVE the banned construction entirely and replace\n"
    "#    it with a genuine reasoning step.\n"
    "# 3. Sketch 3 structurally distinct mutations along different\n"
    "#    complexity axes (1-6 above). Pick the one that adds the\n"
    "#    most genuine reasoning depth without breaking decidability.\n"
    "# 4. Do NOT merely rename variables or resize numeric ranges.\n"
    "#    Change the mathematical STRUCTURE.\n"
    "# 5. Before emitting code, mentally execute for seeds 0..4.\n"
    "#    Every seed must terminate, produce a distinct problem, and\n"
    "#    have a SymPy-parseable answer.\n"
)


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
    "# ROLE: You are a research mathematician designing a harder\n"
    "# variant of an existing problem-generation program for RL\n"
    "# training of a math-reasoning model.\n"
    "#\n"
    "# GOAL: A new `generate(seed)` whose problems have genuinely\n"
    "# deeper reasoning structure — NOT just bigger numbers or\n"
    "# longer text. Stay in the SAME mathematical domain as the\n"
    "# parent, but add reasoning depth.\n"
    "#\n"
    "# TARGET METRICS: p_hat ~ 0.5, H in [2.0, 4.0] from *real*\n"
    "# reasoning uncertainty (which technique to invoke, which\n"
    "# case to pick), not from ambiguous wording.\n"
    "#\n"
    "{few_shot}"
    "{score_feedback}"
    "{exec_feedback}"
    "#\n"
    + ANTI_REWARD_HACK_RULES
    + REAL_DIFFICULTY_RUBRIC
    + COMPLEXITY_AXES
    + MUTATION_METHOD_RULE +
    "# Combine at least TWO complexity axes (1-6) with at least one\n"
    "# additional technique beyond what the parent uses.\n"
    "#\n"
    "# Parent program:\n"
    "```python\n{code}\n```\n\n"
    + CONCEPT_DECLARATION
    + SINGLE_ANSWER_RULE +
    "# Now emit the new program (start with the CONCEPT/TECHNIQUES\n"
    "# comment immediately inside `generate`). Output ONLY code.\n"
    "```python\n"
    "import random\n"
)

MUTATE_BREADTH = (
    "# ROLE: You are a research mathematician DIVERSIFYING an\n"
    "# evolutionary archive of problem-generation programs for\n"
    "# RL training of a math-reasoning model.\n"
    "#\n"
    "# GOAL: A program in a COMPLETELY DIFFERENT mathematical\n"
    "# domain from the parent, with matching quality and depth.\n"
    "# The new problem must NOT be a rewording of the parent and\n"
    "# must NOT share its mathematical OBJECT.\n"
    "#\n"
    "# TARGET METRICS: p_hat ~ 0.5, H in [2.0, 4.0] from real\n"
    "# reasoning uncertainty.\n"
    "#\n"
    "{few_shot}"
    "{score_feedback}"
    "{exec_feedback}"
    "#\n"
    "# === DOMAIN SWITCH TABLE (pick a distant row, NOT adjacent) ===\n"
    "#   algebra / quadratics   -> number theory (CRT, LTE, orders)\n"
    "#   euclidean geometry     -> probability on discrete structures\n"
    "#   combinatorics          -> recurrences + generating functions\n"
    "#   probability            -> modular arithmetic / Bezout\n"
    "#   sequences              -> coordinate geometry / vectors\n"
    "#   trigonometry           -> complex numbers / roots of unity\n"
    "#   logarithms             -> inequalities (AM-GM, Jensen)\n"
    "#   linear algebra         -> discrete calculus / telescoping\n"
    "#   calculus               -> inclusion-exclusion / Polya\n"
    "# (If the parent is already in the right column, pick a left-\n"
    "# column entry or invent a further domain, e.g. p-adic tools.)\n"
    "#\n"
    + ANTI_REWARD_HACK_RULES
    + REAL_DIFFICULTY_RUBRIC
    + COMPLEXITY_AXES
    + MUTATION_METHOD_RULE +
    "# Parent program (for context — do NOT reuse its object):\n"
    "```python\n{code}\n```\n\n"
    + CONCEPT_DECLARATION
    + SINGLE_ANSWER_RULE +
    "# Emit a fresh generator in a distant domain. Output ONLY code.\n"
    "```python\n"
    "import random\n"
)

MUTATE_CROSSOVER = (
    "# ROLE: You are a research mathematician MERGING two problem-\n"
    "# generation programs into a hybrid that draws reasoning\n"
    "# depth from both parents.\n"
    "#\n"
    "# GOAL: A `generate(seed)` whose single mathematical OBJECT\n"
    "# is the INTERSECTION of ideas from parent A and parent B\n"
    "# (not a disjoint concatenation of their steps).\n"
    "# Examples of valid intersections:\n"
    "#   geometry ∩ probability   -> geometric probability over a\n"
    "#                              configuration (Buffon-like).\n"
    "#   number theory ∩ combinat.-> counting residues modulo m.\n"
    "#   algebra ∩ recurrence     -> characteristic polynomial of\n"
    "#                              a linear recurrence solved via\n"
    "#                              Vieta + roots.\n"
    "#   trig ∩ complex           -> De Moivre identities on roots\n"
    "#                              of unity.\n"
    "#\n"
    "# Invalid (forbidden): 'compute X from A, then compute Y from\n"
    "# B, then multiply'. That is concatenation, not crossover.\n"
    "#\n"
    "{few_shot}"
    "# Parent A (p_hat={p_hat_a:.2f}, H={h_a:.2f}):\n"
    "```python\n{code_a}\n```\n\n"
    "# Parent B (p_hat={p_hat_b:.2f}, H={h_b:.2f}):\n"
    "```python\n{code_b}\n```\n\n"
    "# TARGET METRICS: p_hat ~ 0.5, H in [2.0, 4.0] from real\n"
    "# reasoning uncertainty — not ambiguous wording.\n"
    "#\n"
    + ANTI_REWARD_HACK_RULES
    + REAL_DIFFICULTY_RUBRIC
    + COMPLEXITY_AXES
    + MUTATION_METHOD_RULE +
    "# Additional crossover rules:\n"
    "# - Identify a SHARED mathematical structure (a group action,\n"
    "#   a graph, a polynomial, a probability space, a modulus).\n"
    "# - Build the hybrid problem AROUND that structure; every\n"
    "#   reasoning step must use it.\n"
    "# - Declare BOTH domains in the CONCEPT line.\n"
    "#\n"
    + CONCEPT_DECLARATION
    + SINGLE_ANSWER_RULE +
    "# Emit the hybrid generator. Output ONLY code.\n"
    "```python\n"
    "import random\n"
)


# ---------------------------------------------------------------------------
# Anti-hack detection for diagnostics and few-shot filtering
# ---------------------------------------------------------------------------

_BANNED_PATTERNS = (
    "% 1",               # integer/float mod 1
    " mod 1",            # worded mod 1
    "divided by 1",      # worded divide by 1
    "divide by 1",
    "divides by 1",
    "multiplied by 1",
    "multiply by 1",
    "raised to 1",
    "to the power of 1",
    "to the power 1",
)


def has_anti_pattern(source_code: str, problem_text: str = "") -> bool:
    """True if the program source or rendered problem shows a known
    reward-hacking pattern. Cheap string check used to filter the
    few-shot pool and to attach a repair hint to the diagnosis."""
    haystack = (source_code + "\n" + problem_text).lower()
    # Avoid false-positives for legitimate `x % n` where n != 1.
    for pat in _BANNED_PATTERNS:
        if pat in haystack:
            return True
    # Catch `round(..., 0)` and similar rounding-as-obfuscation.
    if "round(" in haystack and ", 0)" in haystack:
        return True
    return False


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
    # Repair hint when the parent itself exhibits a banned pattern.
    if has_anti_pattern(parent.source_code):
        diag += "; PARENT CONTAINS A BANNED ANTI-PATTERN"
        action = (
            "REMOVE the banned construction entirely and replace it "
            "with a genuine reasoning step; " + action
        )
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
        # Render one seed to catch problem-text anti-patterns.
        inst = champ.execute(seed=0, timeout=3.0)
        rendered = inst.problem if inst else ""
        if has_anti_pattern(champ.source_code, rendered):
            continue
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
    if has_anti_pattern(parent.source_code, inst0.problem):
        lines.append(
            "# ANTI-PATTERN DETECTED in parent — the mutation MUST\n"
            "# eliminate the banned construction, not propagate it."
        )
    lines.append("# === END PARENT TRACE ===")
    return "\n".join(lines) + "\n"
