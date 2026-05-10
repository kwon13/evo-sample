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

import re

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
    "Invalid child (rejected — inconsistent givens):\n"
    "  \"x^2 + 5x + 6 = 0 has roots whose product is 12; find sum.\"\n"
    "  Vieta forces product = 6, contradicting the stated 12.\n"
    "\n"
    "Invalid child (rejected — under-specified object):\n"
    "  \"A quadratic has root 3; find the other root.\"\n"
    "  Infinitely many quadratics fit; no unique answer.\n"
    "\n"
    "Invalid child (rejected — degenerate parameters):\n"
    "  \"x^2 + 0*x + 0 = 0; find sum of roots.\"\n"
    "  Sum is 0 trivially; no Vieta reasoning required.\n"
    "\n"
    "The deep child requires Vieta's formulas plus a one-line "
    "reciprocal identity. The shallow child only requires reading off "
    "a coefficient. The three invalid children are rejected even at "
    "matching pass rate because they fail mathematical validity.\n"
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

VALIDITY_CONTRACT = (
    "Validity contract (mandatory).\n"
    "\n"
    "Inside generate(seed), REJECT any random parameter combination that "
    "produces a mathematically invalid, contradictory, under-specified, "
    "or degenerate-trivial problem. Use rejection sampling or "
    "deterministic construction — never emit such a problem.\n"
    "\n"
    "Forbidden parameter patterns and required fixes:\n"
    "\n"
    "  1. INCONSISTENT GIVENS — declared invariants must hold.\n"
    "     BAD : sample g, a, b independently then state \"gcd(a,b)=g\" "
    "without enforcing g|a and g|b.\n"
    "     FIX : sample coprime k1, k2 then set a = g*k1, b = g*k2; or "
    "compute g = math.gcd(a, b) deterministically and state that value.\n"
    "\n"
    "  2. IMPOSSIBLE COUNTING — count is 0 or 1 by parity / pigeonhole.\n"
    "     BAD : \"derangements of n with exactly n-1 fixed points\" "
    "(forces 0 by parity), \"exactly n fixed points\" (always 1).\n"
    "     FIX : restrict the count parameter k to 0..n-2 and forbid "
    "k = n-1 and k = n.\n"
    "\n"
    "  3. CONTRADICTORY CONSTRAINTS — extra clauses unsatisfiable.\n"
    "     BAD : \"three positive integers 1, 19, 19 with 1 > 19 and "
    "19 > 19; find their mean\".\n"
    "     FIX : never bolt on numeric inequalities you did not verify "
    "against the chosen values; if you state x > y, x must exceed y.\n"
    "\n"
    "  4. DEGENERATE INPUTS — n = 0, 1, 2 of a quantity that needs >= 3.\n"
    "     BAD : \"sum of first 1 terms of a geometric sequence\", "
    "\"product over a single element\".\n"
    "     FIX : sample n from a domain-appropriate minimum (n >= 3 for "
    "sequences and counting; n >= 4 for non-trivial recursions).\n"
    "\n"
    "  5. UNDER-SPECIFIED OBJECT — missing data needed for unique answer.\n"
    "     BAD : \"triangle with sides 6 and 7, find the area\" — needs "
    "either the third side, an included angle, or a triangle type "
    "(right / equilateral / isosceles).\n"
    "     FIX : provide enough data for a unique solution by construction.\n"
    "\n"
    "  6. ANSWER UNRELATED TO STATEMENT — solver gets it right by chance.\n"
    "     BAD : statement claims gcd(a,b)=g but the answer recipe ignores "
    "g entirely; statement claims a sequence is geometric but the answer "
    "is computed assuming arithmetic.\n"
    "     FIX : derive the answer from exactly the parameters and "
    "relations stated in the problem text.\n"
    "\n"
    "Self-check before returning. From ONLY the rendered problem text "
    "(not the source code), can you reach the stated answer with a clean "
    "derivation? If not, the parameters are inconsistent or the problem "
    "is under-specified — regenerate.\n"
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
    "Mathematical validity is non-negotiable. See the Validity contract in "
    "the user message: every emitted (problem, answer) pair must be "
    "internally consistent, fully specified, and non-degenerate. Use "
    "rejection sampling or deterministic construction to enforce "
    "invariants — do not let randomness produce contradictory or "
    "impossible problems.\n"
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
    + VALIDITY_CONTRACT
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
    + VALIDITY_CONTRACT
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
    + VALIDITY_CONTRACT
)


# Backward-compatible re-export: external code may import SINGLE_ANSWER_RULE.
SINGLE_ANSWER_RULE = HARD_CONTRACT


# ---------------------------------------------------------------------------
# Diagnostic / feedback helpers
# ---------------------------------------------------------------------------

_TRIVIAL_ANSWERS = {"0", "1", "-0", "-1", "0.0", "1.0", "-0.0", "-1.0"}


def score_diagnosis(
    p_hat: float,
    h_score: float,
    *,
    parent_answer: str | None = None,
) -> tuple[str, str]:
    """Verbose diagnosis + action text for a parent's (p_hat, H).

    ``parent_answer`` is the rendered seed=0 answer string; when supplied,
    a trivial 0/1 answer in the mid-p band is flagged as a likely-broken
    problem (impossible counting, contradictory constraint, empty set).
    This catches the failure mode where the answer is mechanically 0 or 1
    by construction and p_hat ~ 0.5 looks healthy from the band check
    alone.

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

    if parent_answer is not None:
        a_norm = str(parent_answer).strip()
        if a_norm in _TRIVIAL_ANSWERS and 0.3 <= p_hat <= 0.7:
            diag += "; SUSPECT broken (trivial 0/1 answer in mid-p band)"
            action += (
                "; verify the problem has a unique non-trivial solution. "
                "If the answer is mechanically 0 or 1 by construction "
                "(impossible counting, contradictory constraint, empty "
                "set), redesign the parameter ranges to exclude that "
                "case and pick parameters where the answer is "
                "non-trivially derived from the stated quantities"
            )

    return diag, action


def build_score_feedback(parent: ProblemProgram) -> str:
    p_hat = getattr(parent, "p_hat", 0.5)
    h_score = getattr(parent, "h_score", 1.0)
    rq_score = getattr(parent, "rq_score", 0.0)
    parent_answer: str | None = None
    try:
        inst = parent.execute(seed=0, timeout=3.0)
        if inst is not None:
            parent_answer = str(inst.answer)
    except Exception:
        parent_answer = None
    diag, action = score_diagnosis(p_hat, h_score, parent_answer=parent_answer)
    return SCORE_FEEDBACK.format(
        p_hat=p_hat, h_score=h_score, rq_score=rq_score,
        diagnosis=diag, action=action,
    )


# ---------------------------------------------------------------------------
# Validity heuristics — block broken champions from few-shot / training
# ---------------------------------------------------------------------------

def looks_broken(problem: str, answer: str) -> bool:
    """Heuristic detector for broken-problem patterns observed in archive
    contamination. Conservative by design — only flags patterns we have
    direct evidence for.

    False positive = champion excluded from few-shot / training; recoverable
    once mutations replace it.
    False negative = contamination propagates; this is the worse failure
    mode, but the conservative scope keeps the heuristic honest.

    Patterns covered:
      1. Trivial 0/1 answer with impossible-counting language.
      2. Self-contradicting numeric inequality clause (n > n, x > y when
         x <= y is in the same sentence).
      3. GCD/LCM consistency — gcd claim with values that don't satisfy it.
      4. Under-specified geometry (triangle area without 3rd side / angle).
      5. Malformed LaTeX (\\frac with run-on digits).
      6. Degenerate sequence count ("first 1 term", "first 2 terms").
    """
    p = (problem or "").lower().strip()
    a = str(answer or "").strip()

    # 1. Trivial 0/1 answer paired with impossible-counting language
    if a in _TRIVIAL_ANSWERS:
        if "exactly" in p and ("fixed point" in p or "their original" in p):
            return True
        if re.search(r"first\s+1\s+term", p):
            return True
        if "single element" in p and "product" in p:
            return True

    # 2. Self-contradicting inequality, e.g. "19 > 19"
    if re.search(r"\b(\d+)\s*>\s*\1\b", p):
        return True
    # "ensure 1 > 19" (left <= right)
    for m in re.finditer(r"\b(\d+)\s*>\s*(\d+)\b", p):
        try:
            if int(m.group(1)) <= int(m.group(2)):
                return True
        except ValueError:
            continue

    # 3. GCD/LCM consistency — if the text states gcd value with explicit
    # numbers and g does not divide both a and b, it is broken. Covers
    # "gcd(a,b) = g", "gcd of A and B is g", and parenthetical-abbreviation
    # form "greatest common divisor (GCD) is g".
    if "gcd" in p or "greatest common divisor" in p:
        gcd_patterns = [
            r"gcd\s*\([^)]*\)\s*(?:=|is|equals)\s*(\d+)",
            r"gcd\s+(?:of\s+)?[^.]{0,80}?\bis\s+(\d+)",
            r"greatest\s+common\s+divisor[^.]{0,80}?\bis\s+(\d+)",
        ]
        for pat in gcd_patterns:
            m = re.search(pat, p)
            if not m:
                continue
            try:
                g_n = int(m.group(1))
            except ValueError:
                continue
            nums = [int(x) for x in re.findall(r"\b\d+\b", p[: m.start()])]
            if len(nums) >= 2 and g_n > 0:
                a_n, b_n = nums[-2], nums[-1]
                if a_n > 0 and b_n > 0 and (a_n % g_n != 0 or b_n % g_n != 0):
                    return True
            break

    # 4. Under-specified triangle area
    if "triangle" in p and "area" in p:
        has_extra_info = bool(
            re.search(r"\b(angle|included|right\s+triangle|equilateral|"
                      r"isosceles|heron|altitude|height|perimeter|hypotenuse|"
                      r"third\s+side)\b", p)
        )
        if not has_extra_info:
            sides_match = re.search(
                r"sides?\s+(?:of\s+(?:length|lengths?)\s+)?"
                r"(\d+)\s*(?:,|and)\s*(\d+)(?:\s*(?:,|and)\s*(\d+))?",
                p,
            )
            if sides_match and not sides_match.group(3):
                return True

    # 5. Malformed LaTeX (e.g. \frac25002 — should be \frac{2500}{2})
    if re.search(r"\\frac\d{4,}", p):
        return True

    # 6. Degenerate sequence count
    if re.search(r"sum\s+of\s+(?:the\s+)?first\s+(\d+)\s+terms?", p):
        m = re.search(r"sum\s+of\s+(?:the\s+)?first\s+(\d+)\s+terms?", p)
        try:
            if m and int(m.group(1)) <= 2:
                return True
        except ValueError:
            pass

    return False


def champion_passes_validity(
    champ: ProblemProgram,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    use_cache: bool = True,
) -> bool:
    """Multi-seed validity check. Reject the champion if ANY seed produces
    a broken-looking instance, OR if all seeds yield identical answers
    (seed-independent generator).

    Strict policy — even one broken seed is enough to reject. Per the user
    review: false positives are recoverable, contamination propagation is
    not.

    Caching:
      Result is stored on ``champ.metadata['validity_check']`` keyed by
      the champion's current ``rq_score``. When re-evaluation updates
      rq_score the cache invalidates and we re-execute the seeds.
    """
    if champ is None:
        return False

    rq_now = getattr(champ, "rq_score", None)
    if rq_now is None:
        rq_now = 0.0

    meta = getattr(champ, "metadata", None) or {}
    cache = meta.get("validity_check") if use_cache else None
    if cache is not None and cache.get("rq_score_at_check") == rq_now:
        return bool(cache.get("passed"))

    answers: list[str] = []
    broken_seeds: list[int] = []
    exec_failed_seeds: list[int] = []
    for s in seeds:
        try:
            inst = champ.execute(seed=s, timeout=3.0)
        except Exception:
            exec_failed_seeds.append(s)
            continue
        if inst is None:
            exec_failed_seeds.append(s)
            continue
        ans = str(inst.answer or "").strip()
        if not ans:
            broken_seeds.append(s)
            continue
        if looks_broken(inst.problem or "", ans):
            broken_seeds.append(s)
            continue
        answers.append(ans)

    n_total = len(seeds)
    n_broken = len(broken_seeds) + len(exec_failed_seeds)
    seed_independent = (len(answers) >= 2 and len(set(answers)) == 1)
    passed = (n_broken == 0) and not seed_independent and len(answers) == n_total

    if champ.metadata is None:
        champ.metadata = {}
    champ.metadata["validity_check"] = {
        "passed": passed,
        "broken_seeds": broken_seeds,
        "exec_failed_seeds": exec_failed_seeds,
        "seed_independent": seed_independent,
        "n_valid": len(answers),
        "n_total": n_total,
        "rq_score_at_check": rq_now,
    }
    return passed


def build_few_shot_examples(
    grid: MAPElitesGrid,
    top_k: int = 3,
    min_rq: float = 0.25,
) -> str:
    """Top-RQ champions as few-shot, with multi-seed validity filtering.

    A champion is excluded from few-shot if any of seeds 0..4 produces
    a broken-looking instance, or if all 5 seeds yield identical answers
    (seed-independent generator). This blocks contamination — once a
    broken champion enters the archive, it must NOT be paraded as a
    high-quality example for the mutator to imitate.

    If no clean champion is found, emit an explicit "examples
    unavailable" note rather than silently leaking dirty examples.
    """
    champions = grid.get_all_champions()
    if not champions:
        return ""

    clean = []
    for champ in champions:
        if (champ.rq_score or 0.0) < min_rq:
            continue
        if not champion_passes_validity(champ):
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
            "No validity-passing champions are available yet. Design "
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
