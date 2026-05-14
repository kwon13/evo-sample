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
from pathlib import Path

from rq_questioner.concepts import concept_prompt_block
from rq_questioner.program import ProblemProgram


# ---------------------------------------------------------------------------
# Static rubric blocks (concatenated into every mutation template)
# ---------------------------------------------------------------------------


MUTATION_SYSTEM_PROMPT = (
    "You design Python generators for competition-math problems. "
    "Each generator defines `generate(seed)` and returns one "
    "(problem_text, answer) pair.\n"
    "\n"
    "Validity is non-negotiable. The user message specifies the "
    "validity contract; enforce its invariants by construction or "
    "rejection sampling, never by hoping randomness cooperates.\n"
    "\n"
    "Output only Python source — no preamble, no markdown, no prose "
    "outside the code. Inside `generate()`, brief one-line comments "
    "are fine; no docstrings or multi-line prose.\n"
    "\n"
    "FIRST, in your private scratch-pad, think step-by-step to design "
    "a brand-new, non-trivial generator whose outputs are "
    "mathematically valid by construction.\n"
    "THEN, without revealing any of your private thoughts, output "
    "only the Python source for the generator."
)


# ---------------------------------------------------------------------------
# Score feedback template
# ---------------------------------------------------------------------------

SCORE_FEEDBACK = (
    "Parent diagnosis.\n"
    "\n"
    "  observed solver pass rate p = {p_hat:.2f}\n"
    "  observed solver entropy   H = {h_score:.2f}\n"
    "\n"
    "{diagnosis}\n"
    "\n"
    "Action: {action}\n"
    "\n"
    "Note on the metrics above. They describe the CURRENT solver's "
    "behaviour on the parent generator — they are diagnostic, not a "
    "target. Use them only to judge direction: e.g. p very high means "
    "the parent is already mastered (make harder), p very low means "
    "the parent is unreachable (preserve structure but reduce numeric "
    "load), low H means the solver is over-confident (introduce a real "
    "ambiguity). Do NOT design the child to hit a specific numerical p "
    "or H value, and do NOT reference these numbers in the generated "
    "code or problem text. The reward signal itself (R_Q) is "
    "deliberately not shown to prevent reward-hacking.\n"
    "\n"
)


# ---------------------------------------------------------------------------
# Top-level mutation templates
# ---------------------------------------------------------------------------

MUTATE_DEPTH = (
    "Task: write a deeper variant of the parent generator. Stay in "
    "the same domain. Depth means wrapping the parent's object in a "
    "classical identity, lifting it one level (scalar -> sequence "
    "term, single root -> root pair), or chaining a second theorem "
    "onto its result — not bigger numbers, not wrapper branches.\n"
    "{few_shot}"
    "\n"
    "Parent program:\n"
    "```python\n{code}\n```\n"
    "\n"
    "{score_feedback}"
    "{exec_feedback}"
    "Constraints for this mutation:\n"
    "  - Add one substantive reasoning stage beyond the parent "
    "(a named identity, theorem, or structural lift).\n"
    "  - Keep CONCEPT_TYPE = '{parent_concept_type}' and "
    "CONCEPT_GROUP = '{parent_concept_group}' exactly.\n"
    "  - Do not inherit reward-hack patterns from the parent; the "
    "child's final answer should stay compact.\n"
    "\n"
)

MUTATE_BREADTH = (
    "Task: write a generator in a different domain from the parent, "
    "at matching quality. Breadth means choosing a new mathematical "
    "object whose core reasoning has no overlap with the parent's "
    "(combinatorics -> geometry via power-of-a-point, algebra -> "
    "number theory via Legendre symbols) — not the same object in "
    "new clothing, not a renaming.\n"
    "{few_shot}"
    "\n"
    "Parent program (context only — do not reuse its object):\n"
    "```python\n{code}\n```\n"
    "\n"
    "{score_feedback}"
    "{exec_feedback}"
    "Constraints for this mutation:\n"
    "  - Pick a new CONCEPT_TYPE whose CONCEPT_GROUP is NOT "
    "'{parent_concept_group}'.\n"
    "  - The child's reasoning chain must not reuse the parent's "
    "central technique (Vieta, Euclid, binomial, etc.) — discover "
    "a new one for the new object.\n"
    "  - Match the parent's depth: at least three reasoning stages "
    "and a compact final answer.\n"
    "\n"
)

MUTATE_CROSSOVER = (
    "Task: merge parents A and B into a hybrid generator. Crossover "
    "means finding a single mathematical object on which BOTH "
    "parents' core ideas act simultaneously (a quadratic over GF(p) "
    "where Vieta and Legendre multiplicativity both apply; C(n,k) "
    "mod p where binomial counting and Lucas's theorem both apply) "
    "— not a concatenation like \"compute X from A, then Y from B\".\n"
    "{few_shot}"
    "\n"
    "Parent A (solver p={p_hat_a:.2f}, H={h_a:.2f}):\n"
    "```python\n{code_a}\n```\n"
    "\n"
    "Parent B (solver p={p_hat_b:.2f}, H={h_b:.2f}):\n"
    "```python\n{code_b}\n```\n"
    "\n"
    "Constraints for this mutation:\n"
    "  - Identify a shared structure (group, graph, polynomial, "
    "probability space, modulus). Every reasoning step must use it.\n"
    "  - Both parents' techniques must fire on the same object — "
    "if either parent's idea could be removed without changing the "
    "answer, the hybrid has failed.\n"
    "  - Declare one CONCEPT_TYPE / CONCEPT_GROUP pair matching the "
    "hybrid object, using the form 'group.snake_case_name'.\n"
    "  - Match the depth of the stronger parent and keep the final "
    "answer compact.\n"
    "\n"
)

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
    parent_answer: str | None = None
    try:
        inst = parent.execute(seed=0, timeout=3.0)
        if inst is not None:
            parent_answer = str(inst.answer)
    except Exception:
        parent_answer = None
    # p_hat / h_score are interpolated (diagnostic direction signal — empirical
    # ablation showed stripping them caused the archive to drift away from the
    # frontier band: p_hat mean 0.43 → 0.63, frontier_fraction 60% → 29%
    # at matched evo budget).
    # rq_score is the reward-function value itself and is NOT interpolated;
    # exposing it would let the mutator reward-hack the metric directly.
    diag, action = score_diagnosis(p_hat, h_score, parent_answer=parent_answer)
    return SCORE_FEEDBACK.format(
        p_hat=p_hat, h_score=h_score, diagnosis=diag, action=action,
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
    mode, but each pattern below has documented archive evidence.

    Patterns covered:
      Math validity (1st-order failures):
        1. Trivial 0/1 answer with impossible-counting language.
        2. Self-contradicting numeric inequality clause (n > n, x > y when
           x <= y is in the same sentence).
        3. GCD/LCM consistency — gcd claim with values that don't satisfy it.
        4. Under-specified geometry (triangle area without 3rd side / angle).
        5. Malformed LaTeX (\\frac with run-on digits).
        6. Degenerate sequence count ("first 1 term", "first 2 terms").
      Quality / hint-leak (2nd-order failures, added later):
        7. Both gcd AND lcm given as premise (task collapses to arithmetic).
        8. "coprime" / "are reciprocals" / "perfect square" hint words that
           leak the discriminating property.
        9. Variable degeneracy a=b (or n=k for committee-style).
       10. Degenerate triangle (0° angle, 0-length side, n-gon with n<=2).
       11. Committee of size n from n people (no choice involved); committee
           of size 1 (trivially anyone).
       12. "Given that <quantity> is X" pattern that states the very
           quantity the solver should derive.
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
    # numbers and g does not divide both a and b, it is broken.
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
                      r"third\s+side|vertices|coordinates|base)\b", p)
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

    # ---- 2nd-order quality patterns (hint leak / degeneracy) ---------------

    # 7. Both gcd AND lcm stated as premise — task collapses
    if (("gcd" in p or "greatest common divisor" in p)
            and ("lcm" in p or "least common multiple" in p)):
        # Two distinct "<noun> is/equals <number>" claims within one sentence
        gcd_value_pat = (
            r"(?:gcd|greatest common divisor)[^.]{0,80}?"
            r"\b(?:is|equals|=)\s*(\d+)"
        )
        lcm_value_pat = (
            r"(?:lcm|least common multiple)[^.]{0,80}?"
            r"\b(?:is|equals|=)\s*(\d+)"
        )
        if re.search(gcd_value_pat, p) and re.search(lcm_value_pat, p):
            return True

    # 8. Hint-leak property names — telling solver the trick by word.
    # Each phrase only fires when it actually short-circuits reasoning.
    # "is a primitive root" was previously here but removed: in modular
    # arithmetic problems (e.g. kth_root_mod_prime) naming the primitive
    # root is a NECESSARY structural premise, not a hint leak. Same for
    # "is a prime" in legitimate Legendre / quadratic-residue setups.
    HINT_LEAK_PHRASES = (
        "coprime",
        "are reciprocals",
        "roots are reciprocal",
        "is a perfect square",
        "are a pythagorean",
        "form a pythagorean",
    )
    for phrase in HINT_LEAK_PHRASES:
        if phrase in p:
            return True

    # 9. Variable degeneracy — same number assigned to two distinct variables
    # in a context where the resulting symmetry trivializes the concept.
    # Narrowed to mean / AM-GM / arithmetic-mean style problems where a=b
    # makes the answer equal to a; symmetric / equilateral / repeated-
    # coefficient problems (Vieta with double roots, Heron with isosceles,
    # symmetric polynomial identities) are legitimate and not blocked.
    _DEGENERACY_TRIGGERS = (
        "arithmetic mean",
        "geometric mean",
        "harmonic mean",
        "average of",
        "am-gm",
        "am gm",
    )
    if any(trig in p for trig in _DEGENERACY_TRIGGERS):
        var_assign_matches = re.findall(
            r"\b([a-z])\s*=\s*(-?\d+(?:\.\d+)?)\b", p
        )
        if len(var_assign_matches) >= 2:
            var_values = {}
            for var, val in var_assign_matches:
                var_values.setdefault(var, val)
            vals = list(var_values.values())
            if len(var_values) >= 2 and len(set(vals)) < len(vals):
                return True

    # 10. Degenerate triangle / polygon geometry
    if "triangle" in p:
        # 0° or 180° anywhere in the text — even without word boundary
        # because the unicode ° has no \b on its left.
        if re.search(r"(?<!\d)0\s*(?:°|degrees?)", p):
            return True
        if re.search(r"(?<!\d)180\s*(?:°|degrees?)", p):
            return True
    if re.search(r"\b(?:polygon|n-gon)\b", p):
        # n=2 vertices or sides
        m = re.search(r"\b(\d+)[\-\s]+(?:vertex|vertices|sided|sides)\b", p)
        if m:
            try:
                if int(m.group(1)) <= 2:
                    return True
            except ValueError:
                pass

    # 11. Committee size collapse — committee of size n from n people, or
    # committee of size 0 / n with trivial answer.
    # Two phrasing orders observed:
    #   "committee of size K from a group of N people"
    #   "in a group of N people, committee of size K"
    if "committee" in p and "people" in p:
        size_match = re.search(
            r"committees?\s+of\s+size[s]?\s+(\d+)(?:\s+and\s+(\d+))?", p
        )
        group_match = re.search(r"group\s+of\s+(\d+)\s+people", p) \
            or re.search(r"(\d+)\s+people", p)
        if size_match and group_match:
            try:
                k = int(size_match.group(1))
                n = int(group_match.group(1))
                if k == n or k == 0:
                    return True
                # "two committees of sizes 7 and 1 from 8 people" = trivial
                if size_match.group(2):
                    k2 = int(size_match.group(2))
                    if k + k2 == n and (k == 1 or k2 == 1):
                        return True
            except ValueError:
                pass

    # 12. "Given that <X> is <num>" pattern that states a derived quantity
    # Examples: "given that their gcd is 27", "given that their sum is S"
    # Catches the cases where the problem hands the solver the answer step.
    if a in _TRIVIAL_ANSWERS or (a.replace(".", "").replace("-", "").isdigit()):
        # If the answer literal appears verbatim in the problem (a known
        # hint-leak failure mode), reject.
        try:
            a_int = int(float(a))
            # Only flag for non-zero answer; 0 is too common to use as anchor
            if abs(a_int) >= 2 and re.search(
                rf"\b{re.escape(str(a_int))}\b", problem or ""
            ):
                # Avoid false positives: the answer must not be a stated
                # parameter (e.g. "a=7" with answer 7 happens legitimately).
                # Require the answer to appear ALSO outside any "x = N"
                # assignment context.
                stripped = re.sub(
                    r"\b[a-z]\s*=\s*\d+", "", problem.lower()
                )
                if re.search(rf"\b{re.escape(str(a_int))}\b", stripped):
                    # Even after stripping assignments, answer appears →
                    # likely hint-leak. But only flag when paired with
                    # other suspicious tokens to avoid massive FP rate.
                    if any(
                        tok in p for tok in (
                            "given that",
                            "such that the",
                            "given the value",
                        )
                    ):
                        return True
        except (ValueError, OverflowError):
            pass

    return False


def champion_passes_validity(
    champ: ProblemProgram,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    use_cache: bool = True,
) -> bool:
    """Multi-seed validity check. Reject the champion if ANY of:
      - any seed produces a broken-looking instance,
      - exec fails on any seed,
      - generated problem texts are not strictly all distinct
        across the seeds (character-wise comparison),
      - generated answers are not strictly all distinct across the seeds.

    Strict policy — anything less than full seed variation indicates a
    generator that is effectively constant (or thinly re-worded around a
    fixed answer) and is therefore treated as too-easy / contaminating.
    Per the user review: false positives are recoverable, contamination
    propagation is not.

    Distinctness is decided by exact string equality (whitespace-stripped),
    not by length, so generators that change only one numeric field while
    keeping the surrounding wording verbatim still count as distinct as
    long as the resulting strings differ at any character position.

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
    problems: list[str] = []
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
        problems.append((inst.problem or "").strip())

    n_total = len(seeds)
    n_broken = len(broken_seeds) + len(exec_failed_seeds)
    n_distinct_problems = len(set(problems))
    n_distinct_answers = len(set(answers))

    # Strict distinctness: every seed must produce a distinct problem text
    # AND a distinct answer. Anything less is treated as constant / thin
    # rewording and rejected. Only meaningful when every seed produced
    # output; if some seeds failed the check falls back to the legacy
    # all-identical-answer rejection.
    if len(answers) == n_total:
        problem_invariant = n_distinct_problems < n_total
        answer_invariant = n_distinct_answers < n_total
        seed_invariant = problem_invariant or answer_invariant
    else:
        problem_invariant = False
        answer_invariant = (len(answers) >= 2 and n_distinct_answers == 1)
        seed_invariant = answer_invariant

    passed = (
        n_broken == 0
        and not seed_invariant
        and len(answers) == n_total
    )

    if champ.metadata is None:
        champ.metadata = {}
    champ.metadata["validity_check"] = {
        "passed": passed,
        "broken_seeds": broken_seeds,
        "exec_failed_seeds": exec_failed_seeds,
        "seed_invariant": seed_invariant,
        "problem_invariant": problem_invariant,
        "answer_invariant": answer_invariant,
        "n_distinct_problems": n_distinct_problems,
        "n_distinct_answers": n_distinct_answers,
        "n_valid": len(answers),
        "n_total": n_total,
        "rq_score_at_check": rq_now,
    }
    return passed


# ---------------------------------------------------------------------------
# Operator-aware few-shot loader
# ---------------------------------------------------------------------------
#
# Few-shot examples are loaded from hand-curated files, one per mutation
# operator. Archive champions are deliberately NOT used as few-shot, so
# archive contamination cannot propagate into the mutator's signal and
# each operator receives examples tailored to its specific shape.


_SHOT_DIR = Path(__file__).parent  # prompts/

_SHOT_FILES: dict[str, Path] = {
    "in_depth":   _SHOT_DIR / "in_depth_shot.txt",
    "in_breadth": _SHOT_DIR / "in_breadth_shot.txt",
    "crossover":  _SHOT_DIR / "crossover_shot.txt",
}

# Per-(op, top_k) cached rendered shot text — files are parsed once.
_SHOT_CACHE: dict[str, str] = {}


def _parse_paired_shots(text: str) -> list[tuple[str, str]]:
    """Parse PARENT_PROGRAM_EXAMPLE_N / MUTATED_PROGRAM_EXAMPLE_N pairs."""
    blocks = re.split(r"\n-{3,}\n", text)
    pairs: list[tuple[str, str]] = []
    for blk in blocks:
        parents = re.findall(
            r"PARENT_PROGRAM_EXAMPLE_\d+:\s*```python\n(.*?)```",
            blk, re.S,
        )
        children = re.findall(
            r"MUTATED_PROGRAM_EXAMPLE_\d+:\s*```python\n(.*?)```",
            blk, re.S,
        )
        if parents and children:
            pairs.append((parents[0].strip(), children[0].strip()))
    return pairs


def _parse_crossover_shots(text: str) -> list[tuple[str, str, str]]:
    """Parse PARENT_A / PARENT_B / CROSSOVER_CHILD triples."""
    blocks = re.split(r"\n-{3,}\n", text)
    triples: list[tuple[str, str, str]] = []
    for blk in blocks:
        as_ = re.findall(
            r"PARENT_A_PROGRAM_EXAMPLE_\d+:\s*```python\n(.*?)```",
            blk, re.S,
        )
        bs_ = re.findall(
            r"PARENT_B_PROGRAM_EXAMPLE_\d+:\s*```python\n(.*?)```",
            blk, re.S,
        )
        cs_ = re.findall(
            r"CROSSOVER_CHILD_PROGRAM_EXAMPLE_\d+:\s*```python\n(.*?)```",
            blk, re.S,
        )
        if as_ and bs_ and cs_:
            triples.append((as_[0].strip(), bs_[0].strip(), cs_[0].strip()))
    return triples


def _no_examples_fallback(op: str) -> str:
    return (
        f"High-quality {op} examples.\n"
        f"\n"
        f"No example pairs are available yet. Design from the rubric "
        f"alone and do not imitate the parent's shape.\n"
        f"\n"
    )


def _format_paired_shots(
    op: str, pairs: list[tuple[str, str]], top_k: int
) -> str:
    parts = [f"High-quality {op} examples (parent shape -> mutated child).\n"]
    for i, (parent_src, child_src) in enumerate(pairs[:top_k], 1):
        parts.append(
            f"Example {i}.\n"
            f"  Parent (before {op}):\n"
            f"  ```python\n{parent_src}\n  ```\n"
            f"  Mutated child (after {op}):\n"
            f"  ```python\n{child_src}\n  ```\n"
        )
    return "\n".join(parts) + "\n"


def _format_crossover_shots(
    triples: list[tuple[str, str, str]], top_k: int
) -> str:
    parts = [
        "High-quality crossover examples (parent A + parent B -> hybrid "
        "child whose single mathematical object is the intersection of "
        "A's and B's ideas, NOT a concatenation of A then B).\n"
    ]
    for i, (a_src, b_src, c_src) in enumerate(triples[:top_k], 1):
        parts.append(
            f"Example {i}.\n"
            f"  Parent A:\n  ```python\n{a_src}\n  ```\n"
            f"  Parent B:\n  ```python\n{b_src}\n  ```\n"
            f"  Hybrid child:\n  ```python\n{c_src}\n  ```\n"
        )
    return "\n".join(parts) + "\n"


def build_few_shot_examples(op: str, top_k: int = 2) -> str:
    """Operator-aware few-shot loaded from prompts/<op>_shot.txt.

    The shot files are hand-curated and parsed once per process; archive
    champions are intentionally not consulted here.

    Returns a rubric-section text block ready to fill the {few_shot}
    slot of a mutation template. A "no examples available" fallback is
    emitted when the file is missing or parses to no pairs/triples.
    """
    if op not in _SHOT_FILES:
        return ""
    cache_key = f"{op}::{top_k}"
    cached = _SHOT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    path = _SHOT_FILES[op]
    if not path.exists():
        rendered = _no_examples_fallback(op)
        _SHOT_CACHE[cache_key] = rendered
        return rendered

    raw = path.read_text()
    if op == "crossover":
        triples = _parse_crossover_shots(raw)
        rendered = (
            _format_crossover_shots(triples, top_k)
            if triples else _no_examples_fallback(op)
        )
    else:
        pairs = _parse_paired_shots(raw)
        rendered = (
            _format_paired_shots(op, pairs, top_k)
            if pairs else _no_examples_fallback(op)
        )
    _SHOT_CACHE[cache_key] = rendered
    return rendered


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


def parent_concept_fields(parent: ProblemProgram) -> tuple[str, str]:
    """Return (concept_type, concept_group) for prompt interpolation.
    Falls back to 'unknown' when parent metadata is missing — concept
    rescue at extraction time handles those cases."""
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
    return ctype, cgroup


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
    if op == "in_depth":
        body = (
            f"CONCEPT_GROUP = \"{cgroup}\"\n"
            "CONCEPT_TYPE = \""
        )
    else:
        body = (
            "CONCEPT_GROUP = \""
        )
    suffix = "```python\n" + body
    return suffix, body
