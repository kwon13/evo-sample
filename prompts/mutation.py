from rq_questioner.map_elites import MAPElitesGrid
from rq_questioner.program import ProblemProgram


SINGLE_ANSWER_RULE = (
    "# IMPORTANT RULES:\n"
    "# 1. Function name MUST be `generate(seed)`\n"
    "# 2. MUST return (problem_text: str, answer: str)\n"
    "# 3. answer MUST be a SINGLE number or simple value (e.g. '42', '3.14', '7/3')\n"
    "#    NOT ranges, NOT multiple values, NOT inequalities\n"
    "# 4. Compute answer FIRST, then build problem from it\n"
    "# 5. Use only standard library + math module\n"
)

SCORE_FEEDBACK = (
    "# === PERFORMANCE OF CURRENT PROGRAM ===\n"
    "# pass_rate (p_hat) = {p_hat:.2f}  (ideal: 0.50, solver gets it right half the time)\n"
    "# entropy (H)       = {h_score:.2f}  (ideal: > 2.0, higher = solver is more uncertain)\n"
    "# R_Q score          = {rq_score:.4f}  (higher is better, max when p_hat~0.5 and H is high)\n"
    "#\n"
    "# DIAGNOSIS: {diagnosis}\n"
    "# ACTION: {action}\n"
)

MUTATE_DEPTH = (
    "# Python function that generates math word problems.\n"
    "# Rewrite to generate HARDER, competition-level problems (AMC/AIME).\n"
    "#\n"
    "{few_shot}"
    "{score_feedback}"
    "{exec_feedback}"
    "#\n"
    "# Requirements:\n"
    "#   - At least 3-5 reasoning steps to solve\n"
    "#   - Combine multiple math concepts (e.g. geometry + algebra)\n"
    "#   - Use larger numbers, fractions, or nested computations\n"
    "#   - Require intermediate results before final answer\n\n"
    "# Original program:\n"
    "```python\n{code}\n```\n\n"
    + SINGLE_ANSWER_RULE +
    "# Improved version (target: p_hat ~ 0.5, H > 2.0):\n"
    "```python\n"
    "import random\n"
)

MUTATE_BREADTH = (
    "# Python function that generates math word problems.\n"
    "# Rewrite to generate a COMPLETELY DIFFERENT type of hard math problem.\n"
    "#\n"
    "{few_shot}"
    "{score_feedback}"
    "{exec_feedback}"
    "#\n"
    "# Choose a different branch of mathematics:\n"
    "#   - If original is geometry -> try number theory or combinatorics\n"
    "#   - If original is algebra -> try probability or modular arithmetic\n"
    "#   - Must require multi-step reasoning (3+ steps)\n"
    "#   - Must produce a single numerical answer\n\n"
    "# Original program:\n"
    "```python\n{code}\n```\n\n"
    + SINGLE_ANSWER_RULE +
    "# Completely different topic (target: p_hat ~ 0.5, H > 2.0):\n"
    "```python\n"
    "import random\n"
)

MUTATE_CROSSOVER = (
    "# Two Python functions that generate different types of math problems.\n"
    "# Combine ideas from BOTH programs to create a NEW hybrid problem generator\n"
    "# that merges the mathematical concepts from both parents.\n"
    "#\n"
    "{few_shot}"
    "# Parent A (p_hat={p_hat_a:.2f}, H={h_a:.2f}):\n"
    "```python\n{code_a}\n```\n\n"
    "# Parent B (p_hat={p_hat_b:.2f}, H={h_b:.2f}):\n"
    "```python\n{code_b}\n```\n\n"
    "# Create a NEW function that combines concepts from both parents.\n"
    "# Example: if A is geometry and B is probability,\n"
    "#   create geometric probability problems.\n"
    + SINGLE_ANSWER_RULE +
    "# Hybrid version combining both concepts (target: p_hat ~ 0.5, H > 2.0):\n"
    "```python\n"
    "import random\n"
)


def score_diagnosis(p_hat: float, h_score: float) -> tuple[str, str]:
    if p_hat > 0.7:
        diag = f"TOO EASY (solver gets {p_hat:.0%} correct)"
        action = "Make problems significantly harder: more steps, combined concepts, larger numbers"
    elif p_hat < 0.2:
        diag = f"TOO HARD (solver gets only {p_hat:.0%} correct)"
        action = "Make problems slightly easier: clearer wording, fewer steps, smaller numbers"
    else:
        diag = f"Good difficulty (p_hat={p_hat:.2f})"
        action = "Keep similar difficulty but increase problem diversity and reasoning depth"

    if h_score < 0.5:
        diag += f"; LOW ENTROPY (H={h_score:.2f}, solver is too confident)"
        action += "; add ambiguity or multi-step reasoning to increase solver uncertainty"

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


def build_few_shot_examples(grid: MAPElitesGrid, top_k: int = 3) -> str:
    champions = grid.get_all_champions()
    if not champions:
        return ""
    ranked = sorted(champions, key=lambda c: -(c.rq_score or 0))[:top_k]
    parts = ["# === HIGH-QUALITY EXAMPLES (for reference) ==="]
    for i, champ in enumerate(ranked, 1):
        rq = getattr(champ, "rq_score", 0)
        p = getattr(champ, "p_hat", 0)
        h = getattr(champ, "h_score", 0)
        parts.append(
            f"#\n# Example {i} (RQ={rq:.3f}, p_hat={p:.2f}, H={h:.2f}):\n"
            f"```python\n{champ.source_code}\n```"
        )
    parts.append("# === END EXAMPLES ===\n")
    return "\n".join(parts)


def build_execution_feedback(parent: ProblemProgram) -> str:
    inst = parent.execute(seed=0, timeout=5.0)
    if inst is None:
        return ""
    p_hat = getattr(parent, "p_hat", 0.5)
    h_score = getattr(parent, "h_score", 1.0)
    difficulty = (
        "TOO EASY" if p_hat > 0.7 else
        "TOO HARD" if p_hat < 0.2 else
        "GOOD"
    )
    return (
        f"# === EXECUTION RESULT ===\n"
        f"# Problem: {inst.problem[:100]}...\n"
        f"# Answer: {inst.answer}\n"
        f"# Solver pass rate: {p_hat:.0%} ({difficulty})\n"
        f"# Solver entropy: {h_score:.2f}\n"
        f"# === END ===\n"
    )
