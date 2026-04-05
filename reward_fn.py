"""
Custom reward function for veRL GRPO training.

veRL calls this function for each generated response to compute the reward.
Signature must be: compute_score(data_source, solution_str, ground_truth, extra_info)

Reward scheme (following R-Zero / veRL conventions):
  - 1.0 if extracted answer matches ground truth
  - 0.1 if response contains \\boxed{} but answer is wrong (format reward)
  - 0.0 if no \\boxed{} found (format penalty)
"""

import re
from typing import Optional


_BOXED_RE = re.compile(
    r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL
)


def _extract_boxed(text: str) -> Optional[str]:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def _normalize(s: str) -> str:
    s = s.strip().lower()
    for o, c in [("{", "}"), ("[", "]"), ("(", ")")]:
        if s.startswith(o) and s.endswith(c):
            s = s[1:-1].strip()
    s = s.rstrip(".").replace(",", "").replace(" ", "")
    return s


def _match(pred: str, gt: str) -> bool:
    if _normalize(pred) == _normalize(gt):
        return True
    try:
        return abs(float(pred) - float(gt)) < 1e-4
    except (ValueError, TypeError):
        pass
    # Try fraction comparison
    try:
        from fractions import Fraction
        return Fraction(pred) == Fraction(gt)
    except Exception:
        pass
    return False


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> float:
    """
    Compute reward score for a single response.

    Args:
        data_source: Dataset identifier (e.g., "rq_evolved")
        solution_str: Model's full response text
        ground_truth: Expected answer string
        extra_info: Optional dict with program_id, niche, rq_score

    Returns:
        float: 1.0 (correct), 0.1 (format ok, wrong answer), 0.0 (no format)
    """
    predicted = _extract_boxed(solution_str)

    if predicted is None:
        return 0.0

    if _match(predicted, ground_truth):
        return 1.0

    return 0.1
