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


def _sympy_equal(a: str, b: str, tol: float = 1e-4) -> bool | None:
    """
    SymPy로 두 수학 표현식이 동치인지 판별.

    지원: 분수(31/45), 제곱근(sqrt(2)), π/e, 거듭제곱(2^10),
          사칙연산((1+2)/3), LaTeX(\\frac{3}{4}), 소수(3.14) 등.

    Returns: True/False or None (파싱 실패).
    """
    from sympy import sympify, N, simplify
    from sympy.parsing.latex import parse_latex

    def _parse(s: str):
        s = s.strip().replace("^", "**")
        if "\\" in s:
            try:
                return parse_latex(s)
            except Exception:
                pass
        try:
            return sympify(s)
        except Exception:
            pass
        return None

    expr_a, expr_b = _parse(a), _parse(b)
    if expr_a is None or expr_b is None:
        return None
    try:
        if simplify(expr_a - expr_b) == 0:
            return True
    except Exception:
        pass
    try:
        return abs(float(N(expr_a)) - float(N(expr_b))) < tol
    except Exception:
        pass
    return None


def _match(pred: str, gt: str) -> bool:
    # 1. 정규화 후 문자열 비교
    if _normalize(pred) == _normalize(gt):
        return True
    # 2. SymPy 기반 수학적 동치 판별
    result = _sympy_equal(pred, gt)
    if result is not None:
        return result
    return False


def _score_single(solution_str: str, ground_truth: str) -> float:
    """단일 응답 채점: 1.0 (정답), 0.1 (boxed 있지만 오답), 0.0 (boxed 없음)"""
    predicted = _extract_boxed(solution_str)
    if predicted is None:
        return 0.0
    if _match(predicted, ground_truth):
        return 1.0
    return 0.1


def compute_score(response_str_list: list[str], ground_truth_list: list[str]) -> list[dict]:
    """
    verl 0.3.1 BatchRewardFunction 형식.

    Args:
        response_str_list: 모델 응답 텍스트 리스트
        ground_truth_list: 정답 리스트

    Returns:
        list of {"overall": float, "accuracy": float, "format": float}
    """
    results = []
    for resp, gt in zip(response_str_list, ground_truth_list):
        score = _score_single(resp, gt)
        results.append({
            "overall": score,
            "accuracy": 1.0 if score == 1.0 else 0.0,
            "format": 1.0 if score > 0.0 else 0.0,
        })
    return results
