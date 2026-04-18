"""
R_Q Score: Computes the Questioner's objective function.

R_Q(x') = p_θ(x') * (1 - p_θ(x')) * (1/T) * Σ H_t(x')

Where:
  - p_θ(x') is estimated from G rollouts as pass rate
  - H_t(x') is per-token entropy from Solver's forward pass
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RQResult:
    """Result of R_Q evaluation for a problem."""
    rq_score: float       # R_Q = p(1-p) * H_bar
    p_hat: float          # Estimated pass rate
    p_variance: float     # p(1-p) = learnability
    h_bar: float          # Mean entropy = gradient richness
    num_rollouts: int     # G
    num_correct: int      # Number of correct rollouts


def compute_rq(
    p_hat: float,
    h_bar: float,
) -> RQResult:
    """
    Compute R_Q from estimated pass rate and mean entropy.
    
    Args:
        p_hat: Estimated pass rate from G rollouts
        h_bar: Mean token-level entropy (1/T * Σ H_t)
    
    Returns:
        RQResult with the computed R_Q score
    """
    p_var = p_hat * (1.0 - p_hat)
    rq = p_var * h_bar

    return RQResult(
        rq_score=rq,
        p_hat=p_hat,
        p_variance=p_var,
        h_bar=h_bar,
        num_rollouts=0,
        num_correct=0,
    )


def estimate_pass_rate(
    correct_flags: list[bool],
) -> float:
    """
    Estimate p(x') from G rollout results.
    
    Args:
        correct_flags: List of bool indicating correctness of each rollout
    
    Returns:
        Estimated pass rate p_hat
    """
    if not correct_flags:
        return 0.0
    return sum(correct_flags) / len(correct_flags)


def h_prefilter(h_bar: float, threshold: float = 0.1) -> bool:
    """
    H pre-filter: If H is too low, skip rollouts.
    
    Based on DPI inequality: p(1-p) ≤ H/2
    If H is very low, p(1-p) must also be low, so R_Q will be low.
    
    Args:
        h_bar: Mean entropy
        threshold: Minimum H to pass filter
    
    Returns:
        True if problem passes the filter (H is high enough)
    """
    return h_bar >= threshold


def p_hat_filter(p_hat: float) -> bool:
    """p=0 또는 p=1이면 R_Q=0이므로 필터링."""
    return 0.0 < p_hat < 1.0


def compute_rq_full(
    correct_flags: list[bool],
    h_bar: float,
) -> RQResult:
    """
    Full R_Q computation from rollout results and entropy.
    
    Args:
        correct_flags: List of bool from G rollouts
        h_bar: Mean entropy from forward pass
    
    Returns:
        Complete RQResult
    """
    p_hat = estimate_pass_rate(correct_flags)
    p_var = p_hat * (1.0 - p_hat)
    rq = p_var * h_bar

    return RQResult(
        rq_score=rq,
        p_hat=p_hat,
        p_variance=p_var,
        h_bar=h_bar,
        num_rollouts=len(correct_flags),
        num_correct=sum(correct_flags),
    )
