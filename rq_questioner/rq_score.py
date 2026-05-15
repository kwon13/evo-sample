"""
R_Q Score: Computes the Questioner's objective function.

R_Q(x') = p_θ(x') * (1 - p_θ(x')) * (1/G) * Σ_i H(y_i)

Where:
  - p_θ(x') is estimated from G rollouts as pass rate
  - H(y_i) is the mean per-token entropy of rollout response y_i

Ablation: R_Q is the product of a learnability term p(1-p) and an
uncertainty term U. configure_rq_ablation() can force either term to
1.0 so a run can isolate the other term's contribution. See
compute_rq_value() — the single source of truth for the scalar R_Q.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# R_Q term ablation
# ---------------------------------------------------------------------------
# R_Q = [p(1-p)] * [U]. Each switch, when True, replaces its term with the
# neutral element 1.0, removing that term's influence on R_Q. Both False
# (default) = standard R_Q. Set once at trainer startup via
# configure_rq_ablation(); evolution does not change them mid-run.
_ABLATE_LEARNABILITY = False   # True -> p(1-p) term forced to 1.0
_ABLATE_UNCERTAINTY = False    # True -> U (entropy) term forced to 1.0


def configure_rq_ablation(
    ablate_learnability: bool = False,
    ablate_uncertainty: bool = False,
) -> None:
    """Set R_Q term ablation. Call once at startup from the rq config."""
    global _ABLATE_LEARNABILITY, _ABLATE_UNCERTAINTY
    _ABLATE_LEARNABILITY = bool(ablate_learnability)
    _ABLATE_UNCERTAINTY = bool(ablate_uncertainty)


def rq_ablation_state() -> dict:
    """Current ablation switches — log this into run metrics so each
    run records which R_Q terms were active."""
    return {
        "ablate_learnability": _ABLATE_LEARNABILITY,
        "ablate_uncertainty": _ABLATE_UNCERTAINTY,
    }


def rq_terms(p_hat: float, uncertainty: float) -> tuple[float, float]:
    """Return (learnability_term, uncertainty_term) with ablation applied.

    learnability_term = p(1-p), or 1.0 if ablated.
    uncertainty_term  = U,      or 1.0 if ablated.
    """
    learn = 1.0 if _ABLATE_LEARNABILITY else p_hat * (1.0 - p_hat)
    unc = 1.0 if _ABLATE_UNCERTAINTY else float(uncertainty)
    return learn, unc


def compute_rq_value(p_hat: float, uncertainty: float) -> float:
    """Single source of truth for the scalar R_Q = p(1-p) * U, with the
    ablation switches applied. Every R_Q computation site routes here."""
    learn, unc = rq_terms(p_hat, uncertainty)
    return learn * unc


@dataclass
class RQResult:
    """Result of R_Q evaluation for a problem."""
    rq_score: float       # R_Q = p(1-p) * H_bar
    p_hat: float          # Estimated pass rate
    p_variance: float     # p(1-p) = learnability
    h_bar: float          # Mean response entropy across G rollouts
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
        h_bar: Mean response entropy across the sampled rollouts
    
    Returns:
        RQResult with the computed R_Q score
    """
    p_var = p_hat * (1.0 - p_hat)        # raw learnability — reporting only
    rq = compute_rq_value(p_hat, h_bar)  # ablation switches applied here

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
        h_bar: Mean response entropy across the sampled rollouts
    
    Returns:
        Complete RQResult
    """
    p_hat = estimate_pass_rate(correct_flags)
    p_var = p_hat * (1.0 - p_hat)        # raw learnability — reporting only
    rq = compute_rq_value(p_hat, h_bar)  # ablation switches applied here

    return RQResult(
        rq_score=rq,
        p_hat=p_hat,
        p_variance=p_var,
        h_bar=h_bar,
        num_rollouts=len(correct_flags),
        num_correct=sum(correct_flags),
    )
