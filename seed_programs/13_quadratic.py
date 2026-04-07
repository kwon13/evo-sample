"""
Quadratic equations & Vieta's formulas.
MATH benchmark category: Intermediate Algebra (Hendrycks et al., NeurIPS 2021)
"""
import random


def generate(seed):
    """Parameter quadratic: given sum/product constraints, find expression."""
    rng = random.Random(seed)
    r1 = rng.randint(-8, 8)
    r2 = rng.randint(-8, 8)
    while r2 == r1 or r1 == 0 or r2 == 0:
        r2 = rng.randint(-8, 8)

    s = r1 + r2       # sum of roots
    p = r1 * r2       # product of roots
    b = -s
    c = p

    # Ask for r1^2 + r2^2 = s^2 - 2p
    sq_sum = s**2 - 2*p

    problem_type = rng.choice(["sq_sum", "reciprocal_sum", "abs_diff"])

    if problem_type == "sq_sum":
        answer = sq_sum
        q = f"the sum of the squares of its roots (r₁² + r₂²)"
    elif problem_type == "reciprocal_sum":
        # 1/r1 + 1/r2 = (r1+r2)/(r1*r2) = s/p
        from fractions import Fraction
        frac = Fraction(s, p)
        answer = str(frac)
        q = f"the sum of the reciprocals of its roots (1/r₁ + 1/r₂) as a fraction"
    else:
        # |r1 - r2| = sqrt((r1+r2)^2 - 4*r1*r2) = sqrt(s^2 - 4p)
        disc = s**2 - 4*p
        import math
        abs_diff = round(math.sqrt(disc), 4)
        answer = str(abs_diff)
        q = f"the absolute difference of its roots |r₁ - r₂|, rounded to 4 decimal places"

    if problem_type != "sq_sum":
        problem = (
            f"The quadratic equation x² + {b}x + {c} = 0 has two real roots. "
            f"Find {q}."
        )
    else:
        problem = (
            f"The quadratic equation x² + {b}x + {c} = 0 has two real roots. "
            f"Find {q}."
        )
        answer = str(answer)

    return problem, str(answer)
