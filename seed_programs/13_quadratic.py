"""
Quadratic equations & Vieta's formulas.
MATH benchmark category: Intermediate Algebra (Hendrycks et al., NeurIPS 2021)
"""
import random


def generate(seed):
    rng = random.Random(seed)

    # Two integer roots → reconstruct quadratic via Vieta's formulas
    r1 = rng.randint(-10, 10)
    r2 = rng.randint(-10, 10)
    while r2 == r1:
        r2 = rng.randint(-10, 10)

    b = -(r1 + r2)   # coefficient of x  (sum of roots = -b/a)
    c = r1 * r2      # constant term      (product of roots = c/a)

    problem_type = rng.choice(["roots", "sum", "product"])

    if problem_type == "roots":
        problem = (
            f"Find all integer solutions to the equation "
            f"x² + {b}x + {c} = 0."
        )
        roots = sorted([r1, r2])
        answer = f"{roots[0]} and {roots[1]}" if roots[0] != roots[1] else str(roots[0])
    elif problem_type == "sum":
        problem = (
            f"If the two roots of x² + {b}x + {c} = 0 are integers, "
            f"what is their sum?"
        )
        answer = str(r1 + r2)
    else:
        problem = (
            f"If the two roots of x² + {b}x + {c} = 0 are integers, "
            f"what is their product?"
        )
        answer = str(r1 * r2)

    return problem, answer
