import random
import math
from fractions import Fraction


def generate(seed):
    rng = random.Random(seed)

    r1 = rng.randint(-8, -1) if rng.random() < 0.5 else rng.randint(1, 8)
    r2 = rng.randint(-8, -1) if rng.random() < 0.5 else rng.randint(1, 8)
    while r2 == r1:
        r2 = rng.randint(-8, -1) if rng.random() < 0.5 else rng.randint(1, 8)

    s = r1 + r2
    p = r1 * r2
    b = -s
    c = p

    problem_type = rng.choice(["sq_sum", "reciprocal_sum", "abs_diff"])

    if problem_type == "sq_sum":
        answer = str(s ** 2 - 2 * p)
        q = "the sum of the squares of its roots (r1^2 + r2^2)"
    elif problem_type == "reciprocal_sum":
        frac = Fraction(s, p)
        answer = str(frac)
        q = "the sum of the reciprocals of its roots (1/r1 + 1/r2) as a fraction"
    else:
        disc = s ** 2 - 4 * p
        abs_diff = round(math.sqrt(disc), 4)
        answer = str(abs_diff)
        q = "the absolute difference of its roots |r1 - r2|, rounded to 4 decimal places"

    problem = (
        f"The quadratic equation x^2 + {b}x + {c} = 0 has two real roots. "
        f"Find {q}."
    )
    return problem, str(answer)
