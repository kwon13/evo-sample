import random
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

    frac = Fraction(s, p)
    answer = str(frac)
    q = "the sum of the reciprocals of its roots (1/r1 + 1/r2) as a fraction"

    problem = (
        f"The quadratic equation x^2 + {b}x + {c} = 0 has two real roots. "
        f"Find {q}."
    )
    return problem, str(answer)
