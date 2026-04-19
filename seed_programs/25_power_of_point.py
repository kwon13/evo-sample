import random
from fractions import Fraction


def generate(seed):
    rng = random.Random(seed)

    if rng.random() < 0.5:
        external_1 = rng.randint(3, 14)
        internal_1 = rng.randint(8, 30)
        whole_1 = external_1 + internal_1
        product = external_1 * whole_1
        # Require PC != PA (avoid trivial symmetry where CD == AB is read off),
        # PC < PD so CD > 0, and PC divides product so PD is an integer ≥ 2.
        valid = [
            d for d in range(2, whole_1)
            if product % d == 0 and product // d > d and d != external_1
        ]
        # Fallback: the divisor d = external_1 is always in range and satisfies
        # product // d = whole_1 > external_1, so CD = internal_1 > 0. The
        # resulting problem has PC = PA numerically but is still well-posed.
        if not valid:
            valid = [external_1]
        external_2 = rng.choice(valid)
        whole_2 = Fraction(product, external_2)
        internal_2 = whole_2 - external_2
        answer = (
            str(internal_2.numerator)
            if internal_2.denominator == 1
            else f"{internal_2.numerator}/{internal_2.denominator}"
        )
        problem = (
            f"From an external point P, two secants intersect a circle at A,B "
            f"and C,D respectively, with A and C nearer to P. Given PA={external_1}, "
            f"AB={internal_1}, and PC={external_2}, find CD."
        )
    else:
        u = rng.randint(2, 6)
        v = rng.randint(u + 1, u + 6)
        external = u * u
        whole = v * v
        internal = whole - external
        tangent = u * v
        answer = tangent
        problem = (
            f"From an external point P, a tangent PT and a secant PAB are drawn "
            f"to a circle, with A nearer to P. If PA={external} and AB={internal}, "
            f"find the tangent length PT."
        )

    return problem, str(answer)
