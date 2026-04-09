import random
import math


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["two_mod", "three_mod"])

    if problem_type == "two_mod":
        m1 = rng.choice([3, 5, 7])
        m2 = rng.choice([4, 7, 9, 11])
        while m2 == m1 or math.gcd(m1, m2) != 1:
            m2 = rng.choice([4, 7, 9, 11, 13])
        r1 = rng.randint(1, m1 - 1)
        r2 = rng.randint(1, m2 - 1)
        product = m1 * m2
        base = next(n for n in range(1, product + 1) if n % m1 == r1 and n % m2 == r2)
        upper = rng.randint(200, 500)
        count = 0
        val = base
        while val <= upper:
            count += 1
            val += product
        answer = count
        problem = (
            f"A positive integer N leaves a remainder of {r1} when divided by {m1}, "
            f"and a remainder of {r2} when divided by {m2}. "
            f"How many such integers N exist in the range 1 to {upper}, inclusive?"
        )

    else:
        m1 = rng.choice([3, 4, 5])
        m2 = rng.choice([7, 11])
        r1 = rng.randint(1, m1 - 1)
        r2 = rng.randint(1, m2 - 1)
        product = m1 * m2
        base = next(n for n in range(1, product + 1) if n % m1 == r1 and n % m2 == r2)
        answer = base
        problem = (
            f"Find the smallest positive integer that leaves a remainder of "
            f"{r1} when divided by {m1} and a remainder of {r2} when divided by {m2}."
        )

    return problem, str(answer)
