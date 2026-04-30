import random
import math

CONCEPT_TYPE = "number_theory.crt_count"
CONCEPT_GROUP = "number_theory"


def generate(seed):
    rng = random.Random(seed)

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

    return problem, str(answer)
