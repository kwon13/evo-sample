import math
import random

CONCEPT_TYPE = "combinatorics.derangement_fixed_points"
CONCEPT_GROUP = "combinatorics"


def _derangement(n):
    if n == 0:
        return 1
    if n == 1:
        return 0
    a, b = 1, 0
    for k in range(2, n + 1):
        a, b = b, (k - 1) * (a + b)
    return b


def generate(seed):
    rng = random.Random(seed)
    n = rng.randint(7, 11)
    fixed = rng.randint(1, 3)
    remaining = n - fixed

    answer = math.comb(n, fixed) * _derangement(remaining)
    problem = (
        f"{n} people each write their name on a card. The cards are shuffled "
        f"and redistributed. In how many redistributions do exactly {fixed} "
        f"people receive their own card?"
    )

    return problem, str(answer)
