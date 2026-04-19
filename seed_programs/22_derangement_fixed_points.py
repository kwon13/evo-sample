import math
import random


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

    if rng.random() < 0.5:
        answer = math.comb(n, fixed) * _derangement(remaining)
        problem = (
            f"{n} people each write their name on a card. The cards are shuffled "
            f"and redistributed. In how many redistributions do exactly {fixed} "
            f"people receive their own card?"
        )
    else:
        distinguished = rng.randint(2, 4)
        answer = _derangement(n - distinguished)
        problem = (
            f"{n} people each write their name on a card. After redistribution, "
            f"the first {distinguished} specified people must receive their own "
            f"cards, while every other person must not receive their own card. "
            f"How many redistributions are possible?"
        )

    return problem, str(answer)
