import random
import math


def generate(seed):
    rng = random.Random(seed)

    n_men = rng.randint(5, 9)
    n_women = rng.randint(4, 8)
    committee = rng.randint(3, 5)
    total = n_men + n_women
    all_ways = math.comb(total, committee)
    all_men = math.comb(n_men, committee) if n_men >= committee else 0
    all_women = math.comb(n_women, committee) if n_women >= committee else 0
    answer = all_ways - all_men - all_women
    problem = (
        f"A club has {n_men} men and {n_women} women. A committee of "
        f"{committee} people is to be formed with at least 1 man and "
        f"at least 1 woman. How many ways can this committee be formed?"
    )

    return problem, str(answer)
