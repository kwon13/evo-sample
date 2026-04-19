import math
import random


def generate(seed):
    rng = random.Random(seed)

    divisors = rng.sample([6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 4)
    upper = rng.randint(650, 1800)
    threshold = rng.choice([2, 3])

    count = 0
    for n in range(1, upper + 1):
        hits = sum(1 for d in divisors if n % d == 0)
        if hits >= threshold:
            count += 1

    problem = (
        f"For the set S={{1,2,...,{upper}}}, let A_d be the subset of numbers "
        f"divisible by d. For d in {sorted(divisors)}, how many elements of S "
        f"belong to at least {threshold} of these four subsets?"
    )
    return problem, str(count)
