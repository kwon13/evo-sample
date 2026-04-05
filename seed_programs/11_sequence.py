import random

def generate(seed):
    rng = random.Random(seed)
    a1 = rng.randint(1, 20)
    d = rng.randint(1, 10)
    n = rng.randint(5, 20)
    answer = n * (2 * a1 + (n - 1) * d) // 2

    problem = (
        f"An arithmetic sequence starts at {a1} and increases by {d} "
        f"each term. What is the sum of the first {n} terms of "
        f"this sequence?"
    )
    return problem, str(answer)
