import random


def generate(seed):
    rng = random.Random(seed)

    s = rng.choice([6, 9, 12, 15, 18])
    max_abc = (s // 3) ** 3
    answer = str(max_abc)
    problem = (
        f"Let a, b, c be positive real numbers with a + b + c = {s}. "
        f"What is the maximum value of the product abc?"
    )

    return problem, str(answer)
