import random


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["linear_recurrence", "sum_of_squares"])

    if problem_type == "linear_recurrence":
        p = rng.choice([2, 3])
        q = rng.randint(1, 5)
        c = rng.randint(1, 5)
        n = rng.randint(6, 10)
        val = c
        for _ in range(n):
            val = p * val + q
        answer = val
        problem = (
            f"A sequence is defined by a(0) = {c} and "
            f"a(n) = {p} * a(n-1) + {q} for n >= 1. Find a({n})."
        )

    else:
        a = rng.randint(10, 25)
        b = rng.randint(a + 5, a + 20)
        s_a = a * (a + 1) * (2 * a + 1) // 6
        s_b = b * (b + 1) * (2 * b + 1) // 6
        answer = s_b - s_a
        problem = (
            f"Compute the sum of squares from {a + 1} to {b}: "
            f"{a + 1}^2 + {a + 2}^2 + ... + {b}^2."
        )

    return problem, str(answer)
