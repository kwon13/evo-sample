import random


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["geo_then_arith", "partial_sum_diff"])

    if problem_type == "geo_then_arith":
        a = rng.randint(2, 5)
        r = rng.choice([2, 3])
        n = rng.randint(4, 8)
        geo_sum = a * (r ** n - 1) // (r - 1)
        d = rng.randint(5, 20)
        m = rng.randint(5, 10)
        arith_sum = m * (2 * geo_sum + (m - 1) * d) // 2
        answer = arith_sum
        problem = (
            f"A geometric sequence has first term {a} and common ratio {r}. "
            f"Let S be the sum of the first {n} terms. "
            f"An arithmetic sequence has first term S and common difference {d}. "
            f"What is the sum of the first {m} terms of this arithmetic sequence?"
        )

    else:
        a = rng.randint(1, 5)
        d = rng.randint(2, 8)
        n1 = rng.randint(10, 20)
        n2 = rng.randint(n1 + 5, n1 + 20)
        s1 = n1 * (2 * a + (n1 - 1) * d) // 2
        s2 = n2 * (2 * a + (n2 - 1) * d) // 2
        answer = s2 - s1
        problem = (
            f"An arithmetic sequence has first term {a} and common difference {d}. "
            f"Let S(n) be the sum of the first n terms. "
            f"Compute S({n2}) - S({n1})."
        )

    return problem, str(answer)
