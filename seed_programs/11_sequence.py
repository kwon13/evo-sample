import random

def generate(seed):
    """Geometric-arithmetic mixed: sum of geometric series then arithmetic operation."""
    rng = random.Random(seed)
    # Geometric series: a, ar, ar^2, ..., ar^(n-1)
    a = rng.randint(2, 5)
    r = rng.choice([2, 3])
    n = rng.randint(4, 8)

    # S = a * (r^n - 1) / (r - 1)
    geo_sum = a * (r**n - 1) // (r - 1)

    # 등차 수열: 첫항 geo_sum, 공차 d, m항까지의 합
    d = rng.randint(5, 20)
    m = rng.randint(5, 10)
    arith_sum = m * (2 * geo_sum + (m - 1) * d) // 2

    answer = arith_sum

    problem = (
        f"A geometric sequence has first term {a} and common ratio {r}. "
        f"Let S be the sum of the first {n} terms. "
        f"Now consider an arithmetic sequence with first term S and "
        f"common difference {d}. What is the sum of the first {m} terms "
        f"of this arithmetic sequence?"
    )
    return problem, str(answer)
