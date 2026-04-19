import random


def _bounded_count(caps, total):
    dp = [0] * (total + 1)
    dp[0] = 1
    for cap in caps:
        nxt = [0] * (total + 1)
        for current in range(total + 1):
            if dp[current] == 0:
                continue
            for add in range(cap + 1):
                if current + add <= total:
                    nxt[current + add] += dp[current]
        dp = nxt
    return dp[total]


def generate(seed):
    rng = random.Random(seed)
    variables = rng.randint(5, 7)
    caps = [rng.randint(2, 7) for _ in range(variables)]
    total = rng.randint(max(4, sum(caps) // 3), max(5, 2 * sum(caps) // 3))
    answer = _bounded_count(caps, total)

    terms = " ".join(f"(1+x+...+x^{cap})" for cap in caps)
    bounds_str = ", ".join(
        f"x_{i + 1} <= {cap}" for i, cap in enumerate(caps)
    )
    problem = (
        f"Find the coefficient of x^{total} in the generating function "
        f"{terms}. Equivalently, count the nonnegative integer solutions to "
        f"x_1+x_2+...+x_{variables}={total} with {bounds_str}."
    )
    return problem, str(answer)
