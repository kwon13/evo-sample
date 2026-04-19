import random


def _stirling_second(n, k):
    table = [[0] * (k + 1) for _ in range(n + 1)]
    table[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            table[i][j] = j * table[i - 1][j] + table[i - 1][j - 1]
    return table[n][k]


def generate(seed):
    rng = random.Random(seed)
    n = rng.randint(8, 12)
    k = rng.randint(3, min(5, n - 3))

    if rng.random() < 0.5:
        answer = _stirling_second(n, k)
        problem = (
            f"How many ways are there to partition {n} distinct students into "
            f"{k} nonempty unlabeled study groups?"
        )
    else:
        answer = _stirling_second(n, k)
        for x in range(2, k + 1):
            answer *= x
        problem = (
            f"{n} distinct tasks must be assigned onto {k} labeled servers, "
            f"with every server receiving at least one task. How many assignments "
            f"are possible?"
        )

    return problem, str(answer)
