import math
import random


def _pairwise_coprime(values):
    for i, a in enumerate(values):
        for b in values[i + 1:]:
            if math.gcd(a, b) != 1:
                return False
    return True


def generate(seed):
    rng = random.Random(seed)

    candidates = [5, 7, 8, 9, 11, 13, 16, 17, 19]
    while True:
        moduli = sorted(rng.sample(candidates, 4))
        if _pairwise_coprime(moduli):
            break

    modulus = math.prod(moduli)
    base = rng.randint(1, modulus)
    residues = [base % m for m in moduli]

    target_count = rng.randint(8, 24)
    slack = rng.randint(0, modulus - 1)
    upper = base + (target_count - 1) * modulus + slack
    answer = (upper - base) // modulus + 1

    clauses = [
        f"N leaves remainder {r} when divided by {m}"
        for r, m in zip(residues, moduli)
    ]
    problem = (
        "A positive integer N satisfies all of the following congruences: "
        + "; ".join(clauses)
        + f". How many such integers N are there with 1 <= N <= {upper}?"
    )
    return problem, str(answer)
