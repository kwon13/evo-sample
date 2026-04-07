"""
Induction & Recursion: compute n-th term or closed form of recurrence.
"""
import random


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["linear_recurrence", "nested_sum", "fibonacci_variant"])

    if problem_type == "linear_recurrence":
        # a(n) = p*a(n-1) + q, a(0) = c
        p = rng.choice([2, 3])
        q = rng.randint(1, 5)
        c = rng.randint(1, 5)
        n = rng.randint(6, 10)
        # compute iteratively
        val = c
        for _ in range(n):
            val = p * val + q
        answer = val
        problem = (
            f"A sequence is defined by a(0) = {c} and a(n) = {p}·a(n−1) + {q} for n ≥ 1. "
            f"Find a({n})."
        )

    elif problem_type == "nested_sum":
        # S(n) = 1^2 + 2^2 + ... + n^2 = n(n+1)(2n+1)/6
        # then compute S(a) - S(b)
        a = rng.randint(15, 30)
        b = rng.randint(1, a - 5)
        s_a = a * (a + 1) * (2*a + 1) // 6
        s_b = b * (b + 1) * (2*b + 1) // 6
        answer = s_a - s_b
        problem = (
            f"Compute the sum {b+1}² + {b+2}² + ... + {a}² "
            f"(i.e., the sum of squares from {b+1} to {a})."
        )

    else:  # fibonacci_variant
        # F(n) = F(n-1) + F(n-2), F(1)=a, F(2)=b
        a = rng.randint(1, 5)
        b = rng.randint(1, 5)
        n = rng.randint(8, 15)
        fib = [0] * (n + 1)
        fib[1] = a
        fib[2] = b
        for i in range(3, n + 1):
            fib[i] = fib[i-1] + fib[i-2]
        answer = fib[n]
        problem = (
            f"A sequence is defined by F(1) = {a}, F(2) = {b}, and "
            f"F(n) = F(n−1) + F(n−2) for n ≥ 3. Find F({n})."
        )

    return problem, str(answer)
