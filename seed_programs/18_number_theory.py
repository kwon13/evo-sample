"""
Advanced number theory: prime factorization, Euler's totient, divisor counting.
MATH benchmark category: Number Theory (Hendrycks et al., NeurIPS 2021)
"""
import random
import math


def prime_factors(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def euler_totient(n):
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def num_divisors(factors):
    result = 1
    for exp in factors.values():
        result *= (exp + 1)
    return result


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["prime_factorization", "num_divisors", "totient"])

    if problem_type == "prime_factorization":
        # Pick two small primes and multiply
        primes = [2, 3, 5, 7, 11, 13]
        p1 = rng.choice(primes)
        p2 = rng.choice(primes)
        e1 = rng.randint(1, 3)
        e2 = rng.randint(1, 3)
        n = (p1 ** e1) * (p2 ** e2)
        factors = prime_factors(n)
        parts = " × ".join(
            f"{p}^{e}" if e > 1 else str(p)
            for p, e in sorted(factors.items())
        )
        problem = (
            f"Find the prime factorization of {n}. "
            f"Express your answer in the form p₁^a × p₂^b × ..."
        )
        answer = parts

    elif problem_type == "num_divisors":
        primes = [2, 3, 5, 7]
        p1 = rng.choice(primes)
        p2 = rng.choice([p for p in primes if p != p1])
        e1 = rng.randint(1, 4)
        e2 = rng.randint(1, 3)
        n = (p1 ** e1) * (p2 ** e2)
        factors = prime_factors(n)
        nd = num_divisors(factors)
        problem = (
            f"How many positive divisors does {n} have?"
        )
        answer = str(nd)

    else:  # totient
        # Use a number whose totient is easy to compute
        primes = [7, 11, 13, 17, 19, 23]
        p = rng.choice(primes)
        q = rng.choice([x for x in primes if x != p])
        n = p * q
        phi = (p - 1) * (q - 1)
        problem = (
            f"Compute Euler's totient function φ({n}), "
            f"which counts the positive integers up to {n} that are coprime to {n}."
        )
        answer = str(phi)

    return problem, answer
