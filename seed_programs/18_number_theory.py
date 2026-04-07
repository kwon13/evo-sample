"""
Advanced number theory: Euler's totient, sum of divisors, combined.
MATH benchmark category: Number Theory (Hendrycks et al., NeurIPS 2021)
"""
import random
import math


def generate(seed):
    """Compute σ(n) - φ(n) where σ = sum of divisors, φ = Euler's totient."""
    rng = random.Random(seed)

    primes = [2, 3, 5, 7, 11, 13]
    p = rng.choice(primes[:4])
    q = rng.choice([x for x in primes if x != p])
    a = rng.randint(1, 3)
    b = rng.randint(1, 2)
    n = (p ** a) * (q ** b)

    # σ(n) = (p^(a+1)-1)/(p-1) * (q^(b+1)-1)/(q-1)
    sigma = ((p**(a+1) - 1) // (p - 1)) * ((q**(b+1) - 1) // (q - 1))

    # φ(n) = n * (1-1/p) * (1-1/q)
    phi = n * (p - 1) * (q - 1) // (p * q)

    answer = sigma - phi

    problem = (
        f"Let n = {n}. Compute σ(n) − φ(n), where σ(n) is the sum of all "
        f"positive divisors of n, and φ(n) is Euler's totient function "
        f"(the count of integers from 1 to n that are coprime to n)."
    )
    return problem, str(answer)
