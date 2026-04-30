import math
import random


def _prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def _primitive_root(p):
    factors = _prime_factors(p - 1)
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in factors):
            return g
    return 2


def generate(seed):
    """
    Count k-th roots of a given residue modulo a prime p.

    Inverse construction:
      - pick prime p and a primitive root g of (Z/pZ)*
      - pick exponent e in [1, p-2] and power k in {2,3,4,5,6,8,10,12}
      - a = g^e mod p  (so solver is given a; exponent is hidden)
      - # solutions to x^k ≡ a (mod p) = gcd(k, p-1) if gcd(k,p-1) | e else 0
    """
    rng = random.Random(seed)
    p = rng.choice([29, 31, 41, 43, 73, 89, 97, 101, 109, 113, 127, 137, 149])
    g = _primitive_root(p)

    k = rng.choice([2, 3, 4, 5, 6, 8, 10, 12])
    exponent = rng.randint(1, p - 2)
    a = pow(g, exponent, p)

    divisor = math.gcd(k, p - 1)
    answer = divisor if exponent % divisor == 0 else 0

    problem = (
        f"Modulo the prime p={p}, g={g} is a primitive root. "
        f"Let a be the least positive residue of g^{exponent} modulo p, "
        f"so a={a}. How many residue classes x modulo {p} satisfy "
        f"x^{k} ≡ a (mod {p})?"
    )
    return problem, str(answer)
