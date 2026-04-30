import random


def _legendre(a, p):
    """Euler's criterion. Returns -1, 0, or 1."""
    a = a % p
    if a == 0:
        return 0
    value = pow(a, (p - 1) // 2, p)
    return -1 if value == p - 1 else value


def generate(seed):
    """
    Compute Legendre symbol (a/p) where a is given as a PRODUCT of two small
    factors modulo p, forcing the solver to apply multiplicativity:

        (b·c / p) = (b/p) · (c/p)

    Inverse construction:
      - pick prime p in [30, 200]
      - pick two small factors b, c coprime to p
      - a = (b·c) mod p  (problem asks for Legendre symbol of this product)
      - answer = (b·c / p) computed by Euler's criterion

    This is harder than plain Euler application because the solver must either
    factor a (non-trivial in the residue class form) or use multiplicativity
    with the factored form shown in the question, then apply Euler twice.
    """
    rng = random.Random(seed)
    p = rng.choice([
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
        97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
        151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
    ])

    # Pick two distinct small factors that are coprime to p.
    factor_pool = [b for b in range(2, 30) if b % p != 0]
    b, c = rng.sample(factor_pool, 2)
    a = (b * c) % p

    answer = _legendre(a, p)

    problem = (
        f"Let p={p} be prime. Compute the Legendre symbol "
        f"( {b}·{c} / p ) modulo {p}. You may use that the Legendre symbol "
        f"is completely multiplicative: ( xy / p ) = ( x / p )( y / p ). "
        f"Give the answer as -1, 0, or 1."
    )
    return problem, str(answer)
