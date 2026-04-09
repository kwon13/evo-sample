import random


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["sigma_phi", "prime_factorization"])

    if problem_type == "sigma_phi":
        primes = [2, 3, 5, 7, 11, 13]
        p = rng.choice(primes[:4])
        q = rng.choice([x for x in primes if x != p])
        a = rng.randint(1, 3)
        b = rng.randint(1, 2)
        n = (p ** a) * (q ** b)
        sigma = ((p ** (a + 1) - 1) // (p - 1)) * ((q ** (b + 1) - 1) // (q - 1))
        phi = n * (p - 1) * (q - 1) // (p * q)
        answer = sigma - phi
        problem = (
            f"Let n = {n}. Compute sigma(n) - phi(n), where sigma(n) is the sum "
            f"of all positive divisors of n, and phi(n) is Euler's totient function."
        )

    else:
        primes = [2, 3, 5, 7, 11]
        k = rng.randint(2, 4)
        chosen = rng.sample(primes, k)
        exponents = [rng.randint(1, 3) for _ in range(k)]
        n = 1
        for p, e in zip(chosen, exponents):
            n *= p ** e
        num_divisors = 1
        for e in exponents:
            num_divisors *= (e + 1)
        answer = num_divisors
        parts = [f"{p}^{e}" if e > 1 else str(p) for p, e in zip(sorted(chosen), exponents)]
        problem = (
            f"The prime factorization of a number n is {' * '.join(parts)}. "
            f"How many positive divisors does n have?"
        )

    return problem, str(answer)
