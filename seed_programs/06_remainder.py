import random

def generate(seed):
    """Chinese Remainder Theorem style: find N given multiple remainder conditions."""
    rng = random.Random(seed)
    # N ≡ r1 (mod m1), N ≡ r2 (mod m2)
    # m1, m2 coprime → unique solution mod m1*m2
    m1 = rng.choice([3, 5, 7])
    m2 = rng.choice([4, 7, 9, 11])
    while m2 == m1 or __import__('math').gcd(m1, m2) != 1:
        m2 = rng.choice([4, 7, 9, 11, 13])

    r1 = rng.randint(1, m1 - 1)
    r2 = rng.randint(1, m2 - 1)

    # CRT로 최소 양수 해 구하기
    product = m1 * m2
    for n in range(1, product + 1):
        if n % m1 == r1 and n % m2 == r2:
            answer = n
            break

    # 범위 내에서 몇 개인지
    upper = rng.randint(200, 500)
    count = 0
    val = answer
    while val <= upper:
        count += 1
        val += product

    problem = (
        f"A positive integer N leaves a remainder of {r1} when divided by {m1}, "
        f"and a remainder of {r2} when divided by {m2}. "
        f"How many such integers N exist in the range 1 to {upper}, inclusive?"
    )
    return problem, str(count)
