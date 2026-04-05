import random
import math

def generate(seed):
    rng = random.Random(seed)
    gcd_val = rng.choice([3, 4, 5, 6, 7, 8, 9, 10, 12])
    m1 = rng.randint(2, 8)
    m2 = rng.randint(2, 8)
    while math.gcd(m1, m2) != 1:
        m2 = rng.randint(2, 8)
    a = gcd_val * m1
    b = gcd_val * m2
    answer = gcd_val

    problem = (
        f"A florist has {a} roses and {b} tulips. She wants to make "
        f"identical bouquets using all the flowers, with each bouquet "
        f"having the same number of roses and the same number of tulips. "
        f"What is the maximum number of bouquets she can make?"
    )
    return problem, str(answer)
