import random
import math

CONCEPT_TYPE = "number_theory.gcd_lcm_sync"
CONCEPT_GROUP = "number_theory"


def generate(seed):
    rng = random.Random(seed)

    a = rng.choice([6, 8, 10, 12, 15])
    b = rng.choice([9, 12, 14, 16, 18, 20])
    c = rng.choice([10, 15, 20, 21, 24, 25, 30])
    lcm_ab = a * b // math.gcd(a, b)
    lcm_abc = lcm_ab * c // math.gcd(lcm_ab, c)
    hours = rng.choice([6, 8, 10, 12])
    total_minutes = hours * 60
    answer = total_minutes // lcm_abc
    problem = (
        f"Three bus routes depart from the same station. Route A departs every "
        f"{a} minutes, Route B every {b} minutes, and Route C every {c} minutes. "
        f"If all three depart simultaneously at 6:00 AM, how many more times "
        f"will all three depart together within the next {hours} hours?"
    )

    return problem, str(answer)
