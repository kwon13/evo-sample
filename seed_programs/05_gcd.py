import random
import math

def generate(seed):
    """GCD + LCM combined: find meeting time of periodic events."""
    rng = random.Random(seed)
    # 세 버스 노선이 각각 a, b, c분마다 출발
    # 동시 출발 후 다시 동시에 출발하는 시간 = LCM(a, b, c)
    a = rng.choice([6, 8, 10, 12, 15])
    b = rng.choice([9, 12, 14, 16, 18, 20])
    c = rng.choice([10, 15, 20, 21, 24, 25, 30])

    lcm_ab = a * b // math.gcd(a, b)
    lcm_abc = lcm_ab * c // math.gcd(lcm_ab, c)

    # 몇 시간 동안 동시 출발 횟수
    hours = rng.choice([6, 8, 10, 12])
    total_minutes = hours * 60
    # 0분을 포함하므로: floor(total_minutes / lcm_abc) + 1 - 1 (처음 제외 가능)
    # "처음 이후" 동시 출발 횟수
    times = total_minutes // lcm_abc

    answer = times

    problem = (
        f"Three bus routes depart from the same station. Route A departs every "
        f"{a} minutes, Route B every {b} minutes, and Route C every {c} minutes. "
        f"If all three routes depart simultaneously at 6:00 AM, how many more times "
        f"will all three routes depart at the same time within the next {hours} hours?"
    )
    return problem, str(answer)
