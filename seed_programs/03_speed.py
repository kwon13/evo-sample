import random

def generate(seed):
    """Multi-leg trip with rest stop and average speed calculation."""
    rng = random.Random(seed)
    # 3구간 여행: d1 at s1, rest t_rest hours, d2 at s2, d3 at s3
    s1 = rng.choice([40, 50, 60, 80])
    t1 = rng.randint(2, 4)
    d1 = s1 * t1

    t_rest = rng.randint(1, 2)

    s2 = rng.choice([30, 40, 50, 60])
    t2 = rng.randint(1, 3)
    d2 = s2 * t2

    s3 = rng.choice([60, 80, 100, 120])
    t3 = rng.randint(1, 3)
    d3 = s3 * t3

    total_dist = d1 + d2 + d3
    total_time = t1 + t_rest + t2 + t3  # 휴식 포함
    avg_speed = total_dist / total_time

    answer = round(avg_speed, 2)
    if answer == int(answer):
        answer = int(answer)

    problem = (
        f"A driver travels {d1} km at {s1} km/h, then rests for {t_rest} hour(s), "
        f"then travels {d2} km at {s2} km/h, and finally travels {d3} km at {s3} km/h. "
        f"What is the average speed for the entire trip, including rest time? "
        f"Round to 2 decimal places if needed."
    )
    return problem, str(answer)
