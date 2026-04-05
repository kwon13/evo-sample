import random

def generate(seed):
    rng = random.Random(seed)
    speed_a = rng.randint(40, 80)
    speed_b = rng.randint(50, 100)
    if speed_a == speed_b:
        speed_b += 10
    time_hours = rng.randint(2, 6)
    distance = speed_a * time_hours
    time_b = distance / speed_b

    if time_b != int(time_b):
        speed_a = 60
        time_hours = rng.choice([2, 3, 4, 5])
        distance = speed_a * time_hours
        speed_b = rng.choice([d for d in [40, 50, 60, 80, 120] if distance % d == 0 and d != speed_a])
        time_b = distance // speed_b

    answer = int(time_b)

    problem = (
        f"A car travels from City A to City B at {speed_a} km/h and "
        f"takes {time_hours} hours. If a bus travels the same route at "
        f"{speed_b} km/h, how many hours will the bus take?"
    )
    return problem, str(answer)
