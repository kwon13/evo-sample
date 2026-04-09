import random
import math


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["sync_time", "tile_floor"])

    if problem_type == "sync_time":
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

    else:
        room_l = rng.randint(8, 20) * 6
        room_w = rng.randint(6, 15) * 6
        tile = math.gcd(room_l, room_w)
        num_tiles = (room_l // tile) * (room_w // tile)
        answer = num_tiles
        problem = (
            f"A rectangular floor measures {room_l} cm by {room_w} cm. "
            f"It is to be tiled with the largest possible square tiles. "
            f"What is the minimum number of square tiles needed?"
        )

    return problem, str(answer)
