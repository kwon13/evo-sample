import random

CONCEPT_GROUP = "geometry"
CONCEPT_TYPE = "geometry.trig_area"
def generate(seed):
    rng = random.Random(seed)

    while True:
        x1 = rng.randint(2, 12)
        y1 = rng.randint(1, 10)
        x2 = rng.randint(1, 12)
        y2 = rng.randint(2, 10)

        twice_area = abs(x1 * y2 - x2 * y1)
        if twice_area > 0 and twice_area % 2 == 0:
            break

    area = twice_area // 2

    problem = (
        f"Triangle ABC has A = (0, 0), B = ({x1}, {y1}), "
        f"and C = ({x2}, {y2}). Find its area."
    )

    return problem, str(area)