import random
import math

CONCEPT_GROUP = "geometry"
CONCEPT_TYPE = "geometry.line_intersection_distance"
def generate(seed):
    rng = random.Random(seed)

    a1 = rng.randint(1, 5)
    b1 = rng.randint(-5, 5)
    c1 = rng.randint(-20, 20)
    a2 = rng.randint(-5, 5)
    b2 = rng.randint(1, 5)
    while a1 * b2 == a2 * b1:
        b2 = rng.randint(1, 5)
    c2 = rng.randint(-20, 20)
    det = a1 * b2 - a2 * b1
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    dist_origin = round(math.sqrt(x ** 2 + y ** 2), 4)
    answer = dist_origin
    problem = (
        f"Two lines are defined by {a1}x + {b1}y = {c1} and "
        f"{a2}x + {b2}y = {c2}. Find the distance from their "
        f"intersection point to the origin. Round to 4 decimal places."
    )

    return problem, str(answer)
