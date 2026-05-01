import random
import math

CONCEPT_TYPE = "geometry.trig_area"
CONCEPT_GROUP = "geometry"


def generate(seed):
    rng = random.Random(seed)

    a = rng.randint(5, 20)
    b = rng.randint(5, 20)
    angle_deg = rng.choice([30, 45, 60, 90, 120])
    angle_rad = math.radians(angle_deg)
    area = round(0.5 * a * b * math.sin(angle_rad), 2)
    answer = area
    problem = (
        f"Two sides of a triangle have lengths {a} and {b}, with an included "
        f"angle of {angle_deg} degrees. Find the area of the triangle. "
        f"Round to 2 decimal places."
    )

    return problem, str(answer)
