import random
import math


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["inradius", "area_two_sides"])

    if problem_type == "inradius":
        a = rng.randint(5, 15)
        b = rng.randint(5, 15)
        angle_deg = rng.choice([30, 45, 60, 90, 120, 150])
        angle_rad = math.radians(angle_deg)
        c = math.sqrt(a ** 2 + b ** 2 - 2 * a * b * math.cos(angle_rad))
        s = (a + b + c) / 2
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        answer = round(area / s, 2)
        problem = (
            f"A triangle has sides a = {a}, b = {b}, and included angle C = {angle_deg} degrees. "
            f"Find the inradius r = Area / s, where s is the semi-perimeter. "
            f"Round to 2 decimal places."
        )

    else:
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
