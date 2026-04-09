import random
import math


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["centroid_distance", "line_intersection"])

    if problem_type == "centroid_distance":
        x1, y1 = rng.randint(-5, 5), rng.randint(-5, 5)
        x2, y2 = rng.randint(-5, 5), rng.randint(-5, 5)
        x3, y3 = rng.randint(-5, 5), rng.randint(-5, 5)
        area2 = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        while area2 == 0:
            x3, y3 = rng.randint(-5, 5), rng.randint(-5, 5)
            area2 = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        cx = (x1 + x2 + x3) / 3
        cy = (y1 + y2 + y3) / 3
        A_c = y2 - y1
        B_c = -(x2 - x1)
        C_c = (x2 - x1) * y1 - (y2 - y1) * x1
        dist = abs(A_c * cx + B_c * cy + C_c) / math.sqrt(A_c ** 2 + B_c ** 2)
        answer = round(dist, 4)
        problem = (
            f"A triangle has vertices A({x1}, {y1}), B({x2}, {y2}), C({x3}, {y3}). "
            f"Find the distance from the centroid to side AB. Round to 4 decimal places."
        )

    else:
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
