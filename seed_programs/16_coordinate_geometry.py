"""
Coordinate geometry: lines, distance, midpoint, intersection.
MATH benchmark category: Geometry (Hendrycks et al., NeurIPS 2021)
"""
import random
import math


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["distance", "midpoint", "slope", "intersection"])

    if problem_type == "distance":
        x1, y1 = rng.randint(-10, 10), rng.randint(-10, 10)
        x2, y2 = rng.randint(-10, 10), rng.randint(-10, 10)
        while (x1, y1) == (x2, y2):
            x2, y2 = rng.randint(-10, 10), rng.randint(-10, 10)
        dist = round(math.sqrt((x2 - x1)**2 + (y2 - y1)**2), 4)
        problem = (
            f"Find the distance between the points ({x1}, {y1}) and ({x2}, {y2}). "
            f"Round to 4 decimal places if needed."
        )
        answer = str(dist)

    elif problem_type == "midpoint":
        x1, y1 = rng.randint(-10, 10), rng.randint(-10, 10)
        x2, y2 = rng.randint(-10, 10), rng.randint(-10, 10)
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        # Use integers or .5
        mx_str = str(int(mx)) if mx == int(mx) else str(mx)
        my_str = str(int(my)) if my == int(my) else str(my)
        problem = (
            f"What is the midpoint of the segment joining ({x1}, {y1}) and ({x2}, {y2})?"
        )
        answer = f"({mx_str}, {my_str})"

    elif problem_type == "slope":
        x1, y1 = rng.randint(-8, 8), rng.randint(-8, 8)
        dx = rng.randint(1, 8)
        dy = rng.randint(-8, 8)
        x2, y2 = x1 + dx, y1 + dy
        from math import gcd
        g = gcd(abs(dy), abs(dx))
        num, den = dy // g, dx // g
        slope_str = str(num) if den == 1 else f"{num}/{den}"
        problem = (
            f"Find the slope of the line passing through ({x1}, {y1}) and ({x2}, {y2})."
        )
        answer = slope_str

    else:  # intersection of two lines y = m1*x + b1 and y = m2*x + b2
        m1 = rng.randint(-3, 3)
        m2 = rng.randint(-3, 3)
        while m1 == m2:
            m2 = rng.randint(-3, 3)
        b1 = rng.randint(-5, 5)
        b2 = rng.randint(-5, 5)
        # m1*x + b1 = m2*x + b2  →  x = (b2 - b1) / (m1 - m2)
        x_num = b2 - b1
        x_den = m1 - m2
        from math import gcd
        g = gcd(abs(x_num), abs(x_den))
        xn, xd = x_num // g, x_den // g
        if xd < 0:
            xn, xd = -xn, -xd
        x_str = str(xn) if xd == 1 else f"{xn}/{xd}"
        y_val_num = m1 * x_num + b1 * x_den
        y_val_den = x_den
        g2 = gcd(abs(y_val_num), abs(y_val_den))
        yn, yd = y_val_num // g2, y_val_den // g2
        if yd < 0:
            yn, yd = -yn, -yd
        y_str = str(yn) if yd == 1 else f"{yn}/{yd}"
        problem = (
            f"Find the intersection point of the lines y = {m1}x + {b1} "
            f"and y = {m2}x + {b2}."
        )
        answer = f"({x_str}, {y_str})"

    return problem, answer
