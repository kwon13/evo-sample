"""
Coordinate geometry: triangle area, point-to-line distance, combined.
MATH benchmark category: Geometry (Hendrycks et al., NeurIPS 2021)
"""
import random
import math


def generate(seed):
    """Triangle area from 3 vertices + distance from centroid to a side."""
    rng = random.Random(seed)

    # Three vertices
    x1, y1 = rng.randint(-5, 5), rng.randint(-5, 5)
    x2, y2 = rng.randint(-5, 5), rng.randint(-5, 5)
    x3, y3 = rng.randint(-5, 5), rng.randint(-5, 5)

    # Ensure non-degenerate triangle
    area2 = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    while area2 == 0:
        x3 = rng.randint(-5, 5)
        y3 = rng.randint(-5, 5)
        area2 = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

    area = area2 / 2

    # Centroid
    cx = (x1 + x2 + x3) / 3
    cy = (y1 + y2 + y3) / 3

    # Distance from centroid to side AB
    # Line through A(x1,y1) and B(x2,y2): (y2-y1)x - (x2-x1)y + (x2-x1)*y1 - (y2-y1)*x1 = 0
    A_coeff = y2 - y1
    B_coeff = -(x2 - x1)
    C_coeff = (x2 - x1)*y1 - (y2 - y1)*x1

    dist = abs(A_coeff * cx + B_coeff * cy + C_coeff) / math.sqrt(A_coeff**2 + B_coeff**2)
    dist_rounded = round(dist, 4)

    problem = (
        f"A triangle has vertices A({x1}, {y1}), B({x2}, {y2}), and C({x3}, {y3}). "
        f"Find the distance from the centroid of the triangle to side AB. "
        f"Round to 4 decimal places."
    )
    return problem, str(dist_rounded)
