"""
Trigonometry: law of cosines, area computation, combined problems.
MATH benchmark category: Precalculus / Geometry (Hendrycks et al., NeurIPS 2021)
"""
import random
import math


def generate(seed):
    """Triangle: law of cosines → find side, then compute area using Heron's formula."""
    rng = random.Random(seed)

    a = rng.randint(5, 15)
    b = rng.randint(5, 15)
    angle_C_deg = rng.choice([60, 90, 120])
    angle_C_rad = math.radians(angle_C_deg)

    # Law of cosines: c^2 = a^2 + b^2 - 2ab*cos(C)
    c_sq = a**2 + b**2 - 2*a*b*math.cos(angle_C_rad)
    c = math.sqrt(c_sq)

    # Heron's formula for area
    s = (a + b + c) / 2
    area = math.sqrt(s * (s - a) * (s - b) * (s - c))
    area_rounded = round(area, 2)

    # Also compute inradius r = area / s
    inradius = round(area / s, 2)

    problem = (
        f"A triangle has sides a = {a}, b = {b}, and the included angle C = {angle_C_deg}°. "
        f"First find side c using the law of cosines, then use Heron's formula "
        f"to compute the area of the triangle. Finally, find the inradius "
        f"r = Area / s, where s is the semi-perimeter. "
        f"What is the inradius? Round to 2 decimal places."
    )
    return problem, str(inradius)
