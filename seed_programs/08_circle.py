import random
import math

def generate(seed):
    """Annulus (ring) area with sector constraint."""
    rng = random.Random(seed)
    R = rng.randint(8, 20)
    r = rng.randint(3, R - 3)
    # sector angle in degrees
    angle = rng.choice([60, 90, 120, 150, 180, 270])

    # annulus sector area = (angle/360) * pi * (R^2 - r^2)
    coeff = round((angle / 360) * (R**2 - r**2), 2)

    problem = (
        f"Two concentric circles have radii {r} and {R}. "
        f"A sector of central angle {angle}° is drawn. "
        f"Find the area of the region between the two circles "
        f"within this sector. Express your answer as a decimal "
        f"coefficient of pi (e.g., if the area is 45.5π, answer 45.5)."
    )
    return problem, str(coeff)
