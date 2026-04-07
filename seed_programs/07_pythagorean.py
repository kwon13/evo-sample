import random
import math

def generate(seed):
    """3D geometry: diagonal of a rectangular box."""
    rng = random.Random(seed)
    a = rng.randint(3, 12)
    b = rng.randint(3, 12)
    c = rng.randint(3, 12)

    # space diagonal = sqrt(a^2 + b^2 + c^2)
    diag_sq = a**2 + b**2 + c**2
    # surface area = 2(ab + bc + ca)
    surface = 2 * (a*b + b*c + c*a)

    # 질문: space diagonal^2 + surface area
    answer = diag_sq + surface

    problem = (
        f"A rectangular box has dimensions {a} × {b} × {c}. "
        f"Let D be the length of the space diagonal of the box, and "
        f"S be the total surface area. "
        f"Compute D² + S."
    )
    return problem, str(answer)
