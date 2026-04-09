import random
import math


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["box_diagonal", "annulus_sector", "inscribed_circle"])

    if problem_type == "box_diagonal":
        a = rng.randint(3, 12)
        b = rng.randint(3, 12)
        c = rng.randint(3, 12)
        diag_sq = a ** 2 + b ** 2 + c ** 2
        surface = 2 * (a * b + b * c + c * a)
        answer = diag_sq + surface
        problem = (
            f"A rectangular box has dimensions {a} x {b} x {c}. "
            f"Let D be the space diagonal. Compute D^2 + S, "
            f"where S is the total surface area."
        )

    elif problem_type == "annulus_sector":
        R = rng.randint(8, 20)
        r = rng.randint(3, R - 3)
        angle = rng.choice([60, 90, 120, 150, 180])
        coeff = round((angle / 360) * (R ** 2 - r ** 2), 2)
        if coeff == int(coeff):
            coeff = int(coeff)
        answer = coeff
        problem = (
            f"Two concentric circles have radii {r} and {R}. "
            f"A sector of {angle} degrees is drawn. Find the area between "
            f"the circles within this sector. Express as a coefficient of pi."
        )

    else:
        triples = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (9, 40, 41)]
        a, b, c = triples[seed % len(triples)]
        scale = rng.randint(1, 3)
        a, b, c = a * scale, b * scale, c * scale
        inradius = (a + b - c) / 2
        answer = round(inradius, 2)
        if answer == int(answer):
            answer = int(answer)
        problem = (
            f"A right triangle has legs {a} and {b}, and hypotenuse {c}. "
            f"Find the radius of the inscribed circle of this triangle."
        )

    return problem, str(answer)
