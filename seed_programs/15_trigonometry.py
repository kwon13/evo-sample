"""
Trigonometry: special angles, identities, and triangle problems.
MATH benchmark category: Precalculus / Geometry (Hendrycks et al., NeurIPS 2021)
"""
import random
import math


# Exact values at special angles (degrees → (sin, cos, tan) as fractions/strings)
_SPECIAL = {
    30:  ("1/2",        "√3/2",  "1/√3"),
    45:  ("√2/2",      "√2/2",   "1"),
    60:  ("√3/2",      "1/2",   "√3"),
    90:  ("1",          "0",     "undefined"),
    0:   ("0",          "1",     "0"),
}

_EXACT_SIN = {30: 0.5, 45: math.sqrt(2)/2, 60: math.sqrt(3)/2, 90: 1.0, 0: 0.0}
_EXACT_COS = {30: math.sqrt(3)/2, 45: math.sqrt(2)/2, 60: 0.5, 90: 0.0, 0: 1.0}


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["special_value", "law_of_cosines", "identity"])

    if problem_type == "special_value":
        angle = rng.choice([0, 30, 45, 60, 90])
        fn = rng.choice(["sin", "cos"])
        vals = _SPECIAL[angle]
        if fn == "sin":
            answer = vals[0]
        else:
            answer = vals[1]
        problem = (
            f"What is the exact value of {fn}({angle}°)?"
        )

    elif problem_type == "law_of_cosines":
        # Integer-sided triangle, find missing side
        a = rng.randint(3, 10)
        b = rng.randint(3, 10)
        angle_C = rng.choice([60, 90])
        cos_C = _EXACT_COS[angle_C]
        c_sq = a**2 + b**2 - 2*a*b*cos_C
        c = round(math.sqrt(c_sq), 4)
        problem = (
            f"A triangle has sides a = {a}, b = {b}, and the included angle C = {angle_C}°. "
            f"Find the length of side c using the law of cosines. "
            f"Round to 4 decimal places if needed."
        )
        answer = str(c)

    else:
        # Pythagorean identity: sin²θ + cos²θ = 1
        angle = rng.choice([30, 45, 60])
        sin_val = round(_EXACT_SIN[angle], 6)
        cos_val = round(_EXACT_COS[angle], 6)
        sq_sum = round(sin_val**2 + cos_val**2, 6)
        problem = (
            f"If sin(θ) = {round(_EXACT_SIN[angle], 4)} and cos(θ) = {round(_EXACT_COS[angle], 4)}, "
            f"what is sin²(θ) + cos²(θ)?"
        )
        answer = "1"

    return problem, answer
