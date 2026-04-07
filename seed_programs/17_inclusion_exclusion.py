"""
Three-set Inclusion-Exclusion and advanced counting.
MATH benchmark category: Counting & Probability (Hendrycks et al., NeurIPS 2021)
"""
import random
import math


def generate(seed):
    """Three-set inclusion-exclusion: students in clubs."""
    rng = random.Random(seed)

    total = rng.randint(80, 150)
    a = rng.randint(30, total - 20)  # math club
    b = rng.randint(25, total - 20)  # science club
    c = rng.randint(20, total - 20)  # art club

    ab = rng.randint(5, min(a, b) - 3)
    bc = rng.randint(5, min(b, c) - 3)
    ac = rng.randint(5, min(a, c) - 3)
    abc = rng.randint(2, min(ab, bc, ac) - 1)

    # |A ∪ B ∪ C| = |A| + |B| + |C| - |A∩B| - |B∩C| - |A∩C| + |A∩B∩C|
    union = a + b + c - ab - bc - ac + abc
    # students in none = total - union
    # clamp to be valid
    if union > total:
        total = union + rng.randint(5, 20)
    none_count = total - union

    # students in exactly one club
    only_a = a - ab - ac + abc
    only_b = b - ab - bc + abc
    only_c = c - ac - bc + abc
    exactly_one = only_a + only_b + only_c

    answer = exactly_one

    problem = (
        f"In a school of {total} students, {a} are in the math club, "
        f"{b} in the science club, and {c} in the art club. "
        f"{ab} are in both math and science, {bc} in both science and art, "
        f"{ac} in both math and art, and {abc} are in all three clubs. "
        f"How many students are in exactly one club?"
    )
    return problem, str(answer)
