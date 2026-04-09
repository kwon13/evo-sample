import random


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["exactly_one", "at_least_two"])

    if problem_type == "exactly_one":
        total = rng.randint(80, 150)
        a = rng.randint(30, total - 20)
        b = rng.randint(25, total - 20)
        c = rng.randint(20, total - 20)
        ab = rng.randint(5, min(a, b) - 3)
        bc = rng.randint(5, min(b, c) - 3)
        ac = rng.randint(5, min(a, c) - 3)
        abc = rng.randint(2, min(ab, bc, ac) - 1)
        union = a + b + c - ab - bc - ac + abc
        if union > total:
            total = union + rng.randint(5, 20)
        only_a = a - ab - ac + abc
        only_b = b - ab - bc + abc
        only_c = c - ac - bc + abc
        answer = only_a + only_b + only_c
        problem = (
            f"In a school of {total} students, {a} are in math club, "
            f"{b} in science club, and {c} in art club. {ab} are in both math "
            f"and science, {bc} in both science and art, {ac} in both math and art, "
            f"and {abc} in all three. How many students are in exactly one club?"
        )

    else:
        total = rng.randint(100, 200)
        a = rng.randint(40, total - 30)
        b = rng.randint(35, total - 30)
        ab = rng.randint(10, min(a, b) - 5)
        union = a + b - ab
        if union > total:
            total = union + rng.randint(10, 30)
        answer = ab
        problem = (
            f"In a survey of {total} people, {a} like coffee and {b} like tea. "
            f"If {total - union} people like neither, how many people like both "
            f"coffee and tea?"
        )

    return problem, str(answer)
