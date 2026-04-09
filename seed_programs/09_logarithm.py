import random
import math


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["change_of_base", "log_equation", "compound"])

    if problem_type == "change_of_base":
        a = rng.choice([2, 3, 5])
        n1 = rng.randint(2, 4)
        n2 = rng.randint(2, 3)
        b = a ** n1
        c = b ** n2
        answer = n1 * n2
        problem = (
            f"Compute log base {a} of {b}, multiplied by log base {b} of {c}."
        )

    elif problem_type == "log_equation":
        base = rng.choice([2, 3, 5, 10])
        n = rng.randint(2, 3)
        target = base ** n
        x = rng.randint(2, 10)
        k = target // x - x
        while k <= 0 or x * (x + k) != target:
            x = rng.randint(2, 10)
            k = target // x - x
        answer = x
        problem = (
            f"Solve for positive x: log base {base} of x plus "
            f"log base {base} of (x + {k}) equals {n}."
        )

    else:
        base = rng.choice([2, 3])
        exp = rng.randint(2, 5)
        a = base ** exp
        coeff = rng.randint(2, 4)
        result = coeff * exp ** 2 + exp
        answer = result
        problem = (
            f"Let x = log base {base} of {a}. Compute {coeff}x^2 + x."
        )

    return problem, str(answer)
