"""
Exponential and logarithm equations.
MATH benchmark category: Precalculus (Hendrycks et al., NeurIPS 2021)
"""
import random
import math


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["exp_solve", "log_eval", "log_property"])

    if problem_type == "exp_solve":
        # b^x = b^n  → x = n
        base = rng.choice([2, 3, 5, 10])
        exponent = rng.randint(2, 6)
        value = base ** exponent
        problem = (
            f"Solve for x: {base}^x = {value}."
        )
        answer = str(exponent)

    elif problem_type == "log_eval":
        # log_base(base^n) = n
        base = rng.choice([2, 3, 5, 10])
        n = rng.randint(2, 5)
        value = base ** n
        problem = (
            f"Evaluate log base {base} of {value}."
        )
        answer = str(n)

    else:
        # log(a) + log(b) = log(a*b)
        a = rng.randint(2, 12)
        b = rng.randint(2, 12)
        product = a * b
        problem = (
            f"Simplify: log({a}) + log({b}), "
            f"where log denotes the common logarithm (base 10). "
            f"Express your answer as log(n) for some integer n."
        )
        answer = f"log({product})"

    return problem, answer
