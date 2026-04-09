import random
import math


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["am_gm_min", "am_gm_sum_product", "constrained_opt"])

    if problem_type == "am_gm_min":
        a = rng.randint(1, 5)
        b = rng.randint(1, 20)
        min_val = round(2 * math.sqrt(a * b), 4)

        if a == 1:
            f_str = f"x + {b}/x"
        else:
            f_str = f"{a}x + {b}/x"

        problem = (
            f"For x > 0, find the minimum value of f(x) = {f_str}. "
            f"Round to 4 decimal places if needed."
        )
        answer = str(min_val)

    elif problem_type == "am_gm_sum_product":
        s = rng.choice([6, 9, 12, 15, 18])
        max_abc = (s / 3) ** 3
        answer = str(round(max_abc, 2))
        if max_abc == int(max_abc):
            answer = str(int(max_abc))

        problem = (
            f"Let a, b, c be positive real numbers with a + b + c = {s}. "
            f"What is the maximum value of the product abc?"
        )

    else:
        s = rng.randint(4, 16)
        min_val = s ** 2 / 2
        answer = str(round(min_val, 2))
        if min_val == int(min_val):
            answer = str(int(min_val))

        problem = (
            f"Let a and b be real numbers with a + b = {s}. "
            f"What is the minimum value of a^2 + b^2?"
        )

    return problem, str(answer)
