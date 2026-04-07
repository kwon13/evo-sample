"""
Inequalities: AM-GM, Cauchy-Schwarz, optimization with constraints.
Answers are always single numerical values.
"""
import random
import math


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["am_gm_min", "am_gm_sum_product", "constrained_opt"])

    if problem_type == "am_gm_min":
        # Minimize f(x) = ax + b/x for x > 0
        # By AM-GM: ax + b/x >= 2*sqrt(ab)
        a = rng.randint(1, 5)
        b = rng.randint(1, 20)
        # min = 2*sqrt(a*b)
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
        # Given a + b + c = S (positive reals), maximize abc
        # By AM-GM: abc <= (S/3)^3, equality when a=b=c=S/3
        s = rng.choice([6, 9, 12, 15, 18])
        max_abc = (s / 3) ** 3
        answer = str(round(max_abc, 2))
        if max_abc == int(max_abc):
            answer = str(int(max_abc))

        problem = (
            f"Let a, b, c be positive real numbers with a + b + c = {s}. "
            f"What is the maximum value of the product abc?"
        )

    else:  # constrained_opt
        # Minimize a^2 + b^2 subject to a + b = S
        # By Cauchy-Schwarz or QM-AM: min when a=b=S/2
        # min = 2*(S/2)^2 = S^2/2
        s = rng.randint(4, 16)
        min_val = s**2 / 2
        answer = str(round(min_val, 2))
        if min_val == int(min_val):
            answer = str(int(min_val))

        problem = (
            f"Let a and b be real numbers with a + b = {s}. "
            f"What is the minimum value of a² + b²?"
        )

    return problem, str(answer)
