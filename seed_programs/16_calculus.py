import random
import math


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["derivative_eval", "definite_integral"])

    if problem_type == "derivative_eval":
        a = rng.randint(1, 5)
        n = rng.randint(2, 4)
        b = rng.randint(-5, 5)
        m = rng.randint(1, n - 1)
        c_val = rng.randint(1, 4)
        deriv = a * n * c_val ** (n - 1) + b * m * c_val ** (m - 1)
        answer = deriv
        f_str = f"{a}x^{n}"
        if b > 0:
            f_str += f" + {b}x^{m}"
        elif b < 0:
            f_str += f" - {-b}x^{m}"
        problem = (
            f"Let f(x) = {f_str}. Compute the derivative f'(x) "
            f"evaluated at x = {c_val}."
        )

    else:
        p = rng.randint(1, 6)
        q = rng.randint(1, 3)
        a = rng.randint(1, 4)
        b_val = rng.randint(0, a - 1)
        upper_val = p * a ** (q + 1) // (q + 1)
        lower_val = p * b_val ** (q + 1) // (q + 1)
        numerator = p * (a ** (q + 1) - b_val ** (q + 1))
        denominator = q + 1
        g = math.gcd(abs(numerator), denominator)
        num = numerator // g
        den = denominator // g
        if den == 1:
            answer = str(num)
        else:
            answer = f"{num}/{den}"
        problem = (
            f"Evaluate the definite integral of {p}x^{q} "
            f"from {b_val} to {a}. Express your answer as a fraction or integer."
        )

    return problem, str(answer)
