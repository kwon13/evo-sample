"""
Calculus foundations: limits, derivatives, definite integrals.
Answers are always single numerical values.
"""
import random
import math


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["derivative_eval", "definite_integral", "limit"])

    if problem_type == "derivative_eval":
        # f(x) = ax^n + bx^m, find f'(c)
        a = rng.randint(1, 5)
        n = rng.randint(2, 4)
        b = rng.randint(-5, 5)
        m = rng.randint(1, n - 1)
        c = rng.randint(1, 4)
        # f'(x) = a*n*x^(n-1) + b*m*x^(m-1)
        deriv_at_c = a * n * c**(n-1) + b * m * c**(m-1)
        answer = deriv_at_c

        f_str = f"{a}x^{n}"
        if b > 0:
            f_str += f" + {b}x^{m}"
        elif b < 0:
            f_str += f" - {-b}x^{m}"

        problem = (
            f"Let f(x) = {f_str}. Compute f'({c}), "
            f"the derivative of f evaluated at x = {c}."
        )

    elif problem_type == "definite_integral":
        # ∫_0^a (px^q) dx = p * a^(q+1) / (q+1)
        p = rng.randint(1, 6)
        q = rng.randint(1, 3)
        a = rng.randint(1, 4)
        # result = p * a^(q+1) / (q+1)
        numerator = p * a**(q+1)
        denominator = q + 1
        g = math.gcd(abs(numerator), denominator)
        num = numerator // g
        den = denominator // g

        if den == 1:
            answer = str(num)
        else:
            answer = f"{num}/{den}"

        problem = (
            f"Evaluate the definite integral ∫₀^{a} {p}x^{q} dx. "
            f"Express your answer as a fraction or integer."
        )

    else:  # limit
        # lim_{x→a} (x^2 - a^2) / (x - a) = 2a
        a = rng.randint(2, 10)
        answer = 2 * a

        problem = (
            f"Compute the limit: lim(x→{a}) (x² − {a**2}) / (x − {a})."
        )

    return problem, str(answer)
