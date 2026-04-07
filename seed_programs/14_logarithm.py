"""
Exponential and logarithm equations.
MATH benchmark category: Precalculus (Hendrycks et al., NeurIPS 2021)
"""
import random
import math


def generate(seed):
    """Multi-step logarithm: solve for x using log properties."""
    rng = random.Random(seed)

    problem_type = rng.choice(["change_of_base", "log_equation", "compound"])

    if problem_type == "change_of_base":
        # log_a(b) * log_b(c) = log_a(c)
        a = rng.choice([2, 3, 5])
        n1 = rng.randint(2, 4)
        n2 = rng.randint(2, 3)
        b = a ** n1
        c = b ** n2  # c = a^(n1*n2)
        # log_a(b) = n1, log_b(c) = n2, product = n1*n2 = log_a(c)
        answer = n1 * n2
        problem = (
            f"Compute log_{a}({b}) × log_{b}({c})."
        )

    elif problem_type == "log_equation":
        # log_b(x) + log_b(x+k) = n → x(x+k) = b^n
        b = rng.choice([2, 3, 5, 10])
        n = rng.randint(2, 4)
        target = b ** n
        # Find x, k such that x*(x+k) = target, x > 0
        k = rng.randint(1, 10)
        # x^2 + kx - target = 0 → x = (-k + sqrt(k^2 + 4*target)) / 2
        disc = k**2 + 4*target
        sqrt_disc = math.isqrt(disc)
        if sqrt_disc * sqrt_disc == disc and (-k + sqrt_disc) % 2 == 0:
            x = (-k + sqrt_disc) // 2
            answer = x
        else:
            # fallback to simpler case
            x = rng.randint(2, 10)
            k = target // x - x
            if k > 0 and x * (x + k) == target:
                answer = x
            else:
                answer = n
                problem = f"Evaluate log_{b}({target})."
                return problem, str(answer)

        problem = (
            f"Solve for x: log_{b}(x) + log_{b}(x + {k}) = {n}. "
            f"Find the positive value of x."
        )

    else:  # compound
        # (log_2(a))^2 + log_2(a) = some integer
        exp = rng.randint(2, 5)
        a = 2 ** exp  # log_2(a) = exp
        result = exp**2 + exp
        answer = result
        problem = (
            f"Let x = log₂({a}). Compute x² + x."
        )

    return problem, str(answer)
