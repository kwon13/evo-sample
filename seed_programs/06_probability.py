import random
import math


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["bayes", "geometric"])

    if problem_type == "bayes":
        pct_a = rng.choice([40, 50, 60, 70])
        pct_b = 100 - pct_a
        d_a = rng.choice([2, 3, 4, 5])
        d_b = rng.choice([6, 8, 10, 12])
        p_def_num = pct_a * d_a + pct_b * d_b
        p_a_given_def = pct_a * d_a
        g = math.gcd(p_a_given_def, p_def_num)
        num = p_a_given_def // g
        den = p_def_num // g
        answer = f"{num}/{den}"
        problem = (
            f"A factory has two machines. Machine A produces {pct_a}% of all items "
            f"with a {d_a}% defect rate. Machine B produces {pct_b}% with a {d_b}% "
            f"defect rate. A randomly selected item is defective. What is the "
            f"probability it was produced by Machine A? Express as a fraction."
        )

    else:
        n = rng.randint(3, 6)
        sides = rng.choice([4, 6, 8])
        target = rng.randint(1, sides)
        num = (sides - 1) ** (n - 1)
        den = sides ** n
        g = math.gcd(num, den)
        answer = f"{num // g}/{den // g}"
        problem = (
            f"A fair {sides}-sided die is rolled {n} times. What is the probability "
            f"that the number {target} appears exactly once? Express as a fraction."
        )

    return problem, str(answer)
