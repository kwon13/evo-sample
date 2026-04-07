import random
import math

def generate(seed):
    """Conditional probability: defective items from multiple machines."""
    rng = random.Random(seed)
    # Machine A produces pct_a% of items, defect rate d_a%
    # Machine B produces pct_b%, defect rate d_b%
    pct_a = rng.choice([40, 50, 60, 70])
    pct_b = 100 - pct_a
    d_a = rng.choice([2, 3, 4, 5])
    d_b = rng.choice([6, 8, 10, 12])

    # P(defective) = pct_a/100 * d_a/100 + pct_b/100 * d_b/100
    p_def_num = pct_a * d_a + pct_b * d_b  # ×10000
    # P(from A | defective) = (pct_a * d_a) / p_def_num
    p_a_given_def_num = pct_a * d_a
    g = math.gcd(p_a_given_def_num, p_def_num)
    num = p_a_given_def_num // g
    den = p_def_num // g

    answer = f"{num}/{den}"

    problem = (
        f"A factory has two machines. Machine A produces {pct_a}% of all items "
        f"with a {d_a}% defect rate. Machine B produces {pct_b}% of all items "
        f"with a {d_b}% defect rate. If a randomly selected item is found to be "
        f"defective, what is the probability it was produced by Machine A? "
        f"Express your answer as a fraction."
    )
    return problem, answer
