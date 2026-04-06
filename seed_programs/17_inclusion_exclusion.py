"""
Inclusion-Exclusion principle and advanced counting.
MATH benchmark category: Counting & Probability (Hendrycks et al., NeurIPS 2021)
"""
import random
import math


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["two_sets", "derangement", "permutation_constraint"])

    if problem_type == "two_sets":
        # |A ∪ B| = |A| + |B| - |A ∩ B|
        total = rng.randint(30, 100)
        a = rng.randint(10, total - 5)
        b = rng.randint(10, total - 5)
        both = rng.randint(2, min(a, b) - 1)
        either = a + b - both
        problem = (
            f"In a group of {total} students, {a} study mathematics, "
            f"{b} study science, and {both} study both. "
            f"How many students study mathematics or science (or both)?"
        )
        answer = str(either)

    elif problem_type == "derangement":
        # D(n) = n! * Σ(-1)^k / k!  for k=0..n
        n = rng.randint(3, 6)
        # Compute D(n)
        d = round(math.factorial(n) * sum(
            (-1)**k / math.factorial(k) for k in range(n + 1)
        ))
        problem = (
            f"In how many ways can {n} letters be placed into {n} addressed envelopes "
            f"so that no letter goes into its correct envelope? "
            f"(This is the number of derangements of {n} elements.)"
        )
        answer = str(d)

    else:
        # Count permutations of n items where specific item is NOT first
        n = rng.randint(4, 7)
        # Total: n!  Restricted (item A is first): (n-1)!
        total_perms = math.factorial(n)
        bad = math.factorial(n - 1)
        good = total_perms - bad
        problem = (
            f"How many ways can {n} distinct books be arranged on a shelf "
            f"so that a specific book (Book A) is NOT in the first position?"
        )
        answer = str(good)

    return problem, answer
