"""Combinatorics counting problem generator using inverse construction."""
import random
import math

def generate(seed):
    rng = random.Random(seed)
    
    # Step 1: Choose answer first
    problem_type = rng.choice(["combination", "permutation", "partition"])
    
    if problem_type == "combination":
        n = rng.randint(5, 15)
        r = rng.randint(2, min(n - 1, 6))
        answer_val = math.comb(n, r)
        problem = (f"How many ways can you choose {r} items from a set of {n} "
                   f"distinct items? (Order does not matter)")
        answer = str(answer_val)
    
    elif problem_type == "permutation":
        n = rng.randint(4, 10)
        r = rng.randint(2, min(n, 5))
        answer_val = math.perm(n, r)
        problem = (f"How many ways can you arrange {r} items from a set of {n} "
                   f"distinct items in a row? (Order matters)")
        answer = str(answer_val)
    
    else:  # partition
        # How many ways to distribute n identical items into k distinct bins
        n = rng.randint(3, 8)
        k = rng.randint(2, 4)
        # Stars and bars: C(n+k-1, k-1)
        answer_val = math.comb(n + k - 1, k - 1)
        problem = (f"How many ways can you distribute {n} identical balls into "
                   f"{k} distinct boxes?")
        answer = str(answer_val)
    
    return problem, answer
