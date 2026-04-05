"""Polynomial equation generator using inverse construction."""
import random
import math

def generate(seed):
    rng = random.Random(seed)
    
    # Step 1: Choose answer first (roots)
    degree = rng.choice([2, 3])
    roots = [rng.randint(-10, 10) for _ in range(degree)]
    
    # Step 2: Construct polynomial from roots
    # (x - r1)(x - r2)... = 0
    # Expand manually
    coeffs = [1]
    for r in roots:
        new_coeffs = [0] * (len(coeffs) + 1)
        for i, c in enumerate(coeffs):
            new_coeffs[i] += c
            new_coeffs[i + 1] -= c * r
        coeffs = new_coeffs
    
    # Step 3: Build problem string
    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if c == 0:
            continue
        if power == 0:
            terms.append(str(c))
        elif power == 1:
            if c == 1:
                terms.append("x")
            elif c == -1:
                terms.append("-x")
            else:
                terms.append(f"{c}x")
        else:
            if c == 1:
                terms.append(f"x^{power}")
            elif c == -1:
                terms.append(f"-x^{power}")
            else:
                terms.append(f"{c}x^{power}")
    
    poly_str = " + ".join(terms).replace("+ -", "- ")
    problem = f"Solve the equation: {poly_str} = 0"
    
    # Answer: sorted unique roots
    answer = str(sorted(set(roots)))
    
    return problem, answer
