"""System of linear equations generator using inverse construction."""
import random

def generate(seed):
    rng = random.Random(seed)
    
    # Step 1: Choose answer first (x, y)
    x = rng.randint(-5, 5)
    y = rng.randint(-5, 5)
    
    # Step 2: Generate two linearly independent equations
    # a1*x + b1*y = c1
    # a2*x + b2*y = c2
    a1, b1 = rng.randint(1, 5), rng.randint(-5, 5)
    a2, b2 = rng.randint(-5, 5), rng.randint(1, 5)
    
    # Ensure linearly independent (determinant != 0)
    while a1 * b2 - a2 * b1 == 0:
        a2 = rng.randint(-5, 5)
        b2 = rng.randint(1, 5)
    
    c1 = a1 * x + b1 * y
    c2 = a2 * x + b2 * y
    
    # Step 3: Build problem
    def format_eq(a, b, c):
        parts = []
        if a == 1: parts.append("x")
        elif a == -1: parts.append("-x")
        elif a != 0: parts.append(f"{a}x")
        
        if b > 0:
            if b == 1: parts.append("+ y")
            else: parts.append(f"+ {b}y")
        elif b < 0:
            if b == -1: parts.append("- y")
            else: parts.append(f"- {-b}y")
        
        return " ".join(parts) + f" = {c}"
    
    eq1 = format_eq(a1, b1, c1)
    eq2 = format_eq(a2, b2, c2)
    
    problem = f"Solve the system of equations:\n{eq1}\n{eq2}"
    answer = f"x = {x}, y = {y}"
    
    return problem, answer
