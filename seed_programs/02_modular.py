"""Modular arithmetic problem generator using inverse construction."""
import random

def generate(seed):
    rng = random.Random(seed)
    
    # Step 1: Choose answer first
    # Generate a prime modulus
    primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    p = rng.choice(primes)
    
    # Choose the answer x
    x = rng.randint(1, p - 1)
    
    # Choose coefficient a (coprime to p, which is guaranteed since p is prime)
    a = rng.randint(2, p - 1)
    
    # Step 2: Compute b = a*x mod p
    b = (a * x) % p
    
    # Step 3: Build problem
    problem = f"Find the value of x such that {a}x ≡ {b} (mod {p}), where 0 < x < {p}."
    answer = str(x)
    
    return problem, answer
