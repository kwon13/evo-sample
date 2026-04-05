import random

def generate(seed):
    rng = random.Random(seed)
    vol_a = rng.randint(2, 10)
    pct_a = rng.choice([10, 15, 20, 25, 30])
    vol_b = rng.randint(2, 10)
    pct_b = rng.choice([40, 50, 60, 70, 80])
    total_vol = vol_a + vol_b
    total_pure = vol_a * pct_a + vol_b * pct_b
    result_pct = total_pure / total_vol
    answer = result_pct

    problem = (
        f"A chemist mixes {vol_a} liters of a {pct_a}% salt solution "
        f"with {vol_b} liters of a {pct_b}% salt solution. "
        f"What is the percentage concentration of the resulting mixture?"
    )
    return problem, str(answer)
