import random

def generate(seed):
    """Multi-stage mixture: mix, remove, add pure solvent."""
    rng = random.Random(seed)
    # Stage 1: vol_a @ pct_a + vol_b @ pct_b
    vol_a = rng.randint(5, 15)
    pct_a = rng.choice([20, 25, 30, 40])
    vol_b = rng.randint(5, 15)
    pct_b = rng.choice([50, 60, 70, 80])

    total_1 = vol_a + vol_b
    pure_1 = vol_a * pct_a / 100 + vol_b * pct_b / 100
    conc_1 = pure_1 / total_1

    # Stage 2: remove some liters
    remove = rng.randint(2, min(vol_a, vol_b))
    total_2 = total_1 - remove
    pure_2 = pure_1 - remove * conc_1  # proportional removal

    # Stage 3: add pure water
    add_water = rng.randint(3, 10)
    total_3 = total_2 + add_water
    final_pct = round(pure_2 / total_3 * 100, 2)

    answer = final_pct

    problem = (
        f"A chemist mixes {vol_a} liters of a {pct_a}% salt solution "
        f"with {vol_b} liters of a {pct_b}% salt solution. "
        f"She then removes {remove} liters of the well-mixed solution, "
        f"and adds {add_water} liters of pure water. "
        f"What is the final percentage concentration of salt? "
        f"Round to 2 decimal places."
    )
    return problem, str(answer)
