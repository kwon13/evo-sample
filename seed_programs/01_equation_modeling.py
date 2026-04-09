import random


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["age_system", "work_rate", "mixture"])

    if problem_type == "age_system":
        b = rng.randint(8, 25)
        d1 = rng.randint(5, 15)
        d2 = rng.randint(3, b - 2)
        a = b + d1
        c = b - d2
        total = a + b + c
        answer = b
        problem = (
            f"Three people A, B, and C have ages summing to {total}. "
            f"A is {d1} years older than B, and B is {d2} years older than C. "
            f"What is B's current age?"
        )

    elif problem_type == "work_rate":
        t_a = rng.choice([4, 5, 6, 8, 10])
        t_b = rng.choice([6, 8, 10, 12, 15])
        t_c = rng.choice([10, 12, 15, 20])
        h1 = rng.randint(1, t_a - 1)
        remaining = 1 - h1 / t_a
        rate_all = 1 / t_a + 1 / t_b + 1 / t_c
        h2 = remaining / rate_all
        answer = round(h1 + h2, 2)
        problem = (
            f"A can finish a job in {t_a}h, B in {t_b}h, C in {t_c}h. "
            f"A works alone for {h1}h, then B and C join. "
            f"Total time to finish? Round to 2 decimal places."
        )

    else:
        v_a = rng.randint(5, 15)
        p_a = rng.choice([20, 25, 30, 40])
        v_b = rng.randint(5, 15)
        p_b = rng.choice([50, 60, 70, 80])
        total = v_a + v_b
        pure = v_a * p_a / 100 + v_b * p_b / 100
        conc = pure / total
        remove = rng.randint(2, min(v_a, v_b))
        pure2 = pure - remove * conc
        add_water = rng.randint(3, 10)
        final_pct = round(pure2 / (total - remove + add_water) * 100, 2)
        answer = final_pct
        problem = (
            f"Mix {v_a}L of {p_a}% solution with {v_b}L of {p_b}% solution. "
            f"Remove {remove}L of the mixture, add {add_water}L pure water. "
            f"Final concentration (%)? Round to 2 decimal places."
        )

    return problem, str(answer)
