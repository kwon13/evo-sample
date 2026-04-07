import random

def generate(seed):
    """Three workers with efficiency loss when working together."""
    rng = random.Random(seed)
    # A alone: t_a hours, B alone: t_b, C alone: t_c
    t_a = rng.choice([4, 5, 6, 8, 10])
    t_b = rng.choice([6, 8, 10, 12, 15])
    t_c = rng.choice([10, 12, 15, 20])

    # A+B work for h1 hours, then C joins
    # A alone does h1 hours first, then all three finish
    h1 = rng.randint(1, t_a - 1)

    # Work done by A in h1 hours
    work_a = h1 / t_a

    remaining = 1 - work_a  # fraction remaining
    # Combined rate of A+B+C
    rate_abc = 1/t_a + 1/t_b + 1/t_c
    h2 = remaining / rate_abc

    total_time = round(h1 + h2, 2)

    problem = (
        f"Worker A can complete a job alone in {t_a} hours, B in {t_b} hours, "
        f"and C in {t_c} hours. A starts working alone for {h1} hour(s), "
        f"then B and C join A, and all three work together until the job is done. "
        f"What is the total time to complete the job? Round to 2 decimal places."
    )
    return problem, str(total_time)
