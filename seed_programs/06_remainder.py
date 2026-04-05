import random

def generate(seed):
    rng = random.Random(seed)
    divisor = rng.choice([7, 8, 9, 11, 12, 13])
    quotient = rng.randint(5, 20)
    remainder = rng.randint(1, divisor - 1)
    dividend = divisor * quotient + remainder
    answer = remainder

    problem = (
        f"A teacher distributes {dividend} stickers equally among "
        f"{divisor} students. After giving each student the same number "
        f"of stickers, how many stickers are left over?"
    )
    return problem, str(answer)
