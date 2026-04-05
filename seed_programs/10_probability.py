import random
import math

def generate(seed):
    rng = random.Random(seed)
    red = rng.randint(3, 10)
    blue = rng.randint(3, 10)
    total = red + blue
    draw = 2
    ways_total = math.comb(total, draw)
    ways_same_red = math.comb(red, draw)
    ways_same_blue = math.comb(blue, draw)
    ways_same = ways_same_red + ways_same_blue
    answer = f"{ways_same}/{ways_total}"

    problem = (
        f"A bag contains {red} red balls and {blue} blue balls. "
        f"If you draw {draw} balls at random without replacement, "
        f"what is the probability that both balls are the same color? "
        f"Express your answer as a fraction."
    )
    return problem, answer
