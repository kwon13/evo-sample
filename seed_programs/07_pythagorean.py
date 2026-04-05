import random

def generate(seed):
    rng = random.Random(seed)
    triples = [(3,4,5),(5,12,13),(8,15,17),(7,24,25),(9,40,41)]
    a, b, c = triples[seed % len(triples)]
    scale = rng.randint(1, 3)
    a, b, c = a * scale, b * scale, c * scale
    answer = c

    problem = (
        f"A ladder leans against a wall. The foot of the ladder is "
        f"{a} meters away from the wall, and the top of the ladder "
        f"reaches {b} meters up the wall. "
        f"How long is the ladder in meters?"
    )
    return problem, str(answer)
