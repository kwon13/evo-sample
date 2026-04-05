import random

def generate(seed):
    rng = random.Random(seed)
    width = rng.randint(3, 15)
    ratio = rng.randint(2, 5)
    length = width * ratio
    area = width * length
    perimeter = 2 * (width + length)
    answer = area

    things = ["garden", "parking lot", "swimming pool", "classroom floor"]
    thing = things[seed % len(things)]

    problem = (
        f"A rectangular {thing} is {ratio} times longer than it is wide. "
        f"If the perimeter of the {thing} is {perimeter} meters, "
        f"what is the area of the {thing} in square meters?"
    )
    return problem, str(answer)
