import random

def generate(seed):
    rng = random.Random(seed)
    radius = rng.randint(3, 20)
    area_coeff = radius * radius
    answer = area_coeff

    things = ["circular pond", "round playground", "circular flower bed"]
    thing = things[seed % len(things)]

    problem = (
        f"A {thing} has a radius of {radius} meters. What is the area "
        f"of the {thing}? Express your answer as a coefficient of pi "
        f"(e.g., if the area is 25*pi square meters, answer 25)."
    )
    return problem, str(answer)
