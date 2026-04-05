import random
import math

def generate(seed):
    rng = random.Random(seed)
    n_people = rng.randint(5, 15)
    answer = n_people * (n_people - 1) // 2

    events = ["a business meeting", "a party", "a school reunion"]
    event = events[seed % len(events)]

    problem = (
        f"At {event}, there are {n_people} people. If each person "
        f"shakes hands exactly once with every other person, "
        f"how many handshakes occur in total?"
    )
    return problem, str(answer)
