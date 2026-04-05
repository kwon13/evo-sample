import random

def generate(seed):
    rng = random.Random(seed)
    child_now = rng.randint(4, 12)
    diff = rng.randint(22, 35)
    parent_now = child_now + diff
    years_ago = rng.randint(1, child_now - 1)
    parent_then = parent_now - years_ago
    child_then = child_now - years_ago
    answer = child_now

    problem = (
        f"A parent is {diff} years older than their child. "
        f"{years_ago} years ago, the parent was {parent_then} years old. "
        f"How old is the child now?"
    )
    return problem, str(answer)
