import random

def generate(seed):
    rng = random.Random(seed)
    lcm_candidates = [12, 15, 20, 24, 30, 36, 40, 48, 60]
    total_work = lcm_candidates[seed % len(lcm_candidates)]

    divisors = [d for d in range(2, total_work) if total_work % d == 0]
    if len(divisors) < 2:
        total_work = 60
        divisors = [d for d in range(2, 60) if 60 % d == 0]

    time_a = divisors[rng.randint(0, len(divisors) // 2)]
    time_b = divisors[rng.randint(len(divisors) // 2, len(divisors) - 1)]
    if time_a == time_b:
        time_b = divisors[-1]

    rate_a = total_work // time_a
    rate_b = total_work // time_b
    combined_rate = rate_a + rate_b
    combined_time = total_work / combined_rate

    answer = round(combined_time, 2)
    if answer == int(answer):
        answer = int(answer)

    workers = [("Alice", "Bob"), ("Pipe A", "Pipe B"), ("Machine X", "Machine Y")]
    w1, w2 = workers[seed % len(workers)]
    tasks = ["paint a fence", "fill a tank", "complete a project"]
    task = tasks[seed % len(tasks)]

    problem = (
        f"{w1} can {task} in {time_a} hours. {w2} can {task} in "
        f"{time_b} hours. If they work together, how many hours will "
        f"it take them to {task}?"
    )
    return problem, str(answer)
