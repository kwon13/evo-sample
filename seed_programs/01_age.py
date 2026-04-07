import random

def generate(seed):
    """Multi-person age problem with ratios and future/past constraints."""
    rng = random.Random(seed)
    # A, B, C 세 사람. A는 B보다 d1살 많고, C는 B보다 d2살 적음
    # k년 후 A의 나이가 C의 나이의 ratio배
    b_now = rng.randint(8, 25)
    d1 = rng.randint(5, 15)
    d2 = rng.randint(3, b_now - 2)
    a_now = b_now + d1
    c_now = b_now - d2

    # k년 후: (a_now + k) = ratio * (c_now + k)
    # ratio를 정수로 만들기 위해 k를 조정
    for k in range(1, 30):
        if (a_now + k) % (c_now + k) == 0:
            ratio = (a_now + k) // (c_now + k)
            if ratio >= 2:
                break
    else:
        k = 5
        ratio = 2
        c_now = (a_now + k) // ratio - k
        d2 = b_now - c_now

    # 답: B의 현재 나이
    answer = b_now
    total_now = a_now + b_now + c_now

    problem = (
        f"Three people A, B, and C have ages that sum to {total_now}. "
        f"A is {d1} years older than B, and B is {d2} years older than C. "
        f"In {k} years, A's age will be exactly {ratio} times C's age. "
        f"What is B's current age?"
    )
    return problem, str(answer)
