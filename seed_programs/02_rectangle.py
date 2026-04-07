import random
import math

def generate(seed):
    """Rectangle with diagonal constraint and inscribed circle."""
    rng = random.Random(seed)
    w = rng.randint(5, 15)
    l = rng.randint(w + 3, 30)
    diag = round(math.sqrt(w**2 + l**2), 4)
    perimeter = 2 * (w + l)
    area = w * l

    # 직사각형 내접원은 없지만, 대각선과 넓이의 관계를 묻는 multi-step
    # 대각선이 만드는 삼각형의 넓이 = area / 2
    # 그 삼각형의 외접원 반지름 R = diag / 2
    # 질문: 대각선으로 나눈 삼각형의 외접원 넓이 (계수)
    R = diag / 2
    # 외접원 넓이 = pi * R^2 = pi * diag^2 / 4
    circle_area_coeff = round(diag**2 / 4, 2)

    problem = (
        f"A rectangle has a perimeter of {perimeter} and an area of {area}. "
        f"A diagonal of this rectangle divides it into two right triangles. "
        f"Find the area of the circumscribed circle of one such triangle. "
        f"Express your answer as a decimal coefficient of pi "
        f"(e.g., if the area is 12.5*pi, answer 12.5)."
    )
    return problem, str(circle_area_coeff)
