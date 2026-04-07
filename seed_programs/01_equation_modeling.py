"""
Equation modeling: multi-step word problems requiring equation setup.
Merges: age, speed, work_rate, mixture, rectangle categories.
"""
import random


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice([
        "age_system", "speed_legs", "work_rate_join",
        "mixture_dilute", "rect_optimize",
    ])

    if problem_type == "age_system":
        # 3인 나이 연립방정식
        b = rng.randint(8, 25)
        d1 = rng.randint(5, 15)
        d2 = rng.randint(3, b - 2)
        a = b + d1
        c = b - d2
        total = a + b + c
        answer = b
        problem = (
            f"Three people A, B, and C have ages summing to {total}. "
            f"A is {d1} years older than B, and B is {d2} years older than C. "
            f"What is B's current age?"
        )

    elif problem_type == "speed_legs":
        # 3구간 + 휴식 → 평균속도
        s1 = rng.choice([40, 50, 60, 80])
        t1 = rng.randint(2, 4)
        d1 = s1 * t1
        t_rest = rng.randint(1, 2)
        s2 = rng.choice([30, 40, 50, 60])
        t2 = rng.randint(1, 3)
        d2 = s2 * t2
        s3 = rng.choice([60, 80, 100])
        t3 = rng.randint(1, 3)
        d3 = s3 * t3
        avg = round((d1 + d2 + d3) / (t1 + t_rest + t2 + t3), 2)
        if avg == int(avg):
            avg = int(avg)
        answer = avg
        problem = (
            f"A driver travels {d1} km at {s1} km/h, rests {t_rest} h, "
            f"then {d2} km at {s2} km/h, then {d3} km at {s3} km/h. "
            f"What is the average speed for the entire trip including rest? "
            f"Round to 2 decimal places."
        )

    elif problem_type == "work_rate_join":
        # A 혼자 h1시간 작업 후 B,C 합류
        t_a = rng.choice([4, 5, 6, 8, 10])
        t_b = rng.choice([6, 8, 10, 12, 15])
        t_c = rng.choice([10, 12, 15, 20])
        h1 = rng.randint(1, t_a - 1)
        work_done = h1 / t_a
        remaining = 1 - work_done
        rate_all = 1/t_a + 1/t_b + 1/t_c
        h2 = remaining / rate_all
        answer = round(h1 + h2, 2)
        problem = (
            f"A can finish a job in {t_a}h, B in {t_b}h, C in {t_c}h. "
            f"A works alone for {h1}h, then B and C join. "
            f"Total time to finish? Round to 2 decimal places."
        )

    elif problem_type == "mixture_dilute":
        # 혼합 → 제거 → 물 추가
        v_a = rng.randint(5, 15)
        p_a = rng.choice([20, 25, 30, 40])
        v_b = rng.randint(5, 15)
        p_b = rng.choice([50, 60, 70, 80])
        total = v_a + v_b
        pure = v_a * p_a / 100 + v_b * p_b / 100
        conc = pure / total
        remove = rng.randint(2, min(v_a, v_b))
        pure2 = pure - remove * conc
        add_water = rng.randint(3, 10)
        final_pct = round(pure2 / (total - remove + add_water) * 100, 2)
        answer = final_pct
        problem = (
            f"Mix {v_a}L of {p_a}% solution with {v_b}L of {p_b}% solution. "
            f"Remove {remove}L of the mixture, add {add_water}L pure water. "
            f"Final concentration (%)? Round to 2 decimal places."
        )

    else:  # rect_optimize
        # 직사각형 → 대각선 → 외접원
        import math
        w = rng.randint(5, 15)
        l = rng.randint(w + 3, 30)
        perim = 2 * (w + l)
        area = w * l
        diag_sq = w**2 + l**2
        circle_coeff = round(diag_sq / 4, 2)
        answer = circle_coeff
        problem = (
            f"A rectangle has perimeter {perim} and area {area}. "
            f"Find the area of the circumscribed circle of the triangle "
            f"formed by its diagonal. Express as a coefficient of pi."
        )

    return problem, str(answer)
